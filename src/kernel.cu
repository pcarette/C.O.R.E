#include <algorithm>
#include <iostream>
#include <vector>

#include "gpu_utils.hpp"
#include "kernel.hpp"

constexpr uint8_t MASK_ATAC = 1 << 0;
constexpr uint8_t MASK_BLACKLIST = 1 << 1;

__device__ __forceinline__ bool check_pam_cas9(uint64_t chunk, int bit_offset) {
	return ((chunk >> (bit_offset + 2)) & 0xF) == 0xA;
}

__global__ void __launch_bounds__(256) k_search_bulge(
	const uint64_t *__restrict__ genome,
	const uint8_t *__restrict__ epigenome,
	size_t n_blocks,
	uint64_t pattern_val,
	uint32_t *__restrict__ match_indices,
	uint32_t *__restrict__ match_count,
	uint32_t max_capacity,
	size_t global_offset_blocks,
	int max_mismatches,
	int max_seed_mismatches
) {
	uint64_t local_pattern = pattern_val;
	local_pattern = __shfl_sync(0xffffffff, local_pattern, 0);

	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	const uint64_t ODD_MASK = 0x5555555555555555ULL;
	const uint64_t SEED_MASK_BASE = 0xFFFFF00000ULL;
	const uint64_t GUIDE_MASK = 0xFFFFFFFFFFULL;

	for (size_t i = idx; i < n_blocks; i += stride) {
		uint64_t current = __ldg(&genome[i]);
		uint64_t next = (i + 1 < n_blocks) ? __ldg(&genome[i+1]) : 0;

		#pragma unroll
		for (int offset = 0; offset < 32; ++offset) {
			int shift = offset * 2;
			uint64_t chunk = (shift == 0) ? current : (current >> shift) | (next << (64 - shift));

			uint64_t xor_d = chunk ^ local_pattern;
			uint64_t diff_d = (xor_d | (xor_d >> 1)) & ODD_MASK & GUIDE_MASK;
			int seed_d = __popcll(diff_d & SEED_MASK_BASE);
			int total_d = (seed_d > max_seed_mismatches) ? 999 : __popcll(diff_d);

			uint64_t pat_l = local_pattern << 2;
			uint64_t xor_l = chunk ^ pat_l;
			uint64_t diff_l = (xor_l | (xor_l >> 1)) & ODD_MASK & GUIDE_MASK;
			int seed_l = __popcll(diff_l & (SEED_MASK_BASE << 2));
			int total_l = (seed_l > max_seed_mismatches) ? 999 : __popcll(diff_l);

			uint64_t pat_r = local_pattern >> 2;
			uint64_t xor_r = chunk ^ pat_r;
			uint64_t diff_r = (xor_r | (xor_r >> 1)) & ODD_MASK & GUIDE_MASK;
			int seed_r = __popcll(diff_r & (SEED_MASK_BASE >> 2));
			int total_r = (seed_r > max_seed_mismatches) ? 999 : __popcll(diff_r);

			int best_score = max_mismatches + 1;
			if (total_d <= max_mismatches && check_pam_cas9(chunk, 40))
				best_score = min(best_score, total_d);
			if (total_l <= max_mismatches && check_pam_cas9(chunk, 42))
				best_score = min(best_score, total_l);
			if (total_r <= max_mismatches && check_pam_cas9(chunk, 38))
				best_score = min(best_score, total_r);
			if (best_score <= max_mismatches) {
				size_t local_byte_idx = i * 32 + offset;
				uint8_t epi_val = epigenome[local_byte_idx];
				bool is_accessible = (epi_val & MASK_ATAC);
				bool not_blacklisted = !(epi_val & MASK_BLACKLIST);
				if (is_accessible && not_blacklisted) {
					uint32_t write_idx = atomicAdd(match_count, 1);
					if (write_idx < max_capacity) {
						size_t global_pos = (global_offset_blocks + i) * 32 + offset;
						match_indices[write_idx] = (uint32_t)global_pos;
					}
				}
			}
		}
	}
}

SearchResults launch_bulge_search(
	const uint64_t *genome_data,
	const uint8_t *host_epigenome,
	size_t num_elements,
	size_t max_epi_size,
	uint64_t pattern,
	int max_mismatches,
	int max_seed_mismatches
) {
	SearchResults results;
	results.count = 0;
	results.capacity = 1024 * 1024;
	results.matches = (uint32_t *)malloc(results.capacity * sizeof(uint32_t));
	results.time_ms = 0.0f;

	CHECK_CUDA(cudaSetDevice(0));

	const int N_STREAMS = 2;
	size_t chunk_elements = 50 * 1024 * 1024;

	std::cout << "[GPU] Pipeline: Double Buffering (2 Streams)" << std::endl;

	std::vector<DeviceBuffer<uint64_t> *> d_genome;
	std::vector<DeviceBuffer<uint8_t> *> d_epigenome;
	std::vector<DeviceBuffer<uint32_t> *> d_indices;
	std::vector<DeviceBuffer<uint32_t> *> d_count;
	std::vector<CudaStream> streams(N_STREAMS);

	PinnedHostBuffer<uint32_t> h_counts(N_STREAMS);
	std::vector<uint32_t *> h_indices_pinned;

	for(int i=0; i<N_STREAMS; ++i) {
		d_genome.push_back(new DeviceBuffer<uint64_t>(chunk_elements));
		d_epigenome.push_back(new DeviceBuffer<uint8_t>(chunk_elements * 32));
		d_indices.push_back(new DeviceBuffer<uint32_t>(results.capacity / 4));
		d_count.push_back(new DeviceBuffer<uint32_t>(1));

		uint32_t *pin_ptr;
		CHECK_CUDA(cudaMallocHost(&pin_ptr, (results.capacity / 4) * sizeof(uint32_t)));
		h_indices_pinned.push_back(pin_ptr);
	}

	CudaEvent start, stop;
	start.record();

	size_t processed = 0;
	int batch_id = 0;
	while (processed < num_elements) {
		int sid = batch_id % N_STREAMS;
		cudaStream_t stream = streams[sid].get();
		if (batch_id >= N_STREAMS) {
			streams[sid].synchronize();
			uint32_t count = h_counts[sid];
			if (count > 0) {
				uint32_t to_copy = std::min(count, results.capacity - results.count);
				if (to_copy > 0) {
					memcpy(results.matches + results.count, h_indices_pinned[sid], to_copy * sizeof(uint32_t));
					results.count += to_copy;
				}
			}
		}

		size_t current_batch = std::min(chunk_elements, num_elements - processed);
		d_genome[sid]->copyFromHostAsync(genome_data + processed, current_batch, stream);

		size_t epi_offset = processed * 32;
		size_t epi_len_req = current_batch * 32;
		size_t epi_len_safe = 0;
		if (epi_offset < max_epi_size)
			epi_len_safe = std::min(epi_len_req, max_epi_size - epi_offset);
		if (epi_len_safe > 0)
			d_epigenome[sid]->copyFromHostAsync(host_epigenome + epi_offset, epi_len_safe, stream);

		if (epi_len_safe < epi_len_req) {
			CHECK_CUDA(cudaMemsetAsync(
				d_epigenome[sid]->data() + epi_len_safe,
				0,
				epi_len_req - epi_len_safe,
				stream
			));
		}

		CHECK_CUDA(cudaMemsetAsync(d_count[sid]->data(), 0, sizeof(uint32_t), stream));
		int threads = 256;
		int blocks = (current_batch + threads - 1) / threads;
		if (blocks > 32768)
			blocks = 32768;
		k_search_bulge<<<blocks, threads, 0, stream>>>(
			d_genome[sid]->data(),
			d_epigenome[sid]->data(),
			current_batch,
			pattern,
			d_indices[sid]->data(),
			d_count[sid]->data(),
			results.capacity / 4,
			processed,
			max_mismatches,
			max_seed_mismatches
		);
		d_count[sid]->copyToHostAsync(&h_counts[sid], 1, stream);
		CHECK_CUDA(cudaMemcpyAsync(
			h_indices_pinned[sid],
			d_indices[sid]->data(),
			(results.capacity / 4) * sizeof(uint32_t),
			cudaMemcpyDeviceToHost, stream)
		);
		processed += current_batch;
		batch_id++;
	}

	for (int i = 0; i < N_STREAMS; ++i) {
		streams[i].synchronize();
		uint32_t count = h_counts[i];
		if (count > 0) {
			uint32_t to_copy = std::min(count, results.capacity - results.count);
			if (to_copy > 0) {
				memcpy(results.matches + results.count, h_indices_pinned[i], to_copy * sizeof(uint32_t));
				results.count += to_copy;
			}
		}
	}

	stop.record();
	stop.synchronize();
	results.time_ms = CudaEvent::elapsed(start, stop);

	for(int i=0; i<N_STREAMS; ++i) {
		delete d_genome[i];
		delete d_epigenome[i];
		delete d_indices[i];
		delete d_count[i];
		cudaFreeHost(h_indices_pinned[i]);
	}

	return results;
}

void free_search_results(SearchResults &results) {
	if (results.matches) {
		free(results.matches);
		results.matches = nullptr;
	}
}
