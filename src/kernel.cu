#include <algorithm>
#include <iostream>

#include "gpu_utils.hpp"
#include "kernel.hpp"

__global__ void __launch_bounds__(256) k_search_exact(
	const uint64_t *__restrict__ genome,
	size_t n,
	uint64_t pattern_val,
	uint32_t *__restrict__ match_indices,
	uint32_t *__restrict__ match_count,
	uint32_t max_capacity,
	size_t global_offset
) {
	uint64_t local_pattern = pattern_val;
	local_pattern = __shfl_sync(0xffffffff, local_pattern, 0);
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	for (size_t i = idx; i < n; i += stride) {
		uint64_t chunk = __ldg(&genome[i]);
		if (__popcll(chunk ^ local_pattern) == 0) {
			uint32_t write_idx = atomicAdd(match_count, 1);
			if (write_idx < max_capacity) {
				match_indices[write_idx] = (uint32_t)(global_offset + i);
			}
		}
	}
}

__global__ void __launch_bounds__(256) k_search_bulge(
	const uint64_t *__restrict__ genome,
	size_t n,
	uint64_t pattern_val,
	uint32_t *__restrict__ match_indices,
	uint32_t *__restrict__ match_count,
	uint32_t max_capacity,
	size_t global_offset,
	int max_mismatches
) {
	uint64_t local_pattern = pattern_val;
	local_pattern = __shfl_sync(0xffffffff, local_pattern, 0);
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	const uint64_t ODD_MASK = 0x5555555555555555ULL;
	for (size_t i = idx; i < n; i += stride) {
		uint64_t chunk = __ldg(&genome[i]);
		uint64_t xor_d = chunk ^ local_pattern;
		uint64_t diff_d = (xor_d | (xor_d >> 1)) & ODD_MASK;
		int score_d = __popcll(diff_d);
		uint64_t pat_l = local_pattern << 2;
		uint64_t xor_l = chunk ^ pat_l;
		uint64_t diff_l = (xor_l | (xor_l >> 1)) & ODD_MASK;
		int score_l = __popcll(diff_l);
		uint64_t pat_r = local_pattern >> 2;
		uint64_t xor_r = chunk ^ pat_r;
		uint64_t diff_r = (xor_r | (xor_r >> 1)) & ODD_MASK;
		int score_r = __popcll(diff_r);
		int best_score = min(score_d, min(score_l, score_r));
		if (best_score <= max_mismatches) {
			uint32_t write_idx = atomicAdd(match_count, 1);
			if (write_idx < max_capacity) {
				match_indices[write_idx] = (uint32_t)(global_offset + i);
			}
		}
	}
}

SearchResults launch_exact_search(const uint64_t *host_genome, size_t num_elements, uint64_t pattern) {
	SearchResults results;
	results.count = 0;
	results.capacity = 1024 * 1024;
	results.matches = nullptr;
	results.time_ms = 0.0f;

	CHECK_CUDA(cudaSetDevice(0));

	size_t free_mem, total_mem;
	CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));

	size_t safety_margin = 500 * 1024 * 1024;
	size_t usable_mem = (free_mem > safety_margin) ? (free_mem - safety_margin) : (free_mem / 2);
	size_t max_chunk_elements = usable_mem / sizeof(uint64_t);
	if (max_chunk_elements > 250000000)
		max_chunk_elements = 250000000;

	std::cout << "[GPU] Architecture: Chunking Strategy" << std::endl;
	std::cout << "[GPU] VRAM Free: " << (free_mem >> 20) << " MB | Chunk Size: " << (max_chunk_elements * 8 >> 20) << " MB" << std::endl;

	DeviceBuffer<uint64_t> d_genome(max_chunk_elements);
	DeviceBuffer<uint32_t> d_indices(results.capacity);
	DeviceBuffer<uint32_t> d_count(1);

	CudaEvent start, stop;
	start.record();

	size_t processed = 0;
	while (processed < num_elements) {
		size_t current_batch = std::min(max_chunk_elements, num_elements - processed);
		d_genome.copyFromHost(host_genome + processed, current_batch);
		int threads = 256;
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		int blocks = (int)((current_batch + threads - 1) / threads);
		if (blocks > prop.maxGridSize[0])
			blocks = prop.maxGridSize[0];
		if (blocks > 32768)
			blocks = 32768;
		k_search_exact<<<blocks, threads>>>(
			d_genome.data(),
			current_batch,
			pattern,
			d_indices.data(),
			d_count.data(),
			results.capacity,
			processed
		);
		CHECK_CUDA(cudaGetLastError());
		processed += current_batch;
	}

	stop.record();
	stop.synchronize();
	results.time_ms = CudaEvent::elapsed(start, stop);

	uint32_t total_found = 0;
	d_count.copyToHost(&total_found, 1);
	results.count = total_found;
	if (results.count > 0) {
		uint32_t fetch_count = std::min(results.count, results.capacity);
		results.matches = (uint32_t*)malloc(fetch_count * sizeof(uint32_t));
		d_indices.copyToHost(results.matches, fetch_count);
	}

	return results;
}

SearchResults launch_bulge_search(const uint64_t* host_genome, size_t num_elements, uint64_t pattern, int max_mismatches) {
    SearchResults results;
    results.count = 0;
    results.capacity = 1024 * 1024;
    results.matches = (uint32_t*)malloc(results.capacity * sizeof(uint32_t));
    results.time_ms = 0.0f;

    CHECK_CUDA(cudaSetDevice(0));

    const int N_STREAMS = 2;
    size_t chunk_elements = 50 * 1024 * 1024;

    std::cout << "[GPU] Pipeline: Double Buffering (2 Streams)" << std::endl;

    std::vector<DeviceBuffer<uint64_t>*> d_genome;
    std::vector<DeviceBuffer<uint32_t>*> d_indices;
    std::vector<DeviceBuffer<uint32_t>*> d_count;
    std::vector<CudaStream> streams(N_STREAMS);

    PinnedHostBuffer<uint32_t> h_counts(N_STREAMS);
    std::vector<uint32_t*> h_indices_pinned;

    for(int i=0; i<N_STREAMS; ++i) {
        d_genome.push_back(new DeviceBuffer<uint64_t>(chunk_elements));
        d_indices.push_back(new DeviceBuffer<uint32_t>(results.capacity / 4));
        d_count.push_back(new DeviceBuffer<uint32_t>(1));

        uint32_t* pin_ptr;
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
        d_genome[sid]->copyFromHostAsync(host_genome + processed, current_batch, stream);
        CHECK_CUDA(cudaMemsetAsync(d_count[sid]->data(), 0, sizeof(uint32_t), stream));
        int threads = 256;
        int blocks = (current_batch + threads - 1) / threads;
        if (blocks > 32768) blocks = 32768;
        k_search_bulge<<<blocks, threads, 0, stream>>>(
             d_genome[sid]->data(), current_batch, pattern,
             d_indices[sid]->data(), d_count[sid]->data(),
             results.capacity / 4, processed, max_mismatches
        );
        d_count[sid]->copyToHostAsync(&h_counts[sid], 1, stream);
        CHECK_CUDA(cudaMemcpyAsync(
            h_indices_pinned[sid],
            d_indices[sid]->data(),
            (results.capacity / 4) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream
        ));
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
