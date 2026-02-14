#include "gpu_utils.hpp"
#include "kernel.hpp"

constexpr uint8_t MASK_BLACKLIST = 1 << 1;

__device__ __forceinline__ int count_biological_mismatches(uint64_t seq, uint64_t pattern, uint64_t care_mask) {
    uint64_t xor_diff = seq ^ pattern;
    uint64_t bases_diff = (xor_diff | (xor_diff >> 1)) & 0x5555555555555555ULL;
    bases_diff &= care_mask;
    return __popcll(bases_diff);
}

__global__ void __launch_bounds__(256) k_search_bulge(
    const uint64_t *__restrict__ genome,
    const uint8_t *__restrict__ epigenome,
    size_t n_blocks,
    size_t epi_size,
    uint64_t pattern_val,
    uint64_t care_mask,
    uint32_t *__restrict__ match_indices,
    uint32_t *__restrict__ match_count,
    uint32_t max_capacity,
    int max_mismatches
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n_blocks; i += stride) {
        uint64_t current = __ldg(&genome[i]);
        uint64_t next = (i + 1 < n_blocks) ? __ldg(&genome[i+1]) : 0;

        #pragma unroll
        for (int offset = 0; offset < 32; ++offset) {
            int shift = offset * 2;
            uint64_t chunk = (shift == 0) ? current : (current >> shift) | (next << (64 - shift));

            int mismatches = count_biological_mismatches(chunk, pattern_val, care_mask);

            if (mismatches <= max_mismatches) {
                size_t global_pos_bp = i * 32 + offset;
                bool keep = true;

                if (epigenome && global_pos_bp < epi_size) {
                    uint8_t epi_val = epigenome[global_pos_bp];
                    if (epi_val & MASK_BLACKLIST) {
                        keep = false;
                    }
                }

                if (keep) {
                    uint32_t write_idx = atomicAdd(match_count, 1);
                    if (write_idx < max_capacity) {
                        match_indices[write_idx] = (uint32_t)global_pos_bp;
                    }
                }
            }
        }
    }
}

SearchResults launch_bulge_search(
    const uint64_t *genome_data,
    const uint8_t *host_epigenome,
    size_t num_blocks,
    size_t epi_size,
    uint64_t pattern,
    uint64_t care_mask,
    int max_mismatches,
    int max_seed_mismatches
) {
    SearchResults results;
    results.count = 0;
    results.capacity = 1024 * 1024;
    results.matches = (uint32_t *)malloc(results.capacity * sizeof(uint32_t));
    results.time_ms = 0.0f;

    int device = 0;
    cudaSetDevice(device);

    uint64_t *d_genome = nullptr;
    uint8_t *d_epigenome = nullptr;
    uint32_t *d_indices = nullptr;
    uint32_t *d_count = nullptr;

    size_t genome_bytes = num_blocks * sizeof(uint64_t);
    size_t epi_bytes = epi_size * sizeof(uint8_t);

    CHECK_CUDA(cudaMalloc(&d_genome, genome_bytes));
    CHECK_CUDA(cudaMemcpy(d_genome, genome_data, genome_bytes, cudaMemcpyHostToDevice));

    if (host_epigenome && epi_size > 0) {
        CHECK_CUDA(cudaMalloc(&d_epigenome, epi_bytes));
        CHECK_CUDA(cudaMemcpy(d_epigenome, host_epigenome, epi_bytes, cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMalloc(&d_indices, results.capacity * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_count, 0, sizeof(uint32_t)));

    int threads = 256;
    int blocks = (num_blocks + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    CudaEvent start, stop;
    start.record();

    k_search_bulge<<<blocks, threads>>>(
        d_genome,
        d_epigenome,
        num_blocks,
        epi_size,
        pattern,
        care_mask,
        d_indices,
        d_count,
        results.capacity,
        max_mismatches
    );

    stop.record();
    stop.synchronize();
    results.time_ms = CudaEvent::elapsed(start, stop);

    uint32_t h_count = 0;
    CHECK_CUDA(cudaMemcpy(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    results.count = std::min(h_count, results.capacity);
    if (results.count > 0) {
        CHECK_CUDA(cudaMemcpy(results.matches, d_indices, results.count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    if (d_genome) cudaFree(d_genome);
    if (d_epigenome) cudaFree(d_epigenome);
    if (d_indices) cudaFree(d_indices);
    if (d_count) cudaFree(d_count);

    return results;
}

void free_search_results(SearchResults &results) {
    if (results.matches) free(results.matches);
    results.matches = nullptr;
}
