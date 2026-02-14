#include "kernel.hpp"
#include "gpu_utils.hpp"
#include <cuda_runtime.h>
#include <algorithm>

constexpr size_t TILE_SIZE_U64 = (128 * 1024 * 1024) / sizeof(uint64_t);
constexpr int MAX_RES = 1000000;

__device__ __forceinline__ int count_mm(uint64_t s, uint64_t p, uint64_t m) {
    return __popcll((s ^ p) & m);
}

__global__ void k_search(const uint64_t* __restrict__ gen, size_t n, size_t g_off, EnzymeConfig cfg, uint64_t pat, uint64_t mask, uint32_t* res, int* count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint64_t chunk = gen[tid];
    uint32_t hits = 0;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        if (((chunk >> (i * 2)) & cfg.pam_care_mask) == cfg.pam_pattern) hits |= (1u << i);
    }

    while (hits) {
        int l_idx = __ffs(hits) - 1;
        size_t g_bit = ((g_off + tid) * 64) + (l_idx * 2);
        int64_t s_bit = (int64_t)g_bit + (cfg.pam_offset_correction * 2);

        if (s_bit >= 0) {
            int64_t loc_bit = s_bit - (int64_t)(g_off * 64);
            if (loc_bit >= 0 && loc_bit < (int64_t)(n * 64 - 64)) {
                uint64_t target = load_window_unaligned(gen, (size_t)loc_bit, n * 64);
                if (count_mm(target, pat, mask) <= (cfg.max_mismatches * 2)) {
                    int out_idx = atomicAdd(count, 1);
                    if (out_idx < MAX_RES) res[out_idx] = (uint32_t)(s_bit / 2);
                }
            }
        }
        hits &= ~(1u << l_idx);
    }
}

SearchResults launch_pipelined_search(const uint64_t* h_gen, size_t n, const EnzymeConfig& cfg, uint64_t pat, uint64_t mask) {
    uint32_t *d_res; int *d_c;
    cudaMalloc(&d_res, MAX_RES * sizeof(uint32_t));
    cudaMalloc(&d_c, sizeof(int));
    cudaMemset(d_c, 0, sizeof(int));

    uint64_t *d_tk[2];
    cudaMalloc(&d_tk[0], TILE_SIZE_U64 * 8);
    cudaMalloc(&d_tk[1], TILE_SIZE_U64 * 8);

    cudaStream_t st[2];
    cudaStreamCreate(&st[0]); cudaStreamCreate(&st[1]);

    size_t off = 0; int t_id = 0;
    while (off < n) {
        size_t cur = std::min(TILE_SIZE_U64, n - off);
        int sid = t_id % 2;
        cudaMemcpyAsync(d_tk[sid], h_gen + off, cur * 8, cudaMemcpyHostToDevice, st[sid]);
        k_search<<<(cur + 255) / 256, 256, 0, st[sid]>>>(d_tk[sid], cur, off, cfg, pat, mask, d_res, d_c);
        off += cur; t_id++;
    }
    cudaDeviceSynchronize();

    SearchResults r;
    int h_count = 0;
    cudaMemcpy(&h_count, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    r.count = std::min(h_count, MAX_RES);

    if (r.count > 0) {
        cudaMallocHost(&r.matches, r.count * sizeof(uint32_t));
        cudaMemcpy(r.matches, d_res, r.count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    } else {
        r.matches = nullptr;
    }

    cudaFree(d_res); cudaFree(d_c); cudaFree(d_tk[0]); cudaFree(d_tk[1]);
    cudaStreamDestroy(st[0]); cudaStreamDestroy(st[1]);
    return r;
}

void free_search_results(SearchResults& res) {
    if (res.matches)
    	cudaFreeHost(res.matches);
    res.matches = nullptr; res.count = 0;
}
