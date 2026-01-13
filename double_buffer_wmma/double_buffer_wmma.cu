#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <mma.h>

using namespace nvcuda; 

#define BLOCK_SIZE 64
#define NUM_BUFFERS 2

template <int BM, int BN, int BK>
__device__ void load_gmem_to_smem(
    __nv_bfloat16* A,  // gmem
    __nv_bfloat16* B,  // gmem
    __nv_bfloat16* sA, // smem
    __nv_bfloat16* sB, // smem
    const int k_tile_idx,
    const int block_row,
    const int block_col,
    const int num_threads,
    const int M,
    const int N,
    const int K
) {
    // constant for vectorized loads
    constexpr int bfloats_per_load = 2;
    const int loads_per_iter = num_threads * bfloats_per_load; 
    const int total_a_loads = (BM * BK) / loads_per_iter;
    const int total_b_loads = (BK * BN) / loads_per_iter;

    // load tile of A from GMEM into SMEM.
    #pragma unroll
    for (int load_idx = 0; load_idx < total_a_loads; load_idx++) {
        const int linear_idx = load_idx * loads_per_iter + (threadIdx.x * bfloats_per_load);
        const int a_thread_row = linear_idx / BK;
        const int a_thread_col = linear_idx % BK;
        const int a_global_row = (block_row * BM) + a_thread_row;
        const int a_global_col = (k_tile_idx * BK) + a_thread_col;
        *reinterpret_cast<__nv_bfloat162*>(&sA[a_thread_row * BK + a_thread_col]) = *reinterpret_cast<__nv_bfloat162*>(&A[a_global_row * K + a_global_col]);
    }

    // Load tile of B from GMEM into SMEM
    #pragma unroll
    for (int load_idx = 0; load_idx < total_b_loads; load_idx++) {
        const int linear_idx = load_idx * loads_per_iter + (threadIdx.x * bfloats_per_load);
        const int b_thread_row = linear_idx / BN;
        const int b_thread_col = linear_idx % BN;
        const int b_global_row = (k_tile_idx * BK) + b_thread_row; 
        const int b_global_col = (block_col * BN) + b_thread_col;
        *reinterpret_cast<__nv_bfloat162*>(&sB[b_thread_row * BN + b_thread_col]) = *reinterpret_cast<__nv_bfloat162*>(&B[b_global_row * N + b_global_col]);
    }
}

template<
    int BM = 128,
    int BN = 128,
    int BK = 16,
    int WMMA_M = 16,
    int WMMA_N = 16,
    int WMMA_K = 16,
    // how many wmma tiles each warp should do
    int WARP_TILES_M = 2,
    int WARP_TILES_N = 2
>
__global__ void gemm(__nv_bfloat16* A, __nv_bfloat16* B, float* C, int M, int N, int K) {
    alignas(16) __shared__ __nv_bfloat16 sA[NUM_BUFFERS][BM * BK]; // 64*16
    alignas(16) __shared__ __nv_bfloat16 sB[NUM_BUFFERS][BK * BN]; // 16*64
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    constexpr int warps_per_row = BN / (WARP_TILES_N * WMMA_N); // 64 / (2*16) = 2
    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;
    const int num_threads = blockDim.x;

    // fragments for A and B (similar to a_reg/b_reg in warptile kernel)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

    // accumlators for each warp tile, init with 0s. accumulate in fp32
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
        for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
            wmma::fill_fragment(c_frag[warp_tile_m][warp_tile_n], 0.0f);
        }
    }

    // prologue - preload first buffer
    int read_buf_idx = 0;
    int write_buf_idx = 0;
    load_gmem_to_smem<BM, BN, BK>(A, B, &sA[write_buf_idx][0], &sB[write_buf_idx][0], 0, block_row, block_col, num_threads, M, N, K);
    write_buf_idx ^= 1; // toggle next buffer to write to

    __syncthreads();

    const int num_k_tiles = (K + BK - 1) / BK;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < num_k_tiles; tile_idx++) {
        // prefetch next a/b tiles into next buffer.
        // note this is not async in this kernel version, so benefits are limited (one fewer syncthreads())
        if (tile_idx + 1 < num_k_tiles)
        {
            load_gmem_to_smem<BM, BN, BK>(A, B, &sA[write_buf_idx][0], &sB[write_buf_idx][0], tile_idx + 1, block_row, block_col, num_threads, M, N, K);
            write_buf_idx ^= 1;
        }

        // wmma on each warp tile this warp is responsible for
        for (int k = 0; k < BK; k += WMMA_K) {
            for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
                for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
                    // cache col fragment of A
                    const int smem_a_row = (warp_row * WARP_TILES_M * WMMA_M) + (warp_tile_m * WMMA_M);
                    const int smem_a_col = k;
                    __nv_bfloat16* smem_tile_a = &sA[read_buf_idx][smem_a_row * BK + smem_a_col];
                    wmma::load_matrix_sync(a_frag, smem_tile_a, BK);

                    // cache row fragment of B
                    const int smem_b_row = k;
                    const int smem_b_col = (warp_col * WARP_TILES_N * WMMA_N) + (warp_tile_n * WMMA_N);
                    __nv_bfloat16* smem_tile_b = &sB[read_buf_idx][smem_b_row * BN + smem_b_col];
                    wmma::load_matrix_sync(b_frag, smem_tile_b, BN);

                    // accumulate outer product
                    wmma::mma_sync(c_frag[warp_tile_m][warp_tile_n], a_frag, b_frag, c_frag[warp_tile_m][warp_tile_n]);
                }
            }
        }
        read_buf_idx ^= 1; // toggle next buffer to read from
        __syncthreads();
    }

    // store output
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
        for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
            int c_row = (block_row * BM) + (warp_row * WARP_TILES_M * WMMA_M) + (warp_tile_m * WMMA_M);
            int c_col = (block_col * BN) + (warp_col * WARP_TILES_N * WMMA_N) + (warp_tile_n * WMMA_N);
            float* c_ptr = &C[c_row * N + c_col];
            wmma::store_matrix_sync(c_ptr, c_frag[warp_tile_m][warp_tile_n], N, wmma::mem_row_major);
        }
    }
}

extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K) {
    // Cast void* to __nv_bfloat16*
    __nv_bfloat16* a_ptr = reinterpret_cast<__nv_bfloat16*>(A);
    __nv_bfloat16* b_ptr = reinterpret_cast<__nv_bfloat16*>(B);
    float* c_ptr = reinterpret_cast<float*>(C);

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_TILES_M = 2;
    constexpr int WARP_TILES_N = 2;
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    // dims for smem tiles
    constexpr int BM = BLOCK_SIZE; // 64
    constexpr int BN = BLOCK_SIZE; // 64
    constexpr int BK = 16;

    constexpr int num_warps = (BLOCK_SIZE * BLOCK_SIZE) / (WARP_TILES_M * WMMA_M * WARP_TILES_N * WMMA_N); // (64*64) / (2*16*2*16) = 4 warps
    constexpr int threadblock_size = num_warps * 32; // 4*32 = 128 threads

    dim3 block_dim(threadblock_size);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    gemm<BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILES_M, WARP_TILES_N><<<grid_dim, block_dim>>>(a_ptr, b_ptr, c_ptr, M, N, K);
}
