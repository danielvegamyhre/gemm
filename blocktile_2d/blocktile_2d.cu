#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 8

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int a_base_row = block_row * BM;
    const int b_base_col = block_col * BN;
    const int c_row = block_row * BM + (threadIdx.x / BN) * TM; // TM rows per thread
    const int c_col = block_col * BN + (threadIdx.x % BN) * TN; // TN cols per thread

    // 4 results per thread along M dim
    float thread_results[TM][TN] = {0.0f};
    const int num_tiles = (K + BK - 1) / BK;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // We have divided BM/TM and BN/TN so we don't have enough threads to load the full required tile size of A/B at once, we must loop.
        // Load tile of A from GMEM into SMEM.
        const int loads_per_iter = blockDim.x;
        const int total_a_loads = (BM * BK) / loads_per_iter;
        for (int load_idx = 0; load_idx < total_a_loads; load_idx++) {
            const int linear_idx = loads_per_iter * load_idx + threadIdx.x;
            const int a_thread_row = linear_idx / BK;
            const int a_thread_col = linear_idx % BK;
            const int a_global_row = a_base_row + a_thread_row;
            const int a_global_col = tile_idx * BK + a_thread_col;
            printf("tile_idx=%d, load_idx=%d, bid_x=%d, bid_y=%d, tid=%d loading sA=[%d, %d] = A[%d, %d]\n", tile_idx, load_idx, blockIdx.x, blockIdx.y, threadIdx.x, a_thread_row, a_thread_col, a_global_row, a_global_col);
            if (a_global_row < M && a_global_col < K)
            {
                sA[a_thread_row * BK + a_thread_col] = A[a_global_row * K + a_global_col];
            }
            else
            {
                sA[a_thread_row * BK + a_thread_col] = 0.0f;
            }
        }

        // Load tile of B from GMEM into SMEM
        const int total_b_loads = (BK * BN) / loads_per_iter;
        for (int load_idx = 0; load_idx < total_b_loads; load_idx++) {
            const int linear_idx = loads_per_iter * load_idx + threadIdx.x;
            const int b_thread_row = linear_idx / BN;
            const int b_thread_col = linear_idx % BN;
            const int b_global_row = tile_idx * BK + b_thread_row;
            const int b_global_col = b_base_col + b_thread_col;
            printf("tile_idx=%d, load_idx=%d, bid_x=%d, bid_y=%d, tid=%d Loading sB=[%d, %d] = B[%d, %d]\n", tile_idx, load_idx, blockIdx.x, blockIdx.y, threadIdx.x, b_thread_row, b_thread_col, b_global_row, b_global_col);
            if (b_global_row < K && b_global_col < N) 
            {
                sB[b_thread_row * BN + b_thread_col] = B[b_global_row * N + b_global_col];
            }
            else 
            {
                sB[b_thread_row * BN + b_thread_col] = 0.0f;
            }
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float a_reg[TM] = {0.0f};
            float b_reg[TN] = {0.0f};

            // cache col of A in registers
            const int a_smem_col = k;
            for (int tm = 0; tm < TM; tm++) {
                const int a_smem_row = (threadIdx.x/BK) * TM + tm;
                a_reg[tm] = sA[a_smem_row * BK + a_smem_col];
            }
            // cache row of B in registers
            const int b_smem_row = k;
            for (int tn = 0; tn < TN; tn++) {
                const int b_smem_col = ((threadIdx.x * TN) + tn) % BN;
                b_reg[tn] = sB[b_smem_row * BN + b_smem_col];
            }
            // accumulate outer product
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    thread_results[tm][tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }
        __syncthreads();
    }
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            if (c_row + tm < M && c_col + tn < N) {
                C[(c_row + tm) * N + (c_col + tn)] = thread_results[tm][tn];
            }
        }
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    constexpr int TM = 4; // 4 results per thread along M dim
    constexpr int TN = 4; // 4 result per thread along N dim
    constexpr int BM = BLOCK_SIZE;
    constexpr int BN = BLOCK_SIZE;
    constexpr int BK = BLOCK_SIZE / TM; 

    printf("TM=%d, TN=%d, BM=%d, BN=%d, BK=%d, blockDim.x=%d, blockDim.y=%d\n", TM, TN, BM, BN, BK, (BN/TN), (BM/TM));

    dim3 block_dim((BM/TM) * (BN/TN));  
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    gemm<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
