#include <cuda_runtime.h>

#define BLOCK_SIZE 64

template <int BM, int BN, int BK, int TM>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int a_base_row = block_row * BM;
    const int b_base_col = block_col * BN;
    const int c_row = block_row * BM + (threadIdx.x / BN) * TM; // TM rows per thread
    const int c_col = block_col * BN + (threadIdx.x % BN);

    // 4 results per thread along M dim
    float thread_results[TM] = {0.0f};
    const int num_tiles = (K + BK - 1) / BK;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load tile of A from GMEM into SMEM
        const int a_block_row = threadIdx.x / BK;
        const int a_block_col = threadIdx.x % BK;
        const int a_global_row = a_base_row + a_block_row;
        const int a_global_col = tile_idx * BK + a_block_col;
        if (a_global_row < M && a_global_col < K)
        {
            sA[a_block_row * BK + a_block_col] = A[a_global_row * K + a_global_col];
        }
        else
        {
            sA[a_block_row * BK + a_block_col] = 0.0f;
        }

        // Load tile of B from GMEM into SMEM
        const int b_block_row = threadIdx.x / BN;
        const int b_block_col = threadIdx.x % BN;
        const int b_global_row = tile_idx * BK + b_block_row;
        const int b_global_col = b_base_col + b_block_col;
        if (b_global_row < K && b_global_col < N)
        {
            sB[b_block_row * BN + b_block_col] = B[b_global_row * N + b_global_col];
        }
        else
        {
            sB[b_block_row * BN + b_block_col] = 0.0f;
        }
        __syncthreads();

        // For each elem in a column from sB, compute TM dot products at once,
        // using a column of TM size from A from SMEM and a single elem from B
        // cached in a register.
        for (int k = 0; k < BK; k++) {
            float b_reg = sB[k * BN + b_block_col];
            for (int tm = 0; tm < TM; tm++) {
                // computing blocks of size (BM, BN), where each thread computes a column of TM results per iteration
                const int compute_row = (threadIdx.x / BN) * TM + tm;
                thread_results[tm] += sA[compute_row * BK + k] * b_reg;
            }
        }

        __syncthreads();
    }
    for (int tm = 0; tm < TM; tm++) {
        if (c_row + tm < M && c_col < N) {
            C[(c_row + tm) * N + c_col] = thread_results[tm];
        }
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    constexpr int TM = 4; // 4 results per thread along M dim
    constexpr int BM = BLOCK_SIZE;
    constexpr int BN = BLOCK_SIZE;
    // divide by TM so we can load (BM, BK) tiles of A and (BK, BN) tiles of B with 1 load per thread, rather than having to do TM loops/loads per thread
    constexpr int BK = BLOCK_SIZE / TM;
    dim3 block_dim((BM/TM) * BN);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    gemm<BM, BN, BK, TM><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
