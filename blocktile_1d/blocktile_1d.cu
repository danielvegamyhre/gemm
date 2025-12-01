#include <cuda_runtime.h>
#define BLOCK_SIZE 16

template <int TM>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = (threadIdx.x / BLOCK_SIZE);
    int thread_col = (threadIdx.x % BLOCK_SIZE);
    int global_row = block_row * BLOCK_SIZE + thread_row * TM;
    int global_col = block_col * BLOCK_SIZE + thread_col;

    // Move pointers to starting points for this block,
    // to simplify pointer arithmetic below (we only have to worry about
    // threadblock local indexing).
    A += block_row * BLOCK_SIZE * K;
    B += block_col * BLOCK_SIZE;

    // 4 results per thread along M dim
    float thread_results[TM] = {0.0f};
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        for (int tm = 0; tm < TM; tm++) {
            // Load tile of A from GMEM into SMEM
            if ((global_row + tm) < M && (tile_idx * BLOCK_SIZE + thread_col) < K)
            {
                sA[(thread_row * TM + tm) * BLOCK_SIZE + thread_col] = A[(thread_row * TM + tm) * K + thread_col];
            }
            else
            {
                sA[(thread_row * TM + tm) * BLOCK_SIZE + thread_col] = 0.0f;
            }

            // Load tile of B from GMEM into SMEM
            if (global_col < N && (tile_idx * BLOCK_SIZE + thread_row) < K)
            {
                sB[(thread_row * TM + tm) * BLOCK_SIZE + thread_col] = B[(thread_row * TM + tm) * N + thread_col];
            }
            else
            {
                sB[(thread_row * TM + tm) * BLOCK_SIZE + thread_col] = 0.0f;
            }
        }
        __syncthreads();

        // Dot products with data in smem
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float b_reg = sB[k * BLOCK_SIZE + thread_col];
            for (int tm = 0; tm < TM; tm++) {
                thread_results[tm] += sA[(thread_row * TM + tm) * BLOCK_SIZE + k] * b_reg;
            }
        }

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }
    for (int tm = 0; tm < TM; tm++) {
        if ((global_row + tm) < M && global_col < N) {
            C[(global_row + tm) * N + global_col] = thread_results[tm];
        }
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    constexpr int TM = 4; // 4 results per thread along M dim
    int block_size_m = BLOCK_SIZE / TM;
    int block_size_n = BLOCK_SIZE;
    dim3 block_dim(block_size_m * block_size_n);
    dim3 grid_dim(ceil_div(N, block_size_n), ceil_div(M, block_size_m));
    gemm<TM><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
