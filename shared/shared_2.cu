#include <cuda_runtime.h>
#define BLOCK_SIZE 16

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = (threadIdx.x / BLOCK_SIZE);
    int thread_col = (threadIdx.x % BLOCK_SIZE);
    int global_row = block_row * BLOCK_SIZE + thread_row;
    int global_col = block_col * BLOCK_SIZE + thread_col;

    // Move pointers to starting points for this block,
    // to simplify pointer arithmetic below (we only have to worry about
    // threadblock local indexing).
    A += block_row * BLOCK_SIZE * K;
    B += block_col * BLOCK_SIZE;

    float acc = 0.0f;
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load tile of A from GMEM into SMEM
        if (global_row < M && (tile_idx * BLOCK_SIZE + thread_col) < K)
        {
            sA[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        }
        else
        {
            sA[thread_row * BLOCK_SIZE + thread_col] = 0.0f;
        }

        // Load tile of B from GMEM into SMEM
        if (global_col < N && (tile_idx * BLOCK_SIZE + thread_row) < K)
        {
            sB[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];
        }
        else
        {
            sB[thread_row * BLOCK_SIZE + thread_col] = 0.0f;
        }
        __syncthreads();

        // Dot products with data in smem
        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += sA[thread_row * BLOCK_SIZE + k] * sB[k * BLOCK_SIZE + thread_col];
        }

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        __syncthreads();
    }
    if ((global_row < M) && (global_col < N)) {
        C[global_row * N + global_col] = acc;
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    dim3 block_dim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 grid_dim(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));
    sgemm<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
