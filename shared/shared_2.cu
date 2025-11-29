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

    // Move pointers to starting points for this block
    A += block_row * BLOCK_SIZE * K;
    B += block_col * BLOCK_SIZE;  
    C += block_row * BLOCK_SIZE * N + block_col * BLOCK_SIZE;

    float acc = 0.0f;
    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Load tiles into smem
        sA[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        sB[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];
        __syncthreads();
        
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
        
        // Dot products with data in smem
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += sA[thread_row * BLOCK_SIZE + k] * sB[k * BLOCK_SIZE + thread_col];
        }
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
