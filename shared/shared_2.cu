#include <cuda_runtime.h>
#define BLOCK_SIZE 16

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE * BLOCK_SIZE];
    int block_row = (threadIdx.x / BLOCK_SIZE);
    int block_col = (threadIdx.x % BLOCK_SIZE);
    int global_row = blockDim.x * blockIdx.x + block_row;
    int global_col = blockDim.x * blockIdx.x + block_col;
    if ((global_row < M) && (global_col < N)) {
        int k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int k_block = 0; k_block < k_blocks; k_block++) {
            // Load tiles into smem
            sA[block_row * BLOCK_SIZE + block_col] = A[global_row * K + (global_col + k_block * BLOCK_SIZE)];
            sB[block_row * BLOCK_SIZE + block_col] = B[(global_row + k_block * BLOCK_SIZE) * N + global_col];
            __syncthreads();

            // Dot products with data in smem
            float acc = 0.0f;
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                acc += sA[block_row * BLOCK_SIZE + k] * sB[k * BLOCK_SIZE + block_col];
            }
            C[global_row * N + global_col] = acc;
        }
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int block_size = 16;
    auto round_up = [](int x, int y) {
        return (x + y - 1) / y;
    };
    int block_dim = block_size * block_size;
    int grid_dim = round_up(M * N, block_dim);
    sgemm<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
