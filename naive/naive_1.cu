#include <cuda_runtime.h>

#define BLOCK_SIZE 16
 
__global__ void naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int i = blockIdx.x * blockDim.x + (threadIdx.x / BLOCK_SIZE);
    int j = blockIdx.x * blockDim.x + (threadIdx.x % BLOCK_SIZE);
    if ((i < M) && (j < N)) {
        float acc = 0.0f;
        for (int k=0; k < K; ++k) {
            acc += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }
}

void launch_naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int block_size = 16;
    auto round_up = [](int x, int y) {
        return (x + y - 1) / y;
    };
    int block_dim = block_size * block_size;
    int grid_dim = round_up(M * N, block_dim);
    naive_gemm<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
