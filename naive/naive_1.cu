#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    int j = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    if ((i < M) && (j < N)) {
        float acc = 0.0f;
        for (int k=0; k < K; k++) {
            acc += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }
}

void launch_naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    dim3 block_dim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 grid_dim(ceil_div(N, BLOCK_SIZE), ceil_div(M, BLOCK_SIZE));
    naive_gemm<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
