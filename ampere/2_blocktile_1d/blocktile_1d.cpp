#include <torch/extension.h>
#include <cuda_runtime.h>

void launch_gemm(float* A, float* B, float* C, int M, int N, int K);

torch::Tensor gemm_cuda(
    torch::Tensor A,    // Input tensor A (M x K)
    torch::Tensor B,    // Input tensor B (K x N)
    torch::Tensor C     // Output tensor C (M x N)
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    launch_gemm(A_ptr, B_ptr, C_ptr, M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cuda", &gemm_cuda, "Naive GEMM CUDA kernel");
}
