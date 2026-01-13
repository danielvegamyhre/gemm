#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

// forward declare
extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K);

torch::Tensor gemm_cuda(
    torch::Tensor A,    // Input tensor A (M x K)
    torch::Tensor B,    // Input tensor B (K x N)
    torch::Tensor C     // Output tensor C (M x N)
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    launch_gemm(
        A.data_ptr<at::BFloat16>(), 
        B.data_ptr<at::BFloat16>(), 
        C.data_ptr<float>(), 
        M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cuda", &gemm_cuda, "Naive GEMM CUDA kernel");
}
