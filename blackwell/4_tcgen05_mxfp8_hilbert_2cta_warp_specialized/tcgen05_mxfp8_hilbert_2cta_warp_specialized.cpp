#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

// forward declare
extern "C" void launch_gemm(void* A, void* B, void* A_scales, void* B_scales, void* C, int M, int N, int K);

torch::Tensor gemm_cuda(
    torch::Tensor A,            // (M x K)
    torch::Tensor B,            // (K x N)
    torch::Tensor A_scales,     // (M, K//32) 
    torch::Tensor B_scales,     // (N, K//32) 
    torch::Tensor C             // (M x N)
) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    assert(A_scales.size(0) % 128 == 0);
    assert(A_scales.size(1) % 4 == 0);
    assert(B_scales.size(0) % 128 == 0);
    assert(B_scales.size(1) % 4 == 0);

    launch_gemm(
        A.data_ptr<at::BFloat16>(), 
        B.data_ptr<at::BFloat16>(), 
        A_scales.data_ptr<at::kFloat8_e8m0fnu>,
        B_scales.data_ptr<at::kFloat8_e8m0fnu>,
        C.data_ptr<float>(), 
        M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cuda", &gemm_cuda, "Naive GEMM CUDA kernel");
}
