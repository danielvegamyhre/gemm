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
    assert(A_scales.size(0) % 512 == 0);
    assert(B_scales.size(0) % 512 == 0);

    launch_gemm(
        A.data_ptr<uint8_t>(), 
        B.data_ptr<uint8_t>(), 
        A_scales.data_ptr<uint8_t>(),
        B_scales.data_ptr<uint8_t>(),
        C.data_ptr<float>(), 
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cuda", &gemm_cuda, "Naive GEMM CUDA kernel");
}
