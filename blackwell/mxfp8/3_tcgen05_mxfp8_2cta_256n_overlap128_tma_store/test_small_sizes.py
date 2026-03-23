import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import to_blocked

custom_gemm = load(
    name="tcgen05_mxfp8_2cta_256n_overlap128_tma_store",
    sources=["tcgen05_mxfp8_2cta_256n_overlap128_tma_store.cpp", "tcgen05_mxfp8_2cta_256n_overlap128_tma_store.cu"],
    extra_cuda_cflags=["-O3", "-lineinfo", "--use_fast_math", "-gencode=arch=compute_100a,code=sm_100a"],
    extra_cflags=["-O3"],
    verbose=False,
)

WARMUP, REP = 50, 500
sizes = [(1024, 1024, 1024), (1536, 1536, 1536), (2048, 2048, 2048), (3072, 3072, 3072)]

for M, K, N in sizes:
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    A_data, A_scales = triton_to_mxfp8_dim0(A)
    B_data, B_scales = triton_to_mxfp8_dim0(B)
    A_scales_blocked, B_scales_blocked = to_blocked(A_scales), to_blocked(B_scales)
    
    custom_us = do_bench(lambda: custom_gemm.gemm_cuda(A_data.view(torch.uint8), B_data.t().view(torch.uint8), 
                                                         A_scales_blocked.view(torch.uint8), B_scales_blocked.view(torch.uint8), C),
                         warmup=WARMUP, rep=REP, return_mode="median") * 1e3
    
    flops = 2.0 * M * N * K
    custom_tflops = (flops / 1e12) / (custom_us / 1e6)
    print(f"{M:4d}³: {custom_us:7.3f} us ({custom_tflops:7.2f} tflops)")
