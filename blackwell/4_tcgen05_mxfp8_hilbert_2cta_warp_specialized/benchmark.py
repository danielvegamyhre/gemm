import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import to_blocked

custom_gemm = load(
    name='tcgen05_mxfp8_hilbert_2cta_warp_specialized',
    sources=['tcgen05_mxfp8_hilbert_2cta_warp_specialized.cpp', 'tcgen05_mxfp8_hilbert_2cta_warp_specialized.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_100a,code=sm_100a', '--ptxas-options=-v'],
    extra_cflags=['-O3'],
    verbose=True
)

def benchmark():
    WARMUP = 50
    REP = 500

    sizes = [
        (4096, 4096, 4096),
        (16384, 16384, 16384),
    ]

    for M, K, N in sizes:
        print(f"\nMatrix size: M={M}, K={K}, N={N}")

        # Create test data
        A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        B = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
        C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

        # Quantize to MXFP8 with scales
        A_data, A_scales = triton_to_mxfp8_dim0(A)
        B_data, B_scales = triton_to_mxfp8_dim0(B)
        A_scales_blocked, B_scales_blocked = to_blocked(A_scales), to_blocked(B_scales)

        # Benchmark custom kernel
        torch.cuda.synchronize()
        custom_us = do_bench(
            lambda: custom_gemm.gemm_cuda(
                A_data.view(torch.uint8),
                B_data.t().view(torch.uint8),
                A_scales_blocked.view(torch.uint8),
                B_scales_blocked.view(torch.uint8),
                C
            ),
            warmup=WARMUP,
            rep=REP,
            return_mode="median"
        ) * 1e3

        # Benchmark PyTorch _scaled_mm
        torch.cuda.synchronize()
        torch_us = do_bench(
            lambda: torch._scaled_mm(
                A_data,
                B_data.t(),
                A_scales_blocked,
                B_scales_blocked,
                out_dtype=torch.float32,
            ),
            warmup=WARMUP,
            rep=REP,
            return_mode="median"
        ) * 1e3

        # Calculate tflops
        flops = 2.0 * M * N * K
        custom_tflops = (flops / 1e12) / (custom_us / 1e6)
        torch_tflops = (flops / 1e12) / (torch_us / 1e6)

        print(f"  Custom kernel:       {custom_us:7.3f} us ({custom_tflops:6.2f} tflops)")
        print(f"  PyTorch _scaled_mm:  {torch_us:7.3f} us ({torch_tflops:6.2f} tflops)")
        print(f"  Speedup: {torch_us/custom_us:.2f}x")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    benchmark()
