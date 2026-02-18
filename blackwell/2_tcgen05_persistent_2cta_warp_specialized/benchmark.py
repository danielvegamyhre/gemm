import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench

custom_gemm = load(
    name='tcgen05_persistent_2cta_warp_specialized',
    sources=['tcgen05_persistent_2cta_warp_specialized.cpp', 'tcgen05_persistent_2cta_warp_specialized.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_100a,code=sm_100a'],
    extra_cflags=['-O3'],
    verbose=False
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

        # Benchmark custom kernel
        torch.cuda.synchronize()
        custom_us = do_bench(lambda: custom_gemm.gemm_cuda(A, B.t(), C), warmup=WARMUP, rep=REP, return_mode="median") * 1e3

        # Benchmark PyTorch
        out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        torch_us = do_bench(lambda: torch.mm(A, B.t(), out_dtype=out.dtype, out=out), warmup=WARMUP, rep=REP, return_mode="median") * 1e3

        # Calculate tflops
        flops = 2.0 * M * N * K
        custom_tflops = (flops / 1e12) / (custom_us / 1e6)
        torch_tflops = (flops / 1e12) / (torch_us / 1e6)

        print(f"  Custom kernel:  {custom_us:7.3f} us ({custom_tflops:6.2f} tflops)")
        print(f"  PyTorch matmul: {torch_us:7.3f} us ({torch_tflops:6.2f} tflops)")
        print(f"  Speedup: {torch_us/custom_us:.2f}x")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    benchmark()
