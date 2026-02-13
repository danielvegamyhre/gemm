import torch
from torch.utils.cpp_extension import load
from triton.testing import do_bench

custom_gemm = load(
    name='tcgen05_hilbert_2cta_warp_specialized',
    sources=['tcgen05_hilbert_2cta_warp_specialized.cpp', 'tcgen05_hilbert_2cta_warp_specialized.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_100a,code=sm_100a', '--ptxas-options=-v'],
    extra_cflags=['-O3'],
    verbose=True
)

def benchmark():
    def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
        return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3

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

        # Warmup
        for _ in range(5):
            custom_gemm.gemm_cuda(A, B.t(), C)

        # Benchmark custom kernel
        custom_us = benchmark_cuda_function_in_microseconds(
            custom_gemm.gemm_cuda,
            A,
            B.t(),
            C,
        )

        # Benchmark PyTorch
        out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        for _ in range(5):
            torch.mm(A, B.t(), out_dtype=torch.float32, out=out)

        torch_us = benchmark_cuda_function_in_microseconds(
            torch.mm,
            A,
            B.t(),
            out_dtype=torch.float32,
            out=out,
        )
        torch.cuda.synchronize()

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
