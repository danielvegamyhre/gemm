import torch
from torch.utils.cpp_extension import load
import time
from triton.testing import do_bench

# Load the extension (will use cached version if already built)
naive_gemm_cuda = load(
    name='naive_gemm_cuda',
    sources=['naive.cpp', 'naive_1.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cflags=['-O3'],
    verbose=False
)

def test_correctness():
    # Small test case
    M, K, N = 4, 3, 5

    # Create simple test matrices
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    result = naive_gemm_cuda.naive_gemm(A, B, C)
    expected = torch.matmul(A, B)
    torch.testing.assert_close(result, expected)
    print("Tests passed")

def benchmark():
    def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
        return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3

    sizes = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)]

    for M, K, N in sizes:
        print(f"\nMatrix size: {M}x{K}x{N}")

        # Create test data
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(5):
            naive_gemm_cuda.naive_gemm(A, B, C)
        torch.cuda.synchronize()

        # Benchmark custom kernel
        custom_us = benchmark_cuda_function_in_microseconds(
            naive_gemm_cuda.naive_gemm,
            A,
            B,
            C,
        )

        # Benchmark PyTorch
        for _ in range(5):
            torch.matmul(A, B, out=C)
        torch.cuda.synchronize()
        
        torch_us = benchmark_cuda_function_in_microseconds(
            torch.matmul,
            A,
            B,
            out=C,
        )

        # Calculate tflops
        flops = 2.0 * M * N * K
        custom_tflops = (flops / 1e12) / (custom_us / 1e6)
        torch_tflops = (flops / 1e12) / (torch_us / 1e6)

        print(f"  Custom kernel:  {custom_us:7.3f} ms ({custom_tflops:6.2f} tflops)")
        print(f"  PyTorch matmul: {torch_us:7.3f} ms ({torch_tflops:6.2f} tflops)")
        print(f"  Speedup: {torch_us/custom_us:.2f}x")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")

    passed = test_correctness()
    benchmark()
