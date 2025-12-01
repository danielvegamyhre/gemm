import pytest
import torch
from torch.utils.cpp_extension import load

naive_gemm_cuda = load(
    name='naive_gemm_cuda',
    sources=['naive.cpp', 'naive_1.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cflags=['-O3'],
    verbose=False
)

@pytest.mark.parametrize("M,K,N", [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_naive_gemm_correctness(M, K, N):
    """Test that naive GEMM produces correct results compared to PyTorch matmul."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    result = naive_gemm_cuda.naive_gemm(A, B, C)
    expected = torch.matmul(A, B)
    print(f"result: {result}")
    print(f"expected: {expected}")
    torch.testing.assert_close(result, expected)
