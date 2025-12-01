import pytest
import torch
from torch.utils.cpp_extension import load

custom_gemm = load(
    name='blocktile_1d',
    sources=['blocktile_1d.cpp', 'blocktile_1d.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    extra_cflags=['-O3'],
    verbose=False
)


@pytest.mark.parametrize("M,K,N", [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024)
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gemm(M, K, N):
    """Test that shared memory GEMM produces correct results compared to PyTorch matmul."""
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    result = custom_gemm.gemm_cuda(A, B, C)
    expected = torch.matmul(A, B)
    
    torch.testing.assert_close(result, expected)
