import pytest
import torch
from torch.utils.cpp_extension import load

custom_gemm = load(
    name='no_swizzle_pipeline_tma_wgmma',
    sources=['no_swizzle_pipeline_tma_wgmma.cpp', 'no_swizzle_pipeline_tma_wgmma.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode=arch=compute_90a,code=sm_90a', '--keep'],
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
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(K, N, device='cuda', dtype=torch.bfloat16).t().contiguous().t()
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    result = custom_gemm.gemm_cuda(A, B, C)

    expected = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    torch.mm(A, B, out_dtype=torch.float32, out=expected)

    print()
    print(result)
    print()
    print(expected)
    torch.testing.assert_close(result, expected)
