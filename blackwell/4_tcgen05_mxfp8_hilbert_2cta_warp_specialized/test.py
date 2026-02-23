import pytest
import torch
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_mm, ScalingType, SwizzleType
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import to_blocked

custom_gemm = load(
    name='tcgen05_mxfp8_hilbert_2cta_warp_specialized',
    sources=['tcgen05_mxfp8_hilbert_2cta_warp_specialized.cpp', 'tcgen05_mxfp8_hilbert_2cta_warp_specialized.cu'],
    extra_cuda_cflags=['-g','-G','--generate-line-info','-gencode=arch=compute_100a,code=sm_100a'],
    extra_cflags=['-O3'],
    verbose=False
)


@pytest.mark.parametrize("M,K,N", [
    (2048, 4096, 8192)
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gemm(M, K, N):
    torch.manual_seed(42)
    A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

    A_data, A_scales = triton_to_mxfp8_dim0(A)
    B_data, B_scales = triton_to_mxfp8_dim0(B)
    A_scales_blocked, B_scales_blocked = to_blocked(A_scales), to_blocked(B_scales)

    result = custom_gemm.gemm_cuda(
        A_data.view(torch.uint8), 
        B_data.t().view(torch.uint8), 
        torch.ones_like(A_scales_blocked).view(torch.uint8), 
        torch.ones_like(B_scales_blocked).view(torch.uint8), 
        C
    )
    
    expected = torch._scaled_mm(
        A_data, 
        B_data.t(), 
        torch.ones_like(A_scales_blocked), 
        torch.ones_like(B_scales_blocked),
        out_dtype=torch.float32,
    )
    # expected = scaled_mm(
    #     A_data, 
    #     B_data.t(), 
    #     scale_a=A_scales_blocked, 
    #     scale_recipe_a=ScalingType.BlockWise1x32, 
    #     scale_b=B_scales_blocked,
    #     scale_recipe_b=ScalingType.BlockWise1x32, 
    #     swizzle_a=SwizzleType.SWIZZLE_32_4_4,
    #     swizzle_b=SwizzleType.SWIZZLE_32_4_4,
    #     output_dtype=torch.float32,
    # )
    print(result)
    print()
    print(expected)
    
    assert torch.allclose(result, expected, atol=0, rtol=0)