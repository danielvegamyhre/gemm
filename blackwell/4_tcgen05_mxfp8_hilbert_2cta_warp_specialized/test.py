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
        A_scales_blocked.view(torch.uint8), 
        B_scales_blocked.view(torch.uint8), 
        C
    )
    
    expected = torch._scaled_mm(
        A_data, 
        B_data.t(), 
        A_scales_blocked, 
        B_scales_blocked,
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

    # Check for mismatches
    rtol = 1e-2
    atol = 1e-2

    diff = torch.abs(result - expected)
    rel_diff = diff / (torch.abs(expected) + 1e-8)

    mismatch_mask = (diff > atol) & (rel_diff > rtol)
    num_mismatches = mismatch_mask.sum().item()
    total_elements = M * N

    print(f"\n{'='*80}")
    print(f"GEMM Verification Results: M={M}, N={N}, K={K}")
    print(f"{'='*80}")
    print(f"Total elements: {total_elements}")
    print(f"Mismatched elements: {num_mismatches} ({100.0 * num_mismatches / total_elements:.2f}%)")

    if num_mismatches > 0:
        mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)

        # Analyze mismatch distribution
        mismatch_rows = mismatch_indices[:, 0]
        mismatch_cols = mismatch_indices[:, 1]

        print(f"\nMismatch distribution:")
        print(f"  Row range: [{mismatch_rows.min().item()}, {mismatch_rows.max().item()}]")
        print(f"  Col range: [{mismatch_cols.min().item()}, {mismatch_cols.max().item()}]")

        # Count mismatches per 128-row block (BM=128)
        BM = 128
        BN = 256
        print(f"\nMismatches per {BM}-row block:")
        for block_m in range((M + BM - 1) // BM):
            start_row = block_m * BM
            end_row = min(start_row + BM, M)
            block_mask = (mismatch_rows >= start_row) & (mismatch_rows < end_row)
            block_count = block_mask.sum().item()
            if block_count > 0:
                print(f"  Rows [{start_row:4d}:{end_row:4d}): {block_count:6d} mismatches")

        print(f"\nMismatches per {BN}-col block:")
        for block_n in range((N + BN - 1) // BN):
            start_col = block_n * BN
            end_col = min(start_col + BN, N)
            block_mask = (mismatch_cols >= start_col) & (mismatch_cols < end_col)
            block_count = block_mask.sum().item()
            if block_count > 0:
                print(f"  Cols [{start_col:4d}:{end_col:4d}): {block_count:6d} mismatches")

        # Show first 10 mismatches
        print(f"\nFirst 10 mismatches (showing up to 10):")
        for idx in range(min(10, num_mismatches)):
            i, j = mismatch_indices[idx]
            result_val = result[i, j].item()
            expected_val = expected[i, j].item()
            diff_val = diff[i, j].item()
            print(f"  [{i:4d}, {j:4d}]: result={result_val:10.4f}, expected={expected_val:10.4f}, diff={diff_val:10.4f}")

        print(f"\n{'='*80}")
        raise AssertionError(f"Found {num_mismatches} mismatches out of {total_elements} elements")
    else:
        print(f"\n✓ All elements match within tolerance (rtol={rtol}, atol={atol})")
        print(f"{'='*80}\n")
