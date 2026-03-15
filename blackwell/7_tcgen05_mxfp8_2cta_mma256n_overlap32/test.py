import pytest
import torch
from torch.utils.cpp_extension import load
from torch.nn.functional import scaled_mm, ScalingType, SwizzleType
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import to_blocked

custom_gemm = load(
    name="tcgen05_mxfp8_2cta_256n_overlap32",
    sources=[
        "tcgen05_mxfp8_2cta_256n_overlap32.cpp",
        "tcgen05_mxfp8_2cta_256n_overlap32.cu",
    ],
    extra_cuda_cflags=[
        "-g",
        "-G",
        "--generate-line-info",
        "-gencode=arch=compute_100a,code=sm_100a",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)


@pytest.mark.parametrize("M,K,N", [(256, 256, 256), (4096, 4096, 4096), (2048, 4096, 8192), (16384, 16384, 16384)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gemm(M, K, N):
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    A_data, A_scales = triton_to_mxfp8_dim0(A)
    B_data, B_scales = triton_to_mxfp8_dim0(B)
    A_scales_blocked, B_scales_blocked = to_blocked(A_scales), to_blocked(B_scales)

    result = custom_gemm.gemm_cuda(
        A_data.view(torch.uint8),
        B_data.t().view(torch.uint8),
        A_scales_blocked.view(torch.uint8),
        B_scales_blocked.view(torch.uint8),
        C,
    )

    expected = torch._scaled_mm(
        A_data,
        B_data.t(),
        A_scales_blocked,
        B_scales_blocked,
        out_dtype=torch.float32,
    )

    print(result)
    print()
    print(expected)

    print("=" * 80)
    print("RESULT STATISTICS")
    print("=" * 80)

    # Check for mismatches
    mismatch_mask = ~torch.isclose(result, expected, atol=1e-5, rtol=1e-3)
    num_mismatches = mismatch_mask.sum().item()
    total_elements = result.numel()

    print(f"Total elements: {total_elements}")
    print(f"Mismatches: {num_mismatches} ({100*num_mismatches/total_elements:.2f}%)")

    # Check for zeros in result
    result_zeros = result == 0
    expected_zeros = expected == 0
    unexpected_zeros = result_zeros & ~expected_zeros

    print(f"\nZeros in result: {result_zeros.sum().item()}")
    print(f"Zeros in expected: {expected_zeros.sum().item()}")
    print(f"Unexpected zeros: {unexpected_zeros.sum().item()}")

    # Analyze by row
    print("\n" + "=" * 80)
    print("PER-ROW ANALYSIS")
    print("=" * 80)
    mismatch_per_row = mismatch_mask.sum(dim=1)
    unexpected_zeros_per_row = unexpected_zeros.sum(dim=1)

    print(f"Rows with all correct: {(mismatch_per_row == 0).sum().item()}/{M}")
    print(f"Rows with any mismatches: {(mismatch_per_row > 0).sum().item()}/{M}")
    print(
        f"Rows with unexpected zeros: {(unexpected_zeros_per_row > 0).sum().item()}/{M}"
    )

    # Show which rows have issues
    rows_with_issues = torch.where(mismatch_per_row > 0)[0]
    if len(rows_with_issues) > 0:
        print(f"\nFirst 20 rows with issues: {rows_with_issues[:20].tolist()}")
        print(f"Last 20 rows with issues: {rows_with_issues[-20:].tolist()}")

    # Analyze individual columns
    print("\n" + "=" * 80)
    print("PER-COLUMN ANALYSIS")
    print("=" * 80)
    mismatch_per_col = mismatch_mask.sum(dim=0)
    cols_fully_correct = (mismatch_per_col == 0).sum().item()
    print(f"Columns fully correct: {cols_fully_correct}/{N}")
    print(f"Columns with any mismatches: {N - cols_fully_correct}/{N}")

    # Show which columns are correct/incorrect
    correct_cols = torch.where(mismatch_per_col == 0)[0]
    incorrect_cols = torch.where(mismatch_per_col > 0)[0]

    if len(correct_cols) > 0:
        print(f"\nFirst 20 fully correct columns: {correct_cols[:20].tolist()}")
        if len(correct_cols) > 20:
            print(f"Last 20 fully correct columns: {correct_cols[-20:].tolist()}")

    if len(incorrect_cols) > 0:
        print(f"\nFirst 20 incorrect columns: {incorrect_cols[:20].tolist()}")
        if len(incorrect_cols) > 20:
            print(f"Last 20 incorrect columns: {incorrect_cols[-20:].tolist()}")

    # Analyze by column ranges
    print("\n" + "=" * 80)
    print("PER-COLUMN RANGE ANALYSIS")
    print("=" * 80)
    col_chunks = 8
    chunk_size = N // col_chunks
    for i in range(col_chunks):
        start_col = i * chunk_size
        end_col = (i + 1) * chunk_size
        chunk_mismatches = mismatch_mask[:, start_col:end_col].sum().item()
        chunk_unexpected_zeros = unexpected_zeros[:, start_col:end_col].sum().item()
        chunk_cols_correct = (mismatch_per_col[start_col:end_col] == 0).sum().item()
        print(
            f"Cols [{start_col:4d}-{end_col:4d}): mismatches={chunk_mismatches:6d}, unexpected_zeros={chunk_unexpected_zeros:6d}, fully_correct_cols={chunk_cols_correct:3d}/{chunk_size}"
        )

    # Show sample mismatches
    if num_mismatches > 0:
        print("\n" + "=" * 80)
        print("SAMPLE MISMATCHES (first 20)")
        print("=" * 80)
        mismatch_indices = torch.where(mismatch_mask)
        sample_size = min(20, len(mismatch_indices[0]))
        for i in range(sample_size):
            row = mismatch_indices[0][i].item()
            col = mismatch_indices[1][i].item()
            result_val = result[row, col].item()
            expected_val = expected[row, col].item()
            diff = abs(result_val - expected_val)
            print(
                f"  [{row:4d}, {col:4d}]: result={result_val:12.6f}, expected={expected_val:12.6f}, diff={diff:12.6f}"
            )

    # Check for pattern: first half vs second half
    print("\n" + "=" * 80)
    print("HALF-SPLIT ANALYSIS")
    print("=" * 80)
    half_n = N // 2
    first_half_mismatches = mismatch_mask[:, :half_n].sum().item()
    second_half_mismatches = mismatch_mask[:, half_n:].sum().item()
    first_half_zeros = unexpected_zeros[:, :half_n].sum().item()
    second_half_zeros = unexpected_zeros[:, half_n:].sum().item()

    print(
        f"First half cols [0-{half_n}): mismatches={first_half_mismatches}, unexpected_zeros={first_half_zeros}"
    )
    print(
        f"Second half cols [{half_n}-{N}): mismatches={second_half_mismatches}, unexpected_zeros={second_half_zeros}"
    )

    print("\n" + "=" * 80)

    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-3)
