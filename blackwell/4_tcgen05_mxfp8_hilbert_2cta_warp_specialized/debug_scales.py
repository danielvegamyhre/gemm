import torch
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import to_blocked

# Create a simple test with known scales
M, K, N = 256, 128, 128

# Create matrices with simple patterns
torch.manual_seed(42)
A = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
B = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

# Quantize to MXFP8
A_data, A_scales = triton_to_mxfp8_dim0(A)
B_data, B_scales = triton_to_mxfp8_dim0(B)

print(f"A_data shape: {A_data.shape}")
print(f"A_scales shape (before blocking): {A_scales.shape}")
print(f"B_scales shape (before blocking): {B_scales.shape}")

# Apply blocking
A_scales_blocked = to_blocked(A_scales)
B_scales_blocked = to_blocked(B_scales)

print(f"\nA_scales_blocked shape: {A_scales_blocked.shape}")
print(f"B_scales_blocked shape: {B_scales_blocked.shape}")

# Check some scale values
print(f"\nA_scales (first 8 values): {A_scales[0, :8]}")
print(f"A_scales_blocked (first 8 bytes): {A_scales_blocked[0, :8]}")

# Print scale statistics
print(f"\nA_scales stats:")
print(f"  Min: {A_scales.min().item()}, Max: {A_scales.max().item()}, Mean: {A_scales.float().mean().item()}")
print(f"  Unique values: {torch.unique(A_scales).numel()}")

print(f"\nB_scales stats:")
print(f"  Min: {B_scales.min().item()}, Max: {B_scales.max().item()}, Mean: {B_scales.float().mean().item()}")
print(f"  Unique values: {torch.unique(B_scales).numel()}")

# Try with all-ones scales to verify
print("\n" + "="*80)
print("Testing with all-ones scales")
print("="*80)
A_scales_ones = torch.ones_like(A_scales)
B_scales_ones = torch.ones_like(B_scales)
A_scales_blocked_ones = to_blocked(A_scales_ones)
B_scales_blocked_ones = to_blocked(B_scales_ones)

print(f"Ones A_scales_blocked (first 8 bytes): {A_scales_blocked_ones[0, :8]}")
print(f"All same value? {torch.all(A_scales_blocked_ones == A_scales_blocked_ones[0, 0]).item()}")

# Check E8M0 encoding
print("\n" + "="*80)
print("E8M0 scale encoding check")
print("="*80)
print("E8M0 scale = 1.0 should be encoded as:")
# In E8M0: value = 2^(exponent - 127)
# For 1.0: 2^(exponent - 127) = 1.0 => exponent = 127 = 0x7F
print(f"  Expected byte value: 0x7F = {0x7F}")
print(f"  Actual ones byte value: {A_scales_blocked_ones[0, 0].item()}")

# Check if scale 2.0 is different
scale_2 = torch.full_like(A_scales, 2.0)
scale_2_blocked = to_blocked(scale_2)
print(f"  Scale 2.0 byte value: {scale_2_blocked[0, 0].item()}")
