# MXFP8 GEMM Kernels Benchmark Results

Benchmarks run on NVIDIA B200 with PyTorch 2.12.0.dev20260319+cu130 using CUDA_VISIBLE_DEVICES=2.

## Performance Summary

All kernels implement MXFP8 matrix multiplication compared against PyTorch's `_scaled_mm` baseline.

| Kernel | Description | M=2048 | M=4096 | M=8192 | M=16384 |
|--------|-------------|--------|--------|--------|---------|
| 0_warp_specialized | Basic 2CTA warp specialized | 0.48x | 0.35x | 0.44x | 0.49x |
| 1_256n_overlap128 | 256N with 128 column overlap | **0.99x** | 0.87x | 0.97x | 0.95x |
| 2_256n_overlap64 | 256N with 64 column overlap | **0.99x** | 0.87x | 0.92x | 0.94x |
| 3_256n_overlap128_tma_store | 256N overlap128 + TMA store | **0.99x** | 0.87x | **0.98x** | **0.97x** |
| 4_256n_overlap64_tma_store | 256N overlap64 + TMA store | **0.99x** | 0.87x | **0.98x** | 0.96x |
| 5_256n_triple_buffer_tmem | 256N triple buffered TMEM | 0.89x | 0.84x | 0.95x | 0.96x |

## Detailed Results

### Kernel 0: tcgen05_mxfp8_2cta_warp_specialized

Basic warp-specialized implementation with 2 CTAs.

| Matrix Size | Custom Kernel | PyTorch _scaled_mm | Speedup |
|-------------|---------------|-------------------|---------|
| M=2048, K=2048, N=2048 | 35.904 us (478.49 tflops) | 17.248 us (996.05 tflops) | 0.48x |
| M=4096, K=4096, N=4096 | 156.512 us (878.14 tflops) | 54.112 us (2539.90 tflops) | 0.35x |
| M=8192, K=8192, N=8192 | 942.144 us (1167.03 tflops) | 418.656 us (2626.29 tflops) | 0.44x |
| M=16384, K=16384, N=16384 | 7118.912 us (1235.60 tflops) | 3507.024 us (2508.14 tflops) | 0.49x |

### Kernel 1: tcgen05_mxfp8_2cta_256n_overlap128

256N block size with 128 column accumulator overlap.

| Matrix Size | Custom Kernel | PyTorch _scaled_mm | Speedup |
|-------------|---------------|-------------------|---------|
| M=2048, K=2048, N=2048 | 17.472 us (983.28 tflops) | 17.248 us (996.05 tflops) | 0.99x |
| M=4096, K=4096, N=4096 | 62.400 us (2202.55 tflops) | 54.112 us (2539.90 tflops) | 0.87x |
| M=8192, K=8192, N=8192 | 433.152 us (2538.40 tflops) | 418.720 us (2625.89 tflops) | 0.97x |
| M=16384, K=16384, N=16384 | 3695.728 us (2380.07 tflops) | 3521.568 us (2497.78 tflops) | 0.95x |

### Kernel 2: tcgen05_mxfp8_2cta_256n_overlap64

256N block size with 64 column accumulator overlap.

| Matrix Size | Custom Kernel | PyTorch _scaled_mm | Speedup |
|-------------|---------------|-------------------|---------|
| M=2048, K=2048, N=2048 | 17.504 us (981.48 tflops) | 17.248 us (996.05 tflops) | 0.99x |
| M=4096, K=4096, N=4096 | 62.304 us (2205.94 tflops) | 54.112 us (2539.90 tflops) | 0.87x |
| M=8192, K=8192, N=8192 | 455.616 us (2413.24 tflops) | 418.656 us (2626.29 tflops) | 0.92x |
| M=16384, K=16384, N=16384 | 3689.392 us (2384.16 tflops) | 3483.424 us (2525.13 tflops) | 0.94x |

### Kernel 3: tcgen05_mxfp8_2cta_256n_overlap128_tma_store

256N with 128 column overlap + TMA store for epilogue. Uses heuristic-based epilogue strategy selection.

| Matrix Size | Custom Kernel | PyTorch _scaled_mm | Speedup |
|-------------|---------------|-------------------|---------|
| M=2048, K=2048, N=2048 | 17.504 us (981.48 tflops) | 17.248 us (996.05 tflops) | 0.99x |
| M=4096, K=4096, N=4096 | 62.432 us (2201.42 tflops) | 54.112 us (2539.90 tflops) | 0.87x |
| M=8192, K=8192, N=8192 | 427.104 us (2574.34 tflops) | 418.688 us (2626.09 tflops) | 0.98x |
| M=16384, K=16384, N=16384 | 3625.088 us (2426.45 tflops) | 3507.552 us (2507.76 tflops) | 0.97x |

### Kernel 4: tcgen05_mxfp8_2cta_256n_overlap64_tma_store

256N with 64 column overlap + TMA store. Uses heuristic-based epilogue strategy selection.

| Matrix Size | Custom Kernel | PyTorch _scaled_mm | Speedup |
|-------------|---------------|-------------------|---------|
| M=2048, K=2048, N=2048 | 17.440 us (985.08 tflops) | 17.248 us (996.05 tflops) | 0.99x |
| M=4096, K=4096, N=4096 | 62.336 us (2204.81 tflops) | 54.112 us (2539.90 tflops) | 0.87x |
| M=8192, K=8192, N=8192 | 429.184 us (2561.87 tflops) | 418.656 us (2626.29 tflops) | 0.98x |
| M=16384, K=16384, N=16384 | 3613.568 us (2434.18 tflops) | 3482.560 us (2525.75 tflops) | 0.96x |

### Kernel 5: tcgen05_mxfp8_2cta_256n_triple_buffer_tmem

256N with triple buffering in TMEM.

| Matrix Size | Custom Kernel | PyTorch _scaled_mm | Speedup |
|-------------|---------------|-------------------|---------|
| M=2048, K=2048, N=2048 | 19.424 us (884.47 tflops) | 17.248 us (996.05 tflops) | 0.89x |
| M=4096, K=4096, N=4096 | 64.352 us (2135.74 tflops) | 54.112 us (2539.90 tflops) | 0.84x |
| M=8192, K=8192, N=8192 | 441.312 us (2491.46 tflops) | 418.688 us (2626.09 tflops) | 0.95x |
| M=16384, K=16384, N=16384 | 3613.616 us (2434.15 tflops) | 3483.712 us (2524.92 tflops) | 0.96x |

