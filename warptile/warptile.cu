#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>

#define BLOCK_SIZE 128
#define DEBUG

template <int BM, int BN, int BK, int WM, int WN, int WSUBM, int WSUBN, int TM, int TN>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    alignas(16) __shared__ float sA[BM * BK];
    alignas(16) __shared__ float sB[BK * BN];
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int a_base_row = block_row * BM;
    const int b_base_col = block_col * BN;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_row = BN / WN;
    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;

    // how many iterations each warp has to do within a warp tile
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    // update base pointers for this warp
    A += (a_base_row + warp_row * WM) * N;
    B += b_base_col + (warp_col * WN);

    // 4 results per thread along M dim
    float thread_results[WM * WN] = {0.0f};
    const int num_tiles = (K + BK - 1) / BK;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        for (int wmiter = 0; wmiter < WMITER; wmiter++) {
            // increment base pointer for load from A 
            A += WSUBM * N;
            for (int wniter = 0; wniter < WNITER; wniter++) {
                // increment base pointer for load from B
                B += WSUBN;

                // load tile of A from GMEM into SMEM.
                const int floats_per_load = 4;
                const int loads_per_iter = blockDim.x * floats_per_load; 
                const int total_a_loads = (BM * BK) / loads_per_iter;

                #pragma unroll
                for (int load_idx = 0; load_idx < total_a_loads; load_idx++) {
                    const int linear_idx = load_idx * loads_per_iter + (threadIdx.x * floats_per_load);
                    const int a_thread_row = linear_idx / BK;
                    const int a_thread_col = linear_idx % BK;
                    const int a_global_row = a_base_row + a_thread_row;
                    const int a_global_col = tile_idx * BK + a_thread_col;
                    float4 data = make_float4(0, 0, 0, 0); 
                    if (a_global_row < M && a_global_col < K)
                    {
                        data = *reinterpret_cast<float4*>(&A[a_global_row * K + a_global_col]);
                    }
                    // Store in transposed layout for coalesced smem reads of A column fragments into registers later
                    sA[(a_thread_col + 0) * BM + a_thread_row] = data.x;
                    sA[(a_thread_col + 1) * BM + a_thread_row] = data.y;
                    sA[(a_thread_col + 2) * BM + a_thread_row] = data.z;
                    sA[(a_thread_col + 3) * BM + a_thread_row] = data.w;
                }

                // Load tile of B from GMEM into SMEM
                const int total_b_loads = (BK * BN) / loads_per_iter;
                #pragma unroll
                for (int load_idx = 0; load_idx < total_b_loads; load_idx++) {
                    const int linear_idx = load_idx * loads_per_iter + (threadIdx.x * floats_per_load);
                    const int b_thread_row = linear_idx / BN;
                    const int b_thread_col = linear_idx % BN;
                    const int b_global_row = tile_idx * BK + b_thread_row;
                    const int b_global_col = b_base_col + b_thread_col;
                    if (b_global_row < K && b_global_col < N) 
                    {
                        *reinterpret_cast<float4*>(&sB[b_thread_row * BN + b_thread_col]) = *reinterpret_cast<float4*>(&B[b_global_row * N + b_global_col]);
                    }
                    else 
                    {
                        *reinterpret_cast<float4*>(&sB[b_thread_row * BN + b_thread_col]) = make_float4(0, 0, 0, 0);
                    }
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float a_reg[TM] = {0.0f};
            float b_reg[TN] = {0.0f};

            // cache col of A in registers
            const int a_smem_col = k;
            const int a_smem_base_row = (threadIdx.x / (BN/TN)) * TM;
            for (int tm = 0; tm < TM; tm++) {
                const int a_smem_row = a_smem_base_row + tm;
                a_reg[tm] = sA[a_smem_col * BM + a_smem_row];
            }
            
            // cache row of B in registers
            const int b_smem_row = k;
            const int b_smem_base_col = (threadIdx.x % (BN/TN)) * TN;
            for (int tn = 0; tn < TN; tn++) {
                const int b_smem_col = b_smem_base_col + tn;
                b_reg[tn] = sB[b_smem_row * BN + b_smem_col];
            }

            // accumulate outer product
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    thread_results[tm * TN + tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }
        __syncthreads();
    }

    // store output
    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
            if (c_row + tm < M && c_col + tn < N)
            {
                C[(c_row + tm) * N + (c_col + tn)] = thread_results[tm * TN + tn];
            }
        }
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    // each thread computes 4x4 tile
    constexpr int TM = 4;
    constexpr int TN = 4;

    // warp's 32 threads arranged in 8x4, each computing 4x4 tile
    constexpr int WARP_SUBTILE_M = 8 * TM; // 8*4 = 32
    constexpr int WARP_SUBTILE_N = 4 * TN; // 4*4 = 16

    // warp subtiles arranged in 2x2 layout
    constexpr int WARP_TILE_M = 2 * WARP_SUBTILE_M; // 2*32 = 64
    constexpr int WARP_TILE_N = 2 * WARP_SUBTILE_N; // 2*16 = 32

    // thread block target size divided into warp tiles
    constexpr int BM = BLOCK_SIZE / WARP_TILE_M; // 128/64 = 2
    constexpr int BN = BLOCK_SIZE / WARP_TILE_N; // 128/32 = 4
    constexpr int BK = 16;

#ifdef DEBUG
    printf("TM=%d, TN=%d, BM=%d, BN=%d, BK=%d, WARP_SUBTILE_M=%d, WARP_SUBTILE_N=%d, WARP_TILE_M=%d, WARP_TILE_N=%d\n", TM, TN, BM, BN, BK, WARP_SUBTILE_M, WARP_SUBTILE_N, WARP_TILE_M, WARP_TILE_N);
#endif 
    
    constexpr int num_warps = (BLOCK_SIZE * BLOCK_SIZE) / (WARP_TILE_M * WARP_TILE_N); // 128*128 / (64*32) = 8 warps
    constexpr int threadblock_size = num_warps * 32; // 8 * 32 = 256 threads

    dim3 block_dim(threadblock_size);  
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    gemm<BM, BN, BK, WARP_TILE_M, WARP_TILE_N, WARP_SUBTILE_M, WARP_SUBTILE_N, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
