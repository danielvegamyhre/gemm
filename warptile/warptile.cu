#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>

#define BLOCK_SIZE 128

template <int BM, int BN, int BK, int WM, int WN, int WSUBM, int WSUBN, int TM, int TN>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    alignas(16) __shared__ float sA[BM * BK];
    alignas(16) __shared__ float sB[BK * BN];
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_row = BN / WN;
    const int warp_row_subtiles_per_warp = (WM / WSUBM);
    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;
    const int lane_id = threadIdx.x % 32;
    const int thread_row_in_warp = lane_id / (WSUBN / TN);
    const int thread_col_in_warp = lane_id % (WSUBN / TN);

    // how many iterations each warp has to do within a warp tile
    constexpr int WMITER = WM / WSUBM;
    constexpr int WNITER = WN / WSUBN;

    // accumulator for this warp tile
    float accum[(WMITER * TM) * (WNITER * TN)] = {0.0f};
    const int num_tiles = (K + BK - 1) / BK;

    // constant for vectorized loads
    const int floats_per_load = 4;
    const int loads_per_iter = blockDim.x * floats_per_load; 
    const int total_a_loads = (BM * BK) / loads_per_iter;
    const int total_b_loads = (BK * BN) / loads_per_iter;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {

        // load tile of A from GMEM into SMEM.
        #pragma unroll
        for (int load_idx = 0; load_idx < total_a_loads; load_idx++) {
            const int linear_idx = load_idx * loads_per_iter + (threadIdx.x * floats_per_load);
            const int a_thread_row = linear_idx / BK;
            const int a_thread_col = linear_idx % BK;
            const int a_global_row = (block_row * BM) + a_thread_row;
            const int a_global_col = (tile_idx * BK) + a_thread_col;
            float4 data = make_float4(0, 0, 0, 0); 
            data = *reinterpret_cast<float4*>(&A[a_global_row * K + a_global_col]);

            // Store in transposed layout for coalesced smem reads of A column fragments into registers later
            sA[(a_thread_col + 0) * BM + a_thread_row] = data.x;
            sA[(a_thread_col + 1) * BM + a_thread_row] = data.y;
            sA[(a_thread_col + 2) * BM + a_thread_row] = data.z;
            sA[(a_thread_col + 3) * BM + a_thread_row] = data.w;
        }

        // Load tile of B from GMEM into SMEM
        #pragma unroll
        for (int load_idx = 0; load_idx < total_b_loads; load_idx++) {
            const int linear_idx = load_idx * loads_per_iter + (threadIdx.x * floats_per_load);
            const int b_thread_row = linear_idx / BN;
            const int b_thread_col = linear_idx % BN;
            const int b_global_row = (tile_idx * BK) + b_thread_row; 
            const int b_global_col = (block_col * BN) + b_thread_col;
            *reinterpret_cast<float4*>(&sB[b_thread_row * BN + b_thread_col]) = *reinterpret_cast<float4*>(&B[b_global_row * N + b_global_col]);
        }

        __syncthreads();

        #ifdef DEBUG
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            // print smem for debuggin
            for (int i = 0; i < BM; i++) {
                for (int j = 0; j < BK; j++) {
                    printf("sA[%d,%d]=%f\n", i, j, sA[j * BM + i]);
                }
                printf("\n");
            }
            for (int i = 0; i < BK; i++) {
                for (int j = 0; j < BN; j++) {
                    printf("sB[%d,%d]=%f\n", i, j, sB[i * BN + j]);
                }
                printf("\n");
            }
        }
        #endif

        for (int k = 0; k < BK; k++) {
            float a_reg[TM * WMITER] = {0.0f};
            float b_reg[TN * WNITER] = {0.0f};

            // cache col of A in registers
            const int a_smem_col = k;
            for (int wmiter = 0; wmiter < WMITER; wmiter++) {
                for (int tm = 0; tm < TM; tm++) {
                    const int a_smem_row = (warp_row * WM) + (wmiter * WSUBM) + (thread_row_in_warp * TM) + tm;
                    a_reg[wmiter * TM + tm] = sA[a_smem_col * BM + a_smem_row];
                }
            }
                
            // cache row of B in registers
            const int b_smem_row = k;
            for (int wniter = 0; wniter < WNITER; wniter++) {
                for (int tn = 0; tn < TN; tn++) {
                    const int b_smem_col = (warp_col * WN) + (wniter * WSUBN) + (thread_col_in_warp * TN) + tn;
                    b_reg[wniter * TN + tn] = sB[b_smem_row * BN + b_smem_col];
                }
            }

            // accumulate outer product
            for (int wmiter = 0; wmiter < WMITER; wmiter++) {
                for (int wniter = 0; wniter < WNITER; wniter++) {
                    for (int tm = 0; tm < TM; tm++) {
                        for (int tn = 0; tn < TN; tn++) {
                            int row = (wmiter * TM) + tm;
                            int col = (wniter * TN) + tn;
                            accum[row * (WNITER * TN) + col] += a_reg[row] * b_reg[col];
                        }
                    }
                }
            }
            #ifdef DEBUG
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                for (int i=0; i < WMITER*TM; i++) {
                    for (int j=0; j < WNITER*TN; j++) {
                        printf("a_reg[%d]=%f, b_reg[%d]=%f\n", i, a_reg[i], j, b_reg[j]);
                    }
                }
            }
            #endif
        }

        __syncthreads();
    }


    #ifdef DEBUG
    // print accum
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i=0; i < WMITER*TM; i++) {
            for (int j=0; j < WNITER*TN; j++) {
                printf("accum[%d,%d]=%f\n", i, j, accum[i * (WNITER*TN) + j]);
            }
        }
    }
    #endif

    // store output
    #pragma unroll
    for (int wmiter = 0; wmiter < WMITER; wmiter++) {
        #pragma unroll 
        for (int wniter = 0; wniter < WNITER; wniter++) {
            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    int c_row = (block_row * BLOCK_SIZE) + (warp_row * WM) + (wmiter * WSUBM) + (thread_row_in_warp * TM) + tm;
                    int c_col = (block_col * BLOCK_SIZE) + (warp_col * WN) + (wniter * WSUBN) + (thread_col_in_warp * TN) + tn;
                    int thread_row = (wmiter * TM) + tm;
                    int thread_col = (wniter * TN) + tn;
                    C[c_row * N + c_col] = accum[thread_row * (WNITER*TN) + thread_col];
                }
            }
        }
    }
}

void launch_gemm(float* A, float* B, float* C, int M, int N, int K) {
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    // each thread computes 2x2 tile
    constexpr int TM = 2;
    constexpr int TN = 2;

    // warp's 32 threads arranged in 8x4, each computing2x2 tile
    constexpr int WARP_SUBTILE_M = 8 * TM; // 8*2 = 16
    constexpr int WARP_SUBTILE_N = 4 * TN; // 4*2 = 8

    // warp subtiles arranged in 2x2 layout
    constexpr int WARP_TILE_M = 2 * WARP_SUBTILE_M; // 2*16= 32 
    constexpr int WARP_TILE_N = 2 * WARP_SUBTILE_N; // 2*8 = 16 

    // thread block target size divided into warp tiles
    constexpr int BM = BLOCK_SIZE; // 128
    constexpr int BN = BLOCK_SIZE; // 128
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
