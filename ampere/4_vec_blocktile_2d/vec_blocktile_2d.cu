#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>

#define BLOCK_SIZE 64

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    // align shared memory for float4 vectorized loads
    alignas(16) __shared__ float sA[BM * BK];
    alignas(16) __shared__ float sB[BK * BN];
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int a_base_row = block_row * BM;
    const int b_base_col = block_col * BN;
    const int c_row = block_row * BM + (threadIdx.x / (BN/TN)) * TM; // TM rows per thread
    const int c_col = block_col * BN + (threadIdx.x % (BN/TN)) * TN; // TN cols per thread

    // 4 results per thread along M dim
    float thread_results[TM * TN] = {0.0f};
    const int num_tiles = (K + BK - 1) / BK;
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // load tile of A from GMEM into SMEM.
        // we have divided BM/TM and BN/TN so we don't have enough threads to load the full required tile size of A/B at once, we must loop.
        const int floats_per_load = 4; // each thread loads 4 float32 vals via float4 vectorized load
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
    constexpr int TM = 4; // 4 results per thread along M dim
    constexpr int TN = 4; // 4 result per thread along N dim
    constexpr int BM = BLOCK_SIZE;
    constexpr int BN = BLOCK_SIZE;
    constexpr int BK = BLOCK_SIZE / TM; 

#ifdef DEBUG
    printf("TM=%d, TN=%d, BM=%d, BN=%d, BK=%d, blockDim.x=%d, blockDim.y=%d\n", TM, TN, BM, BN, BK, (BN/TN), (BM/TM));
#endif 
    
    constexpr int threadblock_size = (BM/TM) * (BN/TN);
    assert((BM*BK % (threadblock_size*4) == 0) && "A tile size must be divisible by threadblock_size*4 for float4 loads\n");
    assert((BK*BN % (threadblock_size*4) == 0) && "B tile size must be divisible by threadblock_size*4 for float4 loads\n");

    dim3 block_dim(threadblock_size);  
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    gemm<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
}
