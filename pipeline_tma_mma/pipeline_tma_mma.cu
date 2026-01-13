#include <stdio.h>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <mma.h>

using namespace nvcuda; 

#define BLOCK_SIZE 64
#define NUM_BUFFERS 2

// Overloaded error checking for both CUDA driver API (CUresult) and runtime API (cudaError_t)
inline void cuda_check_impl(CUresult result, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA Driver Error at %s:%d - Error code: %d\n", file, line, (int)result);
        exit(EXIT_FAILURE);
    }
}

inline void cuda_check_impl(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d - %s\n", file, line,
                cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(result) cuda_check_impl((result), __FILE__, __LINE__)

// Get the driver entry point for cuTensorMapEncodeTiled (used for TMA tensor maps)
inline void* get_driver_ptr() {
    static void *driver_ptr = nullptr;
    if (!driver_ptr) {
        cudaDriverEntryPointQueryResult result;
        CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr,
                                cudaEnableDefault, &result));
    }
    return driver_ptr;
}

// Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays
void create_tensor_map(
    void* tensor_ptr,
    CUtensorMap& tensor_map,
    const uint64_t gmem_width,
    const uint64_t gmem_height,
    const uint32_t smem_width,
    const uint32_t smem_height
) {
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {gmem_width, gmem_height};
    uint64_t stride[rank - 1] = {gmem_width}; // Row major, 1 byte per e8m0
    uint32_t box_size[rank] = {smem_width, smem_height};
    uint32_t elem_stride[rank] = {1, 1};

    void *driver_ptr = get_driver_ptr();
    auto cuTensorMapEncodeTiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        tensor_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    CUDA_CHECK(res);
}

// init mbar for sync between tma and regular 
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
    "mbarrier.init.shared.b64 [%0], %1;" 
    : // no outputs
    :"r"(mbar_ptr), "r"(count) // inputs
    : "memory"
  );
}

template <int num_barriers, int THREADS_PER_BLOCK>
__forceinline__ __device__ void initialize_barriers(uint64_t *mbar, const bool is_master_thread) {
  if (is_master_thread) {
    #pragma unroll
    for (int iter = 0; iter < num_barriers; ++iter) {
        mbarrier_init(&mbar[iter], THREADS_PER_BLOCK);
    }
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_ptr, const uint32_t parity) {
  uint32_t waitComplete;
  asm volatile(
    "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(waitComplete)         // outputs
        : "r"(mbar_ptr), "r"(parity) // inputs
        : "memory"
  );
  return static_cast<bool>(waitComplete);
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar, const uint32_t parity) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive(uint64_t *mbar) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
        "mbarrier.arrive.shared.b64 _, [%0];" 
        :                // no outputs
        :"r"(mbar_ptr)   // input
        : "memory"
  );
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void
mbarrier_arrive_expect_tx(uint64_t *mbar, const uint32_t tx_count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" 
      :                             // no outputs
      :"r"(mbar_ptr), "r"(tx_count) // inputs
      : "memory");
}


// async tma load from gmem to smem
__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    uint64_t *dst_shmem,
    const uint64_t *tensor_map_ptr,
    const uint32_t offset_x, 
    const uint32_t offset_y, 
    uint64_t *mbar
) {
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);

  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" 
      : // no outputs
      : // inputs
      "r"(dst_shmem_ptr),
      "l"(tensor_map_ptr), 
      "r"(offset_x), 
      "r"(offset_y), 
      "r"(mbar_ptr)
      : "memory"
  );
}

// wraper for cp_async_bulk_tensor_2d_global_to_shared
__forceinline__ __device__ void copy_2d_to_shared(
    void *dst, 
    const void *src, 
    const size_t global_offset_X,
    const size_t global_offset_Y, 
    const size_t num_bytes,
    uint64_t *mbar, 
    const bool is_master_thread
) {
  if (is_master_thread) {
    cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(dst),
        reinterpret_cast<const uint64_t *>(src), 
        global_offset_X, 
        global_offset_Y, 
        mbar
    );
    mbarrier_arrive_expect_tx(mbar, num_bytes);
  } else {
    mbarrier_arrive(mbar);
  }
}

template<
    int THREADBLOCK_SIZE,
    int BM = 128,
    int BN = 128,
    int BK = 16,
    int WMMA_M = 16,
    int WMMA_N = 16,
    int WMMA_K = 16,
    // how many wmma tiles each warp should do
    int WARP_TILES_M = 2,
    int WARP_TILES_N = 2
>
__global__ void gemm(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    float* C, 
    int M, 
    int N, 
    int K
) {
    __shared__ __align__(16) __nv_bfloat16 sA[NUM_BUFFERS][BM * BK]; // 64*16
    __shared__ __align__(16) __nv_bfloat16 sB[NUM_BUFFERS][BK * BN]; // 16*64
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    constexpr int warps_per_row = BN / (WARP_TILES_N * WMMA_N); // 64 / (2*16) = 2
    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;
    const int is_master_thread = threadIdx.x == 0;

    // init mbarriers (one per buffer, per A/B)
    __shared__ __align__(16) uint64_t mbar_a[NUM_BUFFERS];
    __shared__ __align__(16) uint64_t mbar_b[NUM_BUFFERS];
    int parity_a[NUM_BUFFERS] = {0};
    int parity_b[NUM_BUFFERS] = {0};
    initialize_barriers<NUM_BUFFERS, THREADBLOCK_SIZE>(mbar_a, is_master_thread);
    initialize_barriers<NUM_BUFFERS, THREADBLOCK_SIZE>(mbar_b, is_master_thread);

    // fragments for A and B (similar to a_reg/b_reg in warptile kernel)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

    // accumlators for each warp tile, init with 0s. accumulate in fp32
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[WARP_TILES_M][WARP_TILES_N];
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
        for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
            wmma::fill_fragment(c_frag[warp_tile_m][warp_tile_n], 0.0f);
        }
    }

    // prologue: prefetch first A/B tiles into first SMEM bufffer
    int write_buf_idx = 0;
    int read_buf_idx = 0;
    int a_global_row = block_row * BM;
    int a_global_col = 0;
    int b_global_row = 0;
    int b_global_col = block_col * BN;
    constexpr int num_bytes_a = BM * BK;
    constexpr int num_bytes_b = BK * BN;

    copy_2d_to_shared(
        reinterpret_cast<void*>(&sA[write_buf_idx][0]), 
        reinterpret_cast<const void*>(&a_map),
        a_global_col,
        a_global_row,
        num_bytes_a,
        (uint64_t*)&mbar_a[write_buf_idx],
        is_master_thread
    );
    copy_2d_to_shared(
        reinterpret_cast<void*>(&sB[write_buf_idx][0]),
        reinterpret_cast<const void*>(&b_map),
        b_global_col,
        b_global_row,
        num_bytes_b,
        (uint64_t*)&mbar_b[write_buf_idx],
        is_master_thread
    );

    // toggle buf to write to
    write_buf_idx ^= 1;

    const int num_k_tiles = (K + BK - 1) / BK;
    #pragma unroll
    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        // prefetch next a/b tiles into next buffer.
        // note this is not async in this kernel version, so benefits are limited (one fewer syncthreads())
        if (k_tile_idx + 1 < num_k_tiles)
        {
            a_global_col = (k_tile_idx + 1) * BK;
            b_global_row = (k_tile_idx + 1) * BK;
            copy_2d_to_shared(
                reinterpret_cast<void*>(&sA[write_buf_idx][0]), 
                reinterpret_cast<const void*>(&a_map),
                a_global_col,
                a_global_row,
                num_bytes_a,
                (uint64_t*)&mbar_a[write_buf_idx],
                is_master_thread
            ); 
            copy_2d_to_shared(
                reinterpret_cast<void*>(&sB[write_buf_idx][0]),
                reinterpret_cast<const void*>(&b_map),
                b_global_col,
                b_global_row,
                num_bytes_b,
                (uint64_t*)&mbar_b[write_buf_idx],
                is_master_thread
            );
            write_buf_idx ^= 1;
        }

        // at this point we have 2 tma commit groups in flight.
        // wait only for the one we are about to read from.
        mbarrier_wait_parity(&mbar_a[read_buf_idx], parity_a[read_buf_idx]);
        mbarrier_wait_parity(&mbar_b[read_buf_idx], parity_b[read_buf_idx]);
        asm volatile("fence.proxy.async.shared::cta;");

        // wmma on each warp tile this warp is responsible for
        for (int k = 0; k < BK; k += WMMA_K) {
            for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
                for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
                    const int smem_a_row = (warp_row * WARP_TILES_M * WMMA_M) + (warp_tile_m * WMMA_M);
                    const int smem_a_col = k;
                    __nv_bfloat16* smem_tile_a = &sA[read_buf_idx][smem_a_row * BK + smem_a_col];
                    wmma::load_matrix_sync(a_frag, smem_tile_a, BK);

                    const int smem_b_row = k;
                    const int smem_b_col = (warp_col * WARP_TILES_N * WMMA_N) + (warp_tile_n * WMMA_N);
                    __nv_bfloat16* smem_tile_b = &sB[read_buf_idx][smem_b_row * BN + smem_b_col];
                    wmma::load_matrix_sync(b_frag, smem_tile_b, BN);

                    // accumulate outer product
                    wmma::mma_sync(c_frag[warp_tile_m][warp_tile_n], a_frag, b_frag, c_frag[warp_tile_m][warp_tile_n]);
                }
            }
        }

        // toggle next buffer
        parity_a[read_buf_idx] ^= 1;
        parity_b[read_buf_idx] ^= 1;
        read_buf_idx ^= 1;
        __syncthreads();
    }

    // store output
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
        for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
            int c_row = (block_row * BM) + (warp_row * WARP_TILES_M * WMMA_M) + (warp_tile_m * WMMA_M);
            int c_col = (block_col * BN) + (warp_col * WARP_TILES_N * WMMA_N) + (warp_tile_n * WMMA_N);
            float* c_ptr = &C[c_row * N + c_col];
            wmma::store_matrix_sync(c_ptr, c_frag[warp_tile_m][warp_tile_n], N, wmma::mem_row_major);
        }
    }
}

extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K) {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARP_TILES_M = 2;
    constexpr int WARP_TILES_N = 2;
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };
    // dims for smem tiles
    constexpr int BM = BLOCK_SIZE; // 64
    constexpr int BN = BLOCK_SIZE; // 64
    constexpr int BK = 16;

    constexpr int num_warps = (BLOCK_SIZE * BLOCK_SIZE) / (WARP_TILES_M * WMMA_M * WARP_TILES_N * WMMA_N); // (64*64) / (2*16*2*16) = 4 warps
    constexpr int THREADBLOCK_SIZE = num_warps * 32; // 4*32 = 128 threads

    // create tensor maps
    alignas(64) CUtensorMap a_map = {};
    alignas(64) CUtensorMap b_map = {};
    create_tensor_map(
        A,
        a_map,
        K,
        M,
        BK,
        BM 
    );
    create_tensor_map(
        B,
        b_map,
        N,
        K,
        BN,
        BK
    );
    float* c_ptr = reinterpret_cast<float*>(C);

    dim3 block_dim(THREADBLOCK_SIZE);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    gemm<
        THREADBLOCK_SIZE,
        BM, BN, BK, 
        WMMA_M, WMMA_N, WMMA_K, 
        WARP_TILES_M, WARP_TILES_N
    ><<<grid_dim, block_dim>>>(a_map, b_map, c_ptr, M, N, K);
}
