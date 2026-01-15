#include <stdio.h>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <mma.h>

using namespace nvcuda;

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

void create_2d_tensor_map(
    void* tensor_ptr,
    CUtensorMap& tensor_map,
    const uint64_t global_height,
    const uint64_t global_width,
    const uint32_t smem_height,
    const uint32_t smem_width,
    const uint32_t stride_bytes
) {
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {global_width, global_height};
    uint64_t stride[rank - 1] = {stride_bytes};
    uint32_t box_size[rank] = {smem_width, smem_height};
    uint32_t elem_stride[rank] = {1, 1};

    void *driver_ptr = get_driver_ptr();
    auto cuTensorMapEncodeTiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        tensor_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    CUDA_CHECK(res);
}

// Reference: https://gau-nernst.github.io/tcgen05
void create_3d_tensor_map(
    void* tensor_ptr,
    CUtensorMap& tensor_map,
    const uint64_t global_height,
    const uint64_t global_width,
    const uint32_t smem_height,
    const uint32_t smem_width
) {
    constexpr uint32_t rank = 3;
    uint64_t size[rank] = {8, global_height, global_width/8};
    uint64_t stride[rank - 1] = {global_width * sizeof(__nv_bfloat16), 16};
    uint32_t box_size[rank] = {8, smem_height, smem_width/8};
    uint32_t elem_stride[rank] = {1, 1, 1};

    void *driver_ptr = get_driver_ptr();
    auto cuTensorMapEncodeTiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(driver_ptr);

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
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


template<int SCALE_D, int SCALE_A, int SCALE_B, int TRANS_A, int TRANS_B>
__device__ __forceinline__ void wgmma_m64n16k16(uint64_t smem_desc_a, uint64_t smem_desc_b, float reg_c[8]) {
    // 64x16 output = 1024 elements / 4 warps / 32 threads per warp = 8 outputs per thread
    // outputs are fp32 which is exactly 1 32bit reg
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, " // 8 register vector for accum
        "%8, %9, "
        "%10, %11, %12, %13, %14;\n"
        "}\n"
        : "+f"(reg_c[0]), "+f"(reg_c[1]), "+f"(reg_c[2]), "+f"(reg_c[3]),
          "+f"(reg_c[4]), "+f"(reg_c[5]), "+f"(reg_c[6]), "+f"(reg_c[7])
        : "l"(smem_desc_a), "l"(smem_desc_b),
          "n"((int32_t)SCALE_D), "n"((int32_t)SCALE_A), "n"((int32_t)SCALE_B),
          "n"((int32_t)TRANS_A), "n"((int32_t)TRANS_B)
    );
}

// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ uint64_t matrix_desc_encode(uint64_t x) {
    // grabs 18 rightmost bits and shifts right by 4 to get bits 3-13 (14 bits)
    return (x & 0x3FFFF) >> 4;
}

template <int BK>
__device__ uint64_t make_smem_desc(void* smem_ptr) {
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // bits 0-13: matrix_desc_encode(matrix addr)
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
    uint64_t desc = matrix_desc_encode(shared_addr);

    // bits 16-29: matrix_desc_encode(leading dim byte offset)
    // LBO (leading byte offset) is the stride between core-matrices along the K dim
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-leading-dimension-byte-offset
    uint64_t LBO = 16;
    desc |= (matrix_desc_encode(LBO) << 16);

    // bits 32-45: matrix_desc_encode(stride dim byte offset)
    // SBO (stride byte offset) is stride between rows of core matrices (along M/N) dim
    uint64_t SBO = BK*16;
    desc |= (matrix_desc_encode(SBO) << 32);

    // bits 49-52: matrix base offset (no swizzle = 0)
    // bits 62-63: swizzle mode (no swizzle = 0)
    desc |= (1llu << 62);
    return desc;
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
    const void *tmap,
    const size_t global_offset_X,
    const size_t global_offset_Y,
    const size_t num_bytes,
    uint64_t *mbar,
    const bool is_master_thread
) {
  if (is_master_thread) {
    cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(dst),
        reinterpret_cast<const uint64_t *>(tmap),
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
    int NUM_THREADS,
    int BM = 128,
    int BN = 128,
    int BK = 16,
    int WGMMA_M = 64,
    int WGMMA_N = 16,
    int WGMMA_K = 16
>
__global__ void gemm(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    float* C,
    int M,
    int N,
    int K
) {
    __shared__ __align__(128) __nv_bfloat16 sA[NUM_BUFFERS][BM * BK];
    __shared__ __align__(128) __nv_bfloat16 sB[NUM_BUFFERS][BK * BN];
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int is_master_thread = threadIdx.x == 0;

    // init mbarriers (one per buffer, per A/B)
    __shared__ __align__(128) uint64_t mbar_a[NUM_BUFFERS];
    __shared__ __align__(128) uint64_t mbar_b[NUM_BUFFERS];
    int parity_a[NUM_BUFFERS] = {0};
    int parity_b[NUM_BUFFERS] = {0};
    initialize_barriers<NUM_BUFFERS, NUM_THREADS>(mbar_a, is_master_thread);
    initialize_barriers<NUM_BUFFERS, NUM_THREADS>(mbar_b, is_master_thread);

    // prologue: prefetch first A/B tiles into first SMEM bufffer
    int write_buf_idx = 0;
    int read_buf_idx = 0;
    int a_global_row = block_row * BM;
    int a_global_col = 0;
    int b_global_row = 0;
    int b_global_col = block_col * BN;
    constexpr int num_bytes_a = BM * BK * 2; // 2 bytes per bf16
    constexpr int num_bytes_b = BK * BN * 2;

    // TODO: load data to layout required for wgmma
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
        b_global_row,
        b_global_col, // swapped for col major
        num_bytes_b,
        (uint64_t*)&mbar_b[write_buf_idx],
        is_master_thread
    );

    // toggle buf to write to
    write_buf_idx ^= 1;

    // one 8 register accum buffer for each warpgroup.
    // (64,16) @ (16,16) = (64,16) -> 1024 size output per warpgroup
    // 1024 / 4 warps / 32 threads per warp = 8 elements per thread.
    // These 8 elements are distributed across 16 columns of the output D:
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16
    constexpr int NUM_WG = NUM_THREADS / 128;
    constexpr int ROWS_PER_WG = BM / NUM_WG;
    constexpr int m_iters = ROWS_PER_WG/WGMMA_M;    // each wg covers 64 rows of output *per wgmma iter along M*
    constexpr int n_iters = BN/WGMMA_N;             // each wg covers 16 cols of output *per wgmma iter along N*
    float accum[m_iters][n_iters][8] = {0};         // each thread in wg has 8 registers of accum output
    static_assert(sizeof(accum) * NUM_THREADS == BM * BN * sizeof(float));

    const int num_bk_tiles = (K + BK - 1) / BK;
    #pragma unroll
    for (int bk_tile_idx = 0; bk_tile_idx < num_bk_tiles; bk_tile_idx++) {
        // prefetch next a/b tiles into next buffer.
        // note this is not async in this kernel version, so benefits are limited (one fewer syncthreads())
        if (bk_tile_idx + 1 < num_bk_tiles)
        {
            a_global_col = (bk_tile_idx + 1) * BK;
            b_global_row = (bk_tile_idx + 1) * BK;
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
                b_global_row, // swapped for col major
                b_global_col,
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

        // wgmma executed in the async proxy, so use fence to enforce
        // ordering requirements of warpgroup accesses to registers before wgmma.mma_async
        asm volatile("wgmma.fence.sync.aligned; " ::: "memory");

        // async wgmma for every subtile in the smem.
        // warpgroups are arranged vertically and iterate horizontally over BN
        const int wg_row = threadIdx.x / 128;
        const int k_iters = BK / WGMMA_K;
        for (int m_tile_idx = 0; m_tile_idx < m_iters; m_tile_idx++) {
            for (int n_tile_idx = 0; n_tile_idx < n_iters; n_tile_idx++) {
                for (int k_tile_idx = 0; k_tile_idx < k_iters; k_tile_idx++) {
                    const int smem_a_row = (wg_row * ROWS_PER_WG) + (m_tile_idx * WGMMA_M);
                    const int smem_a_col = k_tile_idx * WGMMA_K;
                    __nv_bfloat16* smem_tile_a = &sA[read_buf_idx][smem_a_row * BK + smem_a_col];
                    uint64_t smem_a_desc = make_smem_desc<BK>((void*)smem_tile_a);

                    const int smem_b_row = k_tile_idx * WGMMA_K;
                    const int smem_b_col = n_tile_idx * WGMMA_N;
                    __nv_bfloat16* smem_tile_b = &sB[read_buf_idx][smem_b_row * BN + smem_b_col];
                    uint64_t smem_b_desc = make_smem_desc<BK>((void*)smem_tile_b);

                    wgmma_m64n16k16<1,1,1,0,0>(smem_a_desc, smem_b_desc, accum[m_tile_idx][n_tile_idx]);
                }
            }
        }

        // commit batch of wgmmas and wait for completion
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

        // toggle next buffer
        parity_a[read_buf_idx] ^= 1;
        parity_b[read_buf_idx] ^= 1;
        read_buf_idx ^= 1;
        __syncthreads();
    }

    // accum register layout is confusing: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n16
    // each warp in the warpgroup computes WGMMA_M/4 rows.
    // each consecutive 4 lanes share the same row.
    // threads in groups of 4, each holding 2 registers of contiguous output memory
    constexpr int rows_per_warp = WGMMA_M / 4;
    const int wg_row = threadIdx.x / 128;
    const int warp_id = threadIdx.x / 32;
    const int warp_idx_in_wg = warp_id % 4;
    const int lane_id = threadIdx.x % 32;
    const int thread_row_c = warp_idx_in_wg * rows_per_warp + lane_id / 4;
    const int thread_col_c = 2 * (lane_id % 4); // 2 reg per thread, 4 threads per row before wrapping to next row

    int c_base_row = block_row * BM;
    int c_base_col = block_col * BN;

    // accum register layout:
    // 2 rows: row, row+8
    // 4 cols: col, col+1, col+8, col+9
    #pragma unroll
    for (int m_tile_idx = 0; m_tile_idx < m_iters; m_tile_idx++) {
        #pragma unroll
        for (int n_tile_idx = 0; n_tile_idx < n_iters; n_tile_idx++) {
            int row = c_base_row + (wg_row * ROWS_PER_WG) + (m_tile_idx * WGMMA_M) + thread_row_c;
            int col = c_base_col + (n_tile_idx * WGMMA_N) + thread_col_c;
            C[row * N + col + 0] = accum[m_tile_idx][n_tile_idx][0];
            C[row * N + col + 1] = accum[m_tile_idx][n_tile_idx][1];
            C[(row+8) * N + col + 0] = accum[m_tile_idx][n_tile_idx][2];
            C[(row+8) * N + col + 1] = accum[m_tile_idx][n_tile_idx][3];
            C[row * N + col + 8] = accum[m_tile_idx][n_tile_idx][4];
            C[row * N + col + 9] = accum[m_tile_idx][n_tile_idx][5];
            C[(row+8) * N + col + 8] = accum[m_tile_idx][n_tile_idx][6];
            C[(row+8) * N + col + 9] = accum[m_tile_idx][n_tile_idx][7];
        }
    }
}

extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K) {
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 16;
    constexpr int WGMMA_K = 16;
    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };

    // dims for smem tiles
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;

    constexpr int NUM_THREADS = 128;

    alignas(64) CUtensorMap a_map = {};
    alignas(64) CUtensorMap b_map = {};
    create_2d_tensor_map(
        A,
        a_map,
        M,
        K,
        BM,
        BK,
        K * sizeof(__nv_bfloat16) // row major stride
    );
    create_2d_tensor_map(
        B,
        b_map,
        N,
        K,
        BN,
        BK,
        K * sizeof(__nv_bfloat16) // col major stride
    );
    float* c_ptr = reinterpret_cast<float*>(C);

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));
    auto kernel = gemm<NUM_THREADS, BM, BN, BK, WGMMA_M, WGMMA_N, WGMMA_K>;
    kernel<<<grid_dim, block_dim>>>(a_map, b_map, c_ptr, M, N, K);
}
