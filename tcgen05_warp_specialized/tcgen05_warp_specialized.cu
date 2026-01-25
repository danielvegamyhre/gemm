#include <stdio.h>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <mma.h>

using namespace nvcuda;

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


// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
__device__ uint64_t matrix_desc_encode(uint64_t x) {
    // grabs 18 rightmost bits and shifts right by 4 to get bits 3-17 (14 bits)
    return (x & 0x3FFFF) >> 4;
}

// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
template <int BK>
__device__ uint64_t make_smem_desc(void* smem_ptr) {
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // bits 0-13: matrix_desc_encode(matrix addr)
    uint64_t desc = matrix_desc_encode(shared_addr);

    // bits 16-29: matrix_desc_encode(leading dim byte offset)
    // ignored for swizzled layout

    // bits 32-45: matrix_desc_encode(stride dim byte offset)
    // SBO (stride byte offset) is stride between rows of core matrices (along M/N) dim
    uint64_t SBO = 8 * BK * 2; // 8 rows in core matrix * BK * 2 bytes per elem
    desc |= (matrix_desc_encode(SBO) << 32);

    // bits 61-63: swizzle mode (1 = 128b swizzle with 32b atomicity)
    desc |= (1llu << 61);
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

    // explicit fence to make mbar init (generic proxy) visible to async proxy (TMA)
    asm volatile("fence.proxy.async.shared::cta;");
  }
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


template <int BN>
__device__ __forceinline__ void tcgen05_alloc(uint32_t tmem_addr_smem) {
    // convert to shared addr
    const int addr = static_cast<int>__cvta_generic_to_shared(tmem_addr_smem);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : // no outputs
        : "r"(addr), "r"(BN) // inputs
    );
}

template <int BN>
__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr_smem) {
    // convert to shared addr
    const int addr = static_cast<int>__cvta_generic_to_shared(tmem_addr_smem);
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.shared::cta.b32 %0, %1;"
        : // no outputs
        : "r"(addr), "r"(BN) // inputs
    );    
}

template <int MMA_M, int MMA_N>
__device__ __forceinline__ void tcgen05_idesc() {
    // create idesc
    // from PTX docs: "The 32-bit register operand idesc is the instruction descriptor as described in 
    //   Instruction descriptor, specifies the shapes, exact types, sparsity and other details of the 
    //   input matrices, output matrix and the matrix multiply and accumulate operation."
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    uint32_t idesc = 0;
    idesc |= (1 << 4);              // fp32 output matrix D
    idesc |= (1 << 7);              // bf16 input matrix A
    idesc |= (1 << 10);             // bf16 input matrix B
    idesc |= (MMA_N >> 3) << 17;    // N dim of matrix B
    idesc |= (MMA_M >> 4) << 24;    // M dim of matrix A
    return idesc;
}

__device__ __forceinline__ void tcgen05_mma(
    uint64_t smem_a_desc,
    uint64_t smem_b_desc,
    int tmem_accum_addr,
    uint32_t idesc
) {
    // tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc, { disable-output-lane }, enable-input-d {, scale-input-d};
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"     // declare predicate register for enable-input-d
        "mov.pred p, 0;\n\t"    // disable since we are just doing "D=A@B" not "D=A@B+D"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}"
        :
        : "r"(tmem_accum_addr), "l"(smem_a_desc), "l"(smem_b_desc), "r"(idesc)
    );
}

// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma
__device__ __forceinline__ void tcgen05_commit(uint64_t* mbar_ptr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :
        : "l"(mbar_ptr)
        : "memory"
    );
}

// from PTX docs:  "The Tensor Memory of a CTA is divided into 4 equal chunks such that 
// each warp of a warpgroup in the CTA can access a chunk of the Tensor Memory. 
// All the columns of the Tensor Memory can be accessed by all the four warps of a warpgroup."
// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st
__device__ __forceinline__ void tcgen05_ld(int tmem_addr_reg) {
    // warp 0 can access lanes 0-31
    // warp 1 can access lanes 32-63
    // warp 2 can access lanes 64-95
    // warp 3 can access lanes 96-128

}

template<
    int NUM_THREADS,
    int QUEUE_SIZE,
    int BM = 128,
    int BN = 256,
    int BK = 16,
    int MMA_M = 128,
    int MMA_N = 256,
    int MMA_K = 16
>
__global__ void ws_gemm(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    float* C,
    int M,
    int N,
    int K
) {
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int is_master_thread = threadIdx.x == 0;
    const int warp_id = threadIdx.x / 32;

    // init smem queue buffers
    __shared__ __align__(128) __nv_bfloat16 sA[QUEUE_SIZE][BM * BK];
    __shared__ __align__(128) __nv_bfloat16 sB[QUEUE_SIZE][BN * BK];  // TMA loads (BN x BK) with K contiguous

    // init mbarriers
    __shared__ __align__(8) uint64_t full_mbar[QUEUE_SIZE];     // for signaling buffer in queue is full/ready 
    __shared__ __align__(8) uint64_t empty_mbar[QUEUE_SIZE];    // for signaling buffer in queue has been read/can be re-used
    __shared__ __align__(8) uint64_t mma_mbar[1];               // for signaling tcgen05.mma batch is done
    __shared__ __align__(8) uint64_t ep_mbar[1];                // for signaling epilogue TMEM read is done / TMEM can be re-used

    initialize_barriers<QUEUE_SIZE, 1>(full_mbar, is_master_thread);    // 1 thread issues TMA, so 1 arrival for mbar
    initialize_barriers<1, 32>(mma_mbar, is_master_thread);             // 1 thread issues mmas and commit
    initialize_barriers<QUEUE_SIZE, 32>(empty_mbar, is_master_thread);  // full warp waits on mma batch and arrives, so 32 arrivals
    initialize_barriers<1, 128>(ep_mbar, is_master_thread);             // full warpgroup 

    // parity for coordination between tma, mma, epilogue warps
    int full_parity[QUEUE_SIZE] = {0};
    int empty_parity[QUEUE_SIZE] = {0};
    int mma_parity = 0;
    int ep_parity = 0;

    // alloc tmem addr for accumulator. allocates BN columns of TMEM (must always alloc full 128 rows)
    // 1 warp must do the allocation, from PTX docs:
    // "When .cta_group::1 is specified, one warp from the CTA must perform the allocation and de-allocation."
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions
    __shared__ int tmem_addr_smem; 
    if (warp_id == 0)
    {
        // after this call, we can load tmem_addr from smem -> to register to use.
       tcgen05_alloc<BN>(tmem_addr_smem);
    }

    // make sure mbarriers and tmem addr are visible to full threadblock
    __syncthreads();

    // prologue: prefetch first A/B tiles into first SMEM buffer
    int write_buf_idx = 0;
    int read_buf_idx = 0;

    // producer 
    if (warp_id == 0)
    {
        int a_global_row = block_row * BM;
        int b_global_col = block_col * BN;
        constexpr int num_bytes_a = BM * BK * 2; // 2 bytes per bf16
        constexpr int num_bytes_b = BK * BN * 2;
        const int num_k_tiles = (K + BK - 1) / BK;
        for (int bk_tile_idx = 0; bk_tile_idx < num_k_tiles; bk_tile_idx++) {
            int a_global_col = bk_tile_idx * BK;
            int b_global_row = bk_tile_idx * BK;

            // once we loop around to beginning of circular buffer for the first time,
            // we need to start waiing for consumer to finish reading from target buffer
            if (bk_tile_idx >= QUEUE_SIZE)
            {
                mbarrier_wait_parity(&empty_mbar[write_buf_idx], empty_parity[write_buf_idx]);
                empty_parity[write_buf_idx] ^= 1;
            }

            // tma loads for next A/B tiles
            if (is_master_thread)
            {
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sA[write_buf_idx][0]),
                    reinterpret_cast<const uint64_t*>(&a_map),
                    (uint32_t)a_global_col,
                    (uint32_t)a_global_row,
                    (uint64_t*)&full_mbar[write_buf_idx]
                );
                cp_async_bulk_tensor_2d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sB[write_buf_idx][0]),
                    reinterpret_cast<const uint64_t*>(&b_map),
                    (uint32_t)b_global_row, // swapped for col major
                    (uint32_t)b_global_col, // swapped for col major
                    (uint64_t*)&full_mbar[write_buf_idx]
                );
                mbarrier_arrive_expect_tx(&full_mbar[write_buf_idx], num_bytes_a + num_bytes_b);
            }
            else
            {
                mbarrier_arrive(&full_mbar[write_buf_idx]);
            }
            write_buf_idx = (write_buf_idx + 1) % QUEUE_SIZE;
        }
    }
    // consumer
    else if (warp_id == 1)
    {
        // only 1 thread in consumer/mma warp issues tcgen05.mma and tcgen05.commit
        const int is_mma_master_thread = threadIdx.x == 32;
        if (is_mma_master_thread) 
        {
            // make tcgen05 mma instruction descriptor, to be re-used at every iter of the loop
            uint32_t idesc = tcgen05_idesc<MMA_M, MMA_N>();

            // read tmem addr from smem -> register
            int tmem_addr = tmem_addr_smem;
            const int num_bk_tiles = (K + BK - 1) / BK;
            for (int bk_tile_idx = 0; bk_tile_idx < num_bk_tiles; bk_tile_idx++) {
                // wait for the tma load to the buffers we are about to read from.
                mbarrier_wait_parity(&full_mbar[read_buf_idx], full_parity[read_buf_idx]);
                full_parity[read_buf_idx] ^= 1;

                // tcgen05 mma for every subtile in the smem.
                #pragma unroll
                for (int m_tile_idx = 0; m_tile_idx < m_iters; m_tile_idx++) {
                    const int smem_a_row = m_tile_idx * MMA_M;
                    constexpr int k_iters = BK / MMA_K;

                    for (int k_tile_idx = 0; k_tile_idx < k_iters; k_tile_idx++) {
                        const int smem_a_col = k_tile_idx * MMA_K;
                        void* smem_tile_a = (void*)&sA[read_buf_idx][smem_a_row * BK + smem_a_col];
                        uint64_t smem_a_desc = make_smem_desc<BK>((void*)smem_tile_a);

                        for (int n_tile_idx = 0; n_tile_idx < n_iters; n_tile_idx++) {
                            const int smem_b_row = k_tile_idx * MMA_K;
                            const int smem_b_col = n_tile_idx * MMA_N;
                            void* smem_tile_b = (void*)&sB[read_buf_idx][smem_b_row + smem_b_col * BK];
                            uint64_t smem_b_desc = make_smem_desc<BK>((void*)smem_tile_b);
                            tcgen05_mma(smem_a_desc, smem_b_desc, tmem_addr, idesc);
                        }
                    }
                }
            }
            // commit batch of mmas
            tcgen05_commit(&mma_mbar[0]);
        }

        // wait for completion of batch of mmas
        mbarrier_try_wait_parity(&mma_mbar[0], mma_parity);
        mma_parity ^= 1;

        // signal we finished reading from this buffer
        mbarrier_arrive(&empty_mbar[read_buf_idx]);
        read_buf_idx = (read_buf_idx + 1) % QUEUE_SIZE;
    }
    // epilogue
    else 
    {
        
    }

    // tmem deallocation
    if (warp_id == 0) 
    {
        tcgen05_dealloc<BN>(tmem_addr_smem);
    }
}

extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K) {
    // dims for smem tiles
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;

    // dims for tcgen05.mma
    constexpr int MMA_M = 64;
    constexpr int MMA_N = 128;
    constexpr int MMA_K = 16;

    assert(BM >= MMA_M && BM % MMA_M == 0);
    assert(BN >= MMA_N && BN % MMA_N == 0);
    assert(BK >= MMA_K && BK % MMA_K == 0);

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

    constexpr int PRODUCER_WARPS = 1;
    constexpr int CONSUMER_WARPS = 1;
    constexpr int EPILOGUE_WARPS = 4;
    constexpr int NUM_THREADS = (PRODUCER_WARPS + CONSUMER_WARPS + EPILOGUE_WARPS) * 32;
    constexpr int QUEUE_SIZE = 2;

    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));

    auto kernel = ws_gemm<NUM_THREADS, QUEUE_SIZE, BM, BN, BK, MMA_M, MMA_N, MMA_K>;
    kernel<<<grid_dim, block_dim>>>(a_map, b_map, c_ptr, M, N, K);
}
