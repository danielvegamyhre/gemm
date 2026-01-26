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


void create_3d_tensor_map(
    void* tensor_ptr,
    CUtensorMap& tensor_map,
    const uint64_t global_dims[3],
    const uint32_t smem_dims[3],
    const uint32_t stride_bytes[2]
) {
    constexpr uint32_t rank = 3;
    uint64_t size[rank] = {global_dims[0], global_dims[1], global_dims[2]};
    uint64_t stride[rank - 1] = {stride_bytes[0], stride_bytes[1]};
    uint32_t box_size[rank] = {smem_dims[0], smem_dims[1], smem_dims[2]};
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
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    CUDA_CHECK(res);
}

// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
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
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar, const uint32_t tx_count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
      :                             // no outputs
      :"r"(mbar_ptr), "r"(tx_count) // inputs
      : "memory");
}


__device__ __forceinline__ void cp_async_bulk_tensor_3d_global_to_shared(
    uint64_t *dst_shmem,
    const uint64_t *tensor_map_ptr,
    const uint32_t offset_x,
    const uint32_t offset_y,
    const uint32_t offset_z,
    uint64_t *mbar
) {
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);    
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :
        :
        "r"(dst_shmem_ptr),
        "l"(tensor_map_ptr),
        "r"(offset_x),
        "r"(offset_y),
        "r"(offset_z),
        "r"(mbar_ptr)
        : "memory"
    );
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

template <int BN>
__device__ __forceinline__ void tcgen05_alloc(int* tmem_addr_smem) {
    // convert to shared addr
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr_smem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        : // no outputs
        : "r"(addr), "r"(BN) // inputs
    );
}

template <int BN>
__device__ __forceinline__ void tcgen05_dealloc(int tmem_addr_reg) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        : // no outputs
        : "r"(tmem_addr_reg), "r"(BN) // inputs
    );    
}

template <int MMA_M, int MMA_N>
__device__ __forceinline__ void tcgen05_encode_idesc(uint32_t& idesc) {
    // create idesc
    // from PTX docs: "The 32-bit register operand idesc is the instruction descriptor as described in 
    //   Instruction descriptor, specifies the shapes, exact types, sparsity and other details of the 
    //   input matrices, output matrix and the matrix multiply and accumulate operation."
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    idesc |= (1 << 4);              // fp32 output matrix D
    idesc |= (1 << 7);              // bf16 input matrix A
    idesc |= (1 << 10);             // bf16 input matrix B
    idesc |= (MMA_N >> 3) << 17;    // N dim of matrix B
    idesc |= (MMA_M >> 4) << 24;    // M dim of matrix A
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
// 
template <int BN>
__device__ __forceinline__ void tcgen05_ld(int tmem_base_addr_reg, int ep_warp_id, float c_reg[32]) {
    // warp 0 can access lanes 0-31
    // warp 1 can access lanes 32-63
    // warp 2 can access lanes 64-95
    // warp 3 can access lanes 96-128
    int row = ep_warp_id * 32;
    int col = 0;

    // TMEM address is 32bit and composed of 2 components:
    // - bits 0-15: column index
    // - bits 16-31: row index
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing
    int tmem_addr = tmem_base_addr_reg + (row << 16) + col;

    // - each warp will read 32 rows x 32b (8 bytes, 4 bf16 elem). this is the easiest data movement shape to work with
    //   given our MMA_M is 128, which evenly divides into 32 rows per warp in the warpgroup.
    // - BN is 128, so 128 elem / 4 elems per 8b load = 32 loads. We can use .num=x32 to do this with one instruction.
    // - matrix fragment layout docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-3232b
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {"
        "%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31"
        "}, [%32];"
        :   "=f"(c_reg[0]), "=f"(c_reg[1]), "=f"(c_reg[2]), "=f"(c_reg[3]),
            "=f"(c_reg[4]), "=f"(c_reg[5]), "=f"(c_reg[6]), "=f"(c_reg[7]),
            "=f"(c_reg[8]), "=f"(c_reg[9]), "=f"(c_reg[10]), "=f"(c_reg[11]),
            "=f"(c_reg[12]), "=f"(c_reg[13]), "=f"(c_reg[14]), "=f"(c_reg[15]),
            "=f"(c_reg[16]), "=f"(c_reg[17]), "=f"(c_reg[18]), "=f"(c_reg[19]),
            "=f"(c_reg[20]), "=f"(c_reg[21]), "=f"(c_reg[22]), "=f"(c_reg[23]),
            "=f"(c_reg[24]), "=f"(c_reg[25]), "=f"(c_reg[26]), "=f"(c_reg[27]),
            "=f"(c_reg[28]), "=f"(c_reg[29]), "=f"(c_reg[30]), "=f"(c_reg[31])
        : "r"(tmem_addr)
    );
    // wait for tmem -> reg load to complete and be visible to thread
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

template<
    int NUM_THREADS,
    int QUEUE_SIZE,
    int BM = 128,
    int BN = 128,
    int BK = 64,
    int MMA_M = 128,
    int MMA_N = 128,
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
    const int is_master_thread = threadIdx.x == 128; // 4 epilogue warps -> 1 producer warp -> 1 consumer warp
    const int warp_id = threadIdx.x / 32;

    // init smem buffer, 1024 alignment for 128b swizzle
    extern __shared__ __align__(1024) __nv_bfloat16 smem_buffer[];

    constexpr int SMEM_A_SIZE = QUEUE_SIZE * BM * BK * sizeof(__nv_bfloat16);
    constexpr int SMEM_B_SIZE = QUEUE_SIZE * BN * BK * sizeof(__nv_bfloat16);
    constexpr int SMEM_FULL_MBAR_SIZE = QUEUE_SIZE * sizeof(uint64_t);
    constexpr int SMEM_EMPTY_MBAR_SIZE = QUEUE_SIZE * sizeof(uint64_t);
    constexpr int SMEM_MMA_MBAR_SIZE = 1 * sizeof(uint64_t);
    constexpr int SMEM_TMEM_MBAR_SIZE = 1 * sizeof(uint64_t);

    constexpr int SMEM_B_OFFSET = SMEM_A_SIZE;
    constexpr int SMEM_FULL_MBAR_OFFSET = SMEM_B_OFFSET + SMEM_B_SIZE;
    constexpr int SMEM_EMPTY_MBAR_OFFSET = SMEM_FULL_MBAR_OFFSET + SMEM_FULL_MBAR_SIZE;
    constexpr int SMEM_MMA_MBAR_OFFSET = SMEM_EMPTY_MBAR_OFFSET + SMEM_EMPTY_MBAR_SIZE;
    constexpr int SMEM_TMEM_MBAR_OFFSET = SMEM_MMA_MBAR_OFFSET + SMEM_MMA_MBAR_SIZE;
    constexpr int SMEM_TMEM_ADDR_OFFSET = SMEM_TMEM_MBAR_OFFSET + SMEM_TMEM_MBAR_SIZE;

    auto* sA = reinterpret_cast<__nv_bfloat16*[BM*BK]>(smem_buffer);
    auto* sB = reinterpret_cast<__nv_bfloat16*[BN*BK]>(smem_buffer + SMEM_A_SIZE);

     // for signaling buffer in queue is full/ready 
    auto* smem_full_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_FULL_MBAR_OFFSET);

    // for signaling buffer in queue has been read/can be re-used
    auto* smem_empty_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_EMPTY_MBAR_OFFSET);

    // for signaling tcgen05.mma batch is done
    auto* mma_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_MMA_MBAR_OFFSET);

    // for mma warp to signal TMEM buffer is ready
    auto* tmem_full_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_TMEM_MBAR_OFFSET);

    // for holding tmem addr in smem
    auto* tmem_addr_smem = reinterpret_cast<int*>(smem_buffer + SMEM_TMEM_ADDR_OFFSET);

    initialize_barriers<QUEUE_SIZE, 1>(smem_full_mbar, is_master_thread);   // 1 thread issues tma and arrive + expect tx
    initialize_barriers<QUEUE_SIZE, 1>(smem_empty_mbar, is_master_thread);  // 1 mma-issuing/waiting thread arrives when smem buffer can be re-used
    initialize_barriers<1, 1>(&mma_mbar, is_master_thread);                  // 1 thread issues mma batch, commit, wait
    initialize_barriers<1, 1>(&tmem_full_mbar, is_master_thread);            // 1 thread arrives once tmem buff ready

    // parity for coordination between tma, mma, epilogue warps
    int smem_full_parity[QUEUE_SIZE] = {0};
    int smem_empty_parity[QUEUE_SIZE] = {0};
    int mma_parity = 0;
    int tmem_full_parity = 0;

    // alloc tmem addr for accumulator. allocates BN columns of TMEM (must always alloc full 128 rows)
    // 1 warp must do the allocation, from PTX docs:
    // "When .cta_group::1 is specified, one warp from the CTA must perform the allocation and de-allocation."
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions
    __shared__ int tmem_addr_smem[1]; 
    if (warp_id == 0)
    {
        // after this call, we can load tmem_addr from smem -> to register to use.
       tcgen05_alloc<BN>(tmem_addr_smem);
    }

    // make sure mbarriers and tmem addr are visible to full threadblock
    __syncthreads();

    // read tmem addr from smem -> register
    int tmem_addr_reg = tmem_addr_smem[0];

    // prologue: prefetch first A/B tiles into first SMEM buffer
    int tma_buf_idx = 0;
    int mma_buf_idx = 0;

    constexpr int PRODUCER_WARP_ID = 4, CONSUMER_WARP_ID = 5;

    // producer 
    if (warp_id == PRODUCER_WARP_ID)
    {
        int global_m_off = block_row * BM;
        int global_n_off = block_col * BN;
        constexpr int num_bytes_a = BM * BK * 2; // 2 bytes per bf16
        constexpr int num_bytes_b = BK * BN * 2;
        const int num_blocks_k = (K + BK - 1) / BK;

        for (int block_k_idx = 0; block_k_idx < num_blocks_k; block_k_idx++) {
            // once we loop around to beginning of circular buffer for the first time,
            // we need to start waiing for consumer to finish reading from target buffer
            if (block_k_idx >= QUEUE_SIZE)
            {
                mbarrier_wait_parity(&smem_empty_mbar[tma_buf_idx], smem_empty_parity[tma_buf_idx]);
                smem_empty_parity[tma_buf_idx] ^= 1;
            }

            // tma loads for next A/B tiles
            if (is_master_thread)
            {
                int global_k_off = block_k_idx * BK;
                cp_async_bulk_tensor_3d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sA[tma_buf_idx][0]),
                    reinterpret_cast<const uint64_t*>(&a_map),
                    0,                          // x (64 dim)
                    (uint32_t)global_m_off,     // y (M dim)
                    (uint32_t)global_k_off/64,  // z (K/64 dim)
                    (uint64_t*)&smem_full_mbar[tma_buf_idx]
                );
                cp_async_bulk_tensor_3d_global_to_shared(
                    reinterpret_cast<uint64_t*>(&sB[tma_buf_idx][0]),
                    reinterpret_cast<const uint64_t*>(&b_map),
                    0,                          // x (64 dim)
                    (uint32_t)global_n_off,     // y (N dim)
                    (uint32_t)global_k_off/64,  // z (K/64 dim)
                    (uint64_t*)&smem_full_mbar[tma_buf_idx]
                );
                mbarrier_arrive_expect_tx(&smem_full_mbar[tma_buf_idx], num_bytes_a + num_bytes_b);
            }
            tma_buf_idx = (tma_buf_idx + 1) % QUEUE_SIZE;
        }
    }
    else if (warp_id == CONSUMER_WARP_ID)
    {
        // only 1 thread in consumer/mma warp issues tcgen05.mma and tcgen05.commit
        const int is_mma_master_thread = threadIdx.x == (5*32); // 4 epilogue warps -> producer warp -> consumer warp
        if (is_mma_master_thread) 
        {
            // make tcgen05 mma instruction descriptor, to be re-used at every iter of the loop
            uint32_t idesc = 0;
            tcgen05_encode_idesc<MMA_M, MMA_N>(idesc);
            printf("idesc: %d\n", idesc);

            const int num_blocks_k = (K + BK - 1) / BK;
            constexpr int mma_tiles_per_block = BK / MMA_K;
            for (int block_k_idx = 0; block_k_idx < num_blocks_k; block_k_idx++) {

                // wait for the tma load to the buffers we are about to read from.
                mbarrier_wait_parity(&smem_full_mbar[mma_buf_idx], smem_full_parity[mma_buf_idx]);
                smem_full_parity[mma_buf_idx] ^= 1;

                // tcgen05 mma for every subtile in the smem.
                for (int tile_k_idx = 0; tile_k_idx < mma_tiles_per_block; tile_k_idx++) {
                    const int smem_k_off = tile_k_idx * MMA_K;

                    void* smem_tile_a = (void*)&sA[mma_buf_idx][smem_k_off];
                    uint64_t smem_a_desc = make_smem_desc<BK>((void*)smem_tile_a);

                    void* smem_tile_b = (void*)&sB[mma_buf_idx][smem_k_off];
                    uint64_t smem_b_desc = make_smem_desc<BK>((void*)smem_tile_b);

                    tcgen05_mma(smem_a_desc, smem_b_desc, tmem_addr_reg, idesc);
                }

                // signal to producer/TMA warp this smem buffer can be re-used
                mbarrier_arrive(&smem_empty_mbar[mma_buf_idx]);

                // move to next mma buf idx in circular buffer
                mma_buf_idx = (mma_buf_idx + 1) % QUEUE_SIZE;
            }

            // commit batch of mmas
            tcgen05_commit(&mma_mbar);

            // wait for completion of batch of mmas
            mbarrier_wait_parity(&mma_mbar, mma_parity);
            mma_parity ^= 1;

            // signal to epilogue warpgroup the tmem buffer is ready
            mbarrier_arrive(&tmem_full_mbar);
        }
    }
    else  // epilogue
    {
        int ep_warp_id = (threadIdx.x / 32);
        int lane_id = threadIdx.x % 32;

        // wait for mmas to complete for this buffer idx
        mbarrier_wait_parity(&tmem_full_mbar, tmem_full_parity);
        tmem_full_parity ^= 1;

        asm volatile("tcgen05.fence::after_thread_sync;");

        // now each warp in the epilogue warpgroup reads 32 cols from TMEM to registers,
        // for a full 128x64 tile of floats. 
        float c_reg[32];
        tcgen05_ld<BN>(tmem_addr_reg, ep_warp_id, c_reg);

        // vectorized but uncoalesced writes to gmem, improve later.
        // each thread writes 16 bytes per iteration.
        int c_row = block_row * BM + ep_warp_id * 32 + lane_id;
        int c_col = block_col * BN;
        int num_iters = 32 / 16;
        for (int iter = 0; iter < num_iters; iter++) {
            *reinterpret_cast<float4*>(C + c_row * N + c_col + iter * 16) = *reinterpret_cast<float4*>(&c_reg[iter * 16]);
        }
    }

    // tmem deallocation
    if (warp_id == 0) 
    {
        tcgen05_dealloc<BN>(tmem_addr_reg);
    }
}

extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K) {
    // dims for smem tiles
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 64;

    // dims for tcgen05.mma
    constexpr int MMA_M = 128;
    constexpr int MMA_N = 128;
    constexpr int MMA_K = 16;

    assert(BM >= MMA_M && BM % MMA_M == 0);
    assert(BN >= MMA_N && BN % MMA_N == 0);
    assert(BK >= MMA_K && BK % MMA_K == 0);

    alignas(64) CUtensorMap a_map = {};
    alignas(64) CUtensorMap b_map = {};


    // BM, BK
    // BM, BK/64, 64 -> (64 elems = 128 bytes of bf16)
    // BK/64, BM, 64 -> BK/64 instances of BM,64 strips
    uint64_t a_global_dims[3] = {64, (uint64_t)M, (uint64_t)(K / 64)};
    uint32_t a_smem_dims[3] = {64, BM, BK / 64};
    uint32_t a_strides[2] = {(uint32_t)(K * sizeof(__nv_bfloat16)), 64 * sizeof(__nv_bfloat16)};
    create_3d_tensor_map(
        A,
        a_map,
        a_global_dims,
        a_smem_dims,
        a_strides
    );

    // BK, BN
    // 64, BK/64, BN
    // 64, BN, BK/64 -> BK/64 instances of BN,64 strips
    uint64_t b_global_dims[3] = {64, (uint64_t)N, (uint64_t)(K / 64)};
    uint32_t b_smem_dims[3] = {64, BN, BK / 64};
    uint32_t b_strides[2] = {(uint32_t)(K * sizeof(__nv_bfloat16)), 64 * sizeof(__nv_bfloat16)};
    create_3d_tensor_map(
        B,
        b_map,
        b_global_dims,
        b_smem_dims,
        b_strides
    );

    float* c_ptr = reinterpret_cast<float*>(C);

    constexpr int PRODUCER_WARPS = 1;
    constexpr int CONSUMER_WARPS = 1;
    constexpr int EPILOGUE_WARPS = 4;
    constexpr int NUM_THREADS = (PRODUCER_WARPS + CONSUMER_WARPS + EPILOGUE_WARPS) * 32;
    constexpr int QUEUE_SIZE = 2;

    // TMEM is 128x512 cells, each cell is 32 bits / 4 bytes.
    // So check bf16 tmem buffer requirements with 2 bytes per elem is <= tmem col width in bytes.
    constexpr int tmem_width_bytes = 512 * 4;
    assert(BN * 2 <= tmem_width_bytes);

    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));

    auto kernel = ws_gemm<NUM_THREADS, QUEUE_SIZE, BM, BN, BK, MMA_M, MMA_N, MMA_K>;

    // increase max smem
    constexpr int smem_a_size = QUEUE_SIZE * BM * BK * sizeof(__nv_bfloat16);
    constexpr int smem_b_size = QUEUE_SIZE * BN * BK * sizeof(__nv_bfloat16);
    constexpr int smem_mbar_size = (QUEUE_SIZE * 2 + 2) * sizeof(uint64_t);
    constexpr int total_smem = smem_a_size + smem_b_size + smem_mbar_size;
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        total_smem
    ));

    kernel<<<grid_dim, block_dim>>>(a_map, b_map, c_ptr, M, N, K);
}
