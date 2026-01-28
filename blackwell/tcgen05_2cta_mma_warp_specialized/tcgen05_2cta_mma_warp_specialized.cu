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
__device__ uint64_t make_smem_desc(uint32_t smem_shared_addr) {

    // bits 0-13: matrix_desc_encode(matrix addr)
    uint64_t desc = matrix_desc_encode(smem_shared_addr);

    // bits 16-29: matrix_desc_encode(leading dim byte offset)
    // implied to be 1 for swizzled layouts, see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-leading-dimension-byte-offset

    // bits 32-45: matrix_desc_encode(stride dim byte offset)
    // SBO (stride byte offset) is stride between rows of swizzle atoms which are (8x64 matrix of bf16 in this 128b swizzle setup)
    uint64_t SBO = 8 * 64 * sizeof(__nv_bfloat16);
    desc |= (matrix_desc_encode(SBO) << 32);

    // bits 46-49: fixed value of 0b001
    desc |= (1ULL << 46ULL);

    // bits 61-63: swizzle mode (2 = 128b swizzle)
    desc |= (2ULL << 61ULL);
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

    // explicit fence to make mbar init (on generic proxy) visible to async proxy (for TMA)
    asm volatile("fence.proxy.async.shared::cta;");
  }
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_addr, const uint32_t parity) {
  uint32_t wait_complete;
  asm volatile(
    "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(wait_complete)         // outputs
        : "r"(mbar_addr), "r"(parity) // inputs
        : "memory"
  );
  return static_cast<bool>(wait_complete);
}

__device__ __forceinline__ void mbarrier_wait_parity(uint32_t mbar_addr, const uint32_t parity) {
  while (!mbarrier_try_wait_parity(mbar_addr, parity)) {
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive(uint32_t mbar_addr) {
  asm volatile(
        "mbarrier.arrive.release.cluster.b64 _, [%0];"
        :                   // no outputs
        :"r"(mbar_addr)     // input
        : "memory"
  );
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mbar_addr, const uint32_t tx_count) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cluster.b64 _, [%0], %1;"
      :                             // no outputs
      :"r"(mbar_addr), "r"(tx_count) // inputs
      : "memory");
}


__device__ __forceinline__ void cp_async_bulk_tensor_3d_global_to_shared(
    const uint32_t dst_shmem,           // shared addr
    const uint64_t *tensor_map_ptr,
    const uint32_t offset_x,
    const uint32_t offset_y,
    const uint32_t offset_z,
    uint32_t mbar_addr                  // shared addr
) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :
        :
        "r"(dst_shmem),
        "l"(tensor_map_ptr),
        "r"(offset_x),
        "r"(offset_y),
        "r"(offset_z),
        "r"(mbar_addr)
        : "memory"
    );
}

template <int BN, int CTA_GROUP>
__device__ __forceinline__ void tcgen05_alloc(int* tmem_addr_smem) {
    // convert to shared addr
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr_smem));
    asm volatile(
        "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
        : // no outputs
        : "r"(addr), "r"(BN), "n"(CTA_GROUP) // inputs
    );
}

template <int BN, int CTA_GROUP>
__device__ __forceinline__ void tcgen05_dealloc(int tmem_addr_reg) {
    asm volatile(
        "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
        : // no outputs
        : "r"(tmem_addr_reg), "r"(BN), "n"(CTA_GROUP)
    );    
}

template <int MMA_M, int MMA_N, int CTA_GROUP>
__device__ __forceinline__ void tcgen05_encode_idesc(uint32_t& idesc) {
    // create idesc
    // from PTX docs: "The 32-bit register operand idesc is the instruction descriptor as described in 
    //   Instruction descriptor, specifies the shapes, exact types, sparsity and other details of the 
    //   input matrices, output matrix and the matrix multiply and accumulate operation."
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    idesc |= (1 << 4);                          // fp32 output matrix D
    idesc |= (1 << 7);                          // bf16 input matrix A
    idesc |= (1 << 10);                         // bf16 input matrix B
    idesc |= (MMA_N >> 3) << 17;                // N dim of matrix B
    idesc |= ((CTA_GROUP * MMA_M) >> 4) << 24;  // M dim of matrix A
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_mma(
    uint64_t smem_a_desc,
    uint64_t smem_b_desc,
    int tmem_accum_addr,
    uint32_t idesc,
    int enable_accum
) {
    // tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc, { disable-output-lane }, enable-input-d {, scale-input-d};
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"              // declare predicate register for enable-input-d
        "setp.ne.s32 p, %4, 0;\n\t"      // p = 0 for mma tile 0, then 1 after
        "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n"
        "}"
        :
        : "r"(tmem_accum_addr), "l"(smem_a_desc), "l"(smem_b_desc), "r"(idesc), "r"(enable_accum), "n"(CTA_GROUP)
    );
}

// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma
template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_commit_multicast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        // tcgen05.commit.cta_group.completion_mechanism{.shared::cluster}{.multicast}.b64
        // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit
        "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :
        : "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP)
        : "memory"
    );
}

// from PTX docs:  "The Tensor Memory of a CTA is divided into 4 equal chunks such that 
// each warp of a warpgroup in the CTA can access a chunk of the Tensor Memory. 
// All the columns of the Tensor Memory can be accessed by all the four warps of a warpgroup."
// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st
__device__ __forceinline__ void tcgen05_ld(int tmem_base_addr_reg, int ep_warp_id, float c_reg[4], int base_col) {
    // warp 0 can access lanes 0-31
    // warp 1 can access lanes 32-63
    // warp 2 can access lanes 64-95
    // warp 3 can access lanes 96-128
    int row = ep_warp_id * 32;

    // TMEM address is 32bit and composed of 2 components:
    // - bits 0-15: column index
    // - bits 16-31: row index
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing
    int tmem_addr = tmem_base_addr_reg + (row << 16) + base_col;

    // with .x4, the warp loads 32 rows × 4 columns, where each lane gets 4 floats in register memory.
    // see: matrix fragment layout docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-3232b
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32 {"
        "%0, %1, %2, %3"
        "}, [%4];"
        :   "=f"(c_reg[0]), "=f"(c_reg[1]), "=f"(c_reg[2]), "=f"(c_reg[3])
        : "r"(tmem_addr)
    );

    // from PTX docs: "Prevents subsequent tcgen05.mma from racing ahead of the tcgen05.ld"
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ uint32_t map_smem_addr_to_cta_rank(uint32_t shared_addr, int cta_rank) {
  uint32_t mapped_addr;
  asm volatile("mapa.shared::cluster.u32  %0, %1, %2;\n"
              : "=r"(mapped_addr)
              : "r"(shared_addr), "r"(cta_rank));
  return mapped_addr;
}

// like __syncthreads() but for the cluster
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

template<
    int NUM_THREADS,
    int QUEUE_SIZE,
    int CTA_GROUP,
    int BM = 128,
    int BN = 128,
    int BK = 64,
    int MMA_M = 128,
    int MMA_N = 128,
    int MMA_K = 16
>
__global__ 
__cluster_dims__(2, 1, 1)  // tb cluster for 2 cta mmma
void ws_gemm(
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
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // convert to shared addr now for easier offset addition with integers, etc
    uint32_t smem = __cvta_generic_to_shared(smem_buffer);
    
    constexpr int SMEM_A_SIZE = BM * BK * sizeof(__nv_bfloat16);
    constexpr int SMEM_B_SIZE = BN * BK * sizeof(__nv_bfloat16);

    // init mbarriers
    __shared__ __align__(8) uint64_t smem_full_mbar[QUEUE_SIZE];    // for signaling buffer in queue is full/ready 
    __shared__ __align__(8) uint64_t smem_empty_mbar[QUEUE_SIZE];   // for signaling buffer in queue has been read/can be re-used
    __shared__ __align__(8) uint64_t mma_mbar;                      // for signaling tcgen05.mma batch is done
    __shared__ __align__(8) uint64_t tmem_full_mbar;                // for mma warp to signal TMEM buffer is ready

    // get shared addrs for mbars - used for mapping to peer CTA for DSMEM access
    int smem_full_mbar_addr = static_cast<int>(__cvta_generic_to_shared((const void*)smem_full_mbar));
    int smem_empty_mbar_addr = static_cast<int>(__cvta_generic_to_shared((const void*)smem_empty_mbar));
    int mma_mbar_addr = static_cast<int>(__cvta_generic_to_shared((const void*)mma_mbar));
    int tmem_full_mbar_addr = static_cast<int>(__cvta_generic_to_shared((const void*)tmem_full_mbar));

    // get threadblock rank in cluster
    // see: https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/include/cute/arch/cluster_sm90.hpp#L158
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %cluster_ctaid.x;" : "=r"(cta_rank));

    if (cta_rank == 0)
    {
        // only owner CTA should init mbarriers
        initialize_barriers<QUEUE_SIZE, 1>(smem_full_mbar, is_master_thread);   // 1 thread issues tma and arrive + expect tx
        initialize_barriers<QUEUE_SIZE, 1>(smem_empty_mbar, is_master_thread);  // 1 mma-issuing/waiting thread arrives when smem buffer can be re-used
        initialize_barriers<1, 1>(&mma_mbar, is_master_thread);                 // 1 thread issues mma batch, commit, wait
        initialize_barriers<1, 1>(&tmem_full_mbar, is_master_thread);           // 1 thread arrives once tmem buff ready
    }
    else
    {
        // peer CTA (1) maps mbarrier smem base addr to owner CTA (0)
        smem_full_mbar_addr = map_smem_addr_to_cta_rank(smem_full_mbar_addr, 0);
        smem_empty_mbar_addr = map_smem_addr_to_cta_rank(smem_empty_mbar_addr, 0);
        mma_mbar_addr = map_smem_addr_to_cta_rank(mma_mbar_addr, 0);
        tmem_full_mbar_addr = map_smem_addr_to_cta_rank(tmem_full_mbar_addr, 0);
    }

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
       tcgen05_alloc<BN, CTA_GROUP>(tmem_addr_smem);
    }

    // make sure mbarriers and tmem addr are visible to full cluster
    cluster_sync();

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
                mbarrier_wait_parity(smem_empty_mbar_addr + tma_buf_idx * 8, smem_empty_parity[tma_buf_idx]);
                smem_empty_parity[tma_buf_idx] ^= 1;
            }

            // tma loads for next A/B tiles
            if (is_master_thread)
            {
                const uint32_t A_smem = smem + tma_buf_idx * (SMEM_A_SIZE + SMEM_B_SIZE);
                const uint32_t B_smem = A_smem + SMEM_A_SIZE; 
                int global_k_off = block_k_idx * BK;
                
                cp_async_bulk_tensor_3d_global_to_shared(
                    A_smem,
                    reinterpret_cast<const uint64_t*>(&a_map),
                    0,                          // x (64 dim)
                    (uint32_t)global_m_off,     // y (M dim)
                    (uint32_t)global_k_off/64,  // z (K/64 dim)
                    smem_full_mbar_addr + tma_buf_idx * 8
                );
                cp_async_bulk_tensor_3d_global_to_shared(
                    B_smem,
                    reinterpret_cast<const uint64_t*>(&b_map),
                    0,                          // x (64 dim)
                    (uint32_t)global_n_off,     // y (N dim)
                    (uint32_t)global_k_off/64,  // z (K/64 dim)
                    smem_full_mbar_addr + tma_buf_idx * 8
                );
                mbarrier_arrive_expect_tx(smem_full_mbar_addr + tma_buf_idx * 8, num_bytes_a + num_bytes_b);
            }
            tma_buf_idx = (tma_buf_idx + 1) % QUEUE_SIZE;
        }
    }
    else if (warp_id == CONSUMER_WARP_ID)
    {
        // only 1 thread in consumer/mma warp issues tcgen05.mma and tcgen05.commit.
        // we have: 4 epilogue warps -> producer warp -> consumer warp.
        // choose first thread in consumer warp as master.
        const int is_mma_master_thread = threadIdx.x == (5*32); 
        if (is_mma_master_thread) 
        {
            // make tcgen05 mma instruction descriptor, to be re-used at every iter of the loop
            uint32_t idesc = 0;
            tcgen05_encode_idesc<MMA_M, MMA_N, CTA_GROUP>(idesc);

            const int num_blocks_k = (K + BK - 1) / BK;
            for (int block_k_idx = 0; block_k_idx < num_blocks_k; block_k_idx++) {

                // wait for the tma load to the buffers we are about to read from.
                mbarrier_wait_parity(smem_full_mbar_addr + mma_buf_idx * 8, smem_full_parity[mma_buf_idx]);
                smem_full_parity[mma_buf_idx] ^= 1;

                // tcgen05 mma for every subtile in the smem.
                // 128b swizzle means we iterate through each swizzle atom:
                // BK/128 byte chunks of (BM,128 bytes) or (BM, 64 elem).
                // Within each swizzle atom, we iterate through mma tiles
                for (int bk_chunk = 0; bk_chunk < BK/64; bk_chunk++) {
                    for (int mma_iter = 0; mma_iter < 64/MMA_K; mma_iter++) {
                        const int a_chunk_off = bk_chunk * BM * 64 * sizeof(__nv_bfloat16);
                        const int b_chunk_off = bk_chunk * BN * 64 * sizeof(__nv_bfloat16);
                        const int a_k_off = a_chunk_off + mma_iter * MMA_K * sizeof(__nv_bfloat16);
                        const int b_k_off = b_chunk_off + mma_iter * MMA_K * sizeof(__nv_bfloat16);

                        uint32_t smem_buff_a = smem + mma_buf_idx * (SMEM_A_SIZE + SMEM_B_SIZE);
                        uint32_t smem_buff_b = smem_buff_a + SMEM_A_SIZE;

                        uint64_t smem_a_desc = make_smem_desc(smem_buff_a + a_k_off);
                        uint64_t smem_b_desc = make_smem_desc(smem_buff_b + b_k_off);

                        int enable_accum = (block_k_idx == 0 && bk_chunk == 0 && mma_iter == 0) ? 0 : 1;
                        tcgen05_mma<CTA_GROUP>(smem_a_desc, smem_b_desc, tmem_addr_reg, idesc, enable_accum);
                    }
                }

                // signal to producer/TMA warp this smem buffer can be re-used
                mbarrier_arrive(smem_empty_mbar_addr + mma_buf_idx * 8);

                // move to next mma buf idx in circular buffer
                mma_buf_idx = (mma_buf_idx + 1) % QUEUE_SIZE;
            }

            // commit batch of mmas. this is multicasted to both CTA mbar addr.
            // cta 0 -> commit using it's own mbar
            // cta 1 -> commit using cta 0's mbar
            // from PTX docs: 
            //   "Operand ctaMask specifies the destination CTAs in the cluster such that 
            //    each bit position in the 16-bit ctaMask operand corresponds to the %ctaid of the destination CTA...
            //    The mbarrier signal is multicasted either to all the odd numbered CTAs or the even numbered CTAs 
            //    within the corresponding CTA-Pair. For each destination CTA specified in the ctaMask, 
            //    the mbarrier signal is sent either to the destination CTA or its peer-CTA based on 
            //    CTAs %cluster_ctarank parity of shared memory where the mbarrier object mbar resides."
            // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk
            uint16_t cta_mask = 0b11; // 1 bit for cta rank 0, 1 bit for cta rank 1
            tcgen05_commit_multicast<CTA_GROUP>(mma_mbar_addr, cta_mask);

            // wait for completion of batch of mmas
            mbarrier_wait_parity(mma_mbar_addr, mma_parity);
            mma_parity ^= 1;

            // signal to epilogue warpgroup the tmem buffer is ready
            mbarrier_arrive(tmem_full_mbar_addr);
        }
    }
    else  // epilogue
    {
        int ep_warp_id = (threadIdx.x / 32);
        int lane_id = threadIdx.x % 32;

        // wait for mmas to complete for this buffer idx
        mbarrier_wait_parity(tmem_full_mbar_addr, tmem_full_parity);
        tmem_full_parity ^= 1;

        asm volatile("tcgen05.fence::after_thread_sync;");

        // now each warp in the epilogue warpgroup reads 32 rows of 16 floats per thread. 
        // these floats are contiguous in the output C.
        constexpr int COLS_PER_THREAD = 4;

        #pragma unroll
        for (int i = 0; i < BN/COLS_PER_THREAD; i++) {
            float c_reg[COLS_PER_THREAD];
            const int base_col = i * COLS_PER_THREAD;
            tcgen05_ld(tmem_addr_reg, ep_warp_id, c_reg, base_col);

            // vectorized but uncoalesced writes to gmem, improve later.
            const int c_row = block_row * BM + ep_warp_id * 32 + lane_id;
            const int c_col = block_col * BN + i * COLS_PER_THREAD;
            *reinterpret_cast<float4*>(C + c_row * N + c_col) = *reinterpret_cast<float4*>(&c_reg);
        }
    }

    // tmem deallocation after all threads finished using tmem.
    // need cluster wide sync for 2 cta mma
    cluster_sync();
    if (warp_id == 0) 
    {
        tcgen05_dealloc<BN, CTA_GROUP>(tmem_addr_reg);
    }
}

extern "C" void launch_gemm(void* A, void* B, void* C, int M, int N, int K) {
    // dims for smem tiles
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 64;

    // dims for tcgen05.mma
    constexpr int MMA_M = 128;
    constexpr int MMA_N = 256;
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
    constexpr int QUEUE_SIZE = 4;
    constexpr int CTA_GROUP = 2;

    // TMEM is 128x512 cells, each cell is 32 bits / 4 bytes.
    // So check bf16 tmem buffer requirements with 2 bytes per elem is <= tmem col width in bytes.
    constexpr int tmem_width_bytes = 512 * 4;
    assert(BN * 2 <= tmem_width_bytes);

    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };

    dim3 block_dim(NUM_THREADS);
    dim3 grid_dim(ceil_div(N, BN), ceil_div(M, BM));

    auto kernel = ws_gemm<NUM_THREADS, QUEUE_SIZE, CTA_GROUP, BM, BN, BK, MMA_M, MMA_N, MMA_K>;

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

    kernel<<<grid_dim, block_dim, total_smem>>>(a_map, b_map, c_ptr, M, N, K);
}
