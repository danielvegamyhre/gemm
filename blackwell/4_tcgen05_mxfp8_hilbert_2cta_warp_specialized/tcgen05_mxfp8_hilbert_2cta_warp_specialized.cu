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
    const uint32_t stride_bytes[2],
    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
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
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        tensor_ptr,
        size,
        stride,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    CUDA_CHECK(res);
}

void create_4d_tensor_map(
    void* tensor_ptr,
    CUtensorMap& tensor_map,
    const uint64_t global_dims[4],
    const uint32_t smem_dims[4],
    const uint32_t stride_bytes[3],
    CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
) {
    constexpr uint32_t rank = 4;
    uint64_t size[rank] = {global_dims[0], global_dims[1], global_dims[2], global_dims[3]};
    uint64_t stride[rank - 1] = {stride_bytes[0], stride_bytes[1], stride_bytes[2]};
    uint32_t box_size[rank] = {smem_dims[0], smem_dims[1], smem_dims[2], smem_dims[3]};
    uint32_t elem_stride[rank] = {1, 1, 1, 1};

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
        swizzle,
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
    // SBO (stride byte offset) is stride between rows of swizzle atoms which are (8x128 matrix of e4m3 in this 128b swizzle setup)
    uint64_t SBO = 8 * 128;
    desc |= (matrix_desc_encode(SBO) << 32);

    // bits 46-49: fixed value of 0b001
    desc |= (1ULL << 46ULL);

    // bits 61-63: swizzle mode (2 = 128b swizzle)
    desc |= (2ULL << 61ULL);
    return desc;
}

__device__ uint64_t make_sf_smem_desc(uint32_t smem_shared_addr) {
    // bits 0-13: matrix_desc_encode(matrix addr)
    uint64_t desc = matrix_desc_encode(smem_shared_addr);

    // bits 16-29: matrix_desc_encode(leading dim byte offset)
    // LBO: stride between rows of core matrices (vertical) = 8 rows × 4 bytes/row within a block
    // uint64_t LBO = 32;
    // desc |= (matrix_desc_encode(LBO) << 16);

    // bits 32-45: matrix_desc_encode(stride dim byte offset)
    // SBO: stride horizontally between blocks = 128 bytes to next block
    uint64_t SBO = 128;
    desc |= (matrix_desc_encode(SBO) << 32);

    // bits 46-49: fixed value of 0b001
    desc |= (1ULL << 46ULL);

    // bits 61-63: swizzle mode (0 = no swizzle)
    return desc;
}

// init mbar for sync between tma and regular
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count) {
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile(
    "mbarrier.init.shared::cta.b64 [%0], %1;"
    : // no outputs
    :"r"(mbar_ptr), "r"(count) // inputs
    : "memory"
  );
}

template <int num_barriers, int THREADS_PER_BLOCK>
__forceinline__ __device__ void initialize_barriers(uint64_t *mbar, const bool is_producer_master_thread) {
  if (is_producer_master_thread) {
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
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  P_OUT, [%1], %2; \n\t" 
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
        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
        :                   // no outputs
        :"r"(mbar_addr)     // input
        : "memory"
  );
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mbar_addr, const uint32_t tx_count) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
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
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
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

__device__ __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
    const uint32_t dst_shmem,           // shared addr
    const uint64_t *tensor_map_ptr,
    const uint32_t offset_x,
    const uint32_t offset_y,
    const uint32_t offset_z,
    const uint32_t offset_w,
    uint32_t mbar_addr                  // shared addr
) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2 "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];"
        :
        :
        "r"(dst_shmem),
        "l"(tensor_map_ptr),
        "r"(offset_x),
        "r"(offset_y),
        "r"(offset_z),
        "r"(offset_w),
        "r"(mbar_addr)
        : "memory"
    );
}

template <int TMEM_COLS>
__device__ __forceinline__ void tcgen05_alloc(uint32_t tmem_addr_smem) {
    // convert to shared addr
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        : // no outputs
        : "r"(tmem_addr_smem), "r"(TMEM_COLS) // inputs
    );
}

template <int BN>
__device__ __forceinline__ void tcgen05_dealloc(int tmem_addr_reg) {
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        : // no outputs
        : "r"(tmem_addr_reg), "r"(BN)
    );
}

template <int MMA_M, int MMA_N>
__device__ __forceinline__ void tcgen05_encode_idesc(uint32_t& idesc, int sfa_id) {
    // create idesc
    // from PTX docs: "The 32-bit register operand idesc is the instruction descriptor as described in
    //   Instruction descriptor, specifies the shapes, exact types, sparsity and other details of the
    //   input matrices, output matrix and the matrix multiply and accumulate operation."
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor

    // e4m3 input matrix A and B => encode 0 for bits 7-9 and bits 10-12 (so no-op here).
    // then:
    idesc |= (1 << 4);                          // fp32 output matrix D
    idesc |= (MMA_N >> 3) << 17;                // N dim of matrix B (without 3 LSBs) for bits 17-22
    idesc |= (1 << 23);                         // scale dtype for both scale_A / scale_B (bit 23). 1=e8m0
    idesc |= (MMA_M >> 7) << 27;                // M dim of matrix A (without 7 LSBs) for bits 27-28
    idesc |= (sfa_id << 29);                    // matrix A scale factor ID (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a)
}

__device__ __forceinline__ void tcgen05_mma_mxfp8(
    uint64_t smem_a_desc,
    uint64_t smem_b_desc,
    int tmem_sfa_addr,
    int tmem_sfb_addr,
    int tmem_accum_addr,
    uint32_t idesc,
    int enable_accum
) {
    // tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    //                                    [d-tmem],  a-desc,  b-desc, idesc,
    //                                    [scale-A-tmem], [scale-B-tmem], enable-input-d;
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"              // declare predicate register for enable-input-d
        "setp.ne.s32 p, %6, 0;\n\t"      // p = 0 for mma tile 0, then 1 after
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}"
        :
        : "r"(tmem_accum_addr), "l"(smem_a_desc), "l"(smem_b_desc), "r"(idesc), "r"(tmem_sfa_addr), "r"(tmem_sfb_addr), "r"(enable_accum)
    );
}

// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma
__device__ __forceinline__ void tcgen05_commit_multicast(uint32_t mbar_addr, uint16_t cta_mask) {
    asm volatile(
        // tcgen05.commit.cta_group.completion_mechanism{.shared::cluster}{.multicast}.b64
        // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :
        : "r"(mbar_addr), "h"(cta_mask)
        : "memory"
    );
}

// from PTX docs:  "The Tensor Memory of a CTA is divided into 4 equal chunks such that 
// each warp of a warpgroup in the CTA can access a chunk of the Tensor Memory. 
// All the columns of the Tensor Memory can be accessed by all the four warps of a warpgroup."
// see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st
__device__ __forceinline__ void tcgen05_ld_tmem_to_reg(int tmem_base_addr_reg, int row, int base_col, float c_reg[4]) {
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

__device__ __forceinline__ void tcgen05_cp_smem_to_tmem(uint64_t smem_desc, int tmem_base_addr_reg, int row, int base_col) {
    // TMEM address is 32bit and composed of 2 components:
    // - bits 0-15: column index
    // - bits 16-31: row index
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing
    int tmem_addr = tmem_base_addr_reg + (row << 16) + base_col;
    
    // PTX docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-cp
    // tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
    // * 32x128b -> 32x16bytes -> one contiguous 512 byte SF tile in ((32,4),4) layout
    // * warpx4 -> multicast to 4 warps (confusing because we only need 1 producer/TMA warp and 1 consumer/MMA warp) 
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(tmem_addr), "l"(smem_desc));
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

// Hilbert curve mapping utilities for cache-aware scheduling
// see: https://en.wikipedia.org/wiki/Hilbert_curve
__device__ __forceinline__ void hilbert_rot(int n, int* x, int* y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        // Swap x and y
        int t = *x;
        *x = *y;
        *y = t;
    }
}

// Convert from Hilbert curve index to 2D coordinates
// n must be a power of 2 (size of the grid dimension)
__device__ __forceinline__ void hilbert_index_to_xy(int n, int index, int* x, int* y) {
    *x = 0;
    *y = 0;
    for (int s = 1; s < n; s *= 2) {
        int rx = 1 & (index / 2);
        int ry = 1 & (index ^ rx);
        hilbert_rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        index /= 4;
    }
}

__device__ __forceinline__ std::pair<int, int> compute_bid_hilbert(int bid, int grid_m, int grid_n) {
    constexpr int CTA_GROUP_SIZE = 2;
    const int group_id = bid / CTA_GROUP_SIZE;

    // Find smallest power of 2 that fits both dimensions
    int max_dim = max(grid_m, grid_n);
    int hilbert_size = 1;
    while (hilbert_size < max_dim) {
        hilbert_size *= 2;
    }

    // Map group_id to 2D coordinates using Hilbert curve
    int group_m, group_n;
    hilbert_index_to_xy(hilbert_size, group_id, &group_n, &group_m);

    // Clamp to actual grid bounds and handle out-of-bounds by wrapping
    // This ensures we still process all tiles even with non-power-of-2 grids
    if (group_m >= grid_m || group_n >= grid_n) {
        // Fallback to linear mapping for out-of-bounds indices
        int valid_group_id = group_id % (grid_m * grid_n);
        group_n = valid_group_id % grid_n;
        group_m = valid_group_id / grid_n;
    }

    // Convert group coordinates to block coordinates
    // Each group is CTA_GROUP_SIZE blocks tall
    const int base_block_m = group_m * CTA_GROUP_SIZE;
    const int block_n = group_n;
    const int block_m = base_block_m + (bid % CTA_GROUP_SIZE);

    return {block_m, block_n};
}

__device__ __forceinline__ std::pair<int, int> compute_bid(int bid, int grid_n) {
    constexpr int CTA_GROUP_SIZE = 2;
    const int group_id = bid / CTA_GROUP_SIZE;

    // group advance along N first (rows of vertically stacked pairs)
    const int block_n = group_id % grid_n;

    // get M/row coordinate of base block in pair
    const int base_block_m = (group_id / grid_n) * CTA_GROUP_SIZE;

    // base block row id + (linear block id % BLOCKS PER GROUP) = final block row id
    const int block_m = base_block_m + (bid % CTA_GROUP_SIZE);

    return {block_m, block_n};
}

template<int QUEUE_SIZE, int BM, int BN, int BK>
__device__ __noinline__
void producer_warp(
    const CUtensorMap* a_map,
    const CUtensorMap* b_map,
    const CUtensorMap* sfa_map,
    const CUtensorMap* sfb_map,
    int K,
    int start_bid,
    int start_group_id,
    int num_groups,
    int total_groups,
    uint32_t cta_rank,
    int grid_m,
    int grid_n,
    uint32_t smem,
    uint32_t smem_full_mbar_addr,
    uint32_t smem_empty_mbar_addr
) {
    constexpr int CTA_GROUP_SIZE = 2;
    constexpr int SF_BK = BK / 32;
    constexpr int SMEM_A_SIZE = BM * BK;
    constexpr int SMEM_B_SIZE = (BN / CTA_GROUP_SIZE) * BK;
    constexpr int SMEM_SFA_SIZE = BM * SF_BK;
    constexpr int SMEM_SFB_SIZE = BN * SF_BK;

    int tma_smem_buf = 0;
    int smem_empty_parity[QUEUE_SIZE] = {0};

    // pre-compute mapped barrier addresses to avoid repeated arithmetic and mapping in loop
    uint32_t smem_full_mbar_addrs[QUEUE_SIZE];
    uint32_t smem_empty_mbar_addrs[QUEUE_SIZE];
    for (int i = 0; i < QUEUE_SIZE; i++) {
        smem_full_mbar_addrs[i] = smem_full_mbar_addr + i * sizeof(uint64_t);
        smem_empty_mbar_addrs[i] = smem_empty_mbar_addr + i * sizeof(uint64_t);
        if (cta_rank == 1) {
            smem_full_mbar_addrs[i] = map_smem_addr_to_cta_rank(smem_full_mbar_addrs[i], 0);
        }
    }

    for (int group_id = start_group_id; group_id < total_groups; group_id += num_groups) {
        const int bid = group_id * CTA_GROUP_SIZE + (start_bid % CTA_GROUP_SIZE);
        auto [block_m, block_n] = compute_bid_hilbert(bid, grid_m, grid_n);

        int global_m_off = block_m * BM;
        int global_n_off = block_n * BN + cta_rank * BN / 2;
        const int num_blocks_k = (K + BK - 1) / BK;

        for (int block_k_idx = 0; block_k_idx < num_blocks_k; block_k_idx++) {
            if (block_k_idx >= QUEUE_SIZE || group_id > start_group_id)
            {
                mbarrier_wait_parity(smem_empty_mbar_addrs[tma_smem_buf], smem_empty_parity[tma_smem_buf]);
                smem_empty_parity[tma_smem_buf] ^= 1;
            }

            // smem offsets for next A/B tiles - layout: [A, B, SFA, SFB]
            constexpr int BUFF_SIZE = SMEM_A_SIZE + SMEM_B_SIZE + SMEM_SFA_SIZE + SMEM_SFB_SIZE;
            const uint32_t A_smem = smem + tma_smem_buf * BUFF_SIZE;
            const uint32_t B_smem = A_smem + SMEM_A_SIZE;
            const uint32_t SFA_smem = B_smem + SMEM_B_SIZE;
            const uint32_t SFB_smem = SFA_smem + SMEM_SFA_SIZE;

            mbarrier_arrive_expect_tx(smem_full_mbar_addrs[tma_smem_buf], SMEM_A_SIZE + SMEM_B_SIZE + SMEM_SFA_SIZE + SMEM_SFB_SIZE);

            // load A tile
            int global_k_off = block_k_idx * BK;
            cp_async_bulk_tensor_3d_global_to_shared(
                A_smem,
                reinterpret_cast<const uint64_t*>(a_map),
                0,
                (uint32_t)global_m_off,
                (uint32_t)global_k_off/128,
                smem_full_mbar_addrs[tma_smem_buf]
            );

            // load B tile
            cp_async_bulk_tensor_3d_global_to_shared(
                B_smem,
                reinterpret_cast<const uint64_t*>(b_map),
                0,
                (uint32_t)global_n_off,
                (uint32_t)global_k_off/128,
                smem_full_mbar_addrs[tma_smem_buf]
            );

            // load SFA/SFB tiles
            cp_async_bulk_tensor_4d_global_to_shared(
                SFA_smem,
                reinterpret_cast<const uint64_t*>(sfa_map),
                0,
                0,
                (uint32_t)(global_k_off/32/4),
                (uint32_t)(global_m_off/128),
                smem_full_mbar_addrs[tma_smem_buf]
            );
            cp_async_bulk_tensor_4d_global_to_shared(
                SFB_smem,
                reinterpret_cast<const uint64_t*>(sfb_map),
                0,
                0,
                (uint32_t)(global_k_off/32/4),
                (uint32_t)(global_n_off/128),
                smem_full_mbar_addrs[tma_smem_buf]
            );

            tma_smem_buf = (tma_smem_buf + 1) % QUEUE_SIZE;
        }
    }
}

template<int QUEUE_SIZE, int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K>
__device__ __noinline__
void consumer_warp(
    int K,
    int start_group_id,
    int num_groups,
    int total_groups,
    uint32_t smem,
    uint32_t smem_full_mbar_addr,
    uint32_t smem_empty_mbar_addr,
    uint32_t mma_mbar_addr,
    uint32_t epilogue_mbar_addr,
    uint32_t tmem_addr_smem
) {
    constexpr int CTA_GROUP_SIZE = 2;
    constexpr int TMEM_BUFFERS = 2;
    constexpr int SF_BK = BK / 32;
    constexpr int SMEM_A_SIZE = BM * BK;
    constexpr int SMEM_B_SIZE = (BN / CTA_GROUP_SIZE) * BK;
    constexpr int SMEM_SFA_SIZE = BM * SF_BK;
    constexpr int SMEM_SFB_SIZE = BN * SF_BK;

    int mma_smem_buf = 0;
    int mma_tmem_buf = 0;
    int smem_full_parity[QUEUE_SIZE] = {0};
    int epilogue_parity[TMEM_BUFFERS] = {0};
    int blocks_processed = 0;
    int tmem_base_addr = *reinterpret_cast<int*>(__cvta_shared_to_generic(tmem_addr_smem));

    uint16_t cta_mask = 0b11;
    const int num_blocks_k = (K + BK - 1) / BK;

    // pre-compute barrier addresses
    uint32_t epilogue_mbar_addrs[TMEM_BUFFERS];
    uint32_t smem_full_mbar_addrs[QUEUE_SIZE];
    uint32_t smem_empty_mbar_addrs[QUEUE_SIZE];
    uint32_t mma_mbar_addrs[TMEM_BUFFERS];
    for (int i = 0; i < TMEM_BUFFERS; i++) {
        epilogue_mbar_addrs[i] = epilogue_mbar_addr + i * sizeof(uint64_t);
        mma_mbar_addrs[i] = mma_mbar_addr + i * sizeof(uint64_t);
    }
    for (int i = 0; i < QUEUE_SIZE; i++) {
        smem_full_mbar_addrs[i] = smem_full_mbar_addr + i * sizeof(uint64_t);
        smem_empty_mbar_addrs[i] = smem_empty_mbar_addr + i * sizeof(uint64_t);
    }

    // calculate tmem addresses
    constexpr int SFA_TILES = (BM * SF_BK) / 512;
    constexpr int SFB_TILES = (BN * SF_BK) / 512;
    constexpr int SFA_COLS = SFA_TILES * 16;
    constexpr int SFB_COLS = SFB_TILES * 16;
    constexpr int TMEM_WIDTH = BN + SFA_COLS + SFB_COLS;

    for (int group_id = start_group_id; group_id < total_groups; group_id += num_groups) {

        int buffer_offset = mma_tmem_buf * TMEM_WIDTH;
        int tmem_accum_addr = tmem_base_addr + buffer_offset;
        int tmem_sfa_base_addr = tmem_base_addr + buffer_offset + BN;
        int tmem_sfb_base_addr = tmem_base_addr + buffer_offset + BN + SFA_COLS;

        if (blocks_processed >= TMEM_BUFFERS) {
            mbarrier_wait_parity(epilogue_mbar_addrs[mma_tmem_buf], epilogue_parity[mma_tmem_buf]);
            epilogue_parity[mma_tmem_buf] ^= 1;
        }

        for (int block_k_idx = 0; block_k_idx < num_blocks_k; block_k_idx++) {
            mbarrier_wait_parity(smem_full_mbar_addrs[mma_smem_buf], smem_full_parity[mma_smem_buf]);
            smem_full_parity[mma_smem_buf] ^= 1;

            // copy SF tiles from SMEM -> TMEM
            constexpr int BUFF_SIZE = SMEM_A_SIZE + SMEM_B_SIZE + SMEM_SFA_SIZE + SMEM_SFB_SIZE;
            const int smem_sfa_base = mma_smem_buf * BUFF_SIZE + SMEM_A_SIZE + SMEM_B_SIZE;
            const int smem_sfb_base = smem_sfa_base + SMEM_SFA_SIZE;

            #pragma unroll
            for (int sfa_off = 0; sfa_off < SMEM_SFA_SIZE; sfa_off += 512) {
                uint64_t sfa_desc = make_sf_smem_desc(smem + smem_sfa_base + sfa_off);

                // 32x16 load broadcasted to all 4 warp zones of TMEM (32 rows each -> 128 rows)
                tcgen05_cp_smem_to_tmem(sfa_desc, tmem_sfa_base_addr, 0, sfa_off / 32); // 512/32= 16 cols per ((32,4),4) SF tile
            }

            #pragma unroll
            for (int sfb_off = 0; sfb_off < SMEM_SFB_SIZE; sfb_off += 512) {
                uint64_t sfb_desc = make_sf_smem_desc(smem + smem_sfb_base + sfb_off);
                tcgen05_cp_smem_to_tmem(sfb_desc, tmem_sfb_base_addr, 0, sfb_off / 32);
            }

            // tcgen05 mma
            constexpr int SWIZZLE_ATOM_K = 128;
            for (int bk_chunk = 0; bk_chunk < BK / SWIZZLE_ATOM_K; bk_chunk++) {
                for (int mma_iter = 0; mma_iter < SWIZZLE_ATOM_K / MMA_K; mma_iter++) {
                    const int a_chunk_off = bk_chunk * BM * SWIZZLE_ATOM_K;
                    const int b_chunk_off = bk_chunk * (BN / CTA_GROUP_SIZE) * SWIZZLE_ATOM_K;
                    const int a_k_off = a_chunk_off + mma_iter * MMA_K;
                    const int b_k_off = b_chunk_off + mma_iter * MMA_K;

                    uint32_t smem_buff_a = smem + mma_smem_buf * BUFF_SIZE;
                    uint32_t smem_buff_b = smem_buff_a + SMEM_A_SIZE;

                    uint64_t smem_a_desc = make_smem_desc(smem_buff_a + a_k_off);
                    uint64_t smem_b_desc = make_smem_desc(smem_buff_b + b_k_off);

                    
                    // encode tcgen05.mma instruction descriptor.
                    // SFA_ID is odd. for .block_32 mma scaling, it basically selects what *was* as 128x1 sf column 
                    // before the transform to ((32,4),4) blocked layout. therefore, this 128x1 sf column corresponds exactly to a 128x32 chunk of A tile,
                    // which matches our MMA_M=128, MMA_K=32.
                    uint32_t idesc = 0;
                    tcgen05_encode_idesc<CTA_GROUP_SIZE * MMA_M, MMA_N>(idesc, mma_iter);

                    // only disable accumo on the first mma of each block
                    int enable_accum = (block_k_idx == 0 && bk_chunk == 0 && mma_iter == 0) ? 0 : 1;

                    // i *think* that the `tmem_base_addr` of SFA and SFB should stay constant 
                    // while we are using the same 512b scale factor tile for four MMAs, 
                    // which use one 128x1 (or four 32x1) sf cols.
                    // we use same tmem base address and increment `sfa_id` to communicate 
                    // which sf cols to select for that MMA. 
                    tcgen05_mma_mxfp8(smem_a_desc, smem_b_desc, tmem_sfa_base_addr, tmem_sfb_base_addr, tmem_accum_addr, idesc, enable_accum);
                }
            }

            // cta 0 signals to both ctas the smem buffer can now be safely re-used
            tcgen05_commit_multicast(smem_empty_mbar_addrs[mma_smem_buf], cta_mask);
            mma_smem_buf = (mma_smem_buf + 1) % QUEUE_SIZE;
        }

        // cta0 signals to both ctas the tmem accum buff is ready for epilogue
        tcgen05_commit_multicast(mma_mbar_addrs[mma_tmem_buf], cta_mask);
        mma_tmem_buf = (mma_tmem_buf + 1) % TMEM_BUFFERS;
        blocks_processed += 1;
    }
}

template<int BM, int BN>
__device__ __noinline__
void epilogue_warpgroup(
    float* C,
    int N,
    int start_bid,
    int start_group_id,
    int num_groups,
    int total_groups,
    uint32_t cta_rank,
    int grid_m,
    int grid_n,
    uint32_t threadIdx_x,
    uint32_t mma_mbar_addr,
    uint32_t epilogue_mbar_addr,
    uint32_t tmem_addr_smem
) {
    constexpr int CTA_GROUP_SIZE = 2;
    constexpr int TMEM_BUFFERS = 2;
    constexpr int COLS_PER_THREAD = 4;

    int epilogue_tmem_buf = 0;
    int mma_parity[TMEM_BUFFERS] = {0};
    int tmem_base_addr = *reinterpret_cast<int*>(__cvta_shared_to_generic(tmem_addr_smem));

    // pre-compute mapped barrier addresses
    uint32_t epilogue_mbar_addrs[TMEM_BUFFERS];
    uint32_t mma_mbar_addrs[TMEM_BUFFERS];
    #pragma unroll
    for (int i = 0; i < TMEM_BUFFERS; i++) {
        epilogue_mbar_addrs[i] = epilogue_mbar_addr + i * sizeof(uint64_t);
        mma_mbar_addrs[i] = mma_mbar_addr + i * sizeof(uint64_t);
        if (cta_rank == 1) {
            epilogue_mbar_addrs[i] = map_smem_addr_to_cta_rank(epilogue_mbar_addrs[i], 0);
        }
    }

    int ep_warp_id = (threadIdx_x / 32);
    int lane_id = threadIdx_x % 32;

    for (int group_id = start_group_id; group_id < total_groups; group_id += num_groups) {
        const int bid = group_id * CTA_GROUP_SIZE + (start_bid % CTA_GROUP_SIZE);
        auto [block_m, block_n] = compute_bid_hilbert(bid, grid_m, grid_n);

        mbarrier_wait_parity(mma_mbar_addrs[epilogue_tmem_buf], mma_parity[epilogue_tmem_buf]);
        mma_parity[epilogue_tmem_buf] ^= 1;

        // fence that ensures all previously committed
        // tcgen05.mma operations are complete with results visible in tmem to this thread
        asm volatile("tcgen05.fence::after_thread_sync;");

        int tmem_addr_reg = tmem_base_addr + epilogue_tmem_buf * BN;

        #pragma unroll
        for (int i = 0; i < BN/COLS_PER_THREAD; i++) {
            float c_reg[COLS_PER_THREAD];
            const int tmem_base_row = ep_warp_id * 32;
            const int tmem_base_col = i * COLS_PER_THREAD;
            tcgen05_ld_tmem_to_reg(tmem_addr_reg, tmem_base_row, tmem_base_col, c_reg);

            const int c_row = block_m * BM + ep_warp_id * 32 + lane_id;
            const int c_col = block_n * BN + i * COLS_PER_THREAD;

            *reinterpret_cast<float4*>(C + c_row * N + c_col) = *reinterpret_cast<float4*>(&c_reg);
        }

        mbarrier_arrive(epilogue_mbar_addrs[epilogue_tmem_buf]);
        epilogue_tmem_buf = (epilogue_tmem_buf + 1) % TMEM_BUFFERS;
    }
}

template<
    int QUEUE_SIZE,
    int BM = 128,
    int BN = 128,
    int BK = 64,
    int MMA_M = 128,
    int MMA_N = 128,
    int MMA_K = 32
>
__global__
__cluster_dims__(2, 1, 1)  // threadblock cluster for 2 cta mma
void ws_gemm_2cta_mma(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap sfa_map,
    const __grid_constant__ CUtensorMap sfb_map,
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int CTA_GROUP_SIZE = 2;
    constexpr int TMEM_BUFFERS = 2;
    constexpr int SF_BK = BK / 32;
    constexpr int SF_BYTES = 512; // ((32,4),4) tile of e8m0 is 512 bytes

    // get start block and group id for grid strided loop
    const int start_bid = blockIdx.x;
    const int blocks_per_grid = gridDim.x;
    const int grid_n = N / BN;
    const int grid_m = M / (CTA_GROUP_SIZE * BM);  // number of groups in M dimension

    // epilogue warps = (0,1,2,3), producer = 4, consumer = 5
    const int is_producer_master_thread = threadIdx.x == 4 * 32;             
    const int is_mma_master_thread = threadIdx.x ==  5 * 32;
    const int warp_id = threadIdx.x / 32;
    const int warpgroup_id = threadIdx.x / 128;

    // init smem buffer, 1024 alignment for 128b swizzle
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // convert to shared addr now for easier offset addition with integers, etc
    uint32_t smem = __cvta_generic_to_shared(smem_buffer);
   
    // calculate smem allocation sizes
    constexpr int SMEM_A_SIZE = BM * BK;                    // for 2cta mma, each cta loads BM rows
    constexpr int SMEM_B_SIZE = (BN / CTA_GROUP_SIZE) * BK; // for 2cta mma, each cta loads BN/2 cols
    // Scale factor tiles are ((32,4),4) layout with 512-byte granularity (128 rows/cols × 4 SF_BK)
    constexpr int SMEM_SFA_SIZE = BM * SF_BK;               // for SFA each CTA gets BM rows (so 2 BMxSF_BK tiles stacked vertically)
    constexpr int SMEM_SFB_SIZE = BN * SF_BK;               // for SFB, full BN cols must be duplicated on both CTAs
    constexpr int SMEM_QUEUE_SIZE = QUEUE_SIZE * (SMEM_A_SIZE + SMEM_B_SIZE + SMEM_SFA_SIZE + SMEM_SFB_SIZE);

    // calculate start offset for each smem buffer
    constexpr int SMEM_FULL_MBAR_OFFSET = SMEM_QUEUE_SIZE;
    constexpr int SMEM_EMPTY_MBAR_OFFSET = SMEM_FULL_MBAR_OFFSET + QUEUE_SIZE * sizeof(uint64_t);
    constexpr int SMEM_MMA_MBAR_OFFSET = SMEM_EMPTY_MBAR_OFFSET + QUEUE_SIZE * sizeof(uint64_t);
    constexpr int SMEM_EPILOGUE_MBAR_OFFSET = SMEM_MMA_MBAR_OFFSET + TMEM_BUFFERS * sizeof(uint64_t);
    constexpr int TMEM_ADDR_OFFSET = SMEM_EPILOGUE_MBAR_OFFSET + TMEM_BUFFERS * sizeof(uint64_t);

    // get shared addrs for mbars - used for mapping to peer CTA for DSMEM access
    uint32_t smem_full_mbar_addr = smem + SMEM_FULL_MBAR_OFFSET;
    uint32_t smem_empty_mbar_addr = smem + SMEM_EMPTY_MBAR_OFFSET;
    uint32_t mma_mbar_addr = smem + SMEM_MMA_MBAR_OFFSET;
    uint32_t epilogue_mbar_addr = smem + SMEM_EPILOGUE_MBAR_OFFSET;
    uint32_t tmem_addr_smem = smem + TMEM_ADDR_OFFSET;

    // initialize mbarriers
    uint64_t* smem_full_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_FULL_MBAR_OFFSET);
    uint64_t* smem_empty_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_EMPTY_MBAR_OFFSET);
    uint64_t* mma_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_MMA_MBAR_OFFSET);
    uint64_t* epilogue_mbar = reinterpret_cast<uint64_t*>(smem_buffer + SMEM_EPILOGUE_MBAR_OFFSET);

    // both CTAs report to CTA 0 mbar
    initialize_barriers<QUEUE_SIZE, 2>(smem_full_mbar, is_producer_master_thread);                     

    // cta 0 signals both ctas when smem buffer can be re-used
    initialize_barriers<QUEUE_SIZE, 1>(smem_empty_mbar, is_producer_master_thread);                     

    // cta 0 signals both ctas when tmem buffer is ready for epilogue
    initialize_barriers<TMEM_BUFFERS, 1>(mma_mbar, is_producer_master_thread);                          

    // epilogue warpgroup signals mma warp when tmem->reg read is done
    initialize_barriers<TMEM_BUFFERS, 128 * CTA_GROUP_SIZE>(epilogue_mbar, is_producer_master_thread);  

    // ensure mbarrier initialization is visible to both ctas in the cluster.
    // mbar smem state init before "release" will be visible to other threads after they "acquire" it
    asm volatile("fence.mbarrier_init.release.cluster;");  

    // get threadblock rank in cluster
    // see: https://github.com/NVIDIA/cutlass/blob/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/include/cute/arch/cluster_sm90.hpp#L158
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %cluster_ctaid.x;" : "=r"(cta_rank));

    // parity for coordination between tma, mma, epilogue warps
    int smem_full_parity[QUEUE_SIZE] = {0};
    int smem_empty_parity[QUEUE_SIZE] = {0};
    int mma_parity[TMEM_BUFFERS] = {0};
    int epilogue_parity[TMEM_BUFFERS] = {0};

    // alloc tmem addr for accumulator. allocates BN columns of TMEM (must always alloc full 128 rows)
    // 1 warp must do the allocation
    // see: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions
    // Per-buffer TMEM layout: [BN cols accum | SFA_COLS | SFB_COLS]
    // Each ((32,4),4) tile occupies 16 TMEM columns
    constexpr int SFA_TILES = (BM * SF_BK) / 512;
    constexpr int SFB_TILES = (BN * SF_BK) / 512; 
    constexpr int SFA_COLS = SFA_TILES * 16;  // ((32,4),4) layout -> 16 cols per sf til
    constexpr int SFB_COLS = SFB_TILES * 16; 
    constexpr int TMEM_WIDTH_PER_BUFFER = BN + SFA_COLS + SFB_COLS;
    constexpr int TMEM_TOTAL_WIDTH = TMEM_BUFFERS * TMEM_WIDTH_PER_BUFFER;

    // round up to nearest power of 2
    constexpr int TMEM_WIDTH_ROUNDED = 1 << (32 - __builtin_clz(TMEM_TOTAL_WIDTH - 1));
    if (warp_id == 0)
    {
        // allocate tmem buffers (must be power of 2)
        tcgen05_alloc<TMEM_WIDTH_ROUNDED>(tmem_addr_smem);
    }

    // make sure mbarriers and tmem addr are visible to full cluster
    cluster_sync();

    constexpr int PRODUCER_WARP_ID = 4, CONSUMER_WARP_ID = 5;
    constexpr int EPILOGUE_WARPGROUP_ID = 0;

    const int start_group_id = start_bid / CTA_GROUP_SIZE;
    const int num_groups = (blocks_per_grid + CTA_GROUP_SIZE - 1) / CTA_GROUP_SIZE;
    const int total_blocks = (M / BM) * (N / BN);
    const int total_groups = total_blocks / CTA_GROUP_SIZE;

    if (warp_id == PRODUCER_WARP_ID && is_producer_master_thread)
    {
        producer_warp<QUEUE_SIZE, BM, BN, BK>(
            &a_map, &b_map, &sfa_map, &sfb_map, K, start_bid, start_group_id, num_groups, total_groups,
            cta_rank, grid_m, grid_n, smem, smem_full_mbar_addr, smem_empty_mbar_addr
        );
    }
    else if (cta_rank == 0 && warp_id == CONSUMER_WARP_ID && is_mma_master_thread)
    {
        consumer_warp<QUEUE_SIZE, BM, BN, BK, MMA_M, MMA_N, MMA_K>(
            K, start_group_id, num_groups, total_groups, smem,
            smem_full_mbar_addr, smem_empty_mbar_addr, mma_mbar_addr,
            epilogue_mbar_addr, tmem_addr_smem
        );
    }
    else if (warpgroup_id == EPILOGUE_WARPGROUP_ID)
    {
        epilogue_warpgroup<BM, BN>(
            C, N, start_bid, start_group_id, num_groups, total_groups,
            cta_rank, grid_m, grid_n, threadIdx.x, mma_mbar_addr,
            epilogue_mbar_addr, tmem_addr_smem
        );
    }

    // tmem location after all threads finished using tmem.
    // cluster sync to make sure both CTAs stay alive for the duration of the peristent pipeline
    cluster_sync();
    if (warp_id == 0)
    {
        int tmem_addr_reg = *reinterpret_cast<int*>(__cvta_shared_to_generic(tmem_addr_smem));
        tcgen05_dealloc<TMEM_WIDTH_ROUNDED>(tmem_addr_reg);
    }
}

extern "C" void launch_gemm(void* A, void* B, void* SFA, void* SFB, void* C, int M, int N, int K) {
    const uint8_t* sfa_ptr = reinterpret_cast<const uint8_t*>(SFA);
    const uint8_t* sfb_ptr = reinterpret_cast<const uint8_t*>(SFB);

    // dims for smem tiles
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 128;
    constexpr int SF_BK = BK / 32;

    // dims for tcgen05.mma
    constexpr int MMA_M = 128;
    constexpr int MMA_N = 128;
    constexpr int MMA_K = 32;

    assert(BM == MMA_M);
    assert(BN == MMA_N);
    assert(BK >= MMA_K && BK % MMA_K == 0);

    // for 2cta mma, assert the problem size evenly divides into cluster size
    assert(M % (2 * MMA_M) == 0);
    assert(N % MMA_N == 0);

    alignas(128) CUtensorMap a_map = {};
    alignas(128) CUtensorMap b_map = {};

    constexpr int CTA_GROUP_SIZE = 2;

    // BM, BK
    // BM, BK/128, 128 -> (128 elems = 128 bytes of fp8)
    // BK/128, BM, 128 -> BK/128 instances of BM,128 strips
    constexpr int SWIZZLE_ATOM_K = 128;
    uint64_t a_global_dims[3] = {SWIZZLE_ATOM_K, (uint64_t)M, (uint64_t)(K / SWIZZLE_ATOM_K)};
    uint32_t a_smem_dims[3] = {SWIZZLE_ATOM_K, BM, BK / SWIZZLE_ATOM_K}; // for 2 CTA mma, each CTA still loads the full BM rows of A into smem
    uint32_t a_strides[2] = {(uint32_t)(K), SWIZZLE_ATOM_K};
    create_3d_tensor_map(
        A,
        a_map,
        a_global_dims,
        a_smem_dims,
        a_strides,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
    );

    // BK, BN
    // 128, BK/128, BN
    // 128, BN, BK/128-> BK/128instances of BN,128 strips
    uint64_t b_global_dims[3] = {SWIZZLE_ATOM_K, (uint64_t)N, (uint64_t)(K / SWIZZLE_ATOM_K)};
    uint32_t b_smem_dims[3] = {SWIZZLE_ATOM_K, BN/CTA_GROUP_SIZE, BK / SWIZZLE_ATOM_K}; // for 2 CTA MMA, each CTA loads BN/2 cols of B into smem
    uint32_t b_strides[2] = {(uint32_t)(K), SWIZZLE_ATOM_K};
    create_3d_tensor_map(
        B,
        b_map,
        b_global_dims,
        b_smem_dims,
        b_strides,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
    );

    alignas(128) CUtensorMap sfa_map = {};
    alignas(128) CUtensorMap sfb_map = {};

    // sfa has shape (M, K/32) with layout ((32,4),4) for 512 byte tile granularity)
    // Each 512-byte tile = 4 blocks × 128 bytes/block
    // Use 4D: {128 bytes per block, 4 blocks per tile, SF_K/4 tile cols, M/128 tile rows}
    const int SF_K = K / 32;
    uint64_t sfa_global_dims[4] = {
        128,                       // bytes per block (within 256 limit)
        4,                         // blocks per tile
        (uint64_t)(K / 32/ 4),     // tile columns
        (uint64_t)(M / 128)        // tile rows
    };
    uint32_t sfa_smem_dims[4] = {
        128,                       // load full block
        4,                         // load all 4 blocks (full tile)
        (uint32_t)(SF_BK / 4),     // tile columns to load
        (uint32_t)(BM / 128)       // tile rows to load
    };
    uint32_t sfa_strides[3] = {
        128,                                    // stride to next block
        512,                                    // stride to next tile column
        (uint32_t)((SF_K / 4) * 512)           // stride to next tile row
    };

    // Use create_4d_tensor_map (need to add this function)
    create_4d_tensor_map(
        SFA,
        sfa_map,
        sfa_global_dims,
        sfa_smem_dims,
        sfa_strides,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
    );

    // sfb has shape (N, K/32) with layout ((32,4),4) for 512 byte tile granularity)
    // Each 512-byte tile = 4 blocks × 128 bytes/block
    // Use 4D: {128 bytes per block, 4 blocks per tile, SF_K/4 tile cols, N/128 tile rows}
    uint64_t sfb_global_dims[4] = {
        128,                       // bytes per block
        4,                         // blocks per tile
        (uint64_t)(SF_K / 4),      // tile columns
        (uint64_t)(N / 128)        // tile rows
    };
    uint32_t sfb_smem_dims[4] = {
        128,                       // load full block
        4,                         // load all 4 blocks
        (uint32_t)(SF_BK / 4),     // tile columns to load
        (uint32_t)(BN / 128)       // tile rows to load
    };
    uint32_t sfb_strides[3] = {
        128,                                    // stride to next block
        512,                                    // stride to next tile column
        (uint32_t)((SF_K / 4) * 512)           // stride to next tile row
    };
    create_4d_tensor_map(
        SFB,
        sfb_map,
        sfb_global_dims,
        sfb_smem_dims,
        sfb_strides,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE
    );

    float* c_ptr = reinterpret_cast<float*>(C);

    constexpr int PRODUCER_WARPS = 1;
    constexpr int CONSUMER_WARPS = 1;
    constexpr int EPILOGUE_WARPS = 4;
    constexpr int BLOCK_SIZE = (PRODUCER_WARPS + CONSUMER_WARPS + EPILOGUE_WARPS) * 32;
    constexpr int TMEM_BUFFERS = 2;

    // validate TMEM allocation fits in hardware limits
    constexpr int tmem_width_cells = 512;
    constexpr int SFA_TILES = (BM * BK / 32) / 512;
    constexpr int SFB_TILES = ((BN / CTA_GROUP_SIZE) * BK / 32) / 512;
    constexpr int SFA_COLS = SFA_TILES * 16;  // 16 TMEM cols per ((32,4),4) tile
    constexpr int SFB_COLS = SFB_TILES * 16;
    constexpr int TMEM_WIDTH_PER_BUFFER = BN + SFA_COLS + SFB_COLS;
    constexpr int TMEM_TOTAL_WIDTH = TMEM_BUFFERS * TMEM_WIDTH_PER_BUFFER;
    // TMEM allocation must be power of 2
    constexpr int TMEM_WIDTH_ROUNDED = 1 << (32 - __builtin_clz(TMEM_TOTAL_WIDTH - 1));
    assert(TMEM_WIDTH_ROUNDED <= tmem_width_cells);

    
    // persistent kernel runs 1 threadblock per SM
    int NUM_SMS;
    cudaDeviceGetAttribute(&NUM_SMS, cudaDevAttrMultiProcessorCount, 0);

    // use max smem per SM on sm100 to determine queue size
    constexpr int max_smem_per_sm = 227 * 1024;

    // increase max smem beyond 48k default limit.
    // smem usage:
    // - smem_a_tile[QUEUE_SIZE][BM * BK]
    // - smem_b_tile[QUEUE_SIZE][BN/2 * BK]
    // - smem_full_mbar[QUEUE_SIZE]
    // - smem_empty_mbar[QUEUE_SIZE]
    // - mma_mbar 
    // - epilogue mbar
    // - tma_addr_in_smem
    constexpr int smem_a_size = BM * BK;
    constexpr int smem_b_size = (BN / CTA_GROUP_SIZE) * BK;
    constexpr int smem_sfa_size = BM * SF_BK;  // each CTA has BMxSF_K *separate* sfa tiles (ctas are 'stacked vertically')
    constexpr int smem_sfb_size = BN * SF_BK;  // each CTA has *same* BNxSF_K sfb tile
    constexpr int smem_full_mbar_size = sizeof(uint64_t);
    constexpr int smem_empty_mbar_size = sizeof(uint64_t);
    constexpr int total_smem_per_stage = smem_a_size + smem_sfa_size + smem_b_size + smem_sfb_size + smem_full_mbar_size + smem_empty_mbar_size;
    constexpr int mma_mbar_size = TMEM_BUFFERS * sizeof(uint64_t);
    constexpr int epilogue_mbar_size = TMEM_BUFFERS * sizeof(uint64_t);
    constexpr int tmem_addr_size = sizeof(int);
    constexpr int QUEUE_SIZE = max_smem_per_sm / (total_smem_per_stage + tmem_addr_size + mma_mbar_size + epilogue_mbar_size);
    constexpr int total_smem = total_smem_per_stage * QUEUE_SIZE;

    const int launch_blocks = 128; 
    dim3 grid_dim(launch_blocks);
    dim3 block_dim(BLOCK_SIZE);
    auto kernel = ws_gemm_2cta_mma<QUEUE_SIZE, BM, BN, BK, MMA_M, MMA_N, MMA_K>;

    CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        total_smem
    ));
    kernel<<<grid_dim, block_dim, total_smem>>>(a_map, b_map, sfa_map, sfb_map, c_ptr, M, N, K);
}
