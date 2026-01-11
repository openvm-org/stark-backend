#pragma once

#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include <cstdint>
#include <device_atomic_functions.h>

namespace sumcheck {

template <typename Field>
__device__ __forceinline__ Field
mle_interpolate_single(const Field *__restrict__ column, Fp x, uint32_t y_int) {
    auto t0 = column[y_int << 1];
    auto t1 = column[(y_int << 1) | 1];
    return t0 + (t1 - t0) * x;
}

// Warp-level reduction: sums FpExt values across threads in a warp (32 threads)
// Uses shuffle instructions for efficient communication within a warp
static __device__ inline FpExt warp_reduce_sum(FpExt val) {
    unsigned mask = __activemask();

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        FpExt other;
        other.elems[0] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[0].asRaw(), offset));
        other.elems[1] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[1].asRaw(), offset));
        other.elems[2] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[2].asRaw(), offset));
        other.elems[3] = Fp::fromRaw(__shfl_down_sync(mask, val.elems[3].asRaw(), offset));
        val = val + other;
    }
    return val;
}

// Reduce FpExt across n threads using warp shuffles (butterfly pattern)
// n must be a power of 2, 1 <= n <= 32
// All participating threads must call this function
__device__ __forceinline__ FpExt warp_reduce_sum_n(FpExt val, uint32_t n) {
    for (uint32_t offset = n >> 1; offset > 0; offset >>= 1) {
        FpExt other;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            other.elems[i] =
                Fp::fromRaw(__shfl_xor_sync(0xffffffff, val.elems[i].asRaw(), offset));
        }
        val = val + other;
    }
    return val;
}

// Reduce FpExt across chunk_size threads within a block
// chunk_size must be a power of 2
// tid_in_chunk: thread's index within its chunk [0, chunk_size)
// chunk_in_block: which chunk this thread belongs to
// smem: shared memory for cross-warp reduction (size >= total_warps_in_block)
//
// Note: When chunk_size > 32, writes to chunk_smem[warp_in_chunk] from different chunks
// may hit the same bank (FpExt = 16 bytes = 4 words). This is a minor inefficiency in the
// reduction phase, not a correctness issue.
__device__ inline FpExt chunk_reduce_sum(
    FpExt val,
    FpExt *smem,
    uint32_t tid_in_chunk,
    uint32_t chunk_size,
    uint32_t chunk_in_block
) {
    if (chunk_size <= 32) {
        return warp_reduce_sum_n(val, chunk_size);
    }

    // Cross-warp case (chunk_size > 32)
    uint32_t warps_per_chunk = chunk_size >> 5;
    uint32_t warp_in_chunk = tid_in_chunk >> 5;
    uint32_t lane_id = tid_in_chunk & 31;

    // Step 1: Warp-level reduction
    val = warp_reduce_sum(val);

    // Step 2: Store warp result to shared memory
    FpExt *chunk_smem = smem + chunk_in_block * warps_per_chunk;
    if (lane_id == 0) {
        chunk_smem[warp_in_chunk] = val;
    }
    __syncthreads();

    // Step 3: First warp of chunk reads and reduces warp results
    if (warp_in_chunk == 0) {
        FpExt zero = {0, 0, 0, 0};
        FpExt warp_val = (lane_id < warps_per_chunk) ? chunk_smem[lane_id] : zero;
        val = warp_reduce_sum(warp_val);
    }

    return val;
}

// ============================================================================
// ATOMIC U64 ACCUMULATION (delayed modular reduction to CPU)
// ============================================================================
//
// BabyBear elements in Montgomery form are < 2^31. Summing N elements in uint64_t
// overflows when N * 2^31 > 2^64, i.e., N > 2^33. Safe for typical GPU workloads.
// Final modular reduction is done on CPU after kernel completion.

// Atomically add Fp raw value to uint64_t accumulator
__device__ __forceinline__ void atomic_add_fp_to_u64(uint64_t *output, Fp val) {
    atomicAdd((unsigned long long *)output, static_cast<uint64_t>(val.asRaw()));
}

// Atomically add FpExt to uint64_t[4] accumulator
// Layout: output[0..3] correspond to elems[0..3]
__device__ __forceinline__ void atomic_add_fpext_to_u64(uint64_t *output, FpExt val) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        atomic_add_fp_to_u64(output + i, val.elems[i]);
    }
}

// ============================================================================
// WARP-AGGREGATED ATOMIC REDUCTION
// ============================================================================
//
// Modern best practice for block-wide sums on Volta+ architectures:
// 1. Warp-level reduction via __shfl_down_sync (zero memory latency)
// 2. Lane 0 of each warp does one atomicAdd to global memory
// 3. No __syncthreads() or shared memory required
//
// Trade-off: More atomics than full block reduction (num_warps vs 1), but
// eliminates barrier latency and shared memory bank conflicts. Modern L2
// caches handle concurrent atomics efficiently.

// Warp-aggregated atomic: reduce within warp, lane 0 does single atomic add
__device__ __forceinline__ void warp_aggregated_atomic_add_fpext(
    uint64_t *output, // Global memory, uint64_t[4] for one FpExt
    FpExt val
) {
    // Step 1: Warp-level reduction via shuffles
    val = warp_reduce_sum(val);

    // Step 2: Lane 0 does the atomic add
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        atomic_add_fpext_to_u64(output, val);
    }
}

// Indexed version for arrays of FpExt outputs
// Layout: output[idx * 4 .. idx * 4 + 3]
__device__ __forceinline__ void warp_aggregated_atomic_add_fpext_indexed(
    uint64_t *output, // Global memory, uint64_t[d * 4]
    uint32_t idx,
    FpExt val
) {
    warp_aggregated_atomic_add_fpext(output + idx * 4, val);
}

// Block-level reduction: sums FpExt values across all threads in a block
// Two-stage process: warp-level reduction, then inter-warp reduction
// Returns result in all threads of first warp (typically only thread 0 uses it)
static __device__ inline FpExt block_reduce_sum(FpExt val, FpExt *shared) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (blockDim.x + 31) / 32; // Round up for blocks < 32 threads

    // Stage 1: Each warp reduces its values
    val = warp_reduce_sum(val);

    // Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Stage 2: First warp reduces the per-warp results
    FpExt zero = {0, 0, 0, 0};
    if (warp_id == 0) {
        // Only the first warp participates in the second reduction. Within that warp we
        // reuse the lane id to index the number of warps stored in shared memory.
        FpExt warp_val = (lane_id < num_warps) ? shared[lane_id] : zero;
        val = warp_reduce_sum(warp_val);
    }

    return val;
}

// Final reduction: combines partial block sums into final result
// Grid dimension: D blocks (one per output value)
// Each block reduces num_blocks partial sums for its assigned output
//
// The number of blocks is constant and determined by template parameter D.
template <int D>
static __global__ void static_final_reduce_block_sums(
    const FpExt *block_sums, // [num_blocks][D]
    FpExt *output,           // [D]
    uint32_t num_blocks
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    int tid = threadIdx.x;

    // blockIdx.x selects which of the D outputs we're computing
    int out_idx = blockIdx.x;
    if (out_idx >= D)
        return;

    FpExt sum = {0, 0, 0, 0};

    // Each thread accumulates subset of blocks
    for (int block_id = tid; block_id < num_blocks; block_id += blockDim.x) {
        sum += block_sums[block_id * D + out_idx];
    }

    // Block-level reduction
    sum = block_reduce_sum(sum, shared);

    if (tid == 0) {
        output[out_idx] = sum;
    }
}

// gridDim.x must be set to equal `d`
static __global__ void final_reduce_block_sums(
    const FpExt *block_sums, // [num_blocks][d]
    FpExt *output,           // [d]
    uint32_t num_blocks
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    int tid = threadIdx.x;

    // blockIdx.x selects which of the D outputs we're computing
    int out_idx = blockIdx.x;
    int d = gridDim.x;

    FpExt sum = {0, 0, 0, 0};

    // Each thread accumulates subset of blocks
    for (int block_id = tid; block_id < num_blocks; block_id += blockDim.x) {
        sum += block_sums[block_id * d + out_idx];
    }

    // Block-level reduction
    sum = block_reduce_sum(sum, shared);

    if (tid == 0) {
        output[out_idx] = sum;
    }
}

// Batched version: reduces partial sums for multiple segments (e.g., AIRs) in a single launch.
// Grid: (num_segments, d) where each block (seg, x) reduces blocks for segment `seg` at output index `x`.
// segment_offsets[seg] gives the start block index; segment_offsets[seg+1] gives the end.
static __global__ void batched_final_reduce_block_sums(
    const FpExt *block_sums,           // [total_blocks][d]
    FpExt *output,                     // [num_segments][d]
    const uint32_t *segment_offsets,   // [num_segments + 1], device memory
    uint32_t d
) {
    extern __shared__ char smem[];
    FpExt *shared = (FpExt *)smem;

    uint32_t seg_idx = blockIdx.x;
    uint32_t out_idx = blockIdx.y;
    int tid = threadIdx.x;

    uint32_t start = segment_offsets[seg_idx];
    uint32_t end = segment_offsets[seg_idx + 1];
    uint32_t num_blocks = end - start;

    FpExt sum = {0, 0, 0, 0};

    // Each thread accumulates a strided subset of this segment's blocks
    for (uint32_t i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_sums[(start + i) * d + out_idx];
    }

    // Block-level reduction
    sum = block_reduce_sum(sum, shared);

    if (tid == 0) {
        output[seg_idx * d + out_idx] = sum;
    }
}

// Folds MLE evaluations using challenge r: output[y] = input[2*y] + r*(input[2*y+1] - input[2*y])
__device__ __forceinline__ void fold_mle(
    const FpExt *__restrict__ const *__restrict__ input_matrices,
    FpExt *__restrict__ const *__restrict__ output_matrices,
    const uint32_t *widths, // Width of each matrix
    const uint8_t log_output_height,
    const FpExt &r_val,
    uint32_t tidx,
    uint32_t mat_idx
) {
    uint32_t width = widths[mat_idx];
    uint32_t output_height = 1 << log_output_height;
    if (tidx >= output_height * width)
        return;
    uint32_t row_idx = tidx & (output_height - 1);
    uint32_t col_idx = tidx >> log_output_height;

    const FpExt *input = input_matrices[mat_idx];
    FpExt *output = output_matrices[mat_idx];

    auto col_offset_out = col_idx * output_height;
    auto col_offset_in = col_offset_out << 1;

    auto idx_0 = col_offset_in + (row_idx << 1);
    auto idx_1 = col_offset_in + (row_idx << 1) + 1;
    auto out_idx = col_offset_out + row_idx;

    FpExt t0 = input[idx_0];
    FpExt t1 = input[idx_1];

    output[out_idx] = t0 + r_val * (t1 - t0);
}

} // namespace sumcheck
