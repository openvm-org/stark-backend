#pragma once

#include "fp.h"
#include "fpext.h"
#include <cstdint>

namespace sumcheck {
// Warp-level reduction: sums FpExt values across threads in a warp (32 threads)
// Uses shuffle instructions for efficient communication within a warp
static __device__ inline FpExt warp_reduce_sum(FpExt val) {
    unsigned mask = __activemask();

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

} // namespace sumcheck
