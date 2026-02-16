// Sumcheck helper structures and reduction utilities for Metal
// Translated from CUDA: cuda-backend/cuda/include/sumcheck.cuh
#pragma once

#include "baby_bear.h"
#include "baby_bear_ext.h"

// Metal SIMD group size (equivalent to CUDA warp size)
#define SIMD_SIZE 32

// SIMD-level reduction: sums FpExt values across threads in a SIMD group
// Metal equivalent of CUDA warp_reduce_sum
inline FpExt simd_reduce_sum(FpExt val, uint simd_lane [[thread_index_in_simdgroup]]) {
    // Use simd_shuffle_down to reduce across the SIMD group
    for (uint offset = SIMD_SIZE / 2; offset > 0; offset /= 2) {
        FpExt other;
        other.elems[0] = Fp(as_type<uint32_t>(simd_shuffle_down(as_type<uint32_t>(val.elems[0].val), offset)));
        other.elems[1] = Fp(as_type<uint32_t>(simd_shuffle_down(as_type<uint32_t>(val.elems[1].val), offset)));
        other.elems[2] = Fp(as_type<uint32_t>(simd_shuffle_down(as_type<uint32_t>(val.elems[2].val), offset)));
        other.elems[3] = Fp(as_type<uint32_t>(simd_shuffle_down(as_type<uint32_t>(val.elems[3].val), offset)));
        val = val + other;
    }
    return val;
}

// SIMD-level reduction using butterfly pattern (xor shuffle)
// n must be a power of 2, 1 <= n <= SIMD_SIZE
inline FpExt simd_reduce_sum_n(FpExt val, uint n) {
    for (uint offset = n >> 1; offset > 0; offset >>= 1) {
        FpExt other;
        other.elems[0] = Fp(as_type<uint32_t>(simd_shuffle_xor(as_type<uint32_t>(val.elems[0].val), offset)));
        other.elems[1] = Fp(as_type<uint32_t>(simd_shuffle_xor(as_type<uint32_t>(val.elems[1].val), offset)));
        other.elems[2] = Fp(as_type<uint32_t>(simd_shuffle_xor(as_type<uint32_t>(val.elems[2].val), offset)));
        other.elems[3] = Fp(as_type<uint32_t>(simd_shuffle_xor(as_type<uint32_t>(val.elems[3].val), offset)));
        val = val + other;
    }
    return val;
}

// Chunk-level reduction: reduces FpExt across chunk_size threads
// chunk_size must be a power of 2
inline FpExt chunk_reduce_sum(
    FpExt val,
    threadgroup FpExt *smem,
    uint tid_in_chunk,
    uint chunk_size,
    uint chunk_in_block
) {
    if (chunk_size <= SIMD_SIZE) {
        return simd_reduce_sum_n(val, chunk_size);
    }

    // Cross-SIMD case (chunk_size > SIMD_SIZE)
    uint simds_per_chunk = chunk_size >> 5;
    uint simd_in_chunk = tid_in_chunk >> 5;
    uint lane_id = tid_in_chunk & 31;

    // Step 1: SIMD-level reduction
    val = simd_reduce_sum(val, lane_id);

    // Step 2: Store SIMD result to threadgroup memory
    threadgroup FpExt *chunk_smem = smem + chunk_in_block * simds_per_chunk;
    if (lane_id == 0) {
        chunk_smem[simd_in_chunk] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: First SIMD of chunk reads and reduces
    if (simd_in_chunk == 0) {
        FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
        FpExt simd_val = (lane_id < simds_per_chunk) ? chunk_smem[lane_id] : zero;
        val = simd_reduce_sum(simd_val, lane_id);
    }

    return val;
}

// Block-level reduction: sums FpExt values across all threads in a threadgroup
// Returns result in all threads of first SIMD group (typically only thread 0 uses it)
inline FpExt block_reduce_sum(
    FpExt val,
    threadgroup FpExt *shared,
    uint tid,
    uint threads_per_group
) {
    uint simd_id = tid / SIMD_SIZE;
    uint lane_id = tid % SIMD_SIZE;
    uint num_simds = (threads_per_group + SIMD_SIZE - 1) / SIMD_SIZE;

    // Stage 1: Each SIMD group reduces its values
    val = simd_reduce_sum(val, lane_id);

    // Lane 0 of each SIMD group writes result to threadgroup memory
    if (lane_id == 0) {
        shared[simd_id] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: First SIMD group reduces the per-SIMD results
    FpExt zero = FpExt{Fp(0u), Fp(0u), Fp(0u), Fp(0u)};
    if (simd_id == 0) {
        FpExt simd_val = (lane_id < num_simds) ? shared[lane_id] : zero;
        val = simd_reduce_sum(simd_val, lane_id);
    }

    return val;
}

// Atomically add Fp raw value to uint64_t accumulator (via atomic)
inline void atomic_add_fp_to_u64(device atomic_uint *output_lo, device atomic_uint *output_hi, Fp val) {
    // For Metal we use a simpler strategy: accumulate into device memory with atomics.
    // Since BabyBear values fit in 32 bits, we use atomic_fetch_add on uint32_t pairs.
    // NOTE: This is a simplified version. Full 64-bit atomic is not natively available
    // on all Metal devices. For correctness, the Rust FFI layer should handle the
    // reduction pattern differently for Metal (e.g., per-SIMD-group partial sums).
    atomic_fetch_add_explicit(output_lo, val.val, memory_order_relaxed);
}

// MLE interpolation: t0 + (t1 - t0) * x
inline FpExt mle_interpolate_single_ext(
    const device FpExt *column,
    FpExt x,
    uint32_t y_int
) {
    FpExt t0 = column[y_int << 1];
    FpExt t1 = column[(y_int << 1) | 1];
    return t0 + (t1 - t0) * x;
}
