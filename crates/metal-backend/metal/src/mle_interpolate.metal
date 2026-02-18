/// MLE interpolation kernels for Metal.
/// Translated from cuda-backend/cuda/src/mle_interpolate.cu.

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "baby_bear_ext.h"

// ============================================================================
// Basic MLE Interpolation Stage (1D)
// ============================================================================

/// Single stage of MLE interpolation on Fp buffer.
/// eval_to_coeff=true:  buffer[second] -= buffer[base]
/// eval_to_coeff=false: buffer[second] += buffer[base]
kernel void mle_interpolate_stage(
    device Fp *buffer [[buffer(0)]],
    constant uint32_t &total_pairs [[buffer(1)]],
    constant uint32_t &step [[buffer(2)]],
    constant uint32_t &is_eval_to_coeff [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_pairs) return;

    uint32_t span = step << 1;
    uint32_t chunk = tid / step;
    uint32_t offset = tid % step;
    uint32_t base = chunk * span + offset;
    uint32_t second = base + step;

    if (is_eval_to_coeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

/// Single stage of MLE interpolation on FpExt buffer.
kernel void mle_interpolate_stage_ext(
    device FpExt *buffer [[buffer(0)]],
    constant uint32_t &total_pairs [[buffer(1)]],
    constant uint32_t &step [[buffer(2)]],
    constant uint32_t &is_eval_to_coeff [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_pairs) return;

    uint32_t span = step << 1;
    uint32_t chunk = tid / step;
    uint32_t offset = tid % step;
    uint32_t base = chunk * span + offset;
    uint32_t second = base + step;

    if (is_eval_to_coeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

// ============================================================================
// 2D MLE Interpolation Stage (column-major matrix)
// ============================================================================

/// MLE interpolation stage on a column-major matrix.
/// Each column is processed independently.
kernel void mle_interpolate_stage_2d(
    device Fp *buffer [[buffer(0)]],
    constant uint32_t &padded_height [[buffer(1)]],
    constant uint32_t &span [[buffer(2)]],
    constant uint32_t &step [[buffer(3)]],
    constant uint32_t &is_eval_to_coeff [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]] // (tidx, col)
) {
    uint32_t tidx = gid.x;
    uint32_t col = gid.y;

    uint32_t chunk = tidx / step;
    uint32_t offset = tidx % step;
    uint32_t base = col * padded_height + chunk * span + offset;
    uint32_t second = base + step;

    if (is_eval_to_coeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

// ============================================================================
// Fused MLE Interpolation using SIMD Shuffle (2D)
// ============================================================================

/// Fused MLE interpolation processing multiple stages via SIMD shuffle.
/// num_stages consecutive stages starting from start_step.
kernel void mle_interpolate_fused_2d(
    device Fp *buffer [[buffer(0)]],
    constant uint32_t &padded_height [[buffer(1)]],
    constant uint32_t &log_stride [[buffer(2)]],
    constant uint32_t &start_step [[buffer(3)]],
    constant uint32_t &num_stages [[buffer(4)]],
    constant uint32_t &is_eval_to_coeff [[buffer(5)]],
    constant uint32_t &right_pad [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]] // (tidx, col)
) {
    uint32_t tidx = gid.x;
    uint32_t col = gid.y;

    uint32_t meaningful_count = padded_height >> log_stride;
    if (tidx >= meaningful_count) return;

    uint32_t physical_idx = right_pad ? (tidx << log_stride) : tidx;
    uint32_t base_idx = col * padded_height + physical_idx;
    Fp val = buffer[base_idx];

    // Process each stage using SIMD shuffle
    for (uint32_t stage = 0; stage < num_stages; stage++) {
        uint32_t stp = start_step << stage;
        Fp other = Fp::fromRaw(simd_shuffle_xor(val.asRaw(), stp));
        bool is_low = (tidx & stp) == 0;

        if (is_eval_to_coeff) {
            if (!is_low) val = val - other;
        } else {
            if (!is_low) val = val + other;
        }
    }

    buffer[base_idx] = val;
}

// ============================================================================
// Shared Memory MLE Interpolation (2D)
// ============================================================================

/// Padding index to avoid threadgroup memory bank conflicts.
/// Adds 1 per 32 elements (Metal has 32 threadgroup memory banks).
inline uint32_t padded_idx(uint32_t i) { return i + (i >> 5); }

/// MLE interpolation using threadgroup memory for stages within a tile.
/// Processes stages where step < tile_size.
kernel void mle_interpolate_shared_2d(
    device Fp *buffer [[buffer(0)]],
    constant uint32_t &padded_height [[buffer(1)]],
    constant uint32_t &log_stride [[buffer(2)]],
    constant uint32_t &start_log_step [[buffer(3)]],
    constant uint32_t &end_log_step [[buffer(4)]],
    constant uint32_t &is_eval_to_coeff [[buffer(5)]],
    constant uint32_t &right_pad [[buffer(6)]],
    constant uint32_t &tile_log_size [[buffer(7)]],
    threadgroup Fp *smem [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]], // (tile_idx, col)
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 block_size2 [[threads_per_threadgroup]]
) {
    uint32_t tile_size = 1u << tile_log_size;
    uint32_t tile_idx = tg_pos.x;
    uint32_t col = tg_pos.y;
    uint32_t tid = tid2.x;
    uint32_t block_size = block_size2.x;

    uint32_t meaningful_count = padded_height >> log_stride;
    uint32_t tile_start = tile_idx * tile_size;
    uint32_t col_offset = col * padded_height;

    // Load tile into threadgroup memory
    for (uint32_t i = tid; i < tile_size; i += block_size) {
        uint32_t logical_idx = tile_start + i;
        if (logical_idx < meaningful_count) {
            uint32_t physical_idx = right_pad ? (logical_idx << log_stride) : logical_idx;
            smem[padded_idx(i)] = buffer[col_offset + physical_idx];
        } else {
            smem[padded_idx(i)] = Fp(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process stages
    for (uint32_t log_step = start_log_step; log_step <= end_log_step; log_step++) {
        uint32_t stp = 1u << log_step;
        uint32_t spn = stp << 1;

        for (uint32_t i = tid; i < tile_size / 2; i += block_size) {
            uint32_t chunk = i / stp;
            uint32_t offset = i % stp;
            uint32_t base = chunk * spn + offset;
            uint32_t second = base + stp;

            uint32_t logical_second = tile_start + second;
            if (logical_second < meaningful_count) {
                if (is_eval_to_coeff) {
                    smem[padded_idx(second)] -= smem[padded_idx(base)];
                } else {
                    smem[padded_idx(second)] += smem[padded_idx(base)];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    for (uint32_t i = tid; i < tile_size; i += block_size) {
        uint32_t logical_idx = tile_start + i;
        if (logical_idx < meaningful_count) {
            uint32_t physical_idx = right_pad ? (logical_idx << log_stride) : logical_idx;
            buffer[col_offset + physical_idx] = smem[padded_idx(i)];
        }
    }
}
