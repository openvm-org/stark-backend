#include "fp.h"
#include "fpext.h"
#include "launcher.cuh"
#include "utils.cuh"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <driver_types.h>
#include <vector_types.h>

// ============================================================================
// KERNELS
// ============================================================================

template <typename Field, bool EvalToCoeff>
__global__ void mle_interpolate_stage_kernel(Field *buffer, size_t total_pairs, uint32_t step) {
    size_t span = size_t(step) << 1;
    size_t pair_idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    size_t chunk = pair_idx / step;
    uint32_t offset = pair_idx % step;
    size_t base = chunk * span + offset;
    size_t second = base + step;
    if constexpr (EvalToCoeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

template <typename Field, bool EvalToCoeff>
__global__ void mle_interpolate_stage_2d_kernel(
    Field *buffer,
    uint32_t padded_height,
    uint32_t span,
    uint32_t step
) {
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y;

    auto chunk = tidx / step;
    auto offset = tidx % step;
    uint32_t base = col * padded_height + chunk * span + offset;
    uint32_t second = base + step;

    if constexpr (EvalToCoeff) {
        buffer[second] -= buffer[base];
    } else {
        buffer[second] += buffer[base];
    }
}

// Fused MLE interpolation kernel that processes multiple stages using warp shuffle.
// This fuses stages with step sizes from `start_step` up to `start_step << (NumStages - 1)`.
// Each thread processes one element, and pairs exchange data via warp shuffle.
//
// For EvalToCoeff: buffer[i + step] -= buffer[i]
// For CoeffToEval: buffer[i + step] += buffer[i]
//
// Template parameters:
// - EvalToCoeff: direction of transformation
// - NumStages: number of stages to fuse (1-5 for warp shuffle only)
// - BitReversed: if true, meaningful indices are tidx << log_stride (strided access)
//                if false, meaningful indices are tidx (contiguous access)
template <bool EvalToCoeff, int NumStages, bool BitReversed>
__global__ void mle_interpolate_fused_2d_kernel(
    Fp *buffer,
    uint32_t padded_height,
    uint32_t log_stride, // meaningful_count = padded_height >> log_stride
    uint32_t start_step  // The step for the first stage (power of 2)
) {
    // Each thread handles one element
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y;

    uint32_t meaningful_count = padded_height >> log_stride;
    if (tidx >= meaningful_count)
        return;

    // Compute physical index based on mode
    uint32_t physical_idx;
    if constexpr (BitReversed) {
        physical_idx = tidx << log_stride;
    } else {
        physical_idx = tidx;
    }

    uint32_t base_idx = col * padded_height + physical_idx;
    Fp val = buffer[base_idx];

    unsigned mask = __activemask();
// Process each stage using warp shuffle
#pragma unroll
    for (int stage = 0; stage < NumStages; stage++) {
        uint32_t step = start_step << stage;

        // For step <= 16, we can use warp shuffle (__shfl_xor_sync)
        // Each thread XORs with `step` to find its pair
        Fp other = Fp::fromRaw(__shfl_xor_sync(mask, val.asRaw(), step));

        // Determine if this thread is the "low" or "high" element of the pair
        bool is_low = (tidx & step) == 0;

        if constexpr (EvalToCoeff) {
            // For eval-to-coeff: high = high - low
            // low stays the same, high becomes (high - low)
            if (!is_low) {
                val = val - other;
            }
        } else {
            // For coeff-to-eval: high = high + low
            // low stays the same, high becomes (high + low)
            if (!is_low) {
                val = val + other;
            }
        }
    }

    buffer[base_idx] = val;
}

// Shared memory MLE interpolation kernel that processes multiple stages within a tile.
// Each thread block handles a tile of elements from one column, loading into shared memory.
// Processes stages where step < tile_size.
//
// Template parameters:
// - EvalToCoeff: direction of transformation
// - TileLogSize: log2(tile_size), determines shared memory usage
// - RightPad: if true, meaningful indices are at strides of 2^log_stride
//                if false, meaningful indices are contiguous
template <bool EvalToCoeff, int TileLogSize, bool RightPad>
__global__ void mle_interpolate_shared_2d_kernel(
    Fp *buffer,
    uint32_t padded_height,
    uint32_t log_stride,     // meaningful_count = padded_height >> log_stride
    uint32_t start_log_step, // log2 of the first step to process
    uint32_t end_log_step    // log2 of the last step to process (inclusive)
) {
    constexpr uint32_t tile_size = 1 << TileLogSize;
    extern __shared__ Fp smem_mle[];

    uint32_t tile_idx = blockIdx.x;
    uint32_t col = blockIdx.y;
    uint32_t tid = threadIdx.x;

    uint32_t meaningful_count = padded_height >> log_stride;
    uint32_t tile_start = tile_idx * tile_size; // in logical (thread) space
    uint32_t col_offset = col * padded_height;

// Load tile into shared memory
#pragma unroll 4
    for (uint32_t i = tid; i < tile_size; i += blockDim.x) {
        uint32_t logical_idx = tile_start + i;
        if (logical_idx < meaningful_count) {
            uint32_t physical_idx;
            if constexpr (RightPad) {
                physical_idx = logical_idx << log_stride;
            } else {
                physical_idx = logical_idx;
            }
            smem_mle[i] = buffer[col_offset + physical_idx];
        } else {
            smem_mle[i] = Fp(0);
        }
    }
    __syncthreads();

    // Process stages from start_log_step to end_log_step
    for (uint32_t log_step = start_log_step; log_step <= end_log_step; log_step++) {
        uint32_t step = 1 << log_step;
        uint32_t span = step << 1;

// Each thread handles multiple pairs
#pragma unroll 4
        for (uint32_t i = tid; i < tile_size / 2; i += blockDim.x) {
            uint32_t chunk = i / step;
            uint32_t offset = i % step;
            uint32_t base = chunk * span + offset;
            uint32_t second = base + step;

            // Only process if both elements are within valid range
            uint32_t logical_second = tile_start + second;
            if (logical_second < meaningful_count) {
                if constexpr (EvalToCoeff) {
                    smem_mle[second] -= smem_mle[base];
                } else {
                    smem_mle[second] += smem_mle[base];
                }
            }
        }
        __syncthreads();
    }

// Write back to global memory
#pragma unroll 4
    for (uint32_t i = tid; i < tile_size; i += blockDim.x) {
        uint32_t logical_idx = tile_start + i;
        if (logical_idx < meaningful_count) {
            uint32_t physical_idx;
            if constexpr (RightPad) {
                physical_idx = logical_idx << log_stride;
            } else {
                physical_idx = logical_idx;
            }
            buffer[col_offset + physical_idx] = smem_mle[i];
        }
    }
}

// ============================================================================
// LAUNCHERS
// ============================================================================

template <typename Field, bool EvalToCoeff>
int launch_mle_interpolate_stage(Field *buffer, size_t buffer_len, uint32_t step) {
    size_t total_pairs = buffer_len >> 1;
    auto [grid, block] = kernel_launch_params(total_pairs);
    mle_interpolate_stage_kernel<Field, EvalToCoeff><<<grid, block>>>(buffer, total_pairs, step);
    return CHECK_KERNEL();
}

extern "C" int _mle_interpolate_stage(
    Fp *buffer,
    size_t buffer_len,
    uint32_t step,
    bool is_eval_to_coeff
) {
    if (buffer_len < 2 || step == 0) {
        return 0;
    }

    if (is_eval_to_coeff) {
        return launch_mle_interpolate_stage<Fp, true>(buffer, buffer_len, step);
    } else {
        return launch_mle_interpolate_stage<Fp, false>(buffer, buffer_len, step);
    }
}

extern "C" int _mle_interpolate_stage_ext(
    FpExt *buffer,
    size_t buffer_len,
    uint32_t step,
    bool is_eval_to_coeff
) {
    if (buffer_len < 2 || step == 0) {
        return 0;
    }

    if (is_eval_to_coeff) {
        return launch_mle_interpolate_stage<FpExt, true>(buffer, buffer_len, step);
    } else {
        return launch_mle_interpolate_stage<FpExt, false>(buffer, buffer_len, step);
    }
}

template <typename Field, bool EvalToCoeff>
int launch_mle_interpolate_stage_2d(
    Field *buffer,
    uint16_t width,
    uint32_t height,
    uint32_t padded_height,
    uint32_t step
) {
    auto span = step * 2;
    auto [grid, block] = kernel_launch_params(height >> 1);
    grid.y = width;
    mle_interpolate_stage_2d_kernel<Field, EvalToCoeff>
        <<<grid, block>>>(buffer, padded_height, span, step);
    return CHECK_KERNEL();
}

extern "C" int _mle_interpolate_stage_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t height,
    uint32_t padded_height,
    uint32_t step,
    bool is_eval_to_coeff
) {
    if (is_eval_to_coeff) {
        return launch_mle_interpolate_stage_2d<Fp, true>(
            buffer, width, height, padded_height, step
        );
    } else {
        return launch_mle_interpolate_stage_2d<Fp, false>(
            buffer, width, height, padded_height, step
        );
    }
}

// Launcher for fused MLE interpolation that processes multiple stages via warp shuffle.
// num_stages: number of consecutive stages to fuse (1-5)
// start_step: the step for the first stage (must be power of 2, <= 16 / (1 << (num_stages-1)))
// log_stride: meaningful_count = padded_height >> log_stride
// RightPad: if true, physical index = tidx << log_stride; if false, physical index = tidx
template <bool EvalToCoeff, int NumStages, bool RightPad>
int launch_mle_interpolate_fused_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t padded_height,
    uint32_t log_stride,
    uint32_t start_step
) {
    uint32_t meaningful_count = padded_height >> log_stride;
    auto [grid, block] = kernel_launch_params(meaningful_count);
    grid.y = width;
    mle_interpolate_fused_2d_kernel<EvalToCoeff, NumStages, RightPad>
        <<<grid, block>>>(buffer, padded_height, log_stride, start_step);
    return CHECK_KERNEL();
}

// Recursive template dispatch for num_stages (1-5)
template <bool EvalToCoeff, bool RightPad, int N = LOG_WARP_SIZE>
int dispatch_mle_interpolate_fused_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t padded_height,
    uint32_t log_stride,
    uint32_t start_step,
    uint32_t num_stages
) {
    if constexpr (N == 0) {
        return cudaErrorInvalidValue;
    } else {
        if (num_stages == N) {
            return launch_mle_interpolate_fused_2d<EvalToCoeff, N, RightPad>(
                buffer, width, padded_height, log_stride, start_step
            );
        }
        return dispatch_mle_interpolate_fused_2d<EvalToCoeff, RightPad, N - 1>(
            buffer, width, padded_height, log_stride, start_step, num_stages
        );
    }
}

extern "C" int _mle_interpolate_fused_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t padded_height,
    uint32_t log_stride,
    uint32_t start_step,
    uint32_t num_stages,
    bool is_eval_to_coeff,
    bool right_pad
) {
    return DISPATCH_BOOL_PAIR(
        dispatch_mle_interpolate_fused_2d,
        is_eval_to_coeff,
        right_pad,
        buffer,
        width,
        padded_height,
        log_stride,
        start_step,
        num_stages
    );
}

// Default tile log size for shared memory kernel.
// tile_size = 2^12 = 4096 elements = 16KB shared memory (for Fp = 4 bytes).
// This allows processing stages with log_step up to 11 (step up to 2048).
// Trade-off: larger tile = more stages fused, but lower occupancy.
constexpr int MLE_SHARED_TILE_LOG_SIZE = 12;

template <bool EvalToCoeff, bool RightPad>
int launch_mle_interpolate_shared_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t padded_height,
    uint32_t log_stride,
    uint32_t start_log_step,
    uint32_t end_log_step
) {
    constexpr uint32_t tile_size = 1 << MLE_SHARED_TILE_LOG_SIZE;
    constexpr uint32_t block_size = 256;
    size_t smem_size = tile_size * sizeof(Fp);

    uint32_t meaningful_count = padded_height >> log_stride;
    uint32_t num_tiles = (meaningful_count + tile_size - 1) / tile_size;
    dim3 grid(num_tiles, width);

    mle_interpolate_shared_2d_kernel<EvalToCoeff, MLE_SHARED_TILE_LOG_SIZE, RightPad>
        <<<grid, block_size, smem_size>>>(
            buffer, padded_height, log_stride, start_log_step, end_log_step
        );
    return CHECK_KERNEL();
}

extern "C" int _mle_interpolate_shared_2d(
    Fp *buffer,
    uint16_t width,
    uint32_t padded_height,
    uint32_t log_stride,
    uint32_t start_log_step,
    uint32_t end_log_step,
    bool is_eval_to_coeff,
    bool right_pad
) {
    // Validate: end_log_step must be < MLE_SHARED_TILE_LOG_SIZE (step < tile_size)
    if (end_log_step >= MLE_SHARED_TILE_LOG_SIZE) {
        return cudaErrorInvalidValue;
    }
    return DISPATCH_BOOL_PAIR(
        launch_mle_interpolate_shared_2d,
        is_eval_to_coeff,
        right_pad,
        buffer,
        width,
        padded_height,
        log_stride,
        start_log_step,
        end_log_step
    );
}
