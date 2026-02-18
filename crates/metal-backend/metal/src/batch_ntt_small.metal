/// Small NTT kernel for Metal.
/// Translated from cuda-backend/cuda/src/batch_ntt_small.cu.
/// Handles NTTs of size 2^l_skip where l_skip <= MAX_NTT_LEVEL (10).

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "device_ntt.h"

// ============================================================================
// Twiddle Generation
// ============================================================================

/// Generate twiddle factors for all levels 1..MAX_NTT_LEVEL.
/// Layout: level L starts at offset (2^L - 2) and has 2^L entries.
/// Each entry is TWO_ADIC_GENERATORS[level]^index.
kernel void generate_device_ntt_twiddles(
    device Fp *d_twiddles [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= DEVICE_NTT_TWIDDLES_SIZE) return;

    // Find which level this tid belongs to
    uint32_t level = 1;
    uint32_t offset = 0;
    while (level <= MAX_NTT_LEVEL) {
        uint32_t level_size = 1u << level;
        if (tid < offset + level_size) break;
        offset += level_size;
        level++;
    }

    uint32_t index = tid - offset;
    d_twiddles[tid] = pow(TWO_ADIC_GENERATORS[level], index);
}

// ============================================================================
// Batch NTT Small (forward and inverse)
// ============================================================================

/// Batch NTT for small sizes (2^l_skip <= 1024).
/// Decimation-in-frequency: natural order input, bit-reversed order output.
///
/// Each thread handles one element within its NTT block.
/// Multiple NTT blocks can be packed into one threadgroup via threadgroup dimension y.
///
/// Parameters:
///   buffer:    data buffer (multiple NTT blocks packed contiguously)
///   twiddles:  precomputed twiddle factors (DEVICE_NTT_TWIDDLES_SIZE entries)
///   l_skip:    log2(NTT size)
///   cnt_blocks: total number of NTT blocks
///   is_intt:   1 for inverse NTT, 0 for forward
kernel void batch_ntt_small(
    device Fp *buffer [[buffer(0)]],
    const device Fp *twiddles [[buffer(1)]],
    constant uint32_t &l_skip [[buffer(2)]],
    constant uint32_t &cnt_blocks [[buffer(3)]],
    constant uint32_t &is_intt [[buffer(4)]],
    threadgroup Fp *smem [[threadgroup(0)]],
    uint2 t_pos [[thread_position_in_threadgroup]], // (x=element_idx, y=block_within_tg)
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    uint tg_pos_x = tg_pos.x;
    // Each threadgroup handles multiple NTT blocks (packed via y dimension)
    uint32_t threads_x = 1u << l_skip;
    uint32_t threads_y = 1024 / threads_x; // matching CUDA: threads_per_block = 1024
    uint32_t block_idx = tg_pos_x * threads_y + t_pos.y;
    bool active_thread = (block_idx < cnt_blocks);

    device Fp *block_ptr = buffer + (block_idx << l_skip);
    uint32_t i = t_pos.x;

    bool needs_tg = (l_skip > LOG_SIMD_SIZE);

    Fp this_thread_value;

    if (needs_tg) {
        threadgroup Fp *sbuf = smem + (t_pos.y << l_skip);
        if (active_thread) {
            sbuf[i] = block_ptr[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_intt) {
            ntt_natural_to_bitrev<true, true>(this_thread_value, sbuf, twiddles, i, l_skip, active_thread);
        } else {
            ntt_natural_to_bitrev<false, true>(this_thread_value, sbuf, twiddles, i, l_skip, active_thread);
        }
    } else if (active_thread) {
        this_thread_value = block_ptr[i];
        if (is_intt) {
            ntt_natural_to_bitrev<true, false>(this_thread_value, nullptr, twiddles, i, l_skip, true);
        } else {
            ntt_natural_to_bitrev<false, false>(this_thread_value, nullptr, twiddles, i, l_skip, true);
        }
    }

    if (active_thread) {
        uint32_t j = rev_len(i, l_skip);
        block_ptr[j] = this_thread_value;
    }
}
