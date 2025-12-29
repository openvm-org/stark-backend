#pragma once

#include "fp.h"
#include "utils.cuh"
#include <cmath>
#include <cstdint>

namespace device_ntt {

template <bool intt> __device__ __forceinline__ Fp sum_or_semi_sum(Fp &&x) {
    if constexpr (intt) {
        return x.halve();
    } else {
        return x;
    }
}

// Reusable device function for entirely on-device NTT of size `2^l_skip`.
// Uses shared memory for sizes > warp, warp shuffles for sizes <= warp.
//
// Decimation in frequency NTT: natural order to bit reversed order.
//
// Uses exactly `2^l_skip` threads, and `this_thread_value` will be set with the output of the NTT.
// The input should be the evaluation at `omega^i`.
// - If `needs_shmem` is true, then `sbuf[i]` must already contain the input.
// - If `needs_shmem` is false, then `this_thread_value` must already contain the input.
//
// `active_thread` indicates whether this thread has valid data; inactive threads still participate in syncs.
//
// # Assumption
// - No warp has exited early before this call. We use the full `0xffffffff` mask.
template <bool intt, bool needs_shmem>
__device__ __forceinline__ void ntt_natural_to_bitrev(
    Fp &this_thread_value,
    Fp *__restrict__ sbuf, // shared memory buffer for this thread's NTT (size 1 << l_skip)
    uint32_t const i,      // thread index within NTT [0, 1 << l_skip)
    uint32_t const l_skip, // log2 of NTT size
    bool const active_thread = true
) {
    uint32_t log_interwarp;
    if constexpr (needs_shmem) {
        log_interwarp = LOG_WARP_SIZE;
    } else {
        log_interwarp = l_skip;
    }
    // reverse index for iNTT to get the inverse twiddle
    uint32_t const twiddle_idx = intt ? (i ? (1 << l_skip) - i : 0) : i;
    Fp twiddle = pow(TWO_ADIC_GENERATORS[l_skip], twiddle_idx);

    // Shared memory phase (for l_skip > LOG_WARP_SIZE)
    if constexpr (needs_shmem) {
        for (uint32_t log_len = l_skip; log_len-- > LOG_WARP_SIZE;) {
            if (active_thread) {
                uint32_t const len = 1u << log_len;
                if (!(i & len)) {
                    Fp const sum = sbuf[i];
                    Fp const diff = sbuf[i + len];
                    sbuf[i] = sum_or_semi_sum<intt>(sum + diff);
                    sbuf[i + len] = sum_or_semi_sum<intt>(sum - diff) * twiddle;
                }
                twiddle *= twiddle;
            }
            __syncthreads();
        }
        this_thread_value = sbuf[i];
    }
    // Warp shuffle phase (for remaining log_interwarp levels)
    for (uint32_t log_len = log_interwarp; log_len-- > 0;) {
        uint32_t const len = 1u << log_len;
        Fp const other_value =
            Fp::fromRaw(__shfl_xor_sync(0xffffffff, this_thread_value.asRaw(), len));
        if (!(i & len)) {
            // this_thread_value = sum, other_value = diff
            this_thread_value = sum_or_semi_sum<intt>(this_thread_value + other_value);
        } else {
            // this_thread_value = diff, other_value = sum
            this_thread_value = sum_or_semi_sum<intt>(this_thread_value - other_value) * twiddle;
        }
        twiddle *= twiddle;
    }
}

// Decimation in time NTT: bit reversed order to natural order.
//
// Uses exactly `2^l_skip` threads, and `this_thread_value` will be set with the output of the NTT.
// For both `needs_shmem` equals `true` and `false`, `this_thread_value` must already contain the input, which should be the `rev_len(i, l_skip)`-th element.
//
// # Assumption
// - No warp has exited early before this call. We use the full `0xffffffff` mask.
// TODO[jpw]: create constant memory for partial twiddles
template <bool intt, bool needs_shmem>
__device__ __forceinline__ void ntt_bitrev_to_natural(
    Fp &this_thread_value,
    Fp *__restrict__ sbuf,
    uint32_t const i,
    uint32_t const l_skip
) {
    uint32_t const warp_levels = needs_shmem ? LOG_WARP_SIZE : l_skip;

    // Warp shuffle phase: levels 0 to warp_levels-1 (small to large)
    for (uint32_t m = 0; m < warp_levels; m++) {
        uint32_t const len = 1u << m;
        // Twiddle for DIT: primitive 2^(m+1)-th root raised to (i & (len-1))
        uint32_t j = i & (len - 1);
        if constexpr (intt)
            j = j ? ((2u << m) - j) : 0;
        Fp twiddle = pow(TWO_ADIC_GENERATORS[m + 1], j);

        Fp other = Fp::fromRaw(__shfl_xor_sync(0xffffffff, this_thread_value.asRaw(), len));

        if (i & len) {
            // Bottom: result = a - b * twiddle
            Fp b_tw = this_thread_value * twiddle;
            this_thread_value = sum_or_semi_sum<intt>(other - b_tw);
        } else {
            // Top: result = a + b * twiddle
            Fp b_tw = other * twiddle;
            this_thread_value = sum_or_semi_sum<intt>(this_thread_value + b_tw);
        }
    }

    // Shared memory phase: levels LOG_WARP_SIZE to l_skip-1
    if constexpr (needs_shmem) {
        sbuf[i] = this_thread_value;
        __syncthreads();

        for (uint32_t m = LOG_WARP_SIZE; m < l_skip; m++) {
            uint32_t const len = 1u << m;
            uint32_t j = i & (len - 1);
            if constexpr (intt)
                j = j ? ((2u << m) - j) : 0;
            Fp twiddle = pow(TWO_ADIC_GENERATORS[m + 1], j);

            if (!(i & len)) {
                Fp a = sbuf[i];
                Fp b_tw = sbuf[i + len] * twiddle;
                sbuf[i] = sum_or_semi_sum<intt>(a + b_tw);
                sbuf[i + len] = sum_or_semi_sum<intt>(a - b_tw);
            }
            __syncthreads();
        }
        this_thread_value = sbuf[i];
    }
}

} // namespace device_ntt
