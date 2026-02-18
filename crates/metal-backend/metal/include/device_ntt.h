#pragma once

/// NTT structures and device-side NTT routines for Metal.
/// Mirrors device_ntt.cuh from the CUDA backend.

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"
#include "utils.h"

// Maximum NTT level for precomputed twiddles (batch_ntt_small).
// Level L needs 2^L twiddle factors.
// MAX_NTT_LEVEL=10: total = 2^11 - 2 = 2046 elements.
constant uint32_t MAX_NTT_LEVEL = 10;
constant uint32_t DEVICE_NTT_TWIDDLES_SIZE = (1u << (MAX_NTT_LEVEL + 1)) - 2;

/// Get offset into the twiddle buffer for level L (L >= 1)
inline uint32_t twiddle_offset(uint32_t level) {
    return (1u << level) - 2;
}

/// Get precomputed twiddle: omega_level^index from a buffer.
inline Fp get_twiddle(const device Fp *twiddles, uint32_t level, uint32_t index) {
    return twiddles[twiddle_offset(level) + index];
}

/// Halve-or-identity helper for NTT butterfly.
/// For inverse NTT (intt=true), returns x.halve(); for forward, returns x.
inline Fp sum_or_semi_sum_fwd(Fp x) { return x; }
inline Fp sum_or_semi_sum_inv(Fp x) { return x.halve(); }

/// Decimation-in-frequency NTT: natural order to bit-reversed order.
/// Uses SIMD shuffle for sizes <= 32, threadgroup memory for larger.
///
/// For `needs_tg_mem = false`: `this_thread_value` must contain input on entry.
/// For `needs_tg_mem = true`:  `sbuf[i]` must contain input on entry.
///
/// After return, `this_thread_value` holds the output in bit-reversed order.
template <bool intt, bool needs_tg_mem>
inline void ntt_natural_to_bitrev(
    thread Fp &this_thread_value,
    threadgroup Fp *sbuf,
    const device Fp *twiddles,
    uint32_t i,
    uint32_t l_skip,
    bool active_thread
) {
    uint32_t log_interwarp = needs_tg_mem ? LOG_SIMD_SIZE : l_skip;

    // Twiddle index: reversed for iNTT
    uint32_t twiddle_idx = intt ? (i ? (1u << l_skip) - i : 0u) : i;
    Fp twiddle = get_twiddle(twiddles, l_skip, twiddle_idx);

    // Threadgroup memory phase (for l_skip > LOG_SIMD_SIZE)
    if (needs_tg_mem) {
        for (uint32_t log_len = l_skip; log_len-- > LOG_SIMD_SIZE;) {
            if (active_thread) {
                uint32_t len = 1u << log_len;
                if (!(i & len)) {
                    Fp s = sbuf[i];
                    Fp d = sbuf[i + len];
                    sbuf[i] = intt ? (s + d).halve() : (s + d);
                    sbuf[i + len] = intt ? (s - d).halve() * twiddle : (s - d) * twiddle;
                }
                twiddle *= twiddle;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        this_thread_value = sbuf[i];
    }

    // SIMD shuffle phase (for remaining log_interwarp levels)
    for (uint32_t log_len = log_interwarp; log_len-- > 0;) {
        uint32_t len = 1u << log_len;
        Fp other_value = Fp::fromRaw(simd_shuffle_xor(this_thread_value.asRaw(), len));
        if (!(i & len)) {
            this_thread_value = intt ? (this_thread_value + other_value).halve()
                                     : (this_thread_value + other_value);
        } else {
            this_thread_value = intt ? (this_thread_value - other_value).halve() * twiddle
                                     : (this_thread_value - other_value) * twiddle;
        }
        twiddle *= twiddle;
    }
}

/// Decimation-in-time NTT: bit-reversed order to natural order.
template <bool intt, bool needs_tg_mem>
inline void ntt_bitrev_to_natural(
    thread Fp &this_thread_value,
    threadgroup Fp *sbuf,
    const device Fp *twiddles,
    uint32_t i,
    uint32_t l_skip
) {
    uint32_t simd_levels = needs_tg_mem ? LOG_SIMD_SIZE : l_skip;

    // SIMD shuffle phase: levels 0 to simd_levels-1 (small to large)
    for (uint32_t m = 0; m < simd_levels; m++) {
        uint32_t len = 1u << m;
        uint32_t j = i & (len - 1);
        if (intt) j = j ? ((2u << m) - j) : 0u;
        Fp tw = get_twiddle(twiddles, m + 1, j);
        Fp other = Fp::fromRaw(simd_shuffle_xor(this_thread_value.asRaw(), len));

        if (i & len) {
            Fp b_tw = this_thread_value * tw;
            this_thread_value = intt ? (other - b_tw).halve() : (other - b_tw);
        } else {
            Fp b_tw = other * tw;
            this_thread_value = intt ? (this_thread_value + b_tw).halve()
                                     : (this_thread_value + b_tw);
        }
    }

    // Threadgroup memory phase: levels LOG_SIMD_SIZE to l_skip-1
    if (needs_tg_mem) {
        sbuf[i] = this_thread_value;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint32_t m = LOG_SIMD_SIZE; m < l_skip; m++) {
            uint32_t len = 1u << m;
            uint32_t j = i & (len - 1);
            if (intt) j = j ? ((2u << m) - j) : 0u;
            Fp tw = get_twiddle(twiddles, m + 1, j);

            if (!(i & len)) {
                Fp a = sbuf[i];
                Fp b_tw = sbuf[i + len] * tw;
                sbuf[i] = intt ? (a + b_tw).halve() : (a + b_tw);
                sbuf[i + len] = intt ? (a - b_tw).halve() : (a - b_tw);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        this_thread_value = sbuf[i];
    }
}
