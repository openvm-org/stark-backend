#pragma once

#include "fp.h"
#include <cstdint>

constexpr uint32_t LOG_WARP_SIZE = 5;

/// Compute x^{2^power_log} by repeated squaring.
__device__ __forceinline__ Fp exp_power_of_2(Fp x, uint32_t power_log) {
    Fp res = x;
    while (power_log--) {
        res.sqr();
    }
    return res;
}

// Given x and 2^n, computes 1/2^n * (1 + x + ... + x^{2^n - 1}).
__device__ __forceinline__ Fp avg_gp(Fp x, uint32_t n) {
#ifdef CUDA_DEBUG
    assert(n && !(n & (n - 1)));
#endif
    Fp res = Fp::one();
    for (uint32_t i = 1; i < n; i <<= 1) {
        res *= Fp::one() + x;
        res = res.halve();
        x *= x;
    }
    return res;
}

/// Assuming `x` is a `len`-bit number, reverses its `len` bits.
__device__ __forceinline__ uint32_t rev_len(uint32_t x, uint32_t len) {
    return __brev(x) >> (32 - len);
}

__device__ __forceinline__ uint32_t with_rev_bits(uint32_t x, uint32_t buf_size) { return x; }

/// Given an index and the buffer size (must be a power of two), turns on some last bits as provided.
/// Example: with_rev_bits(0b110, 32, 1, 0) == 0b10110.
template <typename... Bool>
__device__ __forceinline__ uint32_t
with_rev_bits(uint32_t x, uint32_t buf_size, bool first, Bool &&...others) {
    buf_size >>= 1;
    return with_rev_bits(x | (first * buf_size), buf_size, others...);
}

#define DISPATCH_BOOL(func, b1, ...) ((b1) ? func<true>(__VA_ARGS__) : func<false>(__VA_ARGS__))

// Dispatch helper for two bool template parameters
#define DISPATCH_BOOL_PAIR(func, b1, b2, ...)                                                      \
    ((b1) ? ((b2) ? func<true, true>(__VA_ARGS__) : func<true, false>(__VA_ARGS__))                \
          : ((b2) ? func<false, true>(__VA_ARGS__) : func<false, false>(__VA_ARGS__)))

// Generic dispatcher for <uint32_t N, bool B1, bool B2> template functions.
// Generates recursive template to dispatch runtime n (1..MAX_N) to compile-time N.
// Usage: DEFINE_DISPATCH_N_B1_B2(dispatcher_name, target_func, MAX_N)
// Then call: dispatcher_name(n, b1, b2, args...)
#define DEFINE_DISPATCH_N_B1_B2(name, func, max_n)                                                 \
    template <uint32_t N, typename... Args>                                                        \
    inline int name##_impl(uint32_t n, bool b1, bool b2, Args &&...args) {                         \
        if (n == N) {                                                                              \
            if (b1) {                                                                              \
                return b2 ? func<N, true, true>(std::forward<Args>(args)...)                       \
                          : func<N, true, false>(std::forward<Args>(args)...);                     \
            } else {                                                                               \
                return b2 ? func<N, false, true>(std::forward<Args>(args)...)                      \
                          : func<N, false, false>(std::forward<Args>(args)...);                    \
            }                                                                                      \
        } else if constexpr (N == 1) {                                                             \
            return -1;                                                                             \
        } else {                                                                                   \
            return name##_impl<N - 1>(n, b1, b2, std::forward<Args>(args)...);                     \
        }                                                                                          \
    }                                                                                              \
    template <typename... Args> inline int name(uint32_t n, bool b1, bool b2, Args &&...args) {    \
        return name##_impl<max_n>(n, b1, b2, std::forward<Args>(args)...);                         \
    }
