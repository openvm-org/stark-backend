#pragma once

#include "fp.h"
#include <cstdint>

constexpr uint32_t LOG_WARP_SIZE = 5;

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

// Dispatch helper for two bool template parameters
#define DISPATCH_BOOL_PAIR(func, b1, b2, ...)                                                      \
    ((b1) ? ((b2) ? func<true, true>(__VA_ARGS__) : func<true, false>(__VA_ARGS__))                \
          : ((b2) ? func<false, true>(__VA_ARGS__) : func<false, false>(__VA_ARGS__)))
