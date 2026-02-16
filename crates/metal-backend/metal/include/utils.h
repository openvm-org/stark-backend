#pragma once

/// Shared utility functions for Metal compute kernels.

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"

constant uint32_t LOG_SIMD_SIZE = 5; // Metal SIMD width = 32

/// Reverse the bottom `len` bits of `x`.
inline uint32_t rev_len(uint32_t x, uint32_t len) {
    return reverse_bits(x) >> (32 - len);
}

/// Compute x^{2^power_log} by repeated squaring.
inline Fp exp_power_of_2(Fp x, uint32_t power_log) {
    Fp res = x;
    while (power_log--) {
        res *= res;
    }
    return res;
}

/// Ceiling division: (a + b - 1) / b
inline uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

/// Find the position of the lowest set bit (1-indexed), like CUDA's __ffs.
inline uint32_t accel_ffs(uint32_t x) {
    return 32 - clz(x & (~x + 1));
}

/// MLE interpolation helper: fold two evaluations using challenge r.
/// output = t0 + r * (t1 - t0)
template <typename Field>
inline Field mle_interpolate_single(const device Field *column, Fp x, uint32_t y_int) {
    Field t0 = column[y_int << 1];
    Field t1 = column[(y_int << 1) | 1];
    return t0 + (t1 - t0) * x;
}
