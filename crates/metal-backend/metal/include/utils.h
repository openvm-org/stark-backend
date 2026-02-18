// Utility functions for Metal Shading Language kernels.
//
// Based on openvm-org cuda-backend/cuda/include/utils.cuh

#pragma once

#include <metal_stdlib>

#include "baby_bear.h"

using namespace metal;

/// Compute x^{2^power_log} by repeated squaring.
inline Fp exp_power_of_2(Fp x, uint32_t power_log) {
    Fp res = x;
    while (power_log--) {
        res = res.sqr();
    }
    return res;
}

/// Given x and n (a power of 2), computes (1/n) * (1 + x + ... + x^{n-1}).
/// This is the average of the geometric progression with ratio x.
inline Fp avg_gp(Fp x, uint32_t n) {
    Fp res = Fp::one();
    for (uint32_t i = 1; i < n; i <<= 1) {
        res *= Fp::one() + x;
        res = res.halve();
        x *= x;
    }
    return res;
}

/// Reverse the bottom `len` bits of `x`.
/// Metal provides reverse_bits() which reverses all 32 bits; we then shift.
inline uint32_t rev_len(uint32_t x, uint32_t len) {
    return reverse_bits(x) >> (32 - len);
}
