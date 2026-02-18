// Fractional extension field for Metal
// Translated from CUDA: cuda-backend/cuda/include/frac_ext.cuh
#pragma once

#include "baby_bear.h"
#include "baby_bear_ext.h"

struct FracExt {
    FpExt p;
    FpExt q;
};

inline void frac_add_inplace(thread FracExt &lhs, const thread FracExt &rhs) {
    lhs.p = lhs.p * rhs.q + lhs.q * rhs.p;
    lhs.q = lhs.q * rhs.q;
}

inline FracExt frac_add(FracExt a, const thread FracExt &b) {
    frac_add_inplace(a, b);
    return a;
}

inline void frac_unadd_inplace(thread FracExt &lhs, const thread FracExt &rhs) {
    FpExt rhs_denom_inv = fpext_inv(rhs.q);
    lhs.q = lhs.q * rhs_denom_inv;
    lhs.p = (lhs.p - lhs.q * rhs.p) * rhs_denom_inv;
}

// Find such `c` that `frac(add(c, b)) == a`.
inline FracExt frac_unadd(FracExt a, const thread FracExt &b) {
    frac_unadd_inplace(a, b);
    return a;
}

inline FpExt make_bool_ext(bool value) {
    return FpExt{Fp(value ? 1u : 0u), Fp(0u), Fp(0u), Fp(0u)};
}
