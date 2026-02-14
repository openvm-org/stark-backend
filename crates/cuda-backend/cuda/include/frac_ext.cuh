#pragma once

#include "fp.h"
#include "fpext.h"

struct FracExt {
    FpExt p;
    FpExt q;
};

__device__ __forceinline__ void frac_add_inplace(FracExt& lhs, FracExt const& rhs) {
    lhs.p = lhs.p * rhs.q + lhs.q * rhs.p;
    lhs.q = lhs.q * rhs.q;
}

__device__ __forceinline__ FracExt frac_add(FracExt a, const FracExt &b) {
    frac_add_inplace(a, b);
    return a;
}

__device__ __forceinline__ void frac_unadd_inplace(FracExt& lhs, FracExt const& rhs) {
    FpExt rhs_denom_inv = inv(rhs.q);
    lhs.q = lhs.q * rhs_denom_inv;
    lhs.p = (lhs.p - lhs.q * rhs.p) * rhs_denom_inv;
}

/// Find such `c` that `frac(add(c, b)) == a`.
/// Note that it's not "same as add but minus instead of plus", because we want to find
/// the exact numerator and denominator, not multiplied by a constant
__device__ __forceinline__ FracExt frac_unadd(FracExt a, const FracExt &b) {
    frac_unadd_inplace(a, b);
    return a;
}

__device__ __forceinline__ FpExt make_bool_ext(bool value) { return FpExt(Fp(value ? 1u : 0u)); }
