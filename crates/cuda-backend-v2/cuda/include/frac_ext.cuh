#pragma once

#include "fp.h"
#include "fpext.h"

struct FracExt {
    FpExt p;
    FpExt q;
};

__device__ __forceinline__ void frac_add_comps_inplace(FpExt& lhs_num, FpExt& lhs_denom, FpExt const& rhs_num, FpExt const& rhs_denom) {
    lhs_num = lhs_num * rhs_denom + lhs_denom * rhs_num;
    lhs_denom = lhs_denom * rhs_denom;
}

__device__ __forceinline__ FracExt frac_add(FracExt a, const FracExt &b) {
    frac_add_comps_inplace(a.p, a.q, b.p, b.q);
    return a;
}

__device__ __forceinline__ void frac_unadd_comps_inplace(FpExt& lhs_num, FpExt& lhs_denom, FpExt const& rhs_num, FpExt const& rhs_denom) {
    FpExt rhs_denom_inv = inv(rhs_denom);
    lhs_denom = lhs_denom * rhs_denom_inv;
    lhs_num = (lhs_num - lhs_denom * rhs_num) * rhs_denom_inv;
}

/// Find such `c` that `frac(add(c, b)) == a`.
/// Note that it's not "same as add but minus instead of plus", because we want to find
/// the exact numerator and denominator, not multiplied by a constant
__device__ __forceinline__ FracExt frac_unadd(FracExt a, const FracExt &b) {
    frac_unadd_comps_inplace(a.p, a.q, b.p, b.q);
    return a;
}

__device__ __forceinline__ FpExt make_bool_ext(bool value) { return FpExt(Fp(value ? 1u : 0u)); }
