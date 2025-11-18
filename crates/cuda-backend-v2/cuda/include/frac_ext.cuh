#pragma once

#include "fp.h"
#include "fpext.h"

struct FracExt {
    FpExt p;
    FpExt q;
};

__device__ __forceinline__ FracExt frac_add(const FracExt &a, const FracExt &b) {
    FracExt out;
    out.p = a.p * b.q + a.q * b.p;
    out.q = a.q * b.q;
    return out;
}

__device__ __forceinline__ FpExt make_bool_ext(bool value) { return FpExt(Fp(value ? 1u : 0u)); }
