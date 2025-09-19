/*
 * Source: https://github.com/risc0/risc0 (ref=093f46925a0366d72df505256d35c54a033ddad0)
 * Status: MODIFIED from risc0/sys/kernels/zkp/cuda/fpext.h
 * Imported: 2025-01-25 by @gaxiom
 * 
 * LOCAL CHANGES (high level):
 * - 2025-03-25: add operator-()
 * - 2025-09-19: use sppark's bb31_4_t impl for core arithmetic operations
 */

// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

/// \file
/// Defines FpExt, a finite field F_p^4, based on Fp via the irreducible polynomial x^4 - 11.

#include "fp.h"

/// Instances of FpExt are element of a finite field F_p^4.  They are represented as elements of
/// F_p[X] / (X^4 - 11). Basically, this is a 'big' finite field (about 2^128 elements), which is
/// used when the security of various operations depends on the size of the field.  It has the field
/// Fp as a subfield, which means operations by the two are compatible, which is important.  The
/// irreducible polynomial was chosen to be the simplest possible one, x^4 - B, where 11 is the
/// smallest B which makes the polynomial irreducible.
struct FpExt {
    /// The elements of FpExt, elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*x^4
    union {
        Fp elems[4];
        bb31_4_t rep;
    };
    
    /// Default constructor makes the zero elements
    __device__ FpExt() : rep(0) {}

    /// Initialize from uint32_t
    __device__ explicit FpExt(uint32_t x) : rep(bb31_t{x}) {}

    /// Convert from Fp to FpExt.
    __device__ explicit FpExt(Fp x) : rep(static_cast<bb31_t>(x)) {}

    /// Explicitly construct an FpExt from parts
    __device__ FpExt(Fp a, Fp b, Fp c, Fp d) {
        elems[0] = a;
        elems[1] = b;
        elems[2] = c;
        elems[3] = d;
    }

    // Implement the addition/subtraction overloads
    __device__ FpExt operator+=(FpExt rhs) {
        rep += rhs.rep;
        return *this;
    }

    __device__ FpExt operator-=(FpExt rhs) {
        rep -= rhs.rep;
        return *this;
    }

    __device__ FpExt operator+(FpExt rhs) const {
        FpExt result = *this;
        result += rhs;
        return result;
    }

    __device__ FpExt operator-(FpExt rhs) const {
        FpExt result = *this;
        result -= rhs;
        return result;
    }

    __device__ FpExt operator-(Fp rhs) {
        FpExt result = *this;
        result.elems[0] -= rhs;
        return result;
    }

    __device__ FpExt operator-() const { return FpExt() - *this; }

    // Implement the simple multiplication case by the subfield Fp
    // Fp * FpExt is done as a free function due to C++'s operator overloading rules.
    __device__ FpExt operator*=(Fp rhs) {
        rep *= static_cast<bb31_t>(rhs);
        return *this;
    }

    __device__ FpExt operator*(Fp rhs) const {
        FpExt result = *this;
        result *= rhs;
        return result;
    }

    __device__ FpExt operator*=(FpExt rhs) {
        rep *= rhs.rep;
        return *this;
    }

    __device__ FpExt operator*(FpExt rhs) const {
        FpExt result = *this;
        result *= rhs;
        return result;
    }

    // Equality
    __device__ bool operator==(FpExt rhs) const { return rep == rhs.rep; }
    __device__ bool operator!=(FpExt rhs) const { return rep != rhs.rep; }

    __device__ Fp constPart() const { return elems[0]; }
};

/// Overload for case where LHS is Fp (RHS case is handled as a method)
__device__ inline FpExt operator*(Fp a, FpExt b) { return b * a; }

/// Raise an FpExt to a power
__device__ inline FpExt pow(FpExt x, uint32_t n) {
    FpExt result;
    result.rep = x.rep ^ n;
    return result;
}

/// Compute the multiplicative inverse of an FpExt.
__device__ inline FpExt inv(FpExt in) {
    FpExt result;
    result.rep = in.rep.reciprocal();
    return result;
}

// TODO: find the difference between binomial_inversion and inv
__device__ __inline__ FpExt binomial_inversion(const FpExt &in) {
    constexpr uint32_t dth_root_u32 = 1728404513;
    constexpr uint32_t w = 11;
    Fp D(dth_root_u32);
    Fp W(w);

    FpExt f(1);
    Fp D2 = D * D;
    Fp D3 = D2 * D;
    for (int i = 1; i < 4; ++i) {
        f = f * in;
        f.elems[1] *= D;
        f.elems[2] *= D2;
        f.elems[3] *= D3;
    }
    Fp g = (in.elems[1] * f.elems[3] + in.elems[2] * f.elems[2] + in.elems[3] * f.elems[1]) * W +
           in.elems[0] * f.elems[0];
    return f * FpExt(inv(g));
}