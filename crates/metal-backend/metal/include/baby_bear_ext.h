// BabyBear degree-4 extension field arithmetic for Metal Shading Language.
//
// Based on risc0/risc0/build_kernel/kernels/metal/fpext.h (Apache-2.0)
// and openvm-org cuda-common/include/fpext.h
//
// F_p^4 = F_p[X] / (X^4 - 11)
// BETA = 11 (the smallest value making x^4 - BETA irreducible over BabyBear)

#pragma once

#include <metal_stdlib>

#include "baby_bear.h"

// BETA and NBETA are used in the multiplication and inverse formulae.
// Defined as macros (following risc0 convention) and undef'd at end of file.
#define BETA Fp(11)
#define NBETA Fp(Fp::P - 11)

/// FpExt represents an element of the degree-4 extension field F_p^4.
///
/// An element is a[0] + a[1]*X + a[2]*X^2 + a[3]*X^3 in F_p[X] / (X^4 - 11).
struct FpExt {
    /// Coefficients: elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*X^3
    Fp elems[4];

    // ----------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------

    /// Default: zero element
    constexpr FpExt() {
        elems[0] = Fp();
        elems[1] = Fp();
        elems[2] = Fp();
        elems[3] = Fp();
    }

    /// Construct from a uint32_t (embed as constant polynomial)
    explicit constexpr FpExt(uint32_t x) {
        elems[0] = Fp(x);
        elems[1] = Fp();
        elems[2] = Fp();
        elems[3] = Fp();
    }

    /// Embed a base-field element into the extension
    explicit constexpr FpExt(Fp x) {
        elems[0] = x;
        elems[1] = Fp();
        elems[2] = Fp();
        elems[3] = Fp();
    }

    /// Explicit 4-coefficient constructor
    constexpr FpExt(Fp a, Fp b, Fp c, Fp d) {
        elems[0] = a;
        elems[1] = b;
        elems[2] = c;
        elems[3] = d;
    }

    // ----------------------------------------------------------------
    // Special operations
    // ----------------------------------------------------------------

    /// Replace invalid sentinel components with zero
    constexpr FpExt zeroize() {
        for (uint32_t i = 0; i < 4; i++) {
            elems[i].zeroize();
        }
        return *this;
    }

    /// Return the constant part (coefficient of X^0)
    constexpr Fp constPart() const { return elems[0]; }

    // ----------------------------------------------------------------
    // Addition / subtraction
    // ----------------------------------------------------------------

    constexpr FpExt operator+=(FpExt rhs) {
        for (uint32_t i = 0; i < 4; i++) {
            elems[i] += rhs.elems[i];
        }
        return *this;
    }

    constexpr FpExt operator-=(FpExt rhs) {
        for (uint32_t i = 0; i < 4; i++) {
            elems[i] -= rhs.elems[i];
        }
        return *this;
    }

    constexpr FpExt operator+(FpExt rhs) const {
        FpExt result = *this;
        result += rhs;
        return result;
    }

    constexpr FpExt operator-(FpExt rhs) const {
        FpExt result = *this;
        result -= rhs;
        return result;
    }

    constexpr FpExt operator-(Fp rhs) const {
        FpExt result = *this;
        result.elems[0] -= rhs;
        return result;
    }

    constexpr FpExt operator-() const { return FpExt() - *this; }

    // ----------------------------------------------------------------
    // Multiplication by base field Fp
    // ----------------------------------------------------------------

    constexpr FpExt operator*=(Fp rhs) {
        for (uint32_t i = 0; i < 4; i++) {
            elems[i] *= rhs;
        }
        return *this;
    }

    constexpr FpExt operator*(Fp rhs) const {
        FpExt result = *this;
        result *= rhs;
        return result;
    }

    // ----------------------------------------------------------------
    // Multiplication of two extension field elements
    //
    // Multiply out the polynomial representations and reduce modulo X^4 - 11.
    // Powers >= 4 get shifted back by 4 and multiplied by NBETA (= -11 mod P).
    // ----------------------------------------------------------------

    constexpr FpExt operator*(FpExt rhs) const {
#define a elems
#define b rhs.elems
        return FpExt(a[0] * b[0] + NBETA * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]),
                     a[0] * b[1] + a[1] * b[0] + NBETA * (a[2] * b[3] + a[3] * b[2]),
                     a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + NBETA * (a[3] * b[3]),
                     a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0]);
#undef a
#undef b
    }

    constexpr FpExt operator*=(FpExt rhs) {
        *this = *this * rhs;
        return *this;
    }

    // ----------------------------------------------------------------
    // Equality
    // ----------------------------------------------------------------

    constexpr bool operator==(FpExt rhs) const {
        for (uint32_t i = 0; i < 4; i++) {
            if (elems[i] != rhs.elems[i]) {
                return false;
            }
        }
        return true;
    }

    constexpr bool operator!=(FpExt rhs) const { return !(*this == rhs); }

    // ----------------------------------------------------------------
    // device-qualified overloads
    // ----------------------------------------------------------------

    constexpr FpExt operator+=(FpExt rhs) device {
        for (uint32_t i = 0; i < 4; i++) {
            elems[i] += rhs.elems[i];
        }
        return *this;
    }

    constexpr FpExt operator+(FpExt rhs) device const { return FpExt(*this) + rhs; }
    constexpr FpExt operator-(FpExt rhs) device const { return FpExt(*this) - rhs; }
    constexpr FpExt operator-() device const { return -FpExt(*this); }
    constexpr FpExt operator*(FpExt rhs) device const { return FpExt(*this) * rhs; }
    constexpr FpExt operator*(Fp rhs) device const { return FpExt(*this) * rhs; }
    constexpr bool operator==(FpExt rhs) device const { return FpExt(*this) == rhs; }
    constexpr bool operator!=(FpExt rhs) device const { return FpExt(*this) != rhs; }
    constexpr Fp constPart() device const { return FpExt(*this).constPart(); }
};

// ============================================================================
// Free functions
// ============================================================================

/// Fp * FpExt (the reverse direction is handled by the member operator*)
constexpr inline FpExt operator*(Fp a, FpExt b) {
    return b * a;
}

/// Raise an FpExt to the n-th power
constexpr inline FpExt pow(FpExt x, size_t n) {
    FpExt tot(1);
    while (n != 0) {
        if (n % 2 == 1) {
            tot *= x;
        }
        n = n / 2;
        x *= x;
    }
    return tot;
}

/// Multiplicative inverse of an FpExt element.
///
/// Uses the composite-field inversion technique (analogous to complex number
/// inversion). See risc0 fpext.h for the detailed derivation.
///
/// For zero input, returns zero (safe inverse convention).
constexpr inline FpExt inv(FpExt in) {
#define a in.elems
    // Step 1: compute b = a * a' where a' has negated odd components.
    // By construction b has zero odd components. Let b = (b0, 0, b2, 0).
    Fp b0 = a[0] * a[0] + BETA * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
    Fp b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + BETA * (a[3] * a[3]);

    // Step 2: compute c = b * b' which is a base-field element.
    Fp c = b0 * b0 + BETA * b2 * b2;

    // Step 3: invert c in the base field.
    Fp ic = inv(c);

    // Step 4: multiply a' * b' * ic to get the full inverse.
    b0 *= ic;
    b2 *= ic;
    return FpExt(a[0] * b0 + BETA * a[2] * b2,
                 -a[1] * b0 + NBETA * a[3] * b2,
                 -a[0] * b2 + a[2] * b0,
                 a[1] * b2 - a[3] * b0);
#undef a
}

/// Helper: construct an FpExt from a boolean value (0 or 1)
inline FpExt make_bool_ext(bool value) { return FpExt(Fp(value ? 1u : 0u)); }

#undef BETA
#undef NBETA
