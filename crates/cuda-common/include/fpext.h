/*
 * FpExt - Quintic Extension of Baby Bear Field (BabyBear^5)
 *
 * Defines FpExt, a finite field F_p^5, based on Fp via the irreducible polynomial x^5 - 2.
 *
 * Field size: p^5 ~ 2^155, provides ~155 bits of security.
 *
 * Element representation: a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
 * where each ai is a Baby Bear element and x^5 = 2.
 *
 * Multiplication uses schoolbook algorithm with Fp-level PTX.
 *
 * Originally based on risc0 fpext.h (degree-4), now rewritten for degree-5.
 */

#pragma once

#include "fp.h"

/// FpExt is a degree-5 extension of the Baby Bear field.
/// Elements are represented as polynomials in F_p[x] / (x^5 - 2).
constexpr size_t D_EF = 5;

struct FpExt {
    /// Coefficients: elems[0] + elems[1]*x + elems[2]*x^2 + elems[3]*x^3 + elems[4]*x^4
    union { Fp elems[D_EF]; uint32_t u[D_EF]; };

    /// The non-residue W such that x^5 - W is irreducible
    static constexpr uint32_t W = 2;

    /// Default constructor - zero element
    __device__ FpExt() : elems{Fp(0), Fp(0), Fp(0), Fp(0), Fp(0)} {}

    /// Construct from uint32_t
    __device__ explicit FpExt(uint32_t x) : elems{Fp(x), Fp(0), Fp(0), Fp(0), Fp(0)} {}

    /// Convert from Fp to FpExt
    __device__ explicit FpExt(Fp x) : elems{x, Fp(0), Fp(0), Fp(0), Fp(0)} {}

    /// Explicitly construct from all 5 coefficients
    __device__ FpExt(Fp a, Fp b, Fp c, Fp d, Fp e) {
        elems[0] = a;
        elems[1] = b;
        elems[2] = c;
        elems[3] = d;
        elems[4] = e;
    }

    /// Zero element
    __device__ static FpExt zero() { return FpExt(); }

    /// One element
    __device__ static FpExt one() { return FpExt(Fp::one()); }

    /// Access coefficient
    __device__ Fp& operator[](int i) { return elems[i]; }
    __device__ const Fp& operator[](int i) const { return elems[i]; }

    /// Get constant part
    __device__ Fp constPart() const { return elems[0]; }

    // ========================================================================
    // Addition / Subtraction (component-wise)
    // ========================================================================

    __device__ FpExt operator+=(FpExt rhs) {
        elems[0] += rhs.elems[0];
        elems[1] += rhs.elems[1];
        elems[2] += rhs.elems[2];
        elems[3] += rhs.elems[3];
        elems[4] += rhs.elems[4];
        return *this;
    }

    __device__ FpExt operator-=(FpExt rhs) {
        elems[0] -= rhs.elems[0];
        elems[1] -= rhs.elems[1];
        elems[2] -= rhs.elems[2];
        elems[3] -= rhs.elems[3];
        elems[4] -= rhs.elems[4];
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

    __device__ FpExt operator-(Fp rhs) const {
        FpExt result = *this;
        result.elems[0] -= rhs;
        return result;
    }

    __device__ FpExt operator-() const { return FpExt() - *this; }

    // ========================================================================
    // Scalar multiplication
    // ========================================================================

    __device__ FpExt operator*=(Fp rhs) {
        elems[0] *= rhs;
        elems[1] *= rhs;
        elems[2] *= rhs;
        elems[3] *= rhs;
        elems[4] *= rhs;
        return *this;
    }

    __device__ FpExt operator*(Fp rhs) const {
        FpExt result = *this;
        result *= rhs;
        return result;
    }

    // ========================================================================
    // Full extension field multiplication
    // ========================================================================

    /// c[0] = a0*b0 + 2*(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    /// c[1] = a0*b1 + a1*b0 + 2*(a2*b4 + a3*b3 + a4*b2)
    /// c[2] = a0*b2 + a1*b1 + a2*b0 + 2*(a3*b4 + a4*b3)
    /// c[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + 2*(a4*b4)
    /// c[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    __device__ FpExt operator*(FpExt rhs) const {
        FpExt ret;

        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        const Fp b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2], b3 = rhs.elems[3], b4 = rhs.elems[4];

        Fp c0 = a0 * b0;
        Fp t5_half = a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1;
        ret.elems[0] = c0 + t5_half + t5_half;

        Fp c1 = a0 * b1 + a1 * b0;
        Fp t6_half = a2 * b4 + a3 * b3 + a4 * b2;
        ret.elems[1] = c1 + t6_half + t6_half;

        Fp c2 = a0 * b2 + a1 * b1 + a2 * b0;
        Fp t7_half = a3 * b4 + a4 * b3;
        ret.elems[2] = c2 + t7_half + t7_half;

        Fp c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        Fp a4b4 = a4 * b4;
        ret.elems[3] = c3 + a4b4 + a4b4;

        ret.elems[4] = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        return ret;
    }

    __device__ FpExt& operator*=(FpExt rhs) {
        *this = *this * rhs;
        return *this;
    }

    /// Squaring (optimized using symmetry: a[i]*a[j] appears twice for i != j)
    __device__ FpExt square() const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];

        Fp a0sq = a0 * a0;
        Fp a1sq = a1 * a1;
        Fp a2sq = a2 * a2;
        Fp a3sq = a3 * a3;
        Fp a4sq = a4 * a4;

        Fp a0a1 = a0 * a1;
        Fp a0a2 = a0 * a2;
        Fp a0a3 = a0 * a3;
        Fp a0a4 = a0 * a4;
        Fp a1a2 = a1 * a2;
        Fp a1a3 = a1 * a3;
        Fp a1a4 = a1 * a4;
        Fp a2a3 = a2 * a3;
        Fp a2a4 = a2 * a4;
        Fp a3a4 = a3 * a4;

        // c0 = a0^2 + 4*(a1a4 + a2a3)
        Fp sum14_23 = a1a4 + a2a3;
        Fp c0 = a0sq + sum14_23 + sum14_23 + sum14_23 + sum14_23;

        // c1 = 2*a0a1 + 2*(2*a2a4 + a3^2)
        Fp c1 = a0a1 + a0a1 + a2a4 + a2a4 + a2a4 + a2a4 + a3sq + a3sq;

        // c2 = 2*a0a2 + a1^2 + 4*a3a4
        Fp c2 = a0a2 + a0a2 + a1sq + a3a4 + a3a4 + a3a4 + a3a4;

        // c3 = 2*(a0a3 + a1a2) + 2*a4^2
        Fp sum03_12 = a0a3 + a1a2;
        Fp c3 = sum03_12 + sum03_12 + a4sq + a4sq;

        // c4 = 2*(a0a4 + a1a3) + a2^2
        Fp c4 = a0a4 + a0a4 + a1a3 + a1a3 + a2sq;

        return FpExt(c0, c1, c2, c3, c4);
    }

    // ========================================================================
    // Equality
    // ========================================================================

    __device__ bool operator==(FpExt rhs) const {
        return elems[0] == rhs.elems[0] &&
               elems[1] == rhs.elems[1] &&
               elems[2] == rhs.elems[2] &&
               elems[3] == rhs.elems[3] &&
               elems[4] == rhs.elems[4];
    }

    __device__ bool operator!=(FpExt rhs) const { return !(*this == rhs); }
};

/// Scalar * FpExt
__device__ inline FpExt operator*(Fp a, FpExt b) { return b * a; }

/// Power function using square-and-multiply
__device__ inline FpExt pow(FpExt base, uint32_t exp) {
    FpExt result = FpExt::one();
    while (exp > 0) {
        if (exp & 1) {
            result = result * base;
        }
        base = base.square();
        exp >>= 1;
    }
    return result;
}

template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
__device__ inline FpExt pow(FpExt x, I n) {
    return pow(x, static_cast<uint32_t>(n));
}

/// Frobenius endomorphism: phi^power(a) = a^{p^power}
/// For X^5 - W with p = 1 (mod 5), phi(a_i * x^i) = FROB[i] * a_i * x^i
/// where FROB[i] = W^(i*(p-1)/5)
__device__ inline FpExt frobenius(FpExt a, int power) {
    static constexpr uint32_t FROB_TABLE[4][4] = {
        {0x309476E5, 0x24553874, 0x749B8749, 0x267AC95F},  // FROB[1]^1..4
        {0x24553874, 0x267AC95F, 0x309476E5, 0x749B8749},  // FROB[2]^1..4
        {0x749B8749, 0x309476E5, 0x267AC95F, 0x24553874},  // FROB[3]^1..4
        {0x267AC95F, 0x749B8749, 0x24553874, 0x309476E5},  // FROB[4]^1..4
    };

    int j = power - 1;  // 0-indexed for array access
    return FpExt(
        a.elems[0],
        a.elems[1] * Fp(FROB_TABLE[0][j]),
        a.elems[2] * Fp(FROB_TABLE[1][j]),
        a.elems[3] * Fp(FROB_TABLE[2][j]),
        a.elems[4] * Fp(FROB_TABLE[3][j])
    );
}

/// Inversion using Frobenius norm with Itoh-Tsujii chain.
///
/// For a in Fp5, N(a) = a * phi(a) * phi^2(a) * phi^3(a) * phi^4(a) in Fp.
/// Then a^{-1} = phi(a) * phi^2(a) * phi^3(a) * phi^4(a) / N(a).
///
/// Itoh-Tsujii: f1=phi(a), f2=phi^2(a), f12=f1*f2, f34=phi^2(f12), conj=f12*f34
/// Cost: 3 Frobenius + 2 FpExt muls + partial norm + 1 Fp inv + 5 Fp muls
__device__ inline FpExt inv(FpExt x) {
    if (x == FpExt::zero()) {
        return FpExt::zero();
    }

    FpExt f1 = frobenius(x, 1);
    FpExt f2 = frobenius(x, 2);
    FpExt f12 = f1 * f2;
    FpExt f34 = frobenius(f12, 2);
    FpExt conj = f12 * f34;

    // Partial norm: N(x) = x * conj in Fp
    Fp t5 = x[1]*conj[4] + x[2]*conj[3] + x[3]*conj[2] + x[4]*conj[1];
    Fp norm = x[0]*conj[0] + t5 + t5;

    Fp norm_inv = ::inv(norm);
    return conj * norm_inv;
}

/// Alternative name for compatibility with existing code
__device__ __inline__ FpExt binomial_inversion(const FpExt &in) {
    return inv(in);
}

static_assert(sizeof(FpExt) == 20, "FpExt must be 20 bytes");
