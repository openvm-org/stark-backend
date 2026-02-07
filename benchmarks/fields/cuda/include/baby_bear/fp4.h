/*
 * Fp4 - Quartic Extension of BabyBear Field (Non-optimized version)
 * 
 * This is a simple schoolbook implementation of BabyBear[4] for benchmarking
 * comparison against the optimized bb31_4_t (FpExt) implementation.
 * 
 * Polynomial: X^4 - 11 (same as FpExt/bb31_4_t)
 * 
 * Element representation: a0 + a1*α + a2*α² + a3*α³
 * where each ai is a BabyBear element and α^4 = 11.
 * 
 * Reduction rule: α^4 = 11
 */

#pragma once

#include "fp.h"

/// Fp4 is a degree-4 extension of the BabyBear field (simple implementation).
struct Fp4 {
    Fp elems[4];  // [a0, a1, a2, a3]
    
    static constexpr uint32_t W = 11;  // α^4 = 11
    
    __device__ Fp4() : elems{Fp(0), Fp(0), Fp(0), Fp(0)} {}
    
    __device__ explicit Fp4(Fp x) : elems{x, Fp(0), Fp(0), Fp(0)} {}
    
    __device__ explicit Fp4(uint32_t x) : elems{Fp(x), Fp(0), Fp(0), Fp(0)} {}
    
    __device__ Fp4(Fp a0, Fp a1, Fp a2, Fp a3) : elems{a0, a1, a2, a3} {}
    
    __device__ static Fp4 zero() { return Fp4(); }
    __device__ static Fp4 one() { return Fp4(Fp::one()); }
    
    __device__ Fp& operator[](int i) { return elems[i]; }
    __device__ const Fp& operator[](int i) const { return elems[i]; }
    
    __device__ Fp constPart() const { return elems[0]; }
    
    // ========================================================================
    // Addition / Subtraction
    // ========================================================================
    
    __device__ Fp4 operator+(Fp4 rhs) const {
        return Fp4(
            elems[0] + rhs.elems[0],
            elems[1] + rhs.elems[1],
            elems[2] + rhs.elems[2],
            elems[3] + rhs.elems[3]
        );
    }
    
    __device__ Fp4 operator-(Fp4 rhs) const {
        return Fp4(
            elems[0] - rhs.elems[0],
            elems[1] - rhs.elems[1],
            elems[2] - rhs.elems[2],
            elems[3] - rhs.elems[3]
        );
    }
    
    __device__ Fp4 operator-() const {
        return Fp4() - *this;
    }
    
    __device__ Fp4& operator+=(Fp4 rhs) {
        *this = *this + rhs;
        return *this;
    }
    
    __device__ Fp4& operator-=(Fp4 rhs) {
        *this = *this - rhs;
        return *this;
    }
    
    // ========================================================================
    // Multiplication
    // ========================================================================
    
    __device__ Fp4 operator*(Fp rhs) const {
        return Fp4(
            elems[0] * rhs,
            elems[1] * rhs,
            elems[2] * rhs,
            elems[3] * rhs
        );
    }
    
    __device__ Fp4& operator*=(Fp rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Full extension field multiplication (schoolbook)
    /// Uses reduction α^4 = 11
    __device__ Fp4 operator*(Fp4 rhs) const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3];
        const Fp b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2], b3 = rhs.elems[3];
        
        // Schoolbook convolution: t[k] = sum_{i+j=k} a[i]*b[j]
        Fp t0 = a0 * b0;
        Fp t1 = a0 * b1 + a1 * b0;
        Fp t2 = a0 * b2 + a1 * b1 + a2 * b0;
        Fp t3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        Fp t4 = a1 * b3 + a2 * b2 + a3 * b1;
        Fp t5 = a2 * b3 + a3 * b2;
        Fp t6 = a3 * b3;
        
        // Reduction: α^4 = 11, α^5 = 11α, α^6 = 11α²
        // c0 = t0 + 11*t4
        // c1 = t1 + 11*t5
        // c2 = t2 + 11*t6
        // c3 = t3
        Fp w(W);
        return Fp4(
            t0 + w * t4,
            t1 + w * t5,
            t2 + w * t6,
            t3
        );
    }
    
    __device__ Fp4& operator*=(Fp4 rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Squaring (optimized using symmetry)
    __device__ Fp4 square() const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3];
        
        // Squares
        Fp a0sq = a0 * a0;
        Fp a1sq = a1 * a1;
        Fp a2sq = a2 * a2;
        Fp a3sq = a3 * a3;
        
        // Cross products (each appears twice)
        Fp a0a1 = a0 * a1;
        Fp a0a2 = a0 * a2;
        Fp a0a3 = a0 * a3;
        Fp a1a2 = a1 * a2;
        Fp a1a3 = a1 * a3;
        Fp a2a3 = a2 * a3;
        
        // t0 = a0², t1 = 2*a0*a1, t2 = 2*a0*a2 + a1², ...
        Fp t0 = a0sq;
        Fp t1 = a0a1 + a0a1;
        Fp t2 = a0a2 + a0a2 + a1sq;
        Fp t3 = a0a3 + a0a3 + a1a2 + a1a2;
        Fp t4 = a1a3 + a1a3 + a2sq;
        Fp t5 = a2a3 + a2a3;
        Fp t6 = a3sq;
        
        Fp w(W);
        return Fp4(
            t0 + w * t4,
            t1 + w * t5,
            t2 + w * t6,
            t3
        );
    }
    
    // ========================================================================
    // Equality
    // ========================================================================
    
    __device__ bool operator==(Fp4 rhs) const {
        return elems[0] == rhs.elems[0] && 
               elems[1] == rhs.elems[1] && 
               elems[2] == rhs.elems[2] &&
               elems[3] == rhs.elems[3];
    }
    
    __device__ bool operator!=(Fp4 rhs) const {
        return !(*this == rhs);
    }
};

__device__ inline Fp4 operator*(Fp a, Fp4 b) { return b * a; }

/// Power function
__device__ inline Fp4 pow(Fp4 base, uint32_t exp) {
    Fp4 result = Fp4::one();
    while (exp > 0) {
        if (exp & 1) {
            result = result * base;
        }
        base = base.square();
        exp >>= 1;
    }
    return result;
}

/// Inversion using Gaussian elimination on the 4×4 multiplication matrix
/// Matrix for a = a0 + a1*α + a2*α² + a3*α³ with α^4 = W:
/// Col 0: [a0, a1, a2, a3]
/// Col 1: [W*a3, a0, a1, a2]
/// Col 2: [W*a2, W*a3, a0, a1]
/// Col 3: [W*a1, W*a2, W*a3, a0]
__device__ inline Fp4 inv(Fp4 x) {
    if (x == Fp4::zero()) return Fp4::zero();
    
    const Fp* a = x.elems;
    Fp w(Fp4::W);
    
    // Build augmented matrix [M | I]
    Fp m[4][5];
    
    // Col 0: [a0, a1, a2, a3]
    m[0][0] = a[0]; m[1][0] = a[1]; m[2][0] = a[2]; m[3][0] = a[3];
    
    // Col 1: [W*a3, a0, a1, a2]
    m[0][1] = w * a[3]; m[1][1] = a[0]; m[2][1] = a[1]; m[3][1] = a[2];
    
    // Col 2: [W*a2, W*a3, a0, a1]
    m[0][2] = w * a[2]; m[1][2] = w * a[3]; m[2][2] = a[0]; m[3][2] = a[1];
    
    // Col 3: [W*a1, W*a2, W*a3, a0]
    m[0][3] = w * a[1]; m[1][3] = w * a[2]; m[2][3] = w * a[3]; m[3][3] = a[0];
    
    // Augmented column (e0 = [1, 0, 0, 0])
    m[0][4] = Fp::one();
    m[1][4] = Fp::zero();
    m[2][4] = Fp::zero();
    m[3][4] = Fp::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 4; col++) {
        // Find non-zero pivot
        int pivot_row = col;
        while (pivot_row < 4 && m[pivot_row][col] == Fp::zero()) {
            pivot_row++;
        }
        if (pivot_row >= 4) return Fp4::zero();  // Singular
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 5; j++) {
                Fp tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row
        Fp pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 5; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 4; row++) {
            if (row != col) {
                Fp factor = m[row][col];
                for (int j = col; j < 5; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    return Fp4(m[0][4], m[1][4], m[2][4], m[3][4]);
}

static_assert(sizeof(Fp4) == 16, "Fp4 must be 16 bytes");
