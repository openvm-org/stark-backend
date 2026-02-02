/*
 * Fp5 - Quintic Extension of Baby Bear Field
 * 
 * Defines Fp5, a finite field F_p^5, based on Fp via the irreducible polynomial x^5 - 2.
 * 
 * Field size: p^5 ≈ 2^155, provides ~155 bits of security.
 * 
 * Element representation: a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
 * where each ai is a Baby Bear element and x^5 = 2.
 */

#pragma once

#include "fp.h"

/// Fp5 is a degree-5 extension of the Baby Bear field.
/// Elements are represented as polynomials in F_p[x] / (x^5 - 2).
struct Fp5 {
    /// Coefficients: elems[0] + elems[1]*x + elems[2]*x^2 + elems[3]*x^3 + elems[4]*x^4
    Fp elems[5];
    
    /// The non-residue W such that x^5 - W is irreducible
    /// W = 2 is the smallest valid choice for Baby Bear
    static constexpr uint32_t W = 2;
    
    /// Default constructor - zero element
    __device__ Fp5() : elems{Fp(0), Fp(0), Fp(0), Fp(0), Fp(0)} {}
    
    /// Construct from a single Fp (embed base field)
    __device__ explicit Fp5(Fp x) : elems{x, Fp(0), Fp(0), Fp(0), Fp(0)} {}
    
    /// Construct from uint32_t
    __device__ explicit Fp5(uint32_t x) : elems{Fp(x), Fp(0), Fp(0), Fp(0), Fp(0)} {}
    
    /// Construct from all 5 coefficients
    __device__ Fp5(Fp a0, Fp a1, Fp a2, Fp a3, Fp a4) 
        : elems{a0, a1, a2, a3, a4} {}
    
    /// Zero element
    __device__ static Fp5 zero() { return Fp5(); }
    
    /// One element
    __device__ static Fp5 one() { return Fp5(Fp::one()); }
    
    /// Access coefficient
    __device__ Fp& operator[](int i) { return elems[i]; }
    __device__ const Fp& operator[](int i) const { return elems[i]; }
    
    /// Get constant part (coefficient of x^0)
    __device__ Fp constPart() const { return elems[0]; }
    
    // ========================================================================
    // Addition / Subtraction (component-wise)
    // ========================================================================
    
    __device__ Fp5 operator+(Fp5 rhs) const {
        return Fp5(
            elems[0] + rhs.elems[0],
            elems[1] + rhs.elems[1],
            elems[2] + rhs.elems[2],
            elems[3] + rhs.elems[3],
            elems[4] + rhs.elems[4]
        );
    }
    
    __device__ Fp5 operator-(Fp5 rhs) const {
        return Fp5(
            elems[0] - rhs.elems[0],
            elems[1] - rhs.elems[1],
            elems[2] - rhs.elems[2],
            elems[3] - rhs.elems[3],
            elems[4] - rhs.elems[4]
        );
    }
    
    __device__ Fp5 operator-() const {
        return Fp5() - *this;
    }
    
    __device__ Fp5& operator+=(Fp5 rhs) {
        *this = *this + rhs;
        return *this;
    }
    
    __device__ Fp5& operator-=(Fp5 rhs) {
        *this = *this - rhs;
        return *this;
    }
    
    // ========================================================================
    // Multiplication
    // ========================================================================
    
    /// Multiply by scalar (base field element)
    __device__ Fp5 operator*(Fp rhs) const {
        return Fp5(
            elems[0] * rhs,
            elems[1] * rhs,
            elems[2] * rhs,
            elems[3] * rhs,
            elems[4] * rhs
        );
    }
    
    __device__ Fp5& operator*=(Fp rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Full extension field multiplication
    /// Uses schoolbook multiplication + reduction mod (x^5 - 2)
    /// Optimized to minimize register pressure by combining t[i] and t[i+5] directly.
    __device__ Fp5 operator*(Fp5 rhs) const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        const Fp b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2], b3 = rhs.elems[3], b4 = rhs.elems[4];
        
        // Result c[i] = t[i] + 2*t[i+5] where t[k] = sum_{i+j=k} a[i]*b[j]
        // Compute each result coefficient directly, fusing t[i] and 2*t[i+5]
        
        // c0 = t0 + 2*t5 = a0*b0 + 2*(a1*b4 + a2*b3 + a3*b2 + a4*b1)
        Fp c0 = a0 * b0;
        Fp t5_half = a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1;
        c0 = c0 + t5_half + t5_half;
        
        // c1 = t1 + 2*t6 = (a0*b1 + a1*b0) + 2*(a2*b4 + a3*b3 + a4*b2)
        Fp c1 = a0 * b1 + a1 * b0;
        Fp t6_half = a2 * b4 + a3 * b3 + a4 * b2;
        c1 = c1 + t6_half + t6_half;
        
        // c2 = t2 + 2*t7 = (a0*b2 + a1*b1 + a2*b0) + 2*(a3*b4 + a4*b3)
        Fp c2 = a0 * b2 + a1 * b1 + a2 * b0;
        Fp t7_half = a3 * b4 + a4 * b3;
        c2 = c2 + t7_half + t7_half;
        
        // c3 = t3 + 2*t8 = (a0*b3 + a1*b2 + a2*b1 + a3*b0) + 2*(a4*b4)
        Fp c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        Fp a4b4 = a4 * b4;
        c3 = c3 + a4b4 + a4b4;
        
        // c4 = t4 = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
        Fp c4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
        
        return Fp5(c0, c1, c2, c3, c4);
    }
    
    __device__ Fp5& operator*=(Fp5 rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Squaring (optimized using symmetry: a[i]*a[j] appears twice for i != j)
    /// Uses 15 multiplications instead of 25
    __device__ Fp5 square() const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        
        // Squares (5 muls)
        Fp a0sq = a0 * a0;
        Fp a1sq = a1 * a1;
        Fp a2sq = a2 * a2;
        Fp a3sq = a3 * a3;
        Fp a4sq = a4 * a4;
        
        // Cross products that will be doubled (10 muls, each used twice)
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
        
        // Result: c[i] = t[i] + 2*t[i+5]
        // where t[k] = sum_{i+j=k} a[i]*a[j]
        
        // c0 = t0 + 2*t5 = a0^2 + 2*(2*(a1*a4 + a2*a3)) = a0^2 + 4*(a1a4 + a2a3)
        Fp sum14_23 = a1a4 + a2a3;
        Fp c0 = a0sq + sum14_23 + sum14_23 + sum14_23 + sum14_23;
        
        // c1 = t1 + 2*t6 = 2*a0a1 + 2*(2*a2a4 + a3^2)
        Fp c1 = a0a1 + a0a1 + a2a4 + a2a4 + a2a4 + a2a4 + a3sq + a3sq;
        
        // c2 = t2 + 2*t7 = (2*a0a2 + a1^2) + 2*(2*a3a4) = 2*a0a2 + a1^2 + 4*a3a4
        Fp c2 = a0a2 + a0a2 + a1sq + a3a4 + a3a4 + a3a4 + a3a4;
        
        // c3 = t3 + 2*t8 = 2*(a0a3 + a1a2) + 2*a4^2
        Fp sum03_12 = a0a3 + a1a2;
        Fp c3 = sum03_12 + sum03_12 + a4sq + a4sq;
        
        // c4 = t4 = 2*(a0a4 + a1a3) + a2^2
        Fp c4 = a0a4 + a0a4 + a1a3 + a1a3 + a2sq;
        
        return Fp5(c0, c1, c2, c3, c4);
    }
    
    // ========================================================================
    // Equality
    // ========================================================================
    
    __device__ bool operator==(Fp5 rhs) const {
        return elems[0] == rhs.elems[0] && 
               elems[1] == rhs.elems[1] && 
               elems[2] == rhs.elems[2] && 
               elems[3] == rhs.elems[3] && 
               elems[4] == rhs.elems[4];
    }
    
    __device__ bool operator!=(Fp5 rhs) const {
        return !(*this == rhs);
    }
};

/// Scalar * Fp5
__device__ inline Fp5 operator*(Fp a, Fp5 b) { return b * a; }

/// Power function using square-and-multiply
__device__ inline Fp5 pow(Fp5 base, uint32_t exp) {
    Fp5 result = Fp5::one();
    while (exp > 0) {
        if (exp & 1) {
            result = result * base;
        }
        base = base.square();
        exp >>= 1;
    }
    return result;
}

/// Inversion using Gaussian elimination on the multiplication matrix.
/// 
/// For a in Fp5, we solve a * b = 1 by setting up the 5x5 linear system
/// where the matrix M represents "multiply by a" and solving M * b = e0.
/// 
/// The matrix for a = a0 + a1*x + ... + a4*x^4 with x^5 = W is:
/// [a0    W*a4  W*a3  W*a2  W*a1]
/// [a1    a0    W*a4  W*a3  W*a2]
/// [a2    a1    a0    W*a4  W*a3]
/// [a3    a2    a1    a0    W*a4]
/// [a4    a3    a2    a1    a0  ]
/// 
/// This is O(n^3) = O(125) Fp operations, much faster than Fermat's ~4250 Fp ops.
__device__ inline Fp5 inv(Fp5 x) {
    // Handle zero case
    if (x == Fp5::zero()) {
        return Fp5::zero();
    }
    
    const Fp W_val(Fp5::W);
    const Fp* a = x.elems;
    
    // Build augmented matrix [M | I] for Gauss-Jordan elimination
    // M is the multiplication matrix, I is identity
    // We'll solve M * result = e0 = (1, 0, 0, 0, 0)
    
    // Matrix storage: m[row][col], augmented with result column
    Fp m[5][6];
    
    // Row 0: [a0, W*a4, W*a3, W*a2, W*a1 | 1]
    m[0][0] = a[0]; m[0][1] = W_val * a[4]; m[0][2] = W_val * a[3]; 
    m[0][3] = W_val * a[2]; m[0][4] = W_val * a[1]; m[0][5] = Fp::one();
    
    // Row 1: [a1, a0, W*a4, W*a3, W*a2 | 0]
    m[1][0] = a[1]; m[1][1] = a[0]; m[1][2] = W_val * a[4];
    m[1][3] = W_val * a[3]; m[1][4] = W_val * a[2]; m[1][5] = Fp::zero();
    
    // Row 2: [a2, a1, a0, W*a4, W*a3 | 0]
    m[2][0] = a[2]; m[2][1] = a[1]; m[2][2] = a[0];
    m[2][3] = W_val * a[4]; m[2][4] = W_val * a[3]; m[2][5] = Fp::zero();
    
    // Row 3: [a3, a2, a1, a0, W*a4 | 0]
    m[3][0] = a[3]; m[3][1] = a[2]; m[3][2] = a[1];
    m[3][3] = a[0]; m[3][4] = W_val * a[4]; m[3][5] = Fp::zero();
    
    // Row 4: [a4, a3, a2, a1, a0 | 0]
    m[4][0] = a[4]; m[4][1] = a[3]; m[4][2] = a[2];
    m[4][3] = a[1]; m[4][4] = a[0]; m[4][5] = Fp::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 5; col++) {
        // Find pivot (largest element in column)
        int pivot_row = col;
        for (int row = col + 1; row < 5; row++) {
            if (m[row][col].asRaw() > m[pivot_row][col].asRaw()) {
                pivot_row = row;
            }
        }
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 6; j++) {
                Fp tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row to make pivot = 1
        Fp pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 6; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 5; row++) {
            if (row != col) {
                Fp factor = m[row][col];
                for (int j = col; j < 6; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    // Result is in the augmented column
    return Fp5(m[0][5], m[1][5], m[2][5], m[3][5], m[4][5]);
}

static_assert(sizeof(Fp5) == 20, "Fp5 must be 20 bytes");
