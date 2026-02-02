/*
 * Fp6 - Sextic Extension of Baby Bear Field
 * 
 * Constructed as Fp[x] / (x^6 - 31) where 31 is neither a quadratic
 * nor cubic residue in Baby Bear.
 * 
 * W = 31 is chosen because:
 * 1. It's the smallest valid W with nice computational properties
 * 2. 31 = 2^5 - 1, so 31*x = (x << 5) - x (1 subtraction)
 * 3. 31 is the multiplicative generator of Baby Bear, a conventional constant
 * 
 * Element representation: a = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
 * Stored as array [a0, a1, a2, a3, a4, a5] where each ai is an Fp element.
 * Memory: 6 * 4 = 24 bytes per element.
 */

#pragma once

#include "fp.h"

struct Fp6 {
    Fp elems[6];
    
    // W = 31 = 2^5 - 1 (multiplicative generator of Baby Bear)
    // x^6 = 31 in this extension
    static constexpr uint32_t W = 31;
    
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /// Default constructor (zero)
    __device__ Fp6() : elems{Fp(), Fp(), Fp(), Fp(), Fp(), Fp()} {}
    
    /// Construct from base field element (embed Fp into Fp6)
    __device__ explicit Fp6(Fp x) : elems{x, Fp(), Fp(), Fp(), Fp(), Fp()} {}
    
    /// Construct from raw u32 (will be converted to Fp)
    __device__ explicit Fp6(uint32_t x) : elems{Fp(x), Fp(), Fp(), Fp(), Fp(), Fp()} {}
    
    /// Construct from all 6 coefficients
    __device__ Fp6(Fp a0, Fp a1, Fp a2, Fp a3, Fp a4, Fp a5) 
        : elems{a0, a1, a2, a3, a4, a5} {}
    
    /// Zero element
    __device__ static Fp6 zero() { return Fp6(); }
    
    /// One element (multiplicative identity)
    __device__ static Fp6 one() { return Fp6(Fp::one()); }
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    __device__ Fp& operator[](int i) { return elems[i]; }
    __device__ const Fp& operator[](int i) const { return elems[i]; }
    
    /// Get the constant part (coefficient of x^0)
    __device__ Fp constPart() const { return elems[0]; }
    
    // ========================================================================
    // Addition / Subtraction
    // ========================================================================
    
    __device__ Fp6 operator+(Fp6 rhs) const {
        return Fp6(
            elems[0] + rhs.elems[0],
            elems[1] + rhs.elems[1],
            elems[2] + rhs.elems[2],
            elems[3] + rhs.elems[3],
            elems[4] + rhs.elems[4],
            elems[5] + rhs.elems[5]
        );
    }
    
    __device__ Fp6 operator-(Fp6 rhs) const {
        return Fp6(
            elems[0] - rhs.elems[0],
            elems[1] - rhs.elems[1],
            elems[2] - rhs.elems[2],
            elems[3] - rhs.elems[3],
            elems[4] - rhs.elems[4],
            elems[5] - rhs.elems[5]
        );
    }
    
    __device__ Fp6 operator-() const {
        return Fp6(
            -elems[0], -elems[1], -elems[2], 
            -elems[3], -elems[4], -elems[5]
        );
    }
    
    __device__ Fp6& operator+=(Fp6 rhs) {
        *this = *this + rhs;
        return *this;
    }
    
    __device__ Fp6& operator-=(Fp6 rhs) {
        *this = *this - rhs;
        return *this;
    }
    
    // ========================================================================
    // Multiplication
    // ========================================================================
    
    /// Multiply by scalar (base field element)
    __device__ Fp6 operator*(Fp rhs) const {
        return Fp6(
            elems[0] * rhs,
            elems[1] * rhs,
            elems[2] * rhs,
            elems[3] * rhs,
            elems[4] * rhs,
            elems[5] * rhs
        );
    }
    
    __device__ Fp6& operator*=(Fp rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Helper: multiply by W = 31 = 2^5 - 1 using subtraction
    __device__ static Fp mulByW(Fp x) {
        // 31 * x = 32x - x = (x << 5) - x
        // In Montgomery form, shifting doesn't work directly, so we use Fp(31) * x
        // But we can optimize: 31x = 32x - x where 32x = x + x + x + ... (5 doublings) - x
        // Actually simpler: just multiply by Fp(31)
        // For maximum performance, we precompute Fp(31) as a constant
        return Fp(W) * x;
    }
    
    /// Full extension field multiplication
    /// Uses schoolbook multiplication + reduction mod (x^6 - 31)
    __device__ Fp6 operator*(Fp6 rhs) const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2];
        const Fp a3 = elems[3], a4 = elems[4], a5 = elems[5];
        const Fp b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2];
        const Fp b3 = rhs.elems[3], b4 = rhs.elems[4], b5 = rhs.elems[5];
        
        // Result c[i] = t[i] + W*t[i+6] where t[k] = sum_{i+j=k} a[i]*b[j]
        // W = 31
        
        // c0 = t0 + W*t6 = a0*b0 + W*(a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1)
        Fp t6 = a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1;
        Fp c0 = a0*b0 + mulByW(t6);
        
        // c1 = t1 + W*t7 = (a0*b1 + a1*b0) + W*(a2*b5 + a3*b4 + a4*b3 + a5*b2)
        Fp t7 = a2*b5 + a3*b4 + a4*b3 + a5*b2;
        Fp c1 = a0*b1 + a1*b0 + mulByW(t7);
        
        // c2 = t2 + W*t8 = (a0*b2 + a1*b1 + a2*b0) + W*(a3*b5 + a4*b4 + a5*b3)
        Fp t8 = a3*b5 + a4*b4 + a5*b3;
        Fp c2 = a0*b2 + a1*b1 + a2*b0 + mulByW(t8);
        
        // c3 = t3 + W*t9 = (a0*b3 + a1*b2 + a2*b1 + a3*b0) + W*(a4*b5 + a5*b4)
        Fp t9 = a4*b5 + a5*b4;
        Fp c3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 + mulByW(t9);
        
        // c4 = t4 + W*t10 = (a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0) + W*(a5*b5)
        Fp c4 = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + mulByW(a5*b5);
        
        // c5 = t5 = a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0
        Fp c5 = a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0;
        
        return Fp6(c0, c1, c2, c3, c4, c5);
    }
    
    __device__ Fp6& operator*=(Fp6 rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Squaring (optimized using symmetry)
    /// Uses 21 multiplications instead of 36
    __device__ Fp6 square() const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2];
        const Fp a3 = elems[3], a4 = elems[4], a5 = elems[5];
        
        // Squares (6 muls)
        Fp a0sq = a0 * a0;
        Fp a1sq = a1 * a1;
        Fp a2sq = a2 * a2;
        Fp a3sq = a3 * a3;
        Fp a4sq = a4 * a4;
        Fp a5sq = a5 * a5;
        
        // Cross products (15 muls, each used twice)
        Fp a0a1 = a0 * a1;
        Fp a0a2 = a0 * a2;
        Fp a0a3 = a0 * a3;
        Fp a0a4 = a0 * a4;
        Fp a0a5 = a0 * a5;
        Fp a1a2 = a1 * a2;
        Fp a1a3 = a1 * a3;
        Fp a1a4 = a1 * a4;
        Fp a1a5 = a1 * a5;
        Fp a2a3 = a2 * a3;
        Fp a2a4 = a2 * a4;
        Fp a2a5 = a2 * a5;
        Fp a3a4 = a3 * a4;
        Fp a3a5 = a3 * a5;
        Fp a4a5 = a4 * a5;
        
        // t values (before reduction):
        // t0 = a0^2
        // t1 = 2*a0*a1
        // t2 = 2*a0*a2 + a1^2
        // t3 = 2*(a0*a3 + a1*a2)
        // t4 = 2*(a0*a4 + a1*a3) + a2^2
        // t5 = 2*(a0*a5 + a1*a4 + a2*a3)
        // t6 = 2*(a1*a5 + a2*a4) + a3^2
        // t7 = 2*(a2*a5 + a3*a4)
        // t8 = 2*a3*a5 + a4^2
        // t9 = 2*a4*a5
        // t10 = a5^2
        
        // Result: c[i] = t[i] + W*t[i+6]
        Fp t6 = (a1a5 + a2a4) + (a1a5 + a2a4) + a3sq;
        Fp c0 = a0sq + mulByW(t6);
        
        Fp t7 = (a2a5 + a3a4) + (a2a5 + a3a4);
        Fp c1 = a0a1 + a0a1 + mulByW(t7);
        
        Fp t8 = a3a5 + a3a5 + a4sq;
        Fp c2 = a0a2 + a0a2 + a1sq + mulByW(t8);
        
        Fp t9 = a4a5 + a4a5;
        Fp c3 = (a0a3 + a1a2) + (a0a3 + a1a2) + mulByW(t9);
        
        Fp c4 = (a0a4 + a1a3) + (a0a4 + a1a3) + a2sq + mulByW(a5sq);
        
        Fp c5 = (a0a5 + a1a4 + a2a3) + (a0a5 + a1a4 + a2a3);
        
        return Fp6(c0, c1, c2, c3, c4, c5);
    }
    
    // ========================================================================
    // Equality
    // ========================================================================
    
    __device__ bool operator==(Fp6 rhs) const {
        return elems[0] == rhs.elems[0] && 
               elems[1] == rhs.elems[1] && 
               elems[2] == rhs.elems[2] &&
               elems[3] == rhs.elems[3] && 
               elems[4] == rhs.elems[4] && 
               elems[5] == rhs.elems[5];
    }
    
    __device__ bool operator!=(Fp6 rhs) const {
        return !(*this == rhs);
    }
};

/// Scalar * Fp6
__device__ inline Fp6 operator*(Fp a, Fp6 b) { return b * a; }

/// Power function using square-and-multiply
__device__ inline Fp6 pow(Fp6 base, uint32_t exp) {
    Fp6 result = Fp6::one();
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
/// For a in Fp6, we solve a * b = 1 by setting up the 6x6 linear system
/// where the matrix M represents "multiply by a" and solving M * b = e0.
/// 
/// The matrix for a = a0 + a1*x + ... + a5*x^5 with x^6 = W is:
/// [a0    W*a5  W*a4  W*a3  W*a2  W*a1]
/// [a1    a0    W*a5  W*a4  W*a3  W*a2]
/// [a2    a1    a0    W*a5  W*a4  W*a3]
/// [a3    a2    a1    a0    W*a5  W*a4]
/// [a4    a3    a2    a1    a0    W*a5]
/// [a5    a4    a3    a2    a1    a0  ]
/// 
/// This is O(n^3) = O(216) Fp operations, much faster than Fermat's ~6000+ Fp ops.
__device__ inline Fp6 inv(Fp6 x) {
    // Handle zero case
    if (x == Fp6::zero()) {
        return Fp6::zero();
    }
    
    const Fp W_val(Fp6::W);
    const Fp* a = x.elems;
    
    // Build augmented matrix [M | I] for Gauss-Jordan elimination
    // M is the multiplication matrix, I is identity
    // We'll solve M * result = e0 = (1, 0, 0, 0, 0, 0)
    
    // Matrix storage: m[row][col], augmented with result column
    Fp m[6][7];
    
    // Row 0: [a0, W*a5, W*a4, W*a3, W*a2, W*a1 | 1]
    m[0][0] = a[0]; m[0][1] = W_val * a[5]; m[0][2] = W_val * a[4];
    m[0][3] = W_val * a[3]; m[0][4] = W_val * a[2]; m[0][5] = W_val * a[1]; 
    m[0][6] = Fp::one();
    
    // Row 1: [a1, a0, W*a5, W*a4, W*a3, W*a2 | 0]
    m[1][0] = a[1]; m[1][1] = a[0]; m[1][2] = W_val * a[5];
    m[1][3] = W_val * a[4]; m[1][4] = W_val * a[3]; m[1][5] = W_val * a[2];
    m[1][6] = Fp::zero();
    
    // Row 2: [a2, a1, a0, W*a5, W*a4, W*a3 | 0]
    m[2][0] = a[2]; m[2][1] = a[1]; m[2][2] = a[0];
    m[2][3] = W_val * a[5]; m[2][4] = W_val * a[4]; m[2][5] = W_val * a[3];
    m[2][6] = Fp::zero();
    
    // Row 3: [a3, a2, a1, a0, W*a5, W*a4 | 0]
    m[3][0] = a[3]; m[3][1] = a[2]; m[3][2] = a[1];
    m[3][3] = a[0]; m[3][4] = W_val * a[5]; m[3][5] = W_val * a[4];
    m[3][6] = Fp::zero();
    
    // Row 4: [a4, a3, a2, a1, a0, W*a5 | 0]
    m[4][0] = a[4]; m[4][1] = a[3]; m[4][2] = a[2];
    m[4][3] = a[1]; m[4][4] = a[0]; m[4][5] = W_val * a[5];
    m[4][6] = Fp::zero();
    
    // Row 5: [a5, a4, a3, a2, a1, a0 | 0]
    m[5][0] = a[5]; m[5][1] = a[4]; m[5][2] = a[3];
    m[5][3] = a[2]; m[5][4] = a[1]; m[5][5] = a[0];
    m[5][6] = Fp::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 6; col++) {
        // Find pivot (largest element in column)
        int pivot_row = col;
        for (int row = col + 1; row < 6; row++) {
            if (m[row][col].asRaw() > m[pivot_row][col].asRaw()) {
                pivot_row = row;
            }
        }
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 7; j++) {
                Fp tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row to make pivot = 1
        Fp pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 7; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 6; row++) {
            if (row != col) {
                Fp factor = m[row][col];
                for (int j = col; j < 7; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    // Result is in the augmented column
    return Fp6(m[0][6], m[1][6], m[2][6], m[3][6], m[4][6], m[5][6]);
}

static_assert(sizeof(Fp6) == 24, "Fp6 must be 24 bytes");
