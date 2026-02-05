/*
 * Kb6 - Sextic Extension of KoalaBear Field
 * 
 * Defines Kb6, a finite field F_p^6, based on Kb via the irreducible polynomial x^6 + x^3 + 1.
 * 
 * NOTE: Unlike BabyBear, KoalaBear has gcd(3, p-1) = 1, so the cube map x → x^3 is an
 * automorphism. This means every element is a cube, so no binomial x^6 - W is irreducible.
 * We use the sparse trinomial x^6 + x^3 + 1 instead.
 * 
 * Field size: p^6 ≈ 2^186, provides ~186 bits of security.
 * 
 * Element representation: a0 + a1*α + a2*α^2 + a3*α^3 + a4*α^4 + a5*α^5
 * where each ai is a KoalaBear element and α^6 = -α^3 - 1.
 * 
 * Higher powers:
 *   α^6 = -α^3 - 1
 *   α^7 = -α^4 - α
 *   α^8 = -α^5 - α^2
 *   α^9 = 1
 *   α^10 = α
 * 
 * Reduction rule for product giving t0..t10:
 *   c0 = t0 - t6 + t9
 *   c1 = t1 - t7 + t10
 *   c2 = t2 - t8
 *   c3 = t3 - t6
 *   c4 = t4 - t7
 *   c5 = t5 - t8
 * 
 * This reduction requires only additions/subtractions (no multiplications by constants),
 * making it very efficient on CUDA.
 * 
 * Multiplication matrix for inversion uses Gaussian elimination: O(216) Kb operations.
 */

#pragma once

#include "kb.h"

/// Kb6 is a degree-6 extension of the KoalaBear field.
/// Elements are represented as polynomials in F_p[x] / (x^6 + x^3 + 1).
struct Kb6 {
    /// Coefficients: elems[0] + elems[1]*α + ... + elems[5]*α^5
    Kb elems[6];
    
    /// Default constructor - zero element
    __device__ Kb6() : elems{Kb(0), Kb(0), Kb(0), Kb(0), Kb(0), Kb(0)} {}
    
    /// Construct from a single Kb (embed base field)
    __device__ explicit Kb6(Kb x) : elems{x, Kb(0), Kb(0), Kb(0), Kb(0), Kb(0)} {}
    
    /// Construct from uint32_t
    __device__ explicit Kb6(uint32_t x) : elems{Kb(x), Kb(0), Kb(0), Kb(0), Kb(0), Kb(0)} {}
    
    /// Construct from all 6 coefficients
    __device__ Kb6(Kb a0, Kb a1, Kb a2, Kb a3, Kb a4, Kb a5) 
        : elems{a0, a1, a2, a3, a4, a5} {}
    
    /// Zero element
    __device__ static Kb6 zero() { return Kb6(); }
    
    /// One element
    __device__ static Kb6 one() { return Kb6(Kb::one()); }
    
    /// Access coefficient
    __device__ Kb& operator[](int i) { return elems[i]; }
    __device__ const Kb& operator[](int i) const { return elems[i]; }
    
    /// Get constant part (coefficient of α^0)
    __device__ Kb constPart() const { return elems[0]; }

    /// Multiply two degree-2 polynomials (a0 + a1*x + a2*x^2)(b0 + b1*x + b2*x^2)
    /// Using Toom-2.5 (6 muls)
    __device__ static inline void mul_deg2(
        Kb a0, Kb a1, Kb a2, Kb b0, Kb b1, Kb b2,
        Kb& r0, Kb& r1, Kb& r2, Kb& r3, Kb& r4
    ) {
        Kb v0 = a0 * b0;
        Kb v1 = a1 * b1;
        Kb v2 = a2 * b2;
        Kb v01 = (a0 + a1) * (b0 + b1);
        Kb v12 = (a1 + a2) * (b1 + b2);
        Kb v02 = (a0 + a2) * (b0 + b2);

        r0 = v0;
        r1 = v01 - v0 - v1;
        r2 = v02 - v0 - v2 + v1;
        r3 = v12 - v1 - v2;
        r4 = v2;
    }
    
    // ========================================================================
    // Addition / Subtraction (component-wise)
    // ========================================================================
    
    __device__ Kb6 operator+(Kb6 rhs) const {
        return Kb6(
            elems[0] + rhs.elems[0],
            elems[1] + rhs.elems[1],
            elems[2] + rhs.elems[2],
            elems[3] + rhs.elems[3],
            elems[4] + rhs.elems[4],
            elems[5] + rhs.elems[5]
        );
    }
    
    __device__ Kb6 operator-(Kb6 rhs) const {
        return Kb6(
            elems[0] - rhs.elems[0],
            elems[1] - rhs.elems[1],
            elems[2] - rhs.elems[2],
            elems[3] - rhs.elems[3],
            elems[4] - rhs.elems[4],
            elems[5] - rhs.elems[5]
        );
    }
    
    __device__ Kb6 operator-() const {
        return Kb6() - *this;
    }
    
    __device__ Kb6& operator+=(Kb6 rhs) {
        *this = *this + rhs;
        return *this;
    }
    
    __device__ Kb6& operator-=(Kb6 rhs) {
        *this = *this - rhs;
        return *this;
    }
    
    // ========================================================================
    // Multiplication
    // ========================================================================
    
    /// Multiply by scalar (base field element)
    __device__ Kb6 operator*(Kb rhs) const {
        return Kb6(
            elems[0] * rhs,
            elems[1] * rhs,
            elems[2] * rhs,
            elems[3] * rhs,
            elems[4] * rhs,
            elems[5] * rhs
        );
    }
    
    __device__ Kb6& operator*=(Kb rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Full extension field multiplication
    /// Uses Karatsuba split (3+3) + reduction mod (x^6 + x^3 + 1)
    /// 
    /// Reduction: α^6 = -α^3 - 1, α^7 = -α^4 - α, α^8 = -α^5 - α^2, α^9 = 1, α^10 = α
    /// For product t0..t10:
    ///   c0 = t0 - t6 + t9
    ///   c1 = t1 - t7 + t10
    ///   c2 = t2 - t8
    ///   c3 = t3 - t6
    ///   c4 = t4 - t7
    ///   c5 = t5 - t8
    __device__ Kb6 operator*(Kb6 rhs) const {
        const Kb a0 = elems[0], a1 = elems[1], a2 = elems[2];
        const Kb a3 = elems[3], a4 = elems[4], a5 = elems[5];
        const Kb b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2];
        const Kb b3 = rhs.elems[3], b4 = rhs.elems[4], b5 = rhs.elems[5];

        // Split: A = A0 + x^3*A1, B = B0 + x^3*B1, where A0/B0 are degree-2
        Kb p0_0, p0_1, p0_2, p0_3, p0_4;
        mul_deg2(a0, a1, a2, b0, b1, b2, p0_0, p0_1, p0_2, p0_3, p0_4);

        Kb p1_0, p1_1, p1_2, p1_3, p1_4;
        mul_deg2(a3, a4, a5, b3, b4, b5, p1_0, p1_1, p1_2, p1_3, p1_4);

        Kb s0 = a0 + a3;
        Kb s1 = a1 + a4;
        Kb s2 = a2 + a5;
        Kb t0s = b0 + b3;
        Kb t1s = b1 + b4;
        Kb t2s = b2 + b5;

        Kb p2_0, p2_1, p2_2, p2_3, p2_4;
        mul_deg2(s0, s1, s2, t0s, t1s, t2s, p2_0, p2_1, p2_2, p2_3, p2_4);

        // D0 = P0 - P1
        Kb d0 = p0_0 - p1_0;
        Kb d1 = p0_1 - p1_1;
        Kb d2 = p0_2 - p1_2;
        Kb d3 = p0_3 - p1_3;
        Kb d4 = p0_4 - p1_4;

        // D1 = P2 - P0 - 2*P1
        Kb p1_0_tw = p1_0 + p1_0;
        Kb p1_1_tw = p1_1 + p1_1;
        Kb p1_2_tw = p1_2 + p1_2;
        Kb p1_3_tw = p1_3 + p1_3;
        Kb p1_4_tw = p1_4 + p1_4;
        Kb e0 = p2_0 - p0_0 - p1_0_tw;
        Kb e1 = p2_1 - p0_1 - p1_1_tw;
        Kb e2 = p2_2 - p0_2 - p1_2_tw;
        Kb e3 = p2_3 - p0_3 - p1_3_tw;
        Kb e4 = p2_4 - p0_4 - p1_4_tw;

        // Reduce: x^6 = -x^3 - 1, x^7 = -x^4 - x
        Kb c0 = d0 - e3;
        Kb c1 = d1 - e4;
        Kb c2 = d2;
        Kb c3 = d3 + e0 - e3;
        Kb c4 = d4 + e1 - e4;
        Kb c5 = e2;

        return Kb6(c0, c1, c2, c3, c4, c5);
    }
    
    __device__ Kb6& operator*=(Kb6 rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Squaring (Karatsuba split 3+3)
    __device__ Kb6 square() const {
        const Kb a0 = elems[0], a1 = elems[1], a2 = elems[2];
        const Kb a3 = elems[3], a4 = elems[4], a5 = elems[5];

        Kb p0_0, p0_1, p0_2, p0_3, p0_4;
        mul_deg2(a0, a1, a2, a0, a1, a2, p0_0, p0_1, p0_2, p0_3, p0_4);

        Kb p1_0, p1_1, p1_2, p1_3, p1_4;
        mul_deg2(a3, a4, a5, a3, a4, a5, p1_0, p1_1, p1_2, p1_3, p1_4);

        Kb s0 = a0 + a3;
        Kb s1 = a1 + a4;
        Kb s2 = a2 + a5;

        Kb p2_0, p2_1, p2_2, p2_3, p2_4;
        mul_deg2(s0, s1, s2, s0, s1, s2, p2_0, p2_1, p2_2, p2_3, p2_4);

        // D0 = P0 - P1
        Kb d0 = p0_0 - p1_0;
        Kb d1 = p0_1 - p1_1;
        Kb d2 = p0_2 - p1_2;
        Kb d3 = p0_3 - p1_3;
        Kb d4 = p0_4 - p1_4;

        // D1 = P2 - P0 - 2*P1
        Kb p1_0_tw = p1_0 + p1_0;
        Kb p1_1_tw = p1_1 + p1_1;
        Kb p1_2_tw = p1_2 + p1_2;
        Kb p1_3_tw = p1_3 + p1_3;
        Kb p1_4_tw = p1_4 + p1_4;
        Kb e0 = p2_0 - p0_0 - p1_0_tw;
        Kb e1 = p2_1 - p0_1 - p1_1_tw;
        Kb e2 = p2_2 - p0_2 - p1_2_tw;
        Kb e3 = p2_3 - p0_3 - p1_3_tw;
        Kb e4 = p2_4 - p0_4 - p1_4_tw;

        // Reduce: x^6 = -x^3 - 1, x^7 = -x^4 - x
        Kb c0 = d0 - e3;
        Kb c1 = d1 - e4;
        Kb c2 = d2;
        Kb c3 = d3 + e0 - e3;
        Kb c4 = d4 + e1 - e4;
        Kb c5 = e2;

        return Kb6(c0, c1, c2, c3, c4, c5);
    }
    
    // ========================================================================
    // Equality
    // ========================================================================
    
    __device__ bool operator==(Kb6 rhs) const {
        return elems[0] == rhs.elems[0] && 
               elems[1] == rhs.elems[1] && 
               elems[2] == rhs.elems[2] && 
               elems[3] == rhs.elems[3] && 
               elems[4] == rhs.elems[4] &&
               elems[5] == rhs.elems[5];
    }
    
    __device__ bool operator!=(Kb6 rhs) const {
        return !(*this == rhs);
    }
};

/// Scalar * Kb6
__device__ inline Kb6 operator*(Kb a, Kb6 b) { return b * a; }

/// Power function using square-and-multiply
__device__ inline Kb6 pow(Kb6 base, uint32_t exp) {
    Kb6 result = Kb6::one();
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
/// For trinomial x^6 + x^3 + 1, the multiplication matrix is derived from
/// computing a(x) * x^j mod f(x) for j = 0..5.
/// 
/// The matrix columns (coefficients of a*α^j) are:
/// Col 0: [a0, a1, a2, a3, a4, a5]
/// Col 1: [-a5, a0, a1, a2-a5, a3, a4]
/// Col 2: [-a4, -a5, a0, a1-a4, a2-a5, a3]
/// Col 3: [-a3, -a4, -a5, a0-a3, a1-a4, a2-a5]
/// Col 4: [a5-a2, -a3, -a4, -a2, a0-a3, a1-a4]
/// Col 5: [a4-a1, a5-a2, -a3, -a1, -a2, a0-a3]
__device__ inline Kb6 inv(Kb6 x) {
    // Handle zero case
    if (x == Kb6::zero()) {
        return Kb6::zero();
    }
    
    const Kb* a = x.elems;
    
    // Build augmented matrix [M | I] for Gauss-Jordan elimination
    Kb m[6][7];
    
    // Precompute some terms
    Kb neg_a1 = Kb::zero() - a[1];
    Kb neg_a2 = Kb::zero() - a[2];
    Kb neg_a3 = Kb::zero() - a[3];
    Kb neg_a4 = Kb::zero() - a[4];
    Kb neg_a5 = Kb::zero() - a[5];
    Kb a0_m_a3 = a[0] - a[3];
    Kb a1_m_a4 = a[1] - a[4];
    Kb a2_m_a5 = a[2] - a[5];
    Kb a4_m_a1 = a[4] - a[1];
    Kb a5_m_a2 = a[5] - a[2];
    
    // Col 0: [a0, a1, a2, a3, a4, a5]
    m[0][0] = a[0]; m[1][0] = a[1]; m[2][0] = a[2];
    m[3][0] = a[3]; m[4][0] = a[4]; m[5][0] = a[5];
    
    // Col 1: [-a5, a0, a1, a2-a5, a3, a4]
    m[0][1] = neg_a5; m[1][1] = a[0]; m[2][1] = a[1];
    m[3][1] = a2_m_a5; m[4][1] = a[3]; m[5][1] = a[4];
    
    // Col 2: [-a4, -a5, a0, a1-a4, a2-a5, a3]
    m[0][2] = neg_a4; m[1][2] = neg_a5; m[2][2] = a[0];
    m[3][2] = a1_m_a4; m[4][2] = a2_m_a5; m[5][2] = a[3];
    
    // Col 3: [-a3, -a4, -a5, a0-a3, a1-a4, a2-a5]
    m[0][3] = neg_a3; m[1][3] = neg_a4; m[2][3] = neg_a5;
    m[3][3] = a0_m_a3; m[4][3] = a1_m_a4; m[5][3] = a2_m_a5;
    
    // Col 4: [a5-a2, -a3, -a4, -a2, a0-a3, a1-a4]
    m[0][4] = a5_m_a2; m[1][4] = neg_a3; m[2][4] = neg_a4;
    m[3][4] = neg_a2; m[4][4] = a0_m_a3; m[5][4] = a1_m_a4;
    
    // Col 5: [a4-a1, a5-a2, -a3, -a1, -a2, a0-a3]
    m[0][5] = a4_m_a1; m[1][5] = a5_m_a2; m[2][5] = neg_a3;
    m[3][5] = neg_a1; m[4][5] = neg_a2; m[5][5] = a0_m_a3;
    
    // Augmented column (e0 = [1, 0, 0, 0, 0, 0])
    m[0][6] = Kb::one();
    m[1][6] = Kb::zero();
    m[2][6] = Kb::zero();
    m[3][6] = Kb::zero();
    m[4][6] = Kb::zero();
    m[5][6] = Kb::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 6; col++) {
        // Find first non-zero pivot
        int pivot_row = col;
        while (pivot_row < 6 && m[pivot_row][col] == Kb::zero()) {
            pivot_row++;
        }
        
        if (pivot_row >= 6) {
            return Kb6::zero();  // Singular (shouldn't happen)
        }
        
        // Find largest raw value for stability
        for (int row = pivot_row + 1; row < 6; row++) {
            if (m[row][col] != Kb::zero() && m[row][col].asRaw() > m[pivot_row][col].asRaw()) {
                pivot_row = row;
            }
        }
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 7; j++) {
                Kb tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row
        Kb pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 7; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 6; row++) {
            if (row != col) {
                Kb factor = m[row][col];
                for (int j = col; j < 7; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    // Result is in the augmented column
    return Kb6(m[0][6], m[1][6], m[2][6], m[3][6], m[4][6], m[5][6]);
}

static_assert(sizeof(Kb6) == 24, "Kb6 must be 24 bytes");
