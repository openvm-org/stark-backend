/*
 * Kb3x2 - Sextic Extension via 3×2 Tower for KoalaBear
 * 
 * Tower construction: Kb → Kb3 → Kb6
 *   Kb3 = Kb[w] / (w³ + w + 4)  where w³ = -w - 4 (trinomial, since binomial fails)
 *   Kb6 = Kb3[z] / (z² - 3)    where z² = 3
 * 
 * Why trinomial for Kb3:
 *   KoalaBear has gcd(3, p-1) = 1, so the cube map is an automorphism.
 *   Every element is a cube → no binomial w³ = W is irreducible.
 *   w³ + w + 4 is irreducible (verified: no roots in Kb).
 * 
 * Why 3 works for quadratic step:
 *   3 is nonsquare in Kb. Since deg(Kb3/Kb) = 3 is odd, an element is a
 *   square in Kb3 iff it was a square in Kb. So 3 remains nonsquare in Kb3.
 * 
 * Element representation:
 *   A Kb6 element is: a0 + a1*z  where each ai ∈ Kb3
 *   Each Kb3 element is: b0 + b1*w + b2*w²  where bi ∈ Kb
 *   
 * Total storage: 6 Kb elements (24 bytes)
 * 
 * Reduction rules:
 *   w³ = -w - 4
 *   w⁴ = -w² - 4w
 *   z² = 3
 */

#pragma once

#include "kb.h"

// ============================================================================
// Kb3 = Kb[w] / (w³ + w + 4)
// ============================================================================

struct Kb3 {
    Kb c0, c1, c2;  // c0 + c1*w + c2*w²
    
    // Reduction: w³ = -w - 4
    
    __device__ Kb3() : c0(), c1(), c2() {}
    __device__ Kb3(Kb a0) : c0(a0), c1(), c2() {}
    __device__ Kb3(Kb a0, Kb a1, Kb a2) : c0(a0), c1(a1), c2(a2) {}
    __device__ explicit Kb3(uint32_t x) : c0(Kb(x)), c1(), c2() {}
    
    __device__ static Kb3 zero() { return Kb3(); }
    __device__ static Kb3 one() { return Kb3(Kb::one()); }
    
    __device__ Kb3 operator+(Kb3 rhs) const {
        return Kb3(c0 + rhs.c0, c1 + rhs.c1, c2 + rhs.c2);
    }
    
    __device__ Kb3 operator-(Kb3 rhs) const {
        return Kb3(c0 - rhs.c0, c1 - rhs.c1, c2 - rhs.c2);
    }
    
    __device__ Kb3 operator-() const {
        return Kb3(-c0, -c1, -c2);
    }
    
    __device__ Kb3 operator*(Kb rhs) const {
        return Kb3(c0 * rhs, c1 * rhs, c2 * rhs);
    }
    
    // (a0 + a1*w + a2*w²) * (b0 + b1*w + b2*w²) with w³ = -w - 4
    // Schoolbook then reduce using w³ = -w - 4, w⁴ = -w² - 4w
    __device__ Kb3 operator*(Kb3 rhs) const {
        Kb a0 = c0, a1 = c1, a2 = c2;
        Kb b0 = rhs.c0, b1 = rhs.c1, b2 = rhs.c2;
        
        // Schoolbook convolution
        Kb t0 = a0 * b0;
        Kb t1 = a0 * b1 + a1 * b0;
        Kb t2 = a0 * b2 + a1 * b1 + a2 * b0;
        Kb t3 = a1 * b2 + a2 * b1;
        Kb t4 = a2 * b2;
        
        // Reduction: w³ = -w - 4, w⁴ = -w² - 4w
        // t3*w³ = t3*(-w - 4) = -t3*w - 4*t3
        // t4*w⁴ = t4*(-w² - 4w) = -t4*w² - 4*t4*w
        // 
        // c0 = t0 - 4*t3
        // c1 = t1 - t3 - 4*t4
        // c2 = t2 - t4
        Kb four = Kb(4);
        return Kb3(
            t0 - four * t3,
            t1 - t3 - four * t4,
            t2 - t4
        );
    }
    
    __device__ Kb3 square() const {
        Kb a0 = c0, a1 = c1, a2 = c2;
        
        Kb a0sq = a0 * a0;
        Kb a1sq = a1 * a1;
        Kb a2sq = a2 * a2;
        Kb a0a1 = a0 * a1;
        Kb a0a2 = a0 * a2;
        Kb a1a2 = a1 * a2;
        
        Kb t0 = a0sq;
        Kb t1 = a0a1 + a0a1;
        Kb t2 = a0a2 + a0a2 + a1sq;
        Kb t3 = a1a2 + a1a2;
        Kb t4 = a2sq;
        
        Kb four = Kb(4);
        return Kb3(
            t0 - four * t3,
            t1 - t3 - four * t4,
            t2 - t4
        );
    }
    
    __device__ bool operator==(Kb3 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1 && c2 == rhs.c2;
    }
    
    __device__ bool operator!=(Kb3 rhs) const {
        return !(*this == rhs);
    }
};

// Inversion for Kb3 using Gaussian elimination
// Multiplication matrix for a = a0 + a1*w + a2*w² with w³ = -w - 4:
// Computing a*w^j for j = 0, 1, 2:
//   j=0: [a0, a1, a2]
//   j=1: [-4*a2, a0-a2, a1]
//   j=2: [-4*a1, -4*a2-a1, a0-a2]
__device__ inline Kb3 inv(Kb3 x) {
    if (x == Kb3::zero()) return Kb3::zero();
    
    Kb a0 = x.c0, a1 = x.c1, a2 = x.c2;
    Kb four = Kb(4);
    Kb neg_four = Kb::zero() - four;
    
    // Build 3x4 augmented matrix [M | e0]
    // Column j is the result of a * w^j reduced mod (w³ + w + 4)
    Kb m[3][4];
    
    // Col 0: a * 1 = [a0, a1, a2]
    m[0][0] = a0; m[1][0] = a1; m[2][0] = a2;
    
    // Col 1: a * w, where w³ = -w - 4
    // a0*w + a1*w² + a2*w³ = a0*w + a1*w² + a2*(-w - 4)
    // = -4*a2 + (a0 - a2)*w + a1*w²
    m[0][1] = neg_four * a2;
    m[1][1] = a0 - a2;
    m[2][1] = a1;
    
    // Col 2: a * w²
    // a0*w² + a1*w³ + a2*w⁴
    // = a0*w² + a1*(-w - 4) + a2*(-w² - 4w)
    // = -4*a1 + (-a1 - 4*a2)*w + (a0 - a2)*w²
    m[0][2] = neg_four * a1;
    m[1][2] = Kb::zero() - a1 - four * a2;
    m[2][2] = a0 - a2;
    
    // Augmented column (identity's first row)
    m[0][3] = Kb::one();
    m[1][3] = Kb::zero();
    m[2][3] = Kb::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 3; col++) {
        // Find non-zero pivot
        int pivot_row = col;
        while (pivot_row < 3 && m[pivot_row][col] == Kb::zero()) {
            pivot_row++;
        }
        if (pivot_row >= 3) return Kb3::zero();  // Singular
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 4; j++) {
                Kb tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row
        Kb pivot_inv = inv(m[col][col]);
        for (int j = col; j < 4; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 3; row++) {
            if (row != col) {
                Kb factor = m[row][col];
                for (int j = col; j < 4; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    return Kb3(m[0][3], m[1][3], m[2][3]);
}

// ============================================================================
// Kb3x2 = Kb3[z] / (z² - 3)
// ============================================================================

struct Kb3x2 {
    Kb3 c0, c1;  // c0 + c1*z
    
    static constexpr uint32_t W = 3;  // z² = 3
    
    __device__ Kb3x2() : c0(), c1() {}
    __device__ Kb3x2(Kb a) : c0(a), c1() {}
    __device__ Kb3x2(Kb3 a0) : c0(a0), c1() {}
    __device__ Kb3x2(Kb3 a0, Kb3 a1) : c0(a0), c1(a1) {}
    __device__ explicit Kb3x2(uint32_t x) : c0(Kb(x)), c1() {}
    
    // Construct from 6 Kb elements (for initialization from raw data)
    __device__ Kb3x2(Kb a0, Kb a1, Kb a2, Kb a3, Kb a4, Kb a5)
        : c0(a0, a1, a2), c1(a3, a4, a5) {}
    
    __device__ static Kb3x2 zero() { return Kb3x2(); }
    __device__ static Kb3x2 one() { return Kb3x2(Kb3::one()); }
    
    __device__ Kb3x2 operator+(Kb3x2 rhs) const {
        return Kb3x2(c0 + rhs.c0, c1 + rhs.c1);
    }
    
    __device__ Kb3x2 operator-(Kb3x2 rhs) const {
        return Kb3x2(c0 - rhs.c0, c1 - rhs.c1);
    }
    
    __device__ Kb3x2 operator-() const {
        return Kb3x2(-c0, -c1);
    }
    
    __device__ Kb3x2 operator*(Kb rhs) const {
        return Kb3x2(c0 * rhs, c1 * rhs);
    }
    
    // Multiply by Kb3 scalar
    __device__ Kb3x2 operator*(Kb3 rhs) const {
        return Kb3x2(c0 * rhs, c1 * rhs);
    }
    
    // Full multiplication: (a0 + a1*z) * (b0 + b1*z) with z² = 3
    // = (a0*b0 + 3*a1*b1) + (a0*b1 + a1*b0)*z
    __device__ Kb3x2 operator*(Kb3x2 rhs) const {
        Kb3 a0b0 = c0 * rhs.c0;
        Kb3 a1b1 = c1 * rhs.c1;
        Kb3 a0b1_a1b0 = c0 * rhs.c1 + c1 * rhs.c0;
        
        // 3*a1*b1 = a1*b1 + a1*b1 + a1*b1
        return Kb3x2(a0b0 + a1b1 + a1b1 + a1b1, a0b1_a1b0);
    }
    
    __device__ Kb3x2 square() const {
        // (a0 + a1*z)² = (a0² + 3*a1²) + 2*a0*a1*z
        Kb3 a0sq = c0.square();
        Kb3 a1sq = c1.square();
        Kb3 a0a1 = c0 * c1;
        
        return Kb3x2(a0sq + a1sq + a1sq + a1sq, a0a1 + a0a1);
    }
    
    __device__ bool operator==(Kb3x2 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1;
    }
    
    __device__ bool operator!=(Kb3x2 rhs) const {
        return !(*this == rhs);
    }
};

__device__ inline Kb3x2 operator*(Kb a, Kb3x2 b) { return b * a; }

// Inversion using norm to Kb3: (a + b*z)^(-1) = (a - b*z) / (a² - 3*b²)
// where a² - 3*b² is computed in Kb3, then inverted
__device__ inline Kb3x2 inv(Kb3x2 x) {
    if (x == Kb3x2::zero()) return Kb3x2::zero();
    
    Kb3 a = x.c0, b = x.c1;
    Kb3 bsq = b.square();
    Kb3 norm = a.square() - bsq - bsq - bsq;  // a² - 3*b²
    Kb3 norm_inv = inv(norm);
    
    return Kb3x2(a * norm_inv, (Kb3::zero() - b) * norm_inv);
}

static_assert(sizeof(Kb3x2) == 24, "Kb3x2 must be 24 bytes");
