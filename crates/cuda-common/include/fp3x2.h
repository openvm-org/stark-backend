/*
 * Fp3x2 - Sextic Extension via 3×2 Tower
 * 
 * Tower construction: Fp → Fp3 → Fp6
 *   Fp3 = Fp[u] / (u³ - 2)
 *   Fp6 = Fp3[v] / (v² - 11)
 * 
 * Element representation:
 *   An Fp6 element is: a0 + a1*v  where each ai ∈ Fp3
 *   Each Fp3 element is: b0 + b1*u + b2*u²  where bi ∈ Fp
 *   
 * Total storage: 6 Fp elements (24 bytes)
 * 
 * Constants:
 *   W3 = 2  (for Fp3: u³ = 2)
 *   W2 = 11 (for Fp6: v² = 11)
 */

#pragma once

#include "fp.h"

// ============================================================================
// Fp3 = Fp[u] / (u³ - 2)
// ============================================================================

struct Fp3 {
    Fp c0, c1, c2;  // c0 + c1*u + c2*u²
    
    static constexpr uint32_t W = 2;  // u³ = 2
    
    __device__ Fp3() : c0(), c1(), c2() {}
    __device__ Fp3(Fp a0) : c0(a0), c1(), c2() {}
    __device__ Fp3(Fp a0, Fp a1, Fp a2) : c0(a0), c1(a1), c2(a2) {}
    __device__ explicit Fp3(uint32_t x) : c0(Fp(x)), c1(), c2() {}
    
    __device__ static Fp3 zero() { return Fp3(); }
    __device__ static Fp3 one() { return Fp3(Fp::one()); }
    
    __device__ Fp3 operator+(Fp3 rhs) const {
        return Fp3(c0 + rhs.c0, c1 + rhs.c1, c2 + rhs.c2);
    }
    
    __device__ Fp3 operator-(Fp3 rhs) const {
        return Fp3(c0 - rhs.c0, c1 - rhs.c1, c2 - rhs.c2);
    }
    
    __device__ Fp3 operator-() const {
        return Fp3(-c0, -c1, -c2);
    }
    
    __device__ Fp3 operator*(Fp rhs) const {
        return Fp3(c0 * rhs, c1 * rhs, c2 * rhs);
    }
    
    // (a0 + a1*u + a2*u²) * (b0 + b1*u + b2*u²) with u³ = 2
    __device__ Fp3 operator*(Fp3 rhs) const {
        Fp a0 = c0, a1 = c1, a2 = c2;
        Fp b0 = rhs.c0, b1 = rhs.c1, b2 = rhs.c2;
        
        // Schoolbook
        Fp t0 = a0 * b0;
        Fp t1 = a0 * b1 + a1 * b0;
        Fp t2 = a0 * b2 + a1 * b1 + a2 * b0;
        Fp t3 = a1 * b2 + a2 * b1;
        Fp t4 = a2 * b2;
        
        // Reduction: u³ = 2, so u³ → 2, u⁴ → 2u
        // c0 = t0 + 2*t3
        // c1 = t1 + 2*t4
        // c2 = t2
        return Fp3(
            t0 + t3 + t3,
            t1 + t4 + t4,
            t2
        );
    }
    
    __device__ Fp3 square() const {
        Fp a0 = c0, a1 = c1, a2 = c2;
        
        Fp a0sq = a0 * a0;
        Fp a1sq = a1 * a1;
        Fp a2sq = a2 * a2;
        Fp a0a1 = a0 * a1;
        Fp a0a2 = a0 * a2;
        Fp a1a2 = a1 * a2;
        
        Fp t0 = a0sq;
        Fp t1 = a0a1 + a0a1;
        Fp t2 = a0a2 + a0a2 + a1sq;
        Fp t3 = a1a2 + a1a2;
        Fp t4 = a2sq;
        
        return Fp3(
            t0 + t3 + t3,
            t1 + t4 + t4,
            t2
        );
    }
    
    __device__ bool operator==(Fp3 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1 && c2 == rhs.c2;
    }
    
    __device__ bool operator!=(Fp3 rhs) const {
        return !(*this == rhs);
    }
};

// Inversion for Fp3 using Gaussian elimination on 3x3 matrix
__device__ inline Fp3 inv(Fp3 x) {
    if (x == Fp3::zero()) return Fp3::zero();
    
    Fp a0 = x.c0, a1 = x.c1, a2 = x.c2;
    Fp W_val(Fp3::W);
    
    // Matrix for multiplication by x:
    // [a0, W*a2, W*a1]
    // [a1, a0,   W*a2]
    // [a2, a1,   a0  ]
    
    Fp m[3][4];
    m[0][0] = a0; m[0][1] = W_val * a2; m[0][2] = W_val * a1; m[0][3] = Fp::one();
    m[1][0] = a1; m[1][1] = a0; m[1][2] = W_val * a2; m[1][3] = Fp::zero();
    m[2][0] = a2; m[2][1] = a1; m[2][2] = a0; m[2][3] = Fp::zero();
    
    // Gaussian elimination
    for (int col = 0; col < 3; col++) {
        Fp pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 4; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        for (int row = 0; row < 3; row++) {
            if (row != col) {
                Fp factor = m[row][col];
                for (int j = col; j < 4; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    return Fp3(m[0][3], m[1][3], m[2][3]);
}

// ============================================================================
// Fp3x2 = Fp3[v] / (v² - 11)
// ============================================================================

struct Fp3x2 {
    Fp3 c0, c1;  // c0 + c1*v
    
    static constexpr uint32_t W = 11;  // v² = 11
    
    __device__ Fp3x2() : c0(), c1() {}
    __device__ Fp3x2(Fp a) : c0(a), c1() {}
    __device__ Fp3x2(Fp3 a0) : c0(a0), c1() {}
    __device__ Fp3x2(Fp3 a0, Fp3 a1) : c0(a0), c1(a1) {}
    __device__ explicit Fp3x2(uint32_t x) : c0(Fp(x)), c1() {}
    
    // Construct from 6 Fp elements (for initialization from raw data)
    __device__ Fp3x2(Fp a0, Fp a1, Fp a2, Fp a3, Fp a4, Fp a5)
        : c0(a0, a1, a2), c1(a3, a4, a5) {}
    
    __device__ static Fp3x2 zero() { return Fp3x2(); }
    __device__ static Fp3x2 one() { return Fp3x2(Fp3::one()); }
    
    __device__ Fp3x2 operator+(Fp3x2 rhs) const {
        return Fp3x2(c0 + rhs.c0, c1 + rhs.c1);
    }
    
    __device__ Fp3x2 operator-(Fp3x2 rhs) const {
        return Fp3x2(c0 - rhs.c0, c1 - rhs.c1);
    }
    
    __device__ Fp3x2 operator-() const {
        return Fp3x2(-c0, -c1);
    }
    
    __device__ Fp3x2 operator*(Fp rhs) const {
        return Fp3x2(c0 * rhs, c1 * rhs);
    }
    
    __device__ Fp3x2 operator*(Fp3 rhs) const {
        return Fp3x2(c0 * rhs, c1 * rhs);
    }
    
    // (a0 + a1*v) * (b0 + b1*v) = (a0*b0 + 11*a1*b1) + (a0*b1 + a1*b0)*v
    __device__ Fp3x2 operator*(Fp3x2 rhs) const {
        Fp3 a0b0 = c0 * rhs.c0;
        Fp3 a1b1 = c1 * rhs.c1;
        Fp3 a0b1_a1b0 = c0 * rhs.c1 + c1 * rhs.c0;
        return Fp3x2(a0b0 + a1b1 * Fp(W), a0b1_a1b0);
    }
    
    __device__ Fp3x2 square() const {
        // (a0 + a1*v)² = (a0² + 11*a1²) + 2*a0*a1*v
        Fp3 a0sq = c0.square();
        Fp3 a1sq = c1.square();
        Fp3 a0a1 = c0 * c1;
        return Fp3x2(a0sq + a1sq * Fp(W), a0a1 + a0a1);
    }
    
    __device__ bool operator==(Fp3x2 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1;
    }
    
    __device__ bool operator!=(Fp3x2 rhs) const {
        return !(*this == rhs);
    }
};

__device__ inline Fp3x2 operator*(Fp a, Fp3x2 b) { return b * a; }

// Inversion for Fp3x2: (a + b*v)^(-1) = (a - b*v) / (a² - 11*b²)
__device__ inline Fp3x2 inv(Fp3x2 x) {
    if (x == Fp3x2::zero()) return Fp3x2::zero();
    
    // Norm = a² - W*b² where W = 11
    Fp3 norm = x.c0.square() - x.c1.square() * Fp(Fp3x2::W);
    Fp3 norm_inv = inv(norm);
    
    // Conjugate is (a - b*v)
    return Fp3x2(x.c0 * norm_inv, -x.c1 * norm_inv);
}

static_assert(sizeof(Fp3x2) == 24, "Fp3x2 must be 24 bytes");
