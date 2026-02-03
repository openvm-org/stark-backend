/*
 * Kb2x3 - Sextic Extension via 2×3 Tower for KoalaBear
 * 
 * Tower construction: Kb → Kb2 → Kb6
 *   Kb2 = Kb[u] / (u² - 3)      where u² = 3
 *   Kb6 = Kb2[v] / (v³ - (1+u)) where v³ = 1+u
 * 
 * Constants:
 *   3 is the smallest nonsquare in KoalaBear
 *   (1+u) is not a cube in Kb2 (verified: norm(1+u) = -2, and gcd(3, p²-1) = 3)
 * 
 * Element representation:
 *   A Kb6 element is: a0 + a1*v + a2*v²  where each ai ∈ Kb2
 *   Each Kb2 element is: b0 + b1*u  where bi ∈ Kb
 *   
 * Total storage: 6 Kb elements (24 bytes)
 * 
 * Reduction rules:
 *   u² = 3
 *   v³ = 1 + u
 *   v⁴ = (1+u)*v
 *   v⁵ = (1+u)*v²
 */

#pragma once

#include "kb.h"

// ============================================================================
// Kb2 = Kb[u] / (u² - 3)
// ============================================================================

struct Kb2 {
    Kb c0, c1;  // c0 + c1*u
    
    static constexpr uint32_t W = 3;  // u² = 3
    
    __device__ Kb2() : c0(), c1() {}
    __device__ Kb2(Kb a0) : c0(a0), c1() {}
    __device__ Kb2(Kb a0, Kb a1) : c0(a0), c1(a1) {}
    __device__ explicit Kb2(uint32_t x) : c0(Kb(x)), c1() {}
    
    __device__ static Kb2 zero() { return Kb2(); }
    __device__ static Kb2 one() { return Kb2(Kb::one()); }
    
    __device__ Kb2 operator+(Kb2 rhs) const {
        return Kb2(c0 + rhs.c0, c1 + rhs.c1);
    }
    
    __device__ Kb2 operator-(Kb2 rhs) const {
        return Kb2(c0 - rhs.c0, c1 - rhs.c1);
    }
    
    __device__ Kb2 operator-() const {
        return Kb2(-c0, -c1);
    }
    
    __device__ Kb2 operator*(Kb rhs) const {
        return Kb2(c0 * rhs, c1 * rhs);
    }
    
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 + 3*a1*b1) + (a0*b1 + a1*b0)*u
    __device__ Kb2 operator*(Kb2 rhs) const {
        Kb a0b0 = c0 * rhs.c0;
        Kb a1b1 = c1 * rhs.c1;
        Kb a0b1_a1b0 = c0 * rhs.c1 + c1 * rhs.c0;
        // 3*a1*b1 = a1*b1 + a1*b1 + a1*b1 (avoid Kb(3) multiplication)
        return Kb2(a0b0 + a1b1 + a1b1 + a1b1, a0b1_a1b0);
    }
    
    __device__ Kb2 square() const {
        // (a0 + a1*u)² = (a0² + 3*a1²) + 2*a0*a1*u
        Kb a0sq = c0 * c0;
        Kb a1sq = c1 * c1;
        Kb a0a1 = c0 * c1;
        return Kb2(a0sq + a1sq + a1sq + a1sq, a0a1 + a0a1);
    }
    
    __device__ bool operator==(Kb2 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1;
    }
    
    __device__ bool operator!=(Kb2 rhs) const {
        return !(*this == rhs);
    }
};

// Inversion for Kb2: (a + b*u)^(-1) = (a - b*u) / (a² - 3*b²)
__device__ inline Kb2 inv(Kb2 x) {
    Kb a = x.c0, b = x.c1;
    Kb bsq = b * b;
    Kb norm = a * a - bsq - bsq - bsq;  // a² - 3*b²
    Kb norm_inv = inv(norm);
    return Kb2(a * norm_inv, -b * norm_inv);
}

// ============================================================================
// Kb2x3 = Kb2[v] / (v³ - (1+u))
// ============================================================================

struct Kb2x3 {
    Kb2 c0, c1, c2;  // c0 + c1*v + c2*v²
    
    // W = 1 + u = Kb2(1, 1) for v³ = 1+u
    __device__ static Kb2 W_val() { return Kb2(Kb::one(), Kb::one()); }
    
    __device__ Kb2x3() : c0(), c1(), c2() {}
    __device__ Kb2x3(Kb a) : c0(a), c1(), c2() {}
    __device__ Kb2x3(Kb2 a0) : c0(a0), c1(), c2() {}
    __device__ Kb2x3(Kb2 a0, Kb2 a1, Kb2 a2) : c0(a0), c1(a1), c2(a2) {}
    __device__ explicit Kb2x3(uint32_t x) : c0(Kb(x)), c1(), c2() {}
    
    // Construct from 6 Kb elements (for initialization from raw data)
    __device__ Kb2x3(Kb a0, Kb a1, Kb a2, Kb a3, Kb a4, Kb a5)
        : c0(a0, a1), c1(a2, a3), c2(a4, a5) {}
    
    __device__ static Kb2x3 zero() { return Kb2x3(); }
    __device__ static Kb2x3 one() { return Kb2x3(Kb2::one()); }
    
    __device__ Kb2x3 operator+(Kb2x3 rhs) const {
        return Kb2x3(c0 + rhs.c0, c1 + rhs.c1, c2 + rhs.c2);
    }
    
    __device__ Kb2x3 operator-(Kb2x3 rhs) const {
        return Kb2x3(c0 - rhs.c0, c1 - rhs.c1, c2 - rhs.c2);
    }
    
    __device__ Kb2x3 operator-() const {
        return Kb2x3(-c0, -c1, -c2);
    }
    
    __device__ Kb2x3 operator*(Kb rhs) const {
        return Kb2x3(c0 * rhs, c1 * rhs, c2 * rhs);
    }
    
    // Multiply by Kb2 scalar
    __device__ Kb2x3 operator*(Kb2 rhs) const {
        return Kb2x3(c0 * rhs, c1 * rhs, c2 * rhs);
    }
    
    // Full multiplication: (a0 + a1*v + a2*v²) * (b0 + b1*v + b2*v²)
    // with reduction v³ = (1+u)
    __device__ Kb2x3 operator*(Kb2x3 rhs) const {
        Kb2 a0 = c0, a1 = c1, a2 = c2;
        Kb2 b0 = rhs.c0, b1 = rhs.c1, b2 = rhs.c2;
        
        // Schoolbook: t[k] = sum_{i+j=k} a[i]*b[j]
        Kb2 t0 = a0 * b0;
        Kb2 t1 = a0 * b1 + a1 * b0;
        Kb2 t2 = a0 * b2 + a1 * b1 + a2 * b0;
        Kb2 t3 = a1 * b2 + a2 * b1;
        Kb2 t4 = a2 * b2;
        
        // Reduction: v³ = (1+u), v⁴ = (1+u)*v
        // c0 = t0 + (1+u)*t3
        // c1 = t1 + (1+u)*t4
        // c2 = t2
        Kb2 w = W_val();
        return Kb2x3(
            t0 + w * t3,
            t1 + w * t4,
            t2
        );
    }
    
    __device__ Kb2x3 square() const {
        Kb2 a0 = c0, a1 = c1, a2 = c2;
        
        Kb2 a0sq = a0.square();
        Kb2 a1sq = a1.square();
        Kb2 a2sq = a2.square();
        Kb2 a0a1 = a0 * a1;
        Kb2 a0a2 = a0 * a2;
        Kb2 a1a2 = a1 * a2;
        
        // t0 = a0², t1 = 2*a0*a1, t2 = 2*a0*a2 + a1², t3 = 2*a1*a2, t4 = a2²
        Kb2 t0 = a0sq;
        Kb2 t1 = a0a1 + a0a1;
        Kb2 t2 = a0a2 + a0a2 + a1sq;
        Kb2 t3 = a1a2 + a1a2;
        Kb2 t4 = a2sq;
        
        Kb2 w = W_val();
        return Kb2x3(
            t0 + w * t3,
            t1 + w * t4,
            t2
        );
    }
    
    __device__ bool operator==(Kb2x3 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1 && c2 == rhs.c2;
    }
    
    __device__ bool operator!=(Kb2x3 rhs) const {
        return !(*this == rhs);
    }
};

__device__ inline Kb2x3 operator*(Kb a, Kb2x3 b) { return b * a; }

// Inversion using Gaussian elimination on 3×3 matrix over Kb2
// For a ∈ Kb6 = Kb2[v]/(v³ - W), the multiplication matrix is:
// Row 0: [a0, W*a2, W*a1]
// Row 1: [a1, a0,   W*a2]
// Row 2: [a2, a1,   a0  ]
__device__ inline Kb2x3 inv(Kb2x3 x) {
    if (x == Kb2x3::zero()) return Kb2x3::zero();
    
    Kb2 a0 = x.c0, a1 = x.c1, a2 = x.c2;
    Kb2 w = Kb2x3::W_val();
    
    // Build 3x4 augmented matrix [M | I]
    Kb2 m[3][4];
    m[0][0] = a0;     m[0][1] = w * a2; m[0][2] = w * a1; m[0][3] = Kb2::one();
    m[1][0] = a1;     m[1][1] = a0;     m[1][2] = w * a2; m[1][3] = Kb2::zero();
    m[2][0] = a2;     m[2][1] = a1;     m[2][2] = a0;     m[2][3] = Kb2::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 3; col++) {
        // Find non-zero pivot
        int pivot_row = col;
        while (pivot_row < 3 && m[pivot_row][col] == Kb2::zero()) {
            pivot_row++;
        }
        if (pivot_row >= 3) return Kb2x3::zero();  // Singular
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 4; j++) {
                Kb2 tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row
        Kb2 pivot_inv = inv(m[col][col]);
        for (int j = col; j < 4; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 3; row++) {
            if (row != col) {
                Kb2 factor = m[row][col];
                for (int j = col; j < 4; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    return Kb2x3(m[0][3], m[1][3], m[2][3]);
}

static_assert(sizeof(Kb2x3) == 24, "Kb2x3 must be 24 bytes");
