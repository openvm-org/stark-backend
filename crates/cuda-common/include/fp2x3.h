/*
 * Fp2x3 - Sextic Extension via 2×3 Tower
 * 
 * Tower construction: Fp → Fp2 → Fp6
 *   Fp2 = Fp[u] / (u² - 11)
 *   Fp6 = Fp2[v] / (v³ - 2)
 * 
 * Element representation:
 *   An Fp6 element is: a0 + a1*v + a2*v²  where each ai ∈ Fp2
 *   Each Fp2 element is: b0 + b1*u  where bi ∈ Fp
 *   
 * Total storage: 6 Fp elements (24 bytes)
 * 
 * Constants:
 *   W2 = 11 (for Fp2: u² = 11)
 *   W3 = 2  (for Fp6: v³ = 2)
 */

#pragma once

#include "fp.h"

// ============================================================================
// Fp2 = Fp[u] / (u² - 11)
// ============================================================================

struct Fp2 {
    Fp c0, c1;  // c0 + c1*u
    
    static constexpr uint32_t W = 11;  // u² = 11
    
    __device__ Fp2() : c0(), c1() {}
    __device__ Fp2(Fp a0) : c0(a0), c1() {}
    __device__ Fp2(Fp a0, Fp a1) : c0(a0), c1(a1) {}
    __device__ explicit Fp2(uint32_t x) : c0(Fp(x)), c1() {}
    
    __device__ static Fp2 zero() { return Fp2(); }
    __device__ static Fp2 one() { return Fp2(Fp::one()); }
    
    __device__ Fp2 operator+(Fp2 rhs) const {
        return Fp2(c0 + rhs.c0, c1 + rhs.c1);
    }
    
    __device__ Fp2 operator-(Fp2 rhs) const {
        return Fp2(c0 - rhs.c0, c1 - rhs.c1);
    }
    
    __device__ Fp2 operator-() const {
        return Fp2(-c0, -c1);
    }
    
    __device__ Fp2 operator*(Fp rhs) const {
        return Fp2(c0 * rhs, c1 * rhs);
    }
    
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 + 11*a1*b1) + (a0*b1 + a1*b0)*u
    __device__ Fp2 operator*(Fp2 rhs) const {
        Fp a0b0 = c0 * rhs.c0;
        Fp a1b1 = c1 * rhs.c1;
        Fp a0b1_a1b0 = c0 * rhs.c1 + c1 * rhs.c0;
        return Fp2(a0b0 + Fp(W) * a1b1, a0b1_a1b0);
    }
    
    __device__ Fp2 square() const {
        // (a0 + a1*u)² = (a0² + 11*a1²) + 2*a0*a1*u
        Fp a0sq = c0 * c0;
        Fp a1sq = c1 * c1;
        Fp a0a1 = c0 * c1;
        return Fp2(a0sq + Fp(W) * a1sq, a0a1 + a0a1);
    }
    
    __device__ bool operator==(Fp2 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1;
    }
    
    __device__ bool operator!=(Fp2 rhs) const {
        return !(*this == rhs);
    }
};

// Inversion for Fp2: (a + b*u)^(-1) = (a - b*u) / (a² - 11*b²)
__device__ inline Fp2 inv(Fp2 x) {
    Fp norm = x.c0 * x.c0 - Fp(Fp2::W) * x.c1 * x.c1;
    Fp norm_inv = inv(norm);
    return Fp2(x.c0 * norm_inv, -x.c1 * norm_inv);
}

// ============================================================================
// Fp2x3 = Fp2[v] / (v³ - 2)
// ============================================================================

struct Fp2x3 {
    Fp2 c0, c1, c2;  // c0 + c1*v + c2*v²
    
    static constexpr uint32_t W = 2;  // v³ = 2
    
    __device__ Fp2x3() : c0(), c1(), c2() {}
    __device__ Fp2x3(Fp a) : c0(a), c1(), c2() {}
    __device__ Fp2x3(Fp2 a0) : c0(a0), c1(), c2() {}
    __device__ Fp2x3(Fp2 a0, Fp2 a1, Fp2 a2) : c0(a0), c1(a1), c2(a2) {}
    __device__ explicit Fp2x3(uint32_t x) : c0(Fp(x)), c1(), c2() {}
    
    // Construct from 6 Fp elements (for initialization from raw data)
    __device__ Fp2x3(Fp a0, Fp a1, Fp a2, Fp a3, Fp a4, Fp a5)
        : c0(a0, a1), c1(a2, a3), c2(a4, a5) {}
    
    __device__ static Fp2x3 zero() { return Fp2x3(); }
    __device__ static Fp2x3 one() { return Fp2x3(Fp2::one()); }
    
    __device__ Fp2x3 operator+(Fp2x3 rhs) const {
        return Fp2x3(c0 + rhs.c0, c1 + rhs.c1, c2 + rhs.c2);
    }
    
    __device__ Fp2x3 operator-(Fp2x3 rhs) const {
        return Fp2x3(c0 - rhs.c0, c1 - rhs.c1, c2 - rhs.c2);
    }
    
    __device__ Fp2x3 operator-() const {
        return Fp2x3(-c0, -c1, -c2);
    }
    
    __device__ Fp2x3 operator*(Fp rhs) const {
        return Fp2x3(c0 * rhs, c1 * rhs, c2 * rhs);
    }
    
    // Multiply by Fp2 scalar
    __device__ Fp2x3 operator*(Fp2 rhs) const {
        return Fp2x3(c0 * rhs, c1 * rhs, c2 * rhs);
    }
    
    // Full multiplication: (a0 + a1*v + a2*v²) * (b0 + b1*v + b2*v²)
    // with reduction v³ = 2
    __device__ Fp2x3 operator*(Fp2x3 rhs) const {
        Fp2 a0 = c0, a1 = c1, a2 = c2;
        Fp2 b0 = rhs.c0, b1 = rhs.c1, b2 = rhs.c2;
        
        // Schoolbook: t[k] = sum_{i+j=k} a[i]*b[j]
        Fp2 t0 = a0 * b0;
        Fp2 t1 = a0 * b1 + a1 * b0;
        Fp2 t2 = a0 * b2 + a1 * b1 + a2 * b0;
        Fp2 t3 = a1 * b2 + a2 * b1;
        Fp2 t4 = a2 * b2;
        
        // Reduction: v³ = 2, so v³ → 2, v⁴ → 2v
        // c0 = t0 + 2*t3
        // c1 = t1 + 2*t4
        // c2 = t2
        Fp w_val(W);
        return Fp2x3(
            t0 + t3 * w_val,
            t1 + t4 * w_val,
            t2
        );
    }
    
    __device__ Fp2x3 square() const {
        Fp2 a0 = c0, a1 = c1, a2 = c2;
        
        Fp2 a0sq = a0.square();
        Fp2 a1sq = a1.square();
        Fp2 a2sq = a2.square();
        Fp2 a0a1 = a0 * a1;
        Fp2 a0a2 = a0 * a2;
        Fp2 a1a2 = a1 * a2;
        
        // t0 = a0², t1 = 2*a0*a1, t2 = 2*a0*a2 + a1², t3 = 2*a1*a2, t4 = a2²
        Fp2 t0 = a0sq;
        Fp2 t1 = a0a1 + a0a1;
        Fp2 t2 = a0a2 + a0a2 + a1sq;
        Fp2 t3 = a1a2 + a1a2;
        Fp2 t4 = a2sq;
        
        Fp w_val(W);
        return Fp2x3(
            t0 + t3 * w_val,
            t1 + t4 * w_val,
            t2
        );
    }
    
    __device__ bool operator==(Fp2x3 rhs) const {
        return c0 == rhs.c0 && c1 == rhs.c1 && c2 == rhs.c2;
    }
    
    __device__ bool operator!=(Fp2x3 rhs) const {
        return !(*this == rhs);
    }
};

__device__ inline Fp2x3 operator*(Fp a, Fp2x3 b) { return b * a; }

// Inversion using norm to Fp2, then to Fp
// For a ∈ Fp6 = Fp2[v]/(v³-2), compute Norm_{Fp6/Fp2}(a) ∈ Fp2
// Then inv(a) = a^(p²-1) * inv(Norm(a))
__device__ inline Fp2x3 inv(Fp2x3 x) {
    if (x == Fp2x3::zero()) return Fp2x3::zero();
    
    // For cubic extension Fp2[v]/(v³ - W):
    // Norm(a0 + a1*v + a2*v²) = a0³ + W*a1³ + W²*a2³ - 3*W*a0*a1*a2
    // But this is complex. Use Gaussian elimination instead (like Fp5/Fp6).
    
    // Alternative: use the formula for cubic extension inverse
    // Let a = c0 + c1*v + c2*v², with v³ = W
    // The inverse uses the matrix approach
    
    Fp2 a0 = x.c0, a1 = x.c1, a2 = x.c2;
    Fp2 W_val(Fp(Fp2x3::W));
    
    // Build 3x3 matrix and solve via Gaussian elimination
    // Matrix M where M * [b0, b1, b2]^T = [1, 0, 0]^T
    // Row 0: [a0, W*a2, W*a1]
    // Row 1: [a1, a0, W*a2]
    // Row 2: [a2, a1, a0]
    
    Fp2 m[3][4];
    m[0][0] = a0; m[0][1] = W_val * a2; m[0][2] = W_val * a1; m[0][3] = Fp2::one();
    m[1][0] = a1; m[1][1] = a0; m[1][2] = W_val * a2; m[1][3] = Fp2::zero();
    m[2][0] = a2; m[2][1] = a1; m[2][2] = a0; m[2][3] = Fp2::zero();
    
    // Gaussian elimination
    for (int col = 0; col < 3; col++) {
        // Scale pivot row
        Fp2 pivot_inv = inv(m[col][col]);
        for (int j = col; j < 4; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        // Eliminate column in other rows
        for (int row = 0; row < 3; row++) {
            if (row != col) {
                Fp2 factor = m[row][col];
                for (int j = col; j < 4; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    return Fp2x3(m[0][3], m[1][3], m[2][3]);
}

static_assert(sizeof(Fp2x3) == 24, "Fp2x3 must be 24 bytes");
