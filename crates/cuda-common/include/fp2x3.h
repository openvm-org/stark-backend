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
    union { Fp elems[2]; uint32_t u[2]; };  // c0 + c1*u
    
    static constexpr uint32_t W = 11;  // u² = 11
    static constexpr uint32_t MOD  = 0x78000001;
    static constexpr uint32_t M    = 0x77ffffff;
    static constexpr uint32_t BETA = 0x37FFFFE9;  // (11 << 32) % MOD
    
    __device__ Fp2() : elems{Fp(), Fp()} {}
    __device__ Fp2(Fp a0) : elems{a0, Fp()} {}
    __device__ Fp2(Fp a0, Fp a1) : elems{a0, a1} {}
    __device__ explicit Fp2(uint32_t x) : elems{Fp(x), Fp()} {}
    
    // Accessors for compatibility
    __device__ Fp& c0() { return elems[0]; }
    __device__ Fp& c1() { return elems[1]; }
    __device__ const Fp& c0() const { return elems[0]; }
    __device__ const Fp& c1() const { return elems[1]; }
    
    __device__ static Fp2 zero() { return Fp2(); }
    __device__ static Fp2 one() { return Fp2(Fp::one()); }
    
    __device__ Fp2 operator+(Fp2 rhs) const {
        return Fp2(elems[0] + rhs.elems[0], elems[1] + rhs.elems[1]);
    }
    
    __device__ Fp2 operator-(Fp2 rhs) const {
        return Fp2(elems[0] - rhs.elems[0], elems[1] - rhs.elems[1]);
    }
    
    __device__ Fp2 operator-() const {
        return Fp2(-elems[0], -elems[1]);
    }
    
    __device__ Fp2 operator*(Fp rhs) const {
        return Fp2(elems[0] * rhs, elems[1] * rhs);
    }
    
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 + 11*a1*b1) + (a0*b1 + a1*b0)*u
    // r0 = a0*b0 + BETA*a1*b1
    // r1 = a0*b1 + a1*b0
    __device__ Fp2 operator*(Fp2 rhs) const {
        Fp2 ret;
        
# if defined(__CUDA_ARCH__) && !defined(__clang__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // Operand mapping: %1=a0, %2=a1, %3=b0, %4=b1, %5=MOD, %6=M, %7=BETA
        
        // r0 = a0*b0 + BETA*a1*b1
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %2, %4;     mul.hi.u32  %hi, %2, %4;\n\t"       // a1*b1
            "mul.lo.u32    %m, %lo, %6;\n\t"
            "mad.lo.cc.u32 %lo, %m, %5, %lo; madc.hi.u32 %hi, %m, %5, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %7;    mul.hi.u32  %hi, %hi, %7;\n\t"      // *BETA
            "mad.lo.cc.u32 %lo, %1, %3, %lo; madc.hi.u32 %hi, %1, %3, %hi;\n\t" // +a0*b0
            "setp.ge.u32   %p, %hi, %5;\n\t"
            "@%p sub.u32   %hi, %hi, %5;\n\t"
            "mul.lo.u32    %m, %lo, %6;\n\t"
            "mad.lo.cc.u32 %lo, %m, %5, %lo; madc.hi.u32 %0, %m, %5, %hi;\n\t"
            "setp.ge.u32   %p, %0, %5;\n\t"
            "@%p sub.u32   %0, %0, %5;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(rhs.u[0]), "r"(rhs.u[1]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // r1 = a0*b1 + a1*b0 (no BETA)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %1, %4;     mul.hi.u32  %hi, %1, %4;\n\t"       // a0*b1
            "mad.lo.cc.u32 %lo, %2, %3, %lo; madc.hi.u32 %hi, %2, %3, %hi;\n\t" // +a1*b0
            "setp.ge.u32   %p, %hi, %5;\n\t"
            "@%p sub.u32   %hi, %hi, %5;\n\t"
            "mul.lo.u32    %m, %lo, %6;\n\t"
            "mad.lo.cc.u32 %lo, %m, %5, %lo; madc.hi.u32 %0, %m, %5, %hi;\n\t"
            "setp.ge.u32   %p, %0, %5;\n\t"
            "@%p sub.u32   %0, %0, %5;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u[1]), "r"(rhs.u[0]), "r"(rhs.u[1]),
                  "r"(MOD), "r"(M), "r"(BETA));
#  undef asm
# else
        Fp a0b0 = elems[0] * rhs.elems[0];
        Fp a1b1 = elems[1] * rhs.elems[1];
        Fp a0b1_a1b0 = elems[0] * rhs.elems[1] + elems[1] * rhs.elems[0];
        ret.elems[0] = a0b0 + Fp(W) * a1b1;
        ret.elems[1] = a0b1_a1b0;
# endif
        return ret;
    }
    
    __device__ Fp2 square() const {
        // (a0 + a1*u)² = (a0² + 11*a1²) + 2*a0*a1*u
        Fp a0sq = elems[0] * elems[0];
        Fp a1sq = elems[1] * elems[1];
        Fp a0a1 = elems[0] * elems[1];
        return Fp2(a0sq + Fp(W) * a1sq, a0a1 + a0a1);
    }
    
    __device__ bool operator==(Fp2 rhs) const {
        return elems[0] == rhs.elems[0] && elems[1] == rhs.elems[1];
    }
    
    __device__ bool operator!=(Fp2 rhs) const {
        return !(*this == rhs);
    }
};

// Inversion for Fp2: (a + b*u)^(-1) = (a - b*u) / (a² - 11*b²)
__device__ inline Fp2 inv(Fp2 x) {
    Fp norm = x.elems[0] * x.elems[0] - Fp(Fp2::W) * x.elems[1] * x.elems[1];
    Fp norm_inv = inv(norm);
    return Fp2(x.elems[0] * norm_inv, -x.elems[1] * norm_inv);
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
    // with reduction v³ = W
    // Using Karatsuba-style: 6 Fp2 muls instead of 9
    __device__ Fp2x3 operator*(Fp2x3 rhs) const {
        Fp2 a0 = c0, a1 = c1, a2 = c2;
        Fp2 b0 = rhs.c0, b1 = rhs.c1, b2 = rhs.c2;
        
        // Karatsuba for degree-3 polynomial multiplication
        // 6 multiplications instead of 9
        Fp2 v0 = a0 * b0;
        Fp2 v1 = a1 * b1;
        Fp2 v2 = a2 * b2;
        Fp2 v01 = (a0 + a1) * (b0 + b1);
        Fp2 v12 = (a1 + a2) * (b1 + b2);
        Fp2 v02 = (a0 + a2) * (b0 + b2);
        
        // Reconstruct coefficients
        Fp2 t0 = v0;
        Fp2 t1 = v01 - v0 - v1;
        Fp2 t2 = v02 - v0 - v2 + v1;
        Fp2 t3 = v12 - v1 - v2;
        Fp2 t4 = v2;
        
        // Reduction: v³ = W, so v³ → W, v⁴ → W*v
        // c0 = t0 + W*t3
        // c1 = t1 + W*t4
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
