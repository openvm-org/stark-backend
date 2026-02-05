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
    union { Kb elems[2]; uint32_t u[2]; };  // c0 + c1*u
    
    static constexpr uint32_t W = 3;  // u² = 3
    static constexpr uint32_t MOD  = 0x7f000001;
    static constexpr uint32_t M    = 0x7effffff;
    static constexpr uint32_t BETA = 0x05FFFFFA;  // (3 << 32) % MOD
    
    __device__ Kb2() : elems{Kb(), Kb()} {}
    __device__ Kb2(Kb a0) : elems{a0, Kb()} {}
    __device__ Kb2(Kb a0, Kb a1) : elems{a0, a1} {}
    __device__ explicit Kb2(uint32_t x) : elems{Kb(x), Kb()} {}
    
    // Accessors for compatibility
    __device__ Kb& c0() { return elems[0]; }
    __device__ Kb& c1() { return elems[1]; }
    __device__ const Kb& c0() const { return elems[0]; }
    __device__ const Kb& c1() const { return elems[1]; }
    
    __device__ static Kb2 zero() { return Kb2(); }
    __device__ static Kb2 one() { return Kb2(Kb::one()); }
    
    __device__ Kb2 operator+(Kb2 rhs) const {
        return Kb2(elems[0] + rhs.elems[0], elems[1] + rhs.elems[1]);
    }
    
    __device__ Kb2 operator-(Kb2 rhs) const {
        return Kb2(elems[0] - rhs.elems[0], elems[1] - rhs.elems[1]);
    }
    
    __device__ Kb2 operator-() const {
        return Kb2(-elems[0], -elems[1]);
    }
    
    __device__ Kb2 operator*(Kb rhs) const {
        return Kb2(elems[0] * rhs, elems[1] * rhs);
    }
    
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 + 3*a1*b1) + (a0*b1 + a1*b0)*u
    // r0 = a0*b0 + BETA*a1*b1
    // r1 = a0*b1 + a1*b0
    __device__ Kb2 operator*(Kb2 rhs) const {
        Kb2 ret;
        
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
        Kb a0b0 = elems[0] * rhs.elems[0];
        Kb a1b1 = elems[1] * rhs.elems[1];
        Kb a0b1_a1b0 = elems[0] * rhs.elems[1] + elems[1] * rhs.elems[0];
        ret.elems[0] = a0b0 + a1b1 + a1b1 + a1b1;
        ret.elems[1] = a0b1_a1b0;
# endif
        return ret;
    }
    
    __device__ Kb2 square() const {
        Kb a0sq = elems[0] * elems[0];
        Kb a1sq = elems[1] * elems[1];
        Kb a0a1 = elems[0] * elems[1];
        return Kb2(a0sq + a1sq + a1sq + a1sq, a0a1 + a0a1);
    }
    
    __device__ bool operator==(Kb2 rhs) const {
        return elems[0] == rhs.elems[0] && elems[1] == rhs.elems[1];
    }
    
    __device__ bool operator!=(Kb2 rhs) const {
        return !(*this == rhs);
    }
};

// Inversion for Kb2: (a + b*u)^(-1) = (a - b*u) / (a² - 3*b²)
__device__ inline Kb2 inv(Kb2 x) {
    Kb a = x.elems[0], b = x.elems[1];
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

    // Multiply by W = 1 + u without a Kb2 mul
    __device__ static inline Kb2 mulByW(Kb2 x) {
        Kb a0 = x.elems[0];
        Kb a1 = x.elems[1];
        Kb a1_2 = a1 + a1;
        Kb c1 = a0 + a1;
        Kb c0 = c1 + a1_2; // a0 + 3*a1
        return Kb2(c0, c1);
    }
    
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
    // Using Karatsuba-style: 6 Kb2 muls instead of 9
    __device__ Kb2x3 operator*(Kb2x3 rhs) const {
        Kb2 a0 = c0, a1 = c1, a2 = c2;
        Kb2 b0 = rhs.c0, b1 = rhs.c1, b2 = rhs.c2;
        
        // Karatsuba for degree-3 polynomial multiplication
        // 6 multiplications instead of 9
        Kb2 v0 = a0 * b0;
        Kb2 v1 = a1 * b1;
        Kb2 v2 = a2 * b2;
        Kb2 v01 = (a0 + a1) * (b0 + b1);
        Kb2 v12 = (a1 + a2) * (b1 + b2);
        Kb2 v02 = (a0 + a2) * (b0 + b2);
        
        // Reconstruct coefficients
        Kb2 t0 = v0;
        Kb2 t1 = v01 - v0 - v1;
        Kb2 t2 = v02 - v0 - v2 + v1;
        Kb2 t3 = v12 - v1 - v2;
        Kb2 t4 = v2;
        
        // Reduction: v³ = (1+u), v⁴ = (1+u)*v
        // c0 = t0 + (1+u)*t3
        // c1 = t1 + (1+u)*t4
        // c2 = t2
        return Kb2x3(
            t0 + mulByW(t3),
            t1 + mulByW(t4),
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
        
        return Kb2x3(
            t0 + mulByW(t3),
            t1 + mulByW(t4),
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

// Inversion using adjugate formula for cubic extension v^3 = W
__device__ inline Kb2x3 inv(Kb2x3 x) {
    if (x == Kb2x3::zero()) return Kb2x3::zero();
    
    Kb2 a0 = x.c0, a1 = x.c1, a2 = x.c2;
    
    // c0 = a0^2 - W*a1*a2
    // c1 = W*a2^2 - a0*a1
    // c2 = a1^2 - a0*a2
    Kb2 a0sq = a0.square();
    Kb2 a1sq = a1.square();
    Kb2 a2sq = a2.square();
    Kb2 a1a2 = a1 * a2;
    Kb2 a0a1 = a0 * a1;
    Kb2 a0a2 = a0 * a2;
    
    Kb2 c0 = a0sq - Kb2x3::mulByW(a1a2);
    Kb2 c1 = Kb2x3::mulByW(a2sq) - a0a1;
    Kb2 c2 = a1sq - a0a2;
    
    // norm = a0*c0 + W*(a1*c2 + a2*c1)
    Kb2 norm = a0 * c0 + Kb2x3::mulByW(a1 * c2 + a2 * c1);
    Kb2 norm_inv = inv(norm);
    
    return Kb2x3(c0 * norm_inv, c1 * norm_inv, c2 * norm_inv);
}

static_assert(sizeof(Kb2x3) == 24, "Kb2x3 must be 24 bytes");
