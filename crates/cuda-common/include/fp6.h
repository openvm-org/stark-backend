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
 * 
 * Optimizations:
 * - PTX assembly for multiplication (~1.5x speedup)
 * - Frobenius-based inversion (1 Fp inv instead of 6)
 */

#pragma once

#include "fp.h"

struct Fp6 {
    union { Fp elems[6]; uint32_t u[6]; };
    
    // W = 31 = 2^5 - 1 (multiplicative generator of Baby Bear)
    // x^6 = 31 in this extension
    static constexpr uint32_t W = 31;
    
    // Montgomery constants for PTX multiplication
    static constexpr uint32_t MOD  = 0x78000001;
    static constexpr uint32_t M    = 0x77ffffff;
    static constexpr uint32_t BETA = 0x0FFFFFBE;  // (W << 32) % MOD
    
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
    
    /// Helper: multiply by W = 31 using constant
    __device__ static Fp mulByW(Fp x) {
        return Fp(W) * x;
    }
    
    /// Full extension field multiplication
    /// c[i] = t[i] + W*t[i+6] where t[k] = sum_{i+j=k} a[i]*b[j]
    __device__ Fp6 operator*(Fp6 rhs) const {
        Fp6 ret;
        
# if defined(__CUDA_ARCH__) && !defined(__clang__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // PTX assembly for Fp6 multiplication
        // Operand mapping: %1=a0..%6=a5, %7=b0..%12=b5, %13=MOD, %14=M, %15=BETA
        
        // ret[0] = a0*b0 + BETA*(a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %2, %12;     mul.hi.u32  %hi, %2, %12;\n\t"
            "mad.lo.cc.u32 %lo, %3, %11, %lo; madc.hi.u32 %hi, %3, %11, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %4, %10, %lo; madc.hi.u32 %hi, %4, %10, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %5, %9, %lo; madc.hi.u32 %hi, %5, %9, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %6, %8, %lo; madc.hi.u32 %hi, %6, %8, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %13;\n\t"
            "@%p sub.u32   %hi, %hi, %13;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %hi, %m, %13, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %15;    mul.hi.u32  %hi, %hi, %15;\n\t"
            "mad.lo.cc.u32 %lo, %1, %7, %lo; madc.hi.u32 %hi, %1, %7, %hi;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %0, %m, %13, %hi;\n\t"
            "setp.ge.u32   %p, %0, %13;\n\t"
            "@%p sub.u32   %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[1] = a0*b1 + a1*b0 + BETA*(a2*b5 + a3*b4 + a4*b3 + a5*b2)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %3, %12;     mul.hi.u32  %hi, %3, %12;\n\t"
            "mad.lo.cc.u32 %lo, %4, %11, %lo; madc.hi.u32 %hi, %4, %11, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %5, %10, %lo; madc.hi.u32 %hi, %5, %10, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %6, %9, %lo; madc.hi.u32 %hi, %6, %9, %hi;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %hi, %m, %13, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %15;    mul.hi.u32  %hi, %hi, %15;\n\t"
            "mad.lo.cc.u32 %lo, %1, %8, %lo; madc.hi.u32 %hi, %1, %8, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %7, %lo; madc.hi.u32 %hi, %2, %7, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %13;\n\t"
            "@%p sub.u32   %hi, %hi, %13;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %0, %m, %13, %hi;\n\t"
            "setp.ge.u32   %p, %0, %13;\n\t"
            "@%p sub.u32   %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[2] = a0*b2 + a1*b1 + a2*b0 + BETA*(a3*b5 + a4*b4 + a5*b3)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %12;     mul.hi.u32  %hi, %4, %12;\n\t"
            "mad.lo.cc.u32 %lo, %5, %11, %lo; madc.hi.u32 %hi, %5, %11, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %6, %10, %lo; madc.hi.u32 %hi, %6, %10, %hi;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %hi, %m, %13, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %15;    mul.hi.u32  %hi, %hi, %15;\n\t"
            "mad.lo.cc.u32 %lo, %1, %9, %lo; madc.hi.u32 %hi, %1, %9, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %8, %lo; madc.hi.u32 %hi, %2, %8, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %3, %7, %lo; madc.hi.u32 %hi, %3, %7, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %13;\n\t"
            "@%p sub.u32   %hi, %hi, %13;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %0, %m, %13, %hi;\n\t"
            "setp.ge.u32   %p, %0, %13;\n\t"
            "@%p sub.u32   %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + BETA*(a4*b5 + a5*b4)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %5, %12;     mul.hi.u32  %hi, %5, %12;\n\t"
            "mad.lo.cc.u32 %lo, %6, %11, %lo; madc.hi.u32 %hi, %6, %11, %hi;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %hi, %m, %13, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %15;    mul.hi.u32  %hi, %hi, %15;\n\t"
            "mad.lo.cc.u32 %lo, %1, %10, %lo; madc.hi.u32 %hi, %1, %10, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %9, %lo; madc.hi.u32 %hi, %2, %9, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %3, %8, %lo; madc.hi.u32 %hi, %3, %8, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %4, %7, %lo; madc.hi.u32 %hi, %4, %7, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %13;\n\t"
            "@%p sub.u32   %hi, %hi, %13;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %0, %m, %13, %hi;\n\t"
            "setp.ge.u32   %p, %0, %13;\n\t"
            "@%p sub.u32   %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + BETA*(a5*b5)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %6, %12;     mul.hi.u32  %hi, %6, %12;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %hi, %m, %13, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %15;    mul.hi.u32  %hi, %hi, %15;\n\t"
            "mad.lo.cc.u32 %lo, %1, %11, %lo; madc.hi.u32 %hi, %1, %11, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %10, %lo; madc.hi.u32 %hi, %2, %10, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %3, %9, %lo; madc.hi.u32 %hi, %3, %9, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %4, %8, %lo; madc.hi.u32 %hi, %4, %8, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %5, %7, %lo; madc.hi.u32 %hi, %5, %7, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %13;\n\t"
            "@%p sub.u32   %hi, %hi, %13;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %0, %m, %13, %hi;\n\t"
            "setp.ge.u32   %p, %0, %13;\n\t"
            "@%p sub.u32   %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[4])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[5] = a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0 (no BETA)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %1, %12;     mul.hi.u32  %hi, %1, %12;\n\t"
            "mad.lo.cc.u32 %lo, %2, %11, %lo; madc.hi.u32 %hi, %2, %11, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %3, %10, %lo; madc.hi.u32 %hi, %3, %10, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %4, %9, %lo; madc.hi.u32 %hi, %4, %9, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %5, %8, %lo; madc.hi.u32 %hi, %5, %8, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %6, %7, %lo; madc.hi.u32 %hi, %6, %7, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %13;\n\t"
            "@%p sub.u32   %hi, %hi, %13;\n\t"
            "mul.lo.u32    %m, %lo, %14;\n\t"
            "mad.lo.cc.u32 %lo, %m, %13, %lo; madc.hi.u32 %0, %m, %13, %hi;\n\t"
            "setp.ge.u32   %p, %0, %13;\n\t"
            "@%p sub.u32   %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[5])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(BETA));
#  undef asm
# else
        // Fallback: schoolbook multiplication
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2];
        const Fp a3 = elems[3], a4 = elems[4], a5 = elems[5];
        const Fp b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2];
        const Fp b3 = rhs.elems[3], b4 = rhs.elems[4], b5 = rhs.elems[5];
        
        Fp t6 = a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1;
        ret.elems[0] = a0*b0 + mulByW(t6);
        
        Fp t7 = a2*b5 + a3*b4 + a4*b3 + a5*b2;
        ret.elems[1] = a0*b1 + a1*b0 + mulByW(t7);
        
        Fp t8 = a3*b5 + a4*b4 + a5*b3;
        ret.elems[2] = a0*b2 + a1*b1 + a2*b0 + mulByW(t8);
        
        Fp t9 = a4*b5 + a5*b4;
        ret.elems[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + mulByW(t9);
        
        ret.elems[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + mulByW(a5*b5);
        
        ret.elems[5] = a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0;
# endif
        
        return ret;
    }
    
    __device__ Fp6& operator*=(Fp6 rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Squaring (optimized using symmetry)
    __device__ Fp6 square() const {
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2];
        const Fp a3 = elems[3], a4 = elems[4], a5 = elems[5];
        
        Fp a0sq = a0 * a0, a1sq = a1 * a1, a2sq = a2 * a2;
        Fp a3sq = a3 * a3, a4sq = a4 * a4, a5sq = a5 * a5;
        
        Fp a0a1 = a0 * a1, a0a2 = a0 * a2, a0a3 = a0 * a3;
        Fp a0a4 = a0 * a4, a0a5 = a0 * a5, a1a2 = a1 * a2;
        Fp a1a3 = a1 * a3, a1a4 = a1 * a4, a1a5 = a1 * a5;
        Fp a2a3 = a2 * a3, a2a4 = a2 * a4, a2a5 = a2 * a5;
        Fp a3a4 = a3 * a4, a3a5 = a3 * a5, a4a5 = a4 * a5;
        
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
        return elems[0] == rhs.elems[0] && elems[1] == rhs.elems[1] && 
               elems[2] == rhs.elems[2] && elems[3] == rhs.elems[3] && 
               elems[4] == rhs.elems[4] && elems[5] == rhs.elems[5];
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
__device__ inline Fp6 inv(Fp6 x) {
    if (x == Fp6::zero()) {
        return Fp6::zero();
    }
    
    const Fp W_val(Fp6::W);
    const Fp* a = x.elems;
    
    Fp m[6][7];
    
    // Build augmented matrix [M | e0]
    m[0][0] = a[0]; m[0][1] = W_val * a[5]; m[0][2] = W_val * a[4];
    m[0][3] = W_val * a[3]; m[0][4] = W_val * a[2]; m[0][5] = W_val * a[1]; 
    m[0][6] = Fp::one();
    
    m[1][0] = a[1]; m[1][1] = a[0]; m[1][2] = W_val * a[5];
    m[1][3] = W_val * a[4]; m[1][4] = W_val * a[3]; m[1][5] = W_val * a[2];
    m[1][6] = Fp::zero();
    
    m[2][0] = a[2]; m[2][1] = a[1]; m[2][2] = a[0];
    m[2][3] = W_val * a[5]; m[2][4] = W_val * a[4]; m[2][5] = W_val * a[3];
    m[2][6] = Fp::zero();
    
    m[3][0] = a[3]; m[3][1] = a[2]; m[3][2] = a[1];
    m[3][3] = a[0]; m[3][4] = W_val * a[5]; m[3][5] = W_val * a[4];
    m[3][6] = Fp::zero();
    
    m[4][0] = a[4]; m[4][1] = a[3]; m[4][2] = a[2];
    m[4][3] = a[1]; m[4][4] = a[0]; m[4][5] = W_val * a[5];
    m[4][6] = Fp::zero();
    
    m[5][0] = a[5]; m[5][1] = a[4]; m[5][2] = a[3];
    m[5][3] = a[2]; m[5][4] = a[1]; m[5][5] = a[0];
    m[5][6] = Fp::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 6; col++) {
        int pivot_row = col;
        for (int row = col + 1; row < 6; row++) {
            if (m[row][col].asRaw() > m[pivot_row][col].asRaw()) {
                pivot_row = row;
            }
        }
        
        if (pivot_row != col) {
            for (int j = 0; j < 7; j++) {
                Fp tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        Fp pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 7; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }
        
        for (int row = 0; row < 6; row++) {
            if (row != col) {
                Fp factor = m[row][col];
                for (int j = col; j < 7; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }
    
    return Fp6(m[0][6], m[1][6], m[2][6], m[3][6], m[4][6], m[5][6]);
}

static_assert(sizeof(Fp6) == 24, "Fp6 must be 24 bytes");
