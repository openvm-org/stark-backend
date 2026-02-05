/*
 * Fp5 - Quintic Extension of Baby Bear Field
 * 
 * Defines Fp5, a finite field F_p^5, based on Fp via the irreducible polynomial x^5 - 2.
 * 
 * Field size: p^5 ≈ 2^155, provides ~155 bits of security.
 * 
 * Element representation: a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
 * where each ai is a Baby Bear element and x^5 = 2.
 * 
 * Multiplication uses inline PTX assembly on CUDA for ~1.67x speedup over schoolbook.
 */

#pragma once

#include "fp.h"

/// Fp5 is a degree-5 extension of the Baby Bear field.
/// Elements are represented as polynomials in F_p[x] / (x^5 - 2).
struct Fp5 {
    /// Coefficients: elems[0] + elems[1]*x + elems[2]*x^2 + elems[3]*x^3 + elems[4]*x^4
    union { Fp elems[5]; uint32_t u[5]; };
    
    /// The non-residue W such that x^5 - W is irreducible
    /// W = 2 is the smallest valid choice for Baby Bear
    static constexpr uint32_t W = 2;
    
    // Montgomery constants for PTX multiplication
    static constexpr uint32_t MOD  = 0x78000001;
    static constexpr uint32_t M    = 0x77ffffff;
    static constexpr uint32_t BETA = 0x1ffffffc;  // (W << 32) % MOD
    
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
    /// c[0] = a0*b0 + 2*(a1*b4 + a2*b3 + a3*b2 + a4*b1)
    /// c[1] = a0*b1 + a1*b0 + 2*(a2*b4 + a3*b3 + a4*b2)
    /// c[2] = a0*b2 + a1*b1 + a2*b0 + 2*(a3*b4 + a4*b3)
    /// c[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + 2*(a4*b4)
    /// c[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    __device__ Fp5 operator*(Fp5 rhs) const {
        Fp5 ret;

# if defined(__CUDA_ARCH__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // PTX assembly for fused multiply-accumulate with inline Montgomery reduction
        // Operand mapping: %1=a0, %2=a1, %3=a2, %4=a3, %5=a4
        //                  %6=b0, %7=b1, %8=b2, %9=b3, %10=b4
        //                  %11=MOD, %12=M, %13=BETA
        
        // ret[0] = a0*b0 + BETA * (a1*b4 + a2*b3 + a3*b2 + a4*b1)
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %2, %10;     mul.hi.u32  %hi, %2, %10;\n\t"   // a1*b4
            "mad.lo.cc.u32 %lo, %3, %9, %lo; madc.hi.u32 %hi, %3, %9, %hi;\n\t"  // +a2*b3
            "mad.lo.cc.u32 %lo, %4, %8, %lo; madc.hi.u32 %hi, %4, %8, %hi;\n\t"  // +a3*b2
            "mad.lo.cc.u32 %lo, %5, %7, %lo; madc.hi.u32 %hi, %5, %7, %hi;\n\t"  // +a4*b1
            // Double reduction for 4 products (hi can be up to ~1.88*MOD)
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %hi, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %lo, %hi, %13;    mul.hi.u32  %hi, %hi, %13;\n\t"  // *BETA
            "mad.lo.cc.u32 %lo, %1, %6, %lo; madc.hi.u32 %hi, %1, %6, %hi;\n\t"  // +a0*b0
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %0, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[1] = a0*b1 + a1*b0 + BETA * (a2*b4 + a3*b3 + a4*b2)
        // After BETA part + 2 products, hi can be up to ~1.94*MOD, need double reduction
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %3, %10;     mul.hi.u32  %hi, %3, %10;\n\t"   // a2*b4
            "mad.lo.cc.u32 %lo, %4, %9, %lo; madc.hi.u32 %hi, %4, %9, %hi;\n\t"  // +a3*b3
            "mad.lo.cc.u32 %lo, %5, %8, %lo; madc.hi.u32 %hi, %5, %8, %hi;\n\t"  // +a4*b2
            // Double reduction for 3 products (hi up to ~1.41*MOD)
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %hi, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %lo, %hi, %13;    mul.hi.u32  %hi, %hi, %13;\n\t"  // *BETA
            "mad.lo.cc.u32 %lo, %1, %7, %lo; madc.hi.u32 %hi, %1, %7, %hi;\n\t"  // +a0*b1
            "mad.lo.cc.u32 %lo, %2, %6, %lo; madc.hi.u32 %hi, %2, %6, %hi;\n\t"  // +a1*b0
            // Double reduction for hi up to ~1.94*MOD
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %0, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[2] = a0*b2 + a1*b1 + a2*b0 + BETA * (a3*b4 + a4*b3)
        // After BETA part + 3 products, hi can be up to ~2.41*MOD, need triple reduction
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %10;     mul.hi.u32  %hi, %4, %10;\n\t"   // a3*b4
            "mad.lo.cc.u32 %lo, %5, %9, %lo; madc.hi.u32 %hi, %5, %9, %hi;\n\t"  // +a4*b3
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %hi, %m, %11, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %13;    mul.hi.u32  %hi, %hi, %13;\n\t"  // *BETA
            "mad.lo.cc.u32 %lo, %1, %8, %lo; madc.hi.u32 %hi, %1, %8, %hi;\n\t"  // +a0*b2
            "mad.lo.cc.u32 %lo, %2, %7, %lo; madc.hi.u32 %hi, %2, %7, %hi;\n\t"  // +a1*b1
            "mad.lo.cc.u32 %lo, %3, %6, %lo; madc.hi.u32 %hi, %3, %6, %hi;\n\t"  // +a2*b0
            // Triple reduction for hi up to ~2.41*MOD
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %0, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + BETA * (a4*b4)
        // After BETA part + 4 products, hi can be up to ~2.88*MOD, need triple reduction
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %5, %10;     mul.hi.u32  %hi, %5, %10;\n\t"   // a4*b4
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %hi, %m, %11, %hi;\n\t"
            "mul.lo.u32    %lo, %hi, %13;    mul.hi.u32  %hi, %hi, %13;\n\t"  // *BETA
            "mad.lo.cc.u32 %lo, %1, %9, %lo; madc.hi.u32 %hi, %1, %9, %hi;\n\t"  // +a0*b3
            "mad.lo.cc.u32 %lo, %2, %8, %lo; madc.hi.u32 %hi, %2, %8, %hi;\n\t"  // +a1*b2
            "mad.lo.cc.u32 %lo, %3, %7, %lo; madc.hi.u32 %hi, %3, %7, %hi;\n\t"  // +a2*b1
            "mad.lo.cc.u32 %lo, %4, %6, %lo; madc.hi.u32 %hi, %4, %6, %hi;\n\t"  // +a3*b0
            // Triple reduction for hi up to ~2.88*MOD
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %0, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 (no BETA factor)
        // 5 products, hi can be up to ~2.35*MOD, need triple reduction
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %1, %10;     mul.hi.u32  %hi, %1, %10;\n\t"   // a0*b4
            "mad.lo.cc.u32 %lo, %2, %9, %lo; madc.hi.u32 %hi, %2, %9, %hi;\n\t"  // +a1*b3
            "mad.lo.cc.u32 %lo, %3, %8, %lo; madc.hi.u32 %hi, %3, %8, %hi;\n\t"  // +a2*b2
            "mad.lo.cc.u32 %lo, %4, %7, %lo; madc.hi.u32 %hi, %4, %7, %hi;\n\t"  // +a3*b1
            "mad.lo.cc.u32 %lo, %5, %6, %lo; madc.hi.u32 %hi, %5, %6, %hi;\n\t"  // +a4*b0
            // Triple reduction for hi up to ~2.35*MOD
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "setp.ge.u32   %p, %hi, %11;\n\t"
            "@%p sub.u32   %hi, %hi, %11;\n\t"
            "mul.lo.u32    %m, %lo, %12;\n\t"
            "mad.lo.cc.u32 %lo, %m, %11, %lo; madc.hi.u32 %0, %m, %11, %hi;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "setp.ge.u32   %p, %0, %11;\n\t"
            "@%p sub.u32   %0, %0, %11;\n\t"
            "}" : "=r"(ret.u[4])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]),
                  "r"(MOD), "r"(M), "r"(BETA));
#  undef asm
# else
        // Fallback: schoolbook multiplication
        const Fp a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        const Fp b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2], b3 = rhs.elems[3], b4 = rhs.elems[4];
        
        Fp c0 = a0 * b0;
        Fp t5_half = a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1;
        ret.elems[0] = c0 + t5_half + t5_half;
        
        Fp c1 = a0 * b1 + a1 * b0;
        Fp t6_half = a2 * b4 + a3 * b3 + a4 * b2;
        ret.elems[1] = c1 + t6_half + t6_half;
        
        Fp c2 = a0 * b2 + a1 * b1 + a2 * b0;
        Fp t7_half = a3 * b4 + a4 * b3;
        ret.elems[2] = c2 + t7_half + t7_half;
        
        Fp c3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        Fp a4b4 = a4 * b4;
        ret.elems[3] = c3 + a4b4 + a4b4;
        
        ret.elems[4] = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
# endif

        return ret;
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

/// Frobenius endomorphism: φ(a) = a^p
/// For X^5 - W with p ≡ 1 (mod 5), φ(a_i * x^i) = FROB[i] * a_i * x^i
/// where FROB[i] = W^(i*(p-1)/5)
__device__ inline Fp5 frobenius(Fp5 a, int power) {
    // Frobenius constants: FROB[i]^j for i=1..4, j=1..4
    // FROB[i] = W^(i*k) where k = (p-1)/5
    static constexpr uint32_t FROB_TABLE[4][4] = {
        {0x309476E5, 0x24553874, 0x749B8749, 0x267AC95F},  // FROB[1]^1..4
        {0x24553874, 0x267AC95F, 0x309476E5, 0x749B8749},  // FROB[2]^1..4
        {0x749B8749, 0x309476E5, 0x267AC95F, 0x24553874},  // FROB[3]^1..4
        {0x267AC95F, 0x749B8749, 0x24553874, 0x309476E5},  // FROB[4]^1..4
    };
    
    int j = power - 1;  // 0-indexed for array access
    return Fp5(
        a.elems[0],
        a.elems[1] * Fp(FROB_TABLE[0][j]),
        a.elems[2] * Fp(FROB_TABLE[1][j]),
        a.elems[3] * Fp(FROB_TABLE[2][j]),
        a.elems[4] * Fp(FROB_TABLE[3][j])
    );
}

/// Inversion using Frobenius-based norm computation.
/// 
/// For a in Fp5, we use:
///   N(a) = a * φ(a) * φ²(a) * φ³(a) * φ⁴(a)  (norm, in Fp)
///   a⁻¹ = φ(a) * φ²(a) * φ³(a) * φ⁴(a) / N(a)
/// 
/// This requires only 1 base field inversion vs 5 for Gaussian elimination.
/// Cost: 4 Frobenius (16 Fp muls) + 4 Fp5 muls + 1 Fp inv + 5 Fp muls
__device__ inline Fp5 inv(Fp5 x) {
    // Handle zero case
    if (x == Fp5::zero()) {
        return Fp5::zero();
    }
    
    // Compute Frobenius conjugates
    Fp5 phi1 = frobenius(x, 1);  // φ(a)
    Fp5 phi2 = frobenius(x, 2);  // φ²(a)
    Fp5 phi3 = frobenius(x, 3);  // φ³(a)
    Fp5 phi4 = frobenius(x, 4);  // φ⁴(a)
    
    // Compute conjugate product: c = φ(a) * φ²(a) * φ³(a) * φ⁴(a)
    // Use ladder: c12 = φ(a) * φ²(a), c34 = φ³(a) * φ⁴(a), c = c12 * c34
    Fp5 c12 = phi1 * phi2;
    Fp5 c34 = phi3 * phi4;
    Fp5 conj = c12 * c34;
    
    // Compute norm: N(a) = a * conj
    // The result should be in Fp (all higher coefficients are 0)
    Fp5 norm_full = x * conj;
    Fp norm = norm_full.elems[0];  // Norm is in the base field
    
    // Compute inverse: a⁻¹ = conj / N(a)
    Fp norm_inv = ::inv(norm);
    return Fp5(
        conj.elems[0] * norm_inv,
        conj.elems[1] * norm_inv,
        conj.elems[2] * norm_inv,
        conj.elems[3] * norm_inv,
        conj.elems[4] * norm_inv
    );
}

static_assert(sizeof(Fp5) == 20, "Fp5 must be 20 bytes");
