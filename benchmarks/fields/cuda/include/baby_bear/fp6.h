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
    static constexpr uint32_t BETA = 0x0FFFFFBE;  // (W << 32) % MOD = 31 * 2^32 mod MOD
    static constexpr uint32_t R2MOD = 0x0FFFFFFE; // 2^32 mod MOD (for 64-bit reduction)
    
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
    /// 
    /// PTX Strategy: Group products into <=2 terms so Montgomery reduction inputs
    /// stay below R*MOD. Apply one Montgomery reduction per group, then add group
    /// results modulo MOD before the final BETA multiplication.
    __device__ Fp6 operator*(Fp6 rhs) const {
        Fp6 ret;
        
# if defined(__CUDA_ARCH__) && !defined(__clang__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // PTX assembly for Fp6 multiplication
        // Key insight: REDC is linear, so we can reduce grouped sums and then add.
        // We split large sums into <=2-product groups, REDC each group, then combine and apply BETA.
        // Operand mapping: %1=a0..%6=a5, %7=b0..%12=b5, %13=MOD, %14=M, %15=R2MOD, %16=BETA
        
        // Helper macro for Montgomery reduction of 64-bit sum/product
        // Input in (lo, hi), result in hi after reduction
        #define MONT_REDUCE() \
            "mul.lo.u32 %m, %lo, %14;\n\t" \
            "mad.lo.cc.u32 %lo, %m, %13, %lo;\n\t" \
            "madc.hi.u32 %hi, %m, %13, %hi;\n\t"

        // ret[0] = a0*b0 + BETA*(a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1)
        asm("{ .reg.b32 %lo, %hi, %m, %r_beta1, %r_beta2, %r_beta3, %r_beta, %r_nb, %tmp; .reg.pred %p;\n\t"
            // === BETA part: 5 products (2 + 2 + 1 groups) ===
            // group1: a1*b5 + a2*b4
            "mul.lo.u32 %lo, %2, %12;\n\t" "mul.hi.u32 %hi, %2, %12;\n\t"
            "mad.lo.cc.u32 %lo, %3, %11, %lo;\n\t" "madc.hi.u32 %hi, %3, %11, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta1, %hi;\n\t"
            // group2: a3*b3 + a4*b2
            "mul.lo.u32 %lo, %4, %10;\n\t" "mul.hi.u32 %hi, %4, %10;\n\t"
            "mad.lo.cc.u32 %lo, %5, %9, %lo;\n\t" "madc.hi.u32 %hi, %5, %9, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta2, %hi;\n\t"
            // group3: a5*b1
            "mul.lo.u32 %lo, %6, %8;\n\t" "mul.hi.u32 %hi, %6, %8;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta3, %hi;\n\t"
            // sum beta groups
            "add.u32 %tmp, %r_beta1, %r_beta2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            "add.u32 %tmp, %tmp, %r_beta3;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            // multiply by BETA and reduce
            "mul.lo.u32 %lo, %tmp, %16;\n\t"
            "mul.hi.u32 %hi, %tmp, %16;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: a0*b0 ===
            "mul.lo.u32 %lo, %1, %7;\n\t" "mul.hi.u32 %hi, %1, %7;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb, %hi;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %r_nb;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[1] = a0*b1 + a1*b0 + BETA*(a2*b5 + a3*b4 + a4*b3 + a5*b2)
        asm("{ .reg.b32 %lo, %hi, %m, %r_beta1, %r_beta2, %r_beta, %r_nb, %tmp; .reg.pred %p;\n\t"
            // === BETA part: 4 products (2 + 2 groups) ===
            // group1: a2*b5 + a3*b4
            "mul.lo.u32 %lo, %3, %12;\n\t" "mul.hi.u32 %hi, %3, %12;\n\t"
            "mad.lo.cc.u32 %lo, %4, %11, %lo;\n\t" "madc.hi.u32 %hi, %4, %11, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta1, %hi;\n\t"
            // group2: a4*b3 + a5*b2
            "mul.lo.u32 %lo, %5, %10;\n\t" "mul.hi.u32 %hi, %5, %10;\n\t"
            "mad.lo.cc.u32 %lo, %6, %9, %lo;\n\t" "madc.hi.u32 %hi, %6, %9, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta2, %hi;\n\t"
            // sum beta groups
            "add.u32 %tmp, %r_beta1, %r_beta2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %tmp, %16;\n\t" "mul.hi.u32 %hi, %tmp, %16;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: a0*b1 + a1*b0 (2 products) ===
            "mul.lo.u32 %lo, %1, %8;\n\t" "mul.hi.u32 %hi, %1, %8;\n\t"
            "mad.lo.cc.u32 %lo, %2, %7, %lo;\n\t" "madc.hi.u32 %hi, %2, %7, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb, %hi;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %r_nb;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[2] = a0*b2 + a1*b1 + a2*b0 + BETA*(a3*b5 + a4*b4 + a5*b3)
        asm("{ .reg.b32 %lo, %hi, %m, %r_beta1, %r_beta2, %r_beta, %r_nb1, %r_nb2, %tmp; .reg.pred %p;\n\t"
            // === BETA part: 3 products (2 + 1 groups) ===
            // group1: a3*b5 + a4*b4
            "mul.lo.u32 %lo, %4, %12;\n\t" "mul.hi.u32 %hi, %4, %12;\n\t"
            "mad.lo.cc.u32 %lo, %5, %11, %lo;\n\t" "madc.hi.u32 %hi, %5, %11, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta1, %hi;\n\t"
            // group2: a5*b3
            "mul.lo.u32 %lo, %6, %10;\n\t" "mul.hi.u32 %hi, %6, %10;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta2, %hi;\n\t"
            // sum beta groups
            "add.u32 %tmp, %r_beta1, %r_beta2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %tmp, %16;\n\t" "mul.hi.u32 %hi, %tmp, %16;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: a0*b2 + a1*b1 + a2*b0 (2 + 1 groups) ===
            // group1: a0*b2 + a1*b1
            "mul.lo.u32 %lo, %1, %9;\n\t" "mul.hi.u32 %hi, %1, %9;\n\t"
            "mad.lo.cc.u32 %lo, %2, %8, %lo;\n\t" "madc.hi.u32 %hi, %2, %8, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb1, %hi;\n\t"
            // group2: a2*b0
            "mul.lo.u32 %lo, %3, %7;\n\t" "mul.hi.u32 %hi, %3, %7;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb2, %hi;\n\t"
            // sum non-BETA groups
            "add.u32 %tmp, %r_nb1, %r_nb2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %tmp;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + BETA*(a4*b5 + a5*b4)
        asm("{ .reg.b32 %lo, %hi, %m, %r_beta, %r_nb1, %r_nb2, %tmp; .reg.pred %p;\n\t"
            // === BETA part: 2 products ===
            "mul.lo.u32 %lo, %5, %12;\n\t" "mul.hi.u32 %hi, %5, %12;\n\t"
            "mad.lo.cc.u32 %lo, %6, %11, %lo;\n\t" "madc.hi.u32 %hi, %6, %11, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %r_beta, %16;\n\t" "mul.hi.u32 %hi, %r_beta, %16;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: 4 products (2 + 2 groups) ===
            // group1: a0*b3 + a1*b2
            "mul.lo.u32 %lo, %1, %10;\n\t" "mul.hi.u32 %hi, %1, %10;\n\t"
            "mad.lo.cc.u32 %lo, %2, %9, %lo;\n\t" "madc.hi.u32 %hi, %2, %9, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb1, %hi;\n\t"
            // group2: a2*b1 + a3*b0
            "mul.lo.u32 %lo, %3, %8;\n\t" "mul.hi.u32 %hi, %3, %8;\n\t"
            "mad.lo.cc.u32 %lo, %4, %7, %lo;\n\t" "madc.hi.u32 %hi, %4, %7, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb2, %hi;\n\t"
            // sum non-BETA groups
            "add.u32 %tmp, %r_nb1, %r_nb2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %tmp;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + BETA*(a5*b5)
        asm("{ .reg.b32 %lo, %hi, %m, %r_beta, %r_nb1, %r_nb2, %r_nb3, %tmp; .reg.pred %p;\n\t"
            // === BETA part: 1 product ===
            "mul.lo.u32 %lo, %6, %12;\n\t" "mul.hi.u32 %hi, %6, %12;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %r_beta, %16;\n\t" "mul.hi.u32 %hi, %r_beta, %16;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: 5 products (2 + 2 + 1 groups) ===
            // group1: a0*b4 + a1*b3
            "mul.lo.u32 %lo, %1, %11;\n\t" "mul.hi.u32 %hi, %1, %11;\n\t"
            "mad.lo.cc.u32 %lo, %2, %10, %lo;\n\t" "madc.hi.u32 %hi, %2, %10, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb1, %hi;\n\t"
            // group2: a2*b2 + a3*b1
            "mul.lo.u32 %lo, %3, %9;\n\t" "mul.hi.u32 %hi, %3, %9;\n\t"
            "mad.lo.cc.u32 %lo, %4, %8, %lo;\n\t" "madc.hi.u32 %hi, %4, %8, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb2, %hi;\n\t"
            // group3: a4*b0
            "mul.lo.u32 %lo, %5, %7;\n\t" "mul.hi.u32 %hi, %5, %7;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb3, %hi;\n\t"
            // sum non-BETA groups
            "add.u32 %tmp, %r_nb1, %r_nb2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            "add.u32 %tmp, %tmp, %r_nb3;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %tmp;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[4])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[5] = a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0 (no BETA)
        asm("{ .reg.b32 %lo, %hi, %m, %r_nb1, %r_nb2, %r_nb3, %tmp; .reg.pred %p;\n\t"
            // group1: a0*b5 + a1*b4
            "mul.lo.u32 %lo, %1, %12;\n\t" "mul.hi.u32 %hi, %1, %12;\n\t"
            "mad.lo.cc.u32 %lo, %2, %11, %lo;\n\t" "madc.hi.u32 %hi, %2, %11, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb1, %hi;\n\t"
            // group2: a2*b3 + a3*b2
            "mul.lo.u32 %lo, %3, %10;\n\t" "mul.hi.u32 %hi, %3, %10;\n\t"
            "mad.lo.cc.u32 %lo, %4, %9, %lo;\n\t" "madc.hi.u32 %hi, %4, %9, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb2, %hi;\n\t"
            // group3: a4*b1 + a5*b0
            "mul.lo.u32 %lo, %5, %8;\n\t" "mul.hi.u32 %hi, %5, %8;\n\t"
            "mad.lo.cc.u32 %lo, %6, %7, %lo;\n\t" "madc.hi.u32 %hi, %6, %7, %hi;\n\t"
            MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_nb3, %hi;\n\t"
            // sum groups
            "add.u32 %tmp, %r_nb1, %r_nb2;\n\t"
            "setp.ge.u32 %p, %tmp, %13;\n\t" "@%p sub.u32 %tmp, %tmp, %13;\n\t"
            "add.u32 %0, %tmp, %r_nb3;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[5])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        #undef MONT_REDUCE
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

/// Frobenius endomorphism: φ^power(a) = a^(p^power)
/// For X^6 - W with p ≡ 1 (mod 6), φ(a_i * x^i) = FROB[i][power] * a_i * x^i
/// where FROB[i][j] = ω^(i*j) and ω = W^((p-1)/6) is a primitive 6th root of unity.
__device__ inline Fp6 frobenius(Fp6 a, int power) {
    // Frobenius constants: ω^(i*j) for i=1..5, j=1..5
    // ω = 31^((p-1)/6) = 31^335544320 mod p = 0x4E5D1534
    // ω^2 = 0x4E5D1533, ω^3 = 0x78000000 = -1, ω^4 = 0x29A2EACD, ω^5 = 0x29A2EACE
    static constexpr uint32_t FROB_TABLE[5][5] = {
        {0x4E5D1534, 0x4E5D1533, 0x78000000, 0x29A2EACD, 0x29A2EACE},  // i=1
        {0x4E5D1533, 0x29A2EACD, 0x00000001, 0x4E5D1533, 0x29A2EACD},  // i=2
        {0x78000000, 0x00000001, 0x78000000, 0x00000001, 0x78000000},  // i=3
        {0x29A2EACD, 0x4E5D1533, 0x00000001, 0x29A2EACD, 0x4E5D1533},  // i=4
        {0x29A2EACE, 0x29A2EACD, 0x78000000, 0x4E5D1533, 0x4E5D1534},  // i=5
    };
    
    int j = power - 1;  // 0-indexed for array access
    return Fp6(
        a.elems[0],
        a.elems[1] * Fp(FROB_TABLE[0][j]),
        a.elems[2] * Fp(FROB_TABLE[1][j]),
        a.elems[3] * Fp(FROB_TABLE[2][j]),
        a.elems[4] * Fp(FROB_TABLE[3][j]),
        a.elems[5] * Fp(FROB_TABLE[4][j])
    );
}

/// Inversion using Itoh-Tsujii with free φ³.
///
/// For x^6 - 31 with p ≡ 1 (mod 6): ω³ = 31^{(p-1)/2} = -1 (31 is not a QR),
/// so φ³ just negates odd coefficients (zero multiplications).
///
/// Chain:  f12 = φ(x)·φ²(x),  f345 = φ³(f12)·φ³(x),  conj = f12·f345
/// Cost: 2 Frobenius (10 Fp muls) + 3 Fp6 muls + partial norm + 1 Fp inv
__device__ inline Fp6 inv(Fp6 x) {
    if (x == Fp6::zero()) { return Fp6::zero(); }

    Fp6 f1 = frobenius(x, 1);     // x^p
    Fp6 f2 = frobenius(x, 2);     // x^{p²}
    Fp6 f12 = f1 * f2;            // x^{p+p²}

    // φ³ is free: (a₀,-a₁,a₂,-a₃,a₄,-a₅)
    Fp6 f3_f12(f12[0], -f12[1], f12[2], -f12[3], f12[4], -f12[5]);  // x^{p⁴+p⁵}
    Fp6 f3_x(x[0], -x[1], x[2], -x[3], x[4], -x[5]);              // x^{p³}

    Fp6 f345 = f3_f12 * f3_x;    // x^{p³+p⁴+p⁵}
    Fp6 conj = f12 * f345;        // x^{p+p²+p³+p⁴+p⁵}

    // Partial norm: constant coefficient of x·conj
    // For x^6 - W: c₀ = x₀c₀ + W·(x₁c₅ + x₂c₄ + x₃c₃ + x₄c₂ + x₅c₁)
    Fp t = x[1]*conj[5] + x[2]*conj[4] + x[3]*conj[3] + x[4]*conj[2] + x[5]*conj[1];
    Fp norm = x[0]*conj[0] + Fp6::mulByW(t);

    Fp norm_inv = ::inv(norm);
    return conj * norm_inv;
}

static_assert(sizeof(Fp6) == 24, "Fp6 must be 24 bytes");
