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
    /// PTX Strategy: Use 64-bit accumulator to handle overflow when summing
    /// Montgomery-reduced products. Each reduced product is < 2*MOD, and we 
    /// sum up to 6 products (max sum < 12*MOD). The 64-bit sum is reduced
    /// using: (acc_lo + acc_hi * 2^32) mod MOD = (acc_lo + acc_hi * R2MOD) mod MOD
    __device__ Fp6 operator*(Fp6 rhs) const {
        Fp6 ret;
        
# if defined(__CUDA_ARCH__) && !defined(__clang__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // PTX assembly for Fp6 multiplication
        // Key insight: schoolbook computes ret[i] = sum(REDC(a*b)) + REDC(BETA * sum(REDC(a*b)))
        // We must do Montgomery reduction BEFORE summing, then BETA multiply, then add.
        // Operand mapping: %1=a0..%6=a5, %7=b0..%12=b5, %13=MOD, %14=M, %15=R2MOD, %16=BETA
        
        // Helper macro for Montgomery reduction of single product
        // Product in (lo, hi), result in hi after reduction
        #define MONT_REDUCE() \
            "mul.lo.u32 %m, %lo, %14;\n\t" \
            "mad.lo.cc.u32 %lo, %m, %13, %lo;\n\t" \
            "madc.hi.u32 %hi, %m, %13, %hi;\n\t"

        // ret[0] = a0*b0 + BETA*(a1*b5 + a2*b4 + a3*b3 + a4*b2 + a5*b1)
        asm("{ .reg.b32 %lo, %hi, %m, %acc_lo, %acc_hi, %t, %r_beta, %r_nb; .reg.pred %p;\n\t"
            // === BETA part: 5 products ===
            "mov.u32 %acc_lo, 0;\n\t"
            "mov.u32 %acc_hi, 0;\n\t"
            // a1*b5
            "mul.lo.u32 %lo, %2, %12;\n\t" "mul.hi.u32 %hi, %2, %12;\n\t"
            MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // a2*b4
            "mul.lo.u32 %lo, %3, %11;\n\t" "mul.hi.u32 %hi, %3, %11;\n\t"
            MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // a3*b3
            "mul.lo.u32 %lo, %4, %10;\n\t" "mul.hi.u32 %hi, %4, %10;\n\t"
            MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // a4*b2
            "mul.lo.u32 %lo, %5, %9;\n\t" "mul.hi.u32 %hi, %5, %9;\n\t"
            MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // a5*b1
            "mul.lo.u32 %lo, %6, %8;\n\t" "mul.hi.u32 %hi, %6, %8;\n\t"
            MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit sum to 32-bit
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t"
            "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t"
            "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            // Reduce to < MOD
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            // Multiply by BETA and Montgomery reduce
            "mul.lo.u32 %lo, %acc_lo, %16;\n\t"
            "mul.hi.u32 %hi, %acc_lo, %16;\n\t"
            MONT_REDUCE()
            // Reduce hi to < MOD, store in r_beta
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
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[1] = a0*b1 + a1*b0 + BETA*(a2*b5 + a3*b4 + a4*b3 + a5*b2)
        asm("{ .reg.b32 %lo, %hi, %m, %acc_lo, %acc_hi, %t, %r_beta, %r_nb; .reg.pred %p;\n\t"
            // === BETA part: 4 products ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %3, %12;\n\t" "mul.hi.u32 %hi, %3, %12;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %4, %11;\n\t" "mul.hi.u32 %hi, %4, %11;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %5, %10;\n\t" "mul.hi.u32 %hi, %5, %10;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %6, %9;\n\t" "mul.hi.u32 %hi, %6, %9;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %acc_lo, %16;\n\t" "mul.hi.u32 %hi, %acc_lo, %16;\n\t" MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: a0*b1 + a1*b0 (2 products) ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %1, %8;\n\t" "mul.hi.u32 %hi, %1, %8;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %2, %7;\n\t" "mul.hi.u32 %hi, %2, %7;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce (acc_hi max 1 for 2 products)
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "mov.u32 %r_nb, %acc_lo;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %r_nb;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[2] = a0*b2 + a1*b1 + a2*b0 + BETA*(a3*b5 + a4*b4 + a5*b3)
        asm("{ .reg.b32 %lo, %hi, %m, %acc_lo, %acc_hi, %t, %r_beta, %r_nb; .reg.pred %p;\n\t"
            // === BETA part: 3 products ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %4, %12;\n\t" "mul.hi.u32 %hi, %4, %12;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %5, %11;\n\t" "mul.hi.u32 %hi, %5, %11;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %6, %10;\n\t" "mul.hi.u32 %hi, %6, %10;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit with carry handling
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %acc_lo, %16;\n\t" "mul.hi.u32 %hi, %acc_lo, %16;\n\t" MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: a0*b2 + a1*b1 + a2*b0 (3 products) ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %1, %9;\n\t" "mul.hi.u32 %hi, %1, %9;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %2, %8;\n\t" "mul.hi.u32 %hi, %2, %8;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %3, %7;\n\t" "mul.hi.u32 %hi, %3, %7;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit with carry handling
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "mov.u32 %r_nb, %acc_lo;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %r_nb;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0 + BETA*(a4*b5 + a5*b4)
        asm("{ .reg.b32 %lo, %hi, %m, %acc_lo, %acc_hi, %t, %r_beta, %r_nb; .reg.pred %p;\n\t"
            // === BETA part: 2 products ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %5, %12;\n\t" "mul.hi.u32 %hi, %5, %12;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %6, %11;\n\t" "mul.hi.u32 %hi, %6, %11;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit with carry handling
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %acc_lo, %16;\n\t" "mul.hi.u32 %hi, %acc_lo, %16;\n\t" MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: 4 products ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %1, %10;\n\t" "mul.hi.u32 %hi, %1, %10;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %2, %9;\n\t" "mul.hi.u32 %hi, %2, %9;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %3, %8;\n\t" "mul.hi.u32 %hi, %3, %8;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %4, %7;\n\t" "mul.hi.u32 %hi, %4, %7;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit with carry handling
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "mov.u32 %r_nb, %acc_lo;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %r_nb;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[4] = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + BETA*(a5*b5)
        asm("{ .reg.b32 %lo, %hi, %m, %acc_lo, %acc_hi, %t, %r_beta, %r_nb; .reg.pred %p;\n\t"
            // === BETA part: 1 product ===
            "mul.lo.u32 %lo, %6, %12;\n\t" "mul.hi.u32 %hi, %6, %12;\n\t" MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            // BETA multiply + reduce
            "mul.lo.u32 %lo, %hi, %16;\n\t" "mul.hi.u32 %hi, %hi, %16;\n\t" MONT_REDUCE()
            "setp.ge.u32 %p, %hi, %13;\n\t" "@%p sub.u32 %hi, %hi, %13;\n\t"
            "mov.u32 %r_beta, %hi;\n\t"
            // === Non-BETA part: 5 products ===
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %1, %11;\n\t" "mul.hi.u32 %hi, %1, %11;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %2, %10;\n\t" "mul.hi.u32 %hi, %2, %10;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %3, %9;\n\t" "mul.hi.u32 %hi, %3, %9;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %4, %8;\n\t" "mul.hi.u32 %hi, %4, %8;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %5, %7;\n\t" "mul.hi.u32 %hi, %5, %7;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit with carry handling
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "mov.u32 %r_nb, %acc_lo;\n\t"
            // === Final sum ===
            "add.u32 %0, %r_beta, %r_nb;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "setp.ge.u32 %p, %0, %13;\n\t" "@%p sub.u32 %0, %0, %13;\n\t"
            "}" : "=r"(ret.u[4])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]),
                  "r"(rhs.u[0]), "r"(rhs.u[1]), "r"(rhs.u[2]), "r"(rhs.u[3]), "r"(rhs.u[4]), "r"(rhs.u[5]),
                  "r"(MOD), "r"(M), "r"(R2MOD), "r"(BETA));

        // ret[5] = a0*b5 + a1*b4 + a2*b3 + a3*b2 + a4*b1 + a5*b0 (no BETA)
        asm("{ .reg.b32 %lo, %hi, %m, %acc_lo, %acc_hi, %t; .reg.pred %p;\n\t"
            "mov.u32 %acc_lo, 0;\n\t" "mov.u32 %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %1, %12;\n\t" "mul.hi.u32 %hi, %1, %12;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %2, %11;\n\t" "mul.hi.u32 %hi, %2, %11;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %3, %10;\n\t" "mul.hi.u32 %hi, %3, %10;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %4, %9;\n\t" "mul.hi.u32 %hi, %4, %9;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %5, %8;\n\t" "mul.hi.u32 %hi, %5, %8;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            "mul.lo.u32 %lo, %6, %7;\n\t" "mul.hi.u32 %hi, %6, %7;\n\t" MONT_REDUCE()
            "add.cc.u32 %acc_lo, %acc_lo, %hi;\n\t" "addc.u32 %acc_hi, %acc_hi, 0;\n\t"
            // Reduce 64-bit to 32-bit
            "mul.lo.u32 %t, %acc_hi, %15;\n\t"
            "add.cc.u32 %acc_lo, %acc_lo, %t;\n\t" "addc.u32 %acc_hi, 0, 0;\n\t"
            "setp.ne.u32 %p, %acc_hi, 0;\n\t" "@%p add.u32 %acc_lo, %acc_lo, %15;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "setp.ge.u32 %p, %acc_lo, %13;\n\t" "@%p sub.u32 %acc_lo, %acc_lo, %13;\n\t"
            "mov.u32 %0, %acc_lo;\n\t"
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

/// Inversion using Frobenius-based norm computation.
/// 
/// For a in Fp6, we use:
///   N(a) = a * φ(a) * φ²(a) * φ³(a) * φ⁴(a) * φ⁵(a)  (norm, in Fp)
///   a⁻¹ = φ(a) * φ²(a) * φ³(a) * φ⁴(a) * φ⁵(a) / N(a)
/// 
/// This requires only 1 base field inversion vs 6 for Gaussian elimination.
/// Cost: 5 Frobenius (25 Fp muls) + 5 Fp6 muls + 1 Fp inv + 6 Fp muls
__device__ inline Fp6 inv(Fp6 x) {
    // Handle zero case
    if (x == Fp6::zero()) {
        return Fp6::zero();
    }
    
    // Compute Frobenius conjugates
    Fp6 phi1 = frobenius(x, 1);  // φ(a)
    Fp6 phi2 = frobenius(x, 2);  // φ²(a)
    Fp6 phi3 = frobenius(x, 3);  // φ³(a)
    Fp6 phi4 = frobenius(x, 4);  // φ⁴(a)
    Fp6 phi5 = frobenius(x, 5);  // φ⁵(a)
    
    // Compute conjugate product: c = φ(a) * φ²(a) * φ³(a) * φ⁴(a) * φ⁵(a)
    // Use balanced tree: ((φ1 * φ2) * (φ3 * φ4)) * φ5
    Fp6 c12 = phi1 * phi2;
    Fp6 c34 = phi3 * phi4;
    Fp6 c1234 = c12 * c34;
    Fp6 conj = c1234 * phi5;
    
    // Compute norm: N(a) = a * conj
    // The result should be in Fp (all higher coefficients are 0)
    Fp6 norm_full = x * conj;
    Fp norm = norm_full.elems[0];  // Norm is in the base field
    
    // Compute inverse: a⁻¹ = conj / N(a)
    Fp norm_inv = ::inv(norm);
    return Fp6(
        conj.elems[0] * norm_inv,
        conj.elems[1] * norm_inv,
        conj.elems[2] * norm_inv,
        conj.elems[3] * norm_inv,
        conj.elems[4] * norm_inv,
        conj.elems[5] * norm_inv
    );
}

static_assert(sizeof(Fp6) == 24, "Fp6 must be 24 bytes");
