/*
 * KoalaBear field implementation using mont32_t template
 * 
 * KoalaBear prime: p = 2^31 - 2^24 + 1 = 2130706433 = 0x7F000001
 * 2-adicity: 24 (p-1 = 2^24 * 127)
 * 
 * Montgomery constants:
 *   MOD = 0x7F000001  (the prime)
 *   M0  = 0x7EFFFFFF  (-p^{-1} mod 2^32)
 *   RR  = 0x17F7EFE4  (R^2 mod p, where R = 2^32)
 *   ONE = 0x01FFFFFE  (R mod p, i.e., 1 in Montgomery form)
 * 
 * NOTE: The optimized sqr_n path in mont32_t has a bug that affects KoalaBear
 * (but not BabyBear). We provide a corrected sqr_n here and use it in reciprocal().
 */

#ifndef __SPPARK_FF_KOALA_BEAR_HPP__
#define __SPPARK_FF_KOALA_BEAR_HPP__

#ifdef __CUDACC__
# include <cassert>
# include "ff/mont32_t.cuh"
# define inline __device__ __forceinline__

using kb31_base = mont32_t<31, 0x7F000001, 0x7EFFFFFF, 0x17F7EFE4, 0x01FFFFFE>;

struct kb31_t : public kb31_base {
    using mem_t = kb31_t;
    static constexpr uint32_t MOD = 0x7F000001;
    static constexpr uint32_t M0  = 0x7EFFFFFF;

    inline kb31_t() {}
    inline kb31_t(const kb31_base& a) : kb31_base(a) {}
    inline kb31_t(const uint32_t *p)  : kb31_base(p) {}
    // this is used in constant declaration, e.g. as kb31_t{11}
    __host__ __device__ constexpr kb31_t(int a)      : kb31_base(a) {}
    __host__ __device__ constexpr kb31_t(uint32_t a) : kb31_base(a) {}

    // Corrected squaring loop (avoids buggy mont32_t::sqr_n path)
    static inline kb31_t sqr_n(kb31_t s, uint32_t n)
    {
# if defined(__CUDA_ARCH__) && !defined(__clang_analyzer__)
#  ifdef __GNUC__
#   define KB_ASM __asm__ __volatile__
#  else
#   define KB_ASM asm volatile
#  endif
        while (n--) {
            uint32_t tmp0, tmp1, red;
            KB_ASM("mul.lo.u32 %0, %2, %2; mul.hi.u32 %1, %2, %2;"
                : "=r"(tmp0), "=r"(tmp1)
                : "r"(*s));
            KB_ASM("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp0), "r"(M0));
            KB_ASM("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
                : "+r"(tmp0), "=r"(tmp1)
                : "r"(red), "r"(MOD), "r"(tmp1));
            if (tmp1 >= MOD) tmp1 -= MOD;
            *s = tmp1;
        }
#  undef KB_ASM
# else
        while (n--) {
            s = s * s;
        }
# endif
        return s;
    }
    
    static inline kb31_t sqr_n_mul(kb31_t s, uint32_t n, kb31_t m)
    {
        s = sqr_n(s, n);
        return s * m;
    }
    
    // Compute x^(p-2) = x^{-1} using optimized addition chain
    // p - 2 = 0x7EFFFFFF = 2130706431 = 127 * 2^24 - 1
    // Strategy: x^(p-2) = x^(126*2^24) * x^(2^24-1)
    inline kb31_t reciprocal() const
    {
        kb31_t x03, x07, x15, x31, x63, x12m1, x24m1, ret = *this;

        // Build x^3
        x03 = sqr_n_mul(ret, 1, ret);   // x^2 * x = x^3
        
        // Build x^(2^n - 1) chain
        x07 = sqr_n_mul(x03, 1, ret);   // x^6 * x = x^7
        x15 = sqr_n_mul(x07, 1, ret);   // x^14 * x = x^15
        x31 = sqr_n_mul(x15, 1, ret);   // x^30 * x = x^31
        x63 = sqr_n_mul(x31, 1, ret);   // x^62 * x = x^63
        
        // x^(2^12 - 1) = x^4095
        x12m1 = sqr_n_mul(x63, 6, x63); // x^(63*64) * x^63 = x^4095
        
        // x^(2^24 - 1) = x^16777215
        x24m1 = sqr_n_mul(x12m1, 12, x12m1); // x^(4095*4096) * x^4095
        
        // x^126 = (x^63)^2
        ret = sqr_n(x63, 1);
        
        // x^(126 * 2^24) = (x^126)^(2^24)
        ret = sqr_n(ret, 24);
        
        // x^(p-2) = x^(126*2^24 + 2^24-1) = x^(127*2^24 - 1)
        ret = ret * x24m1;

        return ret;
    }

    inline kb31_t& operator^=(int p)
    {
        kb31_base::operator^=(static_cast<uint32_t>(p));
        return *this;
    }
    
    friend inline kb31_t operator/(int one, kb31_t a)
    {   assert(one == 1); return a.reciprocal();   }
    friend inline kb31_t operator/(kb31_t a, kb31_t b)
    {   return a * b.reciprocal();   }
    inline kb31_t& operator/=(const kb31_t a)
    {   *this *= a.reciprocal(); return *this;   }
};

# undef inline
#endif // __CUDACC__

#endif // __SPPARK_FF_KOALA_BEAR_HPP__
