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
 * (but not BabyBear). The reciprocal() function uses simple squaring to avoid this.
 */

#ifndef __SPPARK_FF_KOALA_BEAR_HPP__
#define __SPPARK_FF_KOALA_BEAR_HPP__

#ifdef __CUDACC__
# include <cassert>
# include "mont32_t.cuh"
# define inline __device__ __forceinline__

using kb31_base = mont32_t<31, 0x7F000001, 0x7EFFFFFF, 0x17F7EFE4, 0x01FFFFFE>;

struct kb31_t : public kb31_base {
    using mem_t = kb31_t;

    inline kb31_t() {}
    inline kb31_t(const kb31_base& a) : kb31_base(a) {}
    inline kb31_t(const uint32_t *p)  : kb31_base(p) {}
    // this is used in constant declaration, e.g. as kb31_t{11}
    __host__ __device__ constexpr kb31_t(int a)      : kb31_base(a) {}
    __host__ __device__ constexpr kb31_t(uint32_t a) : kb31_base(a) {}

    // Simple squaring that avoids the buggy optimized path in mont32_t::sqr_n
    // The optimized sqr_n in mont32_t has issues with KoalaBear's Montgomery constants
    inline kb31_t simple_sqr_n(kb31_t s, uint32_t n) const
    {
        for (uint32_t i = 0; i < n; i++) {
            s = s * s;
        }
        return s;
    }
    
    inline kb31_t simple_sqr_n_mul(kb31_t s, uint32_t n, kb31_t m) const
    {
        s = simple_sqr_n(s, n);
        return s * m;
    }
    
    // Compute x^(p-2) = x^{-1} using optimized addition chain
    // p - 2 = 0x7EFFFFFF = 2130706431 = 127 * 2^24 - 1
    // Strategy: x^(p-2) = x^(126*2^24) * x^(2^24-1)
    inline kb31_t reciprocal() const
    {
        kb31_t x03, x07, x15, x31, x63, x12m1, x24m1, ret = *this;

        // Build x^3
        x03 = simple_sqr_n_mul(ret, 1, ret);   // x^2 * x = x^3
        
        // Build x^(2^n - 1) chain
        x07 = simple_sqr_n_mul(x03, 1, ret);   // x^6 * x = x^7
        x15 = simple_sqr_n_mul(x07, 1, ret);   // x^14 * x = x^15
        x31 = simple_sqr_n_mul(x15, 1, ret);   // x^30 * x = x^31
        x63 = simple_sqr_n_mul(x31, 1, ret);   // x^62 * x = x^63
        
        // x^(2^12 - 1) = x^4095
        x12m1 = simple_sqr_n_mul(x63, 6, x63); // x^(63*64) * x^63 = x^4095
        
        // x^(2^24 - 1) = x^16777215
        x24m1 = simple_sqr_n_mul(x12m1, 12, x12m1); // x^(4095*4096) * x^4095
        
        // x^126 = (x^63)^2
        ret = simple_sqr_n(x63, 1);
        
        // x^(126 * 2^24) = (x^126)^(2^24)
        ret = simple_sqr_n(ret, 24);
        
        // x^(p-2) = x^(126*2^24 + 2^24-1) = x^(127*2^24 - 1)
        ret = ret * x24m1;

        return ret;
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
