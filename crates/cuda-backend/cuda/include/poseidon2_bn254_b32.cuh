#pragma once
#include "poseidon2_bn254_common.cuh"
#include <cstdint>

namespace bn254_b32 {

// ---------------------------------------------------------------------------
// 256-bit helpers: wrapping add/sub with carry/borrow
// ---------------------------------------------------------------------------

// Inline PTX: emit an explicit add.cc.u32 / addc.cc.u32 chain 
static __device__ __forceinline__ uint32_t
add256_ret(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t carry;
    asm volatile("add.cc.u32  %0, %9,  %17;\n\t"
                 "addc.cc.u32 %1, %10, %18;\n\t"
                 "addc.cc.u32 %2, %11, %19;\n\t"
                 "addc.cc.u32 %3, %12, %20;\n\t"
                 "addc.cc.u32 %4, %13, %21;\n\t"
                 "addc.cc.u32 %5, %14, %22;\n\t"
                 "addc.cc.u32 %6, %15, %23;\n\t"
                 "addc.cc.u32 %7, %16, %24;\n\t"
                 "addc.u32    %8, 0,   0;"
                 : "=r"(r[0]),
                   "=r"(r[1]),
                   "=r"(r[2]),
                   "=r"(r[3]),
                   "=r"(r[4]),
                   "=r"(r[5]),
                   "=r"(r[6]),
                   "=r"(r[7]),
                   "=r"(carry)
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(a[4]),
                   "r"(a[5]),
                   "r"(a[6]),
                   "r"(a[7]),
                   "r"(b[0]),
                   "r"(b[1]),
                   "r"(b[2]),
                   "r"(b[3]),
                   "r"(b[4]),
                   "r"(b[5]),
                   "r"(b[6]),
                   "r"(b[7]));
    return carry;
}

// returns 1 if borrow (a < b), else 0
//
// sub.cc/subc.cc set CC.CF such that CF=1 means *no* borrow and CF=0 means
// borrow (opposite convention from add). After the chain, `addc.u32 nb,0,0`
// captures the inverted flag (1 = no borrow), which we XOR with 1 to return
// the canonical borrow bit.
static __device__ __forceinline__ uint32_t
sub256_ret(uint32_t r[8], const uint32_t a[8], const uint32_t b[8]) {
    uint32_t not_borrow;
    asm volatile("sub.cc.u32  %0, %9,  %17;\n\t"
                 "subc.cc.u32 %1, %10, %18;\n\t"
                 "subc.cc.u32 %2, %11, %19;\n\t"
                 "subc.cc.u32 %3, %12, %20;\n\t"
                 "subc.cc.u32 %4, %13, %21;\n\t"
                 "subc.cc.u32 %5, %14, %22;\n\t"
                 "subc.cc.u32 %6, %15, %23;\n\t"
                 "subc.cc.u32 %7, %16, %24;\n\t"
                 "addc.u32    %8, 0,   0;"
                 : "=r"(r[0]),
                   "=r"(r[1]),
                   "=r"(r[2]),
                   "=r"(r[3]),
                   "=r"(r[4]),
                   "=r"(r[5]),
                   "=r"(r[6]),
                   "=r"(r[7]),
                   "=r"(not_borrow)
                 : "r"(a[0]),
                   "r"(a[1]),
                   "r"(a[2]),
                   "r"(a[3]),
                   "r"(a[4]),
                   "r"(a[5]),
                   "r"(a[6]),
                   "r"(a[7]),
                   "r"(b[0]),
                   "r"(b[1]),
                   "r"(b[2]),
                   "r"(b[3]),
                   "r"(b[4]),
                   "r"(b[5]),
                   "r"(b[6]),
                   "r"(b[7]));
    return not_borrow ^ 1u;
}

// ---------------------------------------------------------------------------
// Montgomery helpers (32-bit-limb, inline PTX, rhs as u32)
//
// rhs is a single 32-bit limb, so:
//   - mul_small computes lhs * rhs  (256 × 32 → 288 bits = 9 × u32 limbs)
//   - mul_small_and_acc computes lhs * rhs + add  (still ≤ 2^288, so 9 limbs)
//
// Both functions return the lowest 32-bit limb (r[0]) and write the upper
// 8 limbs (r[1..8]) into `high`. Each iteration uses mul.lo/hi instead of mul.wide 
// because the former performed better
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint32_t
mul_small(uint32_t high[8], const uint32_t lhs[8], uint32_t rhs) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8;
    asm volatile(
        // Phase 1: r[0..7] = lo(lhs[i] * rhs)  — independent muls, no chain.
        "mul.lo.u32     %0, %9,  %17;\n\t"
        "mul.lo.u32     %1, %10, %17;\n\t"
        "mul.lo.u32     %2, %11, %17;\n\t"
        "mul.lo.u32     %3, %12, %17;\n\t"
        "mul.lo.u32     %4, %13, %17;\n\t"
        "mul.lo.u32     %5, %14, %17;\n\t"
        "mul.lo.u32     %6, %15, %17;\n\t"
        "mul.lo.u32     %7, %16, %17;\n\t"
        // Phase 2: r[1..8] += hi(lhs[i] * rhs). Single-bit carry chain is
        // sufficient because each column gets one hi + one carry-in, and
        // hi ≤ 2^32-2 so the sum is bounded by 2^33-2.
        "mad.hi.cc.u32  %1, %9,  %17, %1;\n\t"
        "madc.hi.cc.u32 %2, %10, %17, %2;\n\t"
        "madc.hi.cc.u32 %3, %11, %17, %3;\n\t"
        "madc.hi.cc.u32 %4, %12, %17, %4;\n\t"
        "madc.hi.cc.u32 %5, %13, %17, %5;\n\t"
        "madc.hi.cc.u32 %6, %14, %17, %6;\n\t"
        "madc.hi.cc.u32 %7, %15, %17, %7;\n\t"
        "madc.hi.u32    %8, %16, %17, 0;"
        : "=&r"(r0),
          "=&r"(r1),
          "=&r"(r2),
          "=&r"(r3),
          "=&r"(r4),
          "=&r"(r5),
          "=&r"(r6),
          "=&r"(r7),
          "=&r"(r8)
        : "r"(lhs[0]),
          "r"(lhs[1]),
          "r"(lhs[2]),
          "r"(lhs[3]),
          "r"(lhs[4]),
          "r"(lhs[5]),
          "r"(lhs[6]),
          "r"(lhs[7]),
          "r"(rhs)
    );
    high[0] = r1;
    high[1] = r2;
    high[2] = r3;
    high[3] = r4;
    high[4] = r5;
    high[5] = r6;
    high[6] = r7;
    high[7] = r8;
    return r0;
}

// Two-phase schoolbook with mul.lo / mul.hi.
//   Phase 1 chains lo(lhs[i] * rhs) + add[i] across all 8 limbs via
//           mad.lo.cc / madc.lo.cc, capturing the final carry into r[8].
//   Phase 2 chains hi(lhs[i] * rhs) into r[i+1] via mad.hi.cc / madc.hi.cc,
//           folding into the existing r[i+1] (which holds the phase-1 result)
//           and the previous-limb carry. The final r[8] accumulates the
//           phase-1 overflow plus the top hi; total result is bounded by
//           2^288 so the high u32 of r[8] cannot overflow.
static __device__ __forceinline__ uint32_t
mul_small_and_acc(uint32_t high[8], const uint32_t lhs[8], uint32_t rhs, const uint32_t add[8]) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8;
    asm volatile(
        // Phase 1: r[i] = lo(lhs[i] * rhs) + add[i] with carry chain
        "mad.lo.cc.u32  %0, %9,  %17, %18;\n\t"
        "madc.lo.cc.u32 %1, %10, %17, %19;\n\t"
        "madc.lo.cc.u32 %2, %11, %17, %20;\n\t"
        "madc.lo.cc.u32 %3, %12, %17, %21;\n\t"
        "madc.lo.cc.u32 %4, %13, %17, %22;\n\t"
        "madc.lo.cc.u32 %5, %14, %17, %23;\n\t"
        "madc.lo.cc.u32 %6, %15, %17, %24;\n\t"
        "madc.lo.cc.u32 %7, %16, %17, %25;\n\t"
        "addc.u32       %8, 0, 0;\n\t"
        // Phase 2: r[1..8] += hi(lhs[i] * rhs) with carry chain
        "mad.hi.cc.u32  %1, %9,  %17, %1;\n\t"
        "madc.hi.cc.u32 %2, %10, %17, %2;\n\t"
        "madc.hi.cc.u32 %3, %11, %17, %3;\n\t"
        "madc.hi.cc.u32 %4, %12, %17, %4;\n\t"
        "madc.hi.cc.u32 %5, %13, %17, %5;\n\t"
        "madc.hi.cc.u32 %6, %14, %17, %6;\n\t"
        "madc.hi.cc.u32 %7, %15, %17, %7;\n\t"
        "madc.hi.u32    %8, %16, %17, %8;"
        : "=&r"(r0),
          "=&r"(r1),
          "=&r"(r2),
          "=&r"(r3),
          "=&r"(r4),
          "=&r"(r5),
          "=&r"(r6),
          "=&r"(r7),
          "=&r"(r8)
        : "r"(lhs[0]),
          "r"(lhs[1]),
          "r"(lhs[2]),
          "r"(lhs[3]),
          "r"(lhs[4]),
          "r"(lhs[5]),
          "r"(lhs[6]),
          "r"(lhs[7]),
          "r"(rhs),
          "r"(add[0]),
          "r"(add[1]),
          "r"(add[2]),
          "r"(add[3]),
          "r"(add[4]),
          "r"(add[5]),
          "r"(add[6]),
          "r"(add[7])
    );
    high[0] = r1;
    high[1] = r2;
    high[2] = r3;
    high[3] = r4;
    high[4] = r5;
    high[5] = r6;
    high[6] = r7;
    high[7] = r8;
    return r0;
}

/// Single-step Montgomery reduction by 32 bits. Given a 9-limb value laid out
/// as (acc0, acc[0..7]) — acc0 the lowest u32 limb, acc the upper 8 — returns
///   r  ≡  ((acc0 + acc · 2^32) − t·P) / 2^32   (mod P)
/// where  t = acc0 · BN254_MU_32 mod 2^32. By construction of t, the bottom
/// limb of t·P matches acc0 so the lowest 32 bits cancel; we shift right by
/// 32 by dropping the low limb of t·P and subtracting the upper 8 from acc.
static __device__ __forceinline__ void imr(uint32_t r[8], uint32_t acc0, const uint32_t acc[8]) {
    uint32_t t = acc0 * BN254_MU_32; // wrapping mod 2^32
    uint32_t u[8];
    // mul_small returns the lowest limb of t·P (which equals acc0, discarded)
    // and writes the upper 8 limbs of t·P into u.
    mul_small(u, BN254_P_32, t);
    uint32_t sub[8];
    uint32_t borrow = sub256_ret(sub, acc, u);
    if (borrow) {
        add256_ret(r, sub, BN254_P_32);
    } else {
        for (int i = 0; i < 8; i++)
            r[i] = sub[i];
    }
}

/// Montgomery multiplication: r = lhs · rhs · R^{-1} mod P, with R = 2^256.
/// CIOS: each iteration absorbs one u32 limb of rhs (8 iters total, vs 4 in
/// the b64 version which absorbs u64 limbs). Each imr step shifts out 32 bits
/// of "Montgomery factor"; 8 steps × 32 = 256, matching b64's 4 × 64.
static __device__ __forceinline__ void bn254_monty_mul(
    uint32_t r[8],
    const uint32_t lhs[8],
    const uint32_t rhs[8]
) {
    uint32_t acc0;
    uint32_t acc[8];
    uint32_t tmp[8];

    acc0 = mul_small(acc, lhs, rhs[0]);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[1], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[2], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[3], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[4], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[5], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[6], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[7], tmp);
    imr(r, acc0, acc);
}

// ---------------------------------------------------------------------------
// Field arithmetic (Montgomery form throughout, 8 × u32 limbs)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ Bn254Fr32 bn254_add(Bn254Fr32 a, Bn254Fr32 b) {
    Bn254Fr32 r;
    uint32_t sum[8];
    uint32_t overflow = add256_ret(sum, a.limbs, b.limbs);
    // If overflow OR sum >= P, subtract P.
    uint32_t sub[8];
    uint32_t borrow = sub256_ret(sub, sum, BN254_P_32);
    if (overflow || !borrow) {
        for (int i = 0; i < 8; i++)
            r.limbs[i] = sub[i];
    } else {
        for (int i = 0; i < 8; i++)
            r.limbs[i] = sum[i];
    }
    return r;
}

static __device__ __forceinline__ Bn254Fr32 bn254_sub(Bn254Fr32 a, Bn254Fr32 b) {
    Bn254Fr32 r;
    uint32_t diff[8];
    uint32_t borrow = sub256_ret(diff, a.limbs, b.limbs);
    if (borrow) {
        add256_ret(r.limbs, diff, BN254_P_32);
    } else {
        for (int i = 0; i < 8; i++)
            r.limbs[i] = diff[i];
    }
    return r;
}

static __device__ __forceinline__ Bn254Fr32 bn254_neg(Bn254Fr32 a) {
    bool is_zero = (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3] | a.limbs[4] | a.limbs[5] |
                    a.limbs[6] | a.limbs[7]) == 0u;
    if (is_zero)
        return a;
    // r = P - a
    Bn254Fr32 r;
    sub256_ret(r.limbs, BN254_P_32, a.limbs);
    return r;
}

static __device__ __forceinline__ Bn254Fr32 bn254_double(Bn254Fr32 a) { return bn254_add(a, a); }

static __device__ __forceinline__ Bn254Fr32 bn254_mul(Bn254Fr32 a, Bn254Fr32 b) {
    Bn254Fr32 r;
    bn254_monty_mul(r.limbs, a.limbs, b.limbs);
    return r;
}

/// x^5 S-box
static __device__ __forceinline__ Bn254Fr32 bn254_sbox(Bn254Fr32 x) {
    Bn254Fr32 x2 = bn254_mul(x, x);
    Bn254Fr32 x4 = bn254_mul(x2, x2);
    return bn254_mul(x4, x);
}

// ---------------------------------------------------------------------------
// Canonical ↔ Montgomery conversions (32-bit-limb)
// ---------------------------------------------------------------------------

/// Convert canonical [u32; 8] → Montgomery form Bn254Fr32.
static __device__ __forceinline__ Bn254Fr32 bn254_from_canonical(const uint32_t canonical[8]) {
    Bn254Fr32 r;
    bn254_monty_mul(r.limbs, BN254_R2_32, canonical);
    return r;
}

/// Convert Montgomery form → canonical [u32; 8].
static __device__ __forceinline__ void bn254_to_canonical(uint32_t canonical[8], Bn254Fr32 x) {
    const uint32_t one[8] = {1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    bn254_monty_mul(canonical, x.limbs, one);
}

// ---------------------------------------------------------------------------
// Multi-field packing helper (32-bit-limb)
// ---------------------------------------------------------------------------

/// Pack `count` (1..8) canonical BabyBear u32 values into one Bn254Fr32.
/// result = bb[0] + bb[1]*2^31 + bb[2]*2^62 + ... (as a BN254 scalar).
///
/// Each bb[i] < 2^31 so the 31-bit slots don't overlap; we shift+OR into a
/// canonical [u32; 8] buffer and pass it through Montgomery conversion once
/// at the end. Span across two u32 limbs happens whenever shift > 1.
static __device__ __forceinline__ Bn254Fr32 bn254_pack_base_2_31(const uint32_t *bb, int count) {
    uint32_t canonical[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    for (int i = 0; i < count; i++) {
        int bit_pos = i * 31;
        int limb = bit_pos >> 5;  // bit_pos / 32
        int shift = bit_pos & 31; // bit_pos % 32
        canonical[limb] |= bb[i] << shift;
        // shift + 31 > 32 iff shift > 1: bb[i] spans into the next limb.
        if (shift > 1 && limb < 7) {
            canonical[limb + 1] |= bb[i] >> (32 - shift);
        }
    }
    return bn254_from_canonical(canonical);
}

// ---------------------------------------------------------------------------
// Poseidon2 MDS layers (32-bit-limb)
// ---------------------------------------------------------------------------

template <int WIDTH> static __device__ __forceinline__ void bn254_mds_external(Bn254Fr32 s[WIDTH]) {
    Bn254Fr32 sum = s[0];
    for (int i = 1; i < WIDTH; i++)
        sum = bn254_add(sum, s[i]);
    for (int i = 0; i < WIDTH; i++)
        s[i] = bn254_add(s[i], sum);
}

template <int WIDTH> static __device__ __forceinline__ void bn254_mds_internal(Bn254Fr32 s[WIDTH]) {
    Bn254Fr32 sum = s[0];
    for (int i = 1; i < WIDTH; i++)
        sum = bn254_add(sum, s[i]);
    for (int i = 0; i < WIDTH - 1; i++)
        s[i] = bn254_add(s[i], sum);
    s[WIDTH - 1] = bn254_add(bn254_double(s[WIDTH - 1]), sum);
}

} // namespace bn254_b32
