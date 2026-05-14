/// BN254 field arithmetic and Poseidon2 permutation helpers for CUDA.
///
/// This header provides:
///   - CIOS Montgomery multiplication matching p3-bn254::helpers::monty_mul
///   - Field add / sub / neg / double / x^5 S-box
///   - Templated MDS layers (external and internal, any WIDTH)
///   - bn254_from_canonical / bn254_to_canonical conversions
///   - bn254_pack_base_2_31 / u256_mod_u32 helpers for MultiFieldTranscript
///     packing and sampling during grinding
///
/// All functions are pure computation (no globals). The Poseidon2 permutation
/// function is declared here but defined in bn254_poseidon2.cu (where the
/// round-constant globals live).
#pragma once

#include "poseidon2_bn254_common.cuh"

namespace bn254_noinline {

// ---------------------------------------------------------------------------
// 256-bit helpers: wrapping add/sub with carry/borrow
// ---------------------------------------------------------------------------

static __device__ uint64_t add256_ret(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a[i] + b[i] + carry;
        r[i] = (uint64_t)t;
        carry = (uint64_t)(t >> 64);
    }
    return carry;
}

// returns 1 if borrow (a < b), else 0
static __device__ uint64_t sub256_ret(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a[i] - b[i] - borrow;
        r[i] = (uint64_t)t;
        // If a[i] < b[i]+borrow, we borrowed
        borrow = (t >> 127) ? 1 : 0;
    }
    return borrow;
}

// ---------------------------------------------------------------------------
// Montgomery helpers (matching p3-bn254::helpers exactly)
// ---------------------------------------------------------------------------

/// Compute lhs * rhs as a 5-limb product.
/// Returns the lowest limb and the upper 4 limbs (matching mul_small in Rust).
static __device__ uint64_t mul_small(uint64_t high4[4], const uint64_t lhs[4], uint64_t rhs) {
    __uint128_t acc = (__uint128_t)lhs[0] * rhs;
    uint64_t low = (uint64_t)acc;
    acc >>= 64;
    for (int i = 1; i < 4; i++) {
        acc += (__uint128_t)lhs[i] * rhs;
        high4[i - 1] = (uint64_t)acc;
        acc >>= 64;
    }
    high4[3] = (uint64_t)acc;
    return low;
}

/// Compute lhs * rhs + add as a 5-limb product.
static __device__ uint64_t __noinline__
mul_small_and_acc(uint64_t high4[4], const uint64_t lhs[4], uint64_t rhs, const uint64_t add[4]) {
    __uint128_t acc = (__uint128_t)lhs[0] * rhs + add[0];
    uint64_t low = (uint64_t)acc;
    acc >>= 64;
    for (int i = 1; i < 4; i++) {
        acc += (__uint128_t)lhs[i] * rhs + add[i];
        high4[i - 1] = (uint64_t)acc;
        acc >>= 64;
    }
    high4[3] = (uint64_t)acc;
    return low;
}

/// Single-step Montgomery reduction: given a 5-limb number (acc0, acc[4]),
/// returns (acc - t*P) >> 64  where t = acc0 * MU mod 2^64.
/// Matches interleaved_monty_reduction in p3-bn254.
static __device__ void imr(uint64_t r[4], uint64_t acc0, const uint64_t acc[4]) {
    uint64_t t = acc0 * BN254_MU; // wrapping multiply mod 2^64
    // u = (t * P) upper 4 limbs (discard lowest)
    uint64_t u[4];
    mul_small(u, BN254_P, t); // u = upper 4 limbs of t*P
    // sub = acc - u
    uint64_t sub[4];
    uint64_t borrow = sub256_ret(sub, acc, u);
    if (borrow) {
        add256_ret(r, sub, BN254_P);
    } else {
        for (int i = 0; i < 4; i++)
            r[i] = sub[i];
    }
}

/// Montgomery multiplication: r = lhs * rhs * R^{-1} mod P.
/// Requires lhs < P. Matches monty_mul in p3-bn254::helpers.
static __device__ void __noinline__
bn254_monty_mul(uint64_t r[4], const uint64_t lhs[4], const uint64_t rhs[4]) {
    uint64_t acc0, acc[4], tmp[4];

    acc0 = mul_small(acc, lhs, rhs[0]);
    imr(tmp, acc0, acc);

    acc0 = mul_small_and_acc(acc, lhs, rhs[1], tmp);
    imr(tmp, acc0, acc);

    acc0 = mul_small_and_acc(acc, lhs, rhs[2], tmp);
    imr(tmp, acc0, acc);

    acc0 = mul_small_and_acc(acc, lhs, rhs[3], tmp);
    imr(r, acc0, acc);
}

// ---------------------------------------------------------------------------
// Field arithmetic (Montgomery form throughout)
// ---------------------------------------------------------------------------

static __device__ __noinline__ Bn254Fr bn254_add(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    uint64_t sum[4];
    uint64_t overflow = add256_ret(sum, a.limbs, b.limbs);
    // If overflow OR sum >= P, subtract P
    uint64_t sub[4];
    uint64_t borrow = sub256_ret(sub, sum, BN254_P);
    if (overflow || !borrow) {
        for (int i = 0; i < 4; i++)
            r.limbs[i] = sub[i];
    } else {
        for (int i = 0; i < 4; i++)
            r.limbs[i] = sum[i];
    }
    return r;
}

static __device__ __noinline__ Bn254Fr bn254_sub(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    uint64_t diff[4];
    uint64_t borrow = sub256_ret(diff, a.limbs, b.limbs);
    if (borrow) {
        add256_ret(r.limbs, diff, BN254_P);
    } else {
        for (int i = 0; i < 4; i++)
            r.limbs[i] = diff[i];
    }
    return r;
}

static __device__ __noinline__ Bn254Fr bn254_neg(Bn254Fr a) {
    // Check if a == 0
    bool is_zero = (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
    if (is_zero)
        return a;
    // r = P - a
    Bn254Fr r;
    sub256_ret(r.limbs, BN254_P, a.limbs);
    return r;
}

static __device__ Bn254Fr bn254_double(Bn254Fr a) { return bn254_add(a, a); }

static __device__ Bn254Fr bn254_mul(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    bn254_monty_mul(r.limbs, a.limbs, b.limbs);
    return r;
}

/// x^5 S-box
static __device__ Bn254Fr bn254_sbox(Bn254Fr x) {
    Bn254Fr x2 = bn254_mul(x, x);
    Bn254Fr x4 = bn254_mul(x2, x2);
    return bn254_mul(x4, x);
}

// ---------------------------------------------------------------------------
// Canonical ↔ Montgomery conversions
// ---------------------------------------------------------------------------

/// Convert canonical [u64; 4] → Montgomery form Bn254Fr.
/// Equivalent to Bn254Scalar::from_biguint (for values already < P).
static __device__ Bn254Fr bn254_from_canonical(const uint64_t canonical[4]) {
    Bn254Fr r;
    bn254_monty_mul(r.limbs, BN254_R2, canonical);
    return r;
}

/// Convert Montgomery form → canonical [u64; 4].
/// Equivalent to monty_mul(x, [1,0,0,0]).
static __device__ void bn254_to_canonical(uint64_t canonical[4], Bn254Fr x) {
    const uint64_t one[4] = {1, 0, 0, 0};
    bn254_monty_mul(canonical, x.limbs, one);
}

// ---------------------------------------------------------------------------
// Multi-field packing helpers
// ---------------------------------------------------------------------------

/// Pack `count` (1..8) canonical BabyBear u32 values into one Bn254Fr.
/// result = bb[0] + bb[1]*2^31 + bb[2]*2^62 + ... (as a BN254 scalar).
///
/// Since each bb[i] < 2^31, the terms don't overlap when bit-shifted,
/// so the entire packing is done with shifts and ORs in canonical form,
/// followed by a single Montgomery conversion.
static __device__ Bn254Fr bn254_pack_base_2_31(const uint32_t *bb, int count) {
    uint64_t canonical[4] = {0, 0, 0, 0};
    for (int i = 0; i < count; i++) {
        int bit_pos = i * 31;
        int limb = bit_pos >> 6;  // bit_pos / 64
        int shift = bit_pos & 63; // bit_pos % 64
        canonical[limb] |= (uint64_t)(bb[i]) << shift;
        // If the value spans two limbs (shift + 31 > 64), write the overflow.
        if (shift > 33 && limb < 3) { // shift + 31 > 64 iff shift > 33
            canonical[limb + 1] |= (uint64_t)(bb[i]) >> (64 - shift);
        }
    }
    return bn254_from_canonical(canonical);
}

/// 256-bit unsigned mod by a 32-bit divisor.
static __device__ uint32_t u256_mod_u32(const uint64_t x[4], uint32_t d) {
    uint64_t rem = 0;
    for (int i = 3; i >= 0; i--) {
        rem = ((rem << 32) | (x[i] >> 32)) % d;
        rem = ((rem << 32) | (x[i] & 0xFFFFFFFFULL)) % d;
    }
    return (uint32_t)rem;
}

// ---------------------------------------------------------------------------
// Poseidon2 MDS layers (generic over WIDTH)
//
// External MDS: M_E = 1 + ones  (circ(2,1,...,1))
//   sum = s[0] + s[1] + ... + s[WIDTH-1]
//   s[i] += sum  for all i
//
// Internal MDS: M_I = I + diag(1,...,1,2)
//   Hardcodes mat_internal_diag_m_1 = [1,...,1,2] for performance (avoids monty mul).
//   Must match the Rust-side constants for width 2 and width 3:
//     width 2: [Bn254::ONE, Bn254::TWO]
//     width 3: [Bn254::ONE, Bn254::ONE, Bn254::TWO]
//   sum = s[0] + s[1] + ... + s[WIDTH-1]
//   s[i] += sum  for i < WIDTH-1
//   s[WIDTH-1] = 2*s[WIDTH-1] + sum
// ---------------------------------------------------------------------------

template <int WIDTH> static __device__ __noinline__ void bn254_mds_external(Bn254Fr s[WIDTH]) {
    Bn254Fr sum = s[0];
    for (int i = 1; i < WIDTH; i++)
        sum = bn254_add(sum, s[i]);
    for (int i = 0; i < WIDTH; i++)
        s[i] = bn254_add(s[i], sum);
}

template <int WIDTH> static __device__ __noinline__ void bn254_mds_internal(Bn254Fr s[WIDTH]) {
    Bn254Fr sum = s[0];
    for (int i = 1; i < WIDTH; i++)
        sum = bn254_add(sum, s[i]);
    for (int i = 0; i < WIDTH - 1; i++)
        s[i] = bn254_add(s[i], sum);
    s[WIDTH - 1] = bn254_add(bn254_double(s[WIDTH - 1]), sum);
}

} // namespace bn254_noinline
