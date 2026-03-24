/// BN254 field arithmetic and Poseidon2 permutation helpers for CUDA.
///
/// This header provides:
///   - Bn254Fr type (4 × u64 Montgomery form, little-endian limbs)
///   - CIOS Montgomery multiplication matching p3-bn254::helpers::monty_mul
///   - Field add / sub / neg / double / x^5 S-box
///   - MDS layers for WIDTH = 3 (external and internal)
///   - bn254_from_canonical / bn254_to_canonical conversions
///   - reduce_32 (pack BabyBear → Bn254Fr) for both num_f_elms=8 (Merkle) and num_f_elms=3 (challenger)
///   - split_32 (unpack Bn254Fr → BabyBear canonical u32) for num_f_elms=3 (challenger)
///
/// All functions are pure computation (no globals). The Poseidon2 permutation
/// function is declared here but defined in bn254_poseidon2.cu (where the
/// round-constant globals live).
#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// BN254 scalar field type
// ---------------------------------------------------------------------------

/// A BN254 scalar field element stored in Montgomery form.
/// `limbs[i]` are little-endian 64-bit words: limbs[0] = bits 0-63, etc.
struct Bn254Fr {
    uint64_t limbs[4];
};

// ---------------------------------------------------------------------------
// Field constants (all in Montgomery form)
// ---------------------------------------------------------------------------

// P = BN254 prime (canonical representation)
__device__ __constant__ static const uint64_t BN254_P[4] = {
    0x43e1f593f0000001ULL,
    0x2833e84879b97091ULL,
    0xb85045b68181585dULL,
    0x30644e72e131a029ULL,
};

// MU = P^{-1} mod 2^64  (used in Montgomery reduction)
static const uint64_t BN254_MU = 0x3d1e0a6c10000001ULL;

// R^2 mod P  (converts canonical → Montgomery: mont(x) = monty_mul(R2, x))
__device__ __constant__ static const uint64_t BN254_R2[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL,
};

// 1 in Montgomery form = R mod P = 2^256 mod P
__device__ __constant__ static const uint64_t BN254_R_ONE[4] = {
    0xac96341c4ffffffbULL,
    0x36fc76959f60cd29ULL,
    0x666ea36f7879462eULL,
    0x0e0a77c19a07df2fULL,
};

// 2 in Montgomery form
__device__ __constant__ static const uint64_t BN254_R_TWO[4] = {
    0x592c68389ffffff6ULL,
    0x6df8ed2b3ec19a53ULL,
    0xccdd46def0f28c5cULL,
    0x1c14ef83340fbe5eULL,
};

// BabyBear prime
static const uint64_t BABYBEAR_PRIME = 0x78000001ULL;

// ---------------------------------------------------------------------------
// 256-bit helpers: wrapping add/sub with carry/borrow
// ---------------------------------------------------------------------------

static __device__ __forceinline__
uint64_t add256_ret(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a[i] + b[i] + carry;
        r[i] = (uint64_t)t;
        carry = (uint64_t)(t >> 64);
    }
    return carry;
}

// returns 1 if borrow (a < b), else 0
static __device__ __forceinline__
uint64_t sub256_ret(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
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
static __device__ __forceinline__
uint64_t mul_small(uint64_t high4[4], const uint64_t lhs[4], uint64_t rhs) {
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
static __device__ __forceinline__
uint64_t mul_small_and_acc(uint64_t high4[4], const uint64_t lhs[4], uint64_t rhs,
                            const uint64_t add[4]) {
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
static __device__ __forceinline__
void imr(uint64_t r[4], uint64_t acc0, const uint64_t acc[4]) {
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
        for (int i = 0; i < 4; i++) r[i] = sub[i];
    }
}

/// Montgomery multiplication: r = lhs * rhs * R^{-1} mod P.
/// Requires lhs < P. Matches monty_mul in p3-bn254::helpers.
static __device__ __forceinline__
void bn254_monty_mul(uint64_t r[4], const uint64_t lhs[4], const uint64_t rhs[4]) {
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

static __device__ __forceinline__
Bn254Fr bn254_add(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    uint64_t sum[4];
    uint64_t overflow = add256_ret(sum, a.limbs, b.limbs);
    // If overflow OR sum >= P, subtract P
    uint64_t sub[4];
    uint64_t borrow = sub256_ret(sub, sum, BN254_P);
    if (overflow || !borrow) {
        for (int i = 0; i < 4; i++) r.limbs[i] = sub[i];
    } else {
        for (int i = 0; i < 4; i++) r.limbs[i] = sum[i];
    }
    return r;
}

static __device__ __forceinline__
Bn254Fr bn254_sub(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    uint64_t diff[4];
    uint64_t borrow = sub256_ret(diff, a.limbs, b.limbs);
    if (borrow) {
        add256_ret(r.limbs, diff, BN254_P);
    } else {
        for (int i = 0; i < 4; i++) r.limbs[i] = diff[i];
    }
    return r;
}

static __device__ __forceinline__
Bn254Fr bn254_neg(Bn254Fr a) {
    // Check if a == 0
    bool is_zero = (a.limbs[0] | a.limbs[1] | a.limbs[2] | a.limbs[3]) == 0;
    if (is_zero) return a;
    // r = P - a
    Bn254Fr r;
    sub256_ret(r.limbs, BN254_P, a.limbs);
    return r;
}

static __device__ __forceinline__
Bn254Fr bn254_double(Bn254Fr a) {
    return bn254_add(a, a);
}

static __device__ __forceinline__
Bn254Fr bn254_mul(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    bn254_monty_mul(r.limbs, a.limbs, b.limbs);
    return r;
}

/// x^5 S-box
static __device__ __forceinline__
Bn254Fr bn254_sbox(Bn254Fr x) {
    Bn254Fr x2 = bn254_mul(x, x);
    Bn254Fr x4 = bn254_mul(x2, x2);
    return bn254_mul(x4, x);
}

// ---------------------------------------------------------------------------
// Canonical ↔ Montgomery conversions
// ---------------------------------------------------------------------------

/// Convert canonical [u64; 4] → Montgomery form Bn254Fr.
/// Equivalent to Bn254Scalar::from_biguint (for values already < P).
static __device__ __forceinline__
Bn254Fr bn254_from_canonical(const uint64_t canonical[4]) {
    Bn254Fr r;
    bn254_monty_mul(r.limbs, BN254_R2, canonical);
    return r;
}

/// Convert Montgomery form → canonical [u64; 4].
/// Equivalent to monty_mul(x, [1,0,0,0]).
static __device__ __forceinline__
void bn254_to_canonical(uint64_t canonical[4], Bn254Fr x) {
    const uint64_t one[4] = {1, 0, 0, 0};
    bn254_monty_mul(canonical, x.limbs, one);
}

// ---------------------------------------------------------------------------
// reduce_32: pack BabyBear canonical u32 values into one Bn254Fr
//
//   num_f_elms = 8  (Merkle hash, MultiField32PaddingFreeSponge)
//   num_f_elms = 3  (Challenger, MultiField32Challenger)
//
// Formula: result = sum_{i=0}^{n-1} bb[i] * 2^{32*i}  (as a BN254 scalar)
// Implementation: canonical = { bb[0]|(bb[1]<<32), bb[2]|(bb[3]<<32), ... }
//                 r = bn254_from_canonical(canonical)
// ---------------------------------------------------------------------------

static __device__ __forceinline__
Bn254Fr bn254_reduce_32(const uint32_t* bb, int count) {
    // Pack bb[0..count-1] into canonical 256-bit integer, then to Montgomery.
    // canonical = bb[0] + bb[1]*2^32 + bb[2]*2^64 + ...
    // Limb i = bb[2i] | (bb[2i+1] << 32) for pairs; last odd one goes into low32 of a limb.
    uint64_t canonical[4] = {0, 0, 0, 0};
    for (int i = 0; i < count; i++) {
        // bit offset = i * 32; limb index = i / 2; within limb bit = (i & 1) * 32
        int limb = i >> 1;       // 0, 0, 1, 1, 2, 2, 3, 3
        int shift = (i & 1) << 5; // 0, 32, 0, 32, 0, 32, 0, 32
        canonical[limb] |= (uint64_t)(bb[i]) << shift;
    }
    return bn254_from_canonical(canonical);
}

// ---------------------------------------------------------------------------
// split_32 (num_f_elms = 3, for MultiField32Challenger output):
// Given a Bn254Fr in Montgomery form, extract 3 BabyBear canonical u32 values.
// Each result = canonical_limb_i % BABYBEAR_PRIME.
// ---------------------------------------------------------------------------

static __device__ __forceinline__
void bn254_split_32_3(uint32_t bb[3], Bn254Fr x) {
    uint64_t canonical[4];
    bn254_to_canonical(canonical, x);
    bb[0] = (uint32_t)(canonical[0] % BABYBEAR_PRIME);
    bb[1] = (uint32_t)(canonical[1] % BABYBEAR_PRIME);
    bb[2] = (uint32_t)(canonical[2] % BABYBEAR_PRIME);
    // canonical[3] is the 4th limb; split_32 with n=3 ignores it
}

// ---------------------------------------------------------------------------
// Poseidon2 MDS layers for WIDTH = 3
// ---------------------------------------------------------------------------

/// External MDS (same as mds_light_permutation for WIDTH=3):
///   sum = s[0] + s[1] + s[2]
///   s[i] += sum  for all i
static __device__ __forceinline__
void bn254_mds_external(Bn254Fr s[3]) {
    Bn254Fr sum = bn254_add(bn254_add(s[0], s[1]), s[2]);
    s[0] = bn254_add(s[0], sum);
    s[1] = bn254_add(s[1], sum);
    s[2] = bn254_add(s[2], sum);
}

/// Internal MDS (bn254_matmul_internal):
///   sum = s[0] + s[1] + s[2]
///   s[0] += sum
///   s[1] += sum
///   s[2] = 2*s[2] + sum
static __device__ __forceinline__
void bn254_mds_internal(Bn254Fr s[3]) {
    Bn254Fr sum = bn254_add(bn254_add(s[0], s[1]), s[2]);
    s[0] = bn254_add(s[0], sum);
    s[1] = bn254_add(s[1], sum);
    s[2] = bn254_add(bn254_double(s[2]), sum);
}
