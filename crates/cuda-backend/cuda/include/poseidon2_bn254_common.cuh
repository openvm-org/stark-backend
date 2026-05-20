
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

struct Bn254Fr32 {
    uint32_t limbs[8];
};


// ---------------------------------------------------------------------------
// BN254 Merkle digest: a single Bn254Fr element (32 bytes)
// Matches Digest = [Bn254Scalar; 1] on the Rust side.
// ---------------------------------------------------------------------------

struct bn254_digest_t {
    Bn254Fr elem;
};

static const int BN254_BABY_BEAR_RATE = 16;
static const int BN254_NUM_F_ELMS = 8;

static_assert(
    BN254_BABY_BEAR_RATE % 4 == 0,
    "BN254_BABY_BEAR_RATE must be a multiple of FpExt degree (4)"
);


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

// BN254 prime as 8 × u32 limbs, least-significant limb first. Same numerical
// value as BN254_P (in poseidon2_bn254_common.cuh) — just split into 32-bit
// halves: each adjacent pair (lo, hi) is one u64 limb of BN254_P.
__device__ __constant__ static const uint32_t BN254_P_32[8] = {
    4026531841u, 
    1138881939u, 
    2042196113u, 
    674490440u,  
    2172737629u, 
    3092268470u, 
    3778125865u, 
    811880050u,  
};

// MU = P^{-1} mod 2^64  (used in Montgomery reduction)
static const uint64_t BN254_MU = 0x3d1e0a6c10000001ULL;
// same as above but for 32 limbs
static const uint32_t BN254_MU_32 = 268435457;

// R^2 mod P  (converts canonical → Montgomery: mont(x) = monty_mul(R2, x))
__device__ __constant__ static const uint64_t BN254_R2[4] = {
    0x1bb8e645ae216da7ULL,
    0x53fe3ab1e35c59e3ULL,
    0x8c49833d53bb8085ULL,
    0x0216d0b17f4e44a5ULL,
};

// same as above but for 32 limbs
__device__ __constant__ static const uint32_t BN254_R2_32[8] = {
    2921426343u,
    465102405u,
    3814480355u,
    1409170097u,
    1404797061u,
    2353627965u,
    2135835813u,
    35049649u,
};

// BabyBear prime
static const uint64_t BABYBEAR_PRIME = 0x78000001ULL;
