
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
