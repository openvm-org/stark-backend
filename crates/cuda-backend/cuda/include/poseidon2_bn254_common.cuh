
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

// BabyBear prime
static const uint64_t BABYBEAR_PRIME = 0x78000001ULL;
