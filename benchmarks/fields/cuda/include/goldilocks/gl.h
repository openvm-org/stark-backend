/*
 * Goldilocks field element (Gl) - wrapper around sppark's gl64_t
 * 
 * Prime: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
 * 
 * This is a 64-bit prime field used in STARKs (e.g., Plonky2, Polygon zkEVM).
 */

#pragma once

#include <cstdint>
#include "ff/gl64_t.cuh"

/// Goldilocks field element
class Gl {
public:
    gl64_t val;
    
    static constexpr uint64_t P = 0xFFFFFFFF00000001ULL;
    
    __device__ Gl() : val() {}
    __device__ Gl(gl64_t v) : val(v) {}
    __device__ explicit Gl(uint64_t v) : val(v) {}
    
    __device__ static Gl zero() { Gl r; r.val.zero(); return r; }
    __device__ static Gl one() { return Gl(gl64_t::one()); }
    
    __device__ Gl operator+(Gl rhs) const { return Gl(val + rhs.val); }
    __device__ Gl operator-(Gl rhs) const { return Gl(val - rhs.val); }
    __device__ Gl operator*(Gl rhs) const { return Gl(val * rhs.val); }
    __device__ Gl operator-() const { return Gl(-val); }
    
    __device__ Gl& operator+=(Gl rhs) { val += rhs.val; return *this; }
    __device__ Gl& operator-=(Gl rhs) { val -= rhs.val; return *this; }
    __device__ Gl& operator*=(Gl rhs) { val *= rhs.val; return *this; }
    
    __device__ bool operator==(Gl rhs) const { return val == rhs.val; }
    __device__ bool operator!=(Gl rhs) const { return val != rhs.val; }
    
    __device__ uint64_t asRaw() const { return val[0]; }
};

/// Inversion using reciprocal
__device__ inline Gl inv(Gl x) {
    return Gl(x.val.reciprocal());
}

static_assert(sizeof(Gl) == 8, "Gl must be 8 bytes");
