/**
 * KoalaBear field element (Kb)
 * 
 * The Kb class is an element of the finite field F_p, where p = 2^31 - 2^24 + 1 = 2130706433.
 * This is the KoalaBear prime, similar to BabyBear but with different 2-adicity.
 * 
 * Properties:
 * - Prime: p = 2^31 - 2^24 + 1 = 0x7F000001
 * - 2-adicity: 24 (p-1 = 2^24 * 127)
 * - Fits in 31 bits, uses Montgomery representation
 */

#pragma once

#include <cassert>
#include <cstdint>
#include "ff/koala_bear.hpp"

/// KoalaBear field element
class Kb : public kb31_t {
public:
    /// The value of P, the modulus of Kb.
    static constexpr uint32_t P = (uint32_t(1) << 31) - (uint32_t(1) << 24) + 1; // 0x7F000001
    static constexpr uint32_t M = 0x81000001; // -M0 = 2^32 - 0x7EFFFFFF
    static constexpr uint32_t R2 = 0x17F7EFE4; // RR
    static constexpr uint32_t MONTY_BITS = 32;
    static constexpr uint32_t MONTY_MASK = 0xffffffff;
    static constexpr uint32_t HALF_P_PLUS_1 = (P + 1) >> 1;
    static constexpr uint32_t TWO_ADICITY = 24;

private:
    __device__ uint32_t val() const {
        return static_cast<uint32_t>(*this);
    }

public:
    /// Default constructor
    __device__ constexpr Kb() : kb31_t(0) {}
    
    /// Constructor from kb31_t for explicit conversion
    __device__ explicit constexpr Kb(const kb31_t& b) : kb31_t(b) {}

    /// Constructor from kb31_base for implicit conversion
    __device__ Kb(const kb31_base& b) : kb31_t(b) {}

    /// Constructor from uint32_t that forces encoding
    /// Reduces mod P first to handle values >= INT_MAX correctly
    __host__ __device__ constexpr Kb(uint32_t v) : kb31_t(static_cast<int>(v % P)) {}

    /// Constructor from any numerical type that forces encoding
    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    __host__ __device__ constexpr Kb(I v) : kb31_t(static_cast<int>(static_cast<uint32_t>(v) % P)) {}

    /// Construct a Kb from an already-encoded raw value
    __device__ static constexpr Kb fromRaw(uint32_t val) { 
        return Kb(kb31_t(val));
    }

    /// Kb with zero value
    __device__ static constexpr Kb zero() { return Kb(0); }

    /// Kb with one value
    __device__ static constexpr Kb one() { return Kb(1); }

    /// Kb with negative one value
    __device__ static constexpr Kb neg_one() { return maxVal(); }

    /// Convert to a uint32_t
    __device__ uint32_t asUInt32() const { return val(); }

    /// Return the raw underlying word
    __device__ uint32_t asRaw() const { return kb31_t::operator*(); }

    /// Get the largest value, basically P - 1.
    __device__ static constexpr Kb maxVal() { return P - 1; }

    /// Get an 'invalid' Kb value
    __device__ static constexpr Kb invalid() { return Kb::fromRaw(0xfffffffful); }

    /// get value
    __device__ uint32_t get() const { return asRaw(); }

    /// force set the value
    __device__ void set(uint32_t input) { kb31_t::operator=(input); }

    /// Equality operators
    friend __device__ inline bool operator==(const Kb& a, const Kb& b) {
        return a.asRaw() == b.asRaw();
    }
    friend __device__ inline bool operator!=(const Kb& a, const Kb& b) {
        return a.asRaw() != b.asRaw();
    }
    friend __device__ inline bool operator==(const Kb& a, int b) {
        return a.asRaw() == Kb(b).asRaw();
    }
    friend __device__ inline bool operator==(int a, const Kb& b) {
        return Kb(a).asRaw() == b.asRaw();
    }

    /// Comparison operators
    __device__ bool operator<(Kb rhs) const { return val() < rhs.val(); }
    __device__ bool operator<=(Kb rhs) const { return val() <= rhs.val(); }
    __device__ bool operator>(Kb rhs) const { return val() > rhs.val(); }
    __device__ bool operator>=(Kb rhs) const { return val() >= rhs.val(); }

    /// Return a new Kb that is double of this value
    __device__ Kb doubled() const { return *this + *this; }

    /// Given an element x from a 31 bit field F_P compute x/2.
    static __device__ inline uint32_t halve_u32(uint32_t input) {
        uint32_t shr = input >> 1;
        uint32_t lo_bit = input & 1;
        return shr + (lo_bit * HALF_P_PLUS_1);
    }

    /// Return a new Kb that is half of this value
    __device__ Kb halve() const { return Kb::fromRaw(halve_u32(asRaw())); }
};

/// Raise a value to a power
__device__ inline Kb pow(Kb x, uint32_t n) {
    return Kb(static_cast<kb31_t>(x) ^ n);
}

template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
__device__ inline Kb pow(Kb x, I n) {
    return pow(x, static_cast<uint32_t>(n));
}

/// Compute the multiplicative inverse using Fermat's little theorem
/// For x = 0, returns 0.
__device__ inline Kb inv(Kb x) {
    if (x.asRaw() == 0u) {
        return Kb::zero();
    }
    return Kb(static_cast<kb31_t>(x).reciprocal());
}

static_assert(std::is_trivially_copyable<Kb>::value, "Kb must be POD-ish");
static_assert(sizeof(Kb) == 4, "Kb must be 4 bytes");
static_assert(alignof(Kb) == 4, "Kb must be 4-byte aligned");
static_assert(sizeof(Kb) == sizeof(kb31_t), "Kb and kb31_t sizes must match");
static_assert(alignof(Kb) == alignof(kb31_t), "Kb and kb31_t align must match");
