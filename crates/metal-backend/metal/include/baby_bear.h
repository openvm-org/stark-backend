// BabyBear field arithmetic for Metal Shading Language.
//
// Based on risc0/risc0/build_kernel/kernels/metal/fp.h (Apache-2.0)
// and openvm-org cuda-common/include/fp.h
//
// BabyBear prime: P = 15 * 2^27 + 1 = 2013265921
// Montgomery form: values stored as a * R mod P where R = 2^32
// Montgomery constant M = 0x88000001
// R^2 mod P = 1172168163

#pragma once

#include <metal_stdlib>

using namespace metal;

/// Fp is an element of the BabyBear prime field F_P where P = 15 * 2^27 + 1.
///
/// Internally values are stored in Montgomery form. All arithmetic operations
/// work directly on the Montgomery representation so that multiplication can
/// use the efficient Montgomery reduction (no expensive division by P).
class Fp {
public:
    /// The prime modulus P = 2013265921
    static constant uint32_t P = 15 * (uint32_t(1) << 27) + 1;

    /// Montgomery constant: M = -P^{-1} mod 2^32 = 0x88000001
    static constant uint32_t M = 0x88000001;

    /// R^2 mod P, used for encoding into Montgomery form
    static constant uint32_t R2 = 1172168163;

    /// Sentinel value for "invalid" field element
    static constant uint32_t INVALID = 0xffffffff;

    /// (P + 1) / 2 -- used by halve()
    static constant uint32_t HALF_P_PLUS_1 = (P + 1) >> 1;

    /// Two-adicity of the multiplicative group (P - 1 = 15 * 2^27)
    static constant uint32_t TWO_ADICITY = 27;

private:
    /// The stored value, always in Montgomery form and < P.
    uint32_t val;

    // ----------------------------------------------------------------
    // Core modular operations (Montgomery form)
    // ----------------------------------------------------------------

    /// Modular addition: (a + b) mod P
    static constexpr uint32_t add(uint32_t a, uint32_t b) {
        uint32_t r = a + b;
        return (r >= P ? r - P : r);
    }

    /// Modular subtraction: (a - b) mod P
    static constexpr uint32_t sub(uint32_t a, uint32_t b) {
        uint32_t r = a - b;
        return (r > P ? r + P : r);
    }

    /// Montgomery multiplication: returns (a * b * R^{-1}) mod P
    static constexpr uint32_t mul(uint32_t a, uint32_t b) {
        uint64_t o64 = uint64_t(a) * uint64_t(b);
        uint32_t low = -uint32_t(o64);
        uint32_t red = M * low;
        o64 += uint64_t(red) * uint64_t(P);
        uint32_t ret = o64 >> 32;
        return (ret >= P ? ret - P : ret);
    }

    /// Encode a normal integer into Montgomery form: a * R mod P
    static constexpr uint32_t encode(uint32_t a) { return mul(R2, a); }

    /// Decode from Montgomery form back to a normal integer: a * R^{-1} mod P
    static constexpr uint32_t decode(uint32_t a) { return mul(1, a); }

    /// Private constructor from raw Montgomery-form value.
    /// The bool parameter is unused, it just differentiates from the public ctor.
    constexpr Fp(uint32_t v, bool) : val(v) {}

public:
    // ----------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------

    /// Default constructor: zero element
    constexpr Fp() : val(0) {}

    /// Construct from a normal uint32_t. The value is encoded into Montgomery form.
    constexpr Fp(uint32_t v) : val(encode(v % P)) {}

    /// Construct from an already-encoded raw Montgomery value (no encoding step).
    static constexpr Fp fromRaw(uint32_t v) { return Fp(v, true); }

    // ----------------------------------------------------------------
    // Named constructors / constants
    // ----------------------------------------------------------------

    /// Zero element
    static constexpr Fp zero() { return Fp(0, true); }

    /// One element (1 in Montgomery form = R mod P)
    static constexpr Fp one() { return Fp(encode(1), true); }

    /// Negative one element: P - 1 in Montgomery form
    static constexpr Fp neg_one() { return Fp(sub(0, encode(1)), true); }

    /// Maximum representable value (P - 1 as a field element)
    static constexpr Fp maxVal() { return Fp(P - 1); }

    /// Sentinel invalid value
    static constexpr Fp invalid() { return Fp(INVALID, true); }

    // ----------------------------------------------------------------
    // Conversions
    // ----------------------------------------------------------------

    /// Decode from Montgomery form and return as a plain uint32_t
    constexpr uint32_t asUInt32() const { return decode(val); }

    /// Return the raw Montgomery-form word (no decoding)
    constexpr uint32_t asRaw() const { return val; }

    // ----------------------------------------------------------------
    // Special operations
    // ----------------------------------------------------------------

    /// Replace invalid sentinel with zero
    constexpr Fp zeroize() {
        if (val == INVALID) {
            val = 0;
        }
        return *this;
    }

    /// Return 2 * this (mod P) without a full multiply
    constexpr Fp doubled() const { return Fp(add(val, val), true); }

    /// Return this / 2 (mod P)
    constexpr Fp halve() const {
        uint32_t v = val;
        uint32_t shr = v >> 1;
        uint32_t lo_bit = v & 1;
        return Fp(shr + (lo_bit * HALF_P_PLUS_1), true);
    }

    /// Square in-place (returns the squared value for chaining)
    constexpr Fp sqr() const { return Fp(mul(val, val), true); }

    // ----------------------------------------------------------------
    // Assignment
    // ----------------------------------------------------------------

    constexpr void operator=(uint32_t rhs) { val = encode(rhs % P); }

    // ----------------------------------------------------------------
    // Arithmetic operators
    // ----------------------------------------------------------------

    constexpr Fp operator+(Fp rhs) const { return Fp(add(val, rhs.val), true); }
    constexpr Fp operator-() const { return Fp(sub(0, val), true); }
    constexpr Fp operator-(Fp rhs) const { return Fp(sub(val, rhs.val), true); }
    constexpr Fp operator*(Fp rhs) const { return Fp(mul(val, rhs.val), true); }

    constexpr Fp operator+=(Fp rhs) { val = add(val, rhs.val); return *this; }
    constexpr Fp operator-=(Fp rhs) { val = sub(val, rhs.val); return *this; }
    constexpr Fp operator*=(Fp rhs) { val = mul(val, rhs.val); return *this; }

    // ----------------------------------------------------------------
    // Comparison operators
    // ----------------------------------------------------------------

    constexpr bool operator==(Fp rhs) const { return val == rhs.val; }
    constexpr bool operator!=(Fp rhs) const { return val != rhs.val; }
    constexpr bool operator<(Fp rhs) const { return decode(val) < decode(rhs.val); }
    constexpr bool operator<=(Fp rhs) const { return decode(val) <= decode(rhs.val); }
    constexpr bool operator>(Fp rhs) const { return decode(val) > decode(rhs.val); }
    constexpr bool operator>=(Fp rhs) const { return decode(val) >= decode(rhs.val); }

    // ----------------------------------------------------------------
    // device-qualified overloads (required by MSL for device pointers)
    // ----------------------------------------------------------------

    constexpr Fp operator+(Fp rhs) device const { return Fp(add(val, rhs.val), true); }
    constexpr Fp operator-() device const { return Fp(sub(0, val), true); }
    constexpr Fp operator-(Fp rhs) device const { return Fp(sub(val, rhs.val), true); }
    constexpr Fp operator*(Fp rhs) device const { return Fp(mul(val, rhs.val), true); }

    constexpr Fp operator+=(Fp rhs) device { val = add(val, rhs.val); return *this; }
    constexpr Fp operator-=(Fp rhs) device { val = sub(val, rhs.val); return *this; }
    constexpr Fp operator*=(Fp rhs) device { val = mul(val, rhs.val); return *this; }

    constexpr bool operator==(Fp rhs) device const { return val == rhs.val; }
    constexpr bool operator!=(Fp rhs) device const { return val != rhs.val; }

    constexpr uint32_t asUInt32() device const { return decode(val); }
    constexpr uint32_t asRaw() device const { return val; }

    constexpr void operator=(uint32_t rhs) device { val = encode(rhs % P); }

    // ----------------------------------------------------------------
    // Increment / decrement
    // ----------------------------------------------------------------

    constexpr Fp operator++(int) { Fp r = *this; val = add(val, encode(1)); return r; }
    constexpr Fp operator--(int) { Fp r = *this; val = sub(val, encode(1)); return r; }
    constexpr Fp operator++() { val = add(val, encode(1)); return *this; }
    constexpr Fp operator--() { val = sub(val, encode(1)); return *this; }
};

// ============================================================================
// Free functions
// ============================================================================

/// Raise x to the n-th power via square-and-multiply
constexpr inline Fp pow(Fp x, size_t n) {
    Fp tot = Fp(1);
    while (n != 0) {
        if (n % 2 == 1) {
            tot *= x;
        }
        n = n / 2;
        x *= x;
    }
    return tot;
}

/// Multiplicative inverse via Fermat's little theorem: x^{P-2} mod P.
/// inv(0) returns 0 by convention.
constexpr inline Fp inv(Fp x) {
    return pow(x, Fp::P - 2);
}

/// Two-adic generators for the BabyBear field.
/// TWO_ADIC_GENERATORS[k] is a primitive 2^k-th root of unity.
constant Fp TWO_ADIC_GENERATORS[28] = {
    Fp(0x1),
    Fp(0x78000000),
    Fp(0x67055c21),
    Fp(0x5ee99486),
    Fp(0xbb4c4e4),
    Fp(0x2d4cc4da),
    Fp(0x669d6090),
    Fp(0x17b56c64),
    Fp(0x67456167),
    Fp(0x688442f9),
    Fp(0x145e952d),
    Fp(0x4fe61226),
    Fp(0x4c734715),
    Fp(0x11c33e2a),
    Fp(0x62c3d2b1),
    Fp(0x77cad399),
    Fp(0x54c131f4),
    Fp(0x4cabd6a6),
    Fp(0x5cf5713f),
    Fp(0x3e9430e8),
    Fp(0xba067a3),
    Fp(0x18adc27d),
    Fp(0x21fd55bc),
    Fp(0x4b859b3d),
    Fp(0x3bd57996),
    Fp(0x4483d85a),
    Fp(0x3a26eef8),
    Fp(0x1a427a41)
};
