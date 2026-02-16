#pragma once

/// BabyBear field arithmetic in Montgomery form.
/// P = 15 * 2^27 + 1 = 2013265921
/// Montgomery constant M = 0x88000001
/// R^2 mod P = 1172168163

#include <metal_stdlib>
using namespace metal;

struct Fp {
    static constant uint32_t P = 15 * (uint32_t(1) << 27) + 1; // 2013265921
    static constant uint32_t M = 0x88000001u;
    static constant uint32_t R2 = 1172168163u;
    static constant uint32_t INVALID = 0xffffffffu;

    uint32_t val; // Always in Montgomery form, < P

    // --- Core modular ops (private-style) ---

    static constexpr uint32_t add_impl(uint32_t a, uint32_t b) {
        uint32_t r = a + b;
        return (r >= P) ? (r - P) : r;
    }

    static constexpr uint32_t sub_impl(uint32_t a, uint32_t b) {
        uint32_t r = a - b;
        return (r > P) ? (r + P) : r;
    }

    static constexpr uint32_t mul_impl(uint32_t a, uint32_t b) {
        uint64_t o64 = uint64_t(a) * uint64_t(b);
        uint32_t low = -uint32_t(o64);
        uint32_t red = M * low;
        o64 += uint64_t(red) * uint64_t(P);
        uint32_t ret = uint32_t(o64 >> 32);
        return (ret >= P) ? (ret - P) : ret;
    }

    static constexpr uint32_t encode(uint32_t a) { return mul_impl(R2, a); }
    static constexpr uint32_t decode(uint32_t a) { return mul_impl(1, a); }

    // --- Constructors ---

    constexpr Fp() thread : val(0) {}
    constexpr Fp() threadgroup : val(0) {}
    constexpr Fp() device : val(0) {}

    constexpr Fp(uint32_t v) thread : val(encode(v)) {}

    // Private-style: raw constructor (value already in Montgomery form)
    struct RawTag {};
    constexpr Fp(uint32_t v, RawTag) thread : val(v) {}

    static constexpr Fp fromRaw(uint32_t v) { return Fp(v, RawTag{}); }

    // --- Assignment ---

    constexpr void operator=(Fp rhs) thread { val = rhs.val; }
    constexpr void operator=(Fp rhs) device { val = rhs.val; }
    constexpr void operator=(Fp rhs) threadgroup { val = rhs.val; }

    // --- Accessors ---

    constexpr uint32_t asUInt32() const thread { return decode(val); }
    constexpr uint32_t asUInt32() const device { return decode(val); }
    constexpr uint32_t asUInt32() const threadgroup { return decode(val); }
    constexpr uint32_t asRaw() const thread { return val; }
    constexpr uint32_t asRaw() const device { return val; }
    constexpr uint32_t asRaw() const threadgroup { return val; }

    // --- Arithmetic operators (thread address space) ---

    constexpr Fp operator+(Fp rhs) const thread { return Fp(add_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-(Fp rhs) const thread { return Fp(sub_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator*(Fp rhs) const thread { return Fp(mul_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-() const thread { return Fp(sub_impl(0, val), RawTag{}); }

    // --- Arithmetic operators (threadgroup address space) ---

    constexpr Fp operator+(Fp rhs) const threadgroup { return Fp(add_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-(Fp rhs) const threadgroup { return Fp(sub_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator*(Fp rhs) const threadgroup { return Fp(mul_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-() const threadgroup { return Fp(sub_impl(0, val), RawTag{}); }

    // --- Arithmetic operators (device address space) ---

    constexpr Fp operator+(Fp rhs) const device { return Fp(add_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-(Fp rhs) const device { return Fp(sub_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator*(Fp rhs) const device { return Fp(mul_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-() const device { return Fp(sub_impl(0, val), RawTag{}); }

    // --- Arithmetic operators (constant address space) ---

    constexpr Fp operator+(Fp rhs) const constant { return Fp(add_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-(Fp rhs) const constant { return Fp(sub_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator*(Fp rhs) const constant { return Fp(mul_impl(val, rhs.val), RawTag{}); }
    constexpr Fp operator-() const constant { return Fp(sub_impl(0, val), RawTag{}); }

    constexpr Fp operator+=(Fp rhs) thread { val = add_impl(val, rhs.val); return *this; }
    constexpr Fp operator-=(Fp rhs) thread { val = sub_impl(val, rhs.val); return *this; }
    constexpr Fp operator*=(Fp rhs) thread { val = mul_impl(val, rhs.val); return *this; }

    constexpr Fp operator+=(Fp rhs) device { val = add_impl(val, rhs.val); return *this; }
    constexpr Fp operator-=(Fp rhs) device { val = sub_impl(val, rhs.val); return *this; }
    constexpr Fp operator*=(Fp rhs) device { val = mul_impl(val, rhs.val); return *this; }

    constexpr Fp operator+=(Fp rhs) threadgroup { val = add_impl(val, rhs.val); return *this; }
    constexpr Fp operator-=(Fp rhs) threadgroup { val = sub_impl(val, rhs.val); return *this; }
    constexpr Fp operator*=(Fp rhs) threadgroup { val = mul_impl(val, rhs.val); return *this; }

    // --- Comparison ---

    constexpr bool operator==(Fp rhs) const thread { return val == rhs.val; }
    constexpr bool operator!=(Fp rhs) const thread { return val != rhs.val; }
    constexpr bool operator==(Fp rhs) const threadgroup { return val == rhs.val; }
    constexpr bool operator!=(Fp rhs) const threadgroup { return val != rhs.val; }
    constexpr bool operator==(Fp rhs) const device { return val == rhs.val; }
    constexpr bool operator!=(Fp rhs) const device { return val != rhs.val; }
    constexpr bool operator==(Fp rhs) const constant { return val == rhs.val; }
    constexpr bool operator!=(Fp rhs) const constant { return val != rhs.val; }

    // --- Halve: multiply by inverse of 2 ---
    // 2^{-1} mod P = (P+1)/2 = 1006632961
    // In Montgomery form: if val is even, val/2; if odd, (val+P)/2
    constexpr Fp halve() const thread {
        uint32_t r = (val & 1) ? ((val >> 1) + ((P + 1) >> 1)) : (val >> 1);
        return Fp(r, RawTag{});
    }
    constexpr Fp halve() const threadgroup {
        uint32_t r = (val & 1) ? ((val >> 1) + ((P + 1) >> 1)) : (val >> 1);
        return Fp(r, RawTag{});
    }
    constexpr Fp halve() const device {
        uint32_t r = (val & 1) ? ((val >> 1) + ((P + 1) >> 1)) : (val >> 1);
        return Fp(r, RawTag{});
    }

    // --- Zeroize invalid ---
    constexpr Fp zeroize() const thread {
        return Fp((val == INVALID) ? 0u : val, RawTag{});
    }
    constexpr Fp zeroize() const threadgroup {
        return Fp((val == INVALID) ? 0u : val, RawTag{});
    }
    constexpr Fp zeroize() const device {
        return Fp((val == INVALID) ? 0u : val, RawTag{});
    }

    static constexpr Fp invalid() { return Fp(INVALID, RawTag{}); }
};

/// Raise Fp to a power (binary exponentiation)
constexpr inline Fp pow(Fp x, uint32_t n) {
    Fp tot = Fp(1);
    while (n != 0) {
        if (n & 1) {
            tot *= x;
        }
        n >>= 1;
        x *= x;
    }
    return tot;
}

/// Multiplicative inverse via Fermat's little theorem: x^{P-2}
constexpr inline Fp inv(Fp x) {
    return pow(x, Fp::P - 2);
}

/// Double an Fp value: 2*x
inline Fp fp_doubled(Fp x) {
    return x + x;
}

/// Halve an Fp value: x/2
inline Fp fp_halve(Fp x) {
    return x.halve();
}

/// Convert Fp to uint32_t (decode from Montgomery form)
inline uint32_t fp_to_uint(Fp x) {
    return x.asUInt32();
}

// BabyBear two-adic generators (in Montgomery form / raw representation).
// TWO_ADIC_GENERATORS[i] is a primitive 2^i-th root of unity.
// These match forward_roots_of_unity from sppark's baby_bear parameters.
constant Fp TWO_ADIC_GENERATORS[] = {
    Fp::fromRaw(0x0ffffffeu),  // level 0: 1 (trivial)
    Fp::fromRaw(0x68000003u),  // level 1: primitive 2nd root
    Fp::fromRaw(0x1c38d511u),  // level 2
    Fp::fromRaw(0x3d85298fu),  // level 3
    Fp::fromRaw(0x5f06e481u),  // level 4
    Fp::fromRaw(0x3f5c39ecu),  // level 5
    Fp::fromRaw(0x5516a97au),  // level 6
    Fp::fromRaw(0x3d6be592u),  // level 7
    Fp::fromRaw(0x5bb04149u),  // level 8
    Fp::fromRaw(0x4907f9abu),  // level 9
    Fp::fromRaw(0x548b8e90u),  // level 10
    Fp::fromRaw(0x1d8ca617u),  // level 11
    Fp::fromRaw(0x2ce7f0e6u),  // level 12
    Fp::fromRaw(0x621b371fu),  // level 13
    Fp::fromRaw(0x6d4d2d78u),  // level 14
    Fp::fromRaw(0x18716fcdu),  // level 15
    Fp::fromRaw(0x3b30a682u),  // level 16
    Fp::fromRaw(0x1c6f4728u),  // level 17
    Fp::fromRaw(0x59b01f7cu),  // level 18
    Fp::fromRaw(0x1a7f97acu),  // level 19
    Fp::fromRaw(0x0732561cu),  // level 20
    Fp::fromRaw(0x2b5a1cd4u),  // level 21
    Fp::fromRaw(0x6f7d26f9u),  // level 22
    Fp::fromRaw(0x16e2f919u),  // level 23
    Fp::fromRaw(0x285ab85bu),  // level 24
    Fp::fromRaw(0x0dd5a9ecu),  // level 25
    Fp::fromRaw(0x43f13568u),  // level 26
    Fp::fromRaw(0x57fab6eeu),  // level 27
};

// Inverse two-adic generators (in Montgomery form / raw representation).
constant Fp INVERSE_TWO_ADIC_GENERATORS[] = {
    Fp::fromRaw(0x0ffffffeu),  // level 0
    Fp::fromRaw(0x68000003u),  // level 1
    Fp::fromRaw(0x5bc72af0u),  // level 2
    Fp::fromRaw(0x02ec07f3u),  // level 3
    Fp::fromRaw(0x67e027cau),  // level 4
    Fp::fromRaw(0x5e1a0700u),  // level 5
    Fp::fromRaw(0x4bcc008cu),  // level 6
    Fp::fromRaw(0x0bed94d1u),  // level 7
    Fp::fromRaw(0x330b2e00u),  // level 8
    Fp::fromRaw(0x6b469805u),  // level 9
    Fp::fromRaw(0x0d83fad2u),  // level 10
    Fp::fromRaw(0x26e64394u),  // level 11
    Fp::fromRaw(0x0855523bu),  // level 12
    Fp::fromRaw(0x5c9f0045u),  // level 13
    Fp::fromRaw(0x5a7ba8c3u),  // level 14
    Fp::fromRaw(0x3c8b04e2u),  // level 15
    Fp::fromRaw(0x0c0f2066u),  // level 16
    Fp::fromRaw(0x1b51d34cu),  // level 17
    Fp::fromRaw(0x59f9bc12u),  // level 18
    Fp::fromRaw(0x3511f012u),  // level 19
    Fp::fromRaw(0x061ec85fu),  // level 20
    Fp::fromRaw(0x5fd09c6bu),  // level 21
    Fp::fromRaw(0x26bdc06cu),  // level 22
    Fp::fromRaw(0x1272832eu),  // level 23
    Fp::fromRaw(0x052ce2e8u),  // level 24
    Fp::fromRaw(0x02ff110du),  // level 25
    Fp::fromRaw(0x216ce204u),  // level 26
    Fp::fromRaw(0x5e12c8e9u),  // level 27
};

// Domain size inverse: 1/2^i in Montgomery form.
constant Fp DOMAIN_SIZE_INVERSE[] = {
    Fp::fromRaw(0x0ffffffeu),  // 1/2^0 = 1
    Fp::fromRaw(0x07ffffffu),  // 1/2^1
    Fp::fromRaw(0x40000000u),  // 1/2^2
    Fp::fromRaw(0x20000000u),  // 1/2^3
    Fp::fromRaw(0x10000000u),  // 1/2^4
    Fp::fromRaw(0x08000000u),  // 1/2^5
    Fp::fromRaw(0x04000000u),  // 1/2^6
    Fp::fromRaw(0x02000000u),  // 1/2^7
    Fp::fromRaw(0x01000000u),  // 1/2^8
    Fp::fromRaw(0x00800000u),  // 1/2^9
    Fp::fromRaw(0x00400000u),  // 1/2^10
    Fp::fromRaw(0x00200000u),  // 1/2^11
    Fp::fromRaw(0x00100000u),  // 1/2^12
    Fp::fromRaw(0x00080000u),  // 1/2^13
    Fp::fromRaw(0x00040000u),  // 1/2^14
    Fp::fromRaw(0x00020000u),  // 1/2^15
    Fp::fromRaw(0x00010000u),  // 1/2^16
    Fp::fromRaw(0x00008000u),  // 1/2^17
    Fp::fromRaw(0x00004000u),  // 1/2^18
    Fp::fromRaw(0x00002000u),  // 1/2^19
    Fp::fromRaw(0x00001000u),  // 1/2^20
    Fp::fromRaw(0x00000800u),  // 1/2^21
    Fp::fromRaw(0x00000400u),  // 1/2^22
    Fp::fromRaw(0x00000200u),  // 1/2^23
    Fp::fromRaw(0x00000100u),  // 1/2^24
    Fp::fromRaw(0x00000080u),  // 1/2^25
    Fp::fromRaw(0x00000040u),  // 1/2^26
    Fp::fromRaw(0x00000020u),  // 1/2^27
};
