#pragma once

/// BabyBear extension field F_p^4 = F_p[X] / (X^4 - 11).
/// Degree 4 extension with irreducible polynomial x^4 - 11.

#include <metal_stdlib>
using namespace metal;

#include "baby_bear.h"

#define BETA Fp(11)
#define NBETA Fp(Fp::P - 11)

struct FpExt {
    Fp elems[4]; // elems[0] + elems[1]*X + elems[2]*X^2 + elems[3]*X^3

    // --- Constructors ---

    constexpr FpExt() thread {}
    constexpr FpExt() threadgroup {}
    constexpr FpExt() device {}

    explicit constexpr FpExt(uint32_t x) thread {
        elems[0] = Fp(x);
        elems[1] = Fp(0);
        elems[2] = Fp(0);
        elems[3] = Fp(0);
    }

    explicit constexpr FpExt(Fp x) thread {
        elems[0] = x;
        elems[1] = Fp(0);
        elems[2] = Fp(0);
        elems[3] = Fp(0);
    }

    constexpr FpExt(Fp a, Fp b, Fp c, Fp d) thread {
        elems[0] = a;
        elems[1] = b;
        elems[2] = c;
        elems[3] = d;
    }

    // --- Assignment ---
    constexpr void operator=(FpExt rhs) thread { for (int i = 0; i < 4; i++) elems[i] = rhs.elems[i]; }
    constexpr void operator=(FpExt rhs) device { for (int i = 0; i < 4; i++) elems[i] = rhs.elems[i]; }
    constexpr void operator=(FpExt rhs) threadgroup { for (int i = 0; i < 4; i++) elems[i] = rhs.elems[i]; }

    constexpr FpExt zeroize() const thread {
        FpExt r;
        for (uint32_t i = 0; i < 4; i++) {
            r.elems[i] = elems[i].zeroize();
        }
        return r;
    }
    constexpr FpExt zeroize() const device {
        FpExt r;
        for (uint32_t i = 0; i < 4; i++) {
            r.elems[i] = Fp::fromRaw(elems[i].val).zeroize();
        }
        return r;
    }

    // --- Addition / Subtraction ---

    constexpr FpExt operator+=(FpExt rhs) thread {
        for (uint32_t i = 0; i < 4; i++) elems[i] += rhs.elems[i];
        return *this;
    }

    constexpr FpExt operator+=(FpExt rhs) device {
        for (uint32_t i = 0; i < 4; i++) elems[i] += rhs.elems[i];
        return *this;
    }

    constexpr FpExt operator+=(FpExt rhs) threadgroup {
        for (uint32_t i = 0; i < 4; i++) elems[i] += rhs.elems[i];
        return *this;
    }

    constexpr FpExt operator-=(FpExt rhs) thread {
        for (uint32_t i = 0; i < 4; i++) elems[i] -= rhs.elems[i];
        return *this;
    }

    constexpr FpExt operator-=(FpExt rhs) device {
        for (uint32_t i = 0; i < 4; i++) elems[i] -= rhs.elems[i];
        return *this;
    }

    constexpr FpExt operator-=(FpExt rhs) threadgroup {
        for (uint32_t i = 0; i < 4; i++) elems[i] -= rhs.elems[i];
        return *this;
    }

    constexpr FpExt operator+(FpExt rhs) const thread {
        FpExt r = *this;
        r += rhs;
        return r;
    }

    constexpr FpExt operator-(FpExt rhs) const thread {
        FpExt r = *this;
        r -= rhs;
        return r;
    }

    constexpr FpExt operator-() const thread { return FpExt() - *this; }

    // --- Scalar multiplication (Fp * FpExt) ---

    constexpr FpExt operator*=(Fp rhs) thread {
        for (uint32_t i = 0; i < 4; i++) elems[i] *= rhs;
        return *this;
    }

    constexpr FpExt operator*=(Fp rhs) device {
        for (uint32_t i = 0; i < 4; i++) elems[i] *= rhs;
        return *this;
    }

    constexpr FpExt operator*(Fp rhs) const thread {
        FpExt r = *this;
        r *= rhs;
        return r;
    }

    // --- Full extension multiplication ---
    // Multiply modulo x^4 - 11. Powers >= 4 wrap with x^4 = 11.

    constexpr FpExt operator*(FpExt rhs) const thread {
        return FpExt(
            elems[0] * rhs.elems[0] + BETA * (elems[1] * rhs.elems[3] + elems[2] * rhs.elems[2] + elems[3] * rhs.elems[1]),
            elems[0] * rhs.elems[1] + elems[1] * rhs.elems[0] + BETA * (elems[2] * rhs.elems[3] + elems[3] * rhs.elems[2]),
            elems[0] * rhs.elems[2] + elems[1] * rhs.elems[1] + elems[2] * rhs.elems[0] + BETA * (elems[3] * rhs.elems[3]),
            elems[0] * rhs.elems[3] + elems[1] * rhs.elems[2] + elems[2] * rhs.elems[1] + elems[3] * rhs.elems[0]
        );
    }

    constexpr FpExt operator*=(FpExt rhs) thread {
        *this = *this * rhs;
        return *this;
    }

    constexpr FpExt operator*=(FpExt rhs) device {
        *this = FpExt(*this) * rhs;
        return *this;
    }

    // --- Equality ---

    constexpr bool operator==(FpExt rhs) const thread {
        for (uint32_t i = 0; i < 4; i++) {
            if (elems[i] != rhs.elems[i]) return false;
        }
        return true;
    }

    constexpr bool operator!=(FpExt rhs) const thread { return !(*this == rhs); }

    constexpr Fp constPart() const thread { return elems[0]; }

    // --- Device address space operators ---

    constexpr FpExt operator+(FpExt rhs) const device { return FpExt(*this) + rhs; }
    constexpr FpExt operator-(FpExt rhs) const device { return FpExt(*this) - rhs; }
    constexpr FpExt operator-() const device { return -FpExt(*this); }
    constexpr FpExt operator*(FpExt rhs) const device { return FpExt(*this) * rhs; }
    constexpr FpExt operator*(Fp rhs) const device { return FpExt(*this) * rhs; }
    constexpr bool operator==(FpExt rhs) const device { return FpExt(*this) == rhs; }
    constexpr bool operator!=(FpExt rhs) const device { return FpExt(*this) != rhs; }
    constexpr Fp constPart() const device { return FpExt(*this).constPart(); }

    // --- Threadgroup address space operators ---

    constexpr FpExt operator+(FpExt rhs) const threadgroup { return FpExt(*this) + rhs; }
    constexpr FpExt operator-(FpExt rhs) const threadgroup { return FpExt(*this) - rhs; }
    constexpr FpExt operator-() const threadgroup { return -FpExt(*this); }
    constexpr FpExt operator*(FpExt rhs) const threadgroup { return FpExt(*this) * rhs; }
    constexpr FpExt operator*(Fp rhs) const threadgroup { return FpExt(*this) * rhs; }
    constexpr bool operator==(FpExt rhs) const threadgroup { return FpExt(*this) == rhs; }
    constexpr bool operator!=(FpExt rhs) const threadgroup { return FpExt(*this) != rhs; }
    constexpr Fp constPart() const threadgroup { return FpExt(*this).constPart(); }

    // --- Constant address space operators ---

    constexpr FpExt operator+(FpExt rhs) const constant { return FpExt(*this) + rhs; }
    constexpr FpExt operator-(FpExt rhs) const constant { return FpExt(*this) - rhs; }
    constexpr FpExt operator-() const constant { return -FpExt(*this); }
    constexpr FpExt operator*(FpExt rhs) const constant { return FpExt(*this) * rhs; }
    constexpr FpExt operator*(Fp rhs) const constant { return FpExt(*this) * rhs; }
    constexpr bool operator==(FpExt rhs) const constant { return FpExt(*this) == rhs; }
    constexpr bool operator!=(FpExt rhs) const constant { return FpExt(*this) != rhs; }
    constexpr Fp constPart() const constant { return FpExt(*this).constPart(); }
};

/// Fp * FpExt (free function for commutativity)
constexpr inline FpExt operator*(Fp a, FpExt b) {
    return b * a;
}

/// Raise FpExt to a power
constexpr inline FpExt pow(FpExt x, uint32_t n) {
    FpExt tot(1);
    while (n != 0) {
        if (n & 1) {
            tot *= x;
        }
        n >>= 1;
        x *= x;
    }
    return tot;
}

/// Multiplicative inverse of FpExt using composite field inversion.
constexpr inline FpExt inv(FpExt in) {
    Fp b0 = in.elems[0] * in.elems[0]
        + BETA * (in.elems[2] * in.elems[2] - in.elems[1] * (in.elems[3] + in.elems[3]));
    Fp b2 = in.elems[0] * (in.elems[2] + in.elems[2])
        - in.elems[1] * in.elems[1]
        - BETA * (in.elems[3] * in.elems[3]);
    Fp c = b0 * b0 - BETA * b2 * b2;
    Fp ic = inv(c);
    b0 *= ic;
    b2 *= ic;
    return FpExt(
        in.elems[0] * b0 - BETA * in.elems[2] * b2,
        -in.elems[1] * b0 + BETA * in.elems[3] * b2,
        -in.elems[0] * b2 + in.elems[2] * b0,
        in.elems[1] * b2 - in.elems[3] * b0
    );
}

/// Negate FpExt (free function for convenience in DAG evaluation)
constexpr inline FpExt fpext_neg(FpExt x) {
    return -x;
}

/// Multiplicative inverse of FpExt (free function alias)
constexpr inline FpExt fpext_inv(FpExt x) {
    return inv(x);
}

#undef BETA
#undef NBETA
