/*
 * Kb5 - Quintic Extension of KoalaBear Field (Plonky3 polynomial)
 *
 * Defines Kb5, a finite field F_p^5, based on Kb via the irreducible polynomial x^5 + x^2 - 1.
 * This is the same polynomial used in Plonky3: https://github.com/Plonky3/Plonky3/pull/1293
 *
 * NOTE: Unlike BabyBear, KoalaBear has gcd(5, p-1) = 1, so x^5 - W is NEVER irreducible
 * (every element is a 5th power). We use the sparse trinomial x^5 + x^2 - 1 instead.
 *
 * Field size: p^5 ≈ 2^155, provides ~155 bits of security.
 *
 * Element representation: a0 + a1*α + a2*α^2 + a3*α^3 + a4*α^4
 * where each ai is a KoalaBear element and α^5 = -α^2 + 1.
 *
 * Reduction rule: α^5 = -α^2 + 1
 * Powers: α^6 = -α^3 + α, α^7 = -α^4 + α^2, α^8 = α^3 + α^2 - 1
 *
 * For product giving c0..c8:
 *   r0 = c0 + c5 - c8
 *   r1 = c1 + c6
 *   r2 = c2 - c5 + c7 + c8
 *   r3 = c3 - c6 + c8
 *   r4 = c4 - c7
 *
 * Inversion via Gaussian elimination: O(125) Kb operations.
 */

#pragma once

#include "kb.h"

/// Kb5 is a degree-5 extension of the KoalaBear field using x^5 + x^2 - 1.
/// Elements are represented as polynomials in F_p[x] / (x^5 + x^2 - 1).
struct Kb5 {
    /// Coefficients: elems[0] + elems[1]*α + elems[2]*α^2 + elems[3]*α^3 + elems[4]*α^4
    Kb elems[5];

    /// Default constructor - zero element
    __device__ Kb5() : elems{Kb(0), Kb(0), Kb(0), Kb(0), Kb(0)} {}

    /// Construct from a single Kb (embed base field)
    __device__ explicit Kb5(Kb x) : elems{x, Kb(0), Kb(0), Kb(0), Kb(0)} {}

    /// Construct from uint32_t
    __device__ explicit Kb5(uint32_t x) : elems{Kb(x), Kb(0), Kb(0), Kb(0), Kb(0)} {}

    /// Construct from all 5 coefficients
    __device__ Kb5(Kb a0, Kb a1, Kb a2, Kb a3, Kb a4)
        : elems{a0, a1, a2, a3, a4} {}

    /// Zero element
    __device__ static Kb5 zero() { return Kb5(); }

    /// One element
    __device__ static Kb5 one() { return Kb5(Kb::one()); }

    /// Multiply two degree-1 polynomials (a0 + a1*x)(b0 + b1*x)
    __device__ static inline void mul_deg1(
        Kb a0, Kb a1, Kb b0, Kb b1,
        Kb& r0, Kb& r1, Kb& r2
    ) {
        Kb v0 = a0 * b0;
        Kb v1 = a1 * b1;
        Kb v01 = (a0 + a1) * (b0 + b1);
        r0 = v0;
        r1 = v01 - v0 - v1;
        r2 = v1;
    }

    /// Multiply two degree-2 polynomials (a0 + a1*x + a2*x^2)(b0 + b1*x + b2*x^2)
    /// Using Toom-2.5 (6 muls)
    __device__ static inline void mul_deg2(
        Kb a0, Kb a1, Kb a2, Kb b0, Kb b1, Kb b2,
        Kb& r0, Kb& r1, Kb& r2, Kb& r3, Kb& r4
    ) {
        Kb v0 = a0 * b0;
        Kb v1 = a1 * b1;
        Kb v2 = a2 * b2;
        Kb v01 = (a0 + a1) * (b0 + b1);
        Kb v12 = (a1 + a2) * (b1 + b2);
        Kb v02 = (a0 + a2) * (b0 + b2);

        r0 = v0;
        r1 = v01 - v0 - v1;
        r2 = v02 - v0 - v2 + v1;
        r3 = v12 - v1 - v2;
        r4 = v2;
    }

    /// Access coefficient
    __device__ Kb& operator[](int i) { return elems[i]; }
    __device__ const Kb& operator[](int i) const { return elems[i]; }

    /// Get constant part (coefficient of α^0)
    __device__ Kb constPart() const { return elems[0]; }

    // ========================================================================
    // Addition / Subtraction (component-wise)
    // ========================================================================

    __device__ Kb5 operator+(Kb5 rhs) const {
        return Kb5(
            elems[0] + rhs.elems[0],
            elems[1] + rhs.elems[1],
            elems[2] + rhs.elems[2],
            elems[3] + rhs.elems[3],
            elems[4] + rhs.elems[4]
        );
    }

    __device__ Kb5 operator-(Kb5 rhs) const {
        return Kb5(
            elems[0] - rhs.elems[0],
            elems[1] - rhs.elems[1],
            elems[2] - rhs.elems[2],
            elems[3] - rhs.elems[3],
            elems[4] - rhs.elems[4]
        );
    }

    __device__ Kb5 operator-() const {
        return Kb5() - *this;
    }

    __device__ Kb5& operator+=(Kb5 rhs) {
        *this = *this + rhs;
        return *this;
    }

    __device__ Kb5& operator-=(Kb5 rhs) {
        *this = *this - rhs;
        return *this;
    }

    // ========================================================================
    // Multiplication
    // ========================================================================

    /// Multiply by scalar (base field element)
    __device__ Kb5 operator*(Kb rhs) const {
        return Kb5(
            elems[0] * rhs,
            elems[1] * rhs,
            elems[2] * rhs,
            elems[3] * rhs,
            elems[4] * rhs
        );
    }

    __device__ Kb5& operator*=(Kb rhs) {
        *this = *this * rhs;
        return *this;
    }

    /// Full extension field multiplication
    /// Uses Karatsuba split (2+3) + reduction mod (x^5 + x^2 - 1)
    ///
    /// Reduction: α^5 = -α^2 + 1
    /// For degree-8 product t0 + t1*α + ... + t8*α^8:
    ///   c0 = t0 + t5 - t8
    ///   c1 = t1 + t6
    ///   c2 = t2 - t5 + t7 + t8
    ///   c3 = t3 - t6 + t8
    ///   c4 = t4 - t7
    __device__ Kb5 operator*(Kb5 rhs) const {
        const Kb a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        const Kb b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2], b3 = rhs.elems[3], b4 = rhs.elems[4];

        // Split: A = A0 + x^2*A1, where A0 = a0 + a1*x, A1 = a2 + a3*x + a4*x^2
        // Same for B. Uses 15 Kb muls vs 25 schoolbook.
        Kb p0_0, p0_1, p0_2;
        mul_deg1(a0, a1, b0, b1, p0_0, p0_1, p0_2);

        Kb p2_0, p2_1, p2_2, p2_3, p2_4;
        mul_deg2(a2, a3, a4, b2, b3, b4, p2_0, p2_1, p2_2, p2_3, p2_4);

        Kb s0 = a0 + a2;
        Kb s1 = a1 + a3;
        Kb s2 = a4;
        Kb t0s = b0 + b2;
        Kb t1s = b1 + b3;
        Kb t2s = b4;

        Kb p1_0, p1_1, p1_2, p1_3, p1_4;
        mul_deg2(s0, s1, s2, t0s, t1s, t2s, p1_0, p1_1, p1_2, p1_3, p1_4);

        // P1 = (A0+A1)(B0+B1) - P0 - P2
        p1_0 = p1_0 - p0_0 - p2_0;
        p1_1 = p1_1 - p0_1 - p2_1;
        p1_2 = p1_2 - p0_2 - p2_2;
        p1_3 = p1_3 - p2_3;
        p1_4 = p1_4 - p2_4;

        // Compose full product coefficients t0..t8
        Kb t0 = p0_0;
        Kb t1 = p0_1;
        Kb t2 = p0_2 + p1_0;
        Kb t3 = p1_1;
        Kb t4 = p1_2 + p2_0;
        Kb t5 = p1_3 + p2_1;
        Kb t6 = p1_4 + p2_2;
        Kb t7 = p2_3;
        Kb t8 = p2_4;

        // Reduce: α^5 = -α^2 + 1
        Kb c0 = t0 + t5 - t8;
        Kb c1 = t1 + t6;
        Kb c2 = t2 - t5 + t7 + t8;
        Kb c3 = t3 - t6 + t8;
        Kb c4 = t4 - t7;

        return Kb5(c0, c1, c2, c3, c4);
    }

    __device__ Kb5& operator*=(Kb5 rhs) {
        *this = *this * rhs;
        return *this;
    }

    /// Squaring (optimized using symmetry)
    __device__ Kb5 square() const {
        const Kb a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];

        // Squares
        Kb a0sq = a0 * a0;
        Kb a1sq = a1 * a1;
        Kb a2sq = a2 * a2;
        Kb a3sq = a3 * a3;
        Kb a4sq = a4 * a4;

        // Cross products (each appears twice in convolution)
        Kb a0a1 = a0 * a1;
        Kb a0a2 = a0 * a2;
        Kb a0a3 = a0 * a3;
        Kb a0a4 = a0 * a4;
        Kb a1a2 = a1 * a2;
        Kb a1a3 = a1 * a3;
        Kb a1a4 = a1 * a4;
        Kb a2a3 = a2 * a3;
        Kb a2a4 = a2 * a4;
        Kb a3a4 = a3 * a4;

        // Convolution: t[k] = sum_{i+j=k} a[i]*a[j]
        Kb t0 = a0sq;
        Kb t1 = a0a1 + a0a1;  // 2*a0*a1
        Kb t2 = a0a2 + a0a2 + a1sq;  // 2*a0*a2 + a1^2
        Kb t3 = a0a3 + a0a3 + a1a2 + a1a2;  // 2*(a0*a3 + a1*a2)
        Kb t4 = a0a4 + a0a4 + a1a3 + a1a3 + a2sq;  // 2*(a0*a4 + a1*a3) + a2^2
        Kb t5 = a1a4 + a1a4 + a2a3 + a2a3;  // 2*(a1*a4 + a2*a3)
        Kb t6 = a2a4 + a2a4 + a3sq;  // 2*a2*a4 + a3^2
        Kb t7 = a3a4 + a3a4;  // 2*a3*a4
        Kb t8 = a4sq;

        // Reduce: α^5 = -α^2 + 1
        Kb c0 = t0 + t5 - t8;
        Kb c1 = t1 + t6;
        Kb c2 = t2 - t5 + t7 + t8;
        Kb c3 = t3 - t6 + t8;
        Kb c4 = t4 - t7;

        return Kb5(c0, c1, c2, c3, c4);
    }

    // ========================================================================
    // Equality
    // ========================================================================

    __device__ bool operator==(Kb5 rhs) const {
        return elems[0] == rhs.elems[0] &&
               elems[1] == rhs.elems[1] &&
               elems[2] == rhs.elems[2] &&
               elems[3] == rhs.elems[3] &&
               elems[4] == rhs.elems[4];
    }

    __device__ bool operator!=(Kb5 rhs) const {
        return !(*this == rhs);
    }
};

/// Scalar * Kb5
__device__ inline Kb5 operator*(Kb a, Kb5 b) { return b * a; }

/// Power function using square-and-multiply
__device__ inline Kb5 pow(Kb5 base, uint32_t exp) {
    Kb5 result = Kb5::one();
    while (exp > 0) {
        if (exp & 1) {
            result = result * base;
        }
        base = base.square();
        exp >>= 1;
    }
    return result;
}

/// Inversion using Gaussian elimination on the multiplication matrix.
///
/// For trinomial x^5 + x^2 - 1, we need to compute the multiplication matrix.
/// α^5 = -α^2 + 1 gives us a cleaner structure than x^5 + x + 4.
__device__ inline Kb5 inv(Kb5 x) {
    // Handle zero case
    if (x == Kb5::zero()) {
        return Kb5::zero();
    }

    const Kb* a = x.elems;
    // Build augmented matrix [M | I] for Gauss-Jordan elimination
    // M is the multiplication matrix, I is identity
    // We'll solve M * result = e0 = (1, 0, 0, 0, 0)

    // Matrix storage: m[row][col], augmented with result column
    Kb m[5][6];

    // For x^5 + x^2 - 1 (α^5 = -α^2 + 1), compute columns for a*α^j:
    // Column 0 (a*1): [a0, a1, a2, a3, a4]
    // Column 1 (a*α): multiply by α, use α^5 = -α^2 + 1
    // Column 2 (a*α^2): multiply by α^2
    // Column 3 (a*α^3): multiply by α^3
    // Column 4 (a*α^4): multiply by α^4

    // Row 0 (coefficient of α^0)
    m[0][0] = a[0];  // from a*1
    m[0][1] = a[4];  // from a*α: a4*α^5 = a4*(-α^2+1) contributes a4 to constant term
    m[0][2] = a[3];  // from a*α^2: a3*α^5 contributes a3
    m[0][3] = a[2];  // from a*α^3: a2*α^5 contributes a2
    m[0][4] = a[1] - a[4];  // from a*α^4: a1*α^5 - a4*α^8, where α^8 = α^3+α^2-1 contributes -a4
    m[0][5] = Kb::one();

    // Row 1 (coefficient of α^1)
    m[1][0] = a[1];  // from a*1
    m[1][1] = a[0];  // from a*α: a0*α
    m[1][2] = a[4];  // from a*α^2: a4*α^6 = a4*(-α^3+α) contributes a4 to α^1
    m[1][3] = a[3];  // from a*α^3: a3*α^6 contributes a3
    m[1][4] = a[2];  // from a*α^4: a2*α^6 contributes a2
    m[1][5] = Kb::zero();

    // Row 2 (coefficient of α^2)
    m[2][0] = a[2];  // from a*1
    m[2][1] = a[1] - a[4];  // from a*α: a1*α^2 + a4*α^5, α^5 = -α^2+1 contributes -a4 to α^2
    m[2][2] = a[0] - a[3];  // from a*α^2: a0*α^2 + a3*α^5 contributes -a3 to α^2
    m[2][3] = a[4] - a[2];  // from a*α^3: a4*α^7 = a4*(-α^4+α^2), a2*α^5 = a2*(-α^2+1)
    m[2][4] = a[3] + a[4] - a[1];  // from a*α^4: complex term from α^8
    m[2][5] = Kb::zero();

    // Row 3 (coefficient of α^3)
    m[3][0] = a[3];  // from a*1
    m[3][1] = a[2];  // from a*α: a2*α^3
    m[3][2] = a[1] - a[4];  // from a*α^2: a1*α^3 + a4*α^6, α^6 = -α^3+α contributes -a4
    m[3][3] = a[0] - a[3];  // from a*α^3: a0*α^3 + a3*α^6 contributes -a3
    m[3][4] = a[4] - a[2];  // from a*α^4: a4*α^7+a2*α^5 where α^7 = -α^4+α^2, α^8 contributes
    m[3][5] = Kb::zero();

    // Row 4 (coefficient of α^4)
    m[4][0] = a[4];  // from a*1
    m[4][1] = a[3];  // from a*α: a3*α^4
    m[4][2] = a[2];  // from a*α^2: a2*α^4
    m[4][3] = a[1] - a[4];  // from a*α^3: a1*α^4 + a4*α^7, α^7 = -α^4+α^2 contributes -a4
    m[4][4] = a[0] - a[3];  // from a*α^4: a0*α^4 + a3*α^7 contributes -a3
    m[4][5] = Kb::zero();

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 5; col++) {
        // Find pivot
        int pivot_row = col;
        while (pivot_row < 5 && m[pivot_row][col] == Kb::zero()) {
            pivot_row++;
        }

        if (pivot_row >= 5) {
            return Kb5::zero();
        }

        // Pick best pivot for stability
        for (int row = pivot_row + 1; row < 5; row++) {
            if (m[row][col] != Kb::zero() && m[row][col].asRaw() > m[pivot_row][col].asRaw()) {
                pivot_row = row;
            }
        }

        // Swap rows
        if (pivot_row != col) {
            for (int j = 0; j < 6; j++) {
                Kb tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }

        // Scale pivot row
        Kb pivot_inv = ::inv(m[col][col]);
        for (int j = col; j < 6; j++) {
            m[col][j] = m[col][j] * pivot_inv;
        }

        // Eliminate column in other rows
        for (int row = 0; row < 5; row++) {
            if (row != col) {
                Kb factor = m[row][col];
                for (int j = col; j < 6; j++) {
                    m[row][j] = m[row][j] - factor * m[col][j];
                }
            }
        }
    }

    // Result is in the augmented column
    return Kb5(m[0][5], m[1][5], m[2][5], m[3][5], m[4][5]);
}

static_assert(sizeof(Kb5) == 20, "Kb5 must be 20 bytes");
