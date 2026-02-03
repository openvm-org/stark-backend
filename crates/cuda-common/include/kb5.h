/*
 * Kb5 - Quintic Extension of KoalaBear Field
 * 
 * Defines Kb5, a finite field F_p^5, based on Kb via the irreducible polynomial x^5 + x + 4.
 * 
 * NOTE: Unlike BabyBear, KoalaBear has gcd(5, p-1) = 1, so x^5 - W is NEVER irreducible
 * (every element is a 5th power). We use the sparse trinomial x^5 + x + 4 instead.
 * 
 * Field size: p^5 ≈ 2^155, provides ~155 bits of security.
 * 
 * Element representation: a0 + a1*α + a2*α^2 + a3*α^3 + a4*α^4
 * where each ai is a KoalaBear element and α^5 = -α - 4.
 * 
 * Reduction rule: α^5 = -α - 4
 * For product giving c0..c8:
 *   c0_new = c0 - 4*c5
 *   c1_new = c1 - c5 - 4*c6
 *   c2_new = c2 - c6 - 4*c7
 *   c3_new = c3 - c7 - 4*c8
 *   c4_new = c4 - c8
 * 
 * Multiplication matrix for inversion (columns = coefficients of a*α^j):
 * [a0        -4*a4       -4*a3       -4*a2       -4*a1      ]
 * [a1        a0-a4       -a3-4*a4    -a2-4*a3    -a1-4*a2   ]
 * [a2        a1          a0-a4       -a3-4*a4    -a2-4*a3   ]
 * [a3        a2          a1          a0-a4       -a3-4*a4   ]
 * [a4        a3          a2          a1          a0-a4      ]
 * 
 * Inversion via Gaussian elimination: O(125) Kb operations.
 */

#pragma once

#include "kb.h"

/// Kb5 is a degree-5 extension of the KoalaBear field.
/// Elements are represented as polynomials in F_p[x] / (x^5 + x + 4).
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
    /// Uses schoolbook multiplication + reduction mod (x^5 + x + 4)
    /// 
    /// Reduction: α^5 = -α - 4
    /// For degree-8 product t0 + t1*α + ... + t8*α^8:
    ///   c0 = t0 - 4*t5
    ///   c1 = t1 - t5 - 4*t6
    ///   c2 = t2 - t6 - 4*t7
    ///   c3 = t3 - t7 - 4*t8
    ///   c4 = t4 - t8
    __device__ Kb5 operator*(Kb5 rhs) const {
        const Kb a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        const Kb b0 = rhs.elems[0], b1 = rhs.elems[1], b2 = rhs.elems[2], b3 = rhs.elems[3], b4 = rhs.elems[4];
        
        const Kb four(4);
        
        // Compute convolution products t[k] = sum_{i+j=k} a[i]*b[j]
        Kb t0 = a0 * b0;
        Kb t1 = a0 * b1 + a1 * b0;
        Kb t2 = a0 * b2 + a1 * b1 + a2 * b0;
        Kb t3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0;
        Kb t4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;
        Kb t5 = a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1;
        Kb t6 = a2 * b4 + a3 * b3 + a4 * b2;
        Kb t7 = a3 * b4 + a4 * b3;
        Kb t8 = a4 * b4;
        
        // Reduce: α^5 = -α - 4
        // c0 = t0 - 4*t5
        // c1 = t1 - t5 - 4*t6
        // c2 = t2 - t6 - 4*t7
        // c3 = t3 - t7 - 4*t8
        // c4 = t4 - t8
        Kb c0 = t0 - four * t5;
        Kb c1 = t1 - t5 - four * t6;
        Kb c2 = t2 - t6 - four * t7;
        Kb c3 = t3 - t7 - four * t8;
        Kb c4 = t4 - t8;
        
        return Kb5(c0, c1, c2, c3, c4);
    }
    
    __device__ Kb5& operator*=(Kb5 rhs) {
        *this = *this * rhs;
        return *this;
    }
    
    /// Squaring (optimized using symmetry)
    __device__ Kb5 square() const {
        const Kb a0 = elems[0], a1 = elems[1], a2 = elems[2], a3 = elems[3], a4 = elems[4];
        
        const Kb four(4);
        
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
        
        // Reduce
        Kb c0 = t0 - four * t5;
        Kb c1 = t1 - t5 - four * t6;
        Kb c2 = t2 - t6 - four * t7;
        Kb c3 = t3 - t7 - four * t8;
        Kb c4 = t4 - t8;
        
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
/// For trinomial x^5 + x + 4, the multiplication matrix for a = a0 + a1*α + ... + a4*α^4
/// where α^5 = -α - 4 is (columns are coefficients of a*α^j):
/// 
/// [a0        -4*a4       -4*a3       -4*a2       -4*a1      ]  // row 0
/// [a1        a0-a4       -a3-4*a4    -a2-4*a3    -a1-4*a2   ]  // row 1
/// [a2        a1          a0-a4       -a3-4*a4    -a2-4*a3   ]  // row 2
/// [a3        a2          a1          a0-a4       -a3-4*a4   ]  // row 3
/// [a4        a3          a2          a1          a0-a4      ]  // row 4
__device__ inline Kb5 inv(Kb5 x) {
    // Handle zero case
    if (x == Kb5::zero()) {
        return Kb5::zero();
    }
    
    const Kb* a = x.elems;
    const Kb neg_four = Kb(0) - Kb(4);
    
    // Build augmented matrix [M | I] for Gauss-Jordan elimination
    // M is the multiplication matrix, I is identity
    // We'll solve M * result = e0 = (1, 0, 0, 0, 0)
    
    // Matrix storage: m[row][col], augmented with result column
    Kb m[5][6];
    
    // Precompute some terms
    Kb a0_m_a4 = a[0] - a[4];
    Kb n4a4_m_a3 = neg_four * a[4] - a[3];  // -4*a4 - a3
    Kb n4a3_m_a2 = neg_four * a[3] - a[2];  // -4*a3 - a2
    Kb n4a2_m_a1 = neg_four * a[2] - a[1];  // -4*a2 - a1
    Kb n4a1 = neg_four * a[1];              // -4*a1
    
    // Row 0: [a0, -4*a4, -4*a3, -4*a2, -4*a1 | 1]
    m[0][0] = a[0]; 
    m[0][1] = neg_four * a[4]; 
    m[0][2] = neg_four * a[3]; 
    m[0][3] = neg_four * a[2]; 
    m[0][4] = n4a1; 
    m[0][5] = Kb::one();
    
    // Row 1: [a1, a0-a4, -a3-4*a4, -a2-4*a3, -a1-4*a2 | 0]
    m[1][0] = a[1]; 
    m[1][1] = a0_m_a4; 
    m[1][2] = n4a4_m_a3;
    m[1][3] = n4a3_m_a2; 
    m[1][4] = n4a2_m_a1; 
    m[1][5] = Kb::zero();
    
    // Row 2: [a2, a1, a0-a4, -a3-4*a4, -a2-4*a3 | 0]
    m[2][0] = a[2]; 
    m[2][1] = a[1]; 
    m[2][2] = a0_m_a4;
    m[2][3] = n4a4_m_a3; 
    m[2][4] = n4a3_m_a2; 
    m[2][5] = Kb::zero();
    
    // Row 3: [a3, a2, a1, a0-a4, -a3-4*a4 | 0]
    m[3][0] = a[3]; 
    m[3][1] = a[2]; 
    m[3][2] = a[1];
    m[3][3] = a0_m_a4; 
    m[3][4] = n4a4_m_a3; 
    m[3][5] = Kb::zero();
    
    // Row 4: [a4, a3, a2, a1, a0-a4 | 0]
    m[4][0] = a[4]; 
    m[4][1] = a[3]; 
    m[4][2] = a[2];
    m[4][3] = a[1]; 
    m[4][4] = a0_m_a4; 
    m[4][5] = Kb::zero();
    
    // Gaussian elimination with partial pivoting
    for (int col = 0; col < 5; col++) {
        // Find pivot - first non-zero element in column (from row col onwards)
        int pivot_row = col;
        while (pivot_row < 5 && m[pivot_row][col] == Kb::zero()) {
            pivot_row++;
        }
        
        // If no non-zero pivot found, try to find any non-zero
        if (pivot_row >= 5) {
            // Matrix is singular (shouldn't happen for valid extension field elements)
            return Kb5::zero();
        }
        
        // Among non-zero elements, pick one with largest raw value for stability
        for (int row = pivot_row + 1; row < 5; row++) {
            if (m[row][col] != Kb::zero() && m[row][col].asRaw() > m[pivot_row][col].asRaw()) {
                pivot_row = row;
            }
        }
        
        // Swap rows if needed
        if (pivot_row != col) {
            for (int j = 0; j < 6; j++) {
                Kb tmp = m[col][j];
                m[col][j] = m[pivot_row][j];
                m[pivot_row][j] = tmp;
            }
        }
        
        // Scale pivot row to make pivot = 1
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
