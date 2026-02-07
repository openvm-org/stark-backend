/*
 * Goldilocks cubic extension (Gl3) 
 * 
 * Based on Polygon's zisk implementation:
 * https://github.com/0xPolygon/goldilocks/blob/master/src/goldilocks_cubic_extension.cuh
 *
 * Prime: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
 * Irreducible polynomial: X^3 - X - 1
 * 
 * Reduction rule: α³ = α + 1
 * 
 * Element representation: a₀ + a₁α + a₂α²
 */

#pragma once

#include "gl.h"

/// Goldilocks cubic extension field element
class Gl3 {
public:
    Gl c[3];  // c[0] + c[1]*α + c[2]*α²
    
    __device__ Gl3() {}
    __device__ Gl3(Gl c0, Gl c1, Gl c2) { c[0] = c0; c[1] = c1; c[2] = c2; }
    __device__ explicit Gl3(Gl c0) { c[0] = c0; c[1] = Gl::zero(); c[2] = Gl::zero(); }
    
    __device__ static Gl3 zero() { return Gl3(Gl::zero(), Gl::zero(), Gl::zero()); }
    __device__ static Gl3 one() { return Gl3(Gl::one(), Gl::zero(), Gl::zero()); }
    
    __device__ Gl3 operator+(Gl3 rhs) const {
        return Gl3(c[0] + rhs.c[0], c[1] + rhs.c[1], c[2] + rhs.c[2]);
    }
    
    __device__ Gl3 operator-(Gl3 rhs) const {
        return Gl3(c[0] - rhs.c[0], c[1] - rhs.c[1], c[2] - rhs.c[2]);
    }
    
    __device__ Gl3 operator-() const {
        return Gl3(-c[0], -c[1], -c[2]);
    }
    
    __device__ Gl3& operator+=(Gl3 rhs) { *this = *this + rhs; return *this; }
    __device__ Gl3& operator-=(Gl3 rhs) { *this = *this - rhs; return *this; }
    
    /*
     * Multiplication using Karatsuba-style optimization
     * From zisk's Goldilocks3GPU::mul()
     * 
     * For X³ - X - 1, we have α³ = α + 1
     * 
     * Let a = a₀ + a₁α + a₂α², b = b₀ + b₁α + b₂α²
     * 
     * Uses 6 base field multiplications instead of 9:
     *   A = (a₀ + a₁)(b₀ + b₁)
     *   B = (a₀ + a₂)(b₀ + b₂)  
     *   C = (a₁ + a₂)(b₁ + b₂)
     *   D = a₀ * b₀
     *   E = a₁ * b₁
     *   F = a₂ * b₂
     *   G = D - E
     * 
     * Result:
     *   c₀ = C + G - F = C + D - E - F
     *   c₁ = A + C - 2E - D = A + C - E - E - D
     *   c₂ = B - G = B - D + E
     */
    __device__ Gl3 operator*(Gl3 rhs) const {
        Gl A = (c[0] + c[1]) * (rhs.c[0] + rhs.c[1]);
        Gl B = (c[0] + c[2]) * (rhs.c[0] + rhs.c[2]);
        Gl C = (c[1] + c[2]) * (rhs.c[1] + rhs.c[2]);
        Gl D = c[0] * rhs.c[0];
        Gl E = c[1] * rhs.c[1];
        Gl F = c[2] * rhs.c[2];
        Gl G = D - E;
        
        Gl3 result;
        result.c[0] = (C + G) - F;
        result.c[1] = ((((A + C) - E) - E) - D);
        result.c[2] = B - G;
        return result;
    }
    
    __device__ Gl3& operator*=(Gl3 rhs) { *this = *this * rhs; return *this; }
    
    __device__ bool operator==(Gl3 rhs) const {
        return c[0] == rhs.c[0] && c[1] == rhs.c[1] && c[2] == rhs.c[2];
    }
    __device__ bool operator!=(Gl3 rhs) const { return !(*this == rhs); }
};

/*
 * Inversion using direct formula from zisk's Goldilocks3GPU::inv()
 * 
 * For element a = a₀ + a₁α + a₂α², compute 1/a
 * 
 * Uses the norm formula for cubic extensions.
 * Computes various products and then finds the inverse of the norm (scalar).
 */
__device__ inline Gl3 inv(Gl3 x) {
    Gl a0 = x.c[0], a1 = x.c[1], a2 = x.c[2];
    
    // Compute products
    Gl aa = a0 * a0;
    Gl ac = a0 * a2;
    Gl ba = a1 * a0;
    Gl bb = a1 * a1;
    Gl bc = a1 * a2;
    Gl cc = a2 * a2;
    
    // Compute cubic products
    Gl aaa = aa * a0;
    Gl aac = aa * a2;
    Gl abc = ba * a2;
    Gl abb = ba * a1;
    Gl acc = ac * a2;
    Gl bbb = bb * a1;
    Gl bcc = bc * a2;
    Gl ccc = cc * a2;
    
    // Compute norm (scalar)
    // t = 3*abc + abb - aaa - 2*aac - acc - bbb + bcc - ccc
    Gl t = abc + abc + abc + abb - aaa - aac - aac - acc - bbb + bcc - ccc;
    
    // Invert the norm
    Gl tinv = inv(t);
    
    // Compute result coefficients
    // i1 = (bc + bb - aa - 2*ac - cc) / t
    // i2 = (ba - cc) / t  
    // i3 = (ac + cc - bb) / t
    Gl i1 = (bc + bb - aa - ac - ac - cc) * tinv;
    Gl i2 = (ba - cc) * tinv;
    Gl i3 = (ac + cc - bb) * tinv;
    
    return Gl3(i1, i2, i3);
}

static_assert(sizeof(Gl3) == 24, "Gl3 must be 24 bytes (3 x 8 bytes)");
