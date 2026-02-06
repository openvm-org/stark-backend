# Field Benchmarks

CUDA-optimized finite field extensions for zero-knowledge proof systems.

## Usage

### Run Verification Tests

Run verification tests (10 individual field tests):
```bash
cargo test
```

### Run Benchmark

```bash
cargo run --bin field_bench
```


## Structure

```
benchmarks/fields/
├── src/
│   ├── lib.rs              # Benchmark infrastructure and FFI bindings
│   └── bin/
│       └── field_bench.rs  # Benchmark binary
├── tests/
│   └── verification.rs     # Individual verification tests (10 tests)
├── cuda/
│   ├── src/                # CUDA kernel implementations
│   │   ├── ext_field_bench.cu
│   │   └── verification.cu
│   └── include/            # Field extension headers (organized by type)
│       ├── baby_bear/      # Baby Bear extensions (fp4, fp5, fp6, fp2x3, fp3x2)
│       ├── koala_bear/     # KoalaBear extensions (kb, kb5, kb6, kb2x3, kb3x2)
│       ├── goldilocks/     # Goldilocks extensions (gl, gl3)
│       └── ff/             # Field primitives (imported from cuda-common)
├── build.rs                # CUDA build configuration
└── Cargo.toml
```

### Verification Tests

Each field has an individual test that verifies:
- **Inversion**: `a * inv(a) = 1` for 1M random elements
- **Distributivity**: `(a + b) * c = a*c + b*c` for 1M random elements

Tests run in parallel and report precise failure counts on error.

---

# Extension Fields for CUDA

## Table of Contents

1. [Base Fields](#base-fields)
2. [Extension Field Theory](#extension-field-theory)
3. [Baby Bear Extensions](#baby-bear-extensions)
4. [KoalaBear Extensions](#koalabear-extensions)
5. [Goldilocks Extensions](#goldilocks-extensions)
6. [Performance Summary](#performance-summary)
7. [Change History](#change-history)

---

## Base Fields

### Baby Bear (Fp)

```
p = 2^31 - 2^27 + 1 = 0x78000001
```

| Property | Value |
|----------|-------|
| Size | 31 bits |
| Two-adicity | 27 |
| Montgomery R | 2^32 |

### KoalaBear (Kb)

```
p = 2^31 - 2^24 + 1 = 0x7F000001
```

| Property | Value |
|----------|-------|
| Size | 31 bits |
| Two-adicity | 24 |
| gcd(3, p-1) | **1** (every element is a cube) |
| gcd(5, p-1) | **1** (every element is a 5th power) |

**Important**: Binomial extensions `X^n - W` don't work for n=3,5,6. Must use trinomials.

### Goldilocks (Gl)

```
p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
```

| Property | Value |
|----------|-------|
| Size | 64 bits |
| Two-adicity | 32 |

---

## Extension Field Theory

An extension `Fp[n] = Fp[x] / f(x)` where `f(x)` is irreducible of degree n.

**Elements**: Polynomials of degree < n stored as coefficient arrays.

**Multiplication**: Polynomial multiplication followed by reduction mod f(x).

**Inversion**: Gaussian elimination, norm-based, or Frobenius-based methods.

---

## Baby Bear Extensions

### Fp5 (Quintic Extension)

**Polynomial**: `X^5 - 2`

**Multiplication**: PTX assembly with BETA = (2 << 32) % MOD = 0x1FFFFFFC
- Fused accumulation in 64-bit registers
- Single Montgomery reduction per output coefficient

**Inversion**: Frobenius norm (Itoh-Tsujii chain + partial norm)
```cpp
// f12 = φ(x)·φ²(x), f34 = φ²(f12), conj = f12·f34
// 3 Frobenius + 2 Fp5 muls + partial norm + 1 Fp inv
```

**Performance**: 118.0 Gops/s mul, 23.8 Gops/s inv

### Fp6 (Sextic Extension)

**Polynomial**: `X^6 - 31`

**Multiplication**: PTX assembly with 2-product grouping
- Groups 2 products per 64-bit accumulator before reducing
- BETA = (31 << 32) % MOD = 0x0FFFFFBE

**Inversion**: Frobenius norm (Itoh-Tsujii chain, free φ³ + partial norm)
- φ³ just negates odd coefficients (ω³ = -1, since 31 is not a QR)
- 2 Frobenius + 3 Fp6 muls + partial norm + 1 Fp inv

**Performance**: 70.6 Gops/s mul, 14.2 Gops/s inv

### Fp2x3 (2×3 Tower)

**Construction**:
```
Fp2 = Fp[u] / (u² - 11)    -- PTX with BETA = 0x37FFFFE9
Fp6 = Fp2[v] / (v³ - 2)    -- Toom-2.5 (6 muls instead of 9)
```

**mulByW**: `2×x = x + x` (addition only)

**Inversion**: Adjugate formula (1 Fp2 inversion)

**Performance**: 73.0 Gops/s mul, 23.6 Gops/s inv

### Fp3x2 (3×2 Tower) — Best Baby Bear mul

**Construction**:
```
Fp3 = Fp[u] / (u³ - 2)     -- PTX with BETA = 0x1FFFFFFC
Fp6 = Fp3[v] / (v² - 11)   -- Karatsuba (3 muls instead of 4)
```

**Inversion**: Norm to Fp3, then Fp3 adjugate (1 Fp inversion total)

**Performance**: **83.9 Gops/s** mul, **25.9 Gops/s** inv

---

## KoalaBear Extensions

### Kb5 (Quintic Extension)

**Polynomial**: `X^5 + X^2 - 1` (Plonky3 trinomial)

**Reduction**: `α^5 = -α^2 + 1` (cleaner than `α^5 = -α - 4`)

**Multiplication**: Karatsuba 2+3 split (15 muls instead of 25)
```cpp
// Split: A = A0 + x²·A1, where A0 = (a₀,a₁), A1 = (a₂,a₃,a₄)
P0 = A0 × B0    // 3 muls (Karatsuba)
P2 = A1 × B1    // 6 muls (Toom-2.5)
P1 = (A0+A1) × (B0+B1) - P0 - P2  // 6 muls
```

**Inversion**: Frobenius norm (Itoh-Tsujii, 1 Kb inversion, branchless)

**Performance**: 88.7 Gops/s mul, 11.9 Gops/s inv

### Kb6 (Sextic Extension)

**Polynomial**: `X^6 + X^3 + 1` (trinomial)

**Reduction**: `α^6 = -α^3 - 1`, `α^9 = 1` (additions only!)

**Multiplication**: Karatsuba 3+3 split (18 muls instead of 36)
```cpp
// Split: A = A0 + x³·A1, perfectly aligned with polynomial
P0 = A0 × B0    // 6 muls (Toom-2.5)
P1 = A1 × B1    // 6 muls
P2 = (A0+A1) × (B0+B1)  // 6 muls
```

**Inversion**: Frobenius norm (Itoh-Tsujii, 1 Kb inversion, branchless)
- Φ₉(x) cyclotomic polynomial: all Frobenius maps cost zero multiplications!

**Performance**: 68.7 Gops/s mul, 12.8 Gops/s inv

### Kb2x3 (2×3 Tower)

**Construction**:
```
Kb2 = Kb[u] / (u² - 3)       -- PTX with BETA = 0x05FFFFFA
Kb6 = Kb2[v] / (v³ - (1+u))  -- Toom-2.5 (6 muls)
```

**mulByW**: `(a₀ + a₁u)(1+u) = (a₀ + 3a₁) + (a₀ + a₁)u` — **3 additions only!**

**Inversion**: Adjugate formula (1 Kb2 inversion)

**Performance**: **71.0 Gops/s** mul, **21.0 Gops/s** inv

### Kb3x2 (3×2 Tower)

**Construction**:
```
Kb3 = Kb[w] / (w³ + w + 4)   -- Toom-2.5 (6 muls)
Kb6 = Kb3[z] / (z² - 3)      -- Karatsuba (3 muls)
```

**Inversion**: Norm to Kb3, then Kb3 adjugate (1 Kb inversion total)
```cpp
(a + bz)⁻¹ = (a - bz) / (a² - 3b²)   // norm to Kb3
// Kb3 adjugate: 9 Kb muls + 1 Kb inv (branchless)
```

**Performance**: 59.9 Gops/s mul, 20.6 Gops/s inv

---

## Goldilocks Extensions

### Gl3 (Cubic Extension)

**Polynomial**: `X³ - X - 1`

**Reduction**: `α³ = α + 1`

**Multiplication**: Karatsuba-style (6 muls instead of 9)

**Inversion**: Direct norm formula (1 Gl inversion)

**Performance**: 58.5 Gops/s mul, 7.8 Gops/s inv

---

## Performance Summary

### Baby Bear Fields

| Field | mul | inv | Notes |
|-------|-----|-----|-------|
| Fp (base) | 1871 | 59.6 | |
| Fp5 | 118.0 | 23.8 | PTX + Itoh-Tsujii |
| Fp6 | 70.6 | 14.2 | PTX + Itoh-Tsujii (free φ³) |
| Fp2x3 | 73.0 | 23.6 | Toom-2.5 + adjugate inv |
| **Fp3x2** | **83.9** | **25.9** | **PTX + Karatsuba + adjugate inv** |

### KoalaBear Fields

| Field | mul | inv | Notes |
|-------|-----|-----|-------|
| Kb (base) | 1869 | 45.5 | |
| Kb5 | 89.0 | 11.9 | Karatsuba 2+3 + Frobenius inv |
| Kb6 | 68.9 | 12.8 | Karatsuba 3+3 + Frobenius inv |
| **Kb2x3** | **71.8** | **21.1** | **PTX + Toom-2.5 + adjugate inv** |
| Kb3x2 | 59.9 | 20.6 | Toom-2.5 + adjugate inv |

### Goldilocks Fields

| Field | mul | inv | Notes |
|-------|-----|-----|-------|
| Gl (base) | 669.4 | 10.5 | |
| Gl3 | 58.6 | 7.8 | Karatsuba |

### Recommendations

| Use Case | Baby Bear | KoalaBear |
|----------|-----------|-----------|
| **Multiplication-heavy** | Fp3x2 (83.9 Gops/s) | Kb2x3 (71.8 Gops/s) |
| **Inversion-heavy** | Fp3x2 (25.9 Gops/s) | Kb2x3 (21.1 Gops/s) |
| **Simple code** | Direct Fp6 | Direct Kb6 |

---

## Change History

| Field | Change | Impact |
|-------|--------|--------|
| Fp inversion | GCD → Fermat addition chain | +3.2× |
| Fp5 mul | Schoolbook → PTX assembly | +1.8× |
| Fp5 inv | Gaussian → Frobenius | +5.4× |
| Fp6 mul | Schoolbook → PTX 2-product grouping | +2.0× |
| Fp2/Fp3 | Schoolbook → PTX with BETA | +1.3-1.4× |
| Fp2x3/Fp3x2 | Schoolbook → Toom-2.5/Karatsuba | +18-24% |
| Fp2x3 inv | Gaussian → adjugate | +2.2× |
| Kb5 mul | Schoolbook → Karatsuba 2+3 split | +20% |
| Kb5 poly | `x^5 + x + 4` → `x^5 + x^2 - 1` (Plonky3) | 7.2% faster! |
| Kb6 mul | Schoolbook → Karatsuba 3+3 split | +29% |
| Kb2 | Schoolbook → PTX with BETA | +1.2× |
| Kb2x3 | Added mulByW optimization | 3 adds vs 4 muls |
| Kb2x3 inv | Gaussian → adjugate | +6.4× |
| Kb3 | Schoolbook → Toom-2.5 | +33% |
| Kb5 inv | Gaussian → Frobenius norm | +3.2× |
| Kb6 inv | Gaussian → Frobenius norm (zero-mul Frobenius!) | +5.8× |
| Kb3x2 | Added Karatsuba + norm inversion | +31% mul, 4.6× inv |
| Fp5 inv | Naive Frobenius → Itoh-Tsujii chain | +1.22× |
| Fp6 inv | Naive Frobenius → Itoh-Tsujii (free φ³) | +1.22× |
| Fp3 inv | Gaussian elimination → adjugate formula | +1.95× (Fp3x2: 13.3→25.9) |
| Kb3 inv | Gaussian elimination → adjugate formula | +1.98× (Kb3x2: 10.4→20.6) |
