# Extension Fields for CUDA: Design and Implementation

This document describes the mathematical foundations and implementation details of finite field extensions used in our CUDA-accelerated proof system.

## Table of Contents

1. [Base Field: Baby Bear](#base-field-baby-bear)
2. [Extension Field Theory](#extension-field-theory)
3. [Fp5 Implementation](#fp5-implementation)
4. [Fp6 Implementation](#fp6-implementation)
5. [Fp6 Tower Constructions](#fp6-tower-constructions)
6. [Benchmarking System](#benchmarking-system)
7. [Performance Analysis](#performance-analysis)

---

## Base Field: Baby Bear

### Definition

Baby Bear is a prime field with:

```
p = 2^31 - 2^27 + 1 = 2013265921 = 0x78000001
```

### Key Properties

| Property | Value | Significance |
|----------|-------|--------------|
| Size | 31 bits | Fits in `uint32_t` with room for lazy reduction |
| Two-adicity | 27 | `p - 1 = 15 × 2^27`, excellent for FFT |
| Form | `2^31 - 2^27 + 1` | Enables fast modular reduction |

### Montgomery Representation

All field elements are stored in Montgomery form for efficient multiplication:

```
Mont(x) = x * R mod p,  where R = 2^32
```

Multiplication: `Mont(a) * Mont(b) → Mont(a*b)` via Montgomery reduction.

---

## Extension Field Theory

### Binomial Extensions

An extension field `Fp[n]` is constructed as:

```
Fp[n] = Fp[x] / (x^n - W)
```

where `x^n - W` is an irreducible polynomial over `Fp`.

**Irreducibility condition**: For `x^n - W` to be irreducible, `W` must NOT be an n-th power in `Fp`.

**Test**: `W` is an n-th power iff `W^((p-1)/n) ≡ 1 (mod p)`

### Elements

An element `a ∈ Fp[n]` is a polynomial of degree < n:

```
a = a₀ + a₁x + a₂x² + ... + aₙ₋₁xⁿ⁻¹
```

Stored as array `[a₀, a₁, ..., aₙ₋₁]` where each `aᵢ ∈ Fp`.

### Arithmetic

**Addition**: Component-wise
```
(a + b)[i] = a[i] + b[i]
```

**Multiplication**: Polynomial multiplication followed by reduction using `xⁿ = W`:
```
x^n → W
x^(n+1) → W·x
x^(n+k) → W·x^k
```

**Inversion**: Multiple approaches (see Fp5 section).

---

## Fp5 Implementation

### Irreducible Polynomial Selection

For `Fp5 = Fp[x] / (x^5 - W)`, we need `W` that is NOT a 5th power in Baby Bear.

**Verification** (computed `W^((p-1)/5) mod p` where `(p-1)/5 = 402653184`):

| W | W^402653184 mod p | Is 5th power? | Status |
|---|-------------------|---------------|--------|
| 1 | 1 | Yes | ❌ Invalid |
| 2 | 815036133 | No | ✅ **Optimal** |
| 3 | 1956349769 | No | ✅ Valid |
| 4 | 609564788 | No | ✅ Valid |
| 5 | 1 | Yes | ❌ Invalid |

**Result**: `W = 2` is the smallest valid choice.

### Why W = 2 is Optimal

1. **Smallest valid non-residue** - minimizes constant storage
2. **Multiplication by 2 = addition** - `2·x = x + x` (no expensive multiply)
3. **Efficient reduction** - the reduction step becomes additions only

### Irreducible Polynomial

```
x^5 - 2
```

### Element Representation

```cpp
struct Fp5 {
    Fp elems[5];  // [a₀, a₁, a₂, a₃, a₄]
    static constexpr uint32_t W = 2;
};
```

Memory: 5 × 4 = 20 bytes per element.

### Multiplication Algorithm

**Schoolbook multiplication** with fused reduction:

Given `a, b ∈ Fp5`, compute `c = a · b`:

1. Compute product coefficients `t[0..8]` (degree 8 polynomial)
2. Reduce using `x^5 = 2`:
   - `c[0] = t[0] + 2·t[5]`
   - `c[1] = t[1] + 2·t[6]`
   - `c[2] = t[2] + 2·t[7]`
   - `c[3] = t[3] + 2·t[8]`
   - `c[4] = t[4]`

**Optimization**: Since W=2, use `t[i] + t[i]` instead of `Fp(2) * t[i]`.

**Cost**: 25 Fp multiplications + ~28 Fp additions

```cpp
// Optimized: compute each result directly, fusing reduction
Fp c0 = a0*b0;
Fp t5_half = a1*b4 + a2*b3 + a3*b2 + a4*b1;
c0 = c0 + t5_half + t5_half;  // +2·t5 via addition
// ... similar for c1, c2, c3, c4
```

### Squaring Algorithm

Uses symmetry: for `i ≠ j`, `aᵢaⱼ` appears twice in the product.

**Cost**: 15 Fp multiplications (vs 25 for general multiply)

### Inversion Algorithm

**Approach**: Gaussian elimination on the multiplication matrix.

For `a ∈ Fp5`, finding `a⁻¹` is equivalent to solving `M · b = e₀` where:
- `M` is the 5×5 "multiply by a" matrix
- `e₀ = (1, 0, 0, 0, 0)` is the multiplicative identity

The matrix `M` for `a = a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴`:

```
┌ a₀    W·a₄  W·a₃  W·a₂  W·a₁ ┐
│ a₁    a₀    W·a₄  W·a₃  W·a₂ │
│ a₂    a₁    a₀    W·a₄  W·a₃ │
│ a₃    a₂    a₁    a₀    W·a₄ │
└ a₄    a₃    a₂    a₁    a₀   ┘
```

**Algorithm**: Gauss-Jordan elimination with partial pivoting.

**Cost**: O(n³) = O(125) Fp operations + 5 Fp inversions

**Comparison with Fermat's method**:

| Method | Operations | Fp5 Benchmark |
|--------|------------|---------------|
| Fermat (a^(p⁵-2)) | ~155 Fp5 squares + ~77 Fp5 muls ≈ 4250 Fp ops | ~1040 ms |
| Gaussian elimination | ~125 Fp ops + 5 Fp inv | ~110 ms |
| **Speedup** | | **~9.5×** |

### Frobenius Endomorphism Note

For Baby Bear: `p ≡ 1 (mod 5)` (since 2013265921 mod 5 = 1).

This means the Frobenius map `φ(a) = a^p` is the **identity** on Fp5:
```
φ(a) = a
```

This simplifies some algorithms but means we can't use Frobenius for fast exponentiation shortcuts.

### Correctness Verification

The Fp5 implementation was verified using on-device tests with 1024 random elements:

**Test 1: Multiplicative Inverse**
```
∀a ≠ 0: a · inv(a) = 1
```
A CUDA kernel computed `a * inv(a)` for each element and compared against `Fp5::one()`.
Failures were counted via `atomicAdd`. Result: **0 failures**.

**Test 2: Distributivity**
```
∀a, b, c: (a + b) · c = a·c + b·c
```
A CUDA kernel computed both sides independently and compared.
Result: **0 failures**.

These tests validate:
- Addition correctness (used in distributivity LHS)
- Multiplication correctness (both tests)
- Inversion correctness (inverse test)
- The irreducible polynomial choice (incorrect W would cause failures)

---

## Fp6 Implementation

### Irreducible Polynomial Selection

For `Fp6 = Fp[x] / (x^6 - W)`, we need `W` that is:
- **NOT a quadratic residue** (not a square in Fp)
- **NOT a cubic residue** (not a cube in Fp)

This is stricter than Fp5, which only required "not a 5th power."

**Verification** (computed `W^((p-1)/2)` and `W^((p-1)/3)` mod p):

| W | Is square? | Is cube? | Valid for Fp6? |
|---|------------|----------|----------------|
| 2 | Yes | No | ❌ |
| 3 | Yes | No | ❌ |
| 7 | Yes | No | ❌ |
| 11 | No | Yes | ❌ |
| 22 | No | No | ✅ First valid |
| 31 | No | No | ✅ **Chosen** |

### Why W = 31 was Chosen

While W = 22 is the smallest valid choice, we chose **W = 31** for these reasons:

1. **Conventional constant**: 31 is the multiplicative generator of Baby Bear, widely used in the ecosystem
2. **Efficient reduction**: `31 = 2^5 - 1`, so `31·x = (x << 5) - x` (one subtraction vs three additions for W = 22)
3. **Memorable**: As a well-known constant, it's easier to recognize and debug

### Irreducible Polynomial

```
x^6 - 31
```

### Element Representation

```cpp
struct Fp6 {
    Fp elems[6];  // [a₀, a₁, a₂, a₃, a₄, a₅]
    static constexpr uint32_t W = 31;
};
```

Memory: 6 × 4 = 24 bytes per element.

### Multiplication Algorithm

**Schoolbook multiplication** with fused reduction:

Given `a, b ∈ Fp6`, compute `c = a · b`:

1. Compute product coefficients `t[0..10]` (degree 10 polynomial)
2. Reduce using `x^6 = 31`:
   - `c[0] = t[0] + 31·t[6]`
   - `c[1] = t[1] + 31·t[7]`
   - `c[2] = t[2] + 31·t[8]`
   - `c[3] = t[3] + 31·t[9]`
   - `c[4] = t[4] + 31·t[10]`
   - `c[5] = t[5]`

**Cost**: 36 Fp multiplications + ~35 Fp additions

### Squaring Algorithm

Uses symmetry: for `i ≠ j`, `aᵢaⱼ` appears twice in the product.

**Cost**: 21 Fp multiplications (vs 36 for general multiply)

### Inversion Algorithm

Same approach as Fp5: **Gaussian elimination** on the 6×6 multiplication matrix.

**Cost**: O(n³) = O(216) Fp operations + 6 Fp inversions

### Correctness Verification

The Fp6 implementation was verified using on-device tests with 1024 random elements:

**Test 1: Multiplicative Inverse**
```
∀a ≠ 0: a · inv(a) = 1
```
Result: **0 failures**.

**Test 2: Distributivity**
```
∀a, b, c: (a + b) · c = a·c + b·c
```
Result: **0 failures**.

---

## Fp6 Tower Constructions

Instead of building Fp6 directly as `Fp[x]/(x^6 - W)`, we can construct it as a **tower of extensions**. This can improve performance, especially for inversion, by leveraging more efficient algorithms at each level.

### Tower Options

| Tower | Structure | Base → Level 1 → Level 2 |
|-------|-----------|--------------------------|
| **2×3** | Fp → Fp2 → Fp6 | Quadratic then cubic extension |
| **3×2** | Fp → Fp3 → Fp6 | Cubic then quadratic extension |

Both produce isomorphic fields with `p^6` elements, but arithmetic efficiency differs.

### Choosing Tower Constants

For each tower level, we need an irreducible polynomial. The key insight is that constants must remain non-residues when lifted to the extension field.

#### Why Constants 11 and 2 Work for Both Towers

**Constant 11 (non-square in Fp)**:
- Test: `11^((p-1)/2) ≠ 1 (mod p)` ✓
- In Fp3: If 11 were a square in Fp3, then `11 = a²` for some `a ∈ Fp3`. Taking the norm: `N(11) = N(a²) = N(a)² = 11³`. But `11³ = 11 · 11²` implies 11 is a square in Fp (contradiction).
- Therefore 11 remains a non-square in both Fp and Fp3.

**Constant 2 (non-cube in Fp)**:
- Test: `2^((p-1)/3) ≠ 1 (mod p)` ✓
- In Fp2: If 2 were a cube in Fp2, say `2 = a³`, then `N(2) = N(a³) = N(a)³ = 4`. We need `4 = b³` for some `b ∈ Fp`. But `4^((p-1)/3) ≠ 1 (mod p)`, so 4 is not a cube.
- Therefore 2 remains a non-cube in both Fp and Fp2.

### 2×3 Tower (Fp2x3)

**Construction**:
```
Fp2 = Fp[u] / (u² - 11)      -- u² = 11
Fp6 = Fp2[v] / (v³ - 2)      -- v³ = 2
```

**Element Representation**:
```cpp
struct Fp2_11 {
    Fp c0, c1;  // c0 + c1·u where u² = 11
    static constexpr uint32_t W = 11;
};

struct Fp2x3 {
    Fp2_11 c0, c1, c2;  // c0 + c1·v + c2·v² where v³ = 2
    static constexpr uint32_t W = 2;  // for cubic extension
};
```

Memory: 6 × 4 = 24 bytes (same as direct Fp6).

**Arithmetic**:

*Fp2 Multiplication* (Karatsuba-style):
```
(a0 + a1·u)(b0 + b1·u) = (a0·b0 + 11·a1·b1) + (a0·b1 + a1·b0)·u
```
Cost: 3 Fp muls (using Karatsuba) or 4 Fp muls (schoolbook)

*Fp2x3 Multiplication*:
```
(a0 + a1·v + a2·v²)(b0 + b1·v + b2·v²)
```
Reduce using `v³ = 2`:
- `v³ → 2`
- `v⁴ → 2v`

Cost: 9 Fp2 muls + reductions

*Fp2x3 Inversion* (via Gaussian elimination on 3×3 matrix over Fp2):
```
Solve M·x = e₀ where M is the "multiply by a" matrix over Fp2
```
Cost: O(27) Fp2 ops + 3 Fp2 inversions

*Fp2 Inversion* (via norm):
```
(a + b·u)⁻¹ = (a - b·u) / (a² - 11·b²)
```
Cost: 2 Fp muls + 1 Fp square + 1 Fp sub + 1 Fp inv

### 3×2 Tower (Fp3x2)

**Construction**:
```
Fp3 = Fp[u] / (u³ - 2)       -- u³ = 2
Fp6 = Fp3[v] / (v² - 11)     -- v² = 11
```

**Element Representation**:
```cpp
struct Fp3 {
    Fp c0, c1, c2;  // c0 + c1·u + c2·u² where u³ = 2
    static constexpr uint32_t W = 2;
};

struct Fp3x2 {
    Fp3 c0, c1;  // c0 + c1·v where v² = 11
    static constexpr uint32_t W = 11;  // for quadratic extension
};
```

Memory: 6 × 4 = 24 bytes (same as direct Fp6).

**Arithmetic**:

*Fp3 Multiplication*:
```
(a0 + a1·u + a2·u²)(b0 + b1·u + b2·u²)
```
Reduce using `u³ = 2`:
- `u³ → 2`
- `u⁴ → 2u`
- `u⁵ → 2u²`

Cost: 9 Fp muls + reductions

*Fp3x2 Multiplication* (Karatsuba-style):
```
(a0 + a1·v)(b0 + b1·v) = (a0·b0 + 11·a1·b1) + (a0·b1 + a1·b0)·v
```
Cost: 3 Fp3 muls (Karatsuba) = 27 Fp muls

*Fp3x2 Inversion* (via norm to Fp3):
```
(a + b·v)⁻¹ = (a - b·v) / (a² - 11·b²)
```
Where `a² - 11·b²` is computed in Fp3, then inverted.

Cost: 2 Fp3 muls + 1 Fp3 square + 1 Fp3 sub + 1 Fp3 inv

*Fp3 Inversion* (Gaussian elimination on 3×3 matrix):
```
Solve M·x = e₀ where M is the "multiply by a" matrix over Fp
```
Cost: O(27) Fp ops + 3 Fp inversions

### Why 3×2 Tower is Faster for Inversion

The key difference is in how inversion cascades through the tower:

| Tower | Inversion Strategy | Fp Inversions Required |
|-------|-------------------|----------------------|
| Direct Fp6 | 6×6 Gaussian elimination | 6 |
| 2×3 (Fp2x3) | 3×3 over Fp2, then Fp2 inv via norm | 3 × (1 per Fp2 inv) = 3 |
| 3×2 (Fp3x2) | Norm to Fp3, then 3×3 Gaussian | 3 |

However, the **norm-based** approach in Fp3x2 requires fewer total operations:
- Fp3x2: Compute norm (2 Fp3 muls + 1 Fp3 sq), then invert in Fp3
- Fp2x3: Must do 3×3 Gaussian over Fp2, each Fp2 op costs 3-4 Fp ops

**Benchmark result**: Fp3x2 inversion is **~5.6× faster** than direct Fp6 and **~1.8× faster** than Fp2x3.

### Correctness Verification

Both tower implementations were verified using on-device tests with 1024 random elements:

**Fp2x3 (2×3 tower)**:
- Multiplicative inverse test: **0 failures**
- Distributivity test: **0 failures**

**Fp3x2 (3×2 tower)**:
- Multiplicative inverse test: **0 failures**
- Distributivity test: **0 failures**

---

## Benchmarking System

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Rust Test Framework                       │
│  benchmark.rs: FFI bindings, timing, result formatting       │
└──────────────────────────┬──────────────────────────────────┘
                           │ extern "C" calls
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 CUDA Kernel Launchers                        │
│  ext_field_bench.cu: {init,add,mul,inv}_{fp,fpext,fp5,...}  │
└──────────────────────────┬──────────────────────────────────┘
                           │ kernel launches
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Templated CUDA Kernels                         │
│  bench_init_kernel<T>, bench_add_kernel<T>, etc.            │
└─────────────────────────────────────────────────────────────┘
```

### Kernel Design

Each benchmark kernel:
1. Performs `reps` iterations of the operation per element
2. Uses input data to prevent compiler optimization
3. Writes result to prevent dead code elimination

```cpp
template <typename T>
__global__ void bench_mul_kernel(T* a, T* b, T* out, int n, int reps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = a[idx];
        T y = b[idx];
        for (int r = 0; r < reps; r++) {
            x = x * y;
        }
        out[idx] = x;
    }
}
```

### Measurement Methodology

1. **Warmup**: 3 iterations (excluded from timing)
2. **Timed iterations**: 10 iterations (averaged)
3. **Synchronization**: `cudaDeviceSynchronize()` after each kernel
4. **Timing**: Rust `Instant::now()` around kernel + sync

### Metrics

**Throughput (Gops/s)**: Giga-operations per second

```
Gops/s = (num_elements × ops_per_element × reps) / (time_ns)
```

Where:
- `num_elements`: Number of field elements (default: 2²² = 4M)
- `ops_per_element`: Operations per element (100 for compute-bound ops)
- `reps`: Repetitions inside kernel
- `time_ns`: Average time in nanoseconds

### Memory Layout

Three device buffers per benchmark:
1. `d_a`: First operand (initialized from random u32)
2. `d_b`: Second operand (initialized from random u32)
3. `d_out`: Result storage

**Optimization**: `d_a` buffer reused for in-place initialization.

### Launch Configuration

```cpp
constexpr int BENCH_BLOCK_SIZE = 256;

inline void get_launch_config(int n, int& grid_size, int& block_size) {
    block_size = BENCH_BLOCK_SIZE;
    grid_size = (n + block_size - 1) / block_size;
}
```

---

## Performance Analysis

### Benchmark Results (RTX series, 4M elements)

| Field | Size | init | add | mul | inv |
|-------|------|------|-----|-----|-----|
| Fp | 4 B | 308 Gops/s | 3532 Gops/s | 1869 Gops/s | 59.6 Gops/s |
| FpExt (Fp4) | 16 B | 53 Gops/s | 1339 Gops/s | 187 Gops/s | 41.6 Gops/s |
| Fp5 | 20 B | 43 Gops/s | 1201 Gops/s | 77 Gops/s | 3.8 Gops/s |
| Fp6 (direct) | 24 B | 36 Gops/s | 987 Gops/s | 49 Gops/s | 2.2 Gops/s |
| Fp2x3 (2×3 tower) | 24 B | 36 Gops/s | 986 Gops/s | 42 Gops/s | 7.2 Gops/s |
| Fp3x2 (3×2 tower) | 24 B | 36 Gops/s | 987 Gops/s | 49 Gops/s | 12.7 Gops/s |

### Relative Performance (vs Fp baseline)

| Field | init | add | mul | inv |
|-------|------|-----|-----|-----|
| Fp | 1.0× | 1.0× | 1.0× | 1.0× |
| FpExt | 5.8× slower | 2.6× slower | 10.0× slower | 1.4× slower |
| Fp5 | 7.2× slower | 2.9× slower | 24.3× slower | 15.7× slower |
| Fp6 (direct) | 8.6× slower | 3.6× slower | 38.3× slower | 26.6× slower |
| Fp2x3 | 8.6× slower | 3.6× slower | 44.0× slower | 8.2× slower |
| Fp3x2 | 8.7× slower | 3.6× slower | 38.1× slower | 4.7× slower |

### Fp6 Implementation Comparison

| Implementation | Multiplication | Inversion | Best For |
|---------------|----------------|-----------|----------|
| Direct Fp6 | 49 Gops/s | 2.2 Gops/s | General purpose |
| Fp2x3 (2×3 tower) | 42 Gops/s | 7.2 Gops/s | Inversion-heavy workloads |
| Fp3x2 (3×2 tower) | 49 Gops/s | **12.7 Gops/s** | **Best overall** |

**Key findings**:
- **Multiplication**: Direct Fp6 and Fp3x2 are equivalent (~49 Gops/s), Fp2x3 is ~15% slower
- **Inversion**: Fp3x2 is **5.8× faster** than direct, Fp2x3 is **3.3× faster** than direct
- **Recommendation**: Use **Fp3x2** for best overall performance

### Analysis

**Addition**: Near-linear scaling with degree (expected: 4×, 5×, 6× for Fp4/Fp5/Fp6).

**Multiplication**: Super-linear scaling due to:
- Schoolbook: O(n²) multiplications (16, 25, 36 for Fp4/Fp5/Fp6)
- FpExt uses optimized `bb31_4_t` with possible SIMD
- Fp5/Fp6 use manual schoolbook

**Inversion**:
- Fp uses Fermat with optimized addition chain (~31 squares + 7 muls)
- FpExt uses norm-based inversion (reduce to Fp inversion)
- Fp5/Fp6 use Gaussian elimination (much faster than naive Fermat)

### Optimization History

#### Fp Inversion
- **Before**: GCD-based (branch-heavy, warp divergence)
- **After**: Fermat-based (`bb31_t::reciprocal()`)
- **Speedup**: 3.16×

#### Fp5 Inversion
- **Before**: Naive Fermat (155-bit exponent)
- **After**: Gaussian elimination (5×5 matrix solve)
- **Speedup**: 9.5×

#### Fp5 Multiplication
- **Optimization 1**: Use `x + x` instead of `Fp(2) * x` for reduction
- **Optimization 2**: Reduce register pressure by fusing computations
- **Total improvement**: ~9%

---

## References

1. **Plonky3**: https://github.com/Plonky3/Plonky3
   - BabyBear implementation with degree-4 extension
   - Uses `x^4 - 11` as irreducible polynomial

2. **ICICLE**: https://github.com/ingonyama-zk/icicle
   - CUDA-accelerated cryptographic primitives
   - BabyBear support

3. **Montgomery Multiplication**: https://en.wikipedia.org/wiki/Montgomery_modular_multiplication

---

## Future Work

- [ ] Lazy reduction for Fp5/Fp6 multiplication
- [ ] Batch inversion using Montgomery's trick
- [ ] Karatsuba multiplication for Fp5/Fp6 (potential 20-30% speedup)
