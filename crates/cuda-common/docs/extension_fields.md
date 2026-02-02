# Extension Fields for CUDA: Design and Implementation

This document describes the mathematical foundations and implementation details of finite field extensions used in our CUDA-accelerated proof system.

## Table of Contents

1. [Base Field: Baby Bear](#base-field-baby-bear)
2. [Extension Field Theory](#extension-field-theory)
3. [Fp5 Implementation](#fp5-implementation)
4. [Benchmarking System](#benchmarking-system)
5. [Performance Analysis](#performance-analysis)

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
│  ext_field_bench.cu: launch_bench_{init,add,mul,inv}_*      │
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
| Fp | 4 B | 310 Gops/s | 3547 Gops/s | 1824 Gops/s | 59.7 Gops/s |
| FpExt (Fp4) | 16 B | 53 Gops/s | 1480 Gops/s | 203 Gops/s | 41.6 Gops/s |
| Fp5 | 20 B | 43 Gops/s | 960 Gops/s | 77 Gops/s | 3.8 Gops/s |

### Relative Performance (vs Fp baseline)

| Field | init | add | mul | inv |
|-------|------|-----|-----|-----|
| Fp | 1.0× | 1.0× | 1.0× | 1.0× |
| FpExt | 5.8× slower | 2.4× slower | 9.0× slower | 1.4× slower |
| Fp5 | 7.2× slower | 3.7× slower | 23.6× slower | 15.6× slower |

### Analysis

**Addition**: Near-linear scaling with degree (expected: 4× and 5× for Fp4/Fp5).

**Multiplication**: Super-linear scaling due to:
- Schoolbook: O(n²) multiplications
- FpExt uses optimized `bb31_4_t` with possible SIMD
- Fp5 uses manual schoolbook

**Inversion**:
- Fp uses Fermat with optimized addition chain (~31 squares + 7 muls)
- FpExt uses norm-based inversion (reduce to Fp inversion)
- Fp5 uses Gaussian elimination (much faster than naive Fermat)

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

- [ ] Fp6 implementation (consider 2×3 vs 3×2 tower construction)
- [ ] Lazy reduction for Fp5 multiplication
- [ ] Batch inversion using Montgomery's trick
- [ ] Karatsuba evaluation for larger extensions
