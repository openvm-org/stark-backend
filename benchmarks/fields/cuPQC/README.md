# cuPQC Poseidon2 Benchmark

Compares NVIDIA's [cuPQC 0.4.1](https://docs.nvidia.com/cuda/cupqc/) Poseidon2 implementation
against our custom CUDA Poseidon2 for BabyBear and KoalaBear fields — on the same GPU in one binary.

## Results (RTX 5090, sm_120, CUDA 13.1)

```
=== Poseidon2 Benchmark: Our CUDA vs NVIDIA cuPQC 0.4.1 ===

States/run  : 4194304 (2^22)
Reps/state  : 100
Warmup iters: 3
Bench iters : 10

### BabyBear Poseidon2 (p = 2^31 - 2^27 + 1)

| Implementation           |  Time (ms) |       Gops/s |
|--------------------------|------------|--------------|
| Our BB Poseidon2         |     92.741 |          4.5 |
| cuPQC BB Poseidon2       |    147.840 |          2.8 |

  cuPQC / Our speedup: 0.63x  →  Our impl is 1.58x faster

### KoalaBear Poseidon2 (p = 2^31 - 2^24 + 1)

| Implementation           |  Time (ms) |       Gops/s |
|--------------------------|------------|--------------|
| Our KB Poseidon2         |     83.590 |          5.0 |
| cuPQC KB Poseidon2       |    119.061 |          3.5 |

  cuPQC / Our speedup: 0.70x  →  Our impl is 1.43x faster
```

## What's being measured

| | Our implementation | cuPQC |
|---|---|---|
| **API** | Raw `poseidon2_mix()` on a 16-element state | Sponge: `reset + update(8 u32) + finalize + digest(8 u32)` |
| **Width** | 16 field elements | 16 (Width=16) |
| **Permutations/call** | 1 | **exactly 1** — see below |
| **s-box (BB)** | x^7 (4 muls), 13 partial rounds | x^7, 13 partial rounds |
| **s-box (KB)** | x^3 (2 muls), 20 partial rounds | x^3, 20 partial rounds |

### Is the comparison fair?

Yes. We verified that `reset/finalize/digest` overhead is **negligible** — measuring
at reps=1 through reps=100 all give the same Gops/s:

```
reps=  1: 2.87 Gops/s
reps=  5: 2.90 Gops/s
reps= 10: 2.87 Gops/s
reps=100: 2.88 Gops/s
```

When `update(rate=8)` is called, the rate buffer fills and the permutation fires immediately.
`finalize()` is then a no-op (the buffer was just flushed to empty). Each call-cycle does
**exactly 1 permutation**, same as our raw `poseidon2_mix()`.

### Can we call just the permutation via cuPQC?

**No.** cuPQC exposes no raw permutation API — `Poseidon2Context` only has
`reset / update / finalize / digest`. The permutation is internal to `update()`.
Since the sponge overhead is already negligible, the current approach *is* benchmarking
the pure permutation through the only available interface.

## Structure

```
cuPQC/
├── bench.cu                        # Combined benchmark (our + cuPQC side-by-side)
├── Makefile                        # Auto-downloads SDK if absent; builds bench
├── README.md                       # This file
└── cupqc-sdk-0.4.1-x86_64/        # cuPQC SDK (auto-downloaded, gitignored)
    ├── include/cupqc/hash.hpp
    ├── include/commondx/
    └── lib/libcupqc-hash.a
```

## Usage

```bash
# Build (auto-downloads cuPQC SDK ~4 MB on first run)
make

# Run
make run
# or
./bench
```

### Requirements

- CUDA Toolkit ≥ 12.8 (`nvcc`)
- GPU with sm_70 or newer (cuPQC supports sm_70–sm_90; sm_120 works via PTX JIT)
- `wget` (for auto-download of SDK on first build)

### Clean

```bash
make clean        # remove binary only
make distclean    # remove binary + downloaded SDK
```

## Notes

- cuPQC SDK is **not** committed to git — it's downloaded on first `make`.
- cuPQC 0.4.1 officially supports sm_70–sm_90. On sm_120 (RTX 5090) it works
  because the library is compiled to PTX which JIT-compiles for newer architectures.
- Both benchmarks use 4M states × 100 reps × 10 iterations = 4 billion operations
  to amortize launch overhead.
