# Metal Backend Implementation Handoff

**Date**: 2026-02-16
**Status**: Compiles with zero errors. Proof generation stubs (`todo!()`) remain.

---

## Current State Summary

Both new crates compile successfully:
- `openvm-metal-common`: **489 lines** - fully complete, tests pass
- `openvm-metal-backend`: **6,105 lines Rust** + **6,125 lines Metal shaders** - compiles, but proof algorithms are stubbed

All 19 Metal shaders compile and link into `kernels.metallib` via `build.rs`.

---

## Phase Completion Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Scaffolding | DONE | Workspace, directories, Cargo.toml files |
| Phase 1: metal-common | DONE | All 6 files complete, unit tests pass |
| Phase 2A: Core shaders | DONE | baby_bear.h, baby_bear_ext.h, utils.h, device_ntt.h, ntt.metal, matrix.metal, poly.metal, mle_interpolate.metal, batch_ntt_small.metal |
| Phase 2B: Protocol shaders | DONE | poseidon2.h, codec.h, sumcheck.h, monomial.h, merkle_tree.metal, sumcheck.metal, whir.metal, stacked_reduction.metal, sponge.metal, prefix.metal, 8 logup_zerocheck/*.metal files |
| Phase 3: Build system | DONE | build.rs compiles .metal -> .air -> .metallib, embeds via include_bytes! |
| Phase 4: FFI dispatch | DONE | src/metal/*.rs - all dispatch wrappers present |
| Phase 5A: Core backend | DONE | MetalBackend, MetalDevice, engine, data_transporter, base, pkey, monomial, sponge, poly, ntt, merkle_tree, stacked_pcs |
| Phase 5B: Protocol impl | PARTIAL | Three main proof functions are `todo!()`, logup_zerocheck submodules are stubs, sumcheck.rs is a stub |
| Phase 6: Tests | NOT STARTED | tests.rs is empty |

---

## What Is Complete

### metal-common (100%)
All files fully implemented:
- `d_buffer.rs` (245 lines) - MetalBuffer<T> with StorageModeShared
- `copy.rs` (127 lines) - H2D/D2H via unified memory, D2D via blit encoder
- `device.rs` (44 lines) - MetalContext singleton
- `command.rs` (37 lines) - Command buffer helpers
- `error.rs` (31 lines) - Error types

### metal-backend high-level modules (complete)
These files are fully ported from cuda-backend and functional:

| File | Lines | CUDA Equivalent |
|------|-------|-----------------|
| `lib.rs` | 34 | Module exports, prelude |
| `types.rs` | 11 | F, EF, SC type aliases |
| `error.rs` | 134 | ProverError, WhirProverError, KernelError |
| `base.rs` | 202 | MetalMatrix, MetalMatrixView, Basis types |
| `device.rs` | 61 | MetalDevice, MetalProverConfig |
| `engine.rs` | 52 | BabyBearPoseidon2MetalEngine |
| `metal_backend.rs` | 117 | ProverBackend, TraceCommitter, MultiRapProver, OpeningProver impls |
| `data_transporter.rs` | 327 | DeviceDataTransporter trait impl |
| `pkey.rs` | 323 | AirDataMetal, EvalRules, monomial structures |
| `ntt.rs` | 130 | NTT orchestration (forward/inverse) |
| `poly.rs` | 437 | PleMatrix, EqEvalSegments, EqEvalLayers, SqrtEqLayers |
| `monomial.rs` | 453 | Monomial expansion structures |
| `sponge.rs` | 252 | DuplexSpongeMetal (FiatShamirTranscript) |
| `merkle_tree.rs` | 306 | MerkleTreeMetal |
| `stacked_pcs.rs` | 248 | StackedPcsDataMetal, stacked_commit |
| `utils.rs` | 55 | Utility functions |

### metal-backend FFI dispatch (complete)
All kernel dispatch wrappers in `src/metal/`:

| File | Lines | Kernels Dispatched |
|------|-------|--------------------|
| `mod.rs` | 395 | Kernel cache, dispatch helpers, sumcheck/prefix submodules |
| `logup_zerocheck.rs` | 644 | GKR, frac_compute_round, fold, interpolate, monomial precompute |
| `stacked_reduction.rs` | 179 | Stacked reduction round0, fold_ple, mle_round |
| `poly.rs` | 170 | eq_hypercube, algebraic_batch, vector_scalar_multiply |
| `matrix.rs` | 180 | Transpose, get_rows, split, batch_expand_pad, collapse |
| `ntt.rs` | 130 | Twiddle generation, bit_reverse, NTT steps |
| `mle_interpolate.rs` | 129 | MLE interpolation stages |
| `merkle_tree.rs` | 115 | Poseidon2 hashing, Merkle layer compression |
| `whir.rs` | 115 | WHIR batching, sumcheck moments, fold |
| `batch_ntt_small.rs` | 56 | Small batch NTT |
| `sponge.rs` | 48 | Sponge grind |
| `device_info.rs` | 24 | GPU core count query |

### logup_zerocheck rules (complete)
- `rules/mod.rs` (406 lines) - SymbolicRulesMetal, SymbolicRulesBuilder
- `rules/codec.rs` (257 lines) - Codec trait, rule encoding/decoding

### Metal shaders (complete, all compile)
11 header files + 19 kernel files = 6,125 lines of MSL.

---

## What Remains: todo!() and Stubs

### Critical: Three Proof Algorithm Functions

These are the main entry points that orchestrate GPU kernel dispatches for proving. Each is a complex algorithm with state management, multiple kernel calls, and Fiat-Shamir transcript interaction.

#### 1. `prove_zerocheck_and_logup_metal()` in `src/logup_zerocheck/mod.rs`
- **CUDA equivalent**: `src/logup_zerocheck/mod.rs` (~1596 lines)
- **Current state**: `todo!()` placeholder (31 lines)
- **What it does**: Orchestrates the full logup/zerocheck proving pipeline - GKR protocol, batch constraint evaluation, MLE sumcheck rounds
- **CUDA struct**: `LogupZerocheckGpu` holds all state (eq evaluation layers, trace buffers, lambda powers, etc.)
- **Depends on**: All 8 logup_zerocheck submodule files (see below)

#### 2. `prove_stacked_opening_reduction_metal()` in `src/stacked_reduction.rs`
- **CUDA equivalent**: `src/stacked_reduction.rs` (~1005 lines)
- **Current state**: `todo!()` placeholder (43 lines)
- **What it does**: Proves stacked polynomial commitment opening reduction via sumcheck
- **CUDA struct**: `StackedReductionGpu` manages trace pointers, eq eval buffers, sumcheck state
- **Key types already defined**: `UnstackedSlice`, `StackedPcsData2`, `STACKED_REDUCTION_S_DEG`

#### 3. `prove_whir_opening_metal()` in `src/whir.rs`
- **CUDA equivalent**: `src/whir.rs` (~721 lines)
- **Current state**: `todo!()` placeholder (31 lines)
- **What it does**: WHIR polynomial commitment opening proof - algebraic batching, coefficient moments sumcheck, folding, Merkle tree proofs
- **Key type already defined**: `BatchingTracePacket`

### Critical: LogUp/Zerocheck Submodule Stubs

These are called by `prove_zerocheck_and_logup_metal()`. Each is a 1-line stub.

| File | CUDA Lines | What It Does |
|------|-----------|--------------|
| `logup_zerocheck/round0.rs` | 127 | NTT-based zerocheck/logup round 0 evaluation |
| `logup_zerocheck/mle_round.rs` | 113 | MLE sumcheck round evaluation |
| `logup_zerocheck/batch_mle.rs` | 111 | Multi-AIR batched MLE evaluation |
| `logup_zerocheck/batch_mle_monomial.rs` | 124 | Monomial-based batched MLE |
| `logup_zerocheck/fractional.rs` | 96 | GKR fractional sumcheck (tree layer, compute round, fold) |
| `logup_zerocheck/gkr_input.rs` | 63 | GKR input evaluation |
| `logup_zerocheck/fold_ple.rs` | 91 | PLE folding operations |
| `logup_zerocheck/errors.rs` | 60 | Error types for logup/zerocheck |

**Total missing**: ~785 lines across 8 files

### Critical: Sumcheck Module

| File | CUDA Lines | What It Does |
|------|-----------|--------------|
| `sumcheck.rs` | 376 | Sumcheck prover logic - round evaluation, coefficient extraction |

**Current state**: 1-line stub

### Tests

| File | CUDA Lines | What It Does |
|------|-----------|--------------|
| `tests.rs` | 885 | NTT roundtrip, kernel correctness, cross-backend validation |

**Current state**: 1-line stub

---

## Estimated Remaining Work

| Category | Estimated Lines | Complexity |
|----------|----------------|------------|
| `logup_zerocheck/mod.rs` (prove function) | ~1500 | High - complex state machine |
| `stacked_reduction.rs` (prove function) | ~950 | High - multi-round sumcheck |
| `whir.rs` (prove function) | ~690 | High - WHIR protocol |
| `sumcheck.rs` | ~375 | Medium - sumcheck prover |
| `logup_zerocheck/` submodules (8 files) | ~785 | Medium - kernel orchestration |
| `tests.rs` | ~885 | Medium - correctness tests |
| **Total** | **~5185** | |

---

## Agent Team Recommendations

### Parallel Work Streams

The remaining work can be split across 3 agents:

**Agent 1: Sumcheck + Stacked Reduction** (priority: highest)
- Implement `sumcheck.rs` by porting from `cuda-backend/src/sumcheck.rs`
- Implement `stacked_reduction.rs` prove function by porting from `cuda-backend/src/stacked_reduction.rs`
- These two are tightly coupled (stacked reduction uses sumcheck)

**Agent 2: LogUp/Zerocheck** (priority: highest, can run in parallel with Agent 1)
- Implement all 8 submodule files in `logup_zerocheck/`
- Implement `prove_zerocheck_and_logup_metal()` in `logup_zerocheck/mod.rs`
- Reference: `cuda-backend/src/logup_zerocheck/` (all files)

**Agent 3: WHIR + Tests** (priority: high, partially blocked by Agents 1-2)
- Implement `whir.rs` prove function by porting from `cuda-backend/src/whir.rs`
- Implement `tests.rs` after proof functions are complete
- Can start WHIR immediately (independent of logup/zerocheck)

### Translation Rules (CUDA -> Metal)

When porting the remaining CUDA code, apply these substitutions:

| CUDA | Metal |
|------|-------|
| `DeviceBuffer<T>` | `MetalBuffer<T>` |
| `DeviceMatrix<T>` | `MetalMatrix<T>` |
| `GpuBackend` | `MetalBackend` |
| `GpuDevice` | `MetalDevice` |
| `DuplexSpongeGpu` | `DuplexSpongeMetal` |
| `StackedPcsDataGpu` | `StackedPcsDataMetal` |
| `SymbolicRulesGpu` | `SymbolicRulesMetal` |
| `CudaError` | `MetalError` |
| `MemTracker::start(...)` | (remove or adapt - Metal uses unified memory) |
| `cuda_memcpy(...)` | Direct pointer access (unified memory) |
| `LOG_WARP_SIZE` | `LOG_SIMD_SIZE` |
| FFI: `unsafe { _kernel_name(...) }` | `unsafe { kernel_name(...) }?` (Metal dispatch returns Result) |

### Key Architectural Notes

1. **Unified Memory**: Metal's StorageModeShared means no explicit H2D/D2H copies. Where CUDA has `copy_to_device()` / `copy_from_device()`, Metal just reads/writes the buffer directly after GPU sync.

2. **Error Handling**: CUDA FFI returns `i32` error codes checked by the wrapper. Metal dispatch returns `Result<(), MetalError>` - propagate with `?`.

3. **MemTracker**: CUDA uses `MemTracker` extensively for memory profiling. Metal can omit this or use a simpler approach since unified memory means no separate GPU memory accounting.

4. **Kernel Dispatch**: All Metal kernel dispatch wrappers in `src/metal/` are already implemented. The remaining work is the Rust-side algorithm logic that **calls** these dispatch functions - it does not require writing new Metal shaders.

---

## File Reference: CUDA -> Metal Mapping

For each remaining file, the exact CUDA source to port from:

```
cuda-backend/src/sumcheck.rs          -> metal-backend/src/sumcheck.rs
cuda-backend/src/stacked_reduction.rs -> metal-backend/src/stacked_reduction.rs
cuda-backend/src/whir.rs              -> metal-backend/src/whir.rs
cuda-backend/src/logup_zerocheck/mod.rs            -> metal-backend/src/logup_zerocheck/mod.rs
cuda-backend/src/logup_zerocheck/round0.rs         -> metal-backend/src/logup_zerocheck/round0.rs
cuda-backend/src/logup_zerocheck/mle_round.rs      -> metal-backend/src/logup_zerocheck/mle_round.rs
cuda-backend/src/logup_zerocheck/batch_mle.rs      -> metal-backend/src/logup_zerocheck/batch_mle.rs
cuda-backend/src/logup_zerocheck/batch_mle_monomial.rs -> metal-backend/src/logup_zerocheck/batch_mle_monomial.rs
cuda-backend/src/logup_zerocheck/fractional.rs     -> metal-backend/src/logup_zerocheck/fractional.rs
cuda-backend/src/logup_zerocheck/gkr_input.rs      -> metal-backend/src/logup_zerocheck/gkr_input.rs
cuda-backend/src/logup_zerocheck/fold_ple.rs       -> metal-backend/src/logup_zerocheck/fold_ple.rs
cuda-backend/src/logup_zerocheck/errors.rs         -> metal-backend/src/logup_zerocheck/errors.rs
cuda-backend/src/tests.rs             -> metal-backend/src/tests.rs
```

---

## Verification Checklist

After completing the remaining implementation:

- [ ] `cargo check -p openvm-metal-backend` passes (currently passes)
- [ ] `cargo test -p openvm-metal-common` passes (currently passes)
- [ ] `cargo test -p openvm-metal-backend` passes (blocked on tests.rs)
- [ ] No `todo!()` panics remain in proof paths
- [ ] NTT roundtrip test passes
- [ ] End-to-end proof generation succeeds on a small example
- [ ] Cross-backend validation: Metal proof == CPU proof
