# Metal Backend Implementation Plan

## Team Structure

The implementation is organized into **7 phases** with work distributed across a team of Claude Code agents. Phases are designed to maximize parallelism while respecting dependencies.

```
Phase 0: Scaffolding & Permissions (Lead)
    │
    ├── Phase 1: metal-common crate (Agent A)
    │
    ├── Phase 2: Metal shaders - headers + kernels (Agent B + Agent C)
    │       Agent B: headers + core kernels (field arith, NTT, matrix, poly)
    │       Agent C: crypto + protocol kernels (poseidon2, sumcheck, whir, logup)
    │
    └── Phase 3: Build system (Agent D)
            │
            ├── Phase 4: FFI bindings + Rust dispatch layer (Agent E + Agent F)
            │       Agent E: core FFI (ntt, matrix, poly, mle_interpolate)
            │       Agent F: protocol FFI (merkle, sumcheck, whir, logup, sponge)
            │
            └── Phase 5: Backend implementation (Agent G + Agent H)
                    Agent G: MetalBackend, MetalDevice, engine, data_transporter, base
                    Agent H: stacked_pcs, stacked_reduction, whir, sumcheck, logup_zerocheck
                        │
                        └── Phase 6: Tests & validation (Agent I)
```

## Current Gap Snapshot (as of 2026-02-18)

The current `openvm-metal-backend` implementation still routes prover-critical work through CPU code paths. In particular, commitment, RAP constraints, opening reduction, and WHIR opening are delegated to CPU helpers after converting Metal data back to host-oriented structures.

This plan treats those CPU fallbacks as temporary scaffolding. Completion means removing them and reaching end-to-end Metal execution parity with `openvm-cuda-backend`.

## Definition of Done (Hard Release Gate)

The project is not complete until every item below is satisfied:

1. **No CPU prover fallback**  
   During `BabyBearPoseidon2MetalEngine::prove`, no prover-critical stage delegates to CPU proving helpers.

2. **Full prover-stage Metal execution**  
   The following stages execute on Metal kernels + Metal buffers:
   - Stacked PCS commitment and Merkle hashing
   - RAP constraints (zerocheck, logup, GKR/sumcheck)
   - Stacked opening reduction
   - WHIR opening
   - Transcript grinding work

3. **CUDA module parity**  
   Metal implementations exist and are actively used for the CUDA-equivalent components:
   - `ntt`, `poly`, `matrix`, `merkle_tree`, `sponge`
   - `sumcheck`, `prefix`, `whir`, `stacked_pcs`, `stacked_reduction`
   - `logup_zerocheck` (including round-0, MLE, and GKR input paths)

4. **Protocol/output parity with CUDA**  
   For deterministic vectors, Metal and CUDA runs produce protocol-equivalent commitments/challenges/openings and valid proofs under the same verifier.

5. **Kernel correctness parity**  
   Each Metal kernel category is validated against CPU/CUDA references on randomized + edge-case fixtures.

6. **Fallback observability + zero threshold**  
   Fallback metrics/tracing are present; all CI prover tests require zero CPU fallback invocations.

---

## Phase 0: Scaffolding & Permissions (Lead Agent)

**Goal**: Create all crate directories, Cargo.toml files, workspace configuration, and empty module stubs. Perform all operations that may require user permission upfront.

**Permissions needed upfront**:
- File creation across `crates/metal-common/` and `crates/metal-backend/`
- Workspace Cargo.toml modification
- Running `cargo check` to validate structure
- Running `xcrun --sdk macosx metal --version` to verify Metal compiler availability
- Running `cargo add` or editing Cargo.toml for dependencies
- Creating directories for Metal shaders
- Git operations (branch, add, commit)

### Tasks

1. **Verify Metal toolchain availability**
   ```bash
   xcrun --sdk macosx metal --version
   xcrun --sdk macosx metallib --version
   # Verify Xcode command line tools are installed
   ```

2. **Create `crates/metal-common/` crate**
   - `Cargo.toml` with dependencies: `metal`, `thiserror`, `tracing`, `metrics`
   - `src/lib.rs` with module declarations
   - Empty stub files for all modules

3. **Create `crates/metal-backend/` crate**
   - `Cargo.toml` with dependencies on `metal-common`, `openvm-stark-backend`, `openvm-stark-sdk`, `metal`, `p3-*` crates
   - `src/lib.rs` with module declarations
   - `build.rs` stub
   - `metal/include/` and `metal/src/` directories
   - Empty stub files for all modules

4. **Update workspace `Cargo.toml`**
   - Add `metal-common` and `metal-backend` to workspace members
   - Add workspace dependencies for `metal` crate

5. **Initial `cargo check`** to ensure workspace compiles

6. **Commit scaffold**

### Deliverables
- Both crates compile (empty stubs)
- Metal compiler verified working
- All directories and files created
- Workspace configuration updated

---

## Phase 1: metal-common Crate (Agent A)

**Goal**: Implement the core Metal GPU abstractions equivalent to `cuda-common`.

**Dependencies**: Phase 0 complete

**Reference**: `crates/cuda-common/src/`

### Tasks

1. **`error.rs`** - Error types
   - `MetalError` - wrapping Metal/NSError
   - `MemoryError` - allocation failures, OOM
   - `MemCopyError` - transfer errors with size mismatch detection

2. **`device.rs`** - Device management
   - `MetalContext` singleton holding `metal::Device` + `metal::CommandQueue`
   - `get_device()` / `init_device()` functions
   - Device info queries (name, max threadgroup size, max buffer length, recommended working set size)

3. **`d_buffer.rs`** - `MetalBuffer<T>` implementation
   - Wraps `metal::Buffer` with typed access
   - `with_capacity(len)` - allocate with `StorageModeShared`
   - `as_ptr()` / `as_mut_ptr()` - via `buffer.contents()`
   - `as_metal_buffer()` - access underlying Metal buffer for kernel dispatch
   - `fill_zero()` - via blit encoder or memset
   - `fill_zero_suffix(start_idx)`
   - `len()`, `is_empty()`
   - `view()` → `MetalBufferView` (non-owning)
   - `Drop` implementation (Metal handles deallocation via ARC, but we track metrics)
   - Memory tracking via `MemTracker`

4. **`copy.rs`** - Memory copy traits
   - `MemCopyH2D<T>` for `[T]` - direct `ptr::copy_nonoverlapping` to shared buffer
   - `MemCopyD2H<T>` for `MetalBuffer<T>` - direct read from shared buffer (after sync)
   - `MemCopyD2D<T>` for `MetalBuffer<T>` - via blit command encoder
   - Synchronization: must ensure GPU completion before CPU reads

5. **`command.rs`** - Command submission wrappers
   - `MetalCommandQueue` - wraps `metal::CommandQueue`
   - `MetalCommandBuffer` - wraps command buffer with completion tracking
   - `sync_and_check()` - commit + wait + check status
   - Helper for dispatching compute with proper encoding

6. **Unit tests** for buffer allocation, copy, fill operations

### Deliverables
- `MetalBuffer<T>` fully functional
- Copy traits working
- Error handling complete
- `cargo test -p openvm-metal-common` passes

---

## Phase 2: Metal Shader Kernels (Agent B + Agent C)

**Goal**: Write all Metal Shading Language compute kernels, translating from CUDA.

**Dependencies**: Phase 0 complete (directory structure exists)

**Reference**: `crates/cuda-backend/cuda/src/*.cu` and `crates/cuda-backend/cuda/include/*.cuh`

### Agent B: Headers + Core Kernels

1. **Header files** (`metal/include/`)
   - `baby_bear.h` - BabyBear field arithmetic in MSL
     - Montgomery multiplication, addition, subtraction
     - Power/inverse functions
     - Encode/decode from Montgomery form
     - Reference: risc0's `fp.h` (BabyBear, same prime)
   - `baby_bear_ext.h` - Extension field (degree 4)
     - Arithmetic over irreducible polynomial
     - Reference: CUDA's extension field operations in `utils.cuh`
   - `utils.h` - Shared utilities (exp, bit reversal, etc.)
   - `device_ntt.h` - NTT state structures

2. **NTT kernels** (`metal/src/ntt.metal`)
   - `generate_all_twiddles` - precompute twiddle factors
   - `generate_partial_twiddles` - window-based twiddles
   - `bit_reverse` / `bit_reverse_ext` - bit reversal permutation
   - `ntt_forward_step` - Cooley-Tukey butterfly (mixed radix)
   - Reference: CUDA's `cuda/supra/ntt.cu` and `cuda/src/batch_ntt_small.cu`

3. **Matrix kernels** (`metal/src/matrix.metal`)
   - `matrix_transpose_fp` / `matrix_transpose_fpext`
   - `matrix_get_rows_fp`
   - `split_ext_to_base_col_major`
   - `batch_rotate_pad`
   - `batch_expand_pad` / `batch_expand_pad_wide`
   - `collapse_strided_matrix`
   - `lift_padded_matrix_evals`

4. **Polynomial kernels** (`metal/src/poly.metal`)
   - `algebraic_batch_matrices`
   - `eq_hypercube_stage_ext` / `mobius_eq_hypercube_stage_ext`
   - `batch_eq_hypercube_stage`
   - `eval_poly_ext_at_point`
   - `vector_scalar_multiply_ext`
   - `transpose_fp_to_fpext_vec`

5. **MLE interpolation** (`metal/src/mle_interpolate.metal`)
   - All MLE interpolation stage kernels

6. **Batch NTT small** (`metal/src/batch_ntt_small.metal`)
   - Small NTT kernel for l_skip parameter

### Agent C: Crypto + Protocol Kernels

1. **Additional headers** (`metal/include/`)
   - `poseidon2.h` - Poseidon2 round constants, MDS matrices
   - `monomial.h` - Monomial computation structures
   - `codec.h` - Instruction encoding for DAG evaluation
   - `dag_entry.h` - DAG node representation
   - `eval_ctx.h` - Evaluation context
   - `frac_ext.h` - Fractional extension field

2. **Merkle tree / Poseidon2** (`metal/src/merkle_tree.metal`)
   - `poseidon2_compressing_row_hashes` - hash matrix rows to digests
   - `poseidon2_strided_compress_layer` - Merkle tree layer with stride
   - `poseidon2_adjacent_compress_layer` - adjacent pair compression
   - `query_digest_layers` - multi-tree parallel querying
   - Full Poseidon2 permutation in MSL

3. **Sponge / grinding** (`metal/src/sponge.metal`)
   - `sponge_grind` - Proof-of-work witness search

4. **Sumcheck** (`metal/src/sumcheck.metal`)
   - `sumcheck_mle_round` - MLE folding round
   - `fold_mle` / `batch_fold_mle` - column folding
   - `fold_mle_column` - in-place column fold
   - `fold_ple_from_coeffs` - polynomial folding
   - `reduce_over_x_and_cols` - horizontal reduction
   - `triangular_fold_mle` - segment-based folding

5. **Prefix scan** (`metal/src/prefix.metal`)
   - `prefix_scan_block_ext` - parallel scan phases
   - `prefix_scan_block_downsweep_ext`
   - `prefix_scan_epilogue_ext`

6. **WHIR** (`metal/src/whir.metal`)
   - `whir_algebraic_batch_traces`
   - `whir_sumcheck_coeff_moments_round`
   - `whir_fold_coeffs_and_moments`
   - `w_moments_accumulate`

7. **Stacked reduction** (`metal/src/stacked_reduction.metal`)
   - Stacked proof sumcheck kernels

8. **LogUp/Zerocheck** (`metal/src/logup_zerocheck/`)
   - `batch_mle.metal` - batch MLE operations
   - `batch_mle_monomial.metal` - monomial batch MLE
   - `gkr.metal` - GKR protocol kernels
   - `gkr_input.metal` - GKR input preparation
   - `mle.metal` - MLE operations
   - `utils.metal` - shared utilities
   - `zerocheck_round0.metal` - zerocheck round 0
   - `logup_round0.metal` - logup round 0

### Deliverables
- All `.metal` files compile with `xcrun metal -c`
- All header files provide correct field arithmetic
- Kernel function signatures match what the Rust FFI layer expects

---

## Phase 3: Build System (Agent D)

**Goal**: Implement `build.rs` that compiles Metal shaders into a `.metallib` and embeds it.

**Dependencies**: Phase 0 complete, Phase 2 headers exist (at minimum `baby_bear.h`)

### Tasks

1. **`build.rs` implementation**
   - Detect macOS/iOS target
   - Find all `.metal` files in `metal/src/`
   - Stage 1: `xcrun -sdk macosx metal -c <file>.metal -I metal/include/ -o <file>.air -std=metal3.0`
   - Stage 2: `xcrun -sdk macosx metallib *.air -o kernels.metallib`
   - Set `cargo:rustc-env=METAL_KERNELS_PATH=<path>`
   - Set `cargo:rerun-if-changed` for all source and header files
   - Parallel compilation of `.air` files

2. **Caching** (optional optimization)
   - Hash source files to detect changes
   - Cache compiled `.metallib` in `target/` or `~/.cache/`

3. **Error handling**
   - Clear error messages if `xcrun` not found
   - Validation of Metal compiler version
   - Handle missing Xcode command line tools gracefully

4. **Verify build works end-to-end**
   - At minimum with a trivial kernel
   - Test that `include_bytes!(env!("METAL_KERNELS_PATH"))` works

### Deliverables
- `build.rs` compiles all `.metal` files
- `.metallib` is embedded in the binary
- Incremental builds work (rerun-if-changed)

---

## Phase 4: FFI Bindings / Rust Dispatch Layer (Agent E + Agent F)

**Goal**: Implement the Rust-side kernel dispatch wrappers in `src/metal/`.

**Dependencies**: Phase 1 (metal-common), Phase 3 (build system), Phase 2 (kernel signatures)

### Shared: Core Dispatch Infrastructure

Create `src/metal/mod.rs` with:

```rust
use metal::{ComputePipelineState, CommandQueue, Device, Library};
use std::collections::HashMap;

const METAL_LIB: &[u8] = include_bytes!(env!("METAL_KERNELS_PATH"));

pub struct MetalKernels {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl MetalKernels {
    pub fn new(device: &Device, queue: &CommandQueue) -> Self { ... }
    pub fn dispatch(&self, name: &str, args: &[KernelArg], grid: MTLSize, tg: MTLSize) { ... }
}
```

### Agent E: Core FFI Wrappers

1. **`src/metal/ntt.rs`** - NTT dispatch functions
   - `generate_all_twiddles()`
   - `generate_partial_twiddles()`
   - `bit_rev()` / `bit_rev_ext()`
   - `ct_mixed_radix_narrow()` (NTT forward step)

2. **`src/metal/batch_ntt_small.rs`**
   - `batch_ntt_small()`
   - `generate_device_ntt_twiddles()`

3. **`src/metal/matrix.rs`** - Matrix operation dispatch
   - All matrix transpose, get_rows, split, pad, collapse, lift functions

4. **`src/metal/poly.rs`** - Polynomial dispatch
   - All algebraic batch, eq hypercube, eval, multiply functions

5. **`src/metal/mle_interpolate.rs`** - MLE interpolation dispatch

6. **`src/metal/device_info.rs`** - Device query wrappers

### Agent F: Protocol FFI Wrappers

1. **`src/metal/merkle_tree.rs`** - Merkle tree / Poseidon2 dispatch
   - `poseidon2_compressing_row_hashes()`
   - `poseidon2_strided_compress_layer()`
   - `poseidon2_adjacent_compress_layer()`
   - `query_digest_layers()`

2. **`src/metal/sponge.rs`** - Grinding dispatch
   - `sponge_grind()`

3. **`src/metal/mod.rs` sumcheck module** - Sumcheck dispatch
   - All sumcheck, fold, reduce functions

4. **`src/metal/mod.rs` prefix module** - Prefix scan dispatch

5. **`src/metal/whir.rs`** - WHIR dispatch
   - All WHIR kernel dispatch functions

6. **`src/metal/stacked_reduction.rs`** - Stacked reduction dispatch

7. **`src/metal/logup_zerocheck.rs`** - LogUp/Zerocheck dispatch
   - All DAG evaluation, monomial, round0 dispatch functions

### Deliverables
- Every CUDA FFI function in `cuda-backend/src/cuda/` has a Metal equivalent
- Functions have matching signatures (taking `MetalBuffer<T>` instead of `DeviceBuffer<T>`)
- Dispatch parameters (grid/threadgroup sizes) are computed correctly for Metal

---

## Phase 5: Backend Implementation (Agent G + Agent H)

**Goal**: Implement all high-level Rust logic and wire it to Metal dispatch so prover execution does not rely on CPU fallback paths.

**Dependencies**: Phase 4 (FFI layer), Phase 1 (metal-common)

**Reference**: `crates/cuda-backend/src/` (every file)

### Agent G: Core Backend

1. **`types.rs`** - Type aliases (copy from cuda-backend, identical)

2. **`error.rs`** - ProverError types (adapt from cuda-backend)

3. **`base.rs`** - `MetalMatrix<T>`, `MetalMatrixView<T>`, Basis types
   - Uses `Arc<MetalBuffer<T>>` for shared ownership
   - Implements `MatrixDimensions`, `MemCopyD2H`

4. **`metal_backend.rs`** - `MetalBackend` marker struct + `ProverBackend` impl

5. **`device.rs`** - `MetalDevice` struct
   - Holds `MetalKernels`, `SystemParams`, `MetalProverConfig`
   - Device info queries

6. **`engine.rs`** - `BabyBearPoseidon2MetalEngine` implementing `StarkEngine`
   - `new()`, `config()`, `device()`, `initial_transcript()`, `prove()`, `verify()`

7. **`data_transporter.rs`** - `DeviceDataTransporter<SC, MetalBackend>` implementation
   - `transport_pk_to_device()` - proving key → Metal buffers
   - `transport_matrix_to_device()` - trace matrix → Metal buffer
   - `transport_pcs_data_to_device()`
   - `transport_matrix_from_device_to_host()`

8. **`sponge.rs`** - `DuplexSpongeMetal` implementing `FiatShamirTranscript`
   - Hybrid host/device state management
   - GPU-accelerated grinding

9. **`pkey.rs`** - Metal proving key structures (`AirDataMetal`, etc.)

10. **`monomial.rs`** - Monomial expansion (port from cuda-backend)

### Agent H: Protocol Implementation

1. **`ntt.rs`** - NTT orchestration
   - Forward/inverse NTT using Metal dispatch
   - Twiddle factor management
   - Window-based approach for large transforms

2. **`poly.rs`** - MLE interpolation, `PleMatrix` type

3. **`merkle_tree.rs`** - `MerkleTreeMetal` construction and querying

4. **`stacked_pcs.rs`** - Stacked commitment scheme
   - `stacked_commit()` implementation

5. **`stacked_reduction.rs`** - Opening reduction proof

6. **`whir.rs`** - WHIR polynomial commitment opening

7. **`sumcheck.rs`** - Sumcheck prover

8. **`logup_zerocheck/`** - Full logup/zerocheck module
   - `mod.rs`, `batch_mle.rs`, `batch_mle_monomial.rs`
   - `round0.rs`, `mle_round.rs`, `gkr_input.rs`
   - `fractional.rs`, `fold_ple.rs`, `rules/`, `errors.rs`

### Deliverables
- `MetalBackend` implements `ProverBackend`
- `MetalDevice` implements `ProverDevice<MetalBackend, DuplexSpongeMetal>`
- `BabyBearPoseidon2MetalEngine` implements `StarkEngine`
- `cargo check -p openvm-metal-backend` compiles
- `prove()` hot path uses Metal-native prover stages (no CPU delegation for commit/RAP/opening/WHIR)

---

## Phase 6: Tests & Validation (Agent I)

**Goal**: Copy all tests from cuda-backend, adapt for Metal, and verify correctness.

**Dependencies**: Phase 5 complete

### Tasks

1. **Copy and adapt `tests/ntt_roundtrip.rs`**
   - Replace CUDA device init with Metal device init
   - Replace `DeviceBuffer` with `MetalBuffer`
   - Replace CUDA NTT calls with Metal NTT calls
   - Verify forward → inverse roundtrip

2. **Copy and adapt `examples/keccakf.rs`**
   - Replace `BabyBearPoseidon2GpuEngine` with `BabyBearPoseidon2MetalEngine`
   - Note: Metal doesn't support concurrent multi-stream like CUDA
   - Adapt to single-queue or serial proof generation
   - Verify proof generation and verification

3. **Individual kernel correctness tests**
   - For each major kernel category, create tests that:
     a. Generate random input on CPU
     b. Run CPU reference implementation
     c. Run Metal kernel
     d. Compare outputs
   - Categories: NTT, matrix ops, polynomial ops, Poseidon2, sumcheck, WHIR

4. **Cross-backend validation**
   - Run a small proof with `BabyBearPoseidon2GpuEngine` (CUDA)
   - Run same proof with `BabyBearPoseidon2MetalEngine`
   - Verify protocol-equivalent outputs for deterministic fixtures
   - Verify both proofs pass verification

5. **Memory pressure tests**
   - Test with large traces to verify no OOM on typical Apple Silicon configs
   - Test buffer allocation and deallocation patterns

6. **CI configuration** (optional)
   - `.github/workflows/tests-metal.yml`
   - Requires macOS runner with Apple Silicon

### Deliverables
- All tests pass on Apple Silicon
- NTT roundtrip verified
- Keccak-f example proves and verifies
- CUDA-vs-Metal protocol/output parity verified on deterministic fixtures
- No-fallback tests verify zero CPU prover fallback invocations

---

## Dependency Graph

```
Phase 0 (Scaffold)
    │
    ├─── Phase 1 (metal-common) ──────────────────────┐
    │                                                   │
    ├─── Phase 2A (Core shaders - Agent B) ───┐        │
    │                                          │        │
    ├─── Phase 2B (Protocol shaders - Agent C)─┤        │
    │                                          │        │
    └─── Phase 3 (Build system) ──────────────┤        │
                                               │        │
                            ┌──────────────────┘        │
                            │                           │
                    Phase 4A (Core FFI - Agent E) ──────┤
                    Phase 4B (Protocol FFI - Agent F) ──┤
                                                        │
                            ┌───────────────────────────┘
                            │
                    Phase 5A (Core backend - Agent G) ──┐
                    Phase 5B (Protocol impl - Agent H) ─┤
                                                        │
                                                Phase 6 (Tests - Agent I)
```

## Estimated Effort

| Phase | Files | Complexity | Parallelizable |
|-------|-------|------------|----------------|
| 0: Scaffold | ~30 stubs | Low | No (sequential) |
| 1: metal-common | ~6 files | Medium | Independent |
| 2: Metal shaders | ~20 .metal + ~7 .h | High | 2-way parallel (B+C) |
| 3: Build system | 1 file | Medium | Independent (needs headers) |
| 4: FFI bindings | ~12 files | Medium | 2-way parallel (E+F) |
| 5: Backend impl | ~20 files | High | 2-way parallel (G+H) |
| 6: Tests | ~5 files | Medium | After Phase 5 |

## Key Risk Areas

1. **NTT performance**: The SUPRA NTT library is highly optimized for NVIDIA. Metal NTT may need separate optimization passes.

2. **Poseidon2 constants**: Must match exactly with the CPU implementation. Any mismatch causes proof failures.

3. **Extension field degree**: This project uses degree 4 (`D_EF = 4`), same as risc0. This is convenient — risc0's `fpext.h` (degree 4, irreducible polynomial x^4 - 11) can be used as a direct reference.

4. **No CUDA-like streams**: Metal command queues are serial by default. Multi-proof parallelism (as in keccakf.rs) needs different approach (multiple command queues or serial execution).

5. **Memory limits**: Apple Silicon unified memory is shared with the system. Large proofs need careful memory management to avoid system pressure.

6. **Threadgroup memory limits**: Apple Silicon has 32KB threadgroup memory vs CUDA's 48KB+ shared memory. Some kernels may need adaptation.
