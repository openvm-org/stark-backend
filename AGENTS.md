# AGENTS.md

## Build & Test Commands

Always check/test individual crates, never the whole workspace.

```bash
# Check
cargo check -p openvm-stark-backend

# Test (prefer nextest; use cargo test for single tests)
cargo nextest run -p openvm-stark-backend
cargo test -p openvm-stark-backend -- <test_name>

# CUDA (requires GPU, non-default workspace member)
cargo check -p openvm-cuda-backend
cargo nextest run -p openvm-cuda-backend --test-threads=4
```

## Formatting and Linting

Before committing or opening/updating a PR, formatting and clippy must pass for
the affected crate(s). Do not open a PR with known fmt or clippy failures.

```bash
cargo clippy -p <crate> --all-targets --tests -- -D warnings
cargo +nightly fmt          # requires nightly for unstable rustfmt features
cargo +nightly fmt -- --check  # check only
```

## Project Structure

Rust workspace with 5 default-member crates:

- **`openvm-stark-backend`** (`crates/stark-backend`): Core SWIRL proof system — prover, verifier, AIR builders, interactions, keygen, PCS. Re-exports Plonky3 crates.
- **`openvm-codec-derive`** (`crates/stark-backend/codec-derive`): Proc macro for serialization codegen.
- **`openvm-stark-sdk`** (`crates/stark-sdk`): Concrete configs (`BabyBearPoseidon2Config`, `BabyBearBn254Poseidon2Config`), tracing/metrics, benchmarks.
- **`openvm-backend-tests`** (`crates/backend-tests`): Shared backend-generic test suite for the SWIRL proof system.
- **`openvm-cpu-backend`** (`crates/cpu-backend`): Optimized row-major CPU prover backend.

Non-default CUDA support members (require GPU/CUDA toolchain — don't add to default-members):
- **`openvm-cuda-builder`** (`crates/cuda-builder`): Build-time utility for compiling `.cu` files via `nvcc`/`cc`. Used as a `[build-dependency]` by CUDA crates.
- **`openvm-cuda-common`** (`crates/cuda-common`): Shared CUDA runtime utilities. Includes `launcher.cuh` and virtual memory manager (VPMM).
- **`openvm-cuda-backend`** (`crates/cuda-backend`): GPU prover implementation, depends on `cuda-common`.

Non-default benchmark workspace members:
- **`openvm-benchmarks-fields`** (`benchmarks/fields`): CUDA field-extension and Poseidon2 benchmarks and verification tests.
- **`openvm-benchmark-synthetic`** (`benchmarks/synthetic`): Synthetic AIR benchmark suite, including the champ-vs-candidate profile replay workflow and the `mem_meter_runner` memory-metering validation runner documented in its README.

When changing GPU buffer layout, scheduling, or scratch allocations in any proving phase — fractional-GKR, batch-constraint/batch-MLE, stacking/RS encoding, or WHIR opening — update the proving memory model in `crates/stark-backend/src/memory_metering.rs` (see `ProvingMemoryConfig::estimate` and the mirrored constants documented there), and validate against measured peaks with `mem_meter_runner`. For GKR changes, also update the accounting in `docs/cuda-backend/gkr-prover.md`.

Plonky3 crates are exact-version crates.io dependencies pinned in the workspace `Cargo.toml`.

## Code Conventions

- Plonky3 pinned to `=0.4.3` — do NOT upgrade
- Default features: `parallel` (rayon), `metrics`
- `test-utils` feature gate for test fixtures/helpers — tests that need it use `#[cfg(feature = "test-utils")]`
- Integration tests in `crates/stark-backend/tests/`
- Formatting: `imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`
