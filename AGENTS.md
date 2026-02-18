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

Only run before committing.
```bash
cargo clippy --all-targets --tests -- -D warnings
cargo +nightly fmt          # requires nightly for unstable rustfmt features
cargo +nightly fmt -- --check  # check only
```

## Project Structure

Rust workspace with 3 default-member crates:

- **`openvm-stark-backend`** (`crates/stark-backend`): Core SWIRL proof system — prover, verifier, AIR builders, interactions, keygen, PCS. Re-exports Plonky3 crates.
- **`openvm-codec-derive`** (`crates/stark-backend/codec-derive`): Proc macro for serialization codegen.
- **`openvm-stark-sdk`** (`crates/stark-sdk`): Concrete configs (`BabyBearPoseidon2Config`, `BabyBearBn254Poseidon2Config`), tracing/metrics, benchmarks.

Non-default members (require GPU/CUDA toolchain — don't add to default-members):
- **`openvm-cuda-builder`** (`crates/cuda-builder`): Build-time utility for compiling `.cu` files via `nvcc`/`cc`. Used as a `[build-dependency]` by CUDA crates.
- **`openvm-cuda-common`** (`crates/cuda-common`): Shared CUDA runtime utilities. Includes `launcher.cuh` and virtual memory manager (VPMM).
- **`openvm-cuda-backend`** (`crates/cuda-backend`): GPU prover implementation, depends on `cuda-common`.

Plonky3 is a git dependency — find it via `Cargo.toml` or check `../Plonky3`.

## Code Conventions

- Plonky3 pinned to `=0.4.1` — do NOT upgrade
- Default features: `parallel` (rayon), `metrics`
- `test-utils` feature gate for test fixtures/helpers — tests that need it use `#[cfg(feature = "test-utils")]`
- Integration tests in `crates/stark-backend/tests/`
- Formatting: `imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`
