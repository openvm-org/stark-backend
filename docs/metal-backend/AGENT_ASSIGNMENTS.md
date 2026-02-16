# Agent Assignments for Metal Backend Implementation

## Overview

This document provides concrete task assignments for each agent in the implementation team. Each agent has a clear scope, inputs, outputs, and acceptance criteria.

---

## Lead Agent: Phase 0 - Scaffolding & Permissions

### Objective
Set up the entire project structure, verify toolchain, and perform all permission-requiring operations upfront so that subsequent agents can work without permission interruptions.

### Pre-flight Checks
```bash
# 1. Verify Metal compiler
xcrun --sdk macosx metal --version
xcrun --sdk macosx metallib --version

# 2. Verify Rust toolchain
rustup show
cargo --version

# 3. Check current workspace compiles
cargo check --workspace
```

### Scaffold Operations

#### Create metal-common crate
```bash
mkdir -p crates/metal-common/src
```

**`crates/metal-common/Cargo.toml`**:
```toml
[package]
name = "openvm-metal-common"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[dependencies]
metal = "0.29"
thiserror = { workspace = true }
tracing = { workspace = true }
metrics = { workspace = true }
objc = "0.2"

[dev-dependencies]
rand = { workspace = true }
```

**`crates/metal-common/src/lib.rs`**:
```rust
pub mod error;
pub mod d_buffer;
pub mod copy;
pub mod command;
pub mod device;
```

Create empty stub files: `error.rs`, `d_buffer.rs`, `copy.rs`, `command.rs`, `device.rs`

#### Create metal-backend crate
```bash
mkdir -p crates/metal-backend/src/metal
mkdir -p crates/metal-backend/src/logup_zerocheck/rules
mkdir -p crates/metal-backend/metal/include
mkdir -p crates/metal-backend/metal/src/logup_zerocheck
mkdir -p crates/metal-backend/tests
mkdir -p crates/metal-backend/examples
```

**`crates/metal-backend/Cargo.toml`**:
```toml
[package]
name = "openvm-metal-backend"
version.workspace = true
edition.workspace = true
rust-version.workspace = true

[dependencies]
openvm-stark-backend = { path = "../stark-backend" }
openvm-stark-sdk = { path = "../stark-sdk" }
openvm-metal-common = { path = "../metal-common" }
metal = "0.29"
objc = "0.2"
p3-field = { workspace = true }
p3-baby-bear = { workspace = true }
p3-dft = { workspace = true }
p3-util = { workspace = true }
p3-symmetric = { workspace = true }
itertools = { workspace = true }
derive-new = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
getset = { workspace = true }
rustc-hash = { workspace = true }
rand = { workspace = true }

[build-dependencies]
# For compiling Metal shaders

[features]
default = []
test-utils = []

[[test]]
name = "ntt_roundtrip"
harness = true

[[example]]
name = "keccakf"
```

Create stub files for every module listed in the design.

#### Update workspace Cargo.toml
Add to `[workspace]` members:
```toml
"crates/metal-common",
"crates/metal-backend",
```

Add workspace dependency:
```toml
[workspace.dependencies]
metal = "0.29"
```

#### Verify compilation
```bash
cargo check -p openvm-metal-common
cargo check -p openvm-metal-backend
```

#### Create initial git commit
```bash
git add crates/metal-common crates/metal-backend
git add Cargo.toml Cargo.lock
git commit -m "scaffold: metal-common and metal-backend crates"
```

### Acceptance Criteria
- [ ] `xcrun metal --version` succeeds
- [ ] Both new crates compile with `cargo check`
- [ ] All directory structures exist
- [ ] Workspace Cargo.toml includes both crates
- [ ] Git commit created

---

## Agent A: Phase 1 - metal-common Crate

### Objective
Implement the core Metal GPU memory and command abstractions.

### Reference Files
- `crates/cuda-common/src/error.rs` → `crates/metal-common/src/error.rs`
- `crates/cuda-common/src/d_buffer.rs` → `crates/metal-common/src/d_buffer.rs`
- `crates/cuda-common/src/copy.rs` → `crates/metal-common/src/copy.rs`
- `crates/cuda-common/src/stream.rs` → `crates/metal-common/src/command.rs`
- `crates/cuda-common/src/common.rs` → `crates/metal-common/src/device.rs`

### Key Simplifications vs CUDA
1. **No VPMM needed**: Metal uses unified memory with ARC-based buffer lifecycle. Apple's memory allocator handles fragmentation.
2. **No explicit H2D/D2H memcpy**: StorageModeShared buffers are CPU-accessible. But synchronization is still needed.
3. **No stream-per-thread**: Use a global or per-instance CommandQueue.

### Implementation Details

#### MetalBuffer<T> (d_buffer.rs)
```rust
use metal::Buffer as MTLBuffer;
use std::sync::Arc;

pub struct MetalBuffer<T> {
    buffer: MTLBuffer,
    len: usize,
    _marker: PhantomData<T>,
}

// Must be Send + Sync (Metal buffers are thread-safe)
unsafe impl<T: Send> Send for MetalBuffer<T> {}
unsafe impl<T: Sync> Sync for MetalBuffer<T> {}
```

Key methods:
- `with_capacity(device: &metal::Device, len: usize) -> Self`
- `from_slice(device: &metal::Device, data: &[T]) -> Self` where T: Copy
- `as_ptr() -> *const T` (via `buffer.contents()`)
- `as_mut_ptr() -> *mut T`
- `to_host() -> Vec<T>` where T: Copy (requires GPU sync first)
- `copy_from_slice(data: &[T])` where T: Copy
- `fill_zero()` - memset via pointer
- `len() -> usize`
- `gpu_buffer() -> &MTLBuffer` (for kernel dispatch)

#### MetalContext (device.rs)
```rust
use std::sync::OnceLock;

static METAL_CONTEXT: OnceLock<MetalContext> = OnceLock::new();

pub struct MetalContext {
    pub device: metal::Device,
    pub queue: metal::CommandQueue,
}

pub fn get_context() -> &'static MetalContext { ... }
pub fn init_context() -> &'static MetalContext { ... }
```

### Acceptance Criteria
- [ ] `MetalBuffer<T>` allocates, writes, reads correctly
- [ ] Copy traits work (H2D via direct write, D2H via direct read, D2D via blit)
- [ ] `MetalContext` initializes Metal device
- [ ] Error types cover all failure modes
- [ ] Unit tests pass

---

## Agent B: Phase 2A - Metal Headers + Core Kernels

### Objective
Write the MSL header files (field arithmetic) and core compute kernels (NTT, matrix, polynomial, MLE).

### Reference Files
Read the following CUDA files and translate to Metal:
- `cuda/include/utils.cuh` → `metal/include/utils.h`
- `cuda/include/device_ntt.cuh` → `metal/include/device_ntt.h`
- `cuda/src/poly.cu` → `metal/src/poly.metal`
- `cuda/src/matrix.cu` → `metal/src/matrix.metal`
- `cuda/src/batch_ntt_small.cu` → `metal/src/batch_ntt_small.metal`
- `cuda/src/mle_interpolate.cu` → `metal/src/mle_interpolate.metal`
- `cuda/src/device_info.cu` → (not needed, Metal device info via Rust API)
- `cuda/supra/ntt.cu` → `metal/src/ntt.metal`

### Also reference risc0 Metal shaders
- `/Users/jpw/github/risc0/risc0/sys/kernels/zkp/metal/ntt.metal` - for Metal NTT patterns
- `/Users/jpw/github/risc0/risc0/build_kernel/kernels/metal/fp.h` - for BabyBear in MSL

### Translation Rules
1. `__global__ void kernel_name(args)` → `kernel void kernel_name(args [[buffer(N)]], uint tid [[thread_position_in_grid]])`
2. `blockIdx.x * blockDim.x + threadIdx.x` → `thread_position_in_grid`
3. `__shared__ T arr[N]` → `threadgroup T arr[N]`
4. `__syncthreads()` → `threadgroup_barrier(mem_flags::mem_threadgroup)`
5. `__constant__` → `constant` address space or buffer argument
6. `atomicAdd(&x, val)` → `atomic_fetch_add_explicit(&x, val, memory_order_relaxed)`

### BabyBear Field Header (baby_bear.h)
- Prime: P = 2013265921 (same as risc0)
- Montgomery form with M = 0x88000001
- Must match the Rust-side BabyBear exactly

### Extension Field Header (baby_bear_ext.h)
- Degree 4, same as risc0 — can directly reference risc0's `fpext.h`
- Must match `BinomialExtensionField<BabyBear, 4>` from p3-baby-bear
- Irreducible polynomial: x^4 - 11 over BabyBear
- `D_EF = 4` as confirmed in `crates/stark-sdk/src/config/baby_bear_poseidon2.rs`

### Acceptance Criteria
- [ ] All header files compile with `xcrun metal -c test.metal -I include/`
- [ ] Field arithmetic matches CPU implementation (verify constants)
- [ ] All core .metal files compile
- [ ] Kernel function names match what FFI layer expects

---

## Agent C: Phase 2B - Crypto + Protocol Kernels

### Objective
Write MSL kernels for Poseidon2, sumcheck, WHIR, prefix scan, logup/zerocheck, and sponge.

### Reference Files
Read and translate:
- `cuda/src/merkle_tree.cu` → `metal/src/merkle_tree.metal`
- `cuda/src/sponge.cu` → `metal/src/sponge.metal`
- `cuda/src/sumcheck.cu` → `metal/src/sumcheck.metal`
- `cuda/src/prefix.cu` → `metal/src/prefix.metal`
- `cuda/src/whir.cu` → `metal/src/whir.metal`
- `cuda/src/stacked_reduction.cu` → `metal/src/stacked_reduction.metal`
- `cuda/src/logup_zerocheck/*.cu` → `metal/src/logup_zerocheck/*.metal`
- `cuda/include/monomial.cuh` → `metal/include/monomial.h`
- `cuda/include/sumcheck.cuh` → `metal/include/sumcheck.h`
- `cuda/include/codec.cuh` → `metal/include/codec.h`
- `cuda/include/dag_entry.cuh` → `metal/include/dag_entry.h`
- `cuda/include/eval_ctx.cuh` → `metal/include/eval_ctx.h`
- `cuda/include/frac_ext.cuh` → `metal/include/frac_ext.h`

### Also reference risc0 Poseidon2
- `/Users/jpw/github/risc0/risc0/sys/kernels/zkp/metal/poseidon2.metal` - Poseidon2 in MSL

### Poseidon2 Implementation Notes
- Width 16, rate 8 (matches this project's constants)
- Round constants must match `openvm-stark-sdk`'s Poseidon2BabyBear<16> exactly
- Internal matrix (M_INT) and external matrix (M_EXT) must be identical to CPU
- Full rounds: 4 + 4 = 8, partial rounds: check stark-sdk

### Acceptance Criteria
- [ ] All protocol .metal files compile
- [ ] Header files for monomial, DAG, codec structures are correct
- [ ] Poseidon2 round constants match CPU implementation
- [ ] All kernel function names are documented for FFI layer

---

## Agent D: Phase 3 - Build System

### Objective
Implement `build.rs` for the metal-backend crate that compiles all `.metal` files into a `.metallib`.

### Reference
- risc0's build system: `/Users/jpw/github/risc0/risc0/build_kernel/src/lib.rs`
- risc0's sys build: `/Users/jpw/github/risc0/risc0/sys/build.rs`

### Implementation

```rust
// build.rs
use std::process::Command;
use std::path::{Path, PathBuf};
use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        // Skip Metal compilation on non-macOS
        // Create empty metallib or feature-gate
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metal_src_dir = Path::new("metal/src");
    let metal_include_dir = Path::new("metal/include");

    // Find all .metal files
    let metal_files: Vec<_> = glob::glob("metal/src/**/*.metal")
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();

    // Rerun if any source changes
    for f in &metal_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    for entry in glob::glob("metal/include/**/*.h").unwrap().filter_map(|e| e.ok()) {
        println!("cargo:rerun-if-changed={}", entry.display());
    }

    // Stage 1: Compile .metal → .air (parallel)
    let air_files: Vec<PathBuf> = metal_files.iter().map(|src| {
        let air_name = src.file_stem().unwrap().to_str().unwrap();
        let air_path = out_dir.join(format!("{}.air", air_name));

        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal",
                   "-c", src.to_str().unwrap(),
                   "-I", metal_include_dir.to_str().unwrap(),
                   "-std=metal3.0",
                   "-o", air_path.to_str().unwrap()])
            .status()
            .expect("Failed to run Metal compiler. Is Xcode installed?");

        assert!(status.success(), "Metal compilation failed for {}", src.display());
        air_path
    }).collect();

    // Stage 2: Link .air → .metallib
    let metallib_path = out_dir.join("kernels.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air.to_str().unwrap());
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to run metallib linker");
    assert!(status.success(), "metallib linking failed");

    println!("cargo:rustc-env=METAL_KERNELS_PATH={}", metallib_path.display());
}
```

### Acceptance Criteria
- [ ] `build.rs` compiles all Metal shaders
- [ ] `METAL_KERNELS_PATH` env var is set
- [ ] `include_bytes!(env!("METAL_KERNELS_PATH"))` works in lib.rs
- [ ] Incremental builds work (rerun-if-changed)
- [ ] Clear error message if Metal compiler not available

---

## Agents E & F: Phase 4 - FFI / Dispatch Layer

### Shared Setup

**`src/metal/mod.rs`** must provide:
```rust
pub struct MetalKernels {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<String, metal::ComputePipelineState>,
}

pub enum KernelArg<'a> {
    Buffer(&'a metal::Buffer, u64),  // buffer + offset
    U32(u32),
    Null,
}
```

Each dispatch wrapper follows this pattern:
```rust
// Instead of:
//   extern "C" { fn _cuda_kernel(ptr: *mut F, len: u32) -> i32; }
//   CudaError::from_result(_cuda_kernel(buf.as_ptr(), buf.len() as u32))

// We do:
pub fn metal_kernel(
    ctx: &MetalKernels,
    buf: &MetalBuffer<F>,
    len: u32,
) -> Result<(), MetalError> {
    ctx.dispatch(
        "kernel_name",
        &[KernelArg::Buffer(buf.gpu_buffer(), 0), KernelArg::U32(len)],
        MTLSize::new(compute_grid_size(len), 1, 1),
        MTLSize::new(256, 1, 1),  // threadgroup size
    )
}
```

### Agent E Scope
Core FFI wrappers: `ntt.rs`, `batch_ntt_small.rs`, `matrix.rs`, `poly.rs`, `mle_interpolate.rs`, `device_info.rs`

### Agent F Scope
Protocol FFI wrappers: `merkle_tree.rs`, `sponge.rs`, sumcheck (in `mod.rs`), prefix (in `mod.rs`), `whir.rs`, `stacked_reduction.rs`, `logup_zerocheck.rs`

### Acceptance Criteria
- [ ] Every function in `cuda-backend/src/cuda/` has a Metal equivalent
- [ ] Function signatures use `MetalBuffer<T>` / `MetalKernels` consistently
- [ ] Grid/threadgroup sizes are computed correctly
- [ ] Error handling wraps Metal command buffer status

---

## Agents G & H: Phase 5 - Backend Implementation

### Approach
Copy each file from `cuda-backend/src/` and mechanically replace:
- `DeviceBuffer<T>` → `MetalBuffer<T>`
- `DeviceMatrix<T>` → `MetalMatrix<T>`
- `GpuBackend` → `MetalBackend`
- `GpuDevice` → `MetalDevice`
- `GpuProverConfig` → `MetalProverConfig`
- `cuda::kernel_name(args)` → `metal::kernel_name(ctx, args)`
- CUDA error handling → Metal error handling
- Stream sync → command buffer sync

### Agent G Scope
Core: `types.rs`, `error.rs`, `base.rs`, `metal_backend.rs`, `device.rs`, `engine.rs`, `data_transporter.rs`, `sponge.rs`, `pkey.rs`, `monomial.rs`

### Agent H Scope
Protocol: `ntt.rs`, `poly.rs`, `merkle_tree.rs`, `stacked_pcs.rs`, `stacked_reduction.rs`, `whir.rs`, `sumcheck.rs`, `logup_zerocheck/` (all files)

### Acceptance Criteria
- [ ] `cargo check -p openvm-metal-backend` compiles
- [ ] All trait implementations present
- [ ] `BabyBearPoseidon2MetalEngine` implements `StarkEngine`

---

## Agent I: Phase 6 - Tests & Validation

### Tests to Create

1. **`tests/ntt_roundtrip.rs`** - Copy from cuda-backend, replace types
2. **`examples/keccakf.rs`** - Copy from cuda-backend, replace engine type
3. **Kernel unit tests** - For each kernel, test against CPU reference
4. **Cross-backend validation** - Prove with CPU, prove with Metal, compare

### Running Tests
```bash
# Unit tests
cargo test -p openvm-metal-backend

# NTT roundtrip (requires GPU)
cargo test -p openvm-metal-backend --test ntt_roundtrip -- --ignored

# Keccak example
cargo run -p openvm-metal-backend --example keccakf --release
```

### Acceptance Criteria
- [ ] `cargo test -p openvm-metal-backend` passes
- [ ] NTT roundtrip test passes
- [ ] Keccak example generates and verifies proof
- [ ] Cross-backend validation passes (CPU proof == Metal proof verified)
