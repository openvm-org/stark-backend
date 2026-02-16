# Metal Backend Design Document

## Overview

This document describes the design for `openvm-metal-backend`, an Apple Metal GPU implementation of `ProverBackend` for `BabyBearPoseidon2Config`. It is a parallel implementation to the existing `openvm-cuda-backend`, targeting Apple Silicon GPUs via the Metal Shading Language (MSL) and the `metal` Rust crate.

## Architecture

### Crate Structure

Two new crates, mirroring the CUDA crate structure:

```
crates/
├── metal-common/          # Metal GPU memory abstractions (analogous to cuda-common)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── error.rs       # MetalError, MemoryError, MemCopyError
│       ├── d_buffer.rs    # MetalBuffer<T> (analogous to DeviceBuffer<T>)
│       ├── copy.rs        # MemCopyH2D, MemCopyD2H, MemCopyD2D traits
│       ├── command.rs     # CommandQueue, CommandBuffer, Event wrappers
│       └── device.rs      # Metal device selection & info
│
├── metal-backend/         # ProverBackend implementation (analogous to cuda-backend)
│   ├── Cargo.toml
│   ├── build.rs           # Compiles .metal → .metallib
│   ├── metal/             # Metal Shading Language kernels
│   │   ├── include/
│   │   │   ├── baby_bear.h     # BabyBear field arithmetic (Montgomery form)
│   │   │   ├── baby_bear_ext.h # Extension field F_p^5
│   │   │   ├── poseidon2.h     # Poseidon2 constants & permutation
│   │   │   ├── utils.h         # Shared utility functions
│   │   │   ├── monomial.h      # Monomial computation structures
│   │   │   ├── codec.h         # Instruction encoding for DAG eval
│   │   │   └── dag_entry.h     # DAG node representation
│   │   └── src/
│   │       ├── ntt.metal
│   │       ├── poly.metal
│   │       ├── matrix.metal
│   │       ├── merkle_tree.metal
│   │       ├── sponge.metal
│   │       ├── sumcheck.metal
│   │       ├── prefix.metal
│   │       ├── whir.metal
│   │       ├── mle_interpolate.metal
│   │       ├── stacked_reduction.metal
│   │       ├── batch_ntt_small.metal
│   │       └── logup_zerocheck/
│   │           ├── batch_mle.metal
│   │           ├── batch_mle_monomial.metal
│   │           ├── gkr.metal
│   │           ├── gkr_input.metal
│   │           ├── mle.metal
│   │           ├── utils.metal
│   │           ├── zerocheck_round0.metal
│   │           └── logup_round0.metal
│   ├── src/
│   │   ├── lib.rs
│   │   ├── types.rs           # Type aliases (F, EF, SC, Challenger)
│   │   ├── error.rs           # ProverError types
│   │   ├── device.rs          # MetalDevice + MetalProverConfig
│   │   ├── engine.rs          # BabyBearPoseidon2MetalEngine
│   │   ├── metal_backend.rs   # MetalBackend marker + ProverBackend impl
│   │   ├── base.rs            # MetalMatrix<T>, MetalMatrixView<T>, Basis types
│   │   ├── ntt.rs             # NTT orchestration
│   │   ├── poly.rs            # MLE interpolation, PleMatrix
│   │   ├── sponge.rs          # DuplexSpongeGpu (Metal version)
│   │   ├── merkle_tree.rs     # MerkleTreeGpu
│   │   ├── stacked_pcs.rs     # Stacked commitment scheme
│   │   ├── stacked_reduction.rs
│   │   ├── whir.rs            # WHIR opening proof
│   │   ├── sumcheck.rs        # Sumcheck prover
│   │   ├── pkey.rs            # Metal proving key structures
│   │   ├── monomial.rs        # Monomial expansion
│   │   ├── data_transporter.rs
│   │   ├── metal/             # FFI bindings to Metal kernels
│   │   │   ├── mod.rs
│   │   │   ├── ntt.rs
│   │   │   ├── device_info.rs
│   │   │   ├── batch_ntt_small.rs
│   │   │   ├── poly.rs
│   │   │   ├── matrix.rs
│   │   │   ├── merkle_tree.rs
│   │   │   ├── sponge.rs
│   │   │   ├── whir.rs
│   │   │   ├── mle_interpolate.rs
│   │   │   ├── stacked_reduction.rs
│   │   │   └── logup_zerocheck.rs
│   │   └── logup_zerocheck/
│   │       ├── mod.rs
│   │       ├── batch_mle.rs
│   │       ├── batch_mle_monomial.rs
│   │       ├── round0.rs
│   │       ├── mle_round.rs
│   │       ├── gkr_input.rs
│   │       ├── fractional.rs
│   │       ├── fold_ple.rs
│   │       ├── rules/
│   │       └── errors.rs
│   ├── tests/
│   │   └── ntt_roundtrip.rs
│   └── examples/
│       └── keccakf.rs
```

### Trait Implementation Map

| Trait | CUDA Type | Metal Type |
|-------|-----------|------------|
| `ProverBackend` | `GpuBackend` | `MetalBackend` |
| `ProverDevice` | `GpuDevice` | `MetalDevice` |
| `TraceCommitter` | `GpuDevice` | `MetalDevice` |
| `MultiRapProver` | `GpuDevice` | `MetalDevice` |
| `OpeningProver` | `GpuDevice` | `MetalDevice` |
| `StarkEngine` | `BabyBearPoseidon2GpuEngine` | `BabyBearPoseidon2MetalEngine` |
| `FiatShamirTranscript` | `DuplexSpongeGpu` | `DuplexSpongeMetal` |
| `DeviceDataTransporter` | `GpuDevice` | `MetalDevice` |
| `MatrixDimensions` | `DeviceMatrix<T>` | `MetalMatrix<T>` |
| `MemCopyD2H` | `DeviceMatrix<T>` | `MetalMatrix<T>` |

### Type Mapping

```rust
// metal_backend.rs
pub struct MetalBackend;

impl ProverBackend for MetalBackend {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8; // 4
    type Val = F;                                  // BabyBear
    type Challenge = EF;                           // BinomialExtensionField<BabyBear, 4>
    type Commitment = Digest;                      // [F; 8]
    type Matrix = MetalMatrix<F>;                  // Metal buffer wrapper
    type PcsData = StackedPcsDataMetal<F, Digest>;
    type OtherAirData = AirDataMetal;
}
```

## Metal Integration Approach

### Rust-Metal Bridge

We use the `metal` crate (same as risc0) for Metal framework bindings:

```rust
// Dependencies
metal = "0.29"  // Metal framework Rust bindings
```

Key types from the `metal` crate:
- `metal::Device` - GPU device handle
- `metal::CommandQueue` - Command submission
- `metal::Library` - Compiled shader library
- `metal::ComputePipelineState` - Kernel pipeline
- `metal::Buffer` - GPU memory buffer

### Kernel Dispatch Pattern

Unlike CUDA's `extern "C"` FFI to compiled `.cu` kernels, Metal uses a runtime API:

```rust
// CUDA pattern (what we're replacing):
extern "C" { fn _kernel_name(args) -> i32; }

// Metal pattern (what we're implementing):
fn dispatch_kernel(
    pipeline: &ComputePipelineState,
    queue: &CommandQueue,
    buffers: &[(&metal::Buffer, u64)],
    constants: &[u32],
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
) {
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    for (i, (buf, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(i as u64, Some(*buf), *offset);
    }
    for (i, val) in constants.iter().enumerate() {
        encoder.set_bytes(
            (buffers.len() + i) as u64,
            4,
            val as *const u32 as *const _,
        );
    }

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
}
```

### Memory Model

Metal on Apple Silicon uses **unified memory** (shared CPU/GPU address space):

| Aspect | CUDA | Metal |
|--------|------|-------|
| Memory model | Discrete (H2D/D2H copies) | Unified (shared memory) |
| Allocation | `cudaMalloc` | `device.new_buffer(size, StorageModeShared)` |
| H2D transfer | `cudaMemcpy(..., H2D)` | Write directly to `buffer.contents()` pointer |
| D2H transfer | `cudaMemcpy(..., D2H)` | Read directly from `buffer.contents()` pointer |
| D2D transfer | `cudaMemcpy(..., D2D)` | Blit command encoder |
| Sync | Events, stream sync | Command buffer completion, MTLEvent |

This simplifies the memory management significantly:
- No explicit H2D/D2H copies needed for StorageModeShared buffers
- CPU can directly write to / read from Metal buffers
- Synchronization still required (must wait for GPU completion before CPU reads)

### MetalBuffer<T> Design

```rust
pub struct MetalBuffer<T> {
    buffer: metal::Buffer,     // Underlying Metal buffer
    len: usize,                // Number of elements of type T
    _marker: PhantomData<T>,
}

impl<T> MetalBuffer<T> {
    pub fn with_capacity(device: &metal::Device, len: usize) -> Self {
        let bytes = len * std::mem::size_of::<T>();
        let buffer = device.new_buffer(
            bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        Self { buffer, len, _marker: PhantomData }
    }

    pub fn as_ptr(&self) -> *const T {
        self.buffer.contents() as *const T
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    pub fn as_metal_buffer(&self) -> &metal::Buffer {
        &self.buffer
    }

    pub fn to_host(&self) -> Vec<T> where T: Copy {
        // Direct read from shared memory (after GPU sync)
        let slice = unsafe {
            std::slice::from_raw_parts(self.as_ptr(), self.len)
        };
        slice.to_vec()
    }

    pub fn copy_from_host(&self, data: &[T]) where T: Copy {
        // Direct write to shared memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.as_mut_ptr(),
                data.len(),
            );
        }
    }
}
```

### Build System

The build system compiles `.metal` files into a `.metallib` binary at build time:

```rust
// build.rs
fn main() {
    // Stage 1: Compile each .metal file to .air
    // xcrun -sdk macosx metal -c src.metal -I include/ -o out.air

    // Stage 2: Link all .air files into .metallib
    // xcrun -sdk macosx metallib *.air -o kernels.metallib

    // Stage 3: Emit cargo directives
    // println!("cargo:rustc-env=METAL_KERNELS_PATH=path/to/kernels.metallib");
}
```

At runtime, the metallib is embedded via `include_bytes!` and loaded:

```rust
const METAL_LIB: &[u8] = include_bytes!(env!("METAL_KERNELS_PATH"));

fn load_kernels(device: &metal::Device) -> HashMap<String, ComputePipelineState> {
    let library = device.new_library_with_data(METAL_LIB).unwrap();
    let mut kernels = HashMap::new();
    for name in KERNEL_NAMES {
        let func = library.get_function(name, None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&func).unwrap();
        kernels.insert(name.to_string(), pipeline);
    }
    kernels
}
```

## CUDA Kernel → Metal Shader Translation

### Field Arithmetic

BabyBear field (p = 2^31 - 2^27 + 1 = 2013265921) in Metal Shading Language:

```metal
// baby_bear.h
#include <metal_stdlib>
using namespace metal;

constant uint32_t P = 2013265921u;  // BabyBear prime
constant uint32_t M = 0x88000001u;  // Montgomery constant

struct Fp {
    uint32_t val;  // Montgomery form

    Fp() : val(0) {}
    Fp(uint32_t v) : val(v) {}

    Fp operator+(Fp rhs) const {
        uint32_t r = val + rhs.val;
        return Fp(r >= P ? r - P : r);
    }

    Fp operator-(Fp rhs) const {
        uint32_t r = val - rhs.val;
        return Fp(val < rhs.val ? r + P : r);
    }

    Fp operator*(Fp rhs) const {
        uint64_t o64 = uint64_t(val) * uint64_t(rhs.val);
        uint32_t low = uint32_t(o64);
        uint32_t red = M * low;
        o64 += uint64_t(red) * uint64_t(P);
        uint32_t ret = uint32_t(o64 >> 32);
        return Fp(ret >= P ? ret - P : ret);
    }
};
```

Extension field (degree 4, same as risc0):

```metal
// baby_bear_ext.h
constant uint32_t D_EF = 4;

struct FpExt {
    Fp elems[4];
    // Arithmetic defined by irreducible polynomial x^4 - 11 over BabyBear
};
```

### Kernel Translation Mapping

Each CUDA kernel maps to a Metal compute kernel:

| CUDA Kernel | Metal Kernel | Notes |
|-------------|--------------|-------|
| `_generate_all_twiddles` | `generate_all_twiddles` | NTT twiddle factors |
| `_bit_rev` / `_bit_rev_ext` | `bit_reverse` / `bit_reverse_ext` | Bit reversal permutation |
| `_ct_mixed_radix_narrow` | `ntt_forward_step` | NTT butterfly |
| `_algebraic_batch_matrices` | `algebraic_batch_matrices` | Matrix combination |
| `_eq_hypercube_stage_ext` | `eq_hypercube_stage_ext` | Equality polynomial |
| `_matrix_transpose_fp` | `matrix_transpose_fp` | Matrix transpose |
| `_poseidon2_compressing_row_hashes` | `poseidon2_row_hashes` | Merkle hashing |
| `_poseidon2_strided_compress_layer` | `poseidon2_compress_layer` | Merkle tree |
| `_sponge_grind` | `sponge_grind` | Proof-of-work |
| `_sumcheck_mle_round` | `sumcheck_mle_round` | MLE folding |
| `_fold_mle` / `_batch_fold_mle` | `fold_mle` / `batch_fold_mle` | Column folding |
| `_whir_algebraic_batch_traces` | `whir_algebraic_batch` | WHIR batching |
| ... (all other kernels) | ... | Direct translation |

### Thread Model Translation

| CUDA Concept | Metal Concept |
|--------------|---------------|
| Block | Threadgroup |
| Thread | Thread |
| Grid | Grid |
| `blockIdx.x` | `threadgroup_position_in_grid` |
| `threadIdx.x` | `thread_position_in_threadgroup` |
| `blockDim.x` | `threads_per_threadgroup` |
| `gridDim.x` | `threadgroups_per_grid` |
| `__shared__` | `threadgroup` address space |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__global__` | `kernel` function qualifier |

### Key Differences to Handle

1. **No constant memory**: CUDA uses `__constant__` for NTT twiddles. Metal uses `constant` address space or buffer arguments. We'll pass twiddles as buffer arguments.

2. **No warp-level primitives**: CUDA uses `__shfl_sync`, `__ballot_sync`. Metal uses SIMD group functions (`simd_shuffle`, `simd_ballot`). Apple Silicon has 32-wide SIMD groups (same as NVIDIA warps).

3. **Atomic operations**: CUDA `atomicAdd` → Metal `atomic_fetch_add_explicit` with `memory_order_relaxed`.

4. **Thread group size limits**: Apple Silicon M-series supports up to 1024 threads per threadgroup (same as CUDA blocks).

5. **No dynamic parallelism**: CUDA allows kernel launches from device code. Metal does not. Any recursive kernel patterns must be flattened to host-side dispatch loops.

## Synchronization Strategy

Metal command buffers are submitted and completed asynchronously:

```rust
struct MetalCommandContext {
    device: metal::Device,
    queue: metal::CommandQueue,
    kernels: HashMap<String, metal::ComputePipelineState>,
}

impl MetalCommandContext {
    /// Dispatch a compute kernel synchronously (wait for completion)
    fn dispatch_sync(&self, kernel_name: &str, args: &[KernelArg], count: u64) {
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        // ... set pipeline, args, dispatch ...
        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }

    /// Dispatch multiple kernels in a single command buffer (batched)
    fn dispatch_batch(&self, dispatches: &[Dispatch]) {
        let cmd_buffer = self.queue.new_command_buffer();
        for d in dispatches {
            let encoder = cmd_buffer.new_compute_command_encoder();
            // ... set pipeline, args, dispatch for each ...
            encoder.end_encoding();
        }
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    }
}
```

## Testing Strategy

All tests from `cuda-backend` are copied and adapted:

1. **Unit tests** (`tests/ntt_roundtrip.rs`): NTT forward → inverse roundtrip
2. **Integration test** (`examples/keccakf.rs`): Full Keccak-f proving pipeline
3. **Correctness validation**: For each kernel, compare Metal output against CPU reference implementation
4. **Cross-validation**: Run same proof with CPU backend and Metal backend, verify identical proof outputs

## Performance Considerations

1. **Unified memory advantage**: No H2D/D2H copy overhead on Apple Silicon
2. **Shared memory (threadgroup)**: Use for NTT butterfly and Poseidon2 rounds
3. **SIMD group operations**: Leverage 32-wide SIMD for reductions
4. **Command buffer batching**: Minimize command buffer overhead by batching multiple kernels
5. **Memory pressure**: Monitor with `device.currentAllocatedSize` and `recommendedMaxWorkingSetSize`
