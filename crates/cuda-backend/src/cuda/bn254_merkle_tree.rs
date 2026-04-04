//! Rust bindings for BN254 Poseidon2 Merkle-tree CUDA kernels and RC initialization.

use std::{
    collections::BTreeSet,
    sync::{Mutex, OnceLock},
};

use openvm_cuda_common::{
    common::{device_reset_epoch, get_device},
    d_buffer::DeviceBuffer,
    error::CudaError,
    stream::{cudaStreamPerThread, cudaStream_t},
};
use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::Bn254Scalar;

use crate::{
    bn254_sponge::DeviceBn254SpongeState,
    prelude::{EF, F},
    sponge::validate_gpu_grind_bits,
};

// ---------------------------------------------------------------------------
// BN254 Digest type (single Bn254Scalar = [u64; 4] in Montgomery form)
// ---------------------------------------------------------------------------

/// One BN254 scalar element — the digest type for BN254 Poseidon2 Merkle trees.
/// Layout matches `bn254_digest_t` in the CUDA kernel (32 bytes, align 8).
pub type Bn254Digest = [Bn254Scalar; 1];

// Compile-time FFI safety assertions.
// `Bn254Digest` is passed to CUDA as `Bn254Fr*` (32 bytes, 8-aligned).
const _: () = assert!(
    std::mem::size_of::<Bn254Digest>() == 32,
    "Bn254Digest must be 32 bytes to match CUDA bn254_digest_t"
);
const _: () = assert!(
    std::mem::align_of::<Bn254Digest>() == 8,
    "Bn254Digest must be 8-byte aligned to match CUDA Bn254Fr"
);
// `F` (BabyBear) is cast to `*const u32` at the FFI boundary.
const _: () = assert!(
    std::mem::size_of::<F>() == std::mem::size_of::<u32>(),
    "F must be 4 bytes to match CUDA uint32_t"
);
// `EF` (BinomialExtensionField<BabyBear, 4>) is cast to `*const u32` (stride of 4 u32s).
const _: () = assert!(
    std::mem::size_of::<EF>() == 4 * std::mem::size_of::<u32>(),
    "EF must be 16 bytes (4 x uint32_t) for CUDA"
);

// ---------------------------------------------------------------------------
// FFI-safe type alias: 32 bytes = 4 × u64, matching CUDA `Bn254Fr`.
// We use this in extern declarations instead of `Bn254Digest` to avoid
// passing non-repr(C) types across the FFI boundary.
// ---------------------------------------------------------------------------
type Bn254FrRaw = [u64; 4];

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

extern "C" {
    fn _init_bn254_poseidon2_rc(
        initial_rc: *const u64,  // [4*3*4] = 48 u64s
        partial_rc: *const u64,  // [56*4]  = 224 u64s
        terminal_rc: *const u64, // [4*3*4] = 48 u64s
        stream: cudaStream_t,
    ) -> i32;

    fn _init_bn254_poseidon2_rc_w2(
        initial_rc: *const u64,  // [3*2*4] = 24 u64s
        partial_rc: *const u64,  // [50*4]  = 200 u64s
        terminal_rc: *const u64, // [3*2*4] = 24 u64s
        stream: cudaStream_t,
    ) -> i32;

    fn _bn254_poseidon2_compressing_row_hashes(
        out: *mut Bn254FrRaw,
        matrix: *const u32, // F = BabyBear, repr(transparent) over u32
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _bn254_poseidon2_compressing_row_hashes_ext(
        out: *mut Bn254FrRaw,
        matrix: *const u32, // EF = [BabyBear; 4], each BabyBear is repr(transparent) over u32
        width: usize,
        query_stride: usize,
        log_rows_per_query: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _bn254_poseidon2_adjacent_compress_layer(
        output: *mut Bn254FrRaw,
        prev_layer: *const Bn254FrRaw,
        output_size: usize,
        stream: cudaStream_t,
    ) -> i32;

    fn _bn254_sponge_grind(
        init_state: *const DeviceBn254SpongeState,
        bits: u32,
        min_witness: u32,
        max_witness: u32,
        result: *mut u32,
        stream: cudaStream_t,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// Montgomery multiplication helper (mirrors CUDA imr / bn254_monty_mul)
// ---------------------------------------------------------------------------

const BN254_PRIME: [u64; 4] = [
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
];
const BN254_MU: u64 = 0x3d1e0a6c10000001;
const BN254_R_SQ: [u64; 4] = [
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
];

fn mul_small(lhs: [u64; 4], rhs: u64) -> (u64, [u64; 4]) {
    let mut acc = (lhs[0] as u128) * (rhs as u128);
    let low = acc as u64;
    acc >>= 64;
    let mut high = [0u64; 4];
    for i in 1..4 {
        acc += (lhs[i] as u128) * (rhs as u128);
        high[i - 1] = acc as u64;
        acc >>= 64;
    }
    high[3] = acc as u64;
    (low, high)
}

fn mul_small_and_acc(lhs: [u64; 4], rhs: u64, add: [u64; 4]) -> (u64, [u64; 4]) {
    let mut acc = (lhs[0] as u128) * (rhs as u128) + (add[0] as u128);
    let low = acc as u64;
    acc >>= 64;
    let mut high = [0u64; 4];
    for i in 1..4 {
        acc += (lhs[i] as u128) * (rhs as u128) + (add[i] as u128);
        high[i - 1] = acc as u64;
        acc >>= 64;
    }
    high[3] = acc as u64;
    (low, high)
}

fn imr(acc0: u64, acc: [u64; 4]) -> [u64; 4] {
    let t = acc0.wrapping_mul(BN254_MU);
    let (_, u) = mul_small(BN254_PRIME, t);
    let mut sub = [0u64; 4];
    let mut borrow: u64 = 0;
    for i in 0..4 {
        let (d, b1) = acc[i].overflowing_sub(u[i]);
        let (d2, b2) = d.overflowing_sub(borrow);
        sub[i] = d2;
        borrow = (b1 as u64) + (b2 as u64);
    }
    if borrow != 0 {
        let mut carry: u64 = 0;
        let mut result = [0u64; 4];
        for i in 0..4 {
            let t = (sub[i] as u128) + (BN254_PRIME[i] as u128) + (carry as u128);
            result[i] = t as u64;
            carry = (t >> 64) as u64;
        }
        result
    } else {
        sub
    }
}

/// Montgomery multiplication: `lhs * rhs * R^{-1} mod P`.
pub fn monty_mul_bn254(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    let (acc0, acc) = mul_small(lhs, rhs[0]);
    let res0 = imr(acc0, acc);

    let (acc0, acc) = mul_small_and_acc(lhs, rhs[1], res0);
    let res1 = imr(acc0, acc);

    let (acc0, acc) = mul_small_and_acc(lhs, rhs[2], res1);
    let res2 = imr(acc0, acc);

    let (acc0, acc) = mul_small_and_acc(lhs, rhs[3], res2);
    imr(acc0, acc)
}

/// Convert a canonical [u64; 4] BN254 value into Montgomery form.
pub fn canonical_to_monty_bn254(canonical: [u64; 4]) -> [u64; 4] {
    monty_mul_bn254(BN254_R_SQ, canonical)
}

/// Convert a `p3_bn254::Bn254` element to Montgomery-form `[u64; 4]` for CUDA.
fn p3_bn254_to_monty(elem: &Bn254Scalar) -> [u64; 4] {
    use p3_field::PrimeField;
    let bytes = elem.as_canonical_biguint().to_bytes_le();
    let mut canonical = [0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i >= 4 {
            break;
        }
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        canonical[i] = u64::from_le_bytes(buf);
    }
    canonical_to_monty_bn254(canonical)
}

fn flatten_external_rc<const WIDTH: usize>(round_constants: &[[Bn254Scalar; WIDTH]]) -> Vec<u64> {
    round_constants
        .iter()
        .flat_map(|rc| rc.iter())
        .flat_map(p3_bn254_to_monty)
        .collect()
}

fn flatten_internal_rc(round_constants: &[Bn254Scalar]) -> Vec<u64> {
    round_constants.iter().flat_map(p3_bn254_to_monty).collect()
}

// ---------------------------------------------------------------------------
// Round-constant initialization
// ---------------------------------------------------------------------------

/// Compute width-3 BN254 Poseidon2 round constants for CUDA.
/// Returns (initial_flat, partial_flat, terminal_flat) as flat u64 vectors in Montgomery form.
fn compute_bn254_rc_w3_constants() -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    use openvm_stark_sdk::config::bn254_poseidon2::default_bn254_poseidon2_width3_constants;

    let constants = default_bn254_poseidon2_width3_constants();
    (
        flatten_external_rc(constants.initial_external_rc()),
        flatten_internal_rc(constants.internal_rc()),
        flatten_external_rc(constants.terminal_external_rc()),
    )
}

/// Compute width-2 BN254 Poseidon2 round constants for CUDA.
/// Returns (initial_flat, partial_flat, terminal_flat) as flat u64 vectors in Montgomery form.
fn compute_bn254_rc_w2_constants() -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    use openvm_stark_sdk::config::bn254_poseidon2::default_bn254_poseidon2_width2_constants;

    let constants = default_bn254_poseidon2_width2_constants();
    (
        flatten_external_rc(constants.initial_external_rc()),
        flatten_internal_rc(constants.internal_rc()),
        flatten_external_rc(constants.terminal_external_rc()),
    )
}

/// Initialize BN254 Poseidon2 round constants on the GPU (called once at startup).
/// This uploads both width-3 (for leaf hashing) and width-2 (for compression) constants.
pub fn init_bn254_poseidon2_rc() -> Result<(), CudaError> {
    static RC_W3_CONSTANTS: OnceLock<(Vec<u64>, Vec<u64>, Vec<u64>)> = OnceLock::new();
    static RC_W2_CONSTANTS: OnceLock<(Vec<u64>, Vec<u64>, Vec<u64>)> = OnceLock::new();
    static INIT: OnceLock<Mutex<BTreeSet<(i32, u64)>>> = OnceLock::new();

    let device_key = (get_device()?, device_reset_epoch());
    let initialized = INIT.get_or_init(|| Mutex::new(BTreeSet::<(i32, u64)>::new()));
    let mut initialized = initialized.lock().unwrap();
    if initialized.contains(&device_key) {
        return Ok(());
    }

    // Width-3 constants (leaf hashing)
    let (initial, partial, terminal) = RC_W3_CONSTANTS.get_or_init(compute_bn254_rc_w3_constants);
    let code = unsafe {
        _init_bn254_poseidon2_rc(
            initial.as_ptr(),
            partial.as_ptr(),
            terminal.as_ptr(),
            cudaStreamPerThread,
        )
    };
    CudaError::from_result(code)?;

    // Width-2 constants (compression)
    let (initial_w2, partial_w2, terminal_w2) =
        RC_W2_CONSTANTS.get_or_init(compute_bn254_rc_w2_constants);
    let code = unsafe {
        _init_bn254_poseidon2_rc_w2(
            initial_w2.as_ptr(),
            partial_w2.as_ptr(),
            terminal_w2.as_ptr(),
            cudaStreamPerThread,
        )
    };
    CudaError::from_result(code)?;

    initialized.insert(device_key);
    Ok(())
}

// ---------------------------------------------------------------------------
// Public safe wrappers
// ---------------------------------------------------------------------------

/// Computes BN254 Poseidon2 row hashes for a BabyBear matrix.
///
/// # Safety
/// - `out` must have length `>= query_stride`.
/// - `matrix` must have length `>= width * (query_stride << log_rows_per_query)`.
pub unsafe fn bn254_poseidon2_compressing_row_hashes(
    out: &mut DeviceBuffer<Bn254Digest>,
    matrix: &DeviceBuffer<F>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    init_bn254_poseidon2_rc()?;
    super::merkle_tree::validate_merkle_log_rows_per_query(log_rows_per_query)?;
    CudaError::from_result(_bn254_poseidon2_compressing_row_hashes(
        out.as_mut_ptr().cast::<Bn254FrRaw>(),
        matrix.as_ptr().cast::<u32>(),
        width,
        query_stride,
        log_rows_per_query,
        stream,
    ))
}

/// Computes BN254 Poseidon2 row hashes for an extension-field matrix.
///
/// # Safety
/// - `out` must have length `>= query_stride`.
/// - `matrix` must have length `>= width * (query_stride << log_rows_per_query)`.
pub unsafe fn bn254_poseidon2_compressing_row_hashes_ext(
    out: &mut DeviceBuffer<Bn254Digest>,
    matrix: &DeviceBuffer<EF>,
    width: usize,
    query_stride: usize,
    log_rows_per_query: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    init_bn254_poseidon2_rc()?;
    super::merkle_tree::validate_merkle_log_rows_per_query(log_rows_per_query)?;
    CudaError::from_result(_bn254_poseidon2_compressing_row_hashes_ext(
        out.as_mut_ptr().cast::<Bn254FrRaw>(),
        matrix.as_ptr().cast::<u32>(),
        width,
        query_stride,
        log_rows_per_query,
        stream,
    ))
}

/// Compresses adjacent pairs of BN254 digests to produce the next Merkle layer.
///
/// # Safety
/// - `output` must have length `>= output_size`.
/// - `prev_layer` must have length `>= output_size * 2`.
pub unsafe fn bn254_poseidon2_adjacent_compress_layer(
    output: &mut DeviceBuffer<Bn254Digest>,
    prev_layer: &DeviceBuffer<Bn254Digest>,
    output_size: usize,
    stream: cudaStream_t,
) -> Result<(), CudaError> {
    init_bn254_poseidon2_rc()?;
    CudaError::from_result(_bn254_poseidon2_adjacent_compress_layer(
        output.as_mut_ptr().cast::<Bn254FrRaw>(),
        prev_layer.as_ptr().cast::<Bn254FrRaw>(),
        output_size,
        stream,
    ))
}

/// Launch the BN254 sponge grinding kernel.
///
/// # Safety
/// - `init_state` must point to valid device memory.
pub unsafe fn bn254_sponge_grind(
    init_state: *const DeviceBn254SpongeState,
    bits: u32,
    max_witness: u32,
    stream: cudaStream_t,
) -> Result<u32, crate::sponge::GrindError> {
    use openvm_cuda_common::copy::{MemCopyD2H, MemCopyH2D};

    init_bn254_poseidon2_rc()?;
    validate_gpu_grind_bits(bits as usize)?;
    let mut d_result: DeviceBuffer<u32> = DeviceBuffer::with_capacity(1);
    [u32::MAX].copy_to(&mut d_result)?;

    for start in (0..=max_witness).step_by(1 << bits) {
        CudaError::from_result(_bn254_sponge_grind(
            init_state,
            bits,
            start,
            max_witness,
            d_result.as_mut_ptr(),
            stream,
        ))?;

        let result = d_result.to_host()?[0];
        if result < u32::MAX {
            return Ok(result);
        }
    }
    Err(crate::sponge::GrindError::WitnessNotFound)
}
