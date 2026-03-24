//! GPU-accelerated transcript for the BabyBear-BN254 Poseidon2 configuration.
//!
//! This module provides [`MultiField32ChallengerGpu`], which mirrors the logic
//! of `MultiField32Challenger<BabyBear, Bn254Scalar, _, 3, 2>` while exposing
//! its state for host-to-device synchronization before GPU proof-of-work grinding.

use std::ffi::c_void;

use openvm_cuda_common::{copy::cuda_memcpy, d_buffer::DeviceBuffer, error::MemCopyError};
use openvm_stark_backend::FiatShamirTranscript;
use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::{
    default_babybear_bn254_poseidon2, BabyBearBn254Poseidon2Config, Bn254Scalar,
};
use p3_baby_bear::BabyBear;
use p3_field::{reduce_32, split_32, PrimeCharacteristicRing, PrimeField32};
use p3_symmetric::Permutation;

use crate::sponge::{GpuFiatShamirTranscript, GrindError};

/// Bn254 digest type: one BN254 scalar element.
type Digest = [Bn254Scalar; 1];

/// `num_f_elms = Bn254Scalar::bits() / 64`.
/// For the BN254 scalar field (254-bit prime), `254 / 64 = 3`.
const NUM_F_ELMS: usize = 3;

/// Sponge RATE in Bn254Scalar elements (2 per permutation call).
const RATE: usize = 2;

/// Sponge WIDTH in Bn254Scalar elements.
#[allow(dead_code)]
const WIDTH: usize = 3;

/// Max pending input values  = `NUM_F_ELMS * RATE = 6`.
const MAX_INPUT: usize = NUM_F_ELMS * RATE;

/// Max pending output values = `NUM_F_ELMS * WIDTH = 9`.
#[allow(dead_code)]
const MAX_OUTPUT: usize = NUM_F_ELMS * WIDTH;

// ---------------------------------------------------------------------------
// DeviceBn254SpongeState — must match `DeviceBn254SpongeState` in bn254_poseidon2.cu
// ---------------------------------------------------------------------------

/// Device-side BN254 sponge state for GPU grinding.
///
/// Layout must exactly match the CUDA struct `DeviceBn254SpongeState`:
/// ```text
/// struct DeviceBn254SpongeState {
///     Bn254Fr  sponge_state[3];   // 96 bytes
///     uint32_t input_buffer[6];   // 24 bytes
///     uint32_t input_len;         //  4 bytes
///     uint32_t output_buffer[9];  // 36 bytes
///     uint32_t output_len;        //  4 bytes
///     // + 4 bytes trailing padding = 168 bytes total
/// };
/// ```
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct DeviceBn254SpongeState {
    /// BN254 Poseidon2 sponge state — each element as `[u64; 4]` in Montgomery form.
    pub sponge_state: [[u64; 4]; 3], // 96 bytes
    /// Pending absorb values (canonical BabyBear u32); valid range `[0, input_len)`.
    pub input_buffer: [u32; 6], // 24 bytes
    /// Number of valid entries in `input_buffer`.
    pub input_len: u32, // 4 bytes
    /// Pending squeeze values (canonical BabyBear u32); valid range `[0, output_len)`.
    /// Index `output_len - 1` is the next value to be returned (LIFO, matches `Vec::pop`).
    pub output_buffer: [u32; 9], // 36 bytes
    /// Number of valid entries in `output_buffer`.
    pub output_len: u32, /* 4 bytes
                          * 4 bytes implicit trailing padding → total 168 bytes */
}

// Compile-time FFI safety: `Bn254Scalar` ↔ `[u64; 4]` conversion is sound only if
// size and alignment match.  `p3_bn254::Bn254` is a newtype `{ value: [u64; 4] }`
// without `#[repr(C)]`, so we guard against upstream layout changes here.
const _: () = assert!(
    std::mem::size_of::<Bn254Scalar>() == std::mem::size_of::<[u64; 4]>(),
    "Bn254Scalar must be 32 bytes (same as [u64; 4])"
);
const _: () = assert!(
    std::mem::align_of::<Bn254Scalar>() == std::mem::align_of::<[u64; 4]>(),
    "Bn254Scalar alignment must match [u64; 4]"
);
const _: () = assert!(
    std::mem::size_of::<DeviceBn254SpongeState>() == 168,
    "DeviceBn254SpongeState must be 168 bytes to match CUDA struct"
);

/// Extract the Montgomery-form `[u64; 4]` limbs from a `Bn254Scalar`.
///
/// # Safety
///
/// This reinterprets `Bn254Scalar` memory as `[u64; 4]` via a pointer cast,
/// which depends on layout compatibility. `Bn254Scalar` (`p3_bn254::Bn254`) is
/// a single-field newtype `{ value: [u64; 4] }` with identical size and
/// alignment (guarded by the const assertions above). If upstream ever changes
/// the struct layout (e.g. adds fields or changes repr), the const assertions
/// will fail at compile time.
fn bn254_scalar_to_raw(s: Bn254Scalar) -> [u64; 4] {
    // Safety: size and alignment are equal (const-asserted), and Bn254Scalar is
    // a single-field newtype over [u64; 4].
    unsafe { std::ptr::read((&s as *const Bn254Scalar).cast::<[u64; 4]>()) }
}

// ---------------------------------------------------------------------------
// MultiField32ChallengerGpu
// ---------------------------------------------------------------------------

/// GPU-accelerated transcript for the BabyBear-BN254 Poseidon2 proving system.
///
/// This is the GPU counterpart of
/// `openvm_stark_sdk::config::baby_bear_bn254_poseidon2::Transcript` (which wraps
/// `MultiField32Challenger<BabyBear, Bn254Scalar, Poseidon2Bn254<3>, 3, 2>`).
/// It replicates the challenger logic with **public state** so the sponge can be
/// serialised to a device buffer for CUDA grinding.
///
/// # State synchronization
///
/// Call [`sync_h2d`](Self::sync_h2d) before launching GPU grinding kernels.
/// After grinding, the host state is updated directly (observe witness + consume sample)
/// — no device-to-host sync is required.
#[derive(Debug)]
pub struct MultiField32ChallengerGpu {
    /// BN254 Poseidon2 sponge state (3 elements).
    sponge_state: [Bn254Scalar; 3],
    /// Pending absorb buffer (max `NUM_F_ELMS * RATE = 6` elements).
    input_buffer: Vec<BabyBear>,
    /// Pending squeeze buffer — newest last, consumed via `pop()` (max `NUM_F_ELMS * WIDTH = 9`).
    output_buffer: Vec<BabyBear>,
    /// Device buffer for H2D sync before GPU grinding.
    device: DeviceBuffer<DeviceBn254SpongeState>,
}

impl Default for MultiField32ChallengerGpu {
    fn default() -> Self {
        Self {
            sponge_state: [Bn254Scalar::default(); 3],
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            device: DeviceBuffer::new(),
        }
    }
}

impl Clone for MultiField32ChallengerGpu {
    fn clone(&self) -> Self {
        let mut new = Self {
            sponge_state: self.sponge_state,
            input_buffer: self.input_buffer.clone(),
            output_buffer: self.output_buffer.clone(),
            device: DeviceBuffer::new(),
        };
        if !self.device.is_empty() {
            let _ = new.sync_h2d();
        }
        new
    }
}

impl MultiField32ChallengerGpu {
    /// Create a new challenger with zeroed state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the device buffer has been allocated.
    pub fn is_device_allocated(&self) -> bool {
        !self.device.is_empty()
    }

    // --- Core sponge operations (match `MultiField32Challenger`) ---

    /// Absorb pending input into the sponge and refill the output buffer.
    ///
    /// Corresponds to `MultiField32Challenger::duplexing`.
    fn duplexing(&mut self) {
        debug_assert!(self.input_buffer.len() <= MAX_INPUT);

        // Overwrite sponge_state with packed input chunks.
        for (i, chunk) in self.input_buffer.chunks(NUM_F_ELMS).enumerate() {
            self.sponge_state[i] = reduce_32(chunk);
        }
        self.input_buffer.clear();

        // Apply the BN254 Poseidon2 permutation.
        default_babybear_bn254_poseidon2().permute_mut(&mut self.sponge_state);

        // Fill output_buffer from the permuted state.
        self.output_buffer.clear();
        for &pf_val in &self.sponge_state {
            let f_vals = split_32::<Bn254Scalar, BabyBear>(pf_val, NUM_F_ELMS);
            self.output_buffer.extend(f_vals);
        }
    }

    fn observe_inner(&mut self, value: BabyBear) {
        // Any buffered output is now stale.
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == MAX_INPUT {
            self.duplexing();
        }
    }

    fn sample_inner(&mut self) -> BabyBear {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing();
        }
        self.output_buffer
            .pop()
            .expect("output buffer must be non-empty after duplexing")
    }

    // --- Device synchronization ---

    fn ensure_device_allocated(&mut self) {
        if self.device.is_empty() {
            self.device = DeviceBuffer::with_capacity(1);
        }
    }

    /// Copy host sponge state to the device buffer.
    ///
    /// Call this before launching a GPU grinding kernel.
    pub fn sync_h2d(&mut self) -> Result<(), MemCopyError> {
        self.ensure_device_allocated();

        let mut device_state = DeviceBn254SpongeState::default();

        // Sponge state: extract Montgomery-form limbs.
        for (i, &s) in self.sponge_state.iter().enumerate() {
            device_state.sponge_state[i] = bn254_scalar_to_raw(s);
        }

        // Input buffer: canonical u32 values.
        for (i, &bb) in self.input_buffer.iter().enumerate() {
            device_state.input_buffer[i] = bb.as_canonical_u32();
        }
        device_state.input_len = self.input_buffer.len() as u32;

        // Output buffer: canonical u32 values (index 0 is oldest, index output_len-1 is newest).
        // Matches Vec::pop() semantics — CUDA kernel reads from [output_len-1] downward.
        for (i, &bb) in self.output_buffer.iter().enumerate() {
            device_state.output_buffer[i] = bb.as_canonical_u32();
        }
        device_state.output_len = self.output_buffer.len() as u32;

        unsafe {
            cuda_memcpy::<false, true>(
                self.device.as_mut_ptr() as *mut c_void,
                &device_state as *const DeviceBn254SpongeState as *const c_void,
                std::mem::size_of::<DeviceBn254SpongeState>(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl FiatShamirTranscript<BabyBearBn254Poseidon2Config> for MultiField32ChallengerGpu {
    fn observe(&mut self, value: BabyBear) {
        self.observe_inner(value);
    }

    fn sample(&mut self) -> BabyBear {
        self.sample_inner()
    }

    /// Decompose each BN254 digest element into `NUM_F_ELMS` BabyBear values and observe them.
    ///
    /// Matches `CanObserve<Hash<BabyBear, Bn254Scalar, 1>>` in `MultiField32Challenger`.
    fn observe_commit(&mut self, digest: Digest) {
        for pf_val in digest {
            let f_vals = split_32::<Bn254Scalar, BabyBear>(pf_val, NUM_F_ELMS);
            for f in f_vals {
                self.observe_inner(f);
            }
        }
    }
}

impl GpuFiatShamirTranscript<BabyBearBn254Poseidon2Config> for MultiField32ChallengerGpu {
    fn grind_gpu(&mut self, bits: usize) -> Result<BabyBear, GrindError> {
        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return Ok(BabyBear::ZERO);
        }

        // 1. Sync host state to device.
        self.sync_h2d()?;

        // 2. Run the BN254 grinding kernel.
        let witness_u32 = unsafe {
            crate::cuda::bn254_merkle_tree::bn254_sponge_grind(
                self.device.as_ptr(),
                bits as u32,
                BabyBear::ORDER_U32 - 1,
            )?
        };

        let witness = BabyBear::from_u32(witness_u32);

        // 3. Update host state: observe witness + consume one sample.
        self.observe_inner(witness);
        let _: BabyBear = self.sample_inner();

        Ok(witness)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use openvm_stark_backend::{p3_symmetric::CryptographicHasher, FiatShamirTranscript};
    use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::default_transcript;
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::MultiField32PaddingFreeSponge;

    use super::*;

    #[test]
    fn test_device_bn254_sponge_state_size() {
        assert_eq!(
            std::mem::size_of::<DeviceBn254SpongeState>(),
            168,
            "DeviceBn254SpongeState size mismatch with CUDA struct"
        );
    }

    #[test]
    fn test_device_bn254_sponge_state_align() {
        assert_eq!(
            std::mem::align_of::<DeviceBn254SpongeState>(),
            8,
            "DeviceBn254SpongeState must be 8-byte aligned"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Step {
        Observe(u32),
        ObserveCommit(u64),
        Sample,
    }

    fn assert_sample_matches_cpu(
        gpu: &mut MultiField32ChallengerGpu,
        cpu: &mut impl FiatShamirTranscript<BabyBearBn254Poseidon2Config>,
        context: &str,
    ) {
        let gpu_sample = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(gpu);
        let cpu_sample = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(cpu);
        assert_eq!(gpu_sample, cpu_sample, "{context}");
    }

    fn run_steps(steps: &[Step]) {
        let mut gpu = MultiField32ChallengerGpu::new();
        let mut cpu = default_transcript();

        for (idx, step) in steps.iter().copied().enumerate() {
            match step {
                Step::Observe(value) => {
                    let value = BabyBear::from_u32(value);
                    FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, value);
                    FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, value);
                }
                Step::ObserveCommit(value) => {
                    let digest: Digest = [Bn254Scalar::from_u64(value)];
                    FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(
                        &mut gpu, digest,
                    );
                    FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(
                        &mut cpu, digest,
                    );
                }
                Step::Sample => {
                    assert_sample_matches_cpu(
                        &mut gpu,
                        &mut cpu,
                        &format!("sample mismatch at step {idx}: {step:?}"),
                    );
                }
            }
        }
    }

    /// Verify that `MultiField32ChallengerGpu` produces the same samples as
    /// `openvm_stark_sdk::config::baby_bear_bn254_poseidon2::Transcript`.
    #[test]
    fn test_challenger_matches_transcript() {
        let mut gpu = MultiField32ChallengerGpu::new();
        let mut cpu = default_transcript();

        // Observe 10 values.
        for i in 0u32..10 {
            let val = BabyBear::from_u32(i * 13 + 7);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, val);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, val);
        }

        // Sample 5 values — both should match.
        for _ in 0..5 {
            let gpu_s: BabyBear =
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut gpu);
            let cpu_s: BabyBear =
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut cpu);
            assert_eq!(
                gpu_s, cpu_s,
                "sample mismatch between GPU and CPU challengers"
            );
        }

        // Observe a digest element.
        let digest: Digest = [Bn254Scalar::default()];
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(&mut gpu, digest);
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(&mut cpu, digest);

        // Sample 5 more values.
        for _ in 0..5 {
            let gpu_s: BabyBear =
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut gpu);
            let cpu_s: BabyBear =
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut cpu);
            assert_eq!(gpu_s, cpu_s, "sample mismatch after observe_commit");
        }
    }

    #[test]
    fn test_challenger_matches_transcript_for_all_absorb_prefix_lengths() {
        // Sweep prefixes around multiple absorb-capacity boundaries. This catches
        // off-by-one differences in when pending inputs trigger duplexing and when
        // fresh output blocks are generated after sampling drains the buffer.
        for num_observed in 0..=(MAX_INPUT * 2 + 1) {
            let mut gpu = MultiField32ChallengerGpu::new();
            let mut cpu = default_transcript();

            for i in 0..num_observed {
                let value = BabyBear::from_u32((i as u32).wrapping_mul(37).wrapping_add(11));
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, value);
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, value);
            }

            for sample_idx in 0..(MAX_OUTPUT * 2 + 3) {
                assert_sample_matches_cpu(
                    &mut gpu,
                    &mut cpu,
                    &format!(
                        "mismatch after observing {num_observed} values, sample #{sample_idx}"
                    ),
                );
            }
        }
    }

    #[test]
    fn test_challenger_clears_stale_outputs_when_observing_after_sampling() {
        let mut gpu = MultiField32ChallengerGpu::new();
        let mut cpu = default_transcript();

        for i in 0..5u32 {
            let value = BabyBear::from_u32(i * 17 + 3);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, value);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, value);
        }

        for sample_idx in 0..4 {
            assert_sample_matches_cpu(
                &mut gpu,
                &mut cpu,
                &format!("mismatch before clearing stale outputs at sample #{sample_idx}"),
            );
        }

        // Sampling leaves buffered outputs available. A subsequent observe must
        // invalidate them so future samples reflect the new transcript input.
        assert!(
            !gpu.output_buffer.is_empty(),
            "test setup should leave buffered outputs before the next observe"
        );

        let fresh = BabyBear::from_u32(999);
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, fresh);
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, fresh);

        assert!(
            gpu.output_buffer.is_empty(),
            "observing after sampling must invalidate stale buffered outputs"
        );
        assert_eq!(gpu.input_buffer, vec![fresh]);

        for sample_idx in 0..(MAX_OUTPUT + 2) {
            assert_sample_matches_cpu(
                &mut gpu,
                &mut cpu,
                &format!("mismatch after clearing stale outputs at sample #{sample_idx}"),
            );
        }
    }

    #[test]
    fn test_challenger_matches_transcript_for_observe_commit_edge_cases() {
        use Step::{Observe, ObserveCommit, Sample};

        // `observe_commit` decomposes one BN254 element into three BabyBear limbs.
        // These values intentionally straddle partial and full absorb buffers so
        // we verify the GPU path matches CPU chunking and duplex timing exactly.
        let steps = [
            ObserveCommit(u64::MAX - 17),
            Sample,
            ObserveCommit((1u64 << 32) + 9),
            Sample,
            Sample,
            Observe(7),
            Observe(8),
            ObserveCommit(u64::MAX - 0x1234),
            Sample,
            Sample,
            Sample,
            ObserveCommit(42),
            ObserveCommit((1u64 << 40) + 5),
            Sample,
            Sample,
            Sample,
            Sample,
            Observe(12345),
            Sample,
            Sample,
            Sample,
        ];

        run_steps(&steps);
    }

    #[test]
    fn test_challenger_matches_transcript_for_mixed_edge_case_sequence() {
        use Step::{Observe, ObserveCommit, Sample};

        // This sequence mixes empty-state sampling, partial absorbs, full-buffer
        // absorbs, and commit observations. It is meant to stress every place
        // where transcript state can flip between reusing buffered output and
        // forcing a new duplexing step.
        let steps = [
            Sample,
            Sample,
            Observe(1),
            Sample,
            Observe(2),
            Observe(3),
            Observe(4),
            Observe(5),
            Observe(6),
            Sample,
            Sample,
            Observe(7),
            Sample,
            Observe(8),
            Observe(9),
            Observe(10),
            Observe(11),
            Observe(12),
            Observe(13),
            Sample,
            Sample,
            Sample,
            ObserveCommit(u64::MAX),
            Observe(14),
            Observe(15),
            ObserveCommit((1u64 << 32) + 1),
            Sample,
            Sample,
            Sample,
            Sample,
            Sample,
            Sample,
            Observe(16),
            ObserveCommit(0),
            Sample,
            Sample,
            Sample,
            Sample,
            Sample,
        ];

        run_steps(&steps);
    }

    #[test]
    fn test_grind_gpu_matches_cpu_after_large_commit_observes() {
        let mut gpu = MultiField32ChallengerGpu::new();
        let mut cpu = default_transcript();
        let hash = MultiField32PaddingFreeSponge::<BabyBear, Bn254Scalar, _, 3, 16, 1>::new(
            default_babybear_bn254_poseidon2(),
        )
        .unwrap();

        for seed in [17u32, 113, 997] {
            let digest = hash.hash_slice(
                &(0..48)
                    .map(|i| BabyBear::from_u32(seed.wrapping_mul(i + 1).wrapping_add(i * 7)))
                    .collect::<Vec<_>>(),
            );
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(&mut gpu, digest);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(&mut cpu, digest);
        }

        for i in 0..7u32 {
            let value = BabyBear::from_u32(i.wrapping_mul(97).wrapping_add(11));
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, value);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, value);
        }

        let witness = gpu.grind_gpu(12).expect("GPU grind should succeed");
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, witness);
        assert!(
            (FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut cpu)
                .as_canonical_u32()
                & ((1 << 12) - 1))
                == 0,
            "CPU transcript rejected GPU witness"
        );

        for sample_idx in 0..12 {
            let gpu_sample = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut gpu);
            let cpu_sample = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut cpu);
            assert_eq!(
                gpu_sample, cpu_sample,
                "post-grind sample mismatch at index {sample_idx}"
            );
        }
    }
}
