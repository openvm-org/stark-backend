//! GPU-accelerated transcript for the BabyBear-BN254 Poseidon2 configuration.
//!
//! [`MultiFieldTranscriptGpu`] wraps the CPU `MultiFieldTranscript` and adds
//! host-to-device state synchronization for GPU proof-of-work grinding.

use std::ffi::c_void;

use openvm_cuda_common::{
    copy::cuda_memcpy_on, d_buffer::DeviceBuffer, error::MemCopyError, stream::DeviceContext,
};
use openvm_stark_backend::FiatShamirTranscript;
use openvm_stark_sdk::config::{
    baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2Config, Bn254Scalar, Transcript},
    bn254_poseidon2::default_bn254_poseidon2_width3,
};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

use crate::sponge::{validate_gpu_grind_bits, GpuFiatShamirTranscript, GrindError};

/// Bn254 digest type: one BN254 scalar element.
type Digest = [Bn254Scalar; 1];

// ---------------------------------------------------------------------------
// DeviceBn254SpongeState — must match `DeviceBn254SpongeState` in bn254_poseidon2.cu
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of a [`Transcript`]'s
/// [`snapshot`](openvm_stark_backend::transcript::multi_field::MultiFieldTranscript::snapshot)
/// for GPU grinding.
///
/// Populated in [`MultiFieldTranscriptGpu::sync_h2d`], then memcpy'd to the
/// device for CUDA grinding kernels.
///
/// Layout must exactly match the CUDA struct `DeviceBn254SpongeState`:
/// ```text
/// struct DeviceBn254SpongeState {
///     Bn254Fr  sponge_state[3];    // 96 bytes
///     uint32_t absorb_idx;         //  4 bytes
///     uint32_t sample_idx;         //  4 bytes
///     uint32_t observe_buf[8];     // 32 bytes
///     uint32_t observe_buf_len;    //  4 bytes
///     // total = 140 + 4 padding = 144 bytes (aligned to 8)
/// };
/// ```
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct DeviceBn254SpongeState {
    pub sponge_state: [[u64; 4]; 3], // 96 bytes
    pub absorb_idx: u32,             // 4 bytes
    pub sample_idx: u32,             // 4 bytes
    pub observe_buf: [u32; 8],       // 32 bytes
    pub observe_buf_len: u32,        // 4 bytes + 4 padding = 144 total
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
    std::mem::size_of::<DeviceBn254SpongeState>() == 144,
    "DeviceBn254SpongeState must be 144 bytes to match CUDA struct"
);

/// Extract the Montgomery-form `[u64; 4]` limbs from a `Bn254Scalar`.
///
/// # Safety
///
/// This reinterprets `Bn254Scalar` memory as `[u64; 4]` via a pointer cast,
/// which depends on layout compatibility. `Bn254Scalar` (`p3_bn254::Bn254`) is
/// a single-field newtype `{ value: [u64; 4] }` with identical size and
/// alignment (guarded by the const assertions above).
fn bn254_scalar_to_raw(s: Bn254Scalar) -> [u64; 4] {
    unsafe { std::ptr::read((&s as *const Bn254Scalar).cast::<[u64; 4]>()) }
}

// ---------------------------------------------------------------------------
// MultiFieldTranscriptGpu
// ---------------------------------------------------------------------------

/// GPU-accelerated transcript for the BabyBear-BN254 Poseidon2 proving system.
///
/// Wraps the CPU [`Transcript`] (a `MultiFieldTranscript`) and adds a device
/// buffer for GPU grinding. All observe/sample operations delegate to the inner
/// CPU transcript. Only [`grind_gpu`](GpuFiatShamirTranscript::grind_gpu)
/// touches the GPU: it snapshots the transcript state to the device, runs the
/// CUDA grinding kernel, then updates the host transcript with the result.
///
/// The device snapshot is intentionally not a full serialization of
/// [`Transcript`]: it omits the transcript's buffered sampled values
/// (`sample_buf`). This wrapper is therefore intended for the grinding flow,
/// where device-side execution observes a witness and consumes a fresh sample,
/// not for arbitrary continuation from a host transcript with buffered samples.
#[derive(Debug)]
pub struct MultiFieldTranscriptGpu {
    inner: Transcript,
    device: DeviceBuffer<DeviceBn254SpongeState>,
}

impl Default for MultiFieldTranscriptGpu {
    fn default() -> Self {
        Self {
            inner: Transcript::from(default_bn254_poseidon2_width3()),
            device: DeviceBuffer::new(),
        }
    }
}

impl Clone for MultiFieldTranscriptGpu {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            device: DeviceBuffer::new(),
        }
    }
}

impl MultiFieldTranscriptGpu {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_device_allocated(&mut self, ctx: &DeviceContext) {
        if self.device.is_empty() {
            self.device = DeviceBuffer::with_capacity_on(1, ctx);
        }
    }

    /// Snapshot the CPU transcript state to the device buffer.
    ///
    /// This copies the sponge state, sponge indices, and pending `observe_buf`,
    /// but it does not copy the transcript's buffered sampled values
    /// (`sample_buf`).
    ///
    /// Call this before launching a GPU grinding kernel. If the inner
    /// [`Transcript`] still has buffered samples from a prior host-side
    /// `sample()`, device-side sampling after this snapshot can diverge from the
    /// host transcript.
    pub fn sync_h2d(&mut self, ctx: &DeviceContext) -> Result<(), MemCopyError> {
        self.ensure_device_allocated(ctx);

        let mut ds = DeviceBn254SpongeState::default();

        // Sponge state
        for (i, &s) in self.inner.sponge_state().iter().enumerate() {
            ds.sponge_state[i] = bn254_scalar_to_raw(s);
        }
        ds.absorb_idx = self.inner.absorb_idx() as u32;
        ds.sample_idx = self.inner.sample_idx() as u32;

        // Observe buffer
        for (i, &bb) in self.inner.observe_buf().iter().enumerate() {
            ds.observe_buf[i] = bb.as_canonical_u32();
        }
        ds.observe_buf_len = self.inner.observe_buf().len() as u32;

        unsafe {
            cuda_memcpy_on::<false, true>(
                self.device.as_mut_ptr() as *mut c_void,
                &ds as *const DeviceBn254SpongeState as *const c_void,
                std::mem::size_of::<DeviceBn254SpongeState>(),
                ctx,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls — delegate to inner CPU transcript
// ---------------------------------------------------------------------------

impl FiatShamirTranscript<BabyBearBn254Poseidon2Config> for MultiFieldTranscriptGpu {
    fn observe(&mut self, value: BabyBear) {
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut self.inner, value);
    }

    fn sample(&mut self) -> BabyBear {
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut self.inner)
    }

    fn observe_commit(&mut self, digest: Digest) {
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(
            &mut self.inner,
            digest,
        );
    }
}

impl GpuFiatShamirTranscript<BabyBearBn254Poseidon2Config> for MultiFieldTranscriptGpu {
    fn grind_gpu(&mut self, bits: usize, ctx: &DeviceContext) -> Result<BabyBear, GrindError> {
        validate_gpu_grind_bits(bits)?;
        // Trivial case: 0 bits mean no PoW is required and any witness is valid.
        if bits == 0 {
            return Ok(BabyBear::ZERO);
        }

        // 1. Sync host state to device.
        self.sync_h2d(ctx)?;

        // 2. Run the BN254 grinding kernel.
        let witness_u32 = unsafe {
            crate::cuda::bn254_merkle_tree::bn254_sponge_grind(
                self.device.as_ptr(),
                bits as u32,
                BabyBear::ORDER_U32 - 1,
                ctx,
            )?
        };

        let witness = BabyBear::from_u32(witness_u32);

        // 3. Update host state: observe witness + consume one sample.
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut self.inner, witness);
        let _ = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut self.inner);

        Ok(witness)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use openvm_stark_backend::FiatShamirTranscript;
    use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::default_transcript;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    /// Exercises the CUDA grinding kernel end-to-end: the kernel must correctly
    /// implement observe + sample (packing, sponge permutation, base-p decomposition)
    /// to find a valid witness. We verify the witness against the CPU transcript.
    #[test]
    fn test_grind_gpu_witness_valid_on_cpu() {
        let bits = 8;

        // Test with several different transcript states to exercise partial observe buffers,
        // different sponge positions, etc.
        for num_observed in [0, 1, 3, 7, 8, 9, 15, 16, 17] {
            let mut gpu = MultiFieldTranscriptGpu::new();
            let mut cpu = default_transcript();

            for i in 0..num_observed {
                let val = BabyBear::from_u32((i as u32).wrapping_mul(41).wrapping_add(7));
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, val);
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, val);
            }

            let witness = gpu
                .grind_gpu(bits)
                .unwrap_or_else(|e| panic!("grind_gpu failed with {num_observed} observed: {e:?}"));

            // Verify the CUDA-found witness passes check_witness on the CPU transcript.
            assert!(
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::check_witness(
                    &mut cpu, bits, witness
                ),
                "CUDA witness {witness:?} invalid on CPU (observed {num_observed} values)"
            );
        }
    }
}
