//! GPU-accelerated duplex sponge with host/device state synchronization.
//!
//! This module provides [`DuplexSpongeGpu`], a transcript implementation that maintains
//! state on both host and device with explicit synchronization methods.

use std::ffi::c_void;

use openvm_cuda_common::{
    copy::cuda_memcpy,
    d_buffer::DeviceBuffer,
    error::{CudaError, MemCopyError},
};
use openvm_stark_backend::{
    p3_challenger::{CanObserve, CanSample},
    FiatShamirTranscript,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::poseidon2_perm;
use p3_baby_bear::default_babybear_poseidon2_16;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_symmetric::Permutation;

use crate::types::{Challenger, Digest, CHUNK, F, SC, WIDTH};

/// Device-side sponge state, matching the CUDA `DeviceSpongeState` struct.
///
/// This struct is `#[repr(C)]` to ensure ABI compatibility with the CUDA kernel.
/// The state layout matches the Poseidon2 duplex sponge with overwrite mode.
///
/// This struct implements the same logic as `DuplexSponge` from openvm_stark_backend,
/// but with public fields so we can sync state to/from GPU.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DeviceSpongeState {
    /// Full Poseidon2 state (WIDTH = 16 elements)
    pub state: [F; WIDTH],
    /// Current absorb position (0 <= absorb_idx < CHUNK)
    pub absorb_idx: u32,
    /// Current sample position (0 <= sample_idx <= CHUNK)
    pub sample_idx: u32,
}

impl Default for DeviceSpongeState {
    fn default() -> Self {
        Self {
            state: [F::default(); WIDTH],
            absorb_idx: 0,
            sample_idx: 0,
        }
    }
}

impl DeviceSpongeState {
    /// Observe a value into the sponge (absorb phase).
    ///
    /// This matches the behavior of `DuplexSponge::observe`.
    #[inline]
    pub fn observe(&mut self, value: F) {
        self.state[self.absorb_idx as usize] = value;
        self.absorb_idx += 1;
        if self.absorb_idx == CHUNK as u32 {
            poseidon2_perm().permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK as u32;
        }
    }

    /// Sample a value from the sponge (squeeze phase).
    ///
    /// This matches the behavior of `DuplexSponge::sample`.
    #[inline]
    pub fn sample(&mut self) -> F {
        if self.absorb_idx != 0 || self.sample_idx == 0 {
            poseidon2_perm().permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK as u32;
        }
        self.sample_idx -= 1;
        self.state[self.sample_idx as usize]
    }
}

impl FiatShamirTranscript<SC> for DeviceSpongeState {
    #[inline]
    fn observe(&mut self, value: F) {
        DeviceSpongeState::observe(self, value);
    }

    #[inline]
    fn sample(&mut self) -> F {
        DeviceSpongeState::sample(self)
    }

    #[inline]
    fn observe_commit(&mut self, digest: Digest) {
        for x in digest {
            self.observe(x);
        }
    }
}

/// GPU-accelerated duplex sponge that maintains state on both host and device.
///
/// The host-side state uses [`DeviceSpongeState`] which matches the behavior of
/// `DuplexSponge` from openvm_stark_backend (and `p3_challenger::DuplexChallenger`).
/// The device-side state is stored in GPU memory for CUDA kernel operations.
///
/// # State Synchronization
///
/// The host and device states are **independent** and must be explicitly synchronized:
/// - Use [`sync_h2d`](Self::sync_h2d) to copy host state to device before GPU operations
/// - Use [`sync_d2h`](Self::sync_d2h) to copy device state back to host after GPU operations
///
/// # Usage Example
///
/// ```ignore
/// let mut sponge = DuplexSpongeGpu::default();
///
/// // Do some host operations
/// sponge.observe(some_value);
/// let challenge = sponge.sample();
///
/// // Sync to device before GPU grinding
/// sponge.sync_h2d()?;
///
/// // ... GPU grinding kernel runs ...
///
/// // Sync back after GPU modifies state
/// sponge.sync_d2h()?;
///
/// // Continue with host operations
/// let next_challenge = sponge.sample();
/// ```
#[derive(Debug)]
pub struct DuplexSpongeGpu {
    /// Host-side structure
    host: Challenger,
    /// Device-side state buffer (allocated lazily on first sync)
    device: DeviceBuffer<DeviceSpongeState>,
}

impl Default for DuplexSpongeGpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for DuplexSpongeGpu {
    fn clone(&self) -> Self {
        let mut new = Self {
            host: self.host.clone(),
            device: DeviceBuffer::new(),
        };
        // Sync cloned host state to device if device was allocated
        if !self.device.is_empty() {
            let _ = new.sync_h2d();
        }
        new
    }
}

impl DuplexSpongeGpu {
    /// Create a new GPU-accelerated duplex sponge with default (zeroed) state.
    pub fn new() -> Self {
        Self {
            host: Challenger::new(default_babybear_poseidon2_16()),
            device: DeviceBuffer::new(),
        }
    }

    /// Returns true if the device buffer has been allocated.
    pub fn is_device_allocated(&self) -> bool {
        !self.device.is_empty()
    }

    /// Ensure the device buffer is allocated.
    fn ensure_device_allocated(&mut self) {
        if self.device.is_empty() {
            self.device = DeviceBuffer::with_capacity(1);
        }
    }

    /// Synchronize state from host to device (H2D memcpy).
    ///
    /// Call this before running GPU kernels that read/modify the sponge state.
    ///
    /// This converts from `DuplexChallenger`'s representation (with buffered input/output)
    /// to `DeviceSpongeState`'s representation (with indices pointing into state).
    pub fn sync_h2d(&mut self) -> Result<(), MemCopyError> {
        self.ensure_device_allocated();

        // Convert DuplexChallenger state to DeviceSpongeState format:
        // - DuplexChallenger buffers input values before writing to sponge_state
        // - DeviceSpongeState writes directly to state[absorb_idx]
        // We need to overlay the input_buffer onto state[0..len]

        let mut device_state = DeviceSpongeState {
            state: self.host.sponge_state,
            absorb_idx: self.host.input_buffer.len() as u32,
            sample_idx: self.host.output_buffer.len() as u32,
        };

        // Overlay pending input_buffer values onto the beginning of state
        // (DuplexChallenger writes to state[0..N] during duplexing, so pending
        // values that haven't been duplexed yet need to be placed there)
        for (i, &val) in self.host.input_buffer.iter().enumerate() {
            device_state.state[i] = val;
        }

        // SAFETY: Copying a single DeviceSpongeState from host to device
        // - Both pointers are valid and properly aligned
        // - The size matches the struct size
        unsafe {
            cuda_memcpy::<false, true>(
                self.device.as_mut_ptr() as *mut c_void,
                &device_state as *const DeviceSpongeState as *const c_void,
                std::mem::size_of::<DeviceSpongeState>(),
            )
        }
    }

    /// Get a pointer to the device state buffer.
    ///
    /// Returns `None` if the device buffer hasn't been allocated yet.
    /// Call [`sync_h2d`](Self::sync_h2d) to allocate and initialize the device buffer.
    pub fn device_ptr(&self) -> Option<*const DeviceSpongeState> {
        if self.device.is_empty() {
            None
        } else {
            Some(self.device.as_ptr())
        }
    }

    /// Get a mutable pointer to the device state buffer.
    ///
    /// Returns `None` if the device buffer hasn't been allocated yet.
    /// Call [`sync_h2d`](Self::sync_h2d) to allocate and initialize the device buffer.
    pub fn device_ptr_mut(&mut self) -> Option<*mut DeviceSpongeState> {
        if self.device.is_empty() {
            None
        } else {
            Some(self.device.as_mut_ptr())
        }
    }

    /// Perform GPU-accelerated grinding to find a proof-of-work witness.
    ///
    /// This syncs state to device, runs the grinding kernel, and updates
    /// the host state with the witness.
    ///
    /// # Arguments
    /// * `bits` - Number of bits that must be zero in the sampled value
    ///
    /// # Returns
    ///
    /// The PoW witness value that satisfies `sample_bits(bits) == 0` after observing it.
    ///
    /// # Note
    ///
    /// After this call, the host state will have observed the witness and sampled,
    /// matching the state after calling `check_witness(bits, witness)`.
    pub fn grind_gpu(&mut self, bits: usize) -> Result<F, GrindError> {
        // 1. Sync host state to device
        self.sync_h2d()?;

        // 2. Launch grinding kernel
        let witness_u32 = unsafe {
            crate::cuda::sponge::sponge_grind(self.device.as_ptr(), bits as u32, F::ORDER_U32 - 1)?
        };

        let witness = F::from_u32(witness_u32);

        // 3. Update host state to match (observe the witness + sample)
        // This is cheaper than syncing the full state back from device
        debug_assert!(self.clone().check_witness(bits, witness));
        self.host.observe(witness);
        let _: F = self.host.sample(); // Consume the sample to advance state

        Ok(witness)
    }
}

/// Error type for GPU grinding operations.
#[derive(Debug, thiserror::Error)]
pub enum GrindError {
    #[error("Memory copy error: {0}")]
    MemCopy(#[from] MemCopyError),

    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    #[error("Failed to find PoW witness within search space")]
    WitnessNotFound,
}

impl FiatShamirTranscript<SC> for DuplexSpongeGpu {
    #[inline]
    fn observe(&mut self, value: F) {
        self.host.observe(value);
    }

    #[inline]
    fn sample(&mut self) -> F {
        self.host.sample()
    }

    #[inline]
    fn observe_commit(&mut self, digest: Digest) {
        for x in digest {
            self.observe(x);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSponge;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    #[test]
    fn test_device_sponge_state_size() {
        // Verify the struct size is what we expect for FFI
        let expected_size = std::mem::size_of::<[F; WIDTH]>() // state
            + std::mem::size_of::<u32>() // absorb_idx
            + std::mem::size_of::<u32>(); // sample_idx

        assert_eq!(
            std::mem::size_of::<DeviceSpongeState>(),
            expected_size,
            "DeviceSpongeState size mismatch - check repr(C) and padding"
        );
    }

    #[test]
    fn test_device_sponge_state_alignment() {
        // Verify alignment for CUDA compatibility
        assert!(
            std::mem::align_of::<DeviceSpongeState>() >= 4,
            "DeviceSpongeState should be at least 4-byte aligned"
        );
    }

    #[test]
    fn test_default_state() {
        let state = DeviceSpongeState::default();
        assert_eq!(state.absorb_idx, 0);
        assert_eq!(state.sample_idx, 0);
        for elem in state.state.iter() {
            assert_eq!(*elem, F::default());
        }
    }

    #[test]
    fn test_sponge_gpu_new() {
        let sponge = DuplexSpongeGpu::new();
        assert!(!sponge.is_device_allocated());
    }

    #[test]
    fn test_device_sponge_state_matches_duplex_sponge() {
        // Verify our implementation matches DuplexSponge exactly
        let mut device_state = DeviceSpongeState::default();
        let mut duplex_sponge = DuplexSponge::default();

        // Test observe/sample sequence
        for i in 0..20 {
            let val = F::from_u32(i * 42 + 17);
            device_state.observe(val);
            FiatShamirTranscript::observe(&mut duplex_sponge, val);
        }

        for _ in 0..10 {
            let device_sample = device_state.sample();
            let duplex_sample = duplex_sponge.sample();
            assert_eq!(device_sample, duplex_sample);
        }

        // Interleaved observe/sample
        for i in 0..5 {
            let val = F::from_u32(i * 100);
            device_state.observe(val);
            FiatShamirTranscript::observe(&mut duplex_sponge, val);

            let device_sample = device_state.sample();
            let duplex_sample = duplex_sponge.sample();
            assert_eq!(device_sample, duplex_sample);
        }

        // Many samples in a row
        for _ in 0..15 {
            let device_sample = device_state.sample();
            let duplex_sample = duplex_sponge.sample();
            assert_eq!(device_sample, duplex_sample);
        }
    }

    #[test]
    fn test_sponge_gpu_uses_host_transcript() {
        let mut gpu_sponge = DuplexSpongeGpu::default();
        let mut cpu_sponge = DuplexSponge::default();

        // Test that host operations match DuplexSponge
        for i in 0..10 {
            let val = F::from_u32(i * 42 + 17);
            gpu_sponge.observe(val);
            FiatShamirTranscript::observe(&mut cpu_sponge, val);
        }

        for _ in 0..5 {
            let gpu_sample = gpu_sponge.sample();
            let cpu_sample = cpu_sponge.sample();
            assert_eq!(gpu_sample, cpu_sample);
        }
    }

    /// Benchmark test comparing CPU vs GPU grinding performance.
    ///
    /// Run with: `cargo test -p cuda-backend-v2 test_grind_cpu_vs_gpu -- --nocapture`
    ///
    /// Note: GPU has ~20ms fixed overhead for kernel launch + sync. CPU wins for
    /// small search spaces (low bit counts). GPU wins for larger search spaces
    /// where parallelism amortizes the overhead.
    #[test]
    fn test_grind_cpu_vs_gpu() {
        // Warmup: run one GPU grind to initialize CUDA context
        {
            let mut warmup = DuplexSpongeGpu::default();
            let _ = warmup.grind_gpu(8);
        }

        // Test multiple bit counts to see scaling
        let bit_counts = [8, 12, 16, 18, 20]
            .iter()
            .flat_map(|x| std::iter::repeat_n(*x, 5))
            .collect::<Vec<_>>();

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Grinding Performance: CPU vs GPU");
        eprintln!("{}", "=".repeat(60));
        eprintln!(
            "{:>6} {:>12} {:>12} {:>10}",
            "bits", "CPU (ms)", "GPU (ms)", "speedup"
        );
        eprintln!("{:->6} {:->12} {:->12} {:->10}", "", "", "", "");

        let mut seed = 265;
        for bits in bit_counts {
            let mut cpu_sponge = DuplexSponge::default();
            let mut gpu_sponge = DuplexSpongeGpu::default();

            // Add some initial state
            for _ in 0..5 {
                let val = F::from_u32(seed);
                seed += 228;
                FiatShamirTranscript::observe(&mut cpu_sponge, val);
                gpu_sponge.observe(val);
            }

            // Time CPU grinding
            let cpu_start = Instant::now();
            let cpu_witness = cpu_sponge.grind(bits);
            let cpu_time = cpu_start.elapsed();

            // Time GPU grinding
            let gpu_start = Instant::now();
            let gpu_witness = gpu_sponge.grind_gpu(bits).expect("GPU grinding failed");
            let gpu_time = gpu_start.elapsed();

            // Verify both found valid witnesses (witnesses may differ but both should be valid)
            // We already validated inside grind_gpu with debug_assert

            let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

            eprintln!(
                "{:>6} {:>12.2} {:>12.2} {:>10.2}x",
                bits,
                cpu_time.as_secs_f64() * 1000.0,
                gpu_time.as_secs_f64() * 1000.0,
                speedup
            );

            // Verify the witnesses are valid by checking with a fresh sponge
            // (grind() and grind_gpu() already do this internally via check_witness)
            let _ = (cpu_witness, gpu_witness); // suppress unused warnings
        }

        eprintln!("{}\n", "=".repeat(60));
    }
}
