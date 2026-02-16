//! GPU-accelerated duplex sponge with host/device state synchronization.
//!
//! This module provides [`DuplexSpongeMetal`], a transcript implementation that maintains
//! state on both host and device with explicit synchronization methods.

use openvm_metal_common::{
    d_buffer::MetalBuffer,
    error::MetalError,
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

/// Device-side sponge state, matching the Metal `DeviceSpongeState` struct.
///
/// This struct is `#[repr(C)]` to ensure ABI compatibility with the Metal kernel.
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
/// The host-side state uses [`Challenger`] (= `DuplexChallenger`).
/// The device-side state is stored in GPU memory for Metal kernel operations.
///
/// # State Synchronization
///
/// The host and device states are **independent** and must be explicitly synchronized:
/// - Use [`sync_h2d`](Self::sync_h2d) to copy host state to device before GPU operations
/// - Use [`sync_d2h`](Self::sync_d2h) to copy device state back to host after GPU operations
#[derive(Debug)]
pub struct DuplexSpongeMetal {
    /// Host-side structure
    host: Challenger,
    /// Device-side state buffer (allocated lazily on first sync)
    device: MetalBuffer<DeviceSpongeState>,
}

impl Default for DuplexSpongeMetal {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for DuplexSpongeMetal {
    fn clone(&self) -> Self {
        let mut new = Self {
            host: self.host.clone(),
            device: MetalBuffer::with_capacity(1),
        };
        // Sync cloned host state to device if device was allocated
        if !self.device.is_empty() {
            let _ = new.sync_h2d();
        }
        new
    }
}

impl DuplexSpongeMetal {
    /// Create a new GPU-accelerated duplex sponge with default (zeroed) state.
    pub fn new() -> Self {
        Self {
            host: Challenger::new(default_babybear_poseidon2_16()),
            device: MetalBuffer::with_capacity(1),
        }
    }

    /// Returns true if the device buffer has been allocated.
    pub fn is_device_allocated(&self) -> bool {
        !self.device.is_empty()
    }

    /// Synchronize state from host to device (H2D memcpy).
    ///
    /// Call this before running GPU kernels that read/modify the sponge state.
    ///
    /// This converts from `DuplexChallenger`'s representation (with buffered input/output)
    /// to `DeviceSpongeState`'s representation (with indices pointing into state).
    pub fn sync_h2d(&mut self) -> Result<(), MetalError> {
        let mut device_state = DeviceSpongeState {
            state: self.host.sponge_state,
            absorb_idx: self.host.input_buffer.len() as u32,
            sample_idx: self.host.output_buffer.len() as u32,
        };

        // Overlay pending input_buffer values onto the beginning of state
        for (i, &val) in self.host.input_buffer.iter().enumerate() {
            device_state.state[i] = val;
        }

        // With Metal's unified memory (StorageModeShared), H2D is just memcpy
        unsafe {
            std::ptr::copy_nonoverlapping(
                &device_state as *const DeviceSpongeState,
                self.device.as_mut_ptr(),
                1,
            );
        }
        Ok(())
    }

    /// Get a pointer to the device state buffer.
    pub fn device_ptr(&self) -> Option<*const DeviceSpongeState> {
        if self.device.is_empty() {
            None
        } else {
            Some(self.device.as_ptr())
        }
    }

    /// Get a mutable pointer to the device state buffer.
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
    pub fn grind_gpu(&mut self, bits: usize) -> Result<F, GrindError> {
        // 1. Sync host state to device
        self.sync_h2d().map_err(GrindError::MetalError)?;

        // 2. Launch grinding kernel
        let witness_u32 = unsafe {
            crate::metal::sponge::sponge_grind(self.device.as_ptr(), bits as u32, F::ORDER_U32 - 1)?
        };

        let witness = F::from_u32(witness_u32);

        // 3. Update host state to match (observe the witness + sample)
        debug_assert!(self.clone().check_witness(bits, witness));
        self.host.observe(witness);
        let _: F = self.host.sample(); // Consume the sample to advance state

        Ok(witness)
    }
}

/// Error type for GPU grinding operations.
#[derive(Debug, thiserror::Error)]
pub enum GrindError {
    #[error("Metal error: {0}")]
    MetalError(MetalError),

    #[error("Failed to find PoW witness within search space")]
    WitnessNotFound,
}

impl FiatShamirTranscript<SC> for DuplexSpongeMetal {
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
