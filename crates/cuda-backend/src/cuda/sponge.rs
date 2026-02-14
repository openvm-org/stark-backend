//! Rust bindings for CUDA sponge grinding kernel.

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
};

use crate::sponge::{DeviceSpongeState, GrindError};

extern "C" {
    fn _sponge_grind(
        init_state: *const DeviceSpongeState,
        bits: u32,
        min_witness: u32,
        max_witness: u32,
        result: *mut u32,
    ) -> i32;
}

/// Launch the GPU grinding kernel to find a PoW witness.
///
/// # Arguments
/// * `init_state` - Pointer to the initial sponge state on device
/// * `bits` - Number of bits that must be zero in the sampled value
/// * `max_witness` - Maximum witness value to search (typically F::ORDER - 1)
///
/// # Returns
/// The witness value that satisfies the PoW requirement, or an error.
///
/// # Safety
/// - `init_state` must point to valid device memory containing a `DeviceSpongeState`
pub unsafe fn sponge_grind(
    init_state: *const DeviceSpongeState,
    bits: u32,
    max_witness: u32,
) -> Result<u32, GrindError> {
    let mut d_result = DeviceBuffer::with_capacity(1);
    [u32::MAX].copy_to(&mut d_result)?;
    for start in (0..=max_witness).step_by(1 << bits) {
        CudaError::from_result(_sponge_grind(
            init_state,
            bits,
            start,
            max_witness,
            d_result.as_mut_ptr(),
        ))?;

        let result = d_result.to_host()?[0];
        if result < u32::MAX {
            return Ok(result);
        }
    }
    Err(GrindError::WitnessNotFound)
}
