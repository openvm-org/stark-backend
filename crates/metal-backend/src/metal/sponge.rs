//! Sponge grinding kernel dispatch wrappers.
//! Ported from cuda-backend/src/cuda/sponge.rs

use std::ffi::c_void;

use openvm_metal_common::d_buffer::MetalBuffer;

use crate::sponge::{DeviceSpongeState, GrindError};

use super::{dispatch_sync, get_kernels, grid_size_1d, DEFAULT_THREADS_PER_GROUP};

/// Launch the Metal grinding kernel to find a PoW witness.
///
/// # Safety
/// - `init_state` must point to valid device memory containing a `DeviceSpongeState`
pub unsafe fn sponge_grind(
    init_state: &MetalBuffer<DeviceSpongeState>,
    bits: u32,
    max_witness: u32,
) -> Result<u32, GrindError> {
    let d_result = MetalBuffer::<u32>::with_capacity(1);
    d_result.copy_from_slice(&[u32::MAX]);
    let pipeline = get_kernels()
        .get_pipeline("sponge_grind")
        .map_err(|e| GrindError::MetalError(e))?;
    for start in (0..=max_witness).step_by(1 << bits) {
        let total = (1usize << bits).min((max_witness - start + 1) as usize);
        let (grid, group) = grid_size_1d(total, DEFAULT_THREADS_PER_GROUP);
        dispatch_sync(&pipeline, grid, group, |encoder| {
            encoder.set_buffer(0, Some(init_state.gpu_buffer()), 0);
            encoder.set_bytes(1, 4, &bits as *const u32 as *const c_void);
            encoder.set_bytes(2, 4, &start as *const u32 as *const c_void);
            encoder.set_bytes(3, 4, &max_witness as *const u32 as *const c_void);
            encoder.set_buffer(4, Some(d_result.gpu_buffer()), 0);
        })
        .map_err(|e| GrindError::MetalError(e))?;

        let result = d_result.to_vec()[0];
        if result < u32::MAX {
            return Ok(result);
        }
    }
    Err(GrindError::WitnessNotFound)
}
