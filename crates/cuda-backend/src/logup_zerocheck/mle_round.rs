use openvm_cuda_common::{d_buffer::DeviceBuffer, error::CudaError, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use tracing::debug;

use crate::{
    cuda::logup_zerocheck::{
        _logup_mle_intermediates_buffer_size, _logup_mle_temp_sums_buffer_size,
        _zerocheck_mle_intermediates_buffer_size, _zerocheck_mle_temp_sums_buffer_size,
        logup_eval_mle, zerocheck_eval_mle, MainMatrixPtrs,
    },
    error::KernelError,
    prelude::{EF, F},
    ConstraintOnlyRules, InteractionEvalRules,
};

// We interpolate first, so access to vars is free and doesn't need to be buffered
const ZEROCHECK_BUFFER_VARS: bool = false;
const CUDA_GRID_Y_DIM_MAX: u32 = 65535;

fn validate_mle_num_x(num_x: u32) -> Result<(), KernelError> {
    if num_x == 0 || num_x > CUDA_GRID_Y_DIM_MAX {
        return Err(CudaError::new(1).into());
    }
    Ok(())
}

/// Evaluate MLE constraints on GPU.
///
/// Takes device pointers directly, avoiding H2D copies when data is already on device.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_constraints_gpu(
    eq_xi_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    d_main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>,
    public_ptr: *const F,
    lambda_pows: &DeviceBuffer<EF>,
    rules: &ConstraintOnlyRules<ZEROCHECK_BUFFER_VARS>,
    num_y: u32,
    num_x: u32,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<EF>, KernelError> {
    validate_mle_num_x(num_x)?;
    let stream = device_ctx.stream.as_raw();
    let buffer_size = rules.inner.buffer_size;
    let intermed_capacity =
        unsafe { _zerocheck_mle_intermediates_buffer_size(buffer_size, num_x, num_y, stream) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<EF>::with_capacity_on(intermed_capacity, device_ctx)
    } else {
        DeviceBuffer::<EF>::new()
    };
    let temp_sums_buffer_capacity =
        unsafe { _zerocheck_mle_temp_sums_buffer_size(num_x, num_y, stream) };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer =
        DeviceBuffer::<EF>::with_capacity_on(temp_sums_buffer_capacity, device_ctx);
    let mut output = DeviceBuffer::<EF>::with_capacity_on(num_x as usize, device_ctx);

    unsafe {
        zerocheck_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            d_main_ptrs.as_ptr(),
            lambda_pows.as_ptr(),
            lambda_pows.len(),
            public_ptr,
            rules.inner.d_rules.as_raw_ptr(),
            rules.inner.d_rules.len(),
            rules.inner.d_used_nodes.as_ptr(),
            rules.inner.d_used_nodes.len(),
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
            stream,
        )?;
    }
    Ok(output)
}

/// Evaluate MLE interactions on GPU.
///
/// Takes device pointers directly, avoiding H2D copies when data is already on device.
/// See [`crate::logup_zerocheck`] module docs for async-free/peak memory behavior.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_interactions_gpu(
    eq_xi_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    d_main_ptrs: &DeviceBuffer<MainMatrixPtrs<EF>>,
    public_ptr: *const F,
    challenges_ptr: *const EF,
    eq_3bs_ptr: *const EF,
    rules: &InteractionEvalRules,
    num_y: u32,
    num_x: u32,
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<Frac<EF>>, KernelError> {
    validate_mle_num_x(num_x)?;
    let stream = device_ctx.stream.as_raw();
    let buffer_size = rules.inner.buffer_size;
    let intermed_capacity =
        unsafe { _logup_mle_intermediates_buffer_size(buffer_size, num_x, num_y, stream) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<EF>::with_capacity_on(intermed_capacity, device_ctx)
    } else {
        DeviceBuffer::<EF>::new()
    };
    let temp_sums_buffer_capacity =
        unsafe { _logup_mle_temp_sums_buffer_size(num_x, num_y, stream) };
    debug!("logup:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer =
        DeviceBuffer::<Frac<EF>>::with_capacity_on(temp_sums_buffer_capacity, device_ctx);
    let mut output = DeviceBuffer::<Frac<EF>>::with_capacity_on(num_x as usize, device_ctx);

    unsafe {
        logup_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            d_main_ptrs.as_ptr(),
            challenges_ptr,
            eq_3bs_ptr,
            public_ptr,
            rules.inner.d_rules.as_raw_ptr(),
            rules.inner.d_used_nodes.as_ptr(),
            rules.d_pair_idxs.as_ptr(),
            rules.inner.d_used_nodes.len(),
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
            stream,
        )?;
    }
    Ok(output)
}
