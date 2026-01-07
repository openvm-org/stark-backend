use openvm_cuda_common::d_buffer::DeviceBuffer;
use stark_backend_v2::prover::fractional_sumcheck_gkr::Frac;
use tracing::debug;

use crate::{
    ConstraintOnlyRules, EF, F, InteractionEvalRules,
    cuda::logup_zerocheck::{
        _logup_mle_intermediates_buffer_size, _logup_mle_temp_sums_buffer_size,
        _zerocheck_mle_intermediates_buffer_size, _zerocheck_mle_temp_sums_buffer_size,
        MainMatrixPtrs, logup_eval_mle, zerocheck_eval_mle,
    },
};

// We interpolate first, so access to vars is free and doesn't need to be buffered
const ZEROCHECK_BUFFER_VARS: bool = false;

/// Evaluate MLE constraints on GPU.
///
/// Takes device pointers directly, avoiding H2D copies when data is already on device.
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
) -> DeviceBuffer<EF> {
    let buffer_size = rules.inner.buffer_size;
    let intermed_capacity =
        unsafe { _zerocheck_mle_intermediates_buffer_size(buffer_size, num_x, num_y) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<EF>::new()
    };
    let temp_sums_buffer_capacity = unsafe { _zerocheck_mle_temp_sums_buffer_size(num_x, num_y) };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let mut output = DeviceBuffer::<EF>::with_capacity(num_x as usize);

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
        )
        .expect("failed to evaluate MLE constraints on GPU");
    }
    output
}

/// Evaluate MLE interactions on GPU.
///
/// Takes device pointers directly, avoiding H2D copies when data is already on device.
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
) -> DeviceBuffer<Frac<EF>> {
    let buffer_size = rules.inner.buffer_size;
    let intermed_capacity =
        unsafe { _logup_mle_intermediates_buffer_size(buffer_size, num_x, num_y) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<EF>::new()
    };
    let temp_sums_buffer_capacity = unsafe { _logup_mle_temp_sums_buffer_size(num_x, num_y) };
    debug!("logup:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let mut output = DeviceBuffer::<Frac<EF>>::with_capacity(num_x as usize);

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
        )
        .expect("failed to evaluate MLE interactions on GPU");
    }
    output
}
