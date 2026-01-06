use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
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

#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_constraints_gpu(
    eq_xi_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    main_ptrs: &[MainMatrixPtrs<EF>],
    public_vals: &DeviceBuffer<F>,
    lambda_pows: &DeviceBuffer<EF>,
    rules: &ConstraintOnlyRules<ZEROCHECK_BUFFER_VARS>,
    num_y: usize,
    s_deg: usize,
) -> DeviceBuffer<EF> {
    let d_main_ptrs = main_ptrs
        .to_device()
        .expect("failed to copy main_ptrs to device");

    // Calculate dimensions
    let num_x = s_deg as u32;
    let num_y = num_y as u32;

    let buffer_size: u32 = rules.inner.buffer_size;
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
    let mut output = DeviceBuffer::<EF>::with_capacity(s_deg);
    // Launch evaluation kernel
    unsafe {
        zerocheck_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            &d_main_ptrs,
            lambda_pows,
            public_vals,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
        )
        .expect("failed to evaluate MLE constraints on GPU");
    }

    output
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_interactions_gpu(
    eq_sharp_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    main_ptrs: &[MainMatrixPtrs<EF>],
    public_vals: &DeviceBuffer<F>,
    d_challenges: &DeviceBuffer<EF>,
    eq_3bs: &DeviceBuffer<EF>,
    rules: &InteractionEvalRules,
    num_y: usize,
    s_deg: usize,
) -> DeviceBuffer<Frac<EF>> {
    let d_main_ptrs = main_ptrs
        .to_device()
        .expect("failed to copy main_ptrs to device");

    // Calculate dimensions
    let num_y = num_y as u32;
    let num_x = s_deg as u32;

    let buffer_size: u32 = rules.inner.buffer_size;
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
    let mut output = DeviceBuffer::<Frac<EF>>::with_capacity(s_deg);

    // Launch interaction evaluation kernel
    unsafe {
        logup_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_sharp_ptr,
            sels_ptr,
            prep_ptr,
            &d_main_ptrs,
            d_challenges,
            eq_3bs,
            public_vals,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            &rules.d_pair_idxs,
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
        )
        .expect("failed to evaluate MLE interactions on GPU");
    }
    output
}
