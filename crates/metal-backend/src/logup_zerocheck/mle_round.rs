use openvm_metal_common::d_buffer::MetalBuffer;
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use tracing::debug;

use crate::{
    metal::logup_zerocheck::{
        logup_mle_intermediates_buffer_size, logup_mle_temp_sums_buffer_size,
        zerocheck_mle_intermediates_buffer_size, zerocheck_mle_temp_sums_buffer_size,
        MainMatrixPtrs,
    },
    prelude::{EF, F},
    ConstraintOnlyRules, InteractionEvalRules,
};

// We interpolate first, so access to vars is free and doesn't need to be buffered
const ZEROCHECK_BUFFER_VARS: bool = false;

/// Evaluate MLE constraints on Metal.
///
/// Takes device pointers directly (unified memory - no H2D copies needed).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_constraints_metal(
    eq_xi_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    d_main_ptrs: &MetalBuffer<MainMatrixPtrs<EF>>,
    public_ptr: *const F,
    lambda_pows: &MetalBuffer<EF>,
    rules: &ConstraintOnlyRules<ZEROCHECK_BUFFER_VARS>,
    num_y: u32,
    num_x: u32,
) -> MetalBuffer<EF> {
    let buffer_size = rules.inner.buffer_size;
    let intermed_capacity = zerocheck_mle_intermediates_buffer_size(buffer_size, num_x, num_y);
    let mut _intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        MetalBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        MetalBuffer::<EF>::with_capacity(0)
    };
    let temp_sums_buffer_capacity = zerocheck_mle_temp_sums_buffer_size(num_x, num_y);
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut _temp_sums_buffer = MetalBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let output = MetalBuffer::<EF>::with_capacity(num_x as usize);

    // TODO: dispatch zerocheck_eval_mle Metal kernel when available.
    let _ = (
        eq_xi_ptr,
        sels_ptr,
        prep_ptr,
        d_main_ptrs,
        public_ptr,
        lambda_pows,
    );

    output
}

/// Evaluate MLE interactions on Metal.
///
/// Takes device pointers directly (unified memory - no H2D copies needed).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_interactions_metal(
    eq_xi_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    d_main_ptrs: &MetalBuffer<MainMatrixPtrs<EF>>,
    public_ptr: *const F,
    challenges_ptr: *const EF,
    eq_3bs_ptr: *const EF,
    rules: &InteractionEvalRules,
    num_y: u32,
    num_x: u32,
) -> MetalBuffer<Frac<EF>> {
    let buffer_size = rules.inner.buffer_size;
    let intermed_capacity = logup_mle_intermediates_buffer_size(buffer_size, num_x, num_y);
    let mut _intermediates = if intermed_capacity > 0 {
        debug!("logup:intermediates_capacity={intermed_capacity}");
        MetalBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        MetalBuffer::<EF>::with_capacity(0)
    };
    let temp_sums_buffer_capacity = logup_mle_temp_sums_buffer_size(num_x, num_y);
    debug!("logup:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut _temp_sums_buffer = MetalBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let output = MetalBuffer::<Frac<EF>>::with_capacity(num_x as usize);

    // TODO: dispatch logup_eval_mle Metal kernel when available.
    let _ = (
        eq_xi_ptr,
        sels_ptr,
        prep_ptr,
        d_main_ptrs,
        public_ptr,
        challenges_ptr,
        eq_3bs_ptr,
    );

    output
}
