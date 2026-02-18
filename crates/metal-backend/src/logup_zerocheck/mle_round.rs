use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
use openvm_stark_backend::prover::fractional_sumcheck_gkr::Frac;
use tracing::debug;

use crate::{
    metal::logup_zerocheck::{
        logup_batch_eval_mle, logup_mle_intermediates_buffer_size, logup_mle_temp_sums_buffer_size,
        zerocheck_batch_eval_mle, zerocheck_mle_intermediates_buffer_size,
        zerocheck_mle_temp_sums_buffer_size, BlockCtx, EvalCoreCtx, LogupCtx, MainMatrixPtrs,
        ZerocheckCtx,
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
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        MetalBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        MetalBuffer::<EF>::with_capacity(0)
    };
    let temp_sums_buffer_capacity = zerocheck_mle_temp_sums_buffer_size(num_x, num_y);
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let _temp_sums_buffer = MetalBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let output = MetalBuffer::<EF>::with_capacity(num_x as usize);
    let threads_per_block = num_y.min(128).max(1);
    let num_blocks = num_y.div_ceil(threads_per_block);
    let mut block_ctxs_h = Vec::with_capacity(num_blocks as usize);
    for b in 0..num_blocks {
        block_ctxs_h.push(BlockCtx {
            local_block_idx_x: b,
            air_idx: 0,
        });
    }
    let block_ctxs = block_ctxs_h.to_device();
    let air_offsets = vec![0u32, num_blocks].to_device();
    let zc_ctxs = vec![ZerocheckCtx {
        eval_ctx: EvalCoreCtx {
            d_selectors: sels_ptr,
            d_preprocessed: prep_ptr,
            d_main: d_main_ptrs.as_device_ptr(),
            d_public: public_ptr,
        },
        d_intermediates: if buffer_size > 0 {
            intermediates.as_device_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
        num_y,
        d_eq_xi: eq_xi_ptr,
        d_rules: rules.inner.d_rules.as_device_ptr() as *const _,
        rules_len: rules.inner.d_rules.len().try_into().unwrap(),
        d_used_nodes: rules.inner.d_used_nodes.as_device_ptr(),
        used_nodes_len: rules.inner.d_used_nodes.len().try_into().unwrap(),
        buffer_size,
    }]
    .to_device();
    unsafe {
        zerocheck_batch_eval_mle(
            &output,
            &block_ctxs,
            &zc_ctxs,
            &air_offsets,
            lambda_pows,
            lambda_pows.len(),
            num_x,
            threads_per_block,
        )
        .expect("zerocheck_batch_eval_mle failed");
    }

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
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup:intermediates_capacity={intermed_capacity}");
        MetalBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        MetalBuffer::<EF>::with_capacity(0)
    };
    let temp_sums_buffer_capacity = logup_mle_temp_sums_buffer_size(num_x, num_y);
    debug!("logup:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let _temp_sums_buffer = MetalBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let output = MetalBuffer::<Frac<EF>>::with_capacity(num_x as usize);
    let threads_per_block = num_y.min(128).max(1);
    let num_blocks = num_y.div_ceil(threads_per_block);
    let mut block_ctxs_h = Vec::with_capacity(num_blocks as usize);
    for b in 0..num_blocks {
        block_ctxs_h.push(BlockCtx {
            local_block_idx_x: b,
            air_idx: 0,
        });
    }
    let block_ctxs = block_ctxs_h.to_device();
    let air_offsets = vec![0u32, num_blocks].to_device();
    let logup_ctxs = vec![LogupCtx {
        eval_ctx: EvalCoreCtx {
            d_selectors: sels_ptr,
            d_preprocessed: prep_ptr,
            d_main: d_main_ptrs.as_device_ptr(),
            d_public: public_ptr,
        },
        d_intermediates: if buffer_size > 0 {
            intermediates.as_device_mut_ptr()
        } else {
            std::ptr::null_mut()
        },
        num_y,
        d_eq_xi: eq_xi_ptr,
        d_challenges: challenges_ptr,
        d_eq_3bs: eq_3bs_ptr,
        d_rules: rules.inner.d_rules.as_device_ptr() as *const _,
        rules_len: rules.inner.d_rules.len().try_into().unwrap(),
        d_used_nodes: rules.inner.d_used_nodes.as_device_ptr(),
        d_pair_idxs: rules.d_pair_idxs.as_device_ptr(),
        used_nodes_len: rules.inner.d_used_nodes.len().try_into().unwrap(),
        buffer_size,
    }]
    .to_device();
    unsafe {
        logup_batch_eval_mle(
            &output,
            &block_ctxs,
            &logup_ctxs,
            &air_offsets,
            num_x,
            threads_per_block,
        )
        .expect("logup_batch_eval_mle failed");
    }

    output
}
