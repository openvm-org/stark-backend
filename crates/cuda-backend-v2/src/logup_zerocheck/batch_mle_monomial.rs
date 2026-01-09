//! Batched monomial-based MLE evaluation for zerocheck.
//!
//! This module provides a batch evaluator for monomial evaluations across multiple AIRs,
//! enabling efficient GPU kernel launches that process multiple traces in a single launch.

use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::CudaError};
use stark_backend_v2::prover::DeviceMultiStarkProvingKeyV2;
use tracing::debug;

use crate::{
    EF, GpuBackendV2,
    cuda::logup_zerocheck::{
        BlockCtx, EvalCoreCtx, MonomialAirCtx, precompute_lambda_combinations,
        zerocheck_monomial_batched,
    },
    logup_zerocheck::batch_mle::TraceCtx,
};

const THREADS_PER_BLOCK: u32 = 256;

/// Returns true if the trace can use the monomial evaluation path.
///
/// A trace is eligible if it has constraints and the AIR has expanded monomials.
pub(crate) fn trace_has_monomials(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
) -> bool {
    trace.has_constraints
        && pk.per_air[trace.air_idx]
            .other_data
            .zerocheck_monomials
            .as_ref()
            .map(|m| m.num_monomials > 0)
            .unwrap_or(false)
}

/// Precompute lambda combinations for a single AIR's monomials.
///
/// Returns a buffer of length `num_monomials` where each element is
/// `sum_l(coefficient_l * lambda_pows[constraint_idx_l])` for that monomial.
///
/// The AIR must have nonempty monomials.
pub(crate) fn compute_lambda_combinations(
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    air_idx: usize,
    lambda_pows: &DeviceBuffer<EF>,
) -> Result<DeviceBuffer<EF>, CudaError> {
    let monomials = pk.per_air[air_idx]
        .other_data
        .zerocheck_monomials
        .as_ref()
        .expect("AIR must have monomials");
    let mut buf = DeviceBuffer::<EF>::with_capacity(monomials.num_monomials as usize);
    unsafe {
        precompute_lambda_combinations(
            &mut buf,
            monomials.d_headers.as_ptr(),
            monomials.d_lambda_terms.as_ptr(),
            lambda_pows,
            monomials.num_monomials,
        )?;
    }
    Ok(buf)
}

/// Batch evaluator for monomial-based zerocheck MLE evaluation.
///
/// Pre-builds GPU contexts for all traces, then evaluates in a single kernel launch.
///
/// The caller must filter traces using [`trace_has_monomials`] before constructing.
/// The batch must contain at least one trace.
///
/// The struct holds references to `TraceCtx` which guarantees the underlying
/// device buffers (including `main_ptrs_dev`) remain valid for the struct's lifetime.
pub(crate) struct ZerocheckMonomialBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: DeviceBuffer<BlockCtx>,
    air_ctxs: DeviceBuffer<MonomialAirCtx>,
    air_offsets: DeviceBuffer<u32>,
}

impl<'a> ZerocheckMonomialBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `lambda_combinations` must contain one buffer per trace (in iteration order),
    /// each precomputed via [`compute_lambda_combinations`].
    ///
    /// # Panics
    ///
    /// Panics if `traces` is empty or if `lambda_combinations` length doesn't match.
    pub fn new(
        traces: impl Iterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        lambda_combinations: &[&DeviceBuffer<EF>],
    ) -> Self {
        let traces: Vec<_> = traces.collect();
        assert!(
            !traces.is_empty(),
            "ZerocheckMonomialBatch requires at least one trace"
        );
        assert_eq!(
            traces.len(),
            lambda_combinations.len(),
            "lambda_combinations must have one buffer per trace"
        );

        let threads_per_block = THREADS_PER_BLOCK;

        // Build block_ctxs and air_offsets
        // For each AIR: total_blocks = ceil(num_monomials / tpb) * num_y
        // local_block_idx_x encodes (y_int * mono_blocks + mono_block_idx)
        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut air_offsets: Vec<u32> = Vec::with_capacity(traces.len() + 1);
        air_offsets.push(0);

        for (local_air, t) in traces.iter().enumerate() {
            let monomials = pk.per_air[t.air_idx]
                .other_data
                .zerocheck_monomials
                .as_ref()
                .unwrap();
            let mono_blocks = monomials.num_monomials.div_ceil(threads_per_block);
            let total_blocks = mono_blocks * t.num_y;

            for local_idx in 0..total_blocks {
                block_ctxs_h.push(BlockCtx {
                    local_block_idx_x: local_idx,
                    air_idx: local_air as u32,
                });
            }
            air_offsets.push(block_ctxs_h.len() as u32);
        }

        // Build MonomialAirCtx for each trace
        let air_ctxs_h: Vec<MonomialAirCtx> = traces
            .iter()
            .zip(lambda_combinations)
            .map(|(t, lc)| {
                let monomials = pk.per_air[t.air_idx]
                    .other_data
                    .zerocheck_monomials
                    .as_ref()
                    .unwrap();

                let eval_ctx = EvalCoreCtx {
                    d_selectors: t.sels_ptr,
                    d_preprocessed: t.prep_ptr,
                    d_main: t.main_ptrs_dev.as_ptr(),
                    d_public: t.public_ptr,
                };

                MonomialAirCtx {
                    d_headers: monomials.d_headers.as_ptr(),
                    d_variables: monomials.d_variables.as_ptr(),
                    d_lambda_combinations: lc.as_ptr(),
                    num_monomials: monomials.num_monomials,
                    eval_ctx,
                    d_eq_xi: t.eq_xi_ptr,
                    num_y: t.num_y,
                }
            })
            .collect();

        // Upload to device
        let block_ctxs = block_ctxs_h
            .to_device()
            .expect("failed to copy monomial block ctxs to device");
        let air_ctxs = air_ctxs_h
            .to_device()
            .expect("failed to copy monomial air ctxs to device");
        let air_offsets = air_offsets
            .to_device()
            .expect("failed to copy air offsets to device");

        debug!(
            num_airs = traces.len(),
            num_blocks = block_ctxs_h.len(),
            "ZerocheckMonomialBatch created"
        );

        Self {
            traces,
            block_ctxs,
            air_ctxs,
            air_offsets,
        }
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    ///
    /// The buffer contains `num_airs * num_x` elements, laid out as
    /// `[air0_x0, air0_x1, ..., air1_x0, air1_x1, ...]`.
    pub fn evaluate(&self, num_x: u32) -> DeviceBuffer<EF> {
        let num_blocks = self.block_ctxs.len();
        let num_airs = self.air_ctxs.len();

        debug!(
            %num_blocks,
            %num_x,
            %num_airs,
            "zerocheck_monomial_batched"
        );

        let mut tmp_sums = DeviceBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
        let mut output = DeviceBuffer::<EF>::with_capacity(num_airs * num_x as usize);

        debug_assert_eq!(
            self.air_offsets.len(),
            num_airs + 1,
            "air_offsets must have num_airs + 1 elements"
        );
        // SAFETY: All device pointers in block_ctxs and air_ctxs were constructed from
        // valid DeviceBuffers that outlive this call (TraceCtx references, pk monomial data,
        // lambda_combinations). The air_offsets buffer has length num_airs + 1 as required.
        unsafe {
            zerocheck_monomial_batched(
                &mut tmp_sums,
                &mut output,
                &self.block_ctxs,
                &self.air_ctxs,
                &self.air_offsets,
                num_blocks as u32,
                num_x,
                num_airs as u32,
                THREADS_PER_BLOCK,
            )
            .expect("zerocheck monomial batched kernel failed");
        }

        output
    }
}
