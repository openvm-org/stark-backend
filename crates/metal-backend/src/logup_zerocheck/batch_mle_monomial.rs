//! Batched monomial-based MLE evaluation for zerocheck and logup.
//!
//! This module provides batch evaluators for monomial evaluations across multiple AIRs,
//! enabling efficient Metal kernel launches that process multiple traces in a single launch.

use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
use openvm_stark_backend::prover::{fractional_sumcheck_gkr::Frac, DeviceMultiStarkProvingKey};
use p3_field::PrimeCharacteristicRing;
use tracing::debug;

use crate::{
    metal::logup_zerocheck::{
        BlockCtx, EvalCoreCtx, LogupMonomialCommonCtx, LogupMonomialCtx, MonomialAirCtx,
    },
    logup_zerocheck::batch_mle::TraceCtx,
    prelude::EF,
    MetalBackend,
};

const THREADS_PER_BLOCK: u32 = 256;

/// Returns true if the trace can use the monomial evaluation path.
///
/// A trace is eligible if it has constraints and the AIR has expanded monomials.
pub(crate) fn trace_has_monomials(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
) -> bool {
    trace.has_constraints
        && pk.per_air[trace.air_idx]
            .other_data
            .zerocheck_monomials
            .as_ref()
            .map(|m| m.num_monomials > 0)
            .unwrap_or(false)
}

/// Get the number of monomials for a trace. Returns 0 if the trace has no monomials.
pub(crate) fn get_num_monomials(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
) -> u32 {
    pk.per_air[trace.air_idx]
        .other_data
        .zerocheck_monomials
        .as_ref()
        .map(|m| m.num_monomials)
        .unwrap_or(0)
}

/// Get the rules_len for a trace's zerocheck DAG.
pub(crate) fn get_zerocheck_rules_len(
    trace: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
) -> usize {
    pk.per_air[trace.air_idx]
        .other_data
        .zerocheck_mle
        .inner
        .d_rules
        .len()
}

/// Precompute lambda combinations for a single AIR's monomials.
///
/// Returns a buffer of length `num_monomials` where each element is
/// `sum_l(coefficient_l * lambda_pows[constraint_idx_l])` for that monomial.
///
/// The AIR must have nonempty monomials.
pub(crate) fn compute_lambda_combinations(
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    air_idx: usize,
    lambda_pows: &MetalBuffer<EF>,
) -> MetalBuffer<EF> {
    let monomials = pk.per_air[air_idx]
        .other_data
        .zerocheck_monomials
        .as_ref()
        .expect("AIR must have monomials");
    let mut buf = MetalBuffer::<EF>::with_capacity(monomials.num_monomials as usize);
    unsafe {
        crate::metal::logup_zerocheck::precompute_lambda_combinations(
            &mut buf,
            monomials.d_headers.as_ptr(),
            monomials.d_lambda_terms.as_ptr(),
            lambda_pows,
            monomials.num_monomials,
        )
        .expect("precompute_lambda_combinations kernel failed");
    }
    buf
}

/// Batch evaluator for monomial-based zerocheck MLE evaluation.
///
/// Pre-builds Metal contexts for all traces, then evaluates in a single kernel launch.
///
/// The caller must filter traces using [`trace_has_monomials`] before constructing.
/// The batch must contain at least one trace.
pub(crate) struct ZerocheckMonomialBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: MetalBuffer<BlockCtx>,
    air_ctxs: MetalBuffer<MonomialAirCtx>,
    air_offsets: MetalBuffer<u32>,
}

impl<'a> ZerocheckMonomialBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `lambda_combinations` must contain one buffer per trace (in iteration order),
    /// each precomputed via [`compute_lambda_combinations`].
    pub fn new(
        traces: impl IntoIterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<MetalBackend>,
        lambda_combinations: &[&MetalBuffer<EF>],
    ) -> Self {
        let traces: Vec<_> = traces.into_iter().collect();
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

        let block_ctxs = block_ctxs_h.to_device();
        let air_ctxs = air_ctxs_h.to_device();
        let air_offsets = air_offsets.to_device();

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
    pub fn evaluate(&self, num_x: u32) -> MetalBuffer<EF> {
        let num_blocks = self.block_ctxs.len();
        let num_airs = self.air_ctxs.len();

        debug!(
            %num_blocks,
            %num_x,
            %num_airs,
            "zerocheck_monomial_batched"
        );

        let _tmp_sums = MetalBuffer::<EF>::with_capacity(num_blocks * num_x as usize);
        let output = MetalBuffer::<EF>::with_capacity(num_airs * num_x as usize);

        debug_assert_eq!(
            self.air_offsets.len(),
            num_airs + 1,
            "air_offsets must have num_airs + 1 elements"
        );

        // TODO: dispatch zerocheck_monomial_batched Metal kernel when available.
        let _ = (&self.block_ctxs, &self.air_ctxs, &self.air_offsets);

        output
    }
}

// Constants for par-y kernel
const THREADS_PER_BLOCK_PAR_Y: u32 = 128;
const DEFAULT_MAX_MONOMIALS_PER_THREAD: u32 = 64;
const WAVES_TARGET: u32 = 4;

/// Batch evaluator for monomial-based zerocheck MLE evaluation, parallelizing over y_int.
///
/// This variant is optimized for traces with high `num_y`: each thread handles one y_int
/// and loops over a chunk of monomials. The chunk size is auto-tuned based on SM count.
pub(crate) struct ZerocheckMonomialParYBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: MetalBuffer<BlockCtx>,
    air_ctxs: MetalBuffer<MonomialAirCtx>,
    air_offsets: MetalBuffer<u32>,
    num_blocks: u32,
    chunk_size: u32,
}

impl<'a> ZerocheckMonomialParYBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `lambda_combinations` must contain one buffer per trace (in iteration order),
    /// each precomputed via [`compute_lambda_combinations`].
    ///
    /// The `sm_count` and `num_x` parameters are used to auto-tune the chunk size
    /// for optimal GPU utilization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        traces: impl IntoIterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<MetalBackend>,
        lambda_combinations: &[&MetalBuffer<EF>],
        sm_count: u32,
        num_x: u32,
        max_monomials_per_thread: Option<u32>,
    ) -> Self {
        let traces: Vec<_> = traces.into_iter().collect();
        assert!(
            !traces.is_empty(),
            "ZerocheckMonomialParYBatch requires at least one trace"
        );
        assert_eq!(
            traces.len(),
            lambda_combinations.len(),
            "lambda_combinations must have one buffer per trace"
        );

        let threads_per_block = THREADS_PER_BLOCK_PAR_Y;
        let max_mono_per_thread =
            max_monomials_per_thread.unwrap_or(DEFAULT_MAX_MONOMIALS_PER_THREAD);

        let mut per_air_info: Vec<(u32, u32)> = Vec::new(); // (y_blocks, num_monomials)
        let mut max_monomials = 0u32;

        for t in traces.iter() {
            let monomials = pk.per_air[t.air_idx]
                .other_data
                .zerocheck_monomials
                .as_ref()
                .expect("AIR with constraints must have monomials");

            let y_blocks = t.num_y.div_ceil(threads_per_block);
            max_monomials = max_monomials.max(monomials.num_monomials);
            per_air_info.push((y_blocks, monomials.num_monomials));
        }

        // Determine chunk_size based on GPU utilization and cap
        let target_blocks = sm_count * WAVES_TARGET;
        let mut chunk_size = max_mono_per_thread;
        loop {
            let total_blocks: u32 = per_air_info
                .iter()
                .map(|(y_blocks, num_mono)| {
                    let air_mono_chunks = num_mono.div_ceil(chunk_size);
                    y_blocks * air_mono_chunks
                })
                .sum();

            if total_blocks * num_x >= target_blocks || chunk_size <= 1 {
                break;
            }
            chunk_size = (chunk_size / 2).max(1);
        }

        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut air_offsets: Vec<u32> = Vec::with_capacity(traces.len() + 1);
        air_offsets.push(0);

        for (local_air, (y_blocks, num_mono)) in per_air_info.iter().enumerate() {
            let air_mono_chunks = num_mono.div_ceil(chunk_size);

            for y_block in 0..*y_blocks {
                for mono_chunk in 0..air_mono_chunks {
                    let local_idx = y_block * air_mono_chunks + mono_chunk;
                    block_ctxs_h.push(BlockCtx {
                        local_block_idx_x: local_idx,
                        air_idx: local_air as u32,
                    });
                }
            }
            air_offsets.push(block_ctxs_h.len() as u32);
        }

        let num_blocks = block_ctxs_h.len() as u32;

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

        let block_ctxs = block_ctxs_h.to_device();
        let air_ctxs = air_ctxs_h.to_device();
        let air_offsets = air_offsets.to_device();

        debug!(
            num_airs = traces.len(),
            num_blocks, chunk_size, max_monomials, "ZerocheckMonomialParYBatch created"
        );

        Self {
            traces,
            block_ctxs,
            air_ctxs,
            air_offsets,
            num_blocks,
            chunk_size,
        }
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    pub fn evaluate(&self, num_x: u32) -> MetalBuffer<EF> {
        let num_airs = self.air_ctxs.len();

        debug!(
            num_blocks = %self.num_blocks,
            %num_x,
            %num_airs,
            chunk_size = %self.chunk_size,
            "zerocheck_monomial_par_y_batched"
        );

        let _tmp_sums =
            MetalBuffer::<EF>::with_capacity(self.num_blocks as usize * num_x as usize);
        let output = MetalBuffer::<EF>::with_capacity(num_airs * num_x as usize);

        debug_assert_eq!(
            self.air_offsets.len(),
            num_airs + 1,
            "air_offsets must have num_airs + 1 elements"
        );

        // TODO: dispatch zerocheck_monomial_par_y_batched Metal kernel when available.
        let _ = (
            &self.block_ctxs,
            &self.air_ctxs,
            &self.air_offsets,
            self.chunk_size,
        );

        output
    }
}

// ============================================================================
// LOGUP MONOMIAL EVALUATION
// ============================================================================

/// Precomputed logup combinations for a single AIR.
pub struct LogupCombinations {
    pub d_numer_combinations: MetalBuffer<EF>,
    pub d_denom_combinations: MetalBuffer<EF>,
    pub bus_term_sum: EF,
}

/// Precompute logup combinations for a single AIR's interaction monomials.
///
/// The AIR must have nonempty interaction monomials.
pub(crate) fn compute_logup_combinations(
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    air_idx: usize,
    d_beta_pows: &MetalBuffer<EF>,
    d_eq_3bs: &MetalBuffer<EF>,
    eq_3bs_host: &[EF],
    beta_pows_host: &[EF],
) -> LogupCombinations {
    let monomials = pk.per_air[air_idx]
        .other_data
        .interaction_monomials
        .as_ref()
        .expect("AIR must have interaction monomials");

    // Precompute numerator combinations
    let mut d_numer_combinations = if monomials.num_numer_monomials > 0 {
        MetalBuffer::<EF>::with_capacity(monomials.num_numer_monomials as usize)
    } else {
        MetalBuffer::with_capacity(0)
    };
    if monomials.num_numer_monomials > 0 {
        unsafe {
            crate::metal::logup_zerocheck::precompute_logup_numer_combinations(
                &mut d_numer_combinations,
                monomials.d_numer_headers.as_ptr(),
                monomials.d_numer_terms.as_ptr(),
                d_eq_3bs,
                monomials.num_numer_monomials,
            )
            .expect("precompute_logup_numer_combinations kernel failed");
        }
    }

    // Precompute denominator combinations
    let mut d_denom_combinations = if monomials.num_denom_monomials > 0 {
        MetalBuffer::<EF>::with_capacity(monomials.num_denom_monomials as usize)
    } else {
        MetalBuffer::with_capacity(0)
    };
    if monomials.num_denom_monomials > 0 {
        unsafe {
            crate::metal::logup_zerocheck::precompute_logup_denom_combinations(
                &mut d_denom_combinations,
                monomials.d_denom_headers.as_ptr(),
                monomials.d_denom_terms.as_ptr(),
                d_beta_pows,
                d_eq_3bs,
                monomials.num_denom_monomials,
            )
            .expect("precompute_logup_denom_combinations kernel failed");
        }
    }

    // Compute bus_term_sum on CPU
    let interactions = &pk.per_air[air_idx].vk.symbolic_constraints.interactions;
    debug_assert_eq!(
        interactions.len(),
        eq_3bs_host.len(),
        "interaction count must match eq_3bs"
    );
    let mut bus_term_sum = EF::ZERO;
    for (i, interaction) in interactions.iter().enumerate() {
        let beta_len = beta_pows_host[interaction.message.len()];
        let bus_idx = interaction.bus_index as u32;
        bus_term_sum += beta_len * EF::from_u32(bus_idx + 1) * eq_3bs_host[i];
    }

    LogupCombinations {
        d_numer_combinations,
        d_denom_combinations,
        bus_term_sum,
    }
}

const THREADS_PER_BLOCK_LOGUP: u32 = 128;

/// Batch evaluator for logup monomial MLE evaluation.
pub(crate) struct LogupMonomialBatch<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: MetalBuffer<BlockCtx>,
    common_ctxs: MetalBuffer<LogupMonomialCommonCtx>,
    numer_ctxs: MetalBuffer<LogupMonomialCtx>,
    denom_ctxs: MetalBuffer<LogupMonomialCtx>,
    air_offsets: MetalBuffer<u32>,
    num_blocks: u32,
}

impl<'a> LogupMonomialBatch<'a> {
    /// Creates a new batch from an iterator of traces.
    ///
    /// `logup_combinations` must contain one `LogupCombinations` per trace (in iteration order),
    /// each precomputed via [`compute_logup_combinations`].
    pub fn new(
        traces: impl IntoIterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<MetalBackend>,
        logup_combinations: &[&LogupCombinations],
    ) -> Self {
        let traces: Vec<_> = traces.into_iter().collect();
        assert!(
            !traces.is_empty(),
            "LogupMonomialBatch requires at least one trace"
        );
        assert_eq!(
            traces.len(),
            logup_combinations.len(),
            "logup_combinations must have one entry per trace"
        );

        let threads_per_block = THREADS_PER_BLOCK_LOGUP;

        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut air_offsets: Vec<u32> = Vec::with_capacity(traces.len() + 1);
        air_offsets.push(0);

        for (local_air, t) in traces.iter().enumerate() {
            let monomials = pk.per_air[t.air_idx]
                .other_data
                .interaction_monomials
                .as_ref()
                .unwrap();
            let max_monomials = monomials
                .num_numer_monomials
                .max(monomials.num_denom_monomials);
            let mono_blocks = max_monomials.div_ceil(threads_per_block).max(1);
            for y_int in 0..t.num_y {
                for mono_block in 0..mono_blocks {
                    block_ctxs_h.push(BlockCtx {
                        local_block_idx_x: y_int * mono_blocks + mono_block,
                        air_idx: local_air as u32,
                    });
                }
            }
            air_offsets.push(block_ctxs_h.len() as u32);
        }

        let num_blocks = block_ctxs_h.len() as u32;

        let common_ctxs_h: Vec<LogupMonomialCommonCtx> = traces
            .iter()
            .zip(logup_combinations)
            .map(|(t, lc)| {
                let monomials = pk.per_air[t.air_idx]
                    .other_data
                    .interaction_monomials
                    .as_ref()
                    .unwrap();
                let max_monomials = monomials
                    .num_numer_monomials
                    .max(monomials.num_denom_monomials);
                let mono_blocks = max_monomials.div_ceil(threads_per_block).max(1);

                let eval_ctx = EvalCoreCtx {
                    d_selectors: t.sels_ptr,
                    d_preprocessed: t.prep_ptr,
                    d_main: t.main_ptrs_dev.as_ptr(),
                    d_public: t.public_ptr,
                };

                LogupMonomialCommonCtx {
                    eval_ctx,
                    d_eq_xi: t.eq_xi_ptr,
                    bus_term_sum: lc.bus_term_sum,
                    num_y: t.num_y,
                    mono_blocks,
                }
            })
            .collect();
        let numer_ctxs_h: Vec<LogupMonomialCtx> = traces
            .iter()
            .zip(logup_combinations)
            .map(|(t, lc)| {
                let monomials = pk.per_air[t.air_idx]
                    .other_data
                    .interaction_monomials
                    .as_ref()
                    .unwrap();
                LogupMonomialCtx {
                    d_headers: monomials.d_numer_headers.as_ptr(),
                    d_variables: monomials.d_numer_variables.as_ptr(),
                    d_combinations: lc.d_numer_combinations.as_ptr(),
                    num_monomials: monomials.num_numer_monomials,
                }
            })
            .collect();
        let denom_ctxs_h: Vec<LogupMonomialCtx> = traces
            .iter()
            .zip(logup_combinations)
            .map(|(t, lc)| {
                let monomials = pk.per_air[t.air_idx]
                    .other_data
                    .interaction_monomials
                    .as_ref()
                    .unwrap();
                LogupMonomialCtx {
                    d_headers: monomials.d_denom_headers.as_ptr(),
                    d_variables: monomials.d_denom_variables.as_ptr(),
                    d_combinations: lc.d_denom_combinations.as_ptr(),
                    num_monomials: monomials.num_denom_monomials,
                }
            })
            .collect();

        let block_ctxs = block_ctxs_h.to_device();
        let common_ctxs = common_ctxs_h.to_device();
        let numer_ctxs = numer_ctxs_h.to_device();
        let denom_ctxs = denom_ctxs_h.to_device();
        let air_offsets = air_offsets.to_device();

        debug!(
            num_airs = traces.len(),
            num_blocks, "LogupMonomialBatch created"
        );

        Self {
            traces,
            block_ctxs,
            common_ctxs,
            numer_ctxs,
            denom_ctxs,
            air_offsets,
            num_blocks,
        }
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    pub fn evaluate(&self, num_x: u32) -> MetalBuffer<Frac<EF>> {
        let num_airs = self.common_ctxs.len();

        debug!(
            num_blocks = %self.num_blocks,
            %num_x,
            %num_airs,
            "logup_monomial_batched"
        );

        let _tmp_sums =
            MetalBuffer::<Frac<EF>>::with_capacity(self.num_blocks as usize * num_x as usize);
        let output = MetalBuffer::<Frac<EF>>::with_capacity(num_airs * num_x as usize);

        debug_assert_eq!(
            self.air_offsets.len(),
            num_airs + 1,
            "air_offsets must have num_airs + 1 elements"
        );

        // TODO: dispatch logup_monomial_batched Metal kernel when available.
        let _ = (
            &self.block_ctxs,
            &self.common_ctxs,
            &self.numer_ctxs,
            &self.denom_ctxs,
            &self.air_offsets,
        );

        output
    }
}
