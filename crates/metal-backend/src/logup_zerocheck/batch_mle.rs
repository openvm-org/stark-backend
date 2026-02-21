//! Batch MLE evaluation for zerocheck and logup.
//!
//! This module provides builders for batching MLE evaluations across multiple AIRs,
//! enabling efficient Metal kernel launches that process multiple traces in parallel.

use metal::Buffer as MetalRawBuffer;
use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
use openvm_stark_backend::prover::{fractional_sumcheck_gkr::Frac, DeviceMultiStarkProvingKey};

use crate::{
    logup_zerocheck::{
        batch_mle_monomial::{LogupCombinations, LogupMonomialBatch},
        mle_round::evaluate_mle_interactions_metal,
    },
    metal::logup_zerocheck::{
        logup_batch_eval_mle, logup_batch_mle_intermediates_buffer_size, zerocheck_batch_eval_mle,
        zerocheck_batch_mle_intermediates_buffer_size, BlockCtx, EvalCoreCtx, LogupCtx,
        MainMatrixPtrs, ZerocheckCtx,
    },
    prelude::{EF, F},
    MetalBackend,
};

const MAX_THREADS_PER_BLOCK: u32 = 128;
const LOGUP_SINGLE_TRACE_BATCH_CAP_BYTES: usize = 5 << 30; // 5GiB

#[inline]
fn frac_buffer_to_vec(buf: &MetalBuffer<Frac<EF>>) -> Vec<Frac<EF>> {
    let len = buf.len();
    if len == 0 {
        return Vec::new();
    }
    let p_ptr = buf.as_ptr() as *const EF;
    let q_ptr = unsafe { p_ptr.add(len) };
    (0..len)
        .map(|i| unsafe { Frac::new(*p_ptr.add(i), *q_ptr.add(i)) })
        .collect()
}

// ============================================================================
// Memory calculation helpers
// ============================================================================

fn zerocheck_batch_mle_intermediates_buffer_bytes(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    zerocheck_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y)
        * std::mem::size_of::<EF>()
}

pub(crate) fn logup_batch_mle_intermediates_buffer_bytes(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    logup_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y) * std::mem::size_of::<EF>()
}

// ============================================================================
// Batching helpers
// ============================================================================

fn find_batch_end<T, M>(traces: &[T], memory_fn: M, memory_limit_bytes: usize) -> (usize, usize)
where
    M: Fn(&T) -> usize,
{
    let mut batch_end = 0;
    let mut batch_memory = 0usize;

    while batch_end < traces.len() {
        let trace_memory = memory_fn(&traces[batch_end]);

        if batch_end == 0 {
            batch_memory = trace_memory;
            batch_end = 1;
            if trace_memory > memory_limit_bytes {
                break;
            }
        } else if batch_memory + trace_memory <= memory_limit_bytes {
            batch_memory += trace_memory;
            batch_end += 1;
        } else {
            break;
        }
    }
    (batch_end, batch_memory)
}

/// Context for a single trace used in batch MLE evaluation.
pub(crate) struct TraceCtx {
    pub trace_idx: usize,
    pub air_idx: usize,
    #[allow(dead_code)]
    pub n_lift: usize,
    pub num_y: u32,
    pub has_constraints: bool,
    pub has_interactions: bool,
    pub norm_factor: F,
    pub eq_xi_ptr: *const EF,
    pub sels_ptr: *const EF,
    pub prep_ptr: MainMatrixPtrs<EF>,
    pub main_ptrs_dev: MetalBuffer<MainMatrixPtrs<EF>>,
    pub public_ptr: *const F,
    pub eq_3bs_ptr: *const EF,
    pub read_resources: Vec<MetalRawBuffer>,
}

/// Builder for batched zerocheck MLE evaluation.
pub(crate) struct ZerocheckMleBatchBuilder<'a> {
    traces: Vec<&'a TraceCtx>,
    d_block_ctxs: MetalBuffer<BlockCtx>,
    d_zc_ctxs: MetalBuffer<ZerocheckCtx>,
    air_offsets: MetalBuffer<u32>,
    threads_per_block: u32,
    read_resources: Vec<MetalRawBuffer>,
    _intermediates_keepalive: Vec<MetalBuffer<EF>>,
}

impl<'a> ZerocheckMleBatchBuilder<'a> {
    pub fn new(
        traces: impl Iterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<MetalBackend>,
        num_x: u32,
    ) -> Self {
        let traces: Vec<&TraceCtx> = traces.filter(|t| t.has_constraints).collect();

        if traces.is_empty() {
            return Self {
                traces: vec![],
                d_block_ctxs: MetalBuffer::with_capacity(0),
                d_zc_ctxs: MetalBuffer::with_capacity(0),
                air_offsets: MetalBuffer::with_capacity(0),
                threads_per_block: 0,
                read_resources: vec![],
                _intermediates_keepalive: vec![],
            };
        }

        let max_num_y = traces.iter().map(|t| t.num_y).max().unwrap_or(0);
        let threads_per_block = max_num_y.min(MAX_THREADS_PER_BLOCK);

        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut air_offsets: Vec<u32> = Vec::with_capacity(traces.len() + 1);
        air_offsets.push(0);

        for (local_air, t) in traces.iter().enumerate() {
            let nb = t.num_y.div_ceil(threads_per_block);
            for b in 0..nb {
                block_ctxs_h.push(BlockCtx {
                    local_block_idx_x: b,
                    air_idx: local_air as u32,
                });
            }
            air_offsets.push(block_ctxs_h.len() as u32);
        }

        let mut intermediates_keepalive: Vec<MetalBuffer<EF>> = Vec::new();
        let mut zc_ctxs_h: Vec<ZerocheckCtx> = Vec::with_capacity(traces.len());
        let mut read_resources: Vec<MetalRawBuffer> = Vec::new();

        for t in traces.iter() {
            let air_pk = &pk.per_air[t.air_idx];
            let buffer_size = air_pk.other_data.zerocheck_mle.inner.buffer_size;

            let d_intermediates = if buffer_size > 0 {
                let intermediates_len =
                    zerocheck_batch_mle_intermediates_buffer_size(buffer_size, num_x, t.num_y);
                let buf = MetalBuffer::<EF>::with_capacity(intermediates_len);
                let ptr = buf.as_device_mut_ptr();
                read_resources.push(buf.gpu_buffer().to_owned());
                intermediates_keepalive.push(buf);
                ptr
            } else {
                std::ptr::null_mut()
            };

            read_resources.extend(t.read_resources.iter().cloned());
            read_resources.push(
                air_pk
                    .other_data
                    .zerocheck_mle
                    .inner
                    .d_rules
                    .gpu_buffer()
                    .to_owned(),
            );
            read_resources.push(
                air_pk
                    .other_data
                    .zerocheck_mle
                    .inner
                    .d_used_nodes
                    .gpu_buffer()
                    .to_owned(),
            );

            let eval_ctx = EvalCoreCtx {
                d_selectors: t.sels_ptr as u64,
                d_preprocessed: t.prep_ptr,
                d_main: t.main_ptrs_dev.as_device_ptr() as u64,
                d_public: t.public_ptr as u64,
            };

            zc_ctxs_h.push(ZerocheckCtx {
                eval_ctx,
                d_intermediates: d_intermediates as u64,
                num_y: t.num_y,
                d_eq_xi: t.eq_xi_ptr as u64,
                d_rules: air_pk
                    .other_data
                    .zerocheck_mle
                    .inner
                    .d_rules
                    .as_device_ptr() as u64,
                rules_len: air_pk
                    .other_data
                    .zerocheck_mle
                    .inner
                    .d_rules
                    .len()
                    .try_into()
                    .unwrap(),
                d_used_nodes: air_pk
                    .other_data
                    .zerocheck_mle
                    .inner
                    .d_used_nodes
                    .as_device_ptr() as u64,
                used_nodes_len: air_pk
                    .other_data
                    .zerocheck_mle
                    .inner
                    .d_used_nodes
                    .len()
                    .try_into()
                    .unwrap(),
                buffer_size,
            });
        }

        let d_block_ctxs = block_ctxs_h.to_device();
        let d_zc_ctxs = zc_ctxs_h.to_device();
        let air_offsets = air_offsets.to_device();

        Self {
            traces,
            d_block_ctxs,
            d_zc_ctxs,
            air_offsets,
            threads_per_block,
            read_resources,
            _intermediates_keepalive: intermediates_keepalive,
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    pub fn evaluate(&self, lambda_pows: &MetalBuffer<EF>, num_x: u32) -> MetalBuffer<EF> {
        if self.traces.is_empty() {
            return MetalBuffer::with_capacity(0);
        }

        evaluate_mle_constraints_metal_batch(
            &self.d_block_ctxs,
            &self.d_zc_ctxs,
            &self.air_offsets,
            lambda_pows,
            lambda_pows.len(),
            num_x,
            self.threads_per_block,
            &self.read_resources,
        )
    }
}

/// Builder for batched logup MLE evaluation.
pub(crate) struct LogupMleBatchBuilder<'a> {
    traces: Vec<&'a TraceCtx>,
    d_block_ctxs: MetalBuffer<BlockCtx>,
    d_logup_ctxs: MetalBuffer<LogupCtx>,
    air_offsets: MetalBuffer<u32>,
    threads_per_block: u32,
    _intermediates_keepalive: Vec<MetalBuffer<EF>>,
}

impl<'a> LogupMleBatchBuilder<'a> {
    pub fn new(
        traces: impl Iterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<MetalBackend>,
        d_challenges_ptr: *const EF,
        num_x: u32,
    ) -> Self {
        let traces: Vec<&TraceCtx> = traces.filter(|t| t.has_interactions).collect();

        if traces.is_empty() {
            return Self {
                traces: vec![],
                d_block_ctxs: MetalBuffer::with_capacity(0),
                d_logup_ctxs: MetalBuffer::with_capacity(0),
                air_offsets: MetalBuffer::with_capacity(0),
                threads_per_block: 0,
                _intermediates_keepalive: vec![],
            };
        }

        let max_num_y = traces.iter().map(|t| t.num_y).max().unwrap_or(0);
        let threads_per_block = max_num_y.min(MAX_THREADS_PER_BLOCK);

        let mut block_ctxs_h: Vec<BlockCtx> = Vec::new();
        let mut air_offsets: Vec<u32> = Vec::with_capacity(traces.len() + 1);
        air_offsets.push(0);

        for (local_air, t) in traces.iter().enumerate() {
            let nb = t.num_y.div_ceil(threads_per_block);
            for b in 0..nb {
                block_ctxs_h.push(BlockCtx {
                    local_block_idx_x: b,
                    air_idx: local_air as u32,
                });
            }
            air_offsets.push(block_ctxs_h.len() as u32);
        }

        let mut intermediates_keepalive: Vec<MetalBuffer<EF>> = Vec::new();
        let mut logup_ctxs_h: Vec<LogupCtx> = Vec::with_capacity(traces.len());

        for t in traces.iter() {
            let air_pk = &pk.per_air[t.air_idx];
            let buffer_size = air_pk.other_data.interaction_rules.inner.buffer_size;

            let d_intermediates = if buffer_size > 0 {
                let intermediates_len =
                    logup_batch_mle_intermediates_buffer_size(buffer_size, num_x, t.num_y);
                let buf = MetalBuffer::<EF>::with_capacity(intermediates_len);
                let ptr = buf.as_device_mut_ptr();
                intermediates_keepalive.push(buf);
                ptr
            } else {
                std::ptr::null_mut()
            };

            let eval_ctx = EvalCoreCtx {
                d_selectors: t.sels_ptr as u64,
                d_preprocessed: t.prep_ptr,
                d_main: t.main_ptrs_dev.as_device_ptr() as u64,
                d_public: t.public_ptr as u64,
            };

            logup_ctxs_h.push(LogupCtx {
                eval_ctx,
                d_intermediates: d_intermediates as u64,
                num_y: t.num_y,
                d_eq_xi: t.eq_xi_ptr as u64,
                d_challenges: d_challenges_ptr as u64,
                d_eq_3bs: t.eq_3bs_ptr as u64,
                d_rules: air_pk
                    .other_data
                    .interaction_rules
                    .inner
                    .d_rules
                    .as_device_ptr() as u64,
                rules_len: air_pk
                    .other_data
                    .interaction_rules
                    .inner
                    .d_rules
                    .len()
                    .try_into()
                    .unwrap(),
                d_used_nodes: air_pk
                    .other_data
                    .interaction_rules
                    .inner
                    .d_used_nodes
                    .as_device_ptr() as u64,
                d_pair_idxs: air_pk
                    .other_data
                    .interaction_rules
                    .d_pair_idxs
                    .as_device_ptr() as u64,
                used_nodes_len: air_pk
                    .other_data
                    .interaction_rules
                    .inner
                    .d_used_nodes
                    .len()
                    .try_into()
                    .unwrap(),
                buffer_size,
            });
        }

        let d_block_ctxs = block_ctxs_h.to_device();
        let d_logup_ctxs = logup_ctxs_h.to_device();
        let air_offsets = air_offsets.to_device();

        Self {
            traces,
            d_block_ctxs,
            d_logup_ctxs,
            air_offsets,
            threads_per_block,
            _intermediates_keepalive: intermediates_keepalive,
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    pub fn trace_info(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.traces.iter().map(|t| (t.trace_idx, t.norm_factor))
    }

    pub fn evaluate(&self, num_x: u32) -> MetalBuffer<Frac<EF>> {
        if self.traces.is_empty() {
            return MetalBuffer::with_capacity(0);
        }

        evaluate_mle_interactions_metal_batch(
            &self.d_block_ctxs,
            &self.d_logup_ctxs,
            &self.air_offsets,
            num_x,
            self.threads_per_block,
        )
    }
}

// ============================================================================
// Memory-aware batched evaluation
// ============================================================================

pub(crate) fn evaluate_zerocheck_batched<'a>(
    traces: impl IntoIterator<Item = &'a TraceCtx>,
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    lambda_pows: &MetalBuffer<EF>,
    num_x: u32,
    zc_out: &mut [Vec<EF>],
    memory_limit_bytes: usize,
) {
    let mut zc_traces_with_size: Vec<(&TraceCtx, usize)> = traces
        .into_iter()
        .filter(|t| t.has_constraints)
        .map(|t| {
            let buffer_size = pk.per_air[t.air_idx]
                .other_data
                .zerocheck_mle
                .inner
                .buffer_size;
            let mem = zerocheck_batch_mle_intermediates_buffer_bytes(buffer_size, num_x, t.num_y);
            (t, mem)
        })
        .collect();
    if zc_traces_with_size.is_empty() {
        return;
    }

    zc_traces_with_size.sort_by(|a, b| b.1.cmp(&a.1));

    let num_x_usize = num_x as usize;
    let mut batch_start = 0;

    while batch_start < zc_traces_with_size.len() {
        let (batch_count, batch_memory) = find_batch_end(
            &zc_traces_with_size[batch_start..],
            |(_, mem)| *mem,
            memory_limit_bytes,
        );
        let batch: Vec<&TraceCtx> = zc_traces_with_size[batch_start..batch_start + batch_count]
            .iter()
            .map(|(t, _)| *t)
            .collect();
        if batch.len() == 1 && batch_memory > memory_limit_bytes {
            tracing::warn!(
                air_idx = batch[0].air_idx,
                intermediate_buffer_bytes = batch_memory,
                memory_limit_bytes,
                "zerocheck: trace exceeds memory limit"
            );
        }

        // Always use batched dispatch so indirect resources are declared consistently.
        tracing::debug!(
            batch_size = batch.len(),
            batch_memory,
            memory_limit_bytes,
            "zerocheck: batching traces"
        );
        let builder = ZerocheckMleBatchBuilder::new(batch.iter().copied(), pk, num_x);
        let out = builder.evaluate(lambda_pows, num_x);
        let host = out.to_vec();

        for (i, trace_idx) in builder.trace_indices().enumerate() {
            let evals = &host[(i * num_x_usize)..((i + 1) * num_x_usize)];
            zc_out[trace_idx].copy_from_slice(evals);
        }
        batch_start += batch_count;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_logup_batched(
    traces: &[TraceCtx],
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    d_challenges_ptr: *const EF,
    num_x: u32,
    monomial_num_y_threshold: u32,
    logup_combinations: &[Option<LogupCombinations>],
    logup_out: &mut [[Vec<EF>; 2]],
    logup_tilde_evals: &mut [[EF; 2]],
    memory_limit_bytes: usize,
) {
    let (low_traces, high_traces): (Vec<&TraceCtx>, Vec<&TraceCtx>) = traces
        .iter()
        .filter(|t| t.has_interactions)
        .partition(|t| t.num_y <= monomial_num_y_threshold);
    let mut monomial_trace_count = 0usize;
    let mut monomial_block_count = 0u32;
    let mut mle_batch_count = 0usize;
    let mut mle_trace_count = 0usize;
    let mut fallback_count = 0usize;
    let mut memory_limit_raises = 0usize;
    let mut max_batch_memory = 0usize;
    let mut effective_memory_limit_bytes = memory_limit_bytes;

    if !low_traces.is_empty() {
        let logup_combs: Vec<_> = low_traces
            .iter()
            .map(|t| logup_combinations[t.trace_idx].as_ref().unwrap())
            .collect();
        let batch = LogupMonomialBatch::new(low_traces.iter().copied(), pk, &logup_combs);
        monomial_trace_count = low_traces.len();
        monomial_block_count = batch.num_blocks();
        let out = batch.evaluate(num_x);
        let host = frac_buffer_to_vec(&out);
        let num_x_usize = num_x as usize;
        for (i, trace_idx) in batch.trace_indices().enumerate() {
            let fracs = &host[(i * num_x_usize)..((i + 1) * num_x_usize)];
            let norm = low_traces[i].norm_factor;
            if num_x == 1 {
                logup_tilde_evals[trace_idx][0] = fracs[0].p * norm;
                logup_tilde_evals[trace_idx][1] = fracs[0].q;
            } else {
                for (j, frac) in fracs.iter().enumerate() {
                    logup_out[trace_idx][0][j] = frac.p * norm;
                    logup_out[trace_idx][1][j] = frac.q;
                }
            }
        }
    }

    if !high_traces.is_empty() {
        // Collect high traces with interactions and their buffer sizes
        let mut logup_traces_with_size: Vec<(&TraceCtx, usize)> = high_traces
            .iter()
            .copied()
            .map(|t| {
                let buffer_size = pk.per_air[t.air_idx]
                    .other_data
                    .interaction_rules
                    .inner
                    .buffer_size;
                let mem = logup_batch_mle_intermediates_buffer_bytes(buffer_size, num_x, t.num_y);
                (t, mem)
            })
            .collect();

        if !logup_traces_with_size.is_empty() {
            logup_traces_with_size.sort_by(|a, b| b.1.cmp(&a.1));

            let num_x_usize = num_x as usize;
            let mut batch_start = 0;

            while batch_start < logup_traces_with_size.len() {
                let (batch_count, batch_memory) = find_batch_end(
                    &logup_traces_with_size[batch_start..],
                    |(_, mem)| *mem,
                    effective_memory_limit_bytes,
                );
                max_batch_memory = max_batch_memory.max(batch_memory);
                let batch: Vec<&TraceCtx> = logup_traces_with_size
                    [batch_start..batch_start + batch_count]
                    .iter()
                    .map(|(t, _)| *t)
                    .collect();
                tracing::debug!(
                    batch_size = batch.len(),
                    batch_memory,
                    memory_limit_bytes = effective_memory_limit_bytes,
                    "logup: batching traces"
                );

                if batch.len() == 1 && batch_memory > effective_memory_limit_bytes {
                    let t = batch[0];
                    if batch_memory <= LOGUP_SINGLE_TRACE_BATCH_CAP_BYTES {
                        let old_limit = effective_memory_limit_bytes;
                        effective_memory_limit_bytes = batch_memory;
                        memory_limit_raises += 1;
                        tracing::debug!(
                            air_idx = t.air_idx,
                            intermediate_buffer_bytes = batch_memory,
                            old_memory_limit_bytes = old_limit,
                            new_memory_limit_bytes = effective_memory_limit_bytes,
                            "logup: raising memory limit to keep oversized trace batched"
                        );
                        continue;
                    }
                    fallback_count += 1;
                    tracing::warn!(
                        air_idx = t.air_idx,
                        intermediate_buffer_bytes = batch_memory,
                        memory_limit_bytes = effective_memory_limit_bytes,
                        cap_memory_limit_bytes = LOGUP_SINGLE_TRACE_BATCH_CAP_BYTES,
                        "logup: trace exceeds hard memory cap, using non-batch kernel"
                    );
                    evaluate_single_logup(
                        t,
                        pk,
                        d_challenges_ptr,
                        num_x,
                        &mut logup_out[t.trace_idx],
                        &mut logup_tilde_evals[t.trace_idx],
                    );
                    batch_start += batch_count;
                    continue;
                }

                mle_batch_count += 1;
                mle_trace_count += batch.len();
                let builder =
                    LogupMleBatchBuilder::new(batch.iter().copied(), pk, d_challenges_ptr, num_x);
                let out = builder.evaluate(num_x);
                let host = frac_buffer_to_vec(&out);

                for (i, (trace_idx, norm_factor)) in builder.trace_info().enumerate() {
                    let fracs = &host[(i * num_x_usize)..((i + 1) * num_x_usize)];
                    if num_x == 1 {
                        logup_tilde_evals[trace_idx][0] = fracs[0].p * norm_factor;
                        logup_tilde_evals[trace_idx][1] = fracs[0].q;
                    } else {
                        let numer: Vec<EF> = fracs.iter().map(|f| f.p * norm_factor).collect();
                        let denom: Vec<EF> = fracs.iter().map(|f| f.q).collect();
                        logup_out[trace_idx] = [numer, denom];
                    }
                }
                batch_start += batch_count;
            }
        }
    }

    tracing::info!(
        total_logup_traces = low_traces.len() + high_traces.len(),
        monomial_trace_count,
        monomial_block_count,
        mle_trace_count,
        mle_batch_count,
        fallback_count,
        memory_limit_raises,
        max_batch_memory,
        initial_memory_limit_bytes = memory_limit_bytes,
        effective_memory_limit_bytes,
        "logup_batch_metrics"
    );
}

fn evaluate_single_logup(
    t: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<MetalBackend>,
    d_challenges_ptr: *const EF,
    num_x: u32,
    logup_out: &mut [Vec<EF>; 2],
    logup_tilde_eval: &mut [EF; 2],
) {
    let air_pk = &pk.per_air[t.air_idx];
    let out = evaluate_mle_interactions_metal(
        t.eq_xi_ptr,
        t.sels_ptr,
        t.prep_ptr,
        &t.main_ptrs_dev,
        t.public_ptr,
        d_challenges_ptr,
        t.eq_3bs_ptr,
        &air_pk.other_data.interaction_rules,
        t.num_y,
        num_x,
    );
    let fracs = frac_buffer_to_vec(&out);

    if num_x == 1 {
        logup_tilde_eval[0] = fracs[0].p * t.norm_factor;
        logup_tilde_eval[1] = fracs[0].q;
    } else {
        let numer: Vec<EF> = fracs.iter().map(|f| f.p * t.norm_factor).collect();
        let denom: Vec<EF> = fracs.iter().map(|f| f.q).collect();
        *logup_out = [numer, denom];
    }
}

// ============================================================================
// Metal batch dispatch wrappers
// ============================================================================

fn evaluate_mle_constraints_metal_batch(
    block_ctxs: &MetalBuffer<BlockCtx>,
    zc_ctxs: &MetalBuffer<ZerocheckCtx>,
    air_block_offsets: &MetalBuffer<u32>,
    lambda_pows: &MetalBuffer<EF>,
    lambda_len: usize,
    num_x: u32,
    threads_per_block: u32,
    read_resources: &[MetalRawBuffer],
) -> MetalBuffer<EF> {
    let num_airs = zc_ctxs.len();
    let output = MetalBuffer::<EF>::with_capacity(num_airs * num_x as usize);
    unsafe {
        zerocheck_batch_eval_mle(
            &output,
            block_ctxs,
            zc_ctxs,
            air_block_offsets,
            lambda_pows,
            lambda_len,
            num_x,
            threads_per_block,
            read_resources,
        )
        .expect("zerocheck_batch_eval_mle failed");
    }
    output
}

fn evaluate_mle_interactions_metal_batch(
    block_ctxs: &MetalBuffer<BlockCtx>,
    logup_ctxs: &MetalBuffer<LogupCtx>,
    air_block_offsets: &MetalBuffer<u32>,
    num_x: u32,
    threads_per_block: u32,
) -> MetalBuffer<Frac<EF>> {
    let num_airs = logup_ctxs.len();
    let output = MetalBuffer::<Frac<EF>>::with_capacity(num_airs * num_x as usize);
    unsafe {
        logup_batch_eval_mle(
            &output,
            block_ctxs,
            logup_ctxs,
            air_block_offsets,
            num_x,
            threads_per_block,
        )
        .expect("logup_batch_eval_mle failed");
    }
    output
}
