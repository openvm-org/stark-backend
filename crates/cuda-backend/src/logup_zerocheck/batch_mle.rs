//! Batch MLE evaluation for zerocheck and logup.
//!
//! This module provides builders for batching MLE evaluations across multiple AIRs,
//! enabling efficient GPU kernel launches that process multiple traces in parallel.
//!
//! Small per-batch context tables are staged through [`RoundStager`] so an entire
//! sumcheck round's tables go to the device in a single H2D copy (see
//! [`super::stage`]).

use openvm_cuda_common::{copy::MemCopyD2H, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::{fractional_sumcheck_gkr::Frac, DeviceMultiStarkProvingKey};

use crate::{
    cuda::logup_zerocheck::{
        _logup_batch_mle_intermediates_buffer_size, _zerocheck_batch_mle_intermediates_buffer_size,
        logup_batch_eval_mle, zerocheck_batch_eval_mle, BlockCtx, EvalCoreCtx, LogupCtx,
        MainMatrixPtrs, ZerocheckCtx,
    },
    error::KernelError,
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    logup_zerocheck::{
        block_ctxs::build_block_ctxs,
        mle_round::{evaluate_mle_constraints_gpu, evaluate_mle_interactions_gpu},
        stage::{RoundStager, StagedSlice},
    },
    prelude::{EF, F},
};

const MAX_THREADS_PER_BLOCK: u32 = 128;

// ============================================================================
// Memory calculation helpers
// ============================================================================

/// Computes zerocheck intermediate buffer memory in bytes for a trace.
fn zerocheck_batch_mle_intermediates_buffer_bytes(
    buffer_size: u32,
    num_x: u32,
    num_y: u32,
) -> usize {
    unsafe {
        _zerocheck_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y)
            * std::mem::size_of::<EF>()
    }
}

/// Computes logup intermediate buffer memory in bytes for a trace.
fn logup_batch_mle_intermediates_buffer_bytes(buffer_size: u32, num_x: u32, num_y: u32) -> usize {
    unsafe {
        _logup_batch_mle_intermediates_buffer_size(buffer_size, num_x, num_y)
            * std::mem::size_of::<EF>()
    }
}

// ============================================================================
// Batching helpers
// ============================================================================

/// Find how many traces fit in the memory budget.
/// Returns (count, total_memory_of_batch).
fn find_batch_end<T, M>(traces: &[T], memory_fn: M, memory_limit_bytes: usize) -> (usize, usize)
where
    M: Fn(&T) -> usize,
{
    let mut batch_end = 0;
    let mut batch_memory = 0usize;

    while batch_end < traces.len() {
        let trace_memory = memory_fn(&traces[batch_end]);

        if batch_end == 0 {
            // First trace always included (even if oversized)
            batch_memory = trace_memory;
            batch_end = 1;
            if trace_memory > memory_limit_bytes {
                break; // Single oversized trace
            }
        } else if batch_memory + trace_memory <= memory_limit_bytes {
            batch_memory += trace_memory;
            batch_end += 1;
        } else {
            break; // Would exceed limit
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
    // shared eval pointers (same for zerocheck + logup)
    pub eq_xi_ptr: *const EF,
    pub sels_ptr: *const EF,
    pub prep_ptr: MainMatrixPtrs<EF>,
    /// Device pointer to this trace's `MainMatrixPtrs` table, staged via
    /// [`RoundStager`]. Valid for the duration of the round.
    pub main_ptrs: *const MainMatrixPtrs<EF>,
    pub public_ptr: *const F,
    pub eq_3bs_ptr: *const EF,
}

// NOTE[jpw]: we do not expect to use this since most of the time zerocheck will use monomial_par_y.
// We use DAG evaluation primarily for Poseidon2Air. Consider deleting either the non-batch or batch
// dag version to reduce code duplication.
/// Builder for batched zerocheck MLE evaluation.
///
/// Collects traces and pre-builds all GPU contexts, then evaluates in a single kernel launch.
pub(crate) struct ZerocheckMleBatchBuilder<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: StagedSlice<BlockCtx>,
    num_blocks: usize,
    zc_ctxs: StagedSlice<ZerocheckCtx>,
    air_offsets: StagedSlice<u32>,
    threads_per_block: u32,
    _intermediates_keepalive: Vec<DeviceBuffer<EF>>,
    /// Cheap clone: just `(device_id, Arc<CudaStream>)`.
    device_ctx: GpuDeviceCtx,
}

impl<'a> ZerocheckMleBatchBuilder<'a> {
    /// Creates a new builder from an iterator of traces.
    ///
    /// This constructor filters traces with constraints, computes thread configuration,
    /// and stages all block and zerocheck contexts into `stager`. The caller must
    /// commit the stager cycle before calling [`Self::evaluate`].
    pub fn new<HS: GpuHashScheme>(
        traces: impl Iterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        num_x: u32,
        stager: &mut RoundStager,
        device_ctx: &GpuDeviceCtx,
    ) -> Self {
        let traces: Vec<&TraceCtx> = traces.filter(|t| t.has_constraints).collect();

        // Compute threads_per_block from max_num_y
        let max_num_y = traces.iter().map(|t| t.num_y).max().unwrap_or(0);
        let threads_per_block = max_num_y.min(MAX_THREADS_PER_BLOCK);

        let (block_ctxs_h, air_offsets) =
            build_block_ctxs(traces.iter().map(|t| t.num_y.div_ceil(threads_per_block)));

        // Build ZerocheckCtx for each trace
        let mut intermediates_keepalive: Vec<DeviceBuffer<EF>> = Vec::new();
        let mut zc_ctxs_h: Vec<ZerocheckCtx> = Vec::with_capacity(traces.len());

        for t in traces.iter() {
            let air_pk = &pk.per_air[t.air_idx];
            let buffer_size = air_pk.other_data.zerocheck_mle.inner.buffer_size;

            let d_intermediates = if buffer_size > 0 {
                let intermediates_len = unsafe {
                    _zerocheck_batch_mle_intermediates_buffer_size(buffer_size, num_x, t.num_y)
                };
                let buf = DeviceBuffer::<EF>::with_capacity_on(intermediates_len, device_ctx);
                let ptr = buf.as_mut_ptr();
                intermediates_keepalive.push(buf);
                ptr
            } else {
                std::ptr::null_mut()
            };

            let eval_ctx = EvalCoreCtx {
                d_selectors: t.sels_ptr,
                d_preprocessed: t.prep_ptr,
                d_main: t.main_ptrs,
                d_public: t.public_ptr,
            };

            zc_ctxs_h.push(ZerocheckCtx {
                eval_ctx,
                d_intermediates,
                num_y: t.num_y,
                d_eq_xi: t.eq_xi_ptr,
                d_rules: air_pk.other_data.zerocheck_mle.inner.d_rules.as_raw_ptr(),
                rules_len: air_pk.other_data.zerocheck_mle.inner.d_rules.len(),
                d_used_nodes: air_pk.other_data.zerocheck_mle.inner.d_used_nodes.as_ptr(),
                used_nodes_len: air_pk.other_data.zerocheck_mle.inner.d_used_nodes.len(),
                buffer_size,
            });
        }

        // Stage for upload at the caller's next `commit`
        let num_blocks = block_ctxs_h.len();
        let block_ctxs = stager.push(&block_ctxs_h);
        let zc_ctxs = stager.push(&zc_ctxs_h);
        let air_offsets = stager.push(&air_offsets);

        Self {
            traces,
            block_ctxs,
            num_blocks,
            zc_ctxs,
            air_offsets,
            threads_per_block,
            _intermediates_keepalive: intermediates_keepalive,
            device_ctx: device_ctx.clone(),
        }
    }

    /// Returns true if there are no traces to evaluate.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Returns the trace indices in order.
    pub fn trace_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.traces.iter().map(|t| t.trace_idx)
    }

    /// Evaluates the batch and returns the output device buffer.
    ///
    /// The buffer contains `num_airs * num_x` elements, laid out as
    /// `[air0_x0, air0_x1, ..., air1_x0, air1_x1, ...]`.
    ///
    /// The `stager` cycle this builder staged into must be committed.
    pub fn evaluate(
        &self,
        stager: &RoundStager,
        lambda_pows: &DeviceBuffer<EF>,
        num_x: u32,
    ) -> Result<DeviceBuffer<EF>, KernelError> {
        if self.traces.is_empty() {
            return Ok(DeviceBuffer::new());
        }

        let num_blocks = self.num_blocks;
        let num_airs = self.traces.len();
        tracing::debug!(
            %num_blocks,
            %num_x,
            threads_per_block = %self.threads_per_block,
            %num_airs,
            "zerocheck_batch_eval_mle"
        );
        // Need one buffer slot per block
        let mut tmp_sums_buffer =
            DeviceBuffer::<EF>::with_capacity_on(num_blocks * num_x as usize, &self.device_ctx);
        let mut output =
            DeviceBuffer::<EF>::with_capacity_on(num_airs * num_x as usize, &self.device_ctx);
        // SAFETY: All device pointers in the staged contexts were constructed from valid
        // DeviceBuffers that outlive this call, and the staged tables were committed by the
        // caller. The air_offsets table has length num_airs + 1 by construction.
        unsafe {
            zerocheck_batch_eval_mle(
                &mut tmp_sums_buffer,
                &mut output,
                stager.ptr(self.block_ctxs),
                stager.ptr(self.zc_ctxs),
                stager.ptr(self.air_offsets),
                lambda_pows,
                lambda_pows.len(),
                num_blocks as u32,
                num_x,
                num_airs as u32,
                self.threads_per_block,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        Ok(output)
    }
}

/// Builder for batched logup MLE evaluation.
///
/// Collects traces and pre-builds all GPU contexts, then evaluates in a single kernel launch.
pub(crate) struct LogupMleBatchBuilder<'a> {
    traces: Vec<&'a TraceCtx>,
    block_ctxs: StagedSlice<BlockCtx>,
    num_blocks: usize,
    logup_ctxs: StagedSlice<LogupCtx>,
    air_offsets: StagedSlice<u32>,
    threads_per_block: u32,
    _intermediates_keepalive: Vec<DeviceBuffer<EF>>,
    device_ctx: GpuDeviceCtx,
}

impl<'a> LogupMleBatchBuilder<'a> {
    /// Creates a new builder from an iterator of traces.
    ///
    /// This constructor filters traces with interactions, computes thread configuration,
    /// and stages all block and logup contexts into `stager`. The caller must commit the
    /// stager cycle before calling [`Self::evaluate`].
    pub fn new<HS: GpuHashScheme>(
        traces: impl Iterator<Item = &'a TraceCtx>,
        pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
        d_challenges_ptr: *const EF,
        num_x: u32,
        stager: &mut RoundStager,
        device_ctx: &GpuDeviceCtx,
    ) -> Self {
        let traces: Vec<&TraceCtx> = traces.filter(|t| t.has_interactions).collect();

        // Compute threads_per_block from max_num_y
        let max_num_y = traces.iter().map(|t| t.num_y).max().unwrap_or(0);
        let threads_per_block = max_num_y.min(MAX_THREADS_PER_BLOCK);

        let (block_ctxs_h, air_offsets) =
            build_block_ctxs(traces.iter().map(|t| t.num_y.div_ceil(threads_per_block)));

        // Build LogupCtx for each trace
        let mut intermediates_keepalive: Vec<DeviceBuffer<EF>> = Vec::new();
        let mut logup_ctxs_h: Vec<LogupCtx> = Vec::with_capacity(traces.len());

        for t in traces.iter() {
            let air_pk = &pk.per_air[t.air_idx];
            let buffer_size = air_pk.other_data.interaction_rules.inner.buffer_size;

            let d_intermediates = if buffer_size > 0 {
                let intermediates_len = unsafe {
                    _logup_batch_mle_intermediates_buffer_size(buffer_size, num_x, t.num_y)
                };
                let buf = DeviceBuffer::<EF>::with_capacity_on(intermediates_len, device_ctx);
                let ptr = buf.as_mut_ptr();
                intermediates_keepalive.push(buf);
                ptr
            } else {
                std::ptr::null_mut()
            };

            let eval_ctx = EvalCoreCtx {
                d_selectors: t.sels_ptr,
                d_preprocessed: t.prep_ptr,
                d_main: t.main_ptrs,
                d_public: t.public_ptr,
            };

            logup_ctxs_h.push(LogupCtx {
                eval_ctx,
                d_intermediates,
                num_y: t.num_y,
                d_eq_xi: t.eq_xi_ptr,
                d_challenges: d_challenges_ptr,
                d_eq_3bs: t.eq_3bs_ptr,
                d_rules: air_pk
                    .other_data
                    .interaction_rules
                    .inner
                    .d_rules
                    .as_raw_ptr(),
                rules_len: air_pk.other_data.interaction_rules.inner.d_rules.len(),
                d_used_nodes: air_pk
                    .other_data
                    .interaction_rules
                    .inner
                    .d_used_nodes
                    .as_ptr(),
                d_pair_idxs: air_pk.other_data.interaction_rules.d_pair_idxs.as_ptr(),
                used_nodes_len: air_pk.other_data.interaction_rules.inner.d_used_nodes.len(),
                buffer_size,
            });
        }

        // Stage for upload at the caller's next `commit`
        let num_blocks = block_ctxs_h.len();
        let block_ctxs = stager.push(&block_ctxs_h);
        let logup_ctxs = stager.push(&logup_ctxs_h);
        let air_offsets = stager.push(&air_offsets);

        Self {
            traces,
            block_ctxs,
            num_blocks,
            logup_ctxs,
            air_offsets,
            threads_per_block,
            _intermediates_keepalive: intermediates_keepalive,
            device_ctx: device_ctx.clone(),
        }
    }

    /// Returns true if there are no traces to evaluate.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Returns the trace indices and norm factors in order.
    pub fn trace_info(&self) -> impl Iterator<Item = (usize, F)> + '_ {
        self.traces.iter().map(|t| (t.trace_idx, t.norm_factor))
    }

    /// Evaluates the batch and returns the output device buffer.
    ///
    /// The buffer contains `num_airs * num_x` elements, laid out as
    /// `[air0_x0, air0_x1, ..., air1_x0, air1_x1, ...]`.
    ///
    /// The `stager` cycle this builder staged into must be committed.
    pub fn evaluate(
        &self,
        stager: &RoundStager,
        num_x: u32,
    ) -> Result<DeviceBuffer<Frac<EF>>, KernelError> {
        if self.traces.is_empty() {
            return Ok(DeviceBuffer::new());
        }

        let num_blocks = self.num_blocks;
        let num_airs = self.traces.len();
        // Need one buffer slot per block
        let mut tmp_sums_buffer = DeviceBuffer::<Frac<EF>>::with_capacity_on(
            num_blocks * num_x as usize,
            &self.device_ctx,
        );
        let mut output =
            DeviceBuffer::<Frac<EF>>::with_capacity_on(num_airs * num_x as usize, &self.device_ctx);
        // SAFETY: All device pointers in the staged contexts were constructed from valid
        // DeviceBuffers that outlive this call, and the staged tables were committed by the
        // caller. The air_offsets table has length num_airs + 1 by construction.
        unsafe {
            logup_batch_eval_mle(
                &mut tmp_sums_buffer,
                &mut output,
                stager.ptr(self.block_ctxs),
                stager.ptr(self.logup_ctxs),
                stager.ptr(self.air_offsets),
                num_blocks as u32,
                num_x,
                num_airs as u32,
                self.threads_per_block,
                self.device_ctx.stream.as_raw(),
            )?;
        }
        Ok(output)
    }
}

// ============================================================================
// Memory-aware batched evaluation (DAG paths)
// ============================================================================

/// Batched DAG-based zerocheck evaluation over `traces`, with per-batch memory limits.
///
/// Each memory batch is staged (own stager cycle), launched, and read back before the
/// next batch is constructed, bounding concurrent intermediate-buffer memory by
/// `memory_limit_bytes` as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_zerocheck_batched<'a, HS: GpuHashScheme>(
    traces: impl IntoIterator<Item = &'a TraceCtx>,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    lambda_pows: &DeviceBuffer<EF>,
    num_x: u32,
    zc_out: &mut [Vec<EF>],
    memory_limit_bytes: usize,
    stager: &mut RoundStager,
    device_ctx: &GpuDeviceCtx,
) -> Result<(), KernelError> {
    // Collect traces with constraints and their buffer sizes
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
        return Ok(());
    }

    // Sort by buffer size descending for better bin packing (First Fit Decreasing)
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

        if batch.len() == 1 {
            // Single trace: use non-batch kernel
            let t = batch[0];
            if batch_memory > memory_limit_bytes {
                tracing::warn!(
                    air_idx = t.air_idx,
                    intermediate_buffer_bytes = batch_memory,
                    memory_limit_bytes,
                    "zerocheck: trace exceeds memory limit, using non-batch kernel"
                );
            }
            let rules = &pk.per_air[t.air_idx].other_data.zerocheck_mle;
            let out = evaluate_mle_constraints_gpu(
                t.eq_xi_ptr,
                t.sels_ptr,
                t.prep_ptr,
                t.main_ptrs,
                t.public_ptr,
                lambda_pows,
                rules,
                t.num_y,
                num_x,
                device_ctx,
            )?;
            let out_host = out.to_host_on(device_ctx)?;
            zc_out[t.trace_idx].copy_from_slice(&out_host);
        } else {
            // Normal batch using ZerocheckMleBatchBuilder
            tracing::debug!(
                batch_size = batch.len(),
                batch_memory,
                memory_limit_bytes,
                "zerocheck: batching traces"
            );
            let builder =
                ZerocheckMleBatchBuilder::new(batch.iter().copied(), pk, num_x, stager, device_ctx);
            stager.commit(device_ctx)?;
            let out = builder.evaluate(stager, lambda_pows, num_x)?;
            let host = out.to_host_on(device_ctx)?;

            for (i, trace_idx) in builder.trace_indices().enumerate() {
                let evals = &host[(i * num_x_usize)..((i + 1) * num_x_usize)];
                zc_out[trace_idx].copy_from_slice(evals);
            }
        }
        batch_start += batch_count;
    }
    Ok(())
}

/// Batched DAG-based logup evaluation over high-`num_y` `traces`, with per-batch memory
/// limits. The monomial (low-`num_y`) path is handled by the caller.
///
/// Each memory batch is staged (own stager cycle), launched, and read back before the
/// next batch is constructed, bounding concurrent intermediate-buffer memory by
/// `memory_limit_bytes` as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_logup_batched<'a, HS: GpuHashScheme>(
    traces: impl IntoIterator<Item = &'a TraceCtx>,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    d_challenges_ptr: *const EF,
    num_x: u32,
    logup_out: &mut [[Vec<EF>; 2]],
    logup_tilde_evals: &mut [[EF; 2]],
    memory_limit_bytes: usize,
    stager: &mut RoundStager,
    device_ctx: &GpuDeviceCtx,
) -> Result<(), KernelError> {
    // Collect high traces with interactions and their buffer sizes
    let mut logup_traces_with_size: Vec<(&TraceCtx, usize)> = traces
        .into_iter()
        .filter(|t| t.has_interactions)
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
    if logup_traces_with_size.is_empty() {
        return Ok(());
    }

    // Sort by buffer size descending for better bin packing (First Fit Decreasing)
    logup_traces_with_size.sort_by(|a, b| b.1.cmp(&a.1));

    let num_x_usize = num_x as usize;
    let mut batch_start = 0;

    while batch_start < logup_traces_with_size.len() {
        let (batch_count, batch_memory) = find_batch_end(
            &logup_traces_with_size[batch_start..],
            |(_, mem)| *mem,
            memory_limit_bytes,
        );
        let batch: Vec<&TraceCtx> = logup_traces_with_size[batch_start..batch_start + batch_count]
            .iter()
            .map(|(t, _)| *t)
            .collect();

        if batch.len() == 1 && batch_memory > memory_limit_bytes {
            // Single oversized trace: use non-batch kernel
            let t = batch[0];
            tracing::warn!(
                air_idx = t.air_idx,
                intermediate_buffer_bytes = batch_memory,
                memory_limit_bytes,
                "logup: trace exceeds memory limit, using non-batch kernel"
            );
            evaluate_single_logup(
                t,
                pk,
                d_challenges_ptr,
                num_x,
                &mut logup_out[t.trace_idx],
                &mut logup_tilde_evals[t.trace_idx],
                device_ctx,
            )?;
        } else {
            // Normal batch using LogupMleBatchBuilder
            tracing::debug!(
                batch_size = batch.len(),
                batch_memory,
                memory_limit_bytes,
                "logup: batching traces"
            );
            let builder = LogupMleBatchBuilder::new(
                batch.iter().copied(),
                pk,
                d_challenges_ptr,
                num_x,
                stager,
                device_ctx,
            );
            stager.commit(device_ctx)?;
            let out = builder.evaluate(stager, num_x)?;
            let host = out.to_host_on(device_ctx)?;

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
        }
        batch_start += batch_count;
    }
    Ok(())
}

/// Evaluate logup for a single trace using non-batch kernel.
#[allow(clippy::too_many_arguments)]
fn evaluate_single_logup<HS: GpuHashScheme>(
    t: &TraceCtx,
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    d_challenges_ptr: *const EF,
    num_x: u32,
    logup_out: &mut [Vec<EF>; 2],
    logup_tilde_eval: &mut [EF; 2],
    device_ctx: &GpuDeviceCtx,
) -> Result<(), KernelError> {
    let air_pk = &pk.per_air[t.air_idx];
    let out = evaluate_mle_interactions_gpu(
        t.eq_xi_ptr,
        t.sels_ptr,
        t.prep_ptr,
        t.main_ptrs,
        t.public_ptr,
        d_challenges_ptr,
        t.eq_3bs_ptr,
        &air_pk.other_data.interaction_rules,
        t.num_y,
        num_x,
        device_ctx,
    )?;
    let fracs = out.to_host_on(device_ctx)?;

    if num_x == 1 {
        logup_tilde_eval[0] = fracs[0].p * t.norm_factor;
        logup_tilde_eval[1] = fracs[0].q;
        // logup_out not set, will be handled directly from tilde eval in compute_batch_s
    } else {
        let numer: Vec<EF> = fracs.iter().map(|f| f.p * t.norm_factor).collect();
        let denom: Vec<EF> = fracs.iter().map(|f| f.q).collect();
        *logup_out = [numer, denom];
    }
    Ok(())
}
