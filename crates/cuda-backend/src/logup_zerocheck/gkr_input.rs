use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::{
    fractional_sumcheck_gkr::Frac,
    stacked_pcs::{StackedLayout, StackedSlice},
    DeviceMultiStarkProvingKey, ProvingContext,
};
use p3_field::{Field, PrimeCharacteristicRing};
use tracing::instrument;

use super::errors::InteractionGpuError;
use crate::{
    cuda::logup_zerocheck::{
        frac_matrix_vertically_repeat, frac_vector_scalar_multiply_ext_fp,
        gkr_input_intermediates_buffer_size, logup_gkr_input_eval, GkrInputCtx,
    },
    gpu_backend::GenericGpuBackend,
    hash_scheme::GpuHashScheme,
    prelude::{EF, F},
};

const TASK_SIZE: u32 = 65536;

#[allow(dead_code)]
#[derive(Clone)]
pub struct TraceInteractionMeta {
    pub trace_idx: usize,
    pub air_idx: usize,
    pub layout_slices: Vec<StackedSlice>,
}

// TODO[jpw]: revisit if this function is needed
pub fn collect_trace_interactions<HS: GpuHashScheme>(
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    ctx: &ProvingContext<GenericGpuBackend<HS>>,
    layout: &StackedLayout,
) -> Vec<Option<TraceInteractionMeta>> {
    // Pre-group layout slices by trace to avoid repeated scans later.
    let mut slices_by_trace: Vec<Vec<(usize, StackedSlice)>> =
        vec![Vec::new(); ctx.per_trace.len()];
    for &(trace_idx, interaction_idx, ref slice) in &layout.sorted_cols {
        if let Some(entries) = slices_by_trace.get_mut(trace_idx) {
            entries.push((interaction_idx, *slice));
        }
    }

    ctx.per_trace
        .iter()
        .enumerate()
        .map(|(trace_idx, (air_idx, _))| {
            let vk = &pk.per_air[*air_idx].vk;
            if !vk.has_interaction() {
                return None;
            }

            let mut layout_entries = vec![None; vk.num_interactions()];
            for (interaction_idx, slice) in &slices_by_trace[trace_idx] {
                if let Some(slot) = layout_entries.get_mut(*interaction_idx) {
                    *slot = Some(*slice);
                }
            }

            let layout_slices = layout_entries
                .into_iter()
                .enumerate()
                .map(|(idx, maybe_slice)| {
                    maybe_slice.unwrap_or_else(|| {
                        panic!(
                            "missing stacked slice for interaction {} of trace {}",
                            idx, trace_idx
                        )
                    })
                })
                .collect_vec();

            Some(TraceInteractionMeta {
                trace_idx,
                air_idx: *air_idx,
                layout_slices,
            })
        })
        .collect()
}

/// Evaluate interactions from trace evaluation matrices to get (p, q) fractional sumcheck input.
/// Returns real leaves buffer (WITHOUT alpha applied) and alpha value to be applied in first tree
/// layer. Virtual padding is not materialized.
///
/// Uses partial batching: groups AIRs into batches that fit within a memory budget to increase
/// GPU parallelism without increasing peak memory usage. The batch kernel always uses global
/// intermediates, so the memory budget controls peak usage.
#[instrument(name = "prover.rap_constraints.logup_gkr.input_evals", skip_all)]
#[allow(clippy::too_many_arguments)]
pub fn log_gkr_input_evals<HS: GpuHashScheme>(
    trace_interactions: &[Option<TraceInteractionMeta>],
    pk: &DeviceMultiStarkProvingKey<GenericGpuBackend<HS>>,
    proving_ctx: &ProvingContext<GenericGpuBackend<HS>>,
    l_skip: usize,
    alpha_logup: EF,
    d_challenges: &DeviceBuffer<EF>,
    real_len: usize,
    memory_budget_bytes: usize,
    device_ctx: &GpuDeviceCtx,
) -> Result<(DeviceBuffer<Frac<EF>>, EF), InteractionGpuError> {
    if trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok((DeviceBuffer::new(), alpha_logup));
    }

    let leaves = DeviceBuffer::<Frac<EF>>::with_capacity_on(real_len, device_ctx);
    leaves.fill_zero_on(device_ctx)?;
    let null_preprocessed = DeviceBuffer::<F>::new();

    let stream = device_ctx.stream.as_raw();

    // Collect all AIRs with interactions
    let metas: Vec<&TraceInteractionMeta> = trace_interactions.iter().flatten().collect();

    // Pre-compute per-AIR batch planning info
    struct AirPlan {
        height: usize,
        num_interactions: usize,
        intermediate_elements: usize,
        intermediate_bytes: usize,
        lifted_height: usize,
        dst_offset: usize,
        needs_lifting: bool,
    }

    let mut plans: Vec<AirPlan> = Vec::with_capacity(metas.len());
    for meta in &metas {
        let air_ctx = &proving_ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];
        let height = air_ctx.height();
        let num_interactions = pk_air.vk.symbolic_constraints.interactions.len();
        let buffer_size = pk_air.other_data.interaction_rules.inner.buffer_size;

        let intermediate_elements = gkr_input_intermediates_buffer_size(buffer_size);
        let intermediate_bytes = intermediate_elements * std::mem::size_of::<EF>();

        let slice = meta.layout_slices.first().unwrap();
        if slice.col_idx != 0 {
            return Err(InteractionGpuError::Layout);
        }
        let dst_offset = slice.row_idx;
        let lifted_height = max(height, 1 << l_skip);
        debug_assert_eq!(slice.len(l_skip), lifted_height);

        plans.push(AirPlan {
            height,
            num_interactions,
            intermediate_elements,
            intermediate_bytes,
            lifted_height,
            dst_offset,
            needs_lifting: height != lifted_height,
        });
    }

    let mut plan_start: usize = 0;
    while plan_start < plans.len() {
        // Greedy batching: add AIRs while cumulative intermediates fit within budget
        let mut batch_end = plan_start + 1;
        let mut batch_intermediate_bytes = plans[plan_start].intermediate_bytes;
        while batch_end < plans.len() {
            let next_bytes = batch_intermediate_bytes + plans[batch_end].intermediate_bytes;
            if next_bytes > memory_budget_bytes {
                break;
            }
            batch_intermediate_bytes = next_bytes;
            batch_end += 1;
        }

        let count = batch_end - plan_start;

        // Single tmp allocation per batch, partitioned across lifting AIRs by offset.
        // (One alloc beats N separate `with_capacity_on` calls; lifting AIRs in the same batch
        // can't share one buffer because the kernel writes them concurrently.)
        let total_lift_elements: usize = plans[plan_start..batch_end]
            .iter()
            .filter(|p| p.needs_lifting)
            .map(|p| p.height * p.num_interactions)
            .sum();
        let tmp_buf = (total_lift_elements > 0)
            .then(|| DeviceBuffer::<Frac<EF>>::with_capacity_on(total_lift_elements, device_ctx));
        let mut tmp_offset: usize = 0;

        // Build GkrInputCtx for each AIR in this batch
        let mut ctxs_host: Vec<GkrInputCtx> = Vec::with_capacity(count);
        let mut intermediates_keepalive: Vec<DeviceBuffer<EF>> = Vec::with_capacity(count);
        let mut main_ptrs_keepalive: Vec<DeviceBuffer<u64>> = Vec::with_capacity(count);
        let mut public_keepalive: Vec<DeviceBuffer<F>> = Vec::with_capacity(count);

        for i in 0..count {
            let plan = &plans[plan_start + i];
            let meta = metas[plan_start + i];
            let air_ctx = &proving_ctx.per_trace[meta.trace_idx].1;
            let pk_air = &pk.per_air[meta.air_idx];

            let preprocessed_matrix = pk_air
                .preprocessed_data
                .as_ref()
                .map(|committed| &committed.trace);
            let d_preprocessed = preprocessed_matrix
                .as_ref()
                .map(|m| m.buffer())
                .unwrap_or(&null_preprocessed);

            let mut partitioned_main = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
            for committed in &air_ctx.cached_mains {
                partitioned_main.push(&committed.trace);
            }
            partitioned_main.push(&air_ctx.common_main);
            let main_ptrs: Vec<u64> = partitioned_main
                .iter()
                .map(|m| m.buffer().as_ptr() as u64)
                .collect_vec();
            let d_main_ptrs = main_ptrs.to_device_on(device_ctx)?;
            let main_ptr = d_main_ptrs.as_ptr();
            main_ptrs_keepalive.push(d_main_ptrs);

            let d_public_values = if air_ctx.public_values.is_empty() {
                DeviceBuffer::<F>::new()
            } else {
                air_ctx.public_values.to_device_on(device_ctx)?
            };
            let public_ptr = d_public_values.as_ptr();
            public_keepalive.push(d_public_values);

            let intermediates = if plan.intermediate_elements > 0 {
                let buf =
                    DeviceBuffer::<EF>::with_capacity_on(plan.intermediate_elements, device_ctx);
                let ptr = buf.as_mut_ptr();
                intermediates_keepalive.push(buf);
                ptr
            } else {
                std::ptr::null_mut()
            };

            let trace_output_ptr = if plan.needs_lifting {
                let ptr = unsafe { tmp_buf.as_ref().unwrap().as_mut_ptr().add(tmp_offset) };
                tmp_offset += plan.height * plan.num_interactions;
                ptr
            } else {
                unsafe { leaves.as_mut_ptr().add(plan.dst_offset) }
            };

            let num_rows_per_tile = plan.height.div_ceil(TASK_SIZE as usize).max(1);
            let rules = &pk_air.other_data.interaction_rules;

            ctxs_host.push(GkrInputCtx {
                d_fracs: trace_output_ptr,
                d_preprocessed: d_preprocessed.as_ptr(),
                d_main: main_ptr,
                d_public_values: public_ptr,
                d_challenges: d_challenges.as_ptr(),
                d_intermediates: intermediates,
                d_rules: rules.inner.d_rules.as_raw_ptr(),
                d_used_nodes: rules.inner.d_used_nodes.as_ptr(),
                d_pair_idxs: rules.d_pair_idxs.as_ptr(),
                used_nodes_len: rules.inner.d_used_nodes.len(),
                height: plan.height as u32,
                num_rows_per_tile: num_rows_per_tile as u32,
            });
        }

        let d_ctxs = ctxs_host.to_device_on(device_ctx)?;
        unsafe {
            logup_gkr_input_eval(&d_ctxs, count as u32, stream)?;
        }

        // Handle lifting (normalization + vertical repeat) for AIRs that need it
        let mut lift_offset: usize = 0;
        for i in 0..count {
            let plan = &plans[plan_start + i];
            if plan.needs_lifting {
                let len = plan.height * plan.num_interactions;
                debug_assert_eq!(plan.lifted_height % plan.height, 0);
                let norm_factor_denom = plan.lifted_height / plan.height;
                let norm_factor = F::from_usize(norm_factor_denom).inverse();
                let leaves_ptr = unsafe { leaves.as_mut_ptr().add(plan.dst_offset) };
                let slice_ptr = unsafe { tmp_buf.as_ref().unwrap().as_mut_ptr().add(lift_offset) };
                lift_offset += len;
                unsafe {
                    frac_vector_scalar_multiply_ext_fp(slice_ptr, norm_factor, len as u32, stream)?;
                    frac_matrix_vertically_repeat(
                        leaves_ptr,
                        slice_ptr,
                        plan.num_interactions as u32,
                        plan.lifted_height as u32,
                        plan.height as u32,
                        stream,
                    )?;
                }
            }
        }

        plan_start = batch_end;
    }

    // NOTE: alpha is NO LONGER applied here - it will be fused into the first tree layer
    // in fractional_sumcheck_gpu for better performance (eliminates one memory pass)
    Ok((leaves, alpha_logup))
}
