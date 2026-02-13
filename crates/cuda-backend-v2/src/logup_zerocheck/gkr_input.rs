use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::prover::{
    fractional_sumcheck_gkr::Frac,
    stacked_pcs::{StackedLayout, StackedSlice},
    DeviceMultiStarkProvingKey, MatrixDimensions, ProvingContext,
};
use p3_field::{Field, PrimeCharacteristicRing};
use tracing::instrument;

use super::errors::InteractionGpuError;
use crate::{
    cuda::logup_zerocheck::{
        frac_add_alpha, frac_matrix_vertically_repeat, frac_vector_scalar_multiply_ext_fp,
        logup_gkr_input_eval,
    },
    GpuBackend, EF, F,
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
pub fn collect_trace_interactions(
    pk: &DeviceMultiStarkProvingKey<GpuBackend>,
    ctx: &ProvingContext<GpuBackend>,
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
/// Returns separate buffers for numerators (F) and denominators (EF) to save memory.
#[instrument(name = "prover.rap_constraints.logup_gkr.input_evals", skip_all)]
pub fn log_gkr_input_evals(
    trace_interactions: &[Option<TraceInteractionMeta>],
    pk: &DeviceMultiStarkProvingKey<GpuBackend>,
    ctx: &ProvingContext<GpuBackend>,
    l_skip: usize,
    alpha_logup: EF,
    d_challenges: &DeviceBuffer<EF>,
    total_leaves: usize,
) -> Result<DeviceBuffer<Frac<EF>>, InteractionGpuError> {
    if trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok(DeviceBuffer::new());
    }

    let leaves = DeviceBuffer::<Frac<EF>>::with_capacity(total_leaves);
    leaves.fill_zero()?;
    let null_preprocessed = DeviceBuffer::<F>::new();

    let mut d_partition_ptrs = DeviceBuffer::<u64>::new();
    let mut tmp = DeviceBuffer::<Frac<EF>>::new();
    for meta in trace_interactions.iter().flatten() {
        let air_ctx = &ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];

        let preprocessed_matrix = pk_air
            .preprocessed_data
            .as_ref()
            .map(|committed| &committed.trace);

        let mut partitioned_main = Vec::with_capacity(air_ctx.cached_mains.len() + 1);
        for committed in &air_ctx.cached_mains {
            partitioned_main.push(&committed.trace);
        }
        partitioned_main.push(&air_ctx.common_main);

        let rules = &pk_air.other_data.interaction_rules;
        let num_interactions = pk_air.vk.symbolic_constraints.interactions.len();

        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.buffer())
            .unwrap_or(&null_preprocessed);
        let d_public_values = if air_ctx.public_values.is_empty() {
            DeviceBuffer::<F>::new()
        } else {
            air_ctx.public_values.to_device().unwrap()
        };

        let height = air_ctx.height();
        debug_assert_eq!(height, partitioned_main[0].height());
        let partition_ptrs = partitioned_main
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect_vec();
        if partition_ptrs.len() > d_partition_ptrs.len() {
            d_partition_ptrs = DeviceBuffer::with_capacity(partition_ptrs.len());
        }
        partition_ptrs.copy_to(&mut d_partition_ptrs)?;

        let buffer_size = rules.inner.buffer_size;
        // TODO[jpw]: remove magic 10
        let is_global = buffer_size > 10;
        let intermediates = if is_global {
            DeviceBuffer::<EF>::with_capacity((TASK_SIZE as usize) * buffer_size as usize)
        } else {
            DeviceBuffer::<EF>::with_capacity(1)
        };

        let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1);

        let slice = meta.layout_slices.first().unwrap();
        if slice.col_idx != 0 {
            return Err(InteractionGpuError::Layout);
        }
        let dst_offset = slice.row_idx;
        let lifted_height = max(height, 1 << l_skip);
        debug_assert_eq!(slice.len(l_skip), lifted_height);
        // SAFETY: by definition of interactions stacked layout, `leaves` has enough capacity
        let leaves_ptr = unsafe { leaves.as_mut_ptr().add(dst_offset) };

        let trace_output = if height != lifted_height {
            let required = height * num_interactions;
            if required > tmp.len() {
                tmp = DeviceBuffer::with_capacity(required);
            }
            tmp.as_mut_ptr()
        } else {
            leaves_ptr
        };
        unsafe {
            logup_gkr_input_eval(
                is_global,
                trace_output,
                d_preprocessed,
                &d_partition_ptrs,
                &d_public_values,
                d_challenges,
                &intermediates,
                &rules.inner.d_rules,
                &rules.inner.d_used_nodes,
                &rules.d_pair_idxs,
                height as u32,
                num_rows_per_tile as u32,
            )?;
        }
        if height != lifted_height {
            debug_assert_eq!(lifted_height % height, 0);
            debug_assert!(!tmp.is_empty());
            let norm_factor_denom = lifted_height / height;
            let norm_factor = F::from_usize(norm_factor_denom).inverse();
            unsafe {
                // SAFETY: scaling within buffer length
                frac_vector_scalar_multiply_ext_fp(
                    tmp.as_mut_ptr(),
                    norm_factor,
                    tmp.len() as u32,
                )?;
                // SAFETY: stacked interaction layout is defined with respect to lifted height so
                // lifting (i.e., vertically repeating) stays within bounds
                frac_matrix_vertically_repeat(
                    leaves_ptr,
                    tmp.as_ptr(),
                    num_interactions as u32,
                    lifted_height as u32,
                    height as u32,
                )?;
            }
        }
    }

    if !leaves.is_empty() {
        unsafe {
            frac_add_alpha(&leaves, alpha_logup)?;
        }
    }

    Ok(leaves)
}
