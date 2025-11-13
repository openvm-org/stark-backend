use std::{cmp::max, ffi::c_void};

use itertools::Itertools;
use openvm_cuda_backend::transpiler::{SymbolicRulesOnGpu, codec::Codec};
use openvm_cuda_common::{
    copy::{MemCopyH2D, cuda_memcpy},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{
    air_builders::symbolic::{
        SymbolicConstraints, SymbolicConstraintsDag,
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
    },
    interaction::SymbolicInteraction,
    prover::MatrixDimensions,
};
use p3_field::{Field, FieldAlgebra};
use stark_backend_v2::prover::{
    DeviceMultiStarkProvingKeyV2, ProvingContextV2,
    fractional_sumcheck_gkr::Frac,
    stacked_pcs::{StackedLayout, StackedSlice},
};

use super::{errors::InteractionGpuError, matrix_utils::unstack_matrix, state::FractionalGkrState};
use crate::{
    Digest, EF, F, GpuBackendV2,
    cuda::logup_zerocheck::{
        frac_add_alpha, frac_vector_scalar_multiply_ext_fp, zerocheck_eval_interactions_gkr,
    },
    stacked_pcs::StackedPcsDataGpu,
};

const TASK_SIZE: u32 = 65536;

#[allow(dead_code)]
#[derive(Clone)]
pub struct TraceInteractionMeta {
    pub trace_idx: usize,
    pub air_idx: usize,
    // TODO: delete this, can extract from ctx?
    pub lifted_height: usize,
    pub interactions: Vec<SymbolicInteraction<F>>,
    pub layout_slices: Vec<StackedSlice>,
}

pub fn collect_trace_interactions(
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    layout: &StackedLayout,
    l_skip: usize,
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
        .map(|(trace_idx, (air_idx, air_ctx))| {
            let symbolic_constraints =
                SymbolicConstraints::from(&pk.per_air[*air_idx].vk.symbolic_constraints);
            let interactions: Vec<SymbolicInteraction<F>> = symbolic_constraints.interactions;

            if interactions.is_empty() {
                return None;
            }

            let mut layout_entries = vec![None; interactions.len()];
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

            let height = air_ctx.common_main.height();
            let lifted_height = max(height, 1 << l_skip);
            Some(TraceInteractionMeta {
                trace_idx,
                air_idx: *air_idx,
                lifted_height,
                interactions,
                layout_slices,
            })
        })
        .collect()
}

pub fn evaluate_interactions_gpu(
    state: &FractionalGkrState,
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    alpha_logup: EF,
    beta_pows: &[EF],
    total_leaves: usize,
) -> Result<DeviceBuffer<Frac<EF>>, InteractionGpuError> {
    if state.trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok(DeviceBuffer::new());
    }

    let leaves = DeviceBuffer::<Frac<EF>>::with_capacity(total_leaves);
    leaves.fill_zero()?;
    let null_preprocessed = DeviceBuffer::<F>::new();

    for meta in state.trace_interactions.iter().flatten() {
        let air_ctx = &ctx.per_trace[meta.trace_idx].1;
        let pk_air = &pk.per_air[meta.air_idx];

        let preprocessed_matrix = pk_air
            .preprocessed_data
            .as_ref()
            .map(|committed| unstack_matrix(committed.data.as_ref(), 0))
            .transpose()?;

        let mut partitioned_main = Vec::new();
        for committed in &air_ctx.cached_mains {
            partitioned_main.push(unstack_matrix(committed.data.as_ref(), 0)?.0);
        }
        partitioned_main.push(unstack_matrix(&common_main_pcs_data, meta.trace_idx)?.0);

        let all_interactions = &meta.interactions;
        let max_fields_len = all_interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        let betas = beta_pows
            .iter()
            .take(max_fields_len + 1)
            .cloned()
            .collect_vec();

        let challenges = std::iter::once(alpha_logup)
            .chain(betas.iter().cloned())
            .collect_vec();
        let symbolic_challenges: Vec<SymbolicExpression<F>> = (0..challenges.len())
            .map(|index| SymbolicVariable::<F>::new(Entry::Challenge, index).into())
            .collect();

        let mut full_interactions = Vec::new();
        for (interaction_idx, interaction) in all_interactions.iter().enumerate() {
            let mut interaction = interaction.clone();
            let b = SymbolicExpression::from_canonical_u32(interaction.bus_index as u32 + 1);
            let betas = symbolic_challenges[1..].to_vec();
            let mut denom = SymbolicExpression::from_canonical_u32(0);
            for (j, expr) in interaction.message.iter().enumerate() {
                denom += betas[j].clone() * expr.clone();
            }
            denom += betas[interaction.message.len()].clone() * b;
            interaction.message = vec![denom];
            full_interactions.push((interaction_idx, interaction));
        }

        let symbolic_interactions: Vec<SymbolicInteraction<F>> = full_interactions
            .iter()
            .map(|(_, interaction)| interaction.clone())
            .collect();

        let constraints = SymbolicConstraints {
            constraints: vec![],
            interactions: symbolic_interactions.clone(),
        };
        let constraints_dag: SymbolicConstraintsDag<F> = constraints.into();
        let rules = SymbolicRulesOnGpu::new(constraints_dag, true);
        let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();

        let partition_lens = vec![1u32; symbolic_interactions.len()];

        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = rules.used_nodes.to_device()?;
        let d_partition_lens = partition_lens.to_device()?;
        let d_challenges = challenges.to_device()?;

        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.0.buffer())
            .unwrap_or(&null_preprocessed);

        let lifted_height = partitioned_main[0].height();
        debug_assert_eq!(meta.lifted_height, lifted_height);
        let partition_ptrs = partitioned_main
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect_vec();
        let d_partition_ptrs = partition_ptrs.to_device()?;

        let buffer_size = rules.buffer_size;
        let is_global = buffer_size > 10;
        let intermediates = if is_global {
            DeviceBuffer::<EF>::with_capacity((TASK_SIZE as usize) * buffer_size)
        } else {
            DeviceBuffer::<EF>::with_capacity(1)
        };

        let num_rows_per_tile = lifted_height.div_ceil(TASK_SIZE as usize).max(1);

        let trace_output =
            DeviceBuffer::<Frac<EF>>::with_capacity(lifted_height * symbolic_interactions.len());

        unsafe {
            zerocheck_eval_interactions_gkr(
                is_global,
                &trace_output,
                d_preprocessed,
                &d_partition_ptrs,
                &d_challenges,
                &intermediates,
                &d_rules,
                &d_used_nodes,
                &d_partition_lens,
                symbolic_interactions.len(),
                lifted_height as u32,
                num_rows_per_tile as u32,
            )?;
        }
        let height = air_ctx.height();
        if lifted_height != height {
            debug_assert_eq!(lifted_height % height, 0);
            let norm_factor_denom = lifted_height / height;
            let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
            // SAFETY: scaling within buffer length
            unsafe {
                frac_vector_scalar_multiply_ext_fp(
                    trace_output.as_mut_ptr(),
                    norm_factor,
                    trace_output.len() as u32,
                )?;
            }
        }

        // TODO[jpw]: this can be a single memcpy or better to avoid memcpy entirely
        for (local_idx, (orig_idx, _)) in full_interactions.iter().enumerate() {
            let slice = meta.layout_slices[*orig_idx];
            if slice.col_idx != 0 {
                return Err(InteractionGpuError::Layout);
            }
            debug_assert_eq!(slice.len(0), lifted_height);
            let dst_offset = slice.row_idx;

            unsafe {
                cuda_memcpy::<true, true>(
                    leaves.as_mut_ptr().add(dst_offset) as *mut c_void,
                    trace_output.as_ptr().add(local_idx * lifted_height) as *const c_void,
                    lifted_height * size_of::<Frac<EF>>(),
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
