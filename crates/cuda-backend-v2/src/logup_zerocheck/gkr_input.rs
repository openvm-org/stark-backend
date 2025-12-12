use std::cmp::max;

use itertools::Itertools;
use openvm_cuda_backend::transpiler::{SymbolicRulesOnGpu, codec::Codec};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
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
use tracing::instrument;

use super::errors::InteractionGpuError;
use crate::{
    EF, F, GpuBackendV2,
    cuda::logup_zerocheck::{
        frac_add_alpha, frac_matrix_vertically_repeat, frac_vector_scalar_multiply_ext_fp,
        logup_gkr_input_eval,
    },
};

const TASK_SIZE: u32 = 65536;

#[allow(dead_code)]
#[derive(Clone)]
pub struct TraceInteractionMeta {
    pub trace_idx: usize,
    pub air_idx: usize,
    pub interactions: Vec<SymbolicInteraction<F>>,
    pub layout_slices: Vec<StackedSlice>,
}

// TODO[jpw]: revisit if this function is needed / precompute the symbolic interactions
pub fn collect_trace_interactions(
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
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

            Some(TraceInteractionMeta {
                trace_idx,
                air_idx: *air_idx,
                interactions,
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
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    l_skip: usize,
    alpha_logup: EF,
    beta_pows: &[EF],
    total_leaves: usize,
) -> Result<DeviceBuffer<Frac<EF>>, InteractionGpuError> {
    if trace_interactions.iter().all(|meta| meta.is_none()) {
        return Ok(DeviceBuffer::new());
    }

    let leaves = DeviceBuffer::<Frac<EF>>::with_capacity(total_leaves);
    leaves.fill_zero()?;
    let null_preprocessed = DeviceBuffer::<F>::new();

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

        let interactions = &meta.interactions;
        let max_fields_len = interactions
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

        // TODO[jpw]: this is a weird way to express the logup denominator with beta_pows
        // symbolically Revisit if it's possible to do it more directly
        let num_interactions = interactions.len();
        let mut frac_expressions = Vec::with_capacity(num_interactions);
        for interaction in interactions.iter() {
            let mut interaction = interaction.clone();
            let b = SymbolicExpression::from_canonical_u32(interaction.bus_index as u32 + 1);
            let betas = symbolic_challenges[1..].to_vec();
            let mut denom = SymbolicExpression::from_canonical_u32(0);
            for (j, expr) in interaction.message.iter().enumerate() {
                denom += betas[j].clone() * expr.clone();
            }
            denom += betas[interaction.message.len()].clone() * b;
            interaction.message = vec![denom];
            frac_expressions.push(interaction);
        }

        let constraints = SymbolicConstraints {
            constraints: vec![],
            interactions: frac_expressions,
        };
        let constraints_dag: SymbolicConstraintsDag<F> = constraints.into();
        let rules = SymbolicRulesOnGpu::new(constraints_dag, true);
        let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();

        let partition_lens = vec![1u32; num_interactions];

        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = rules.used_nodes.to_device()?;
        let d_partition_lens = partition_lens.to_device()?;
        let d_challenges = challenges.to_device()?;

        let d_preprocessed = preprocessed_matrix
            .as_ref()
            .map(|m| m.buffer())
            .unwrap_or(&null_preprocessed);

        let height = air_ctx.height();
        debug_assert_eq!(height, partitioned_main[0].height());
        let partition_ptrs = partitioned_main
            .iter()
            .map(|m| m.buffer().as_ptr() as u64)
            .collect_vec();
        let d_partition_ptrs = partition_ptrs.to_device()?;

        let buffer_size = rules.buffer_size;
        // TODO[jpw]: remove magic 10
        let is_global = buffer_size > 10;
        let intermediates = if is_global {
            DeviceBuffer::<EF>::with_capacity((TASK_SIZE as usize) * buffer_size)
        } else {
            DeviceBuffer::<EF>::with_capacity(1)
        };

        let num_rows_per_tile = height.div_ceil(TASK_SIZE as usize).max(1);

        let mut tmp = DeviceBuffer::new();
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
            tmp = DeviceBuffer::with_capacity(height * num_interactions);
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
                &d_challenges,
                &intermediates,
                &d_rules,
                &d_used_nodes,
                &d_partition_lens,
                num_interactions,
                height as u32,
                num_rows_per_tile as u32,
            )?;
        }
        if height != lifted_height {
            debug_assert_eq!(lifted_height % height, 0);
            debug_assert!(!tmp.is_empty());
            let norm_factor_denom = lifted_height / height;
            let norm_factor = F::from_canonical_usize(norm_factor_denom).inverse();
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
