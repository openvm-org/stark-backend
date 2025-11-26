#![allow(dead_code)] // Temporary: keeping upsampling implementation
use std::{cmp::min, collections::HashMap, iter::zip, mem::ManuallyDrop, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{
    base::{DeviceMatrix, DeviceMatrixView},
    ntt::batch_ntt,
    transpiler::{SymbolicRulesOnGpu, codec::Codec},
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::{
        SymbolicConstraints, SymbolicConstraintsDag, SymbolicExpressionDag, SymbolicExpressionNode,
        symbolic_variable::Entry, topological_sort_symbolic_expr,
    },
    prover::MatrixDimensions,
};
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;
use rustc_hash::FxHashMap;
use stark_backend_v2::prover::{
    AirProvingContextV2, CommittedTraceDataV2, DeviceStarkProvingKeyV2,
    fractional_sumcheck_gkr::Frac,
};
use tracing::debug;

use super::{
    dag_scheduling::compute_constraint_expr_indices,
    errors::{Round0EvalError, Round0PrepError},
};
use crate::{
    Digest, EF, F, GpuBackendV2,
    cuda::{
        logup_zerocheck::{
            _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
            _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
            MainMatrixPtrs, logup_bary_eval_interactions_round0, zerocheck_bary_eval_constraints,
        },
        matrix::{batch_expand_pad_wide, batch_rotate_pad, lift_padded_matrix_evals},
    },
    stacked_pcs::StackedPcsDataGpu,
};

const TASK_SIZE: u32 = 65_536;

/// For a single AIR.
/// Each matrix comes with boolean `needs_rotation` for whether width is doubled.
#[derive(Debug)]
struct TraceRound0Matrices {
    preprocessed: Option<(DeviceMatrix<F>, bool)>,
    cached: Vec<(DeviceMatrix<F>, bool)>,
    common: (DeviceMatrix<F>, bool),
}

#[derive(Debug)]
pub(crate) struct Round0TraceInput<'a> {
    selectors_large: DeviceMatrix<F>,
    trace_mats: TraceRound0Matrices,
    main_ptrs: DeviceBuffer<MainMatrixPtrs<F>>,
    eq_x: DeviceMatrix<EF>,
    public_values: &'a DeviceBuffer<F>,
}

/// For single AIR
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_round0_trace_input<'a>(
    l_skip: usize,
    log_large_domain: usize,
    pk: &DeviceStarkProvingKeyV2<GpuBackendV2>,
    air_ctx: &AirProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    selectors_base: &DeviceMatrix<F>,
    eq_x: DeviceMatrix<EF>,
    public_values: &'a [DeviceBuffer<F>],
    trace_idx: usize,
) -> Result<Round0TraceInput<'a>, Round0PrepError> {
    // TODO: follow stacked_reduction strategy where univariate variable of sel is evaluated
    // directly
    let selectors_base_buf = selectors_base.buffer();
    let selectors_large_buf =
        DeviceBuffer::with_capacity(selectors_base_buf.len() << (log_large_domain - l_skip));
    let num_poly = (selectors_base_buf.len() >> l_skip) as u32;
    unsafe {
        batch_expand_pad_wide(
            selectors_large_buf.as_mut_ptr(),
            selectors_base_buf.as_ptr(),
            num_poly,
            1 << log_large_domain,
            1 << l_skip,
        )?;
        batch_ntt(
            &selectors_large_buf,
            l_skip as u32,
            (log_large_domain - l_skip) as u32,
            num_poly,
            true,
            true,
        );
        batch_ntt(
            &selectors_large_buf,
            log_large_domain as u32,
            0,
            num_poly,
            true,
            false,
        );
    }
    let selectors_large = DeviceMatrix::new(
        Arc::new(selectors_large_buf),
        selectors_base.height() << (log_large_domain - l_skip),
        selectors_base.width(),
    );
    let trace_mats = prepare_trace_round0_matrices(
        l_skip,
        log_large_domain,
        &pk.vk.symbolic_constraints.constraints,
        pk.preprocessed_data.as_ref(),
        air_ctx,
        common_main_pcs_data,
        trace_idx,
    )?;
    debug_assert_eq!(selectors_large.height(), trace_mats.common.0.height());

    let partition_ptrs_host = collect_partition_ptrs(&trace_mats);
    let d_main_ptrs = partition_ptrs_host.to_device()?;

    let public_values = &public_values[trace_idx];

    Ok(Round0TraceInput {
        selectors_large,
        trace_mats,
        main_ptrs: d_main_ptrs,
        eq_x,
        public_values,
    })
}

/// Evaluate plain AIR constraints (not interactions) for a single AIR, given prepared trace input.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_gpu(
    pk: &DeviceStarkProvingKeyV2<GpuBackendV2>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<F>,
    eq_uni: &DeviceBuffer<EF>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    large_domain: u32,
    skip_domain: u32,
    num_x: u32,
    height: u32,
) -> Result<DeviceBuffer<EF>, Round0EvalError> {
    let constraints_dag = &pk.vk.symbolic_constraints;
    if constraints_dag.constraints.constraint_idx.is_empty() {
        // No plain AIR constraints, return empty buffer
        return Ok(DeviceBuffer::new());
    }

    let lambda_index_map: HashMap<usize, usize> = constraints_dag
        .constraints
        .constraint_idx
        .iter()
        .enumerate()
        .map(|(idx, dag_idx)| (*dag_idx, idx))
        .collect();
    let constraint_dag_indices = compute_constraint_expr_indices(constraints_dag);
    let rules = SymbolicRulesOnGpu::new(constraints_dag.clone(), false);

    let lambda_indices_host: Vec<u32> = rules
        .used_nodes
        .iter()
        .map(|&constraint_idx| {
            constraint_dag_indices
                .get(constraint_idx)
                .and_then(|dag_idx| lambda_index_map.get(dag_idx))
                .copied()
                .unwrap_or(0) as u32
        })
        .collect();
    let d_lambda_indices = lambda_indices_host.to_device()?;

    let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;
    let d_used_nodes = rules.used_nodes.to_device()?;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity =
        unsafe { _zerocheck_r0_intermediates_buffer_size(buffer_size, large_domain, num_x) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<F>::with_capacity(intermed_capacity as usize)
    } else {
        DeviceBuffer::<F>::new()
    };

    let tmp_sums_buffer_capacity =
        unsafe { _zerocheck_r0_temp_sums_buffer_size(buffer_size, large_domain, num_x) };
    debug!("zerocheck:tmp_sums_buffer_capacity={tmp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<EF>::with_capacity(tmp_sums_buffer_capacity as usize);

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut s_evals = DeviceBuffer::<EF>::with_capacity(large_domain as usize);
    // SAFETY:
    // - No bounds checks are done in this kernel. It fully assumes that the Rules are trusted and
    //   all nodes are valid.
    unsafe {
        zerocheck_bary_eval_constraints(
            &mut temp_sums_buffer,
            &mut s_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            omega_skip_pows,
            inv_lagrange_denoms,
            eq_uni,
            eq_cube,
            lambda_pows,
            &d_lambda_indices,
            public_values,
            &d_rules,
            &d_used_nodes,
            buffer_size,
            &mut intermediates,
            large_domain,
            skip_domain,
            num_x,
            height,
        )?;
    }

    Ok(s_evals)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct InteractionNode {
    // Interaction index
    pub idx: u32,
    // 0 means numerator (count)
    // id > 0 means denominator term to multiply by beta_pows[id], i.e., id - 1 is message index
    pub beta_idx: u32,
}

/// Evaluate interaction constraints (excluding plain AIR constraints) for a single AIR, given
/// prepared trace input.
///
/// `constraints` includes interaction expressions for the AIR.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn evaluate_round0_interactions_gpu(
    pk: &DeviceStarkProvingKeyV2<GpuBackendV2>,
    symbolic: &SymbolicConstraints<F>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    omega_skip_pows: &DeviceBuffer<F>,
    inv_lagrange_denoms: &DeviceBuffer<F>,
    eq_sharp_uni: &DeviceBuffer<EF>,
    eq_cube: *const EF,
    beta_pows: &[EF],
    eq_3bs: &[EF],
    large_domain: u32,
    skip_domain: u32,
    num_x: u32,
    height: u32,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    // Check if this trace has interactions
    if eq_3bs.is_empty() {
        return Ok(DeviceBuffer::new());
    }

    // We create a new "interactions DAG" where the new .constraints are the interaction [count,
    // message_0, message_1, ..] expressions themselves, while the .interactions are empty
    // We track the indices with InteractionNode

    // Copied from build_symbolic_constraints_dag to handle sorting of constraints
    // TODO: rework this for interaction chunking
    let (rules, d_numer_weights, d_denom_weights, denom_sum_init) = {
        let mut expr_to_idx = FxHashMap::default();
        let mut nodes = Vec::new();
        let mut sorted_used_dag_idxs = Vec::new();
        for interaction in &symbolic.interactions {
            let count =
                topological_sort_symbolic_expr(&interaction.count, &mut expr_to_idx, &mut nodes);
            sorted_used_dag_idxs.push(count);
            sorted_used_dag_idxs.extend(interaction.message.iter().map(|field_expr| {
                topological_sort_symbolic_expr(field_expr, &mut expr_to_idx, &mut nodes)
            }));
        }
        sorted_used_dag_idxs.sort();
        // TODO: SymbolicRulesOnGpu::new already has this
        let dag_idx_to_used_idx = FxHashMap::from_iter(
            sorted_used_dag_idxs
                .iter()
                .enumerate()
                .map(|(used_idx, dag_idx)| (*dag_idx, used_idx)),
        );
        let constraints = SymbolicExpressionDag {
            nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let interactions_dag = SymbolicConstraintsDag {
            constraints,
            interactions: vec![],
        };
        let rules = SymbolicRulesOnGpu::new(interactions_dag, false);
        let mut numer_weights = vec![EF::ZERO; rules.constraints.len()];
        let mut denom_weights = vec![EF::ZERO; rules.constraints.len()];
        let mut denom_sum_init = EF::ZERO;
        for (interaction_idx, interaction) in symbolic.interactions.iter().enumerate() {
            // CAUTION: an expression node could be used in multiple interactions, and might even be
            // used as `count` in one, but message field in another. We only care about their
            // weighted sum with eq_3b, so we compute the weights ahead of time.
            let count_dag_idx = expr_to_idx[&interaction.count];
            let count_used_idx = dag_idx_to_used_idx[&count_dag_idx];
            let count_rule_idx = rules.used_nodes[count_used_idx];
            numer_weights[count_rule_idx] += eq_3bs[interaction_idx];
            denom_sum_init += eq_3bs[interaction_idx]
                * beta_pows[interaction.message.len()]
                * F::from_canonical_u32(interaction.bus_index as u32 + 1);

            for (message_idx, message) in interaction.message.iter().enumerate() {
                let message_dag_idx = expr_to_idx[message];
                let message_used_idx = dag_idx_to_used_idx[&message_dag_idx];
                let message_rule_idx = rules.used_nodes[message_used_idx];
                denom_weights[message_rule_idx] += eq_3bs[interaction_idx] * beta_pows[message_idx];
            }
        }
        let d_numer_weights = numer_weights.to_device()?;
        let d_denom_weights = denom_weights.to_device()?;
        (rules, d_numer_weights, d_denom_weights, denom_sum_init)
    };

    let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity =
        unsafe { _logup_r0_intermediates_buffer_size(buffer_size, large_domain, num_x) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup_r0:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<F>::with_capacity(intermed_capacity as usize)
    } else {
        DeviceBuffer::<F>::new()
    };

    let tmp_sums_buffer_capacity =
        unsafe { _logup_r0_temp_sums_buffer_size(buffer_size, large_domain, num_x) };
    debug!("logup_r0:tmp_sums_buffer_capacity={tmp_sums_buffer_capacity}");
    let mut temp_sums_buffer =
        DeviceBuffer::<Frac<EF>>::with_capacity(tmp_sums_buffer_capacity as usize);

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut s_evals = DeviceBuffer::<Frac<EF>>::with_capacity(large_domain as usize);

    unsafe {
        logup_bary_eval_interactions_round0(
            &mut temp_sums_buffer,
            &mut s_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            omega_skip_pows,
            inv_lagrange_denoms,
            eq_sharp_uni,
            eq_cube,
            public_values,
            &d_numer_weights,
            &d_denom_weights,
            denom_sum_init,
            &d_rules,
            buffer_size,
            &mut intermediates,
            large_domain,
            skip_domain,
            num_x,
            height,
        )?;
    }

    Ok(s_evals)
}

// For a single present AIR.
// `constraints_dag` includes nodes for plain AIR constraints and interaction expressions.
fn prepare_trace_round0_matrices(
    l_skip: usize,
    log_large_domain: usize,
    constraints_dag: &SymbolicExpressionDag<F>,
    preprocessed: Option<&CommittedTraceDataV2<GpuBackendV2>>,
    air_ctx: &AirProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    trace_idx: usize,
) -> Result<TraceRound0Matrices, Round0PrepError> {
    let mut preprocessed_rot = false;
    let mut parts_main_rot = vec![false; air_ctx.cached_mains.len() + 1];
    for node in &constraints_dag.nodes {
        if let SymbolicExpressionNode::Variable(var) = node {
            match var.entry {
                Entry::Preprocessed { offset } => {
                    preprocessed_rot = preprocessed_rot || offset > 0;
                }
                Entry::Main { offset, part_index } => {
                    parts_main_rot[part_index] = parts_main_rot[part_index] || offset > 0;
                }
                _ => {}
            }
        }
    }
    let common_main_rot = parts_main_rot.pop().unwrap();
    let cached_mains_rot = parts_main_rot;

    let preprocessed_up = if let Some(committed) = preprocessed {
        debug_assert_eq!(committed.data.layout.l_skip(), l_skip);
        let trace = &committed.trace;
        let width = trace.width();
        let upsampled = upsample_matrix(
            l_skip,
            log_large_domain,
            trace,
            committed.data.mixed_view(0, width),
            preprocessed_rot,
        )?;
        Some((upsampled, preprocessed_rot))
    } else {
        None
    };
    let cached = zip(&air_ctx.cached_mains, cached_mains_rot)
        .map(|(committed, needs_rot)| -> Result<_, Round0PrepError> {
            debug_assert_eq!(committed.data.layout.l_skip(), l_skip);
            let trace = &committed.trace;
            let width = trace.width();
            let upsampled = upsample_matrix(
                l_skip,
                log_large_domain,
                trace,
                committed.data.mixed_view(0, width),
                needs_rot,
            )?;
            Ok((upsampled, needs_rot))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let common_main = &air_ctx.common_main;
    let width = common_main.width();
    let common = (
        upsample_matrix(
            l_skip,
            log_large_domain,
            common_main,
            common_main_pcs_data.mixed_view(trace_idx, width),
            common_main_rot,
        )?,
        common_main_rot,
    );

    Ok(TraceRound0Matrices {
        preprocessed: preprocessed_up,
        cached,
        common,
    })
}

/// `trace_evals` is always the honest trace matrix of evaluations. Its height is not lifted.
/// `mixed` is a view of the mixed-interpolation (aka coefficient in Z-, eval in X_i's) form of the
/// _strided_ trace (which does **not** equal lifted trace). Its height is the lifted height.
///
/// Returns matrix of either width `trace_evals.width` if `rotate` is false, or width
/// `trace_evals.width * 2` if `rotate` is true. In the latter case, the second half of the matrix
/// is the upsamping of the lift of the rotation of the trace evaluations (note the order!).
pub fn upsample_matrix(
    l_skip: usize,
    log_large_domain: usize,
    trace_evals: &DeviceMatrix<F>,
    mixed: DeviceMatrixView<F>,
    rotate: bool,
) -> Result<DeviceMatrix<F>, Round0PrepError> {
    debug_assert_eq!(trace_evals.width(), mixed.width());
    let height = trace_evals.height();
    let lifted_height = mixed.height();
    let width = trace_evals.width();
    let log_lifted_height = log2_strict_usize(lifted_height);
    debug_assert!(
        log_lifted_height >= l_skip,
        "log_height ({log_lifted_height}) < l_skip ({l_skip})"
    );
    debug_assert!(
        height == lifted_height || (height < lifted_height && log_lifted_height == l_skip)
    );
    let n_lift = log_lifted_height - l_skip;
    let num_x = 1 << n_lift;
    let domain_size = min(1 << l_skip, height);
    let domain_poly_count = num_x * width;
    let variants = if rotate { 2 } else { 1 };
    let total_poly_count = num_x * width * variants;

    let large_domain_size = 1 << log_large_domain;

    let upsampled = DeviceBuffer::<F>::with_capacity(large_domain_size * total_poly_count);
    // upsampled.fill_zero()?;

    unsafe {
        // 1. Handle the non-rotated part. If height == lifted_height, we can use the already
        //    computed mixed form.
        if height == lifted_height {
            batch_expand_pad_wide(
                upsampled.as_mut_ptr(),
                mixed.as_ptr(),
                domain_poly_count as u32,
                large_domain_size as u32,
                domain_size as u32,
            )?;
        } else {
            debug_assert_eq!(num_x, 1);
            debug_assert_eq!(lifted_height, 1 << l_skip);
            // In this case the mixed form is strided but not lifted, so we lift and perform iNTT to
            // get coefficient form. We will expand directly into the large_domain
            batch_expand_pad_wide(
                upsampled.as_mut_ptr(),
                trace_evals.buffer().as_ptr(),
                width as u32,
                large_domain_size as u32,
                domain_size as u32,
            )?;
            lift_padded_matrix_evals(
                upsampled.as_mut_ptr(),
                width as u32,
                height as u32,
                1 << l_skip,
                large_domain_size as u32,
            )?;
            // iNTT to get coefficient form
            batch_ntt(
                &upsampled,
                l_skip as u32,
                (log_large_domain - l_skip) as u32,
                width as u32,
                true,
                true,
            );
        }

        if rotate {
            // SAFETY: when `rotate = true`, we've allocated `upsampled` for twice the width.
            let dst = upsampled
                .as_mut_ptr()
                .add(large_domain_size * domain_poly_count);
            // Rotation always needs to use the evaluation form, and NOT the mixed form. Rotation
            // should also be done before lifting.
            batch_rotate_pad(
                dst,
                trace_evals.buffer().as_ptr(),
                width as u32,
                num_x as u32,
                domain_size as u32,
                large_domain_size as u32,
            )?;
            if height != lifted_height {
                lift_padded_matrix_evals(
                    dst,
                    width as u32,
                    height as u32,
                    1 << l_skip,
                    large_domain_size as u32,
                )?;
            }
            // NOTE: this is a workaround to pass the raw `dst` pointer to the `batch_ntt` function.
            // We use ManuallyDrop so dropping does not free the underlying memory.
            let upsampled_rot = ManuallyDrop::new(DeviceBuffer::from_raw_parts(
                dst,
                large_domain_size * domain_poly_count,
            ));
            // iNTT to get coefficient form
            batch_ntt(
                &upsampled_rot,
                l_skip as u32,
                (log_large_domain - l_skip) as u32,
                domain_poly_count as u32,
                true,
                true,
            );
        }
    }
    // At this point, we have total_poly_count univariate polynomials in coefficient form, each of
    // degree `< 2^l_skip` but padded with 0 coefficients to have `large_domain_size` coefficients.
    // We perform batch NTT to get evaluations on the larger domain.
    batch_ntt(
        &upsampled,
        log_large_domain as u32,
        0,
        total_poly_count as u32,
        true,
        false, // forward NTT
    );

    Ok(DeviceMatrix::new(
        Arc::new(upsampled),
        large_domain_size * num_x,
        width * variants,
    ))
}

fn collect_partition_ptrs(mats: &TraceRound0Matrices) -> Vec<MainMatrixPtrs<F>> {
    let mut ptrs = Vec::new();
    for (matrix, needs_rot) in &mats.cached {
        let multiplier = if *needs_rot { 2 } else { 1 };
        debug_assert_eq!(matrix.width() % multiplier, 0);
        ptrs.push(MainMatrixPtrs {
            data: matrix.buffer().as_ptr(),
            air_width: (matrix.width() / multiplier) as u32,
        });
    }
    let (common, needs_rot) = &mats.common;
    let multiplier = if *needs_rot { 2 } else { 1 };
    debug_assert_eq!(common.width() % multiplier, 0);
    ptrs.push(MainMatrixPtrs {
        data: common.buffer().as_ptr(),
        air_width: (common.width() / multiplier) as u32,
    });
    ptrs
}

impl TraceRound0Matrices {
    pub fn preprocessed_ptr(&self) -> MainMatrixPtrs<F> {
        if let Some((matrix, needs_rot)) = &self.preprocessed {
            let multiplier = if *needs_rot { 2 } else { 1 };
            debug_assert_eq!(matrix.width() % multiplier, 0);
            MainMatrixPtrs {
                data: matrix.buffer().as_ptr(),
                air_width: (matrix.width() / multiplier) as u32,
            }
        } else {
            MainMatrixPtrs {
                data: std::ptr::null(),
                air_width: 0,
            }
        }
    }
}
