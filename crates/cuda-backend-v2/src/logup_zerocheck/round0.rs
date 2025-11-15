use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::{
    base::DeviceMatrix,
    cuda::kernels::lde::{batch_expand_pad, raw_batch_expand_pad},
    ntt::batch_ntt,
    transpiler::{SymbolicRulesOnGpu, codec::Codec},
};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::{
        SymbolicConstraints, SymbolicConstraintsDag,
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
    },
    prover::MatrixDimensions,
};
use p3_field::{FieldAlgebra, TwoAdicField};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use stark_backend_v2::{
    poly_common::{eval_eq_sharp_uni, eval_eq_uni},
    prover::{
        CommittedTraceDataV2, DeviceMultiStarkProvingKeyV2, ProvingContextV2,
        sumcheck::sumcheck_round0_deg,
    },
};

use super::{
    dag_scheduling::compute_constraint_expr_indices,
    errors::{Round0EvalError, Round0PrepError},
    matrix_utils::unstack_matrix_round0,
    state::Round0Buffers,
};
use crate::{
    Digest, EF, F, GpuBackendV2,
    cuda::{
        logup_zerocheck::{
            MainMatrixPtrs, accumulate_constraints, zerocheck_eval_constraints,
            zerocheck_eval_interactions_round0,
        },
        matrix::{batch_rotate_lift_and_pad, batch_rotate_pad},
    },
    stacked_pcs::StackedPcsDataGpu,
};

const TASK_SIZE: u32 = 65_536;

#[derive(Default, Debug)]
struct TraceRound0Matrices {
    preprocessed: Option<DeviceMatrix<F>>,
    cached: Vec<DeviceMatrix<F>>,
    common: Option<DeviceMatrix<F>>,
}

#[derive(Debug)]
struct Round0TraceInput<'a> {
    selectors_large: DeviceMatrix<F>,
    trace_mats: TraceRound0Matrices,
    main_ptrs: DeviceBuffer<MainMatrixPtrs<F>>,
    eq_x: DeviceMatrix<EF>,
    lambda_indices: DeviceBuffer<u32>,
    rules: DeviceBuffer<u128>,
    used_nodes: DeviceBuffer<usize>,
    public_values: &'a DeviceBuffer<F>,
    buffer_size: u32,
}

fn prepare_round0_trace_inputs<'a>(
    l_skip: usize,
    s_deg: usize,
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    buffers: &'a Round0Buffers,
    public_values: &'a [DeviceBuffer<F>],
) -> Result<Vec<Option<Round0TraceInput<'a>>>, Round0PrepError> {
    let mut inputs = Vec::with_capacity(ctx.per_trace.len());

    for (trace_idx, ((air_idx, air_ctx), selectors_base)) in ctx
        .per_trace
        .iter()
        .zip(buffers.selectors_base.iter())
        .enumerate()
    {
        let per_air = &pk.per_air[*air_idx];
        let constraints = SymbolicConstraints::from(&per_air.vk.symbolic_constraints);
        // Skip only if both constraints and interactions are empty
        if constraints.constraints.is_empty() && constraints.interactions.is_empty() {
            inputs.push(None);
            continue;
        }

        let selectors_large = upsample_matrix(
            l_skip,
            s_deg,
            selectors_base,
            selectors_base.height(), // selectors_base is already lifted
            false,
        )?;
        let trace_mats = prepare_trace_round0_matrices(
            l_skip,
            s_deg,
            trace_idx,
            per_air
                .preprocessed_data
                .as_ref()
                .map(|committed| committed.data.as_ref()),
            &air_ctx.cached_mains,
            common_main_pcs_data,
        )?;
        debug_assert_eq!(
            selectors_large.height(),
            trace_mats.common.as_ref().unwrap().height()
        );

        let constraints_dag: SymbolicConstraintsDag<F> = constraints.into();
        let lambda_index_map: HashMap<usize, usize> = constraints_dag
            .constraints
            .constraint_idx
            .iter()
            .enumerate()
            .map(|(idx, dag_idx)| (*dag_idx, idx))
            .collect();
        let constraint_dag_indices = compute_constraint_expr_indices(&constraints_dag, false);
        let rules = SymbolicRulesOnGpu::new(constraints_dag, false);

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

        let partition_ptrs_host = collect_partition_ptrs(&trace_mats);
        let d_main_ptrs = partition_ptrs_host.to_device()?;

        let eq_x = buffers.eq_xi[trace_idx].clone();
        let public_values = &public_values[trace_idx];

        inputs.push(Some(Round0TraceInput {
            selectors_large,
            trace_mats,
            main_ptrs: d_main_ptrs,
            eq_x,
            lambda_indices: d_lambda_indices,
            rules: d_rules,
            used_nodes: d_used_nodes,
            public_values,
            buffer_size: rules.buffer_size as u32,
        }));
    }

    Ok(inputs)
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_gpu(
    l_skip: usize,
    s_deg: usize,
    n_per_trace: &[isize],
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    buffers: &Round0Buffers,
    public_values: &[DeviceBuffer<F>],
    xi: &[EF],
    lambda_pows: &DeviceBuffer<EF>,
) -> Result<Vec<DeviceBuffer<EF>>, Round0EvalError> {
    let num_traces = ctx.per_trace.len();
    assert_eq!(n_per_trace.len(), num_traces);

    let s0_deg = sumcheck_round0_deg(l_skip, s_deg);
    let log_large_domain = log2_ceil_usize(s0_deg + 1);
    let large_domain = 1 << log_large_domain;

    assert!(!xi.is_empty(), "xi vector must not be empty");
    let omega = F::two_adic_generator(log_large_domain);
    let eq_z_host: Vec<EF> = omega
        .powers()
        .take(large_domain)
        .map(|z| eval_eq_uni(l_skip, xi[0], z.into()))
        .collect();
    let d_eq_z = eq_z_host.to_device()?;

    let skip_stride = (large_domain >> l_skip) as u32;
    let trace_inputs = prepare_round0_trace_inputs(
        l_skip,
        s_deg,
        pk,
        ctx,
        common_main_pcs_data,
        buffers,
        public_values,
    )?;

    let mut sums = Vec::with_capacity(num_traces);

    for (trace_idx, input) in trace_inputs.iter().enumerate() {
        let Some(input) = input else {
            sums.push(DeviceBuffer::new());
            continue;
        };

        // Check if this trace has constraints - if not, skip constraint evaluation
        let per_air = &pk.per_air[ctx.per_trace[trace_idx].0];
        let constraints = SymbolicConstraints::from(&per_air.vk.symbolic_constraints);
        if constraints.constraints.is_empty() {
            // Trace has no constraints, create empty buffers
            sums.push(DeviceBuffer::new());
            continue;
        }

        let num_x = input.eq_x.height();
        let height = input.selectors_large.height();
        debug_assert_eq!(num_x * large_domain, height);
        debug_assert_eq!(input.trace_mats.common.as_ref().unwrap().height(), height);

        let intermediates = if input.buffer_size > 0 {
            let capacity = if input.buffer_size > 10 {
                (TASK_SIZE as usize) * input.buffer_size as usize
            } else {
                input.buffer_size as usize
            };
            Some(DeviceBuffer::<EF>::with_capacity(capacity))
        } else {
            None
        };

        let num_rows_per_tile = {
            let h = height as u32;
            h.div_ceil(TASK_SIZE).max(1)
        };

        let output = DeviceBuffer::<EF>::with_capacity(height);

        unsafe {
            zerocheck_eval_constraints(
                &output,
                &input.selectors_large,
                &input.main_ptrs,
                input.trace_mats.preprocessed.as_ref(),
                &d_eq_z,
                &input.eq_x,
                lambda_pows,
                &input.lambda_indices,
                input.public_values,
                &input.rules,
                &input.used_nodes,
                input.buffer_size,
                intermediates.as_ref(),
                large_domain as u32,
                num_x as u32,
                num_rows_per_tile,
                skip_stride,
            )?;
        }

        let sums_device = DeviceBuffer::<EF>::with_capacity(large_domain);
        unsafe {
            accumulate_constraints(&output, &sums_device, large_domain as u32, num_x as u32)?;
        }

        sums.push(sums_device);
    }

    Ok(sums)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn evaluate_round0_interactions_gpu(
    l_skip: usize,
    s_deg: usize,
    n_per_trace: &[isize],
    pk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
    ctx: &ProvingContextV2<GpuBackendV2>,
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
    buffers: &Round0Buffers,
    public_values: &[DeviceBuffer<F>],
    xi: &[EF],
    omega_skip_pows: &[F],
    beta_pows: &[EF],
    eq_3b_per_trace: &[DeviceBuffer<EF>],
) -> Result<(Vec<DeviceBuffer<EF>>, Vec<DeviceBuffer<EF>>), Round0EvalError> {
    let num_traces = ctx.per_trace.len();
    assert_eq!(n_per_trace.len(), num_traces);

    let s0_deg = stark_backend_v2::prover::sumcheck::sumcheck_round0_deg(l_skip, s_deg);
    let log_large_domain = log2_ceil_usize(s0_deg + 1);
    let large_domain = 1 << log_large_domain;

    assert!(!xi.is_empty(), "xi vector must not be empty");
    let omega = F::two_adic_generator(log_large_domain);

    // Precompute eq_sharp_z (using eq_sharp instead of eq_z)
    let eq_sharp_z_host: Vec<EF> = omega
        .powers()
        .take(large_domain)
        .map(|z| eval_eq_sharp_uni(omega_skip_pows, &xi[..l_skip], z.into()))
        .collect();
    let d_eq_sharp_z = eq_sharp_z_host.to_device()?;

    let skip_stride = (large_domain >> l_skip) as u32;
    let trace_inputs = prepare_round0_trace_inputs(
        l_skip,
        s_deg,
        pk,
        ctx,
        common_main_pcs_data,
        buffers,
        public_values,
    )?;

    let mut sums_numer = Vec::with_capacity(num_traces);
    let mut sums_denom = Vec::with_capacity(num_traces);

    for (trace_idx, input) in trace_inputs.iter().enumerate() {
        let Some(input) = input else {
            sums_numer.push(DeviceBuffer::new());
            sums_denom.push(DeviceBuffer::new());
            continue;
        };

        // Check if this trace has interactions
        // eq_3b_per_trace is non-empty only for traces with interactions
        let eq_3b = &eq_3b_per_trace[trace_idx];
        if eq_3b.is_empty() {
            sums_numer.push(DeviceBuffer::new());
            sums_denom.push(DeviceBuffer::new());
            continue;
        }

        // Get constraints for this trace to access interactions
        let per_air = &pk.per_air[ctx.per_trace[trace_idx].0];
        let constraints = SymbolicConstraints::from(&per_air.vk.symbolic_constraints);

        // Create symbolic challenges for beta
        let max_fields_len = constraints
            .interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);

        // Prepare challenges: [unused_alpha, beta_0, beta_1, ..., beta_{max_fields_len}]
        // Challenge index 0 is unused (would be alpha), indices 1..=max_fields_len+1 are betas
        // symbolic_challenges uses 0..=max_fields_len+1 to cover all beta indices
        let mut challenges_vec = vec![EF::ZERO]; // index 0 unused
        if max_fields_len < beta_pows.len() {
            challenges_vec.extend_from_slice(&beta_pows[..=max_fields_len]);
        } else {
            challenges_vec.extend_from_slice(beta_pows);
            challenges_vec.extend(vec![EF::ZERO; max_fields_len + 1 - beta_pows.len()]);
        }
        let d_challenges = challenges_vec.to_device()?;

        // Create symbolic challenges: indices 0..=max_fields_len+1 (0 unused, 1..=max_fields_len+1
        // for betas)
        let symbolic_challenges: Vec<SymbolicExpression<F>> = (0..=max_fields_len + 1)
            .map(|index| SymbolicVariable::<F>::new(Entry::Challenge, index).into())
            .collect();

        let mut transformed_interactions = Vec::new();
        for interaction in &constraints.interactions {
            let mut interaction = interaction.clone();
            let b = SymbolicExpression::from_canonical_u32(interaction.bus_index as u32 + 1);
            let betas = symbolic_challenges[1..].to_vec();
            let mut denom = SymbolicExpression::from_canonical_u32(0);
            for (j, expr) in interaction.message.iter().enumerate() {
                denom += betas[j].clone() * expr.clone();
            }
            denom += betas[interaction.message.len()].clone() * b;
            interaction.message = vec![denom];
            transformed_interactions.push(interaction);
        }

        // Build interaction DAG with transformed interactions
        let constraints_dag: SymbolicConstraintsDag<F> = SymbolicConstraints {
            constraints: vec![],
            interactions: transformed_interactions,
        }
        .into();
        let rules = SymbolicRulesOnGpu::new(constraints_dag, true);

        let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = rules.used_nodes.to_device()?;

        let num_x = input.eq_x.height();
        let height = input.selectors_large.height();

        let intermediates = if rules.buffer_size > 0 {
            let capacity = if rules.buffer_size > 10 {
                (TASK_SIZE as usize) * rules.buffer_size
            } else {
                rules.buffer_size
            };
            Some(DeviceBuffer::<EF>::with_capacity(capacity))
        } else {
            None
        };

        let num_rows_per_tile = {
            let h = height as u32;
            h.div_ceil(TASK_SIZE).max(1)
        };

        let output_numer = DeviceBuffer::<EF>::with_capacity(height);
        let output_denom = DeviceBuffer::<EF>::with_capacity(height);

        unsafe {
            zerocheck_eval_interactions_round0(
                &output_numer,
                &output_denom,
                &input.selectors_large,
                &input.main_ptrs,
                input.trace_mats.preprocessed.as_ref(),
                &d_eq_sharp_z,
                &input.eq_x,
                eq_3b,
                input.public_values,
                &d_rules,
                &d_used_nodes,
                rules.buffer_size.try_into().unwrap(),
                intermediates.as_ref(),
                large_domain as u32,
                num_x as u32,
                num_rows_per_tile,
                skip_stride,
                &d_challenges,
            )?;
        }

        let sums_numer_device = DeviceBuffer::<EF>::with_capacity(large_domain);
        let sums_denom_device = DeviceBuffer::<EF>::with_capacity(large_domain);
        unsafe {
            accumulate_constraints(
                &output_numer,
                &sums_numer_device,
                large_domain as u32,
                num_x as u32,
            )?;
            accumulate_constraints(
                &output_denom,
                &sums_denom_device,
                large_domain as u32,
                num_x as u32,
            )?;
        }

        sums_numer.push(sums_numer_device);
        sums_denom.push(sums_denom_device);
    }

    Ok((sums_numer, sums_denom))
}

// For a single present AIR.
fn prepare_trace_round0_matrices(
    l_skip: usize,
    s_deg: usize,
    trace_idx: usize,
    preprocessed: Option<&StackedPcsDataGpu<F, Digest>>,
    cached_mains: &[CommittedTraceDataV2<GpuBackendV2>],
    common_main_pcs_data: &StackedPcsDataGpu<F, Digest>,
) -> Result<TraceRound0Matrices, Round0PrepError> {
    let mut mats = TraceRound0Matrices::default();

    if let Some(data) = preprocessed {
        debug_assert_eq!(data.layout.l_skip(), l_skip);
        let (pre_unstacked, height) = unstack_matrix_round0(data, 0)?;
        mats.preprocessed = Some(upsample_matrix(
            l_skip,
            s_deg,
            &pre_unstacked,
            height,
            true,
        )?);
    }

    mats.cached = cached_mains
        .iter()
        .map(|committed| {
            debug_assert_eq!(committed.data.layout.l_skip(), l_skip);
            let (unstacked, height) = unstack_matrix_round0(committed.data.as_ref(), 0)?;
            upsample_matrix(l_skip, s_deg, &unstacked, height, true)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (common_main, height) = unstack_matrix_round0(common_main_pcs_data, trace_idx)?;
    mats.common = Some(upsample_matrix(l_skip, s_deg, &common_main, height, true)?);

    Ok(mats)
}

/// Currently this function is only called after `unstack_matrix_round0`. As a consequence we assume
/// that `matrix.height()` is the lifted height and if this matrix is the lifted matrix when
/// `lifted_height != height`.
fn upsample_matrix(
    l_skip: usize,
    s_deg: usize,
    matrix: &DeviceMatrix<F>,
    height: usize,
    rotate: bool,
) -> Result<DeviceMatrix<F>, Round0PrepError> {
    let lifted_height = matrix.height();
    let width = matrix.width();
    let log_lifted_height = log2_strict_usize(lifted_height);
    assert!(
        log_lifted_height >= l_skip,
        "log_height ({log_lifted_height}) < l_skip ({l_skip})"
    );
    let n_lift = log_lifted_height - l_skip;
    let num_x = 1 << n_lift;
    let domain_size = 1 << l_skip;
    let domain_poly_count = num_x * width;
    let variants = if rotate { 2 } else { 1 };
    let total_poly_count = domain_poly_count * variants;

    let s0_deg = stark_backend_v2::prover::sumcheck::sumcheck_round0_deg(l_skip, s_deg);
    let log_large_domain = log2_ceil_usize(s0_deg + 1);
    let large_domain_size = 1 << log_large_domain;

    let upsampled = DeviceBuffer::<F>::with_capacity(large_domain_size * total_poly_count);
    upsampled.fill_zero()?;

    unsafe {
        if rotate {
            if height == lifted_height {
                // IMPORTANT: in the case n < 0, you must rotate _and then_ lift (which is different
                // from lift and then rotate)
                batch_rotate_pad(
                    &upsampled,
                    matrix.buffer(),
                    width as u32,
                    num_x as u32,
                    domain_size as u32,
                    large_domain_size as u32,
                )?;
            } else {
                debug_assert_eq!(lifted_height, 1 << l_skip);
                debug_assert_eq!(lifted_height % height, 0);
                debug_assert_eq!(num_x, 1);
                raw_batch_expand_pad(
                    upsampled.as_mut_ptr(),
                    matrix.buffer().as_ptr(),
                    width as u32,
                    large_domain_size as u32,
                    domain_size as u32,
                )?;
                batch_rotate_lift_and_pad(
                    upsampled
                        .as_mut_ptr()
                        .add(large_domain_size * domain_poly_count),
                    matrix.buffer().as_ptr(),
                    width as u32,
                    height as u32,
                    domain_size as u32,
                    large_domain_size as u32,
                )?;
            }
        } else {
            batch_expand_pad(
                &upsampled,
                matrix.buffer(),
                domain_poly_count as u32,
                large_domain_size as u32,
                domain_size as u32,
            )?;
        }
    }

    batch_ntt(
        &upsampled,
        l_skip as u32,
        (log_large_domain - l_skip) as u32,
        total_poly_count as u32,
        true,
        true,
    );

    batch_ntt(
        &upsampled,
        log_large_domain as u32,
        0,
        total_poly_count as u32,
        true,
        false,
    );

    Ok(DeviceMatrix::new(
        Arc::new(upsampled),
        large_domain_size * num_x,
        width * variants,
    ))
}

fn collect_partition_ptrs(mats: &TraceRound0Matrices) -> Vec<MainMatrixPtrs<F>> {
    let mut ptrs = Vec::new();
    for matrix in &mats.cached {
        debug_assert_eq!(
            matrix.width() % 2,
            0,
            "rotated cached main should have even width"
        );
        ptrs.push(MainMatrixPtrs {
            data: matrix.buffer().as_ptr(),
            air_width: (matrix.width() / 2) as u32,
        });
    }
    if let Some(common) = &mats.common {
        debug_assert_eq!(
            common.width() % 2,
            0,
            "rotated common main should have even width"
        );
        ptrs.push(MainMatrixPtrs {
            data: common.buffer().as_ptr(),
            air_width: (common.width() / 2) as u32,
        });
    }
    ptrs
}
