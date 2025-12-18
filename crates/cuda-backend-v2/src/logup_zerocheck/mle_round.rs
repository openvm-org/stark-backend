use std::collections::HashMap;

use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, SymbolicConstraintsDag,
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::FieldAlgebra;
use stark_backend_v2::prover::fractional_sumcheck_gkr::Frac;
use tracing::debug;

use crate::{
    EF, F,
    cuda::logup_zerocheck::{
        _logup_mle_intermediates_buffer_size, _logup_mle_temp_sums_buffer_size,
        _zerocheck_mle_intermediates_buffer_size, _zerocheck_mle_temp_sums_buffer_size,
        MainMatrixPtrs, logup_eval_mle, zerocheck_eval_mle,
    },
    logup_zerocheck::rules::{SymbolicRulesOnGpuV2, codec::Codec},
};

#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_constraints_gpu(
    eq_xi_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    main_ptrs: &[MainMatrixPtrs<EF>],
    public_vals: &DeviceBuffer<F>,
    lambda_pows: &DeviceBuffer<EF>,
    symbolic_constraints: &SymbolicConstraintsDag<F>,
    num_y: usize,
    s_deg: usize,
) -> DeviceBuffer<EF> {
    let d_main_ptrs = main_ptrs
        .to_device()
        .expect("failed to copy main_ptrs to device");
    // Prepare lambda indices and rules (same logic as round0)
    let lambda_index_map: HashMap<usize, usize> = symbolic_constraints
        .constraints
        .constraint_idx
        .iter()
        .enumerate()
        .map(|(idx, dag_idx)| (*dag_idx, idx))
        .collect();
    let rules = SymbolicRulesOnGpuV2::new(symbolic_constraints, false, false);

    let lambda_indices_host: Vec<u32> = rules
        .used_nodes
        .iter()
        .map(|&constraint_idx| {
            rules
                .constraint_expr_idxs
                .get(constraint_idx)
                .and_then(|dag_idx| lambda_index_map.get(dag_idx))
                .copied()
                .unwrap_or(0) as u32
        })
        .collect();
    let d_lambda_indices = lambda_indices_host
        .to_device()
        .expect("failed to copy lambda indices to device");

    let encoded_rules: Vec<u128> = rules.constraints.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules
        .to_device()
        .expect("failed to copy rules to device");
    let d_used_nodes = rules
        .used_nodes
        .to_device()
        .expect("failed to copy used_nodes to device");

    // Calculate dimensions
    let num_x = s_deg as u32;
    let num_y = num_y as u32;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity =
        unsafe { _zerocheck_mle_intermediates_buffer_size(buffer_size, num_x, num_y) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<EF>::new()
    };
    let temp_sums_buffer_capacity = unsafe { _zerocheck_mle_temp_sums_buffer_size(num_x, num_y) };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let mut output = DeviceBuffer::<EF>::with_capacity(s_deg);
    // Launch evaluation kernel
    unsafe {
        zerocheck_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            &d_main_ptrs,
            lambda_pows,
            &d_lambda_indices,
            public_vals,
            &d_rules,
            &d_used_nodes,
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
        )
        .expect("failed to evaluate MLE constraints on GPU");
    }

    output
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_mle_interactions_gpu(
    eq_sharp_ptr: *const EF,
    sels_ptr: *const EF,
    prep_ptr: MainMatrixPtrs<EF>,
    main_ptrs: &[MainMatrixPtrs<EF>],
    public_vals: &DeviceBuffer<F>,
    beta_pows: &[EF],
    eq_3bs: &DeviceBuffer<EF>,
    symbolic_constraints: &SymbolicConstraintsDag<F>,
    num_y: usize,
    s_deg: usize,
) -> DeviceBuffer<Frac<EF>> {
    let d_main_ptrs = main_ptrs
        .to_device()
        .expect("failed to copy main_ptrs to device");

    // Prepare interaction evaluation data structures (same pattern as round0)
    let constraints = SymbolicConstraints::from(symbolic_constraints);

    // Create symbolic challenges for beta
    let max_fields_len = constraints
        .interactions
        .iter()
        .map(|interaction| interaction.message.len())
        .max()
        .unwrap_or(0);

    // Prepare challenges: [unused_alpha, beta_0, beta_1, ..., beta_{max_fields_len}]
    // Challenge index 0 is unused (would be alpha), indices 1..=max_fields_len+1 are betas
    let mut challenges_vec = vec![EF::ZERO]; // index 0 unused
    if max_fields_len < beta_pows.len() {
        challenges_vec.extend_from_slice(&beta_pows[..=max_fields_len]);
    } else {
        challenges_vec.extend_from_slice(beta_pows);
        challenges_vec.extend(vec![EF::ZERO; max_fields_len + 1 - beta_pows.len()]);
    }
    let d_challenges = challenges_vec
        .to_device()
        .expect("failed to copy challenges to device");

    // Create symbolic challenges: indices 0..=max_fields_len+1 (0 unused, 1..=max_fields_len+1 for
    // betas)
    let symbolic_challenges: Vec<SymbolicExpression<F>> = (0..=max_fields_len + 1)
        .map(|index| SymbolicVariable::<F>::new(Entry::Challenge, index).into())
        .collect();

    // Transform interactions (same pattern as round0)
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
    let rules = SymbolicRulesOnGpuV2::new(&constraints_dag, true, false);

    let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules
        .to_device()
        .expect("failed to copy rules to device");
    let d_used_nodes = rules
        .used_nodes
        .to_device()
        .expect("failed to copy used_nodes to device");

    // Calculate dimensions
    let num_y = num_y as u32;
    let num_x = s_deg as u32;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity =
        unsafe { _logup_mle_intermediates_buffer_size(buffer_size, num_x, num_y) };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<EF>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<EF>::new()
    };
    let temp_sums_buffer_capacity = unsafe { _logup_mle_temp_sums_buffer_size(num_x, num_y) };
    debug!("logup:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let mut output = DeviceBuffer::<Frac<EF>>::with_capacity(s_deg);

    // Launch interaction evaluation kernel
    unsafe {
        logup_eval_mle(
            &mut temp_sums_buffer,
            &mut output,
            eq_sharp_ptr,
            sels_ptr,
            prep_ptr,
            &d_main_ptrs,
            &d_challenges,
            eq_3bs,
            public_vals,
            &d_rules,
            &d_used_nodes,
            buffer_size,
            &mut intermediates,
            num_y,
            num_x,
        )
        .expect("failed to evaluate MLE interactions on GPU");
    }
    output
}
