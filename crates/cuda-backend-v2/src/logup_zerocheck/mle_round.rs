use std::collections::HashMap;

use itertools::Itertools;
use openvm_cuda_backend::transpiler::{SymbolicRulesOnGpu, codec::Codec};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, SymbolicConstraintsDag,
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::FieldAlgebra;

use super::dag_scheduling::compute_constraint_expr_indices;
use crate::{
    EF, F,
    cuda::logup_zerocheck::{
        MainMatrixPtrs, reduce_hypercube_blocks, reduce_hypercube_final, zerocheck_eval_mle,
        zerocheck_eval_mle_interactions,
    },
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
    domain_size: usize,
    s_deg: usize,
) -> Vec<EF> {
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
    let constraint_dag_indices = compute_constraint_expr_indices(symbolic_constraints, false);
    let rules = SymbolicRulesOnGpu::new(symbolic_constraints.clone(), false);

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
    let num_y = domain_size / s_deg;
    let num_x = s_deg;

    // Allocate output buffer: [num_x * num_y] for f_hat(x, y) = eq_xi_val * constraint_sum
    let evaluated = DeviceBuffer::<EF>::with_capacity(num_x * num_y);

    // Allocate intermediates buffer (same pattern as round0)
    const TASK_SIZE: u32 = 65_536;
    let intermediates = if rules.buffer_size > 0 {
        let capacity = if rules.buffer_size > 10 {
            TASK_SIZE as usize * rules.buffer_size
        } else {
            rules.buffer_size
        };
        Some(DeviceBuffer::<EF>::with_capacity(capacity))
    } else {
        None
    };

    // Calculate num_rows_per_tile (same pattern as round0)
    let num_rows_per_tile = {
        let h = domain_size as u32;
        h.div_ceil(TASK_SIZE).max(1)
    };

    // Launch evaluation kernel
    unsafe {
        zerocheck_eval_mle(
            &evaluated,
            eq_xi_ptr,
            sels_ptr,
            prep_ptr,
            &d_main_ptrs,
            lambda_pows,
            &d_lambda_indices,
            public_vals,
            &d_rules,
            &d_used_nodes,
            rules.buffer_size.try_into().unwrap(),
            intermediates.as_ref(),
            num_y as u32,
            num_x as u32,
            num_rows_per_tile,
        )
        .expect("failed to evaluate MLE constraints on GPU");
    }

    if domain_size == 1 {
        return evaluated
            .to_host()
            .expect("failed to copy evaluation result to host");
    }

    // REDUCTION: Sum over hypercube for each x_idx
    let num_blocks = num_y.div_ceil(256); // BLOCK_SIZE = 256
    let mut block_sums = DeviceBuffer::<EF>::with_capacity(num_blocks * s_deg);
    let mut output = DeviceBuffer::<EF>::with_capacity(s_deg);

    unsafe {
        // Phase 1: Block-level reduction
        reduce_hypercube_blocks(&mut block_sums, &evaluated, s_deg as u32, num_y as u32)
            .expect("failed to reduce hypercube blocks on GPU");

        // Phase 2: Final reduction
        reduce_hypercube_final(&mut output, &block_sums, s_deg as u32, num_blocks as u32)
            .expect("failed to finalize hypercube reduction on GPU");
    }

    // Copy result to host
    output
        .to_host()
        .expect("failed to copy reduction result to host")
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
    domain_size: usize,
    s_deg: usize,
) -> [Vec<EF>; 2] {
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
    let rules = SymbolicRulesOnGpu::new(constraints_dag, true);

    let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules
        .to_device()
        .expect("failed to copy rules to device");
    let d_used_nodes = rules
        .used_nodes
        .to_device()
        .expect("failed to copy used_nodes to device");

    // Calculate dimensions
    let num_y = domain_size / s_deg;
    let num_x = s_deg;

    // Allocate output buffers: [num_x * num_y] for numer and denom separately
    let evaluated_numer = DeviceBuffer::<EF>::with_capacity(num_x * num_y);
    let evaluated_denom = DeviceBuffer::<EF>::with_capacity(num_x * num_y);

    // Allocate intermediates buffer (same pattern as round0)
    const TASK_SIZE: u32 = 65_536;
    let intermediates = if rules.buffer_size > 0 {
        let capacity = if rules.buffer_size > 10 {
            TASK_SIZE as usize * rules.buffer_size
        } else {
            rules.buffer_size
        };
        Some(DeviceBuffer::<EF>::with_capacity(capacity))
    } else {
        None
    };

    // Calculate num_rows_per_tile (same pattern as round0)
    let num_rows_per_tile = {
        let h = domain_size as u32;
        h.div_ceil(TASK_SIZE).max(1)
    };

    // Launch interaction evaluation kernel
    unsafe {
        zerocheck_eval_mle_interactions(
            &evaluated_numer,
            &evaluated_denom,
            eq_sharp_ptr,
            sels_ptr,
            prep_ptr,
            &d_main_ptrs,
            &d_challenges,
            eq_3bs,
            public_vals,
            &d_rules,
            &d_used_nodes,
            rules.buffer_size.try_into().unwrap(),
            intermediates.as_ref(),
            num_y as u32,
            num_x as u32,
            num_rows_per_tile,
        )
        .expect("failed to evaluate MLE interactions on GPU");
    }

    if domain_size == 1 {
        return [
            evaluated_numer
                .to_host()
                .expect("failed to copy numer result to host"),
            evaluated_denom
                .to_host()
                .expect("failed to copy denom result to host"),
        ];
    }

    // REDUCTION: Sum over hypercube for each x_idx (for both numer and denom)
    let num_blocks = num_y.div_ceil(256); // BLOCK_SIZE = 256
    let mut block_sums_numer = DeviceBuffer::<EF>::with_capacity(num_blocks * s_deg);
    let mut block_sums_denom = DeviceBuffer::<EF>::with_capacity(num_blocks * s_deg);
    let mut output_numer = DeviceBuffer::<EF>::with_capacity(s_deg);
    let mut output_denom = DeviceBuffer::<EF>::with_capacity(s_deg);

    unsafe {
        // Phase 1: Block-level reduction for numer
        reduce_hypercube_blocks(
            &mut block_sums_numer,
            &evaluated_numer,
            s_deg as u32,
            num_y as u32,
        )
        .expect("failed to reduce hypercube blocks (numer) on GPU");

        // Phase 1: Block-level reduction for denom
        reduce_hypercube_blocks(
            &mut block_sums_denom,
            &evaluated_denom,
            s_deg as u32,
            num_y as u32,
        )
        .expect("failed to reduce hypercube blocks (denom) on GPU");

        // Phase 2: Final reduction for numer
        reduce_hypercube_final(
            &mut output_numer,
            &block_sums_numer,
            s_deg as u32,
            num_blocks as u32,
        )
        .expect("failed to finalize hypercube reduction (numer) on GPU");

        // Phase 2: Final reduction for denom
        reduce_hypercube_final(
            &mut output_denom,
            &block_sums_denom,
            s_deg as u32,
            num_blocks as u32,
        )
        .expect("failed to finalize hypercube reduction (denom) on GPU");
    }

    // Copy results to host
    [
        output_numer
            .to_host()
            .expect("failed to copy numer reduction result to host"),
        output_denom
            .to_host()
            .expect("failed to copy denom reduction result to host"),
    ]
}
