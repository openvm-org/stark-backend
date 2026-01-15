use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, SymbolicDagBuilder, SymbolicExpressionDag,
    symbolic_expression::SymbolicExpression,
};
use p3_field::FieldAlgebra;
use stark_backend_v2::prover::{DeviceStarkProvingKeyV2, fractional_sumcheck_gkr::Frac};
use tracing::{debug, warn};

use super::errors::Round0EvalError;
use crate::{
    EF, F, GpuBackendV2,
    cuda::logup_zerocheck::{
        _logup_r0_intermediates_buffer_size, _logup_r0_temp_sums_buffer_size,
        _zerocheck_r0_intermediates_buffer_size, _zerocheck_r0_temp_sums_buffer_size,
        logup_bary_eval_interactions_round0, zerocheck_ntt_eval_constraints,
    },
    logup_zerocheck::rules::{SymbolicRulesGpuV2, codec::Codec},
};

/// Evaluate plain AIR constraints (not interactions) for a single AIR, given prepared trace input.
///
/// `num_cosets` should equal `constraint_degree - 1` because we evaluate the quotient polynomial.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_gpu(
    pk: &DeviceStarkProvingKeyV2<GpuBackendV2>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    lambda_pows: &DeviceBuffer<EF>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<EF>, Round0EvalError> {
    let constraints_dag = &pk.vk.symbolic_constraints;
    if constraints_dag.constraints.constraint_idx.is_empty() || num_cosets == 0 {
        // No plain AIR constraints, return empty buffer
        return Ok(DeviceBuffer::new());
    }

    let rules = &pk.other_data.zerocheck_round0;

    let buffer_size: u32 = rules.inner.buffer_size;
    let intermed_capacity = unsafe {
        _zerocheck_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<F>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<F>::new()
    };

    let temp_sums_buffer_capacity = unsafe {
        _zerocheck_r0_temp_sums_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<EF>();
    if used_temp_bytes > max_temp_bytes {
        // We do not error if the required bytes is greater than the requested max, but this may
        // lead to unexpected peak memory usage.
        warn!("zerocheck used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})");
    }

    let preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let mut sp_evals =
        DeviceBuffer::<EF>::with_capacity(num_cosets as usize * skip_domain as usize);
    // SAFETY:
    // - No bounds checks are done in this kernel. It fully assumes that the Rules are trusted and
    //   all nodes are valid.
    unsafe {
        zerocheck_ntt_eval_constraints(
            &mut temp_sums_buffer,
            &mut sp_evals,
            selectors_cube,
            preprocessed_ptr,
            main_parts,
            eq_cube,
            lambda_pows,
            public_values,
            &rules.inner.d_rules,
            &rules.inner.d_used_nodes,
            buffer_size,
            &mut intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
        )?;
    }

    Ok(sp_evals)
}

/// Evaluate interaction constraints (excluding plain AIR constraints) for a single AIR, given
/// prepared trace input.
///
/// `constraints` includes interaction expressions for the AIR.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_interactions_gpu(
    pk: &DeviceStarkProvingKeyV2<GpuBackendV2>,
    symbolic: &SymbolicConstraints<F>,
    selectors_cube: &DeviceBuffer<F>,
    main_parts: &DeviceBuffer<*const F>,
    public_values: &DeviceBuffer<F>,
    eq_cube: *const EF,
    beta_pows: &[EF],
    eq_3bs: &[EF],
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<DeviceBuffer<Frac<EF>>, Round0EvalError> {
    // Check if this trace has interactions
    if eq_3bs.is_empty() {
        return Ok(DeviceBuffer::new());
    }
    let large_domain = num_cosets * skip_domain;

    // We create a new "interactions DAG" where the new .constraints are the interaction [count,
    // message_0, message_1, ..] expressions themselves, while the .interactions are empty
    // We track the indices with InteractionNode

    // Copied from build_symbolic_constraints_dag to handle sorting of constraints
    // NOTE: For logup round0, the kernel uses weights indexed by rule_idx, not constraint_idx.
    // So we deduplicate constraint_idx and use dag_idx_to_rule_idx for weight mapping.
    let (rules, d_numer_weights, d_denom_weights, denom_sum_init) = {
        let mut dag_builder = SymbolicDagBuilder::new();
        let mut sorted_used_dag_idxs = Vec::new();
        for interaction in &symbolic.interactions {
            let count = dag_builder.add_expr(&interaction.count);
            sorted_used_dag_idxs.push(count);
            sorted_used_dag_idxs.extend(
                interaction
                    .message
                    .iter()
                    .map(|field_expr| dag_builder.add_expr(field_expr)),
            );
        }
        sorted_used_dag_idxs.sort();
        // Deduplicate for the dag since logup round0 kernel doesn't use used_nodes
        sorted_used_dag_idxs.dedup();
        let dag = SymbolicExpressionDag {
            nodes: dag_builder.nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let rules = SymbolicRulesGpuV2::new(&dag, true);
        let mut numer_weights = vec![EF::ZERO; rules.rules.len()];
        let mut denom_weights = vec![EF::ZERO; rules.rules.len()];
        let mut denom_sum_init = EF::ZERO;
        for (interaction_idx, interaction) in symbolic.interactions.iter().enumerate() {
            // CAUTION: an expression node could be used in multiple interactions, and might even be
            // used as `count` in one, but message field in another. We only care about their
            // weighted sum with eq_3b, so we compute the weights ahead of time.
            let count_dag_idx =
                dag_builder.expr_to_idx[&(&interaction.count as *const SymbolicExpression<_>)];
            let count_rule_idx = rules.dag_idx_to_rule_idx[&count_dag_idx];
            numer_weights[count_rule_idx] += eq_3bs[interaction_idx];
            denom_sum_init += eq_3bs[interaction_idx]
                * beta_pows[interaction.message.len()]
                * F::from_canonical_u32(interaction.bus_index as u32 + 1);

            for (message_idx, message) in interaction.message.iter().enumerate() {
                let message_dag_idx =
                    dag_builder.expr_to_idx[&(message as *const SymbolicExpression<_>)];
                let message_rule_idx = rules.dag_idx_to_rule_idx[&message_dag_idx];
                denom_weights[message_rule_idx] += eq_3bs[interaction_idx] * beta_pows[message_idx];
            }
        }
        let d_numer_weights = numer_weights.to_device()?;
        let d_denom_weights = denom_weights.to_device()?;
        (rules, d_numer_weights, d_denom_weights, denom_sum_init)
    };

    let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
    let d_rules = encoded_rules.to_device()?;

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity = unsafe {
        _logup_r0_intermediates_buffer_size(
            buffer_size,
            skip_domain,
            num_x,
            num_cosets,
            max_temp_bytes,
        )
    };
    let mut intermediates = if intermed_capacity > 0 {
        debug!("logup_r0:intermediates_capacity={intermed_capacity}");
        DeviceBuffer::<F>::with_capacity(intermed_capacity)
    } else {
        DeviceBuffer::<F>::new()
    };

    let temp_sums_buffer_capacity = unsafe {
        _logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes)
    };
    debug!("logup_r0:tmp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut temp_sums_buffer = DeviceBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<Frac<EF>>();
    if used_temp_bytes > max_temp_bytes {
        warn!(
            "logup_round0 used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})"
        );
    }

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
            eq_cube,
            public_values,
            &d_numer_weights,
            &d_denom_weights,
            denom_sum_init,
            &d_rules,
            buffer_size,
            &mut intermediates,
            skip_domain,
            num_x,
            height,
            num_cosets,
            g_shift,
            max_temp_bytes,
        )?;
    }

    Ok(s_evals)
}
