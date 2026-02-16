use itertools::Itertools;
use openvm_metal_common::{copy::MemCopyH2D, d_buffer::MetalBuffer};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, SymbolicConstraints, SymbolicDagBuilder,
        SymbolicExpressionDag,
    },
    prover::{fractional_sumcheck_gkr::Frac, DeviceStarkProvingKey},
};
use p3_field::PrimeCharacteristicRing;
use tracing::{debug, warn};

use super::errors::Round0EvalError;
use crate::{
    metal::logup_zerocheck::{
        logup_r0_intermediates_buffer_size, logup_r0_temp_sums_buffer_size,
        zerocheck_r0_intermediates_buffer_size, zerocheck_r0_temp_sums_buffer_size,
    },
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesMetal},
    prelude::{EF, F},
    MetalBackend,
};

/// Evaluate plain AIR constraints (not interactions) for a single AIR, given prepared trace input.
///
/// `num_cosets` should equal `constraint_degree - 1` because we evaluate the quotient polynomial.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_constraints_metal(
    pk: &DeviceStarkProvingKey<MetalBackend>,
    selectors_cube: &MetalBuffer<F>,
    main_parts: &MetalBuffer<*const F>,
    public_values: &MetalBuffer<F>,
    eq_cube: *const EF,
    lambda_pows: &MetalBuffer<EF>,
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<MetalBuffer<EF>, Round0EvalError> {
    let constraints_dag = &pk.vk.symbolic_constraints;
    if constraints_dag.constraints.constraint_idx.is_empty() || num_cosets == 0 {
        return Ok(MetalBuffer::with_capacity(0));
    }

    let rules = &pk.other_data.zerocheck_round0;

    let buffer_size: u32 = rules.inner.buffer_size;
    let intermed_capacity = zerocheck_r0_intermediates_buffer_size(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
    );
    let mut _intermediates = if intermed_capacity > 0 {
        debug!("zerocheck:intermediates_capacity={intermed_capacity}");
        MetalBuffer::<F>::with_capacity(intermed_capacity)
    } else {
        MetalBuffer::<F>::with_capacity(0)
    };

    let temp_sums_buffer_capacity = zerocheck_r0_temp_sums_buffer_size(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
    );
    debug!("zerocheck:temp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut _temp_sums_buffer = MetalBuffer::<EF>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<EF>();
    if used_temp_bytes > max_temp_bytes {
        warn!("zerocheck used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})");
    }

    let _preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let sp_evals = MetalBuffer::<EF>::with_capacity(num_cosets as usize * skip_domain as usize);

    // TODO: dispatch zerocheck_ntt_eval_constraints Metal kernel when available.
    // The kernel dispatch infrastructure is in metal/logup_zerocheck.rs but the
    // higher-level eval kernel is not yet ported. This will be wired up when the
    // Metal compute shaders for zerocheck NTT evaluation are compiled.
    let _ = (
        selectors_cube,
        main_parts,
        public_values,
        eq_cube,
        lambda_pows,
        height,
        g_shift,
    );

    Ok(sp_evals)
}

/// Evaluate interaction constraints (excluding plain AIR constraints) for a single AIR, given
/// prepared trace input.
///
/// `constraints` includes interaction expressions for the AIR.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_round0_interactions_metal(
    pk: &DeviceStarkProvingKey<MetalBackend>,
    symbolic: &SymbolicConstraints<F>,
    selectors_cube: &MetalBuffer<F>,
    main_parts: &MetalBuffer<*const F>,
    public_values: &MetalBuffer<F>,
    eq_cube: *const EF,
    beta_pows: &[EF],
    eq_3bs: &[EF],
    skip_domain: u32,
    num_x: u32,
    height: u32,
    num_cosets: u32,
    g_shift: F,
    max_temp_bytes: usize,
) -> Result<MetalBuffer<Frac<EF>>, Round0EvalError> {
    if eq_3bs.is_empty() {
        return Ok(MetalBuffer::with_capacity(0));
    }
    let large_domain = num_cosets * skip_domain;

    let (rules, _d_numer_weights, _d_denom_weights, _denom_sum_init) = {
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
        sorted_used_dag_idxs.dedup();
        let dag = SymbolicExpressionDag {
            nodes: dag_builder.nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let rules = SymbolicRulesMetal::new(&dag, true);
        let mut numer_weights = vec![EF::ZERO; rules.rules.len()];
        let mut denom_weights = vec![EF::ZERO; rules.rules.len()];
        let mut denom_sum_init = EF::ZERO;
        for (interaction_idx, interaction) in symbolic.interactions.iter().enumerate() {
            let count_dag_idx =
                dag_builder.expr_to_idx[&(&interaction.count as *const SymbolicExpression<_>)];
            let count_rule_idx = rules.dag_idx_to_rule_idx[&count_dag_idx];
            numer_weights[count_rule_idx] += eq_3bs[interaction_idx];
            denom_sum_init += eq_3bs[interaction_idx]
                * beta_pows[interaction.message.len()]
                * F::from_u32(interaction.bus_index as u32 + 1);

            for (message_idx, message) in interaction.message.iter().enumerate() {
                let message_dag_idx =
                    dag_builder.expr_to_idx[&(message as *const SymbolicExpression<_>)];
                let message_rule_idx = rules.dag_idx_to_rule_idx[&message_dag_idx];
                denom_weights[message_rule_idx] += eq_3bs[interaction_idx] * beta_pows[message_idx];
            }
        }
        let d_numer_weights = numer_weights.to_device();
        let d_denom_weights = denom_weights.to_device();
        (rules, d_numer_weights, d_denom_weights, denom_sum_init)
    };

    let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
    let _d_rules = encoded_rules.to_device();

    let buffer_size: u32 = rules.buffer_size.try_into().unwrap();
    let intermed_capacity = logup_r0_intermediates_buffer_size(
        buffer_size,
        skip_domain,
        num_x,
        num_cosets,
        max_temp_bytes,
    );
    let mut _intermediates = if intermed_capacity > 0 {
        debug!("logup_r0:intermediates_capacity={intermed_capacity}");
        MetalBuffer::<F>::with_capacity(intermed_capacity)
    } else {
        MetalBuffer::<F>::with_capacity(0)
    };

    let temp_sums_buffer_capacity =
        logup_r0_temp_sums_buffer_size(buffer_size, skip_domain, num_x, num_cosets, max_temp_bytes);
    debug!("logup_r0:tmp_sums_buffer_capacity={temp_sums_buffer_capacity}");
    let mut _temp_sums_buffer = MetalBuffer::<Frac<EF>>::with_capacity(temp_sums_buffer_capacity);
    let used_temp_bytes =
        intermed_capacity * size_of::<F>() + temp_sums_buffer_capacity * size_of::<Frac<EF>>();
    if used_temp_bytes > max_temp_bytes {
        warn!(
            "logup_round0 used_temp_bytes ({used_temp_bytes}) > max_temp_bytes ({max_temp_bytes})"
        );
    }

    let _preprocessed_ptr = pk
        .preprocessed_data
        .as_ref()
        .map(|cd| cd.trace.buffer().as_ptr())
        .unwrap_or(std::ptr::null());

    let s_evals = MetalBuffer::<Frac<EF>>::with_capacity(large_domain as usize);

    // TODO: dispatch logup_bary_eval_interactions_round0 Metal kernel when available.
    let _ = (
        selectors_cube,
        main_parts,
        public_values,
        eq_cube,
        height,
        g_shift,
    );

    Ok(s_evals)
}
