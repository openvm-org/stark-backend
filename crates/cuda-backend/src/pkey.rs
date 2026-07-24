//! Defines the symbolic rule data to precompute and store in the GPU proving key
use itertools::Itertools;
use openvm_cuda_common::{
    copy::MemCopyH2D, d_buffer::DeviceBuffer, error::MemCopyError, stream::GpuDeviceCtx,
};
use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints, SymbolicDagBuilder, SymbolicExpressionDag,
    },
    keygen::types::StarkProvingKey,
    StarkProtocolConfig,
};
use p3_field::PrimeCharacteristicRing;

use crate::{
    logup_zerocheck::rules::{codec::Codec, SymbolicRulesGpu},
    monomial::{
        ExpandedInteractionMonomials, ExpandedMonomials, InteractionMonomialTerm, LambdaTerm,
        MonomialHeader, PackedVar,
    },
    prelude::F,
};

pub struct AirDataGpu {
    pub interaction_rules: InteractionEvalRules,
    /// Whether to buffer vars depends on the performance and memory access patterns of the kernel.
    /// This may be tuned.
    pub zerocheck_round0: ConstraintOnlyRules<true>,
    pub interaction_round0: Option<InteractionRound0Rules>,
    pub zerocheck_mle: ConstraintOnlyRules<false>,
    pub zerocheck_monomials: Option<ZerocheckMonomials>,
    pub interaction_monomials: Option<InteractionMonomials>,
}

/// Used for logup round 0: the interactions DAG structure, which depends only on the AIR
/// (not on challenges), so it is built once at keygen. The per-prove work reduces to
/// computing the challenge-dependent `(numer, denom)` weight vectors from the cached
/// per-interaction rule indices.
///
/// The kernel uses weights indexed by rule idx, not constraint idx, so `constraint_idx`
/// is deduplicated and per-interaction expressions are mapped to rule indices here.
pub struct InteractionRound0Rules {
    /// Encoded rules for the interactions DAG (built with `buffer_vars = true`).
    pub(crate) d_rules: DeviceBuffer<u128>,
    pub(crate) buffer_size: u32,
    /// Number of rules; the length of the weight vectors.
    pub(crate) rules_len: usize,
    /// Rule index of each interaction's `count` expression.
    pub(crate) count_rule_idxs: Vec<u32>,
    /// Rule indices of each interaction's message field expressions.
    pub(crate) message_rule_idxs: Vec<Vec<u32>>,
    /// `(message_len, bus_index)` per interaction, for the denominator init term.
    pub(crate) interaction_meta: Vec<(u32, u32)>,
}

/// Used for GKR input evaluation and logup MLE sumcheck rounds.
pub struct InteractionEvalRules {
    pub(crate) inner: EvalRules,
    /// Constraints consist of all `(numer, denom)` pairs **topologically sorted**. We map the
    /// constraint idx back to unsorted order. ```text
    /// constraint_idx => 2 * interaction_idx + is_denom
    /// ```
    pub(crate) d_pair_idxs: DeviceBuffer<u32>,
    pub(crate) max_fields_len: usize,
}

/// Constraints only, no interactions
pub struct ConstraintOnlyRules<const BUFFER_VARS: bool> {
    pub(crate) inner: EvalRules,
}

pub struct EvalRules {
    /// Encoded rules
    pub d_rules: DeviceBuffer<u128>,
    pub d_used_nodes: DeviceBuffer<usize>,
    pub buffer_size: u32,
}

pub struct ZerocheckMonomials {
    pub d_headers: DeviceBuffer<MonomialHeader>,
    pub d_variables: DeviceBuffer<PackedVar>,
    pub d_lambda_terms: DeviceBuffer<LambdaTerm<F>>,
    pub num_monomials: u32,
}

pub struct InteractionMonomials {
    pub d_numer_headers: DeviceBuffer<MonomialHeader>,
    pub d_numer_variables: DeviceBuffer<PackedVar>,
    pub d_numer_terms: DeviceBuffer<InteractionMonomialTerm<F>>,
    pub num_numer_monomials: u32,
    pub d_denom_headers: DeviceBuffer<MonomialHeader>,
    pub d_denom_variables: DeviceBuffer<PackedVar>,
    pub d_denom_terms: DeviceBuffer<InteractionMonomialTerm<F>>,
    pub num_denom_monomials: u32,
    pub max_fields_len: usize,
    pub num_interactions: u32,
}

fn to_device_or_empty<T>(
    data: &[T],
    device_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<T>, MemCopyError> {
    if data.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        data.to_device_on(device_ctx)
    }
}

impl AirDataGpu {
    pub fn new<S: StarkProtocolConfig<F = F>>(
        pk: &StarkProvingKey<S>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, MemCopyError> {
        let dag = &pk.vk.symbolic_constraints;
        let symbolic_constraints = SymbolicConstraints::from(dag);
        let interaction_rules = InteractionEvalRules::new(&symbolic_constraints, device_ctx)?;
        let zerocheck_round0 = ConstraintOnlyRules::<true>::new(&dag.constraints, device_ctx)?;
        let interaction_round0 = InteractionRound0Rules::new(&symbolic_constraints, device_ctx)?;
        let zerocheck_mle = ConstraintOnlyRules::<false>::new(&dag.constraints, device_ctx)?;

        let zerocheck_monomials = if dag.constraints.num_constraints() > 0 {
            let expanded = ExpandedMonomials::from_dag(&dag.constraints);
            Some(ZerocheckMonomials::from_expanded(&expanded, device_ctx)?)
        } else {
            None
        };
        let interaction_monomials = if !symbolic_constraints.interactions.is_empty() {
            let expanded =
                ExpandedInteractionMonomials::from_symbolic_constraints(&symbolic_constraints);
            Some(InteractionMonomials::from_expanded(&expanded, device_ctx)?)
        } else {
            None
        };
        Ok(Self {
            interaction_rules,
            zerocheck_round0,
            interaction_round0,
            zerocheck_mle,
            zerocheck_monomials,
            interaction_monomials,
        })
    }
}

impl InteractionRound0Rules {
    /// Builds the interactions DAG (structure only; challenge-independent) and the
    /// per-interaction rule index tables. Mirrors the DAG previously rebuilt on every
    /// prove in `evaluate_round0_interactions_gpu`.
    pub fn new(
        symbolic: &SymbolicConstraints<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Option<Self>, MemCopyError> {
        if symbolic.interactions.is_empty() {
            return Ok(None);
        }
        // Copied from build_symbolic_constraints_dag to handle sorting of constraints.
        // NOTE: For logup round0, the kernel uses weights indexed by rule_idx, not
        // constraint_idx. So we deduplicate constraint_idx and use dag_idx_to_rule_idx
        // for weight mapping.
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
        // Deduplicate for the dag since the logup round0 kernel doesn't use used_nodes
        sorted_used_dag_idxs.dedup();
        let dag = SymbolicExpressionDag {
            nodes: dag_builder.nodes,
            constraint_idx: sorted_used_dag_idxs,
        };
        let rules = SymbolicRulesGpu::new(&dag, true);

        // CAUTION: an expression node could be used in multiple interactions, and might
        // even be used as `count` in one, but message field in another. The weight
        // vectors are accumulated per rule index at prove time; here we only record the
        // per-interaction rule indices.
        let mut count_rule_idxs = Vec::with_capacity(symbolic.interactions.len());
        let mut message_rule_idxs = Vec::with_capacity(symbolic.interactions.len());
        let mut interaction_meta = Vec::with_capacity(symbolic.interactions.len());
        for interaction in &symbolic.interactions {
            let count_dag_idx =
                dag_builder.expr_to_idx[&(&interaction.count as *const SymbolicExpression<_>)];
            count_rule_idxs.push(rules.dag_idx_to_rule_idx[&count_dag_idx] as u32);
            message_rule_idxs.push(
                interaction
                    .message
                    .iter()
                    .map(|message| {
                        let message_dag_idx =
                            dag_builder.expr_to_idx[&(message as *const SymbolicExpression<_>)];
                        rules.dag_idx_to_rule_idx[&message_dag_idx] as u32
                    })
                    .collect_vec(),
            );
            interaction_meta.push((
                interaction.message.len() as u32,
                interaction.bus_index as u32,
            ));
        }

        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device_on(device_ctx)?;
        Ok(Some(Self {
            d_rules,
            buffer_size: rules
                .buffer_size
                .try_into()
                .expect("buffer_size exceeds u32"),
            rules_len: rules.rules.len(),
            count_rule_idxs,
            message_rule_idxs,
            interaction_meta,
        }))
    }
}

impl ZerocheckMonomials {
    pub fn from_expanded(
        expanded: &ExpandedMonomials<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, MemCopyError> {
        // Validate bounds for all monomial headers to prevent out-of-bounds access in CUDA kernel
        let num_variables = expanded.variables.len();
        let num_lambda_terms = expanded.lambda_terms.len();
        for (i, hdr) in expanded.headers.iter().enumerate() {
            let var_end = hdr.var_offset as usize + hdr.num_vars as usize;
            let term_end = hdr.term_offset as usize + hdr.num_terms as usize;
            assert!(
                var_end <= num_variables,
                "Monomial {i}: var_offset ({}) + num_vars ({}) = {var_end} exceeds variables.len() ({num_variables})",
                hdr.var_offset,
                hdr.num_vars
            );
            assert!(
                term_end <= num_lambda_terms,
                "Monomial {i}: term_offset ({}) + num_terms ({}) = {term_end} exceeds lambda_terms.len() ({num_lambda_terms})",
                hdr.term_offset,
                hdr.num_terms
            );
        }

        Ok(Self {
            d_headers: expanded.headers.to_device_on(device_ctx)?,
            d_variables: expanded.variables.to_device_on(device_ctx)?,
            d_lambda_terms: expanded.lambda_terms.to_device_on(device_ctx)?,
            num_monomials: expanded.headers.len() as u32,
        })
    }
}

impl InteractionMonomials {
    pub fn from_expanded(
        expanded: &ExpandedInteractionMonomials<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, MemCopyError> {
        // Validate numerator monomial headers
        let num_numer_vars = expanded.numer_variables.len();
        let num_numer_terms = expanded.numer_terms.len();
        for (i, hdr) in expanded.numer_headers.iter().enumerate() {
            let var_end = hdr.var_offset as usize + hdr.num_vars as usize;
            let term_end = hdr.term_offset as usize + hdr.num_terms as usize;
            assert!(
                var_end <= num_numer_vars,
                "Numer monomial {i}: var_offset + num_vars exceeds bounds"
            );
            assert!(
                term_end <= num_numer_terms,
                "Numer monomial {i}: term_offset + num_terms exceeds bounds"
            );
        }

        // Validate denominator monomial headers
        let num_denom_vars = expanded.denom_variables.len();
        let num_denom_terms = expanded.denom_terms.len();
        for (i, hdr) in expanded.denom_headers.iter().enumerate() {
            let var_end = hdr.var_offset as usize + hdr.num_vars as usize;
            let term_end = hdr.term_offset as usize + hdr.num_terms as usize;
            assert!(
                var_end <= num_denom_vars,
                "Denom monomial {i}: var_offset + num_vars exceeds bounds"
            );
            assert!(
                term_end <= num_denom_terms,
                "Denom monomial {i}: term_offset + num_terms exceeds bounds"
            );
        }

        Ok(Self {
            d_numer_headers: to_device_or_empty(&expanded.numer_headers, device_ctx)?,
            d_numer_variables: to_device_or_empty(&expanded.numer_variables, device_ctx)?,
            d_numer_terms: to_device_or_empty(&expanded.numer_terms, device_ctx)?,
            num_numer_monomials: expanded.numer_headers.len() as u32,
            d_denom_headers: to_device_or_empty(&expanded.denom_headers, device_ctx)?,
            d_denom_variables: to_device_or_empty(&expanded.denom_variables, device_ctx)?,
            d_denom_terms: to_device_or_empty(&expanded.denom_terms, device_ctx)?,
            num_denom_monomials: expanded.denom_headers.len() as u32,
            max_fields_len: expanded.max_fields_len,
            num_interactions: expanded.num_interactions,
        })
    }
}

impl InteractionEvalRules {
    pub fn new(
        symbolic_constraints: &SymbolicConstraints<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, MemCopyError> {
        let interactions = &symbolic_constraints.interactions;
        let num_interactions = interactions.len();
        if num_interactions == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),

                max_fields_len: 0,
                d_pair_idxs: DeviceBuffer::new(),
            });
        }
        let max_fields_len = interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        // [alpha, beta^0, ..., beta^max_fields_len]
        let symbolic_challenges: Vec<SymbolicExpression<F>> = (0..max_fields_len + 2)
            .map(|index| SymbolicVariable::<F>::new(Entry::Challenge, index).into())
            .collect();

        let mut frac_pairs = Vec::with_capacity(num_interactions * 2);
        for interaction in interactions.iter() {
            let numer = interaction.count.clone();
            let b = SymbolicExpression::from_u32(interaction.bus_index as u32 + 1);
            let betas = symbolic_challenges[1..].to_vec();
            let mut denom = SymbolicExpression::from_u32(0);
            for (j, expr) in interaction.message.iter().enumerate() {
                denom += betas[j].clone() * expr.clone();
            }
            denom += betas[interaction.message.len()].clone() * b;
            frac_pairs.push(numer);
            frac_pairs.push(denom);
        }
        // build DAG without sorting constraint idxs:
        let (dag, pair_idxs) = {
            let mut dag_builder = SymbolicDagBuilder::new();
            let mut dag_pair_idxs: Vec<(usize, u32)> = frac_pairs
                .iter()
                .enumerate()
                .map(|(pair_idx, expr)| {
                    let dag_idx = dag_builder.add_expr(expr);
                    (dag_idx, pair_idx.try_into().unwrap())
                })
                .collect_vec();
            dag_pair_idxs.sort();
            let (constraint_idx, pair_idxs): (Vec<_>, Vec<_>) = dag_pair_idxs.into_iter().unzip();
            // NOTE: do not sort pair_idxs since we need to keep them in pairs
            let dag = SymbolicExpressionDag {
                nodes: dag_builder.nodes,
                constraint_idx,
            };
            (dag, pair_idxs)
        };
        let rules = SymbolicRulesGpu::new(&dag, false);
        // Build used_nodes with duplicates, preserving order from constraint_idx
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|&dag_idx| rules.dag_idx_to_rule_idx[&dag_idx])
            .collect_vec();
        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device_on(device_ctx)?;
        let d_used_nodes = used_nodes.to_device_on(device_ctx)?;
        let d_pair_idxs = pair_idxs.to_device_on(device_ctx)?;
        assert_eq!(
            used_nodes.len(),
            2 * num_interactions,
            "Rules come in (numer, denom) pairs"
        );

        let inner = EvalRules {
            d_rules,
            d_used_nodes,
            buffer_size: rules
                .buffer_size
                .try_into()
                .expect("buffer_size exceeds u32"),
        };

        Ok(Self {
            inner,
            d_pair_idxs,
            max_fields_len,
        })
    }
}

impl<const BUFFER_VARS: bool> ConstraintOnlyRules<BUFFER_VARS> {
    pub fn new(
        dag: &SymbolicExpressionDag<F>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, MemCopyError> {
        if dag.num_constraints() == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),
            });
        }

        let rules = SymbolicRulesGpu::new(dag, BUFFER_VARS);
        // Build used_nodes with duplicates, preserving order from constraint_idx
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|&dag_idx| rules.dag_idx_to_rule_idx[&dag_idx])
            .collect_vec();

        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device_on(device_ctx)?;
        let d_used_nodes = used_nodes.to_device_on(device_ctx)?;

        let inner = EvalRules {
            d_rules,
            d_used_nodes,
            buffer_size: rules
                .buffer_size
                .try_into()
                .expect("buffer_size exceeds u32"),
        };
        Ok(Self { inner })
    }
}

impl EvalRules {
    pub fn dummy() -> Self {
        Self {
            d_rules: DeviceBuffer::new(),
            d_used_nodes: DeviceBuffer::new(),
            buffer_size: 0,
        }
    }
}
