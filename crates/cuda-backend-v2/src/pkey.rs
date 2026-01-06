//! Defines the symbolic rule data to precompute and store in the GPU proving key
use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::MemCopyError};
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, SymbolicDagBuilder, SymbolicExpressionDag,
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::FieldAlgebra;
use stark_backend_v2::keygen::types::StarkProvingKeyV2;

use crate::{
    F,
    logup_zerocheck::rules::{SymbolicRulesGpuV2, codec::Codec},
};

pub struct AirDataGpu {
    pub interaction_rules: InteractionEvalRules,
    /// Whether to buffer vars depends on the performance and memory access patterns of the kernel.
    /// This may be tuned.
    pub zerocheck_round0: ConstraintOnlyRules<true>,
    pub zerocheck_mle: ConstraintOnlyRules<false>,
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

impl AirDataGpu {
    pub fn new(pk: &StarkProvingKeyV2) -> Result<Self, MemCopyError> {
        let dag = &pk.vk.symbolic_constraints;
        let symbolic_constraints = SymbolicConstraints::from(dag);
        let interaction_rules = InteractionEvalRules::new(&symbolic_constraints)?;
        let zerocheck_round0 = ConstraintOnlyRules::<true>::new(&dag.constraints)?;
        let zerocheck_mle = ConstraintOnlyRules::<false>::new(&dag.constraints)?;
        Ok(Self {
            interaction_rules,
            zerocheck_round0,
            zerocheck_mle,
        })
    }
}

impl InteractionEvalRules {
    pub fn new(symbolic_constraints: &SymbolicConstraints<F>) -> Result<Self, MemCopyError> {
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
            let b = SymbolicExpression::from_canonical_u32(interaction.bus_index as u32 + 1);
            let betas = symbolic_challenges[1..].to_vec();
            let mut denom = SymbolicExpression::from_canonical_u32(0);
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
        let rules = SymbolicRulesGpuV2::new(&dag, false);
        // Build used_nodes with duplicates, preserving order from constraint_idx
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|&dag_idx| rules.dag_idx_to_rule_idx[&dag_idx])
            .collect_vec();
        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = used_nodes.to_device()?;
        let d_pair_idxs = pair_idxs.to_device()?;
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
    pub fn new(dag: &SymbolicExpressionDag<F>) -> Result<Self, MemCopyError> {
        if dag.num_constraints() == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),
            });
        }

        let rules = SymbolicRulesGpuV2::new(dag, BUFFER_VARS);
        // Build used_nodes with duplicates, preserving order from constraint_idx
        let used_nodes = dag
            .constraint_idx
            .iter()
            .map(|&dag_idx| rules.dag_idx_to_rule_idx[&dag_idx])
            .collect_vec();

        let encoded_rules = rules.rules.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = used_nodes.to_device()?;

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
