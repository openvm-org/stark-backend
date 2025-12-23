//! Defines the symbolic rule data to precompute and store in the GPU proving key
use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::MemCopyError};
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, SymbolicConstraintsDag,
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::FieldAlgebra;
use rustc_hash::FxHashMap;
use stark_backend_v2::keygen::types::StarkProvingKeyV2;

use crate::{
    F,
    logup_zerocheck::rules::{SymbolicRulesOnGpuV2, codec::Codec},
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
    pub(crate) max_fields_len: usize,
}

/// Constraints only, no interactions
pub struct ConstraintOnlyRules<const BUFFER_VARS: bool> {
    pub(crate) inner: EvalRules,
    pub d_lambda_indices: DeviceBuffer<u32>,
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
        let zerocheck_round0 = ConstraintOnlyRules::<true>::new(dag)?;
        let zerocheck_mle = ConstraintOnlyRules::<false>::new(dag)?;
        Ok(Self {
            interaction_rules,
            zerocheck_round0,
            zerocheck_mle,
        })
    }
}

impl InteractionEvalRules {
    // TODO[jpw]: this is a weird way to express the logup denominator with beta_pows
    // symbolically. Revisit if it's possible to do it more directly
    pub fn new(symbolic_constraints: &SymbolicConstraints<F>) -> Result<Self, MemCopyError> {
        let interactions = &symbolic_constraints.interactions;
        let num_interactions = interactions.len();
        if num_interactions == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),
                max_fields_len: 0,
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
        let rules = SymbolicRulesOnGpuV2::new(&constraints_dag, true, false);
        let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = rules.used_nodes.to_device()?;
        assert_eq!(
            rules.used_nodes.len(),
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
            max_fields_len,
        })
    }
}

impl<const BUFFER_VARS: bool> ConstraintOnlyRules<BUFFER_VARS> {
    pub fn new(dag: &SymbolicConstraintsDag<F>) -> Result<Self, MemCopyError> {
        if dag.constraints.num_constraints() == 0 {
            return Ok(Self {
                inner: EvalRules::dummy(),
                d_lambda_indices: DeviceBuffer::new(),
            });
        }

        let lambda_index_map: FxHashMap<usize, usize> = dag
            .constraints
            .constraint_idx
            .iter()
            .enumerate()
            .map(|(idx, dag_idx)| (*dag_idx, idx))
            .collect();
        let rules = SymbolicRulesOnGpuV2::new(dag, false, BUFFER_VARS);

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
        let d_lambda_indices = lambda_indices_host.to_device()?;

        let encoded_rules = rules.constraints.iter().map(|c| c.encode()).collect_vec();
        let d_rules = encoded_rules.to_device()?;
        let d_used_nodes = rules.used_nodes.to_device()?;

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
            d_lambda_indices,
        })
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
