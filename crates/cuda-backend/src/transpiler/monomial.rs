// Monomial expansion from DAG for batch MLE evaluation
//
// This module provides functionality to expand a constraint DAG into a sum of monomials
// where coefficients are represented in F[λ] (sparse polynomial in the batching variable λ).

use openvm_stark_backend::air_builders::symbolic::{
    dag::{SymbolicExpressionDag, SymbolicExpressionNode},
    symbolic_variable::SymbolicVariable,
};
use p3_field::{Field, PrimeField32};
use rustc_hash::FxHashMap;

use super::codec::Codec;
use super::Source;

/// A term in the λ-polynomial coefficient: coeff * λ^constraint_idx
#[derive(Clone, Copy, Debug)]
pub struct LambdaTerm<F> {
    /// Index of the constraint (power of λ)
    pub constraint_idx: u16,
    /// Coefficient in the base field
    pub coefficient: F,
}

impl<F: PrimeField32> LambdaTerm<F> {
    /// Serialize to 6 bytes: u16 constraint_idx + u32 coefficient
    pub fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.constraint_idx.to_le_bytes());
        out.extend_from_slice(&self.coefficient.as_canonical_u32().to_le_bytes());
    }
}

/// A variable in a monomial - wraps Source<F> but excludes Constant/Intermediate variants
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MonomialVar<F: Field> {
    /// A trace column variable
    Column(SymbolicVariable<F>),
    /// First row selector
    IsFirst,
    /// Last row selector
    IsLast,
    /// Transition rows selector
    IsTransition,
}

impl<F: Field + PrimeField32> MonomialVar<F> {
    /// Convert to Source for encoding
    pub fn to_source(&self) -> Source<F> {
        match self {
            MonomialVar::Column(v) => Source::Var(*v),
            MonomialVar::IsFirst => Source::IsFirst,
            MonomialVar::IsLast => Source::IsLast,
            MonomialVar::IsTransition => Source::IsTransition,
        }
    }

    /// Serialize using existing codec (8 bytes)
    pub fn serialize(&self, out: &mut Vec<u8>) {
        let encoded = self.to_source().encode();
        out.extend_from_slice(&encoded.to_le_bytes());
    }
}

/// A monomial with coefficient in F[λ]
///
/// Represents: coeff(λ) * ∏ᵢ variables[i]
/// where coeff(λ) = Σⱼ lambda_terms[j].coefficient * λ^{lambda_terms[j].constraint_idx}
#[derive(Clone, Debug)]
pub struct Monomial<F: Field> {
    /// Sparse polynomial in λ
    pub lambda_terms: Vec<LambdaTerm<F>>,
    /// Variables in the monomial (sorted for canonical representation)
    pub variables: Vec<MonomialVar<F>>,
}

impl<F: Field + PrimeField32> Monomial<F> {
    /// Serialize to bytes:
    /// - num_lambda_terms: u8
    /// - num_vars: u8
    /// - lambda_terms: [LambdaTerm; num_lambda_terms] (6 bytes each)
    /// - variables: [encoded Source; num_vars] (8 bytes each)
    pub fn serialize(&self, out: &mut Vec<u8>) {
        assert!(self.lambda_terms.len() <= 255, "Too many lambda terms");
        assert!(self.variables.len() <= 255, "Too many variables");

        out.push(self.lambda_terms.len() as u8);
        out.push(self.variables.len() as u8);

        for term in &self.lambda_terms {
            term.serialize(out);
        }
        for var in &self.variables {
            var.serialize(out);
        }
    }
}

/// Intermediate representation during DAG expansion (coefficient in F, not F[λ])
#[derive(Clone, Debug)]
struct RawMonomial<F: Field> {
    coefficient: F,
    variables: Vec<MonomialVar<F>>,
}

/// Expand a single node to a sum of raw monomials
fn expand_node<F: Field + PrimeField32>(
    dag: &SymbolicExpressionDag<F>,
    node_idx: usize,
    cache: &mut Vec<Option<Vec<RawMonomial<F>>>>,
) -> Vec<RawMonomial<F>> {
    if let Some(cached) = &cache[node_idx] {
        return cached.clone();
    }

    let result = match &dag.nodes[node_idx] {
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            ..
        } => {
            let mut left_monomials = expand_node(dag, *left_idx, cache);
            let right_monomials = expand_node(dag, *right_idx, cache);
            left_monomials.extend(right_monomials);
            left_monomials
        }
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            ..
        } => {
            let mut left_monomials = expand_node(dag, *left_idx, cache);
            let right_monomials = expand_node(dag, *right_idx, cache);
            for mut m in right_monomials {
                m.coefficient = -m.coefficient;
                left_monomials.push(m);
            }
            left_monomials
        }
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            ..
        } => {
            let left_monomials = expand_node(dag, *left_idx, cache);
            let right_monomials = expand_node(dag, *right_idx, cache);
            // Cross product
            let mut result = Vec::with_capacity(left_monomials.len() * right_monomials.len());
            for lm in &left_monomials {
                for rm in &right_monomials {
                    let coefficient = lm.coefficient * rm.coefficient;
                    let mut variables = lm.variables.clone();
                    variables.extend(rm.variables.iter().cloned());
                    variables.sort(); // Canonical order
                    result.push(RawMonomial {
                        coefficient,
                        variables,
                    });
                }
            }
            result
        }
        SymbolicExpressionNode::Neg { idx, .. } => {
            let mut monomials = expand_node(dag, *idx, cache);
            for m in &mut monomials {
                m.coefficient = -m.coefficient;
            }
            monomials
        }
        SymbolicExpressionNode::Constant(c) => {
            vec![RawMonomial {
                coefficient: *c,
                variables: vec![],
            }]
        }
        SymbolicExpressionNode::Variable(var) => {
            vec![RawMonomial {
                coefficient: F::ONE,
                variables: vec![MonomialVar::Column(*var)],
            }]
        }
        SymbolicExpressionNode::IsFirstRow => {
            vec![RawMonomial {
                coefficient: F::ONE,
                variables: vec![MonomialVar::IsFirst],
            }]
        }
        SymbolicExpressionNode::IsLastRow => {
            vec![RawMonomial {
                coefficient: F::ONE,
                variables: vec![MonomialVar::IsLast],
            }]
        }
        SymbolicExpressionNode::IsTransition => {
            vec![RawMonomial {
                coefficient: F::ONE,
                variables: vec![MonomialVar::IsTransition],
            }]
        }
    };

    cache[node_idx] = Some(result.clone());
    result
}

/// Serialized monomials ready for GPU consumption
#[derive(Clone, Debug)]
pub struct SerializedMonomials {
    /// Concatenated serialized monomial data
    pub data: Vec<u8>,
    /// Byte offset of each monomial in `data`
    pub offsets: Vec<u32>,
    /// Number of monomials
    pub num_monomials: usize,
}

use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};

/// Monomials stored on GPU device
pub struct DeviceMonomials {
    /// Serialized monomial data on device
    pub d_data: DeviceBuffer<u8>,
    /// Byte offsets on device
    pub d_offsets: DeviceBuffer<u32>,
    /// Number of monomials
    pub num_monomials: u32,
}

impl DeviceMonomials {
    /// Transfer serialized monomials to device
    pub fn from_serialized(serialized: &SerializedMonomials) -> Self {
        let d_data = serialized.data.as_slice().to_device().unwrap();
        let d_offsets = serialized.offsets.as_slice().to_device().unwrap();
        Self {
            d_data,
            d_offsets,
            num_monomials: serialized.num_monomials as u32,
        }
    }

    /// Create from a constraint DAG, returning None if expansion fails
    pub fn from_dag<F: Field + PrimeField32>(
        dag: &SymbolicExpressionDag<F>,
        config: &MonomialExpansionConfig,
    ) -> Option<Self> {
        match expand_dag_to_monomials(dag, config) {
            MonomialExpansionResult::Success(serialized) => {
                Some(Self::from_serialized(&serialized))
            }
            MonomialExpansionResult::TooManyMonomials { .. } => None,
            MonomialExpansionResult::Monomials(_) => unreachable!(),
        }
    }
}

/// Configuration for monomial expansion
#[derive(Clone, Debug)]
pub struct MonomialExpansionConfig {
    /// Maximum total monomials before falling back (exponential blowup protection)
    pub max_monomials: usize,
}

impl Default for MonomialExpansionConfig {
    fn default() -> Self {
        Self {
            max_monomials: 1_000_000,
        }
    }
}

/// Result of monomial expansion
pub enum MonomialExpansionResult<F: Field> {
    /// Successfully expanded to monomials
    Success(SerializedMonomials),
    /// Fell back because monomial count exceeded threshold
    TooManyMonomials { count: usize },
    /// Contains deduplicated monomials (for inspection/debugging)
    #[allow(dead_code)]
    Monomials(Vec<Monomial<F>>),
}

/// Expand constraint DAG to deduplicated monomials with F[λ] coefficients
pub fn expand_dag_to_monomials<F: Field + PrimeField32>(
    dag: &SymbolicExpressionDag<F>,
    config: &MonomialExpansionConfig,
) -> MonomialExpansionResult<F> {
    // Map from canonical variable list to accumulated lambda terms
    let mut monomial_map: FxHashMap<Vec<MonomialVar<F>>, Vec<LambdaTerm<F>>> = FxHashMap::default();

    let mut cache = vec![None; dag.nodes.len()];
    let mut total_raw_monomials = 0usize;

    // Process each constraint
    for (constraint_idx, &node_idx) in dag.constraint_idx.iter().enumerate() {
        let raw_monomials = expand_node(dag, node_idx, &mut cache);
        total_raw_monomials += raw_monomials.len();

        // Check for blowup
        if total_raw_monomials > config.max_monomials {
            return MonomialExpansionResult::TooManyMonomials {
                count: total_raw_monomials,
            };
        }

        // Add to deduplicated map
        for raw in raw_monomials {
            if raw.coefficient == F::ZERO {
                continue;
            }
            let entry = monomial_map.entry(raw.variables).or_default();
            // Check if we already have a term for this constraint
            if let Some(term) = entry
                .iter_mut()
                .find(|t| t.constraint_idx == constraint_idx as u16)
            {
                term.coefficient += raw.coefficient;
            } else {
                entry.push(LambdaTerm {
                    constraint_idx: constraint_idx as u16,
                    coefficient: raw.coefficient,
                });
            }
        }
    }

    // Convert map to Vec<Monomial> and filter out zero coefficients
    let monomials: Vec<Monomial<F>> = monomial_map
        .into_iter()
        .filter_map(|(variables, mut lambda_terms)| {
            // Remove zero-coefficient terms
            lambda_terms.retain(|t| t.coefficient != F::ZERO);
            if lambda_terms.is_empty() {
                None
            } else {
                Some(Monomial {
                    lambda_terms,
                    variables,
                })
            }
        })
        .collect();

    // Serialize
    let serialized = serialize_monomials(&monomials);
    MonomialExpansionResult::Success(serialized)
}

/// Serialize a list of monomials to bytes
pub fn serialize_monomials<F: Field + PrimeField32>(
    monomials: &[Monomial<F>],
) -> SerializedMonomials {
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(monomials.len());

    for monomial in monomials {
        offsets.push(data.len() as u32);
        monomial.serialize(&mut data);
    }

    SerializedMonomials {
        data,
        offsets,
        num_monomials: monomials.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openvm_stark_backend::air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, symbolic_variable::Entry, SymbolicConstraints,
        SymbolicConstraintsDag,
    };
    use p3_baby_bear::BabyBear;

    type F = BabyBear;

    fn make_dag(constraints: Vec<SymbolicExpression<F>>) -> SymbolicConstraintsDag<F> {
        let sc = SymbolicConstraints {
            constraints,
            interactions: vec![],
        };
        sc.into()
    }

    #[test]
    fn test_simple_monomial_expansion() {
        // Build a simple DAG: x * y + 2 * z
        let x = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            0,
        );
        let y = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            1,
        );
        let z = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            2,
        );

        let x_expr = SymbolicExpression::from(x);
        let y_expr = SymbolicExpression::from(y);
        let z_expr = SymbolicExpression::from(z);
        let two = SymbolicExpression::Constant(F::from_canonical_u32(2));

        // x * y + 2 * z
        let constraint = x_expr * y_expr + two * z_expr;

        let dag = make_dag(vec![constraint]);
        let config = MonomialExpansionConfig::default();

        match expand_dag_to_monomials(&dag.constraints, &config) {
            MonomialExpansionResult::Success(serialized) => {
                assert_eq!(serialized.num_monomials, 2); // x*y and 2*z
            }
            _ => panic!("Expected successful expansion"),
        }
    }

    #[test]
    fn test_monomial_deduplication() {
        // Build a DAG with two constraints that share monomials
        // Constraint 0: x + y
        // Constraint 1: x + 2*y
        // Should result in: x (from both), y (from 0), 2*y (from 1)
        // After dedup by variables: x with lambda_terms [(0,1), (1,1)]
        //                           y with lambda_terms [(0,1), (1,2)]
        let x = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            0,
        );
        let y = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            1,
        );

        let x_expr = SymbolicExpression::from(x);
        let y_expr = SymbolicExpression::from(y);
        let two = SymbolicExpression::Constant(F::from_canonical_u32(2));

        // Constraint 0: x + y
        let constraint0 = x_expr.clone() + y_expr.clone();
        // Constraint 1: x + 2*y
        let constraint1 = x_expr + two * y_expr;

        let dag = make_dag(vec![constraint0, constraint1]);
        let config = MonomialExpansionConfig::default();

        match expand_dag_to_monomials(&dag.constraints, &config) {
            MonomialExpansionResult::Success(serialized) => {
                // Two unique monomials: x and y
                assert_eq!(serialized.num_monomials, 2);
            }
            _ => panic!("Expected successful expansion"),
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let x = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            10,
        );
        let var = MonomialVar::Column(x);

        let mut data = Vec::new();
        var.serialize(&mut data);

        // Should be 8 bytes (u64)
        assert_eq!(data.len(), 8);

        // Verify encoding matches codec
        let encoded = var.to_source().encode();
        assert_eq!(u64::from_le_bytes(data.try_into().unwrap()), encoded);
    }
}
