use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use p3_field::Field;

use super::{
    dag::{SymbolicExpressionDag, SymbolicExpressionNode},
    packing::{
        entry_parts, pack_index, ENTRY_TYPE_MAIN, ENTRY_TYPE_PREPROCESSED, ENTRY_TYPE_PUBLIC,
    },
    symbolic_variable::SymbolicVariable,
};

/// Packed variable following the CUDA monomial layout.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct PackedVar(pub u32);

impl PackedVar {
    pub fn new(entry_type: u8, part_index: u8, offset: u8, col_index: u16) -> Self {
        Self(
            (entry_type as u32)
                | ((part_index as u32) << 4)
                | ((offset as u32) << 12)
                | ((col_index as u32) << 16),
        )
    }

    pub fn from_symbolic_var<F: Field>(var: &SymbolicVariable<F>) -> Self {
        let (entry_type, part_index, offset) = entry_parts(&var.entry);
        match entry_type {
            ENTRY_TYPE_PREPROCESSED | ENTRY_TYPE_MAIN | ENTRY_TYPE_PUBLIC => {}
            _ => panic!("unsupported symbolic entry in zerocheck monomial extraction"),
        }
        Self::new(entry_type, part_index, offset, pack_index(var.index))
    }

    pub fn is_first() -> Self {
        Self::new(8, 0, 0, 0)
    }

    pub fn is_last() -> Self {
        Self::new(9, 0, 0, 0)
    }

    pub fn is_transition() -> Self {
        Self::new(10, 0, 0, 0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonomialHeader {
    pub var_offset: u32,
    pub lambda_offset: u32,
    pub num_vars: u16,
    pub num_lambdas: u16,
}

/// A (constraint_idx, coefficient) pair in F[lambda].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct LambdaTerm<F> {
    pub constraint_idx: u32,
    pub coefficient: F,
}

#[derive(Clone)]
struct ExpandedMonomial<F> {
    pub variables: Vec<PackedVar>,
    pub coefficient: F,
}

/// Expanded monomials serialized for GPU upload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct ExpandedMonomials<F> {
    pub headers: Vec<MonomialHeader>,
    pub variables: Vec<PackedVar>,
    pub lambda_terms: Vec<LambdaTerm<F>>,
}

impl<F: Field> ExpandedMonomials<F> {
    pub fn from_dag(dag: &SymbolicExpressionDag<F>) -> Self {
        let mut cache: FxHashMap<usize, Vec<ExpandedMonomial<F>>> = FxHashMap::default();

        let mut monomial_map: FxHashMap<Vec<PackedVar>, Vec<(u32, F)>> = FxHashMap::default();

        for (constraint_idx, &dag_idx) in dag.constraint_idx.iter().enumerate() {
            let expanded = expand_node_cached(dag, dag_idx, &mut cache);
            for mono in expanded {
                if mono.coefficient == F::ZERO {
                    continue;
                }
                monomial_map
                    .entry(mono.variables)
                    .or_default()
                    .push((constraint_idx as u32, mono.coefficient));
            }
        }

        let mut headers = Vec::with_capacity(monomial_map.len());
        let mut all_vars = Vec::new();
        let mut all_lambda_terms = Vec::new();

        for (vars, lambda_terms) in monomial_map {
            headers.push(MonomialHeader {
                var_offset: all_vars.len() as u32,
                num_vars: vars.len() as u16,
                lambda_offset: all_lambda_terms.len() as u32,
                num_lambdas: lambda_terms.len() as u16,
            });
            all_vars.extend(vars);
            all_lambda_terms.extend(
                lambda_terms
                    .into_iter()
                    .map(|(idx, coeff)| LambdaTerm {
                        constraint_idx: idx,
                        coefficient: coeff,
                    }),
            );
        }

        Self {
            headers,
            variables: all_vars,
            lambda_terms: all_lambda_terms,
        }
    }
}

fn expand_node_cached<F: Field>(
    dag: &SymbolicExpressionDag<F>,
    idx: usize,
    cache: &mut FxHashMap<usize, Vec<ExpandedMonomial<F>>>,
) -> Vec<ExpandedMonomial<F>> {
    if let Some(cached) = cache.get(&idx) {
        return cached.clone();
    }

    let result = match &dag.nodes[idx] {
        SymbolicExpressionNode::Constant(c) => vec![ExpandedMonomial {
            variables: vec![],
            coefficient: *c,
        }],
        SymbolicExpressionNode::Variable(v) => vec![ExpandedMonomial {
            variables: vec![PackedVar::from_symbolic_var(v)],
            coefficient: F::ONE,
        }],
        SymbolicExpressionNode::IsFirstRow => vec![ExpandedMonomial {
            variables: vec![PackedVar::is_first()],
            coefficient: F::ONE,
        }],
        SymbolicExpressionNode::IsLastRow => vec![ExpandedMonomial {
            variables: vec![PackedVar::is_last()],
            coefficient: F::ONE,
        }],
        SymbolicExpressionNode::IsTransition => vec![ExpandedMonomial {
            variables: vec![PackedVar::is_transition()],
            coefficient: F::ONE,
        }],
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            ..
        } => {
            let mut result = expand_node_cached(dag, *left_idx, cache);
            result.extend(expand_node_cached(dag, *right_idx, cache));
            combine_like_terms(result)
        }
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            ..
        } => {
            let mut result = expand_node_cached(dag, *left_idx, cache);
            for mut mono in expand_node_cached(dag, *right_idx, cache) {
                mono.coefficient = -mono.coefficient;
                result.push(mono);
            }
            combine_like_terms(result)
        }
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            ..
        } => {
            let left = expand_node_cached(dag, *left_idx, cache);
            let right = expand_node_cached(dag, *right_idx, cache);
            let mut result = Vec::with_capacity(left.len() * right.len());
            for l in &left {
                for r in &right {
                    let mut vars = l.variables.clone();
                    vars.extend(&r.variables);
                    vars.sort();
                    result.push(ExpandedMonomial {
                        variables: vars,
                        coefficient: l.coefficient * r.coefficient,
                    });
                }
            }
            combine_like_terms(result)
        }
        SymbolicExpressionNode::Neg { idx, .. } => expand_node_cached(dag, *idx, cache)
            .into_iter()
            .map(|mut mono| {
                mono.coefficient = -mono.coefficient;
                mono
            })
            .collect(),
    };

    cache.insert(idx, result.clone());
    result
}

fn combine_like_terms<F: Field>(monomials: Vec<ExpandedMonomial<F>>) -> Vec<ExpandedMonomial<F>> {
    let mut map: FxHashMap<Vec<PackedVar>, F> = FxHashMap::default();
    for mono in monomials {
        *map.entry(mono.variables).or_insert(F::ZERO) += mono.coefficient;
    }
    map.into_iter()
        .filter(|(_, coeff)| *coeff != F::ZERO)
        .map(|(variables, coefficient)| ExpandedMonomial {
            variables,
            coefficient,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;

    use crate::air_builders::symbolic::{
        dag::build_symbolic_constraints_dag,
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
    };

    use super::{ExpandedMonomials, PackedVar};

    type F = BabyBear;

    #[test]
    fn test_expand_monomials_basic() {
        let x = SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            0,
        );

        let expr0 = SymbolicExpression::from(x);
        let expr1 = (SymbolicExpression::from(x) + F::ONE) * (SymbolicExpression::from(x) + F::ONE);

        let constraints = vec![expr0, expr1];
        let dag = build_symbolic_constraints_dag(&constraints, &[]);
        let expanded = ExpandedMonomials::from_dag(&dag.constraints);

        let mut actual: BTreeMap<Vec<PackedVar>, Vec<(u32, F)>> = BTreeMap::new();
        for header in &expanded.headers {
            let vars = expanded.variables
                [header.var_offset as usize..header.var_offset as usize + header.num_vars as usize]
                .to_vec();
            let mut terms: Vec<(u32, F)> = expanded.lambda_terms[header.lambda_offset as usize
                ..header.lambda_offset as usize + header.num_lambdas as usize]
                .iter()
                .map(|t| (t.constraint_idx, t.coefficient))
                .collect();
            terms.sort_by_key(|(idx, _)| *idx);
            actual.insert(vars, terms);
        }

        let packed_x = PackedVar::from_symbolic_var(&x);
        let mut expected: BTreeMap<Vec<PackedVar>, Vec<(u32, F)>> = BTreeMap::new();
        expected.insert(vec![], vec![(1, F::ONE)]);
        expected.insert(
            vec![packed_x],
            vec![(0, F::ONE), (1, F::from_canonical_u8(2))],
        );
        expected.insert(vec![packed_x, packed_x], vec![(1, F::ONE)]);

        assert_eq!(actual, expected);
    }
}
