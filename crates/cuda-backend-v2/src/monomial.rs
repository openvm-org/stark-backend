//! Monomial expansion for GPU zerocheck evaluation.
//!
//! This module expands symbolic constraint DAGs into monomials for efficient
//! GPU evaluation via the monomial kernel.

use openvm_stark_backend::air_builders::symbolic::{
    SymbolicExpressionDag, SymbolicExpressionNode,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::Field;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Packed variable following the CUDA monomial layout.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PackedVar(pub u32);

impl PackedVar {
    pub fn new(entry_type: u8, part_index: u8, offset: u8, col_index: u16) -> Self {
        assert!(offset < 16, "PackedVar offset must fit in 4 bits");
        Self(
            (entry_type as u32)
                | ((part_index as u32) << 4)
                | ((offset as u32) << 12)
                | ((col_index as u32) << 16),
        )
    }

    pub fn from_symbolic_var<F: Field>(var: &SymbolicVariable<F>) -> Self {
        assert!(
            var.index <= u16::MAX as usize,
            "symbolic column index exceeds PackedVar capacity"
        );
        let (entry_type, part_index, offset) = match var.entry {
            Entry::Main { part_index, offset } => (1, part_index as u8, offset as u8),
            Entry::Preprocessed { offset } => (0, 0, offset as u8),
            Entry::Public => (3, 0, 0),
            Entry::Permutation { .. } | Entry::Challenge | Entry::Exposed => {
                panic!("unsupported symbolic entry in zerocheck monomial extraction")
            }
        };
        Self::new(entry_type, part_index, offset, var.index as u16)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonomialHeader {
    pub var_offset: u32,
    pub lambda_offset: u32,
    pub num_vars: u16,
    pub num_lambdas: u16,
}

/// A (constraint_idx, coefficient) pair in F[lambda].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpandedMonomials<F> {
    pub headers: Vec<MonomialHeader>,
    pub variables: Vec<PackedVar>,
    pub lambda_terms: Vec<LambdaTerm<F>>,
}

impl<F: Field> ExpandedMonomials<F> {
    pub fn from_dag(dag: &SymbolicExpressionDag<F>) -> Self {
        let mut cache: FxHashMap<usize, Arc<[ExpandedMonomial<F>]>> = FxHashMap::default();

        let mut monomial_map: FxHashMap<Vec<PackedVar>, Vec<(u32, F)>> = FxHashMap::default();

        for (constraint_idx, &dag_idx) in dag.constraint_idx.iter().enumerate() {
            let expanded = expand_node_cached(dag, dag_idx, &mut cache);
            for mono in expanded.iter() {
                if mono.coefficient == F::ZERO {
                    continue;
                }
                monomial_map
                    .entry(mono.variables.clone())
                    .or_default()
                    .push((constraint_idx as u32, mono.coefficient));
            }
        }

        let mut headers = Vec::with_capacity(monomial_map.len());
        let mut all_vars = Vec::new();
        let mut all_lambda_terms = Vec::new();

        for (vars, lambda_terms) in monomial_map {
            assert!(
                vars.len() <= u16::MAX as usize,
                "monomial has too many variables for PackedVar header"
            );
            assert!(
                lambda_terms.len() <= u16::MAX as usize,
                "monomial has too many lambda terms for PackedVar header"
            );
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
    cache: &mut FxHashMap<usize, Arc<[ExpandedMonomial<F>]>>,
) -> Arc<[ExpandedMonomial<F>]> {
    if let Some(cached) = cache.get(&idx) {
        return Arc::clone(cached);
    }

    let result_vec = match &dag.nodes[idx] {
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
            let left = expand_node_cached(dag, *left_idx, cache);
            let right = expand_node_cached(dag, *right_idx, cache);
            let mut result = Vec::with_capacity(left.len() + right.len());
            result.extend(left.iter().cloned());
            result.extend(right.iter().cloned());
            combine_like_terms(result)
        }
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            ..
        } => {
            let left = expand_node_cached(dag, *left_idx, cache);
            let right = expand_node_cached(dag, *right_idx, cache);
            let mut result = Vec::with_capacity(left.len() + right.len());
            result.extend(left.iter().cloned());
            for mut mono in right.iter().cloned() {
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
            for l in left.iter() {
                for r in right.iter() {
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
            .iter()
            .cloned()
            .map(|mut mono| {
                mono.coefficient = -mono.coefficient;
                mono
            })
            .collect(),
    };

    let result = Arc::from(result_vec);
    cache.insert(idx, Arc::clone(&result));
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
