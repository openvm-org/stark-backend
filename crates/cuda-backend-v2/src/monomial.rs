//! Monomial expansion for GPU zerocheck evaluation.
//!
//! This module expands symbolic constraint DAGs into monomials for efficient
//! GPU evaluation via the monomial kernel.

use std::sync::Arc;

use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraints, SymbolicExpressionDag, SymbolicExpressionNode,
    symbolic_expression::SymbolicExpression,
    symbolic_variable::{Entry, SymbolicVariable},
};
use p3_field::Field;
use rustc_hash::FxHashMap;

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
    pub term_offset: u32,
    pub num_vars: u16,
    pub num_terms: u16,
}

/// A (constraint_idx, coefficient) pair in F[lambda].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LambdaTerm<F> {
    pub constraint_idx: u32,
    pub coefficient: F,
}

/// Term mapping a monomial to its interaction context.
/// For numerator: sum_i(coefficient_i * eq_3bs[interaction_idx_i])
/// For denominator: sum_i(coefficient_i * beta_pows[field_idx_i] * eq_3bs[interaction_idx_i])
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractionMonomialTerm<F> {
    pub coefficient: F,
    pub interaction_idx: u16,
    pub field_idx: u16, // For denom: index into message for beta_pows. For numer: unused.
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

        let mut monomials: Vec<_> = monomial_map.into_iter().collect();
        monomials.sort_by(|(vars_a, _), (vars_b, _)| {
            vars_a
                .len()
                .cmp(&vars_b.len())
                .then_with(|| vars_a.cmp(vars_b))
        });

        let (headers, all_vars, all_lambda_terms) = serialize_monomials(
            monomials,
            |terms| terms.sort_by_key(|(constraint_idx, _)| *constraint_idx),
            |(idx, coeff)| LambdaTerm {
                constraint_idx: idx,
                coefficient: coeff,
            },
        );

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
        SymbolicExpressionNode::Constant(c) => expand_leaf(vec![], *c),
        SymbolicExpressionNode::Variable(v) => {
            expand_leaf(vec![PackedVar::from_symbolic_var(v)], F::ONE)
        }
        SymbolicExpressionNode::IsFirstRow => expand_leaf(vec![PackedVar::is_first()], F::ONE),
        SymbolicExpressionNode::IsLastRow => expand_leaf(vec![PackedVar::is_last()], F::ONE),
        SymbolicExpressionNode::IsTransition => {
            expand_leaf(vec![PackedVar::is_transition()], F::ONE)
        }
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

fn expand_leaf<F: Field>(variables: Vec<PackedVar>, coefficient: F) -> Vec<ExpandedMonomial<F>> {
    vec![ExpandedMonomial {
        variables,
        coefficient,
    }]
}

fn serialize_monomials<TermIn, TermOut, FSort, FMap>(
    monomials: impl IntoIterator<Item = (Vec<PackedVar>, Vec<TermIn>)>,
    mut sort_terms: FSort,
    mut map_term: FMap,
) -> (Vec<MonomialHeader>, Vec<PackedVar>, Vec<TermOut>)
where
    FSort: FnMut(&mut Vec<TermIn>),
    FMap: FnMut(TermIn) -> TermOut,
{
    let iter = monomials.into_iter();
    let (min, _) = iter.size_hint();
    let mut headers = Vec::with_capacity(min);
    let mut all_vars = Vec::new();
    let mut all_terms = Vec::new();

    for (vars, mut terms) in iter {
        sort_terms(&mut terms);
        assert!(
            vars.len() <= u16::MAX as usize,
            "monomial has too many variables for PackedVar header"
        );
        assert!(
            terms.len() <= u16::MAX as usize,
            "monomial has too many terms for PackedVar header"
        );
        headers.push(MonomialHeader {
            var_offset: all_vars.len() as u32,
            num_vars: vars.len() as u16,
            term_offset: all_terms.len() as u32,
            num_terms: terms.len() as u16,
        });
        all_vars.extend(vars);
        all_terms.extend(terms.into_iter().map(&mut map_term));
    }

    (headers, all_vars, all_terms)
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

// ============================================================================
// INTERACTION (LOGUP) MONOMIAL EXPANSION
// ============================================================================

/// Expanded interaction monomials for GPU upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpandedInteractionMonomials<F> {
    /// Numerator monomial headers
    pub numer_headers: Vec<MonomialHeader>,
    /// Numerator monomial variables
    pub numer_variables: Vec<PackedVar>,
    /// Numerator interaction terms: (interaction_idx, 0, coefficient)
    pub numer_terms: Vec<InteractionMonomialTerm<F>>,
    /// Denominator monomial headers
    pub denom_headers: Vec<MonomialHeader>,
    /// Denominator monomial variables
    pub denom_variables: Vec<PackedVar>,
    /// Denominator interaction terms: (interaction_idx, field_idx, coefficient)
    pub denom_terms: Vec<InteractionMonomialTerm<F>>,
    /// Number of interactions
    pub num_interactions: u32,
    /// Maximum message length across all interactions
    pub max_fields_len: usize,
}

impl<F: Field> ExpandedInteractionMonomials<F> {
    /// Create expanded interaction monomials from symbolic constraints.
    pub fn from_symbolic_constraints(symbolic: &SymbolicConstraints<F>) -> Self {
        let interactions = &symbolic.interactions;
        if interactions.is_empty() {
            return Self {
                numer_headers: Vec::new(),
                numer_variables: Vec::new(),
                numer_terms: Vec::new(),
                denom_headers: Vec::new(),
                denom_variables: Vec::new(),
                denom_terms: Vec::new(),
                num_interactions: 0,
                max_fields_len: 0,
            };
        }

        let max_fields_len = interactions
            .iter()
            .map(|i| i.message.len())
            .max()
            .unwrap_or(0);

        // Build numerator monomials: expand count expressions
        // Group by variables -> (interaction_idx, coefficient)
        let mut numer_map: FxHashMap<Vec<PackedVar>, Vec<(u16, F)>> = FxHashMap::default();
        for (interaction_idx, interaction) in interactions.iter().enumerate() {
            let monomials = expand_symbolic_expression(&interaction.count);
            for mono in monomials {
                if mono.coefficient == F::ZERO {
                    continue;
                }
                numer_map
                    .entry(mono.variables)
                    .or_default()
                    .push((interaction_idx as u16, mono.coefficient));
            }
        }

        // Build denominator monomials: expand each message field
        // Each message field contributes: coeff * beta_pows[field_idx] * eq_3bs[interaction_idx]
        // Group by variables -> (interaction_idx, field_idx, coefficient)
        let mut denom_map: FxHashMap<Vec<PackedVar>, Vec<(u16, u16, F)>> = FxHashMap::default();
        for (interaction_idx, interaction) in interactions.iter().enumerate() {
            for (field_idx, field_expr) in interaction.message.iter().enumerate() {
                let monomials = expand_symbolic_expression(field_expr);
                for mono in monomials {
                    if mono.coefficient == F::ZERO {
                        continue;
                    }
                    denom_map.entry(mono.variables).or_default().push((
                        interaction_idx as u16,
                        field_idx as u16,
                        mono.coefficient,
                    ));
                }
            }
        }

        let (numer_headers, numer_variables, numer_terms) = serialize_monomials(
            numer_map,
            |_| {},
            |(idx, coeff)| InteractionMonomialTerm {
                interaction_idx: idx,
                field_idx: 0, // unused for numerator
                coefficient: coeff,
            },
        );

        let (denom_headers, denom_variables, denom_terms) = serialize_monomials(
            denom_map,
            |_| {},
            |(int_idx, field_idx, coeff)| InteractionMonomialTerm {
                interaction_idx: int_idx,
                field_idx,
                coefficient: coeff,
            },
        );

        Self {
            numer_headers,
            numer_variables,
            numer_terms,
            denom_headers,
            denom_variables,
            denom_terms,
            num_interactions: interactions.len() as u32,
            max_fields_len,
        }
    }
}

/// Expand a `SymbolicExpression` into a list of monomials.
fn expand_symbolic_expression<F: Field>(expr: &SymbolicExpression<F>) -> Vec<ExpandedMonomial<F>> {
    match expr {
        SymbolicExpression::Constant(c) => expand_leaf(vec![], *c),
        SymbolicExpression::Variable(v) => {
            expand_leaf(vec![PackedVar::from_symbolic_var(v)], F::ONE)
        }
        SymbolicExpression::IsFirstRow => expand_leaf(vec![PackedVar::is_first()], F::ONE),
        SymbolicExpression::IsLastRow => expand_leaf(vec![PackedVar::is_last()], F::ONE),
        SymbolicExpression::IsTransition => expand_leaf(vec![PackedVar::is_transition()], F::ONE),
        SymbolicExpression::Add { x, y, .. } => {
            let mut result = expand_symbolic_expression(x);
            result.extend(expand_symbolic_expression(y));
            combine_like_terms(result)
        }
        SymbolicExpression::Sub { x, y, .. } => {
            let mut result = expand_symbolic_expression(x);
            for mut mono in expand_symbolic_expression(y) {
                mono.coefficient = -mono.coefficient;
                result.push(mono);
            }
            combine_like_terms(result)
        }
        SymbolicExpression::Mul { x, y, .. } => {
            let left_monomials = expand_symbolic_expression(x);
            let right_monomials = expand_symbolic_expression(y);
            let mut result = Vec::with_capacity(left_monomials.len() * right_monomials.len());
            for l in &left_monomials {
                for r in &right_monomials {
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
        SymbolicExpression::Neg { x, .. } => expand_symbolic_expression(x)
            .into_iter()
            .map(|mut mono| {
                mono.coefficient = -mono.coefficient;
                mono
            })
            .collect(),
    }
}
