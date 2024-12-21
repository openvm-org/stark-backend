use std::sync::Arc;

use p3_field::{AbstractField, Field};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::symbolic_expression::SymbolicEvaluator;
use crate::air_builders::symbolic::{
    symbolic_expression::SymbolicExpression, symbolic_variable::SymbolicVariable,
};

/// A node in symbolic expression DAG.
/// Basically replace `Arc`s in `SymbolicExpression` with node IDs.
/// Intended to be serializable and deserializable.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound = "F: Field")]
#[repr(C)]
pub enum SymbolicExpressionNode<F> {
    Variable(SymbolicVariable<F>),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Constant(F),
    Add {
        x: usize,
        y: usize,
        degree_multiple: usize,
    },
    Sub {
        x: usize,
        y: usize,
        degree_multiple: usize,
    },
    Neg {
        x: usize,
        degree_multiple: usize,
    },
    Mul {
        x: usize,
        y: usize,
        degree_multiple: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound = "F: Field")]
pub struct SymbolicExpressionDag<F> {
    /// Nodes in **topological** order.
    pub(crate) nodes: Vec<SymbolicExpressionNode<F>>,
    /// Node indices of expressions to assert equal zero.
    pub(crate) constraint_idx: Vec<usize>,
}

pub(crate) fn build_symbolic_expr_dag<F: Field>(
    exprs: &[SymbolicExpression<F>],
) -> SymbolicExpressionDag<F> {
    let mut expr_to_idx = FxHashMap::default();
    let mut nodes = Vec::new();
    let constraint_idx = exprs
        .iter()
        .map(|expr| topological_sort_symbolic_expr(expr, &mut expr_to_idx, &mut nodes))
        .collect();
    SymbolicExpressionDag {
        nodes,
        constraint_idx,
    }
}

/// `expr_to_idx` is a cache so that the `Arc<_>` references within symbolic expressions get
/// mapped to the same node ID if their underlying references are the same.
fn topological_sort_symbolic_expr<'a, F: Field>(
    expr: &'a SymbolicExpression<F>,
    expr_to_idx: &mut FxHashMap<&'a SymbolicExpression<F>, usize>,
    nodes: &mut Vec<SymbolicExpressionNode<F>>,
) -> usize {
    if let Some(&idx) = expr_to_idx.get(expr) {
        return idx;
    }
    let node = match expr {
        SymbolicExpression::Variable(var) => SymbolicExpressionNode::Variable(*var),
        SymbolicExpression::IsFirstRow => SymbolicExpressionNode::IsFirstRow,
        SymbolicExpression::IsLastRow => SymbolicExpressionNode::IsLastRow,
        SymbolicExpression::IsTransition => SymbolicExpressionNode::IsTransition,
        SymbolicExpression::Constant(cons) => SymbolicExpressionNode::Constant(*cons),
        SymbolicExpression::Add {
            x,
            y,
            degree_multiple,
        } => {
            let x_id = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            let y_id = topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Add {
                x: x_id,
                y: y_id,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Sub {
            x,
            y,
            degree_multiple,
        } => {
            let x_id = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            let y_id = topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Sub {
                x: x_id,
                y: y_id,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Neg { x, degree_multiple } => {
            let x_id = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Neg {
                x: x_id,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Mul {
            x,
            y,
            degree_multiple,
        } => {
            // An important case to remember: square will have Arc::as_ptr(&x) == Arc::as_ptr(&y)
            // The `expr_to_id` will ensure only one topological sort is done to prevent exponential
            // behavior.
            let x_id = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            let y_id = topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Mul {
                x: x_id,
                y: y_id,
                degree_multiple: *degree_multiple,
            }
        }
    };

    let idx = nodes.len();
    nodes.push(node);
    expr_to_idx.insert(expr, idx);
    idx
}

impl<F: Field> SymbolicExpressionDag<F> {
    /// Returns symbolic expressions for each constraint
    pub fn to_symbolic_expressions(&self) -> Vec<SymbolicExpression<F>> {
        let mut exprs: Vec<Arc<SymbolicExpression<_>>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let expr = match node {
                SymbolicExpressionNode::Variable(var) => SymbolicExpression::Variable(*var),
                SymbolicExpressionNode::IsFirstRow => SymbolicExpression::IsFirstRow,
                SymbolicExpressionNode::IsLastRow => SymbolicExpression::IsLastRow,
                SymbolicExpressionNode::IsTransition => SymbolicExpression::IsTransition,
                SymbolicExpressionNode::Constant(f) => SymbolicExpression::Constant(*f),
                SymbolicExpressionNode::Add {
                    x,
                    y,
                    degree_multiple,
                } => SymbolicExpression::Add {
                    x: exprs[*x].clone(),
                    y: exprs[*y].clone(),
                    degree_multiple: *degree_multiple,
                },
                SymbolicExpressionNode::Sub {
                    x,
                    y,
                    degree_multiple,
                } => SymbolicExpression::Sub {
                    x: exprs[*x].clone(),
                    y: exprs[*y].clone(),
                    degree_multiple: *degree_multiple,
                },
                SymbolicExpressionNode::Neg { x, degree_multiple } => SymbolicExpression::Neg {
                    x: exprs[*x].clone(),
                    degree_multiple: *degree_multiple,
                },
                SymbolicExpressionNode::Mul {
                    x,
                    y,
                    degree_multiple,
                } => SymbolicExpression::Mul {
                    x: exprs[*x].clone(),
                    y: exprs[*y].clone(),
                    degree_multiple: *degree_multiple,
                },
            };
            exprs.push(Arc::new(expr));
        }
        self.constraint_idx
            .iter()
            .map(|&idx| exprs[idx].as_ref().clone())
            .collect()
    }

    /// Evaluate each constraint expression.
    pub fn evaluate_constraints<E, SE>(&self, evaluator: &SE) -> Vec<E>
    where
        E: AbstractField + From<F>,
        SE: SymbolicEvaluator<F, E>,
    {
        // node_idx -> evaluation
        // We do a simple serial evaluation in topological order.
        // This can be parallelized if necessary.
        let mut exprs: Vec<E> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let expr = match *node {
                SymbolicExpressionNode::Variable(var) => evaluator.eval_var(var),
                SymbolicExpressionNode::Constant(f) => E::from(f),
                SymbolicExpressionNode::Add { x, y, .. } => exprs[x].clone() + exprs[y].clone(),
                SymbolicExpressionNode::Sub { x, y, .. } => exprs[x].clone() - exprs[y].clone(),
                SymbolicExpressionNode::Neg { x, .. } => -exprs[x].clone(),
                SymbolicExpressionNode::Mul { x, y, .. } => exprs[x].clone() * exprs[y].clone(),
                _ => unreachable!("unevaluatable expression: {node:?}"),
            };
            exprs.push(expr);
        }
        self.constraint_idx
            .iter()
            .map(|&idx| exprs[idx].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    use crate::air_builders::symbolic::{
        dag::{build_symbolic_expr_dag, SymbolicExpressionDag, SymbolicExpressionNode},
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints,
    };

    type F = BabyBear;

    #[test]
    fn test_symbolic_expressions_dag() {
        let expr = SymbolicExpression::Constant(F::ONE)
            * SymbolicVariable::new(
                Entry::Main {
                    part_index: 1,
                    offset: 2,
                },
                3,
            );
        let exprs = vec![
            SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow
                + SymbolicExpression::Constant(F::ONE)
                + SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow
                + expr.clone(),
            expr.clone() * expr.clone(),
        ];
        let expr_list = build_symbolic_expr_dag(&exprs);
        assert_eq!(
            expr_list,
            SymbolicExpressionDag::<F> {
                nodes: vec![
                    SymbolicExpressionNode::IsFirstRow,
                    SymbolicExpressionNode::IsLastRow,
                    SymbolicExpressionNode::Mul {
                        x: 0,
                        y: 1,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Constant(F::ONE),
                    SymbolicExpressionNode::Add {
                        x: 2,
                        y: 3,
                        degree_multiple: 2
                    },
                    // Currently topological sort does not detect all subgraph isomorphisms. For example each IsFirstRow and IsLastRow is a new reference so ptr::hash is distinct.
                    SymbolicExpressionNode::Mul {
                        x: 0,
                        y: 1,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Add {
                        x: 4,
                        y: 5,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Variable(SymbolicVariable::new(
                        Entry::Main {
                            part_index: 1,
                            offset: 2
                        },
                        3
                    )),
                    SymbolicExpressionNode::Mul {
                        x: 3,
                        y: 7,
                        degree_multiple: 1
                    },
                    SymbolicExpressionNode::Add {
                        x: 6,
                        y: 8,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Mul {
                        x: 8,
                        y: 8,
                        degree_multiple: 2
                    }
                ],
                constraint_idx: vec![9, 10],
            }
        );
        let sc = SymbolicConstraints {
            constraints: exprs,
            interactions: vec![],
        };
        let ser_str = serde_json::to_string(&sc).unwrap();
        let new_sc: SymbolicConstraints<_> = serde_json::from_str(&ser_str).unwrap();
        assert_eq!(sc.constraints, new_sc.constraints);
    }
}
