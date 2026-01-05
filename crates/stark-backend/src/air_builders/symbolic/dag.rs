use std::sync::Arc;

use p3_field::Field;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::SymbolicConstraints;
use crate::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, symbolic_variable::SymbolicVariable,
    },
    interaction::{Interaction, SymbolicInteraction},
};

/// A node in symbolic expression DAG.
/// Basically replace `Arc`s in `SymbolicExpression` with node IDs.
/// Intended to be serializable and deserializable.
#[derive(Clone, Debug, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
#[repr(C)]
pub enum SymbolicExpressionNode<F> {
    Variable(SymbolicVariable<F>),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Constant(F),
    Add {
        left_idx: usize,
        right_idx: usize,
        degree_multiple: usize,
    },
    Sub {
        left_idx: usize,
        right_idx: usize,
        degree_multiple: usize,
    },
    Neg {
        idx: usize,
        degree_multiple: usize,
    },
    Mul {
        left_idx: usize,
        right_idx: usize,
        degree_multiple: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
#[repr(C)]
pub struct SymbolicExpressionDag<F> {
    /// Nodes in **topological** order.
    pub nodes: Vec<SymbolicExpressionNode<F>>,
    /// Node indices of expressions to assert equal zero.
    pub constraint_idx: Vec<usize>,
}

impl<F> SymbolicExpressionDag<F> {
    pub fn max_rotation(&self) -> usize {
        let mut rotation = 0;
        for node in &self.nodes {
            if let SymbolicExpressionNode::Variable(var) = node {
                rotation = rotation.max(var.entry.offset().unwrap_or(0));
            }
        }
        rotation
    }

    pub fn num_constraints(&self) -> usize {
        self.constraint_idx.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
#[repr(C)]
pub struct SymbolicConstraintsDag<F> {
    /// DAG with all symbolic expressions as nodes.
    /// A subset of the nodes represents all constraints that will be
    /// included in the quotient polynomial via DEEP-ALI.
    pub constraints: SymbolicExpressionDag<F>,
    /// List of all interactions, where expressions in the interactions
    /// are referenced by node idx as `usize`.
    ///
    /// This is used by the prover for after challenge trace generation,
    /// and some partial information may be used by the verifier.
    ///
    /// **However**, any contributions to the quotient polynomial from
    /// logup are already included in `constraints` and do not need to
    /// be separately calculated from `interactions`.
    pub interactions: Vec<Interaction<usize>>,
}

pub(crate) fn build_symbolic_constraints_dag<F: Field>(
    constraints: &[SymbolicExpression<F>],
    interactions: &[SymbolicInteraction<F>],
) -> SymbolicConstraintsDag<F> {
    let mut expr_to_idx = FxHashMap::default();
    let mut node_to_idx = FxHashMap::default();
    let mut nodes = Vec::new();
    let mut constraint_idx: Vec<usize> = constraints
        .iter()
        .map(|expr| topological_sort_symbolic_expr(expr, &mut expr_to_idx, &mut node_to_idx, &mut nodes))
        .collect();
    constraint_idx.sort();
    constraint_idx.dedup();
    let interactions: Vec<Interaction<usize>> = interactions
        .iter()
        .map(|interaction| {
            let fields: Vec<usize> = interaction
                .message
                .iter()
                .map(|field_expr| {
                    topological_sort_symbolic_expr(field_expr, &mut expr_to_idx, &mut node_to_idx, &mut nodes)
                })
                .collect();
            let count =
                topological_sort_symbolic_expr(&interaction.count, &mut expr_to_idx, &mut node_to_idx, &mut nodes);
            Interaction {
                message: fields,
                count,
                bus_index: interaction.bus_index,
                count_weight: interaction.count_weight,
            }
        })
        .collect();
    let constraints = SymbolicExpressionDag {
        nodes,
        constraint_idx,
    };
    SymbolicConstraintsDag {
        constraints,
        interactions,
    }
}

/// Converts a symbolic expression tree into a DAG representation with structural deduplication
/// and algebraic simplifications.
///
/// Two caches are used:
/// - `expr_to_idx`: Fast path for expressions with the same Arc pointer
/// - `node_to_idx`: Structural deduplication - catches identical nodes with different Arc pointers
///
/// Algebraic simplifications performed:
/// - `x + 0` → `x`, `0 + x` → `x`
/// - `x - 0` → `x`
/// - `x * 1` → `x`, `1 * x` → `x`
/// - `x * 0` → `0`, `0 * x` → `0`
/// - `-0` → `0`
/// - `x + (-y)` → `x - y` (normalizes Add+Neg to Sub)
/// - `x - (-y)` → `x + y` (double negation)
pub fn topological_sort_symbolic_expr<'a, F: Field>(
    expr: &'a SymbolicExpression<F>,
    expr_to_idx: &mut FxHashMap<&'a SymbolicExpression<F>, usize>,
    node_to_idx: &mut FxHashMap<SymbolicExpressionNode<F>, usize>,
    nodes: &mut Vec<SymbolicExpressionNode<F>>,
) -> usize {
    if let Some(&idx) = expr_to_idx.get(expr) {
        return idx;
    }

    // Helper to check if a node at given index is a constant with specific value
    let is_const = |nodes: &[SymbolicExpressionNode<F>], idx: usize, val: F| -> bool {
        matches!(&nodes[idx], SymbolicExpressionNode::Constant(c) if *c == val)
    };

    // Helper to check if a node is a Neg and return its child index
    let get_neg_child = |nodes: &[SymbolicExpressionNode<F>], idx: usize| -> Option<usize> {
        match &nodes[idx] {
            SymbolicExpressionNode::Neg { idx, .. } => Some(*idx),
            _ => None,
        }
    };

    let idx = match expr {
        SymbolicExpression::Variable(var) => {
            intern_node(SymbolicExpressionNode::Variable(*var), node_to_idx, nodes)
        }
        SymbolicExpression::IsFirstRow => {
            intern_node(SymbolicExpressionNode::IsFirstRow, node_to_idx, nodes)
        }
        SymbolicExpression::IsLastRow => {
            intern_node(SymbolicExpressionNode::IsLastRow, node_to_idx, nodes)
        }
        SymbolicExpression::IsTransition => {
            intern_node(SymbolicExpressionNode::IsTransition, node_to_idx, nodes)
        }
        SymbolicExpression::Constant(cons) => {
            intern_node(SymbolicExpressionNode::Constant(*cons), node_to_idx, nodes)
        }
        SymbolicExpression::Add {
            x,
            y,
            degree_multiple,
        } => {
            let left_idx =
                topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, node_to_idx, nodes);
            let right_idx =
                topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, node_to_idx, nodes);

            // Simplify: 0 + x = x, x + 0 = x
            if is_const(nodes, left_idx, F::ZERO) {
                right_idx
            } else if is_const(nodes, right_idx, F::ZERO) {
                left_idx
            }
            // Normalize: x + (-y) = x - y
            else if let Some(neg_child_idx) = get_neg_child(nodes, right_idx) {
                intern_node(
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx: neg_child_idx,
                        degree_multiple: *degree_multiple,
                    },
                    node_to_idx,
                    nodes,
                )
            } else {
                intern_node(
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx,
                        degree_multiple: *degree_multiple,
                    },
                    node_to_idx,
                    nodes,
                )
            }
        }
        SymbolicExpression::Sub {
            x,
            y,
            degree_multiple,
        } => {
            let left_idx =
                topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, node_to_idx, nodes);
            let right_idx =
                topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, node_to_idx, nodes);

            // Simplify: x - 0 = x
            if is_const(nodes, right_idx, F::ZERO) {
                left_idx
            }
            // Simplify: x - (-y) = x + y (double negation)
            else if let Some(neg_child_idx) = get_neg_child(nodes, right_idx) {
                intern_node(
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx: neg_child_idx,
                        degree_multiple: *degree_multiple,
                    },
                    node_to_idx,
                    nodes,
                )
            } else {
                intern_node(
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx,
                        degree_multiple: *degree_multiple,
                    },
                    node_to_idx,
                    nodes,
                )
            }
        }
        SymbolicExpression::Neg { x, degree_multiple } => {
            let child_idx =
                topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, node_to_idx, nodes);

            // Simplify: -0 = 0
            if is_const(nodes, child_idx, F::ZERO) {
                child_idx
            } else {
                intern_node(
                    SymbolicExpressionNode::Neg {
                        idx: child_idx,
                        degree_multiple: *degree_multiple,
                    },
                    node_to_idx,
                    nodes,
                )
            }
        }
        SymbolicExpression::Mul {
            x,
            y,
            degree_multiple,
        } => {
            // An important case to remember: square will have Arc::as_ptr(&x) == Arc::as_ptr(&y)
            // The `expr_to_idx` will ensure only one topological sort is done to prevent exponential
            // behavior.
            let left_idx =
                topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, node_to_idx, nodes);
            let right_idx =
                topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, node_to_idx, nodes);

            // Simplify: 0 * x = 0, x * 0 = 0
            if is_const(nodes, left_idx, F::ZERO) {
                left_idx
            } else if is_const(nodes, right_idx, F::ZERO) {
                right_idx
            }
            // Simplify: 1 * x = x, x * 1 = x
            else if is_const(nodes, left_idx, F::ONE) {
                right_idx
            } else if is_const(nodes, right_idx, F::ONE) {
                left_idx
            } else {
                intern_node(
                    SymbolicExpressionNode::Mul {
                        left_idx,
                        right_idx,
                        degree_multiple: *degree_multiple,
                    },
                    node_to_idx,
                    nodes,
                )
            }
        }
    };

    expr_to_idx.insert(expr, idx);
    idx
}

/// Intern a node: return existing index if the node already exists, otherwise add it.
fn intern_node<F: Field>(
    node: SymbolicExpressionNode<F>,
    node_to_idx: &mut FxHashMap<SymbolicExpressionNode<F>, usize>,
    nodes: &mut Vec<SymbolicExpressionNode<F>>,
) -> usize {
    *node_to_idx.entry(node.clone()).or_insert_with(|| {
        let idx = nodes.len();
        nodes.push(node);
        idx
    })
}

impl<F: Field> SymbolicExpressionDag<F> {
    /// Convert each node to a [`SymbolicExpression<F>`] reference and return
    /// the full list.
    fn to_symbolic_expressions(&self) -> Vec<Arc<SymbolicExpression<F>>> {
        let mut exprs: Vec<Arc<SymbolicExpression<_>>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let expr = match *node {
                SymbolicExpressionNode::Variable(var) => SymbolicExpression::Variable(var),
                SymbolicExpressionNode::IsFirstRow => SymbolicExpression::IsFirstRow,
                SymbolicExpressionNode::IsLastRow => SymbolicExpression::IsLastRow,
                SymbolicExpressionNode::IsTransition => SymbolicExpression::IsTransition,
                SymbolicExpressionNode::Constant(f) => SymbolicExpression::Constant(f),
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    degree_multiple,
                } => SymbolicExpression::Add {
                    x: exprs[left_idx].clone(),
                    y: exprs[right_idx].clone(),
                    degree_multiple,
                },
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    degree_multiple,
                } => SymbolicExpression::Sub {
                    x: exprs[left_idx].clone(),
                    y: exprs[right_idx].clone(),
                    degree_multiple,
                },
                SymbolicExpressionNode::Neg {
                    idx,
                    degree_multiple,
                } => SymbolicExpression::Neg {
                    x: exprs[idx].clone(),
                    degree_multiple,
                },
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    degree_multiple,
                } => SymbolicExpression::Mul {
                    x: exprs[left_idx].clone(),
                    y: exprs[right_idx].clone(),
                    degree_multiple,
                },
            };
            exprs.push(Arc::new(expr));
        }
        exprs
    }
}

// TEMPORARY conversions until we switch main interfaces to use SymbolicConstraintsDag
impl<'a, F: Field> From<&'a SymbolicConstraintsDag<F>> for SymbolicConstraints<F> {
    fn from(dag: &'a SymbolicConstraintsDag<F>) -> Self {
        let exprs = dag.constraints.to_symbolic_expressions();
        let constraints = dag
            .constraints
            .constraint_idx
            .iter()
            .map(|&idx| exprs[idx].as_ref().clone())
            .collect::<Vec<_>>();
        let interactions = dag
            .interactions
            .iter()
            .map(|interaction| {
                let fields = interaction
                    .message
                    .iter()
                    .map(|&idx| exprs[idx].as_ref().clone())
                    .collect();
                let count = exprs[interaction.count].as_ref().clone();
                Interaction {
                    message: fields,
                    count,
                    bus_index: interaction.bus_index,
                    count_weight: interaction.count_weight,
                }
            })
            .collect::<Vec<_>>();
        SymbolicConstraints {
            constraints,
            interactions,
        }
    }
}

impl<F: Field> From<SymbolicConstraintsDag<F>> for SymbolicConstraints<F> {
    fn from(dag: SymbolicConstraintsDag<F>) -> Self {
        (&dag).into()
    }
}

impl<F: Field> From<SymbolicConstraints<F>> for SymbolicConstraintsDag<F> {
    fn from(sc: SymbolicConstraints<F>) -> Self {
        build_symbolic_constraints_dag(&sc.constraints, &sc.interactions)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;

    use crate::{
        air_builders::symbolic::{
            dag::{build_symbolic_constraints_dag, SymbolicExpressionDag, SymbolicExpressionNode},
            symbolic_expression::SymbolicExpression,
            symbolic_variable::{Entry, SymbolicVariable},
        },
        interaction::Interaction,
    };

    type F = BabyBear;

    #[test]
    fn test_duplicate_constraints_are_deduplicated() {
        // Create a simple expression
        let expr: SymbolicExpression<F> = SymbolicExpression::Variable(SymbolicVariable::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            0,
        ));

        // Simulate calling assert_zero twice on the same expression
        let constraints = vec![expr.clone(), expr.clone()];
        let interactions = vec![];

        let dag = build_symbolic_constraints_dag(&constraints, &interactions);

        // Nodes are deduplicated - there's only 1 node in the DAG
        assert_eq!(dag.constraints.nodes.len(), 1, "Nodes should be deduplicated");

        // constraint_idx should also be deduplicated
        assert_eq!(
            dag.constraints.constraint_idx,
            vec![0],
            "constraint_idx should be deduplicated"
        );

        // Only 1 constraint
        assert_eq!(
            dag.constraints.num_constraints(),
            1,
            "Duplicate constraints should be deduplicated"
        );
    }

    #[test]
    fn test_structural_deduplication() {
        // Create two structurally identical expressions with different Arc pointers
        // This simulates: builder.assert_zero(x - ONE); builder.assert_zero(x - ONE);
        let var = SymbolicVariable::<F>::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            0,
        );
        let expr1 = SymbolicExpression::from(var) - SymbolicExpression::Constant(F::ONE);
        let expr2 = SymbolicExpression::from(var) - SymbolicExpression::Constant(F::ONE);

        // These are different Arc allocations
        assert!(!std::ptr::eq(&expr1, &expr2));

        let constraints = vec![expr1, expr2];
        let dag = build_symbolic_constraints_dag(&constraints, &[]);

        // With structural deduplication, both expressions should map to the same node
        // Nodes: Variable(0), Constant(1), Sub(0,1)
        assert_eq!(dag.constraints.nodes.len(), 3);
        assert_eq!(dag.constraints.constraint_idx, vec![2]);
    }

    #[test]
    fn test_algebraic_simplifications() {
        let var = SymbolicVariable::<F>::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            0,
        );
        let x = SymbolicExpression::from(var);
        let zero = SymbolicExpression::Constant(F::ZERO);
        let one = SymbolicExpression::Constant(F::ONE);

        // Test x + 0 = x
        let expr_add_zero = x.clone() + zero.clone();
        let dag = build_symbolic_constraints_dag(&[expr_add_zero], &[]);
        // Should only have Variable node, no Add node
        assert_eq!(dag.constraints.nodes.len(), 2); // Variable + Constant(0) interned but Add simplified away
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Variable(_)
        ));

        // Test 0 + x = x
        let expr_zero_add = zero.clone() + x.clone();
        let dag = build_symbolic_constraints_dag(&[expr_zero_add], &[]);
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Variable(_)
        ));

        // Test x * 1 = x
        let expr_mul_one = x.clone() * one.clone();
        let dag = build_symbolic_constraints_dag(&[expr_mul_one], &[]);
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Variable(_)
        ));

        // Test 1 * x = x
        let expr_one_mul = one.clone() * x.clone();
        let dag = build_symbolic_constraints_dag(&[expr_one_mul], &[]);
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Variable(_)
        ));

        // Test x * 0 = 0
        let expr_mul_zero = x.clone() * zero.clone();
        let dag = build_symbolic_constraints_dag(&[expr_mul_zero], &[]);
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Constant(c) if c == F::ZERO
        ));

        // Test x - 0 = x
        let expr_sub_zero = x.clone() - zero.clone();
        let dag = build_symbolic_constraints_dag(&[expr_sub_zero], &[]);
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Variable(_)
        ));

        // Test x + (-y) normalizes to x - y (same as Sub)
        let y = SymbolicExpression::from(SymbolicVariable::<F>::new(
            Entry::Main {
                part_index: 0,
                offset: 0,
            },
            1,
        ));
        let expr_add_neg = x.clone() + (-y.clone());
        let expr_sub = x.clone() - y.clone();
        let dag1 = build_symbolic_constraints_dag(&[expr_add_neg], &[]);
        let dag2 = build_symbolic_constraints_dag(&[expr_sub], &[]);
        // Both should produce the same constraint node (Sub)
        assert!(matches!(
            dag1.constraints.nodes[dag1.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Sub { .. }
        ));
        assert_eq!(
            dag1.constraints.nodes[dag1.constraints.constraint_idx[0]],
            dag2.constraints.nodes[dag2.constraints.constraint_idx[0]]
        );

        // Test x - (-y) = x + y
        let expr_sub_neg = x.clone() - (-y.clone());
        let dag = build_symbolic_constraints_dag(&[expr_sub_neg], &[]);
        assert!(matches!(
            dag.constraints.nodes[dag.constraints.constraint_idx[0]],
            SymbolicExpressionNode::Add { .. }
        ));
    }

    #[test]
    fn test_symbolic_constraints_dag() {
        // expr = Constant(1) * Variable, which simplifies to just Variable
        let expr = SymbolicExpression::Constant(F::ONE)
            * SymbolicVariable::new(
                Entry::Main {
                    part_index: 1,
                    offset: 2,
                },
                3,
            );
        let constraints = vec![
            SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow
                + SymbolicExpression::Constant(F::ONE)
                + SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow
                + expr.clone(),
            expr.clone() * expr.clone(),
        ];
        let interactions = vec![Interaction {
            bus_index: 0,
            message: vec![expr.clone(), SymbolicExpression::Constant(F::TWO)],
            count: SymbolicExpression::Constant(F::ONE),
            count_weight: 1,
        }];
        let dag = build_symbolic_constraints_dag(&constraints, &interactions);
        assert_eq!(
            dag.constraints,
            SymbolicExpressionDag::<F> {
                nodes: vec![
                    SymbolicExpressionNode::IsFirstRow,
                    SymbolicExpressionNode::IsLastRow,
                    SymbolicExpressionNode::Mul {
                        left_idx: 0,
                        right_idx: 1,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Constant(F::ONE),
                    SymbolicExpressionNode::Add {
                        left_idx: 2,
                        right_idx: 3,
                        degree_multiple: 2
                    },
                    // With structural deduplication, IsFirstRow * IsLastRow is now reused
                    // instead of duplicated. The second occurrence reuses node index 2.
                    SymbolicExpressionNode::Add {
                        left_idx: 4,
                        right_idx: 2,
                        degree_multiple: 2
                    },
                    // expr = Constant(1) * Variable simplifies to just Variable (1 * x = x)
                    SymbolicExpressionNode::Variable(SymbolicVariable::new(
                        Entry::Main {
                            part_index: 1,
                            offset: 2
                        },
                        3
                    )),
                    // First constraint: ... + expr (which is Variable at index 6)
                    SymbolicExpressionNode::Add {
                        left_idx: 5,
                        right_idx: 6,
                        degree_multiple: 2
                    },
                    // Second constraint: expr * expr = Variable * Variable
                    SymbolicExpressionNode::Mul {
                        left_idx: 6,
                        right_idx: 6,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Constant(F::TWO),
                ],
                constraint_idx: vec![7, 8],
            }
        );
        assert_eq!(
            dag.interactions,
            vec![Interaction {
                bus_index: 0,
                // expr simplified to Variable at index 6, Constant(2) at index 9
                message: vec![6, 9],
                count: 3,
                count_weight: 1,
            }]
        );
    }
}
