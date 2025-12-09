use std::sync::Arc;

use p3_field::Field;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::SymbolicConstraints;
use crate::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression, symbolic_variable::SymbolicVariable,
        SymbolicRapBuilder,
    },
    interaction::{Interaction, InteractionChunks, SymbolicInteraction},
};

/// A node in symbolic expression DAG.
/// Basically replace `Arc`s in `SymbolicExpression` with node IDs.
/// Intended to be serializable and deserializable.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Logup fractions as pairs of symbolic expression node indices for `(numerator,
    /// denominator)`.
    pub logup_frac_nodes: Vec<(usize, usize)>,
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
    /// A subset of the nodes represents all plain AIR constraints. Other nodes are used to express
    /// interactions as LogUp fractions.
    pub constraints: SymbolicExpressionDag<F>,
    /// Partition of `interactions` into chunks. The chunks are specified via indices into
    /// `interactions`.
    ///
    /// This data is implicitly already in the logup fractions within `constraints`, but we store
    /// it separately for convenience.
    pub interaction_chunks: InteractionChunks,
    /// List of all interactions, where expressions in the interactions
    /// are referenced by node idx as `usize`.
    ///
    /// This data is implicitly already in the logup fractions within `constraints`, but we store
    /// it separately for convenience.
    pub interactions: Vec<Interaction<usize>>,
}

impl<F: Field> SymbolicConstraintsDag<F> {
    pub fn from_expressions(
        constraints: &[SymbolicExpression<F>],
        interactions: &[SymbolicInteraction<F>],
        interaction_chunks: InteractionChunks,
    ) -> Self {
        let mut expr_to_idx = FxHashMap::default();
        let mut nodes = Vec::new();
        let mut constraint_idx: Vec<usize> = constraints
            .iter()
            .map(|expr| topological_sort_symbolic_expr(expr, &mut expr_to_idx, &mut nodes))
            .collect();
        // It is more efficient for DAG evaluation if the constraints are in the topological order.
        constraint_idx.sort();

        let max_msg_len = interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .max()
            .unwrap_or(0);
        let beta_pows = SymbolicRapBuilder::new_challenges(&[max_msg_len + 1])
            .pop()
            .unwrap();
        let logup_chunk_fracs =
            interaction_chunks.symbolic_logup_fractions(interactions, &beta_pows);
        // We add the fraction expression directly first to optimize the ordering of the DAG.
        // However there is no guarantee that the fraction nodes are in any sorted order.
        let logup_frac_nodes = logup_chunk_fracs
            .iter()
            .map(|(numer, denom)| {
                let [numer_node, denom_node] = [numer, denom]
                    .map(|expr| topological_sort_symbolic_expr(expr, &mut expr_to_idx, &mut nodes));
                (numer_node, denom_node)
            })
            .collect::<Vec<_>>();

        let interactions: Vec<Interaction<usize>> = interactions
            .iter()
            .map(|interaction| {
                // We are using the exact SymbolicExpression that was used above in the definition
                // of the fraction chunk, so the definition of Eq on
                // SymbolicExpression guarantees this will return the node contained
                // within the fraction's expression.
                let message: Vec<usize> = interaction
                    .message
                    .iter()
                    .map(|expr| expr_to_idx[expr])
                    .collect();
                let count = expr_to_idx[&interaction.count];
                Interaction {
                    message,
                    count,
                    bus_index: interaction.bus_index,
                    count_weight: interaction.count_weight,
                }
            })
            .collect();
        let constraints = SymbolicExpressionDag {
            nodes,
            constraint_idx,
            logup_frac_nodes,
        };
        SymbolicConstraintsDag {
            constraints,
            interaction_chunks,
            interactions,
        }
    }
}

/// `expr_to_idx` is a cache so that the `Arc<_>` references within symbolic expressions get
/// mapped to the same node ID if their underlying references are the same.
pub fn topological_sort_symbolic_expr<'a, F: Field>(
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
            let left_idx = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            let right_idx = topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Sub {
            x,
            y,
            degree_multiple,
        } => {
            let left_idx = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            let right_idx = topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Neg { x, degree_multiple } => {
            let idx = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Neg {
                idx,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Mul {
            x,
            y,
            degree_multiple,
        } => {
            // An important case to remember: square will have Arc::as_ptr(&x) ==
            // Arc::as_ptr(&y) The `expr_to_id` will ensure only one topological
            // sort is done to prevent exponential behavior.
            let left_idx = topological_sort_symbolic_expr(x.as_ref(), expr_to_idx, nodes);
            let right_idx = topological_sort_symbolic_expr(y.as_ref(), expr_to_idx, nodes);
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
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
    /// Convert each node to a [`SymbolicExpression<F>`] reference and return
    /// the full list.
    pub fn to_symbolic_expressions(&self) -> Vec<Arc<SymbolicExpression<F>>> {
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;

    use crate::{
        air_builders::symbolic::{
            dag::{SymbolicExpressionDag, SymbolicExpressionNode},
            symbolic_expression::SymbolicExpression,
            symbolic_variable::{Entry, SymbolicVariable},
            SymbolicConstraintsDag,
        },
        interaction::{find_interaction_chunks, Interaction},
    };

    type F = BabyBear;

    #[test]
    fn test_symbolic_constraints_dag() {
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
        let interaction_chunks = find_interaction_chunks(&interactions, 0);
        let dag = SymbolicConstraintsDag::from_expressions(
            &constraints,
            &interactions,
            interaction_chunks,
        );
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
                    // Currently topological sort does not detect all subgraph isomorphisms. For
                    // example each IsFirstRow and IsLastRow is a new reference so ptr::hash is
                    // distinct.
                    SymbolicExpressionNode::Mul {
                        left_idx: 0,
                        right_idx: 1,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Add {
                        left_idx: 4,
                        right_idx: 5,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Variable(SymbolicVariable::new(
                        Entry::Main {
                            part_index: 1,
                            offset: 2
                        },
                        3
                    )),
                    // expr:
                    SymbolicExpressionNode::Mul {
                        left_idx: 3,
                        right_idx: 7,
                        degree_multiple: 1
                    },
                    SymbolicExpressionNode::Add {
                        left_idx: 6,
                        right_idx: 8,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Mul {
                        left_idx: 8,
                        right_idx: 8,
                        degree_multiple: 2
                    },
                    SymbolicExpressionNode::Variable(SymbolicVariable::new(Entry::Challenge, 0)), /* alpha */
                    SymbolicExpressionNode::Variable(SymbolicVariable::new(Entry::Challenge, 2)), /* beta^{msg_len} */
                    SymbolicExpressionNode::Mul {
                        // beta^msg_len * (bus_index + 1)
                        left_idx: 12,
                        right_idx: 3, // (bus_index + 1) is 1
                        degree_multiple: 0
                    },
                    SymbolicExpressionNode::Add {
                        /* alpha + beta^{msg_len} * (bus_index + 1) */
                        left_idx: 11,
                        right_idx: 13,
                        degree_multiple: 0
                    },
                    SymbolicExpressionNode::Add {
                        // alpha + beta^{msg_len} * (bus_index + 1) + msg[0]
                        left_idx: 14,
                        right_idx: 8,
                        degree_multiple: 1
                    },
                    SymbolicExpressionNode::Constant(F::TWO), // msg[1]
                    SymbolicExpressionNode::Variable(SymbolicVariable::new(Entry::Challenge, 1)),
                    SymbolicExpressionNode::Mul {
                        // beta^1 * msg[1]
                        left_idx: 16,
                        right_idx: 17,
                        degree_multiple: 0
                    },
                    SymbolicExpressionNode::Add {
                        // logup_denom = alpha + beta^2 * (bus_index + 1) + msg[0] + beta^1 * msg[1]
                        left_idx: 15,
                        right_idx: 18,
                        degree_multiple: 1
                    }
                ],
                constraint_idx: vec![9, 10],
                logup_frac_nodes: vec![(3, 19)]
            }
        );
        assert_eq!(
            dag.interactions,
            vec![Interaction {
                bus_index: 0,
                message: vec![8, 16],
                count: 3,
                count_weight: 1,
            }]
        );
    }
}
