use std::sync::Arc;

use p3_field::Field;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::air_builders::symbolic::{
    symbolic_expression::SymbolicExpression, symbolic_variable::SymbolicVariable,
};

/// A node in symbolic expression DAG. Basically replace `Arc`s in `SymbolicExpression` with node IDs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(bound = "F: Field")]
enum SymbolicExpressionNode<F> {
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
    expr_by_id: Vec<SymbolicExpressionNode<F>>,
    /// Node ids of expressions to assert.
    top_expr_ids: Vec<usize>,
}

pub(super) fn build_symbolic_expr_dag<F: Field>(
    exprs: &[SymbolicExpression<F>],
) -> SymbolicExpressionDag<F> {
    let mut expr_to_id = FxHashMap::default();
    let mut id_to_expr = Vec::new();
    let top_expr_ids = exprs
        .iter()
        .map(|expr| topology_sort_symbolic_expr(expr, &mut expr_to_id, &mut id_to_expr))
        .collect();
    SymbolicExpressionDag {
        expr_by_id: id_to_expr,
        top_expr_ids,
    }
}

fn topology_sort_symbolic_expr<'a, F: Field>(
    expr: &'a SymbolicExpression<F>,
    expr_to_id: &mut FxHashMap<&'a SymbolicExpression<F>, usize>,
    id_to_expr: &mut Vec<SymbolicExpressionNode<F>>,
) -> usize {
    if let Some(&id) = expr_to_id.get(&expr) {
        return id;
    }
    let exp_ser = match expr {
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
            let x_id = topology_sort_symbolic_expr(x.as_ref(), expr_to_id, id_to_expr);
            let y_id = topology_sort_symbolic_expr(y.as_ref(), expr_to_id, id_to_expr);
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
            let x_id = topology_sort_symbolic_expr(x.as_ref(), expr_to_id, id_to_expr);
            let y_id = topology_sort_symbolic_expr(y.as_ref(), expr_to_id, id_to_expr);
            SymbolicExpressionNode::Sub {
                x: x_id,
                y: y_id,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpression::Neg { x, degree_multiple } => {
            let x_id = topology_sort_symbolic_expr(x.as_ref(), expr_to_id, id_to_expr);
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
            let x_id = topology_sort_symbolic_expr(x.as_ref(), expr_to_id, id_to_expr);
            let y_id = topology_sort_symbolic_expr(y.as_ref(), expr_to_id, id_to_expr);
            SymbolicExpressionNode::Mul {
                x: x_id,
                y: y_id,
                degree_multiple: *degree_multiple,
            }
        }
    };

    let id = id_to_expr.len();
    id_to_expr.push(exp_ser);
    expr_to_id.insert(expr, id);
    id
}

impl<F: Field> SymbolicExpressionDag<F> {
    pub fn to_symbolic_expressions(&self) -> Vec<SymbolicExpression<F>> {
        let mut exprs: Vec<Arc<SymbolicExpression<_>>> = Vec::with_capacity(self.expr_by_id.len());
        for expr_ser in &self.expr_by_id {
            let expr = match expr_ser {
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
        self.top_expr_ids
            .iter()
            .map(|&id| exprs[id].as_ref().clone())
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
        let exprs = vec![
            SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow
                + SymbolicExpression::Constant(F::ONE)
                + SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow
                + SymbolicExpression::Constant(F::ONE)
                    * SymbolicVariable::new(
                        Entry::Main {
                            part_index: 1,
                            offset: 2,
                        },
                        3,
                    ),
            SymbolicExpression::IsFirstRow * SymbolicExpression::IsLastRow,
        ];
        let expr_list = build_symbolic_expr_dag(&exprs);
        assert_eq!(
            expr_list,
            SymbolicExpressionDag::<F> {
                expr_by_id: vec![
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
                    SymbolicExpressionNode::Add {
                        x: 4,
                        y: 2,
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
                        y: 6,
                        degree_multiple: 1
                    },
                    SymbolicExpressionNode::Add {
                        x: 5,
                        y: 7,
                        degree_multiple: 2
                    }
                ],
                top_expr_ids: vec![8, 2],
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
