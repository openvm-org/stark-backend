extern crate alloc;

use alloc::vec::Vec;
use std::{cmp::Ordering, collections::BinaryHeap};

use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    symbolic_variable::{Entry, SymbolicVariable},
    SymbolicConstraintsDag, SymbolicExpressionNode,
};
use p3_field::{Field, PrimeField32};
use rustc_hash::FxHashMap;
use tracing::instrument;

pub(crate) mod codec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Source<F: Field> {
    Intermediate(usize),
    TerminalIntermediate,
    Var(SymbolicVariable<F>),
    BufferedVar((SymbolicVariable<F>, usize)),
    IsFirst,
    IsLast,
    IsTransition,
    Constant(F),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Constraint<F: Field> {
    // three-address code
    // (x, y, z) => z = x op y
    Add(Source<F>, Source<F>, Source<F>),
    Sub(Source<F>, Source<F>, Source<F>),
    Mul(Source<F>, Source<F>, Source<F>),
    // (x, z) => z = -x
    Neg(Source<F>, Source<F>),
    Variable(Source<F>),
}

#[derive(derive_new::new, Clone, Debug, PartialEq)]
pub struct ConstraintWithFlag<F: Field> {
    pub constraint: Constraint<F>,
    pub need_accumulate: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ExpressionType {
    Intermediate,
    Variable,
    Other,
}

#[derive(Debug, Copy, Clone)]
struct ExpressionInfo {
    pub dag_idx: usize,
    pub buffer_idx: usize,
    pub first_use: usize,
    pub last_use: usize,
    pub use_count: usize,
    pub accumulate: bool,
    pub expr_type: ExpressionType,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct BufferEntry {
    buffer_idx: usize,
    last_use: usize,
}

impl Ord for BufferEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .last_use
            .cmp(&self.last_use)
            .then_with(|| other.buffer_idx.cmp(&self.buffer_idx))
    }
}

impl PartialOrd for BufferEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct SymbolicRulesOnGpu<F: Field> {
    pub constraints: Vec<ConstraintWithFlag<F>>,
    pub used_nodes: Vec<usize>,
    pub buffer_size: usize,
}

impl<F: Field + PrimeField32> SymbolicRulesOnGpu<F> {
    #[instrument(name = "SymbolicRulesOnGpu.new", skip_all, level = "debug")]
    pub fn new(dag: SymbolicConstraintsDag<F>, cache_vars: bool, is_permute: bool) -> Self {
        let mut expr_info = (0..dag.constraints.nodes.len())
            .map(|i| ExpressionInfo {
                dag_idx: i,
                buffer_idx: usize::MAX,
                first_use: usize::MAX,
                last_use: usize::MAX,
                use_count: 0,
                accumulate: false,
                expr_type: ExpressionType::Other,
            })
            .collect::<Vec<_>>();

        for (i, node) in dag.constraints.nodes.iter().enumerate() {
            match node {
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                }
                | SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                }
                | SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    expr_info[i].buffer_idx = i;
                    expr_info[i].expr_type = ExpressionType::Intermediate;
                    expr_info[*left_idx].first_use = expr_info[*left_idx].first_use.min(i);
                    expr_info[*left_idx].last_use = i;
                    expr_info[*left_idx].use_count += 1;
                    expr_info[*right_idx].first_use = expr_info[*right_idx].first_use.min(i);
                    expr_info[*right_idx].last_use = i;
                    expr_info[*right_idx].use_count += 1;
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    expr_info[i].buffer_idx = i;
                    expr_info[i].expr_type = ExpressionType::Intermediate;
                    expr_info[*idx].first_use = expr_info[*idx].first_use.min(i);
                    expr_info[*idx].last_use = i;
                    expr_info[*idx].use_count += 1;
                }
                SymbolicExpressionNode::Variable(var) => {
                    expr_info[i].expr_type = ExpressionType::Variable;
                    if cache_vars {
                        match var.entry {
                            Entry::Main { .. }
                            | Entry::Permutation { .. }
                            | Entry::Preprocessed { .. } => {
                                expr_info[i].buffer_idx = i;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // This should be a list of constraint indices, but we initialize it to be a list of DAG
        // node indices for now. We'll remap each entry later on.
        let used_nodes = if dag.constraints.constraint_idx.is_empty() {
            // This branch is used only for encoding SymbolicInteractions for use during perm
            // trace generation. The `message` is always a single expression for the denominator.
            dag.interactions
                .iter()
                .flat_map(|i| {
                    assert_eq!(i.message.len(), 1);
                    [i.count, *i.message.first().unwrap()]
                })
                .collect::<Vec<_>>()
        } else {
            dag.constraints.constraint_idx.clone()
        };

        let mut max_prev_node = 0;
        for &idx in &used_nodes {
            max_prev_node = max_prev_node.max(idx);
            expr_info[idx].accumulate = true;
            expr_info[idx].last_use = if expr_info[idx].last_use == usize::MAX {
                max_prev_node
            } else {
                expr_info[idx].last_use.max(max_prev_node)
            };
            // Variables should only be read from the LDE once, and hence should be stored in the
            // buffer after their first read. If a variable is accumulated, during quotient eval
            // it will be read here first (due to the topological ordering). Alternatively, during
            // permutation some accumulated intermediates may need to be stored in the buffer for
            // access later - such values must be marked as used.
            if (expr_info[idx].expr_type == ExpressionType::Variable && cache_vars) || is_permute {
                expr_info[idx].first_use = expr_info[idx].first_use.min(idx);
                expr_info[idx].use_count += 1;
            }
        }

        // Expressions don't actually need to be buffered if they're not read/used multiple time.
        // We can use this to reduce the number of buffers needed.
        for expr in expr_info.iter_mut() {
            if expr.use_count == 0
                || (expr.use_count == 1 && expr.expr_type == ExpressionType::Variable)
            {
                expr.buffer_idx = usize::MAX;
            }
        }

        // Collects all the expressions that need to be buffered and sort them by last use
        // last use. We then use the classic scheduling algorithm to minimally assign buffer
        // indices to each expression.
        let buffer_expr_info = expr_info
            .iter()
            .filter(|info| info.buffer_idx != usize::MAX)
            .copied()
            .sorted_by_key(|info| {
                if info.expr_type == ExpressionType::Variable {
                    info.first_use
                } else {
                    info.dag_idx
                }
            })
            .collect::<Vec<_>>();
        let mut buffer = BinaryHeap::<BufferEntry>::new();

        for expr in buffer_expr_info {
            // Variables will be stored to the buffer immediately after their first read, while
            // intermediates will be stored after being computed for the first time.
            let store_point = if expr.expr_type == ExpressionType::Variable {
                expr.first_use
            } else {
                expr.dag_idx
            };

            if buffer.is_empty() || (buffer.peek().unwrap().last_use >= store_point) {
                expr_info[expr.dag_idx].buffer_idx = buffer.len();
                buffer.push(BufferEntry {
                    buffer_idx: expr_info[expr.dag_idx].buffer_idx,
                    last_use: expr.last_use,
                });
            } else {
                let buffer_entry = buffer.pop().unwrap();
                expr_info[expr.dag_idx].buffer_idx = buffer_entry.buffer_idx;
                buffer.push(BufferEntry {
                    buffer_idx: expr_info[expr.dag_idx].buffer_idx,
                    last_use: expr.last_use,
                });
            }
        }

        // Builds the list of constraints that will be encoded into rules that our CUDA kernels can
        // interpret. We need to add only a) intermediate expressions and b) expressions that will
        // be directly accumulated into the quotient value.
        let constraint_expr_idxs = expr_info
            .iter()
            .filter(|info| info.accumulate || info.expr_type == ExpressionType::Intermediate)
            .map(|info| info.dag_idx)
            .collect::<Vec<_>>();

        let mut dag_idx_to_constraint_idx = FxHashMap::default();
        let dag_idx_to_source = |idx: usize, use_idx: usize| {
            let buffer_idx = expr_info[idx].buffer_idx;
            match &dag.constraints.nodes[idx] {
                SymbolicExpressionNode::Variable(var) => {
                    if buffer_idx == usize::MAX {
                        Source::Var(*var)
                    } else if expr_info[idx].first_use == use_idx && cache_vars {
                        Source::BufferedVar((*var, buffer_idx))
                    } else {
                        Source::Intermediate(buffer_idx)
                    }
                }
                SymbolicExpressionNode::IsFirstRow => Source::IsFirst,
                SymbolicExpressionNode::IsLastRow => Source::IsLast,
                SymbolicExpressionNode::IsTransition => Source::IsTransition,
                SymbolicExpressionNode::Constant(c) => Source::Constant(*c),
                SymbolicExpressionNode::Add { .. }
                | SymbolicExpressionNode::Sub { .. }
                | SymbolicExpressionNode::Mul { .. }
                | SymbolicExpressionNode::Neg { .. } => {
                    if buffer_idx == usize::MAX {
                        Source::TerminalIntermediate
                    } else {
                        Source::Intermediate(buffer_idx)
                    }
                }
            }
        };

        let constraints = constraint_expr_idxs
            .iter()
            .enumerate()
            .map(|(constraint_idx, &dag_idx)| {
                if expr_info[dag_idx].accumulate {
                    dag_idx_to_constraint_idx.insert(dag_idx, constraint_idx);
                }
                let current_node = &dag.constraints.nodes[dag_idx];
                let constraint = match current_node {
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx,
                        ..
                    } => Constraint::Add(
                        dag_idx_to_source(*left_idx, dag_idx),
                        dag_idx_to_source(*right_idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx,
                        ..
                    } => Constraint::Sub(
                        dag_idx_to_source(*left_idx, dag_idx),
                        dag_idx_to_source(*right_idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Mul {
                        left_idx,
                        right_idx,
                        ..
                    } => Constraint::Mul(
                        dag_idx_to_source(*left_idx, dag_idx),
                        dag_idx_to_source(*right_idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Neg { idx, .. } => Constraint::Neg(
                        dag_idx_to_source(*idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    // This will always be the first variable read due to topological ordering -
                    // thus it should always be buffered here.
                    _ => Constraint::Variable(dag_idx_to_source(dag_idx, dag_idx)),
                };
                ConstraintWithFlag::new(constraint, expr_info[dag_idx].accumulate)
            })
            .collect::<Vec<_>>();

        SymbolicRulesOnGpu {
            constraints,
            used_nodes: used_nodes
                .iter()
                .map(|idx| dag_idx_to_constraint_idx[idx])
                .collect(),
            buffer_size: buffer.len(),
        }
    }
}
