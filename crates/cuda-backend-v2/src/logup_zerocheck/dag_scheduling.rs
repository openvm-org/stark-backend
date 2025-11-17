use std::{cmp::Ordering, collections::BinaryHeap};

use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    SymbolicConstraintsDag, SymbolicExpressionNode,
};
use p3_field::{Field, PrimeField32};

#[derive(Clone, Copy)]
pub struct ScheduleExpressionInfo {
    pub dag_idx: usize,
    pub buffer_idx: usize,
    pub first_use: usize,
    pub last_use: usize,
    pub use_count: usize,
    pub accumulate: bool,
    pub intermediate: bool,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ScheduleBufferEntry {
    buffer_idx: usize,
    last_use: usize,
}

impl Ord for ScheduleBufferEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .last_use
            .cmp(&self.last_use)
            .then_with(|| other.buffer_idx.cmp(&self.buffer_idx))
    }
}

impl PartialOrd for ScheduleBufferEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn compute_constraint_expr_indices<F: Field + PrimeField32>(
    dag: &SymbolicConstraintsDag<F>,
) -> Vec<usize> {
    let mut expr_info = (0..dag.constraints.nodes.len())
        .map(|i| ScheduleExpressionInfo {
            dag_idx: i,
            buffer_idx: usize::MAX,
            first_use: usize::MAX,
            last_use: usize::MAX,
            use_count: 0,
            accumulate: false,
            intermediate: false,
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
                expr_info[i].intermediate = true;
                expr_info[*left_idx].first_use = expr_info[*left_idx].first_use.min(i);
                expr_info[*left_idx].last_use = i;
                expr_info[*left_idx].use_count += 1;
                expr_info[*right_idx].first_use = expr_info[*right_idx].first_use.min(i);
                expr_info[*right_idx].last_use = i;
                expr_info[*right_idx].use_count += 1;
            }
            SymbolicExpressionNode::Neg { idx, .. } => {
                expr_info[i].buffer_idx = i;
                expr_info[i].intermediate = true;
                expr_info[*idx].first_use = expr_info[*idx].first_use.min(i);
                expr_info[*idx].last_use = i;
                expr_info[*idx].use_count += 1;
            }
            _ => {}
        }
    }

    let used_nodes = &dag.constraints.constraint_idx;

    let mut max_prev_node = 0;
    for &idx in used_nodes {
        max_prev_node = max_prev_node.max(idx);
        expr_info[idx].accumulate = true;
        expr_info[idx].last_use = if expr_info[idx].last_use == usize::MAX {
            max_prev_node
        } else {
            expr_info[idx].last_use.max(max_prev_node)
        };
    }

    for expr in expr_info.iter_mut() {
        if expr.use_count == 0 {
            expr.buffer_idx = usize::MAX;
        }
    }

    let buffer_expr_info = expr_info
        .iter()
        .filter(|info| info.buffer_idx != usize::MAX)
        .copied()
        .sorted_by_key(|info| info.dag_idx)
        .collect::<Vec<_>>();

    let mut buffer = BinaryHeap::<ScheduleBufferEntry>::new();
    for expr in buffer_expr_info {
        if buffer.is_empty() || (buffer.peek().unwrap().last_use > expr.dag_idx) {
            expr_info[expr.dag_idx].buffer_idx = buffer.len();
            buffer.push(ScheduleBufferEntry {
                buffer_idx: expr_info[expr.dag_idx].buffer_idx,
                last_use: expr.last_use,
            });
        } else {
            let buffer_entry = buffer.pop().unwrap();
            expr_info[expr.dag_idx].buffer_idx = buffer_entry.buffer_idx;
            buffer.push(ScheduleBufferEntry {
                buffer_idx: expr_info[expr.dag_idx].buffer_idx,
                last_use: expr.last_use,
            });
        }
    }

    expr_info
        .iter()
        .filter(|info| info.accumulate || info.intermediate)
        .map(|info| info.dag_idx)
        .collect()
}
