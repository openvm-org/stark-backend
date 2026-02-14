extern crate alloc;

use alloc::vec::Vec;
use std::{cmp::Ordering, collections::BinaryHeap};

use getset::Getters;
use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    symbolic_variable::SymbolicVariable, SymbolicExpressionDag, SymbolicExpressionNode,
};
use p3_field::{Field, PrimeField32};
use rustc_hash::FxHashMap;

pub mod codec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Source<F> {
    Intermediate(usize),
    TerminalIntermediate,
    Var(SymbolicVariable<F>),
    IsFirst,
    IsLast,
    IsTransition,
    Constant(F),
}

/// Rule representing a single operation for the GPU kernel to interpret.
#[derive(Clone, Debug, PartialEq)]
pub enum Rule<F> {
    // three-address code
    // (x, y, z) => z = x op y
    Add(Source<F>, Source<F>, Source<F>),
    Sub(Source<F>, Source<F>, Source<F>),
    Mul(Source<F>, Source<F>, Source<F>),
    // (x, z) => z = -x
    Neg(Source<F>, Source<F>),
    Variable(Source<F>),
    // (x, z) => z = x
    BufferVar(Source<F>, Source<F>),
}

#[derive(derive_new::new, Clone, Debug, PartialEq)]
pub struct RuleWithFlag<F> {
    /// Raw rule
    pub inner: Rule<F>,
    /// Flag for whether the rule represents a constraint node that should be included in the
    /// accumulator.
    pub need_accumulate: bool,
}

#[derive(Debug, Copy, Clone)]
struct StaticExpressionInfo {
    // smallest dag_idx of a descendant
    pub first_descendant: usize,
    pub accumulate: bool,
    pub intermediate: bool,
    pub buffer_var: bool,
}

#[derive(Debug, Copy, Clone)]
struct LiveExpressionInfo {
    pub use_count: usize,
    /// Last dag_idx (inclusive) that uses this expr
    pub last_use: usize,
    /// Index within intermediate buffer to store, usize::MAX if not stored
    pub buffer_idx: usize,
}

impl LiveExpressionInfo {
    pub fn update(&mut self, dag_idx: usize) {
        self.last_use = self.last_use.max(dag_idx);
        self.use_count += 1;
    }
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
pub struct SymbolicRulesGpu<F> {
    /// Vector consisting of rules to be sequentially evaluated in a single GPU thread.
    pub rules: Vec<RuleWithFlag<F>>,
    /// Number of `F` elements in the intermediate buffer required for rule evaluation.
    pub buffer_size: usize,
    /// Map from dag_idx to rule_idx for accumulated nodes. Unlike `used_nodes`, this has no
    /// duplicates.
    pub dag_idx_to_rule_idx: FxHashMap<usize, usize>,
}

/// Stateful rule builder that schedules buffer allocation and reuse using a priority queue.
///
/// At any given time, the builder will determine the buffer scheduling for a contiguous range of
/// nodes in a global DAG, where the DAG is already topologically sorted.
#[derive(Getters)]
pub struct SymbolicRulesBuilder<'a, F> {
    /// Global dag that all symbolic expressions will be referenced with respect to.
    dag_nodes: &'a [SymbolicExpressionNode<F>],

    // Contains every node in the DAG.
    // dag_idx => StaticExpressionInfo
    static_expr_info: Vec<StaticExpressionInfo>,

    // Stateful info about node with respect to buffer usage. Changes if nodes are added or removed
    // from what the intermediate buffer needs to account for. dag_idx => LiveExpressionInfo
    live_expr_info: FxHashMap<usize, LiveExpressionInfo>,
    buffer: BinaryHeap<BufferEntry>,

    // Effective range after set_range (inclusive start, exclusive end)
    effective_start: usize,
    effective_end: usize,
}

impl<'a, F: Field> SymbolicRulesBuilder<'a, F> {
    /// The `dag.constraint_idx` must be topologically sorted.
    pub fn new(dag: &'a SymbolicExpressionDag<F>, buffer_vars: bool) -> Self {
        debug_assert!(dag.constraint_idx.iter().is_sorted());
        let mut expr_info = vec![
            StaticExpressionInfo {
                first_descendant: usize::MAX,
                accumulate: false,
                intermediate: false,
                buffer_var: false,
            };
            dag.nodes.len()
        ];
        for &idx in &dag.constraint_idx {
            expr_info[idx].accumulate = true;
        }

        for (i, node) in dag.nodes.iter().enumerate() {
            expr_info[i].first_descendant = expr_info[i].first_descendant.min(i);
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
                    expr_info[i].intermediate = true;
                    expr_info[i].first_descendant = expr_info[i]
                        .first_descendant
                        .min(expr_info[*left_idx].first_descendant)
                        .min(expr_info[*right_idx].first_descendant);
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    expr_info[i].intermediate = true;
                    expr_info[i].first_descendant = expr_info[i]
                        .first_descendant
                        .min(expr_info[*idx].first_descendant);
                }
                SymbolicExpressionNode::Variable(_) => {
                    if buffer_vars {
                        expr_info[i].buffer_var = true;
                    }
                }
                _ => {}
            }
        }
        Self {
            dag_nodes: &dag.nodes,
            static_expr_info: expr_info,
            live_expr_info: FxHashMap::default(),
            buffer: BinaryHeap::new(),
            effective_start: 0,
            effective_end: 0,
        }
    }

    /// Schedules buffer usage for nodes in `[start_dag_idx, end_dag_idx)` (non-inclusive end). Any
    /// previous range is cleared. Any nodes that are needed prior to the start of the range are
    /// also scheduled.
    pub fn set_range(&mut self, mut start_dag_idx: usize, end_dag_idx: usize) {
        debug_assert!(start_dag_idx < end_dag_idx);
        debug_assert!(end_dag_idx <= self.dag_nodes.len());
        self.live_expr_info.clear();
        let _start = start_dag_idx;
        for i in _start..end_dag_idx {
            start_dag_idx = start_dag_idx.min(self.static_expr_info[i].first_descendant);
        }
        // Store effective range for to_rules()
        self.effective_start = start_dag_idx;
        self.effective_end = end_dag_idx;

        // We add to live_expr_info if the node is used (even if not buffered)
        // Add in reverse order so last used is first
        for (dag_idx, (node, info)) in self
            .dag_nodes
            .iter()
            .zip(self.static_expr_info.iter())
            .enumerate()
            .take(end_dag_idx)
            .skip(start_dag_idx)
            .rev()
        {
            if !self.live_expr_info.contains_key(&dag_idx) && !info.accumulate {
                // Not a descendant and not accumulated, so we can skip
                continue;
            }
            let unit_info = LiveExpressionInfo {
                use_count: 1,
                last_use: dag_idx,
                buffer_idx: usize::MAX,
            };
            self.live_expr_info
                .entry(dag_idx)
                .and_modify(|e| {
                    e.use_count += 1;
                })
                .or_insert(unit_info);

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
                    self.live_expr_info
                        .entry(*left_idx)
                        .and_modify(|e| e.update(dag_idx))
                        .or_insert(unit_info);
                    self.live_expr_info
                        .entry(*right_idx)
                        .and_modify(|e| e.update(dag_idx))
                        .or_insert(unit_info);
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    self.live_expr_info
                        .entry(*idx)
                        .and_modify(|e| e.update(dag_idx))
                        .or_insert(unit_info);
                }
                _ => {}
            }
        }

        // Collects all the expressions that need to be buffered. We then use a classic
        // priority-queue scheduling algorithm to minimally assign buffer indices to
        // each expression. Iterate by index to maintain topological order.
        let buffer_dag_idxs = (start_dag_idx..end_dag_idx)
            .filter(|&dag_idx| {
                if let Some(live_info) = self.live_expr_info.get(&dag_idx) {
                    debug_assert!(live_info.use_count > 0);
                    let static_info = &self.static_expr_info[dag_idx];
                    static_info.intermediate || (static_info.buffer_var && live_info.use_count > 1)
                } else {
                    false
                }
            })
            .collect_vec();

        let buffer = &mut self.buffer;
        buffer.clear();

        for dag_idx in buffer_dag_idxs {
            if buffer.is_empty() || buffer.peek().unwrap().last_use > dag_idx {
                let buffer_idx = buffer.len();
                let info = self.live_expr_info.get_mut(&dag_idx).unwrap();
                let last_use = info.last_use;
                info.buffer_idx = buffer_idx;
                buffer.push(BufferEntry {
                    buffer_idx,
                    last_use,
                });
            } else {
                let buffer_entry = buffer.pop().unwrap();
                let buffer_idx = buffer_entry.buffer_idx;
                let info = self.live_expr_info.get_mut(&dag_idx).unwrap();
                let last_use = info.last_use;
                info.buffer_idx = buffer_idx;
                buffer.push(BufferEntry {
                    buffer_idx,
                    last_use,
                });
            }
        }
    }

    pub fn to_rules(&self) -> SymbolicRulesGpu<F> {
        let dag_idx_to_source = |idx: usize, use_idx: usize| {
            match &self.dag_nodes[idx] {
                // Leaf nodes don't need buffer_idx lookup
                SymbolicExpressionNode::IsFirstRow => Source::IsFirst,
                SymbolicExpressionNode::IsLastRow => Source::IsLast,
                SymbolicExpressionNode::IsTransition => Source::IsTransition,
                SymbolicExpressionNode::Constant(c) => Source::Constant(*c),
                SymbolicExpressionNode::Variable(var) => {
                    if self.static_expr_info[idx].buffer_var {
                        // Only use buffer if it was actually assigned (use_count > 1)
                        let buffer_idx = self.live_expr_info[&idx].buffer_idx;
                        if buffer_idx != usize::MAX && idx != use_idx {
                            Source::Intermediate(buffer_idx)
                        } else {
                            Source::Var(*var)
                        }
                    } else {
                        Source::Var(*var)
                    }
                }
                SymbolicExpressionNode::Add { .. }
                | SymbolicExpressionNode::Sub { .. }
                | SymbolicExpressionNode::Mul { .. }
                | SymbolicExpressionNode::Neg { .. } => {
                    // Only access live_expr_info when we need buffer_idx
                    let buffer_idx = self
                        .live_expr_info
                        .get(&idx)
                        .map(|info| info.buffer_idx)
                        .unwrap_or(usize::MAX);
                    if buffer_idx == usize::MAX {
                        Source::TerminalIntermediate
                    } else {
                        Source::Intermediate(buffer_idx)
                    }
                }
            }
        };

        // Build dag_idx -> rule_idx map (no duplicates, uses earliest rule_idx)
        // Iterate by index to maintain topological order (FxHashMap has no ordering)
        let mut dag_idx_to_rule_idx = FxHashMap::default();
        let rules = (self.effective_start..self.effective_end)
            .filter(|dag_idx| self.live_expr_info.contains_key(dag_idx))
            .enumerate()
            .map(|(rule_idx, dag_idx)| {
                let current_node = &self.dag_nodes[dag_idx];
                let constraint = match current_node {
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx,
                        ..
                    } => Rule::Add(
                        dag_idx_to_source(*left_idx, dag_idx),
                        dag_idx_to_source(*right_idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx,
                        ..
                    } => Rule::Sub(
                        dag_idx_to_source(*left_idx, dag_idx),
                        dag_idx_to_source(*right_idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Mul {
                        left_idx,
                        right_idx,
                        ..
                    } => Rule::Mul(
                        dag_idx_to_source(*left_idx, dag_idx),
                        dag_idx_to_source(*right_idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Neg { idx, .. } => Rule::Neg(
                        dag_idx_to_source(*idx, dag_idx),
                        dag_idx_to_source(dag_idx, dag_idx),
                    ),
                    SymbolicExpressionNode::Variable(_) => {
                        let var = dag_idx_to_source(dag_idx, dag_idx);
                        let buffer_idx = self.live_expr_info[&dag_idx].buffer_idx;
                        // Only use BufferVar if variable actually needs buffering
                        // (buffer_var=true AND use_count > 1, which means buffer_idx was set)
                        if self.static_expr_info[dag_idx].buffer_var && buffer_idx != usize::MAX {
                            Rule::BufferVar(var, Source::Intermediate(buffer_idx))
                        } else {
                            Rule::Variable(var)
                        }
                    }
                    _ => Rule::Variable(dag_idx_to_source(dag_idx, dag_idx)),
                };
                let accumulate = self.static_expr_info[dag_idx].accumulate;
                if accumulate {
                    // Use or_insert to keep the earliest rule_idx
                    dag_idx_to_rule_idx.entry(dag_idx).or_insert(rule_idx);
                }
                RuleWithFlag::new(constraint, accumulate)
            })
            .collect::<Vec<_>>();

        SymbolicRulesGpu {
            rules,
            buffer_size: self.buffer.len(),
            dag_idx_to_rule_idx,
        }
    }
}

impl<F: Field + PrimeField32> SymbolicRulesGpu<F> {
    pub fn new(dag: &SymbolicExpressionDag<F>, buffer_vars: bool) -> Self {
        let mut builder = SymbolicRulesBuilder::new(dag, buffer_vars);
        builder.set_range(0, dag.nodes.len());
        builder.to_rules()
    }
}
