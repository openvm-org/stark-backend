#![allow(dead_code)]

extern crate alloc;

use alloc::vec::Vec;
use std::collections::{BTreeMap, HashMap, HashSet};

use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    symbolic_variable::SymbolicVariable, SymbolicConstraintsDag, SymbolicExpressionNode,
};
use p3_field::{Field, PrimeField32};
use rustc_hash::FxHashMap;
use tracing::instrument;

use crate::types::F;

pub(crate) mod codec;

// TODO[stephenh]: If the new rule generation is truly an optimization, we should
// make it the default.
pub mod stephen;

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

impl<F: Field> From<SymbolicExpressionNode<F>> for Source<F> {
    fn from(value: SymbolicExpressionNode<F>) -> Self {
        match value {
            SymbolicExpressionNode::Variable(var) => Source::Var(var),
            SymbolicExpressionNode::IsFirstRow => Source::IsFirst,
            SymbolicExpressionNode::IsLastRow => Source::IsLast,
            SymbolicExpressionNode::IsTransition => Source::IsTransition,
            SymbolicExpressionNode::Constant(c) => Source::Constant(c),
            _ => panic!("Invalid conversion for non-intermediate SymbolicExpressionNode to Source"),
        }
    }
}

fn is_intermediate<F: Field>(node: &SymbolicExpressionNode<F>) -> bool {
    !matches!(
        node,
        SymbolicExpressionNode::Variable(_)
            | SymbolicExpressionNode::IsFirstRow
            | SymbolicExpressionNode::IsLastRow
            | SymbolicExpressionNode::IsTransition
            | SymbolicExpressionNode::Constant(_)
    )
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

#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintWithFlag<F: Field> {
    pub constraint: Constraint<F>,
    pub need_accumulate: bool,
}

impl<F: Field + PrimeField32> ConstraintWithFlag<F> {
    pub fn new(constraint: Constraint<F>, need_accumulate: bool) -> Self {
        Self {
            constraint,
            need_accumulate,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SymbolicRulesOnGpu<F: Field> {
    pub constraints: Vec<ConstraintWithFlag<F>>,
    pub used_nodes: Vec<usize>,
    pub num_intermediates: usize,
}

impl<F: Field + PrimeField32> SymbolicRulesOnGpu<F> {
    /// transpile constraints in dag to a form that can be evaluated by the gpu prover
    /// especially we minimize the vector for storing intermediates to fit in gpu shared memory
    #[instrument(name = "SymbolicRulesOnGpu.new", skip_all, level = "debug")]
    pub fn new(dag_constraints: SymbolicConstraintsDag<F>) -> Self {
        let dag = dag_constraints.constraints;

        let mut used_nodes = if dag.constraint_idx.is_empty() {
            // NOTE: this branch is used only for encoding SymbolicInteractions for use during perm
            // trace generation. The `message` is always a single expression for the
            // denominator.
            dag_constraints
                .interactions
                .iter()
                .flat_map(|i| {
                    assert_eq!(i.message.len(), 1);
                    [i.count, *i.message.first().unwrap()]
                })
                .collect_vec()
        } else {
            dag.constraint_idx.clone()
        };

        // create a mapping from non-intermediate node to their index in dag
        let mut vars_index_map = FxHashMap::default();
        for (idx, node) in dag.nodes.iter().enumerate() {
            if !is_intermediate(node) {
                vars_index_map.insert(idx, node.clone());
            }
        }

        // compute number of intermediates up to current index
        let num_intermediates_array = dag
            .nodes
            .iter()
            .scan(0, |acc, node| {
                let num = *acc;
                *acc += if is_intermediate(node) { 1 } else { 0 };

                Some(num)
            })
            .collect_vec();

        // if node is an intermediate, return its index in the array of all intermediates
        // else return the node itself as a variable
        let get_source = |idx| {
            if vars_index_map.contains_key(&idx) {
                vars_index_map.get(&idx).unwrap().clone().into()
            } else {
                Source::Intermediate(num_intermediates_array[idx])
            }
        };
        let mut compiled_constraints = Vec::with_capacity(dag.nodes.len());
        let constraint_idx_set: HashSet<usize> = HashSet::from_iter(used_nodes.iter().copied());

        for (node_idx, node) in dag.nodes.into_iter().enumerate() {
            let assert_zero = constraint_idx_set.contains(&node_idx);

            let constraint = if !is_intermediate(&node) {
                // We add Variable constraint only if we accumulate it
                if assert_zero {
                    Constraint::Variable(node.into())
                } else {
                    let compiled_constraint_len = compiled_constraints.len();
                    for node in used_nodes.iter_mut() {
                        if *node >= compiled_constraint_len {
                            *node -= 1;
                        }
                    }
                    continue;
                }
            } else {
                match node {
                    SymbolicExpressionNode::Add {
                        left_idx,
                        right_idx,
                        ..
                    } => Constraint::Add(
                        get_source(left_idx),
                        get_source(right_idx),
                        get_source(node_idx),
                    ),
                    SymbolicExpressionNode::Sub {
                        left_idx,
                        right_idx,
                        ..
                    } => Constraint::Sub(
                        get_source(left_idx),
                        get_source(right_idx),
                        get_source(node_idx),
                    ),
                    SymbolicExpressionNode::Neg { idx, .. } => {
                        Constraint::Neg(get_source(idx), get_source(node_idx))
                    }
                    SymbolicExpressionNode::Mul {
                        left_idx,
                        right_idx,
                        ..
                    } => Constraint::Mul(
                        get_source(left_idx),
                        get_source(right_idx),
                        get_source(node_idx),
                    ),
                    _ => unreachable!(),
                }
            };

            compiled_constraints.push(ConstraintWithFlag::new(constraint, assert_zero));
        }

        // have same set of assert-zero constraints as dag
        assert_eq!(
            compiled_constraints
                .iter()
                .enumerate()
                .filter(|(_, c)| c.need_accumulate)
                .map(|(idx, _)| idx)
                .collect::<Vec<usize>>(),
            used_nodes.iter().copied().unique().sorted().collect_vec()
        );

        let num_intermediates_before_reduction = num_intermediates_array
            .last()
            .copied()
            .map(|n| n + 1)
            .unwrap_or(0);

        let mut analyzer = IntermediateAllocator::new(compiled_constraints);
        analyzer.reallocate(used_nodes.clone());

        tracing::debug!(
            "[IntermediateAllocator] size of minimal vector to store intermediates: before = {}, after = {}",
            num_intermediates_before_reduction,
            analyzer.num_intermediates,

        );

        SymbolicRulesOnGpu {
            constraints: analyzer.constraints,
            used_nodes,
            num_intermediates: analyzer.num_intermediates,
        }
    }
}

/// Information about an intermediate variable's usage and dependencies
///
/// Tracks the lifecycle and relationships of intermediate variables:
/// * `def_point` - Instruction index where variable is first defined
/// * `first_use` - Instruction index of first usage
/// * `last_use` - Instruction index of last usage
/// * `use_count` - Total number of times the variable is used
#[derive(Debug)]
struct VariableInfo {
    def_point: usize,
    first_use: usize,
    last_use: usize,
    use_count: usize,
}

/// An allocator that optimizes memory usage by reallocating intermediate variables.
/// It allows multiple variables with non-overlapping lifetimes to share the same index,
/// thus reducing the total size of the intermediates array.
pub struct IntermediateAllocator<F: Field> {
    /// List of arithmetic constraints and their accumulation flags.
    /// Each tuple contains:
    /// * `Constraint<F>` - The arithmetic constraint
    /// * `bool` - Whether this constraint should be accumulated (true) or not (false)
    constraints: Vec<ConstraintWithFlag<F>>,

    /// Stores usage information for each intermediate variable, including:
    /// * Definition point - Where the variable is first defined
    /// * First/Last use points - Lifecycle boundaries
    /// * Usage count - Number of times the variable is used
    variable_info: HashMap<usize, VariableInfo>,

    /// Maps original variable IDs to their optimized/reallocated IDs.
    /// Used during the reallocation process to track how variables are reassigned
    /// to minimize the total number of required intermediate slots.
    variable_mapping: HashMap<usize, usize>,

    /// The minimum number of intermediate slots needed after optimization.
    /// This represents the final length required for the intermediates array.
    num_intermediates: usize,
}

impl<F: Field + PrimeField32> IntermediateAllocator<F> {
    pub fn new(constraints: Vec<ConstraintWithFlag<F>>) -> Self {
        IntermediateAllocator {
            constraints,
            variable_info: HashMap::new(),
            variable_mapping: HashMap::new(),
            num_intermediates: 0,
        }
    }

    /// This process is similar to register allocation in compiler optimization,
    /// where we minimize required storage space by analyzing variable lifetimes.
    ///
    /// The optimization works through three main steps:
    /// 1. Collecting usage information for all variables
    /// 2. Compressing variable IDs by reusing available slots
    /// 3. Applying the optimized ID mapping to all constraints
    ///
    /// # Memory Optimization Strategy
    /// * Tracks variable definition points and last usage
    /// * Reuses variable slots when their previous occupants are no longer needed
    /// * Maintains correctness by respecting variable dependencies
    #[instrument(name = "reallocate intermediates", skip_all, level = "debug")]
    pub fn reallocate(&mut self, used_nodes: Vec<usize>) {
        let nodes_map = self.map_potential_overwrites(&used_nodes);
        self.collect_variable_info(nodes_map);
        self.compact_variable_ids();
        self.apply_variable_mapping();
        if !self.constraints.is_empty() {
            assert!(
                self.constraints
                    .iter()
                    .any(|c| !matches!(c.constraint, Constraint::Variable(_))),
                "All constraints are pure Variable copies; no computation will be performed."
            );
        }
    }

    /// Map the used_nodes to the max previous node for each node in asc order
    fn map_potential_overwrites(&self, used_nodes: &[usize]) -> Vec<usize> {
        // if used_nodes is sorted and unique, we can return it directly for O(N)
        if used_nodes.windows(2).all(|w| w[0] < w[1]) {
            return used_nodes.to_vec();
        }
        // iterate through all used_nodes to find max prev node for O(NlogN)
        let mut m: BTreeMap<usize, usize> = BTreeMap::new();
        let mut max_so_far = 0;

        for &node_idx in used_nodes {
            if node_idx > max_so_far {
                max_so_far = node_idx;
            }
            m.insert(node_idx, max_so_far);
        }

        m.values().cloned().collect()
    }

    fn collect_variable_info(&mut self, nodes_map: Vec<usize>) {
        // Iterate through all constraints to collect for each intermediate variable
        let mut variable_info = HashMap::new();
        let mut nodes_iter = nodes_map.iter().copied();
        for (idx, c) in self.constraints.iter().enumerate() {
            match &c.constraint {
                Constraint::Add(src1, src2, dest)
                | Constraint::Sub(src1, src2, dest)
                | Constraint::Mul(src1, src2, dest) => {
                    match dest {
                        Source::Intermediate(dest_id) => {
                            let info =
                                variable_info
                                    .entry(*dest_id)
                                    .or_insert_with(|| VariableInfo {
                                        def_point: idx,
                                        first_use: usize::MAX,
                                        last_use: idx,
                                        use_count: 0,
                                    });
                            // We need to be sure that previous node will not erase accumulated value
                            if c.need_accumulate {
                                let max_prev_node = nodes_iter
                                    .next()
                                    .expect("nodes map shorter than number of accumulators");
                                info.last_use = info.last_use.max(max_prev_node);
                            }
                        }
                        _ => panic!("Destination should be an intermediate variable"),
                    }

                    // if an intermediate variable is used in the `src`, update its use info
                    if let Source::Intermediate(src1_id) = src1 {
                        self.update_use_info(&mut variable_info, src1_id, idx);
                    }
                    if let Source::Intermediate(src2_id) = src2 {
                        self.update_use_info(&mut variable_info, src2_id, idx);
                    }
                }
                Constraint::Neg(src, dest) => {
                    match dest {
                        Source::Intermediate(dest_id) => {
                            let info =
                                variable_info
                                    .entry(*dest_id)
                                    .or_insert_with(|| VariableInfo {
                                        def_point: idx,
                                        first_use: usize::MAX,
                                        last_use: idx,
                                        use_count: 0,
                                    });
                            // We need to be sure that previous node will not erase accumulated value
                            if c.need_accumulate {
                                let max_prev_node = nodes_iter
                                    .next()
                                    .expect("nodes map shorter than number of accumulators");
                                info.last_use = info.last_use.max(max_prev_node);
                            }
                        }
                        _ => panic!("Destination should be an intermediate variable"),
                    }

                    if let Source::Intermediate(src_id) = src {
                        self.update_use_info(&mut variable_info, src_id, idx);
                    }
                }
                Constraint::Variable(_) => {
                    if c.need_accumulate {
                        nodes_iter.next();
                    }
                }
            }
        }
        self.variable_info = variable_info;
    }

    fn update_use_info(
        &self,
        variable_info: &mut HashMap<usize, VariableInfo>,
        src_id: &usize,
        idx: usize,
    ) {
        if let Some(info) = variable_info.get_mut(src_id) {
            // update use info
            info.first_use = info.first_use.min(idx);
            info.last_use = info.last_use.max(idx);
            info.use_count += 1;
        } else {
            panic!(
                "Variable {} used before definition at instruction {}",
                src_id, idx
            );
        }
    }

    fn compact_variable_ids(&mut self) {
        let mut next_available_id = 0;
        self.variable_mapping.clear();
        let mut id_usage: HashMap<usize, Option<usize>> = HashMap::new();

        // Sort variables by their definition points
        let mut vars: Vec<_> = self.variable_info.keys().cloned().collect();
        vars.sort_by_key(|var| {
            self.variable_info
                .get(var)
                .map(|info| info.def_point)
                .unwrap_or(0)
        });

        // Track usage of each ID:
        // - Some(last_use) indicates ID is in use until last_use point,
        // - None means available
        for var in vars {
            if let Some(info) = self.variable_info.get(&var) {
                // Assign new IDs to each variable:
                let new_id = {
                    let mut selected_id = None;
                    // - Prioritize reusing IDs that are no longer in use
                    for id in 0..next_available_id {
                        if let Some(Some(last_use)) = id_usage.get(&id) {
                            if *last_use < info.def_point {
                                selected_id = Some(id);
                                break;
                            }
                        }
                    }
                    // - Allocate new ID if no reusable ID is available
                    selected_id.unwrap_or_else(|| {
                        let id = next_available_id;
                        next_available_id += 1;
                        id
                    })
                };

                // Record mapping between old and new IDs
                id_usage.insert(new_id, Some(info.last_use));
                self.variable_mapping.insert(var, new_id);
            }
        }

        // Update total number of required intermediates
        self.num_intermediates = next_available_id;
    }

    fn apply_variable_mapping(&mut self) {
        let update_source = |source: &mut Source<F>| {
            if let Source::Intermediate(var_id) = source {
                if let Some(new_var) = self.variable_mapping.get(var_id) {
                    *var_id = *new_var;
                }
            }
        };

        // Traverse all constraints and replace old variable IDs with newly assigned IDs
        for c in &mut self.constraints {
            // Handle all constraint types: Add, Sub, Mul, Neg, Variable
            match &mut c.constraint {
                Constraint::Add(x, y, z) => {
                    update_source(x);
                    update_source(y);
                    update_source(z);
                }
                Constraint::Sub(x, y, z) => {
                    update_source(x);
                    update_source(y);
                    update_source(z);
                }
                Constraint::Mul(x, y, z) => {
                    update_source(x);
                    update_source(y);
                    update_source(z);
                }
                Constraint::Neg(x, z) => {
                    update_source(x);
                    update_source(z);
                }
                Constraint::Variable(x) => {
                    update_source(x);
                }
            }
        }
    }
}

pub fn match_constraints(
    old_rules: Vec<ConstraintWithFlag<F>>,
    new_rules: Vec<ConstraintWithFlag<F>>,
) {
    // Map from write source (z) buffer idx to constraint index i
    let mut old_write_buffer_to_constraint: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut new_write_buffer_to_constraint: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();

    for (i, (old, new)) in old_rules.iter().zip(new_rules.iter()).enumerate() {
        // First, handle write sources (z) and build the mapping
        match (&old.constraint, &new.constraint) {
            (Constraint::Add(old_x, old_y, old_z), Constraint::Add(new_x, new_y, new_z))
            | (Constraint::Sub(old_x, old_y, old_z), Constraint::Sub(new_x, new_y, new_z))
            | (Constraint::Mul(old_x, old_y, old_z), Constraint::Mul(new_x, new_y, new_z)) => {
                match (old_x, new_x) {
                    (Source::Intermediate(old_id), Source::Intermediate(new_id)) => {
                        assert_eq!(
                            *old_write_buffer_to_constraint.get(old_id).unwrap(),
                            *new_write_buffer_to_constraint.get(new_id).unwrap(),
                            "i: {}, old_id: {}, new_id: {}",
                            i,
                            *old_write_buffer_to_constraint.get(old_id).unwrap(),
                            *new_write_buffer_to_constraint.get(new_id).unwrap()
                        );
                    }
                    (Source::Var(_), Source::Var(_))
                    | (Source::IsFirst, Source::IsFirst)
                    | (Source::IsLast, Source::IsLast)
                    | (Source::IsTransition, Source::IsTransition)
                    | (Source::Constant(_), Source::Constant(_)) => {
                        assert_eq!(old_x, new_x);
                    }
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                }
                match (old_y, new_y) {
                    (Source::Intermediate(old_id), Source::Intermediate(new_id)) => {
                        assert_eq!(
                            *old_write_buffer_to_constraint.get(old_id).unwrap(),
                            *new_write_buffer_to_constraint.get(new_id).unwrap(),
                            "i: {}, old_id: {}, new_id: {}",
                            i,
                            *old_write_buffer_to_constraint.get(old_id).unwrap(),
                            *new_write_buffer_to_constraint.get(new_id).unwrap()
                        );
                    }
                    (Source::Var(_), Source::Var(_))
                    | (Source::IsFirst, Source::IsFirst)
                    | (Source::IsLast, Source::IsLast)
                    | (Source::IsTransition, Source::IsTransition)
                    | (Source::Constant(_), Source::Constant(_)) => {
                        assert_eq!(old_y, new_y);
                    }
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                }
                let old_z_buffer_idx = match old_z {
                    Source::Intermediate(id) => *id,
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                };
                let new_z_buffer_idx = match new_z {
                    Source::Intermediate(id) => *id,
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                };

                // Map both old and new write buffer indices to this constraint index
                old_write_buffer_to_constraint.insert(old_z_buffer_idx, i);
                new_write_buffer_to_constraint.insert(new_z_buffer_idx, i);
            }
            (Constraint::Neg(old_x, old_z), Constraint::Neg(new_x, new_z)) => {
                match (old_x, new_x) {
                    (Source::Intermediate(old_id), Source::Intermediate(new_id)) => {
                        assert_eq!(
                            *old_write_buffer_to_constraint.get(old_id).unwrap(),
                            *new_write_buffer_to_constraint.get(new_id).unwrap(),
                            "i: {}, old_id: {}, new_id: {}",
                            i,
                            *old_write_buffer_to_constraint.get(old_id).unwrap(),
                            *new_write_buffer_to_constraint.get(new_id).unwrap()
                        );
                    }
                    (Source::Var(_), Source::Var(_))
                    | (Source::IsFirst, Source::IsFirst)
                    | (Source::IsLast, Source::IsLast)
                    | (Source::IsTransition, Source::IsTransition)
                    | (Source::Constant(_), Source::Constant(_)) => {
                        assert_eq!(old_x, new_x);
                    }
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                }

                let old_z_buffer_idx = match old_z {
                    Source::Intermediate(id) => *id,
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                };
                let new_z_buffer_idx = match new_z {
                    Source::Intermediate(id) => *id,
                    _ => panic!(
                        "Intermediate read sources must be equal at constraint {}",
                        i
                    ),
                };

                // Map both old and new write buffer indices to this constraint index
                old_write_buffer_to_constraint.insert(old_z_buffer_idx, i);
                new_write_buffer_to_constraint.insert(new_z_buffer_idx, i);
            }
            (Constraint::Variable(_), Constraint::Variable(_)) => {
                // Variable constraint has no write source
            }
            _ => {
                panic!("Constraint types don't match at index {}", i);
            }
        }
    }
}
