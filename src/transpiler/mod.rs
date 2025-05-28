extern crate alloc;

use alloc::vec::Vec;
use std::collections::{HashMap, HashSet};

use codec::as_intermediate;
use itertools::Itertools;
use openvm_stark_backend::air_builders::symbolic::{
    symbolic_expression::SymbolicExpression, symbolic_variable::SymbolicVariable,
    SymbolicExpressionDag, SymbolicExpressionNode,
};
use p3_field::{Field, PrimeField32};
use rustc_hash::FxHashMap;
use tracing::instrument;

pub(crate) mod codec;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Source<F: Field> {
    Intermediate(usize),
    Var(SymbolicVariable<F>),
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

pub struct SymbolicRulesOnGpu<F: Field> {
    pub constraints: Vec<ConstraintWithFlag<F>>,
    pub num_intermediates: usize,
}

impl<F: Field + PrimeField32> SymbolicRulesOnGpu<F> {
    /// transpile constraints in dag to a form that can be evaluated by the gpu prover
    /// especially we minimize the vector for storing intermediates to fit in gpu shared memory
    #[instrument(name = "SymbolicRulesOnGpu.new", skip_all, level = "debug")]
    pub fn new(dag: SymbolicExpressionDag<F>) -> Self {
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
        let constraint_idx_set: HashSet<usize> =
            HashSet::from_iter(dag.constraint_idx.iter().copied());
        for (node_idx, node) in dag.nodes.into_iter().enumerate() {
            let assert_zero = constraint_idx_set.contains(&node_idx);

            let constraint = if !is_intermediate(&node) {
                Constraint::Variable(node.into())
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
            dag.constraint_idx
        );

        let num_intermediates_before_reduction = num_intermediates_array
            .last()
            .copied()
            .map(|n| n + 1)
            .unwrap_or(0);

        let mut analyzer = IntermediateAllocator::new(compiled_constraints);
        analyzer.reallocate();

        tracing::debug!(
            "[IntermediateAllocator] size of minimal vector to store intermediates: before = {}, after = {}",
            num_intermediates_before_reduction,
            analyzer.num_intermediates,

        );

        SymbolicRulesOnGpu {
            constraints: analyzer.constraints,
            num_intermediates: analyzer.num_intermediates,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct IntermediateVariable {
    id: usize,
}

/// Information about an intermediate variable's usage and dependencies
///
/// Tracks the lifecycle and relationships of intermediate variables:
/// * `def_point` - Instruction index where variable is first defined
/// * `first_use` - Instruction index of first usage
/// * `last_use` - Instruction index of last usage
/// * `use_count` - Total number of times the variable is used
/// * `dependencies` - Set of variables this variable depends on
/// * `dependents` - Set of variables that depend on this variable
#[derive(Debug)]
struct VariableInfo {
    def_point: usize,
    first_use: usize,
    last_use: usize,
    use_count: usize,
    dependencies: HashSet<IntermediateVariable>,
    dependents: HashSet<IntermediateVariable>,
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
    /// * Dependencies/Dependents - Variable dependency relationships
    variable_info: HashMap<IntermediateVariable, VariableInfo>,

    /// Maps original variable IDs to their optimized/reallocated IDs.
    /// Used during the reallocation process to track how variables are reassigned
    /// to minimize the total number of required intermediate slots.
    variable_mapping: HashMap<IntermediateVariable, IntermediateVariable>,

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
    pub fn reallocate(&mut self) {
        self.collect_variable_info();
        self.compact_variable_ids();
        self.apply_variable_mapping();
        if !self.constraints.is_empty() {
            assert!(self.num_intermediates > 0);
        }
    }

    fn collect_variable_info(&mut self) {
        // Iterate through all constraints to collect for each intermediate variable
        let mut variable_info = HashMap::new();
        for (idx, c) in self.constraints.iter().enumerate() {
            match &c.constraint {
                Constraint::Add(src1, src2, dest)
                | Constraint::Sub(src1, src2, dest)
                | Constraint::Mul(src1, src2, dest) => {
                    // we only care about intermediate variables
                    // an intermediate variable is first defined in the `dest` field
                    if let Source::Intermediate(dest_id) = dest {
                        let dest_var = IntermediateVariable { id: *dest_id };
                        variable_info
                            .entry(dest_var.clone())
                            .or_insert_with(|| VariableInfo {
                                def_point: idx,
                                first_use: usize::MAX,
                                last_use: idx,
                                use_count: 0,
                                dependencies: HashSet::new(),
                                dependents: HashSet::new(),
                            });
                    }

                    // if an intermediate variable is used in the `src`, update its use info
                    if let Source::Intermediate(src1_id) = src1 {
                        self.update_use_info(&mut variable_info, src1_id, idx, dest);
                    }
                    if let Source::Intermediate(src2_id) = src2 {
                        self.update_use_info(&mut variable_info, src2_id, idx, dest);
                    }
                }
                Constraint::Neg(src, dest) => {
                    if let Source::Intermediate(dest_id) = dest {
                        let dest_var = IntermediateVariable { id: *dest_id };
                        variable_info
                            .entry(dest_var.clone())
                            .or_insert_with(|| VariableInfo {
                                def_point: idx,
                                first_use: usize::MAX,
                                last_use: idx,
                                use_count: 0,
                                dependencies: HashSet::new(),
                                dependents: HashSet::new(),
                            });
                    }

                    if let Source::Intermediate(src_id) = src {
                        self.update_use_info(&mut variable_info, src_id, idx, dest);
                    }
                }
                // todo: check
                Constraint::Variable(src) => {
                    if let Source::Intermediate(src_id) = src {
                        let src_var = IntermediateVariable { id: *src_id };
                        let info = variable_info
                            .entry(src_var)
                            .or_insert_with(|| VariableInfo {
                                def_point: idx,
                                first_use: idx,
                                last_use: idx,
                                use_count: 1,
                                dependencies: HashSet::new(),
                                dependents: HashSet::new(),
                            });
                        info.last_use = info.last_use.max(idx);
                        info.use_count += 1;
                    }
                }
            }
        }
        self.variable_info = variable_info;
    }

    fn update_use_info(
        &self,
        variable_info: &mut HashMap<IntermediateVariable, VariableInfo>,
        src_id: &usize,
        idx: usize,
        dest: &Source<F>,
    ) {
        let src_var = IntermediateVariable { id: *src_id };

        if let Some(info) = variable_info.get_mut(&src_var) {
            // update use info
            info.first_use = info.first_use.min(idx);
            info.last_use = info.last_use.max(idx);
            info.use_count += 1;

            // track dependencies
            let dest_id = as_intermediate(dest).unwrap();
            let dest_var = IntermediateVariable { id: *dest_id };
            info.dependents.insert(dest_var.clone());
            if let Some(dest_info) = variable_info.get_mut(&dest_var) {
                dest_info.dependencies.insert(src_var);
            }
        } else {
            panic!(
                "IntermediateVariable t{} used before definition at instruction {}",
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

                let new_var = IntermediateVariable { id: new_id };
                self.variable_mapping.insert(var, new_var);
            }
        }

        // Update total number of required intermediates
        self.num_intermediates = next_available_id;
    }

    fn apply_variable_mapping(&mut self) {
        // Traverse all constraints and replace old variable IDs with newly assigned IDs
        for c in &mut self.constraints {
            let update_source = |source: &mut Source<F>| {
                if let Source::Intermediate(var_id) = source {
                    if let Some(new_var) = self
                        .variable_mapping
                        .get(&IntermediateVariable { id: *var_id })
                    {
                        *var_id = new_var.id;
                    }
                }
            };

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

pub fn dummy_inv<F: Field>(x: SymbolicExpression<F>) -> SymbolicExpression<F> {
    -(-(-x))
}
