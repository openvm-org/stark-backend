use std::{cmp::Reverse, collections::HashMap, fmt, hash::Hash, mem};

use crate::{Budget, ResourceProfile};

/// A unit of schedulable work: an id of the caller's choosing, the ids it must
/// wait for, and what it occupies while it runs.
#[derive(Clone, Debug)]
pub struct Node<I> {
    pub id: I,
    pub dependencies: Vec<I>,
    pub profile: ResourceProfile,
}

impl<I> Node<I> {
    pub fn new(id: I, dependencies: Vec<I>, profile: ResourceProfile) -> Self {
        Self {
            id,
            dependencies,
            profile,
        }
    }
}

/// The outcome of one call to [`Engine::admit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Admission<I> {
    /// These nodes now hold their profiles. The caller runs them and reports
    /// each one back through [`Engine::complete`].
    Admitted(Vec<I>),
    /// A node is ready but does not fit what is left of the budget. Progress
    /// needs an in-flight node to complete first.
    Backpressure,
    /// No pending node is ready while registered work remains. A pending node may
    /// be waiting on an unfinished predecessor, and an already-admitted node is
    /// excluded from later passes until it completes — so a single running root
    /// with no dependents also reports this.
    Blocked,
    /// Every registered node has completed. A caller that may still register
    /// more work reads this as idle rather than final.
    AllComplete,
}

/// Why a graph edit or a completion was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedulerError<I> {
    /// The id is already registered.
    DuplicateNode(I),
    /// `node` names a dependency that is not registered yet.
    UnknownDependency { node: I, dependency: I },
    /// The profile exceeds the budget on some axis, so the node could never be
    /// admitted.
    ExceedsBudget(I),
    /// The node is not currently admitted, so there is nothing to release.
    NotRunning(I),
}

impl<I: fmt::Debug> fmt::Display for SchedulerError<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(f, "node {id:?} is already registered"),
            Self::UnknownDependency { node, dependency } => {
                write!(f, "node {node:?} depends on unregistered {dependency:?}")
            }
            Self::ExceedsBudget(id) => write!(f, "node {id:?} does not fit the budget"),
            Self::NotRunning(id) => write!(f, "node {id:?} is not running"),
        }
    }
}

impl<I: fmt::Debug> std::error::Error for SchedulerError<I> {}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Pending,
    Running,
    Complete,
}

struct NodeState<I> {
    id: I,
    profile: ResourceProfile,
    /// Predecessors that have not completed yet.
    pending_deps: usize,
    /// Dependents still counting this node in their `pending_deps`.
    successors: Vec<usize>,
    state: State,
}

impl<I> NodeState<I> {
    fn is_ready(&self) -> bool {
        self.state == State::Pending && self.pending_deps == 0
    }
}

/// Admits graph nodes against a [`Budget`].
///
/// The engine never runs anything and never blocks: [`Engine::admit`] and
/// [`Engine::complete`] are bookkeeping, so the caller keeps full control over
/// threads, streams, and ordering.
pub struct Engine<I> {
    budget: Budget,
    /// Sum of the profiles of every running node, and so never over `budget` on
    /// any axis.
    in_use: ResourceProfile,
    nodes: Vec<NodeState<I>>,
    index: HashMap<I, usize>,
    completed: usize,
}

impl<I: Clone + Eq + Hash> Engine<I> {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            in_use: ResourceProfile::ZERO,
            nodes: Vec::new(),
            index: HashMap::new(),
            completed: 0,
        }
    }

    /// Register a node. Every dependency must already be registered, which makes
    /// the graph acyclic by construction — there is no cycle check because a
    /// cycle cannot be expressed.
    ///
    /// Nodes may be added while others run, so a caller that discovers work as
    /// it goes can keep extending the graph.
    pub fn add_node(&mut self, node: Node<I>) -> Result<(), SchedulerError<I>> {
        if self.index.contains_key(&node.id) {
            return Err(SchedulerError::DuplicateNode(node.id));
        }
        if !self
            .budget
            .has_room_for(&ResourceProfile::ZERO, &node.profile)
        {
            return Err(SchedulerError::ExceedsBudget(node.id));
        }

        // Resolve every dependency before touching the graph, so a rejected node
        // leaves no partial edges behind.
        let mut dependencies = Vec::with_capacity(node.dependencies.len());
        for dependency in &node.dependencies {
            match self.index.get(dependency) {
                Some(&idx) => dependencies.push(idx),
                None => {
                    return Err(SchedulerError::UnknownDependency {
                        node: node.id,
                        dependency: dependency.clone(),
                    })
                }
            }
        }

        let idx = self.nodes.len();
        let mut pending_deps = 0;
        for dependency in dependencies {
            // Only an unfinished predecessor needs an edge, so each counted
            // dependency has exactly one successor entry to release later.
            if self.nodes[dependency].state != State::Complete {
                pending_deps += 1;
                self.nodes[dependency].successors.push(idx);
            }
        }

        self.index.insert(node.id.clone(), idx);
        self.nodes.push(NodeState {
            id: node.id,
            profile: node.profile,
            pending_deps,
            successors: Vec::new(),
            state: State::Pending,
        });
        Ok(())
    }

    /// Admit every ready node that still fits, GPU-first.
    ///
    /// A ready node that does not fit is skipped rather than reserved, so a
    /// smaller node behind it can still be admitted. Over a finite graph, a
    /// caller that completes what it admits *and* keeps calling this after each
    /// completion admits every node in the end — the engine is passive and makes
    /// no progress on its own, so a caller that stops asking leaves ready work
    /// unadmitted forever. A caller that keeps registering new work can also
    /// starve a large node indefinitely; this admits no fairness rule against
    /// that.
    pub fn admit(&mut self) -> Admission<I> {
        let mut ready: Vec<usize> = (0..self.nodes.len())
            .filter(|&idx| self.nodes[idx].is_ready())
            .collect();
        // The GPU is the scarce axis, so the heaviest GPU consumer claims
        // contended capacity ahead of any lighter or CPU-only node. The sort is
        // stable, so registration order breaks ties.
        ready.sort_by_key(|&idx| Reverse(self.nodes[idx].profile.gpu_bytes));

        let mut admitted = Vec::new();
        for &idx in &ready {
            let profile = self.nodes[idx].profile;
            if !self.budget.has_room_for(&self.in_use, &profile) {
                continue;
            }
            // `has_room_for` just proved these sums stay within the budget.
            self.in_use.gpu_bytes += profile.gpu_bytes;
            self.in_use.host_bytes += profile.host_bytes;
            self.in_use.cpu_threads += profile.cpu_threads;
            self.nodes[idx].state = State::Running;
            admitted.push(self.nodes[idx].id.clone());
        }

        if !admitted.is_empty() {
            Admission::Admitted(admitted)
        } else if !ready.is_empty() {
            Admission::Backpressure
        } else if self.completed == self.nodes.len() {
            Admission::AllComplete
        } else {
            Admission::Blocked
        }
    }

    /// Release a running node's resources and unblock its dependents.
    pub fn complete(&mut self, id: &I) -> Result<(), SchedulerError<I>> {
        let idx = match self.index.get(id) {
            Some(&idx) if self.nodes[idx].state == State::Running => idx,
            _ => return Err(SchedulerError::NotRunning(id.clone())),
        };

        let profile = self.nodes[idx].profile;
        // Mirrors the admission that reserved this profile, so it cannot
        // underflow.
        self.in_use.gpu_bytes -= profile.gpu_bytes;
        self.in_use.host_bytes -= profile.host_bytes;
        self.in_use.cpu_threads -= profile.cpu_threads;
        self.nodes[idx].state = State::Complete;
        self.completed += 1;

        // A node completes once, so its edges are consumed for good.
        for successor in mem::take(&mut self.nodes[idx].successors) {
            self.nodes[successor].pending_deps -= 1;
        }
        Ok(())
    }
}
