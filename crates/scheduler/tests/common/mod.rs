//! Shared driver for the scheduler tests.
//!
//! The driver keeps its own occupancy and completion bookkeeping, so budget and
//! dependency safety are checked against the declared graph rather than against
//! the engine's own accounting.
//!
//! Each test binary compiles this module separately, so not every binary uses
//! every helper.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use openvm_scheduler::{Admission, Budget, Engine, Node, ResourceProfile};

pub type Id = &'static str;

pub const GB: u64 = 1 << 30;

/// What one drive to completion observed.
pub struct Run {
    /// Ids admitted per admission pass, in admission order.
    pub rounds: Vec<Vec<Id>>,
    /// Passes that reported [`Admission::Backpressure`].
    pub backpressure: usize,
    /// Passes that reported [`Admission::Blocked`].
    pub blocked: usize,
    /// Largest simultaneous occupancy the driver itself summed.
    pub peak: ResourceProfile,
}

impl Run {
    /// Widest admission pass — how much concurrency the budget allowed.
    pub fn widest_round(&self) -> usize {
        self.rounds.iter().map(Vec::len).max().unwrap_or(0)
    }

    pub fn admitted_count(&self) -> usize {
        self.rounds.iter().map(Vec::len).sum()
    }

    /// Index of the pass that admitted `id`.
    pub fn round_of(&self, id: Id) -> usize {
        self.rounds
            .iter()
            .position(|round| round.contains(&id))
            .unwrap_or_else(|| panic!("{id} was never admitted"))
    }
}

/// Drive `nodes` to completion under `budget`, completing the oldest in-flight
/// node whenever no further admission is possible.
///
/// Panics if the engine puts occupancy over budget on any axis, admits a node
/// whose predecessors have not completed, or stalls with nothing in flight.
pub fn drive(budget: Budget, nodes: Vec<Node<Id>>) -> Run {
    let profiles: HashMap<Id, ResourceProfile> = nodes.iter().map(|n| (n.id, n.profile)).collect();
    let deps: HashMap<Id, Vec<Id>> = nodes
        .iter()
        .map(|n| (n.id, n.dependencies.clone()))
        .collect();

    let mut engine = Engine::new(budget);
    for node in nodes {
        engine.add_node(node).expect("graph is well formed");
    }

    let mut run = Run {
        rounds: Vec::new(),
        backpressure: 0,
        blocked: 0,
        peak: ResourceProfile::ZERO,
    };
    let mut resident = ResourceProfile::ZERO;
    let mut in_flight: VecDeque<Id> = VecDeque::new();
    let mut completed: HashSet<Id> = HashSet::new();

    loop {
        let admission = engine.admit();
        match &admission {
            Admission::AllComplete => return run,
            Admission::Backpressure => run.backpressure += 1,
            Admission::Blocked => run.blocked += 1,
            Admission::Admitted(ids) => {
                for id in ids {
                    for dep in &deps[id] {
                        assert!(
                            completed.contains(dep),
                            "dependency safety: {id} admitted while {dep} was unfinished"
                        );
                    }
                    let profile = profiles[id];
                    resident.gpu_bytes += profile.gpu_bytes;
                    resident.host_bytes += profile.host_bytes;
                    resident.cpu_threads += profile.cpu_threads;
                    in_flight.push_back(*id);
                }
                assert!(
                    resident.gpu_bytes <= budget.gpu_bytes
                        && resident.host_bytes <= budget.host_bytes
                        && resident.cpu_threads <= budget.cpu_threads,
                    "budget safety: resident {resident:?} exceeds {budget:?}"
                );
                run.peak.gpu_bytes = run.peak.gpu_bytes.max(resident.gpu_bytes);
                run.peak.host_bytes = run.peak.host_bytes.max(resident.host_bytes);
                run.peak.cpu_threads = run.peak.cpu_threads.max(resident.cpu_threads);
                run.rounds.push(ids.clone());
                continue;
            }
        }

        let id = in_flight
            .pop_front()
            .expect("engine stalled with nothing in flight");
        let profile = profiles[id];
        resident.gpu_bytes -= profile.gpu_bytes;
        resident.host_bytes -= profile.host_bytes;
        resident.cpu_threads -= profile.cpu_threads;
        completed.insert(id);
        engine.complete(&id).expect("node was admitted");
    }
}
