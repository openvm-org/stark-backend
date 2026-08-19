//! The engine's load-bearing invariants: budget safety, dependency safety,
//! GPU-first admission order, and the crate's ignorance of its callers.

use common::{drive, Id, GB};
use openvm_scheduler::{Admission, Budget, Engine, Node, ResourceProfile, SchedulerError};

mod common;

/// A CPU-bound node: no GPU residency.
const CPU_BOUND: ResourceProfile = ResourceProfile::new(0, 4 * GB, 8);
/// A GPU-bound node.
const GPU_BOUND: ResourceProfile = ResourceProfile::new(16 * GB, 8 * GB, 2);

/// Occupancy never passes the budget on any axis, and a node too large for what
/// remains waits for a release rather than overrunning it.
#[test]
fn budget_safety_defers_a_node_that_does_not_fit() {
    let budget = Budget::new(32 * GB, 64 * GB, 16);
    let nodes = vec![
        Node::new("A", vec![], ResourceProfile::new(16 * GB, 8 * GB, 1)),
        Node::new("B", vec![], ResourceProfile::new(16 * GB, 8 * GB, 1)),
        // Ready only once A completes, and then too large to join B.
        Node::new("C", vec!["A"], ResourceProfile::new(24 * GB, 8 * GB, 1)),
    ];

    let run = drive(budget, nodes);

    assert_eq!(run.rounds, vec![vec!["A", "B"], vec!["C"]]);
    assert_eq!(run.peak.gpu_bytes, 32 * GB);
    assert_eq!(run.backpressure, 1, "C waits for B to release the GPU axis");
}

/// A serial CPU chain with a fan-out of independent GPU nodes. Under a budget
/// wide enough to hold the whole graph at once, admission order is decided purely
/// by dependencies.
#[test]
fn dependency_safety_holds_under_an_unconstrained_budget() {
    let budget = Budget::new(1024 * GB, 1024 * GB, 1024);
    let nodes = vec![
        Node::new("c0", vec![], CPU_BOUND),
        Node::new("g0", vec!["c0"], GPU_BOUND),
        Node::new("c1", vec!["c0"], CPU_BOUND),
        Node::new("g1", vec!["c1"], GPU_BOUND),
        Node::new("c2", vec!["c1"], CPU_BOUND),
        Node::new("g2", vec!["c2"], GPU_BOUND),
    ];

    let run = drive(budget, nodes);

    assert_eq!(
        run.rounds,
        vec![vec!["c0"], vec!["g0", "c1"], vec!["g1", "c2"], vec!["g2"]]
    );
    assert_eq!(run.admitted_count(), 6);
}

/// A predecessor that is merely *in flight* still blocks. The budget is wide
/// enough that backpressure is impossible, so the only thing keeping the
/// dependent out is its unfinished predecessor — and it is admitted on the very
/// next pass after that predecessor completes.
#[test]
fn a_dependent_waits_while_its_predecessor_is_in_flight() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(1024 * GB, 1024 * GB, 1024));
    engine.add_node(Node::new("c", vec![], CPU_BOUND)).unwrap();
    engine
        .add_node(Node::new("g", vec!["c"], GPU_BOUND))
        .unwrap();

    assert_eq!(engine.admit(), Admission::Admitted(vec!["c"]));
    assert_eq!(engine.admit(), Admission::Blocked);

    engine.complete(&"c").unwrap();

    assert_eq!(engine.admit(), Admission::Admitted(vec!["g"]));
}

/// Repeating a pass over an already-admitted root reports `Blocked` and reserves
/// nothing further. The root is unfinished but waits on no predecessor, so
/// `Blocked` cannot be read as "some predecessor is in flight".
#[test]
fn a_second_admit_pass_over_a_running_root_reports_blocked() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(32 * GB, 64 * GB, 16));
    engine
        .add_node(Node::new("root", vec![], CPU_BOUND))
        .unwrap();

    assert_eq!(engine.admit(), Admission::Admitted(vec!["root"]));
    assert_eq!(engine.admit(), Admission::Blocked);
    assert_eq!(engine.admit(), Admission::Blocked);

    engine.complete(&"root").unwrap();

    assert_eq!(engine.admit(), Admission::AllComplete);
}

/// GPU-first: when a contended axis admits only one of two ready nodes, the GPU
/// consumer goes first even though the CPU-only node was registered earlier.
/// The control case pins that the GPU axis is what flips the order: with GPU
/// demand equal, registration order decides.
#[test]
fn gpu_demand_wins_a_contended_axis() {
    let budget = Budget::new(32 * GB, 10 * GB, 16);
    let cpu_only = ResourceProfile::new(0, 10 * GB, 8);
    let gpu_bound = ResourceProfile::new(16 * GB, 10 * GB, 8);

    let run = drive(
        budget,
        vec![
            Node::new("cpu", vec![], cpu_only),
            Node::new("gpu", vec![], gpu_bound),
        ],
    );
    assert_eq!(run.rounds, vec![vec!["gpu"], vec!["cpu"]]);

    let control = drive(
        budget,
        vec![
            Node::new("cpu", vec![], cpu_only),
            Node::new("also-cpu", vec![], cpu_only),
        ],
    );
    assert_eq!(control.rounds, vec![vec!["cpu"], vec!["also-cpu"]]);
}

/// A node that cannot fit an empty machine would stall the graph forever, so it
/// is refused at registration instead.
#[test]
fn a_node_larger_than_the_budget_is_rejected() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(32 * GB, 64 * GB, 16));

    let err = engine
        .add_node(Node::new(
            "huge",
            vec![],
            ResourceProfile::new(33 * GB, 0, 0),
        ))
        .unwrap_err();

    assert_eq!(err, SchedulerError::ExceedsBudget("huge"));
}

/// Dependencies must be registered before their dependents. That is what makes
/// the graph acyclic by construction.
#[test]
fn a_dependency_must_be_registered_first() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(32 * GB, 64 * GB, 16));

    let err = engine
        .add_node(Node::new("g0", vec!["c0"], GPU_BOUND))
        .unwrap_err();

    assert_eq!(
        err,
        SchedulerError::UnknownDependency {
            node: "g0",
            dependency: "c0",
        }
    );
}

#[test]
fn a_duplicate_id_is_rejected() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(32 * GB, 64 * GB, 16));
    engine.add_node(Node::new("c0", vec![], CPU_BOUND)).unwrap();

    let err = engine
        .add_node(Node::new("c0", vec![], CPU_BOUND))
        .unwrap_err();

    assert_eq!(err, SchedulerError::DuplicateNode("c0"));
}

#[test]
fn completing_a_node_that_was_not_admitted_is_rejected() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(32 * GB, 64 * GB, 16));

    assert_eq!(
        engine.complete(&"absent").unwrap_err(),
        SchedulerError::NotRunning("absent")
    );

    engine.add_node(Node::new("c0", vec![], CPU_BOUND)).unwrap();
    assert_eq!(
        engine.complete(&"c0").unwrap_err(),
        SchedulerError::NotRunning("c0")
    );
}

/// The graph may grow while work is in flight, which is what lets a caller
/// register work it only discovers as it goes.
#[test]
fn a_node_added_after_its_predecessor_completed_is_ready_at_once() {
    let mut engine: Engine<Id> = Engine::new(Budget::new(32 * GB, 64 * GB, 16));
    engine.add_node(Node::new("c0", vec![], CPU_BOUND)).unwrap();

    assert_eq!(engine.admit(), Admission::Admitted(vec!["c0"]));
    engine.complete(&"c0").unwrap();
    assert_eq!(engine.admit(), Admission::AllComplete);

    engine
        .add_node(Node::new("g0", vec!["c0"], GPU_BOUND))
        .unwrap();

    assert_eq!(engine.admit(), Admission::Admitted(vec!["g0"]));
}

/// The engine stays ignorant of its callers. A dependency edge is the only way
/// that could change, so pin it here rather than leave it to review.
#[test]
fn the_manifest_declares_no_dependencies() {
    let manifest = include_str!("../Cargo.toml");
    let mut section = "";
    let declared: Vec<&str> = manifest
        .lines()
        .map(str::trim)
        .filter(|line| {
            if line.starts_with('[') {
                section = line;
                return false;
            }
            section.contains("dependencies") && !line.is_empty() && !line.starts_with('#')
        })
        .collect();

    assert!(
        declared.is_empty(),
        "the engine crate must depend on nothing, found {declared:?}"
    );
}
