//! The subdivision test.
//!
//! The next increment replaces one coarse GPU node with several finer nodes that
//! carry their own profiles and their own edges. Both graphs below are built
//! from the same `Node` type and driven by the same unchanged `common::drive`:
//! subdividing a node is a change to the node list, never to an interface.

use common::{drive, Id, GB};
use openvm_scheduler::{Budget, Node, ResourceProfile};

mod common;

const CPU_BOUND: ResourceProfile = ResourceProfile::new(0, 4 * GB, 8);
const GPU_BOUND: ResourceProfile = ResourceProfile::new(16 * GB, 8 * GB, 2);

fn coarse() -> Vec<Node<Id>> {
    vec![
        Node::new("c0", vec![], CPU_BOUND),
        Node::new("g0", vec!["c0"], GPU_BOUND),
        Node::new("c1", vec!["c0"], CPU_BOUND),
        Node::new("g1", vec!["c1"], GPU_BOUND),
    ]
}

/// Each GPU node split into two heterogeneous stages and a join. Their GPU peaks
/// sum to the coarse node's, so the same budget describes the same machine.
fn subdivided() -> Vec<Node<Id>> {
    let stage_a = ResourceProfile::new(4 * GB, 2 * GB, 1);
    let stage_b = ResourceProfile::new(6 * GB, 3 * GB, 1);
    let join = ResourceProfile::new(6 * GB, 3 * GB, 2);
    vec![
        Node::new("c0", vec![], CPU_BOUND),
        Node::new("g0.a", vec!["c0"], stage_a),
        Node::new("g0.b", vec!["c0"], stage_b),
        Node::new("g0.join", vec!["g0.a", "g0.b"], join),
        Node::new("c1", vec!["c0"], CPU_BOUND),
        Node::new("g1.a", vec!["c1"], stage_a),
        Node::new("g1.b", vec!["c1"], stage_b),
        Node::new("g1.join", vec!["g1.a", "g1.b"], join),
    ]
}

#[test]
fn subdividing_a_node_needs_no_interface_change() {
    let budget = Budget::new(16 * GB, 32 * GB, 16);

    let coarse_run = drive(budget, coarse());
    let fine_run = drive(budget, subdivided());

    assert_eq!(
        coarse_run.rounds,
        vec![vec!["c0"], vec!["g0", "c1"], vec!["g1"]]
    );
    assert_eq!(
        fine_run.rounds,
        vec![
            vec!["c0"],
            vec!["g0.b", "g0.a", "c1"],
            vec!["g0.join"],
            vec!["g1.b", "g1.a"],
            vec!["g1.join"],
        ]
    );

    assert_eq!(coarse_run.admitted_count(), 4);
    assert_eq!(fine_run.admitted_count(), 8);

    // One coarse node saturates the GPU axis alone; the finer nodes share it, so
    // subdivision buys concurrency the budget could not express before.
    assert!(fine_run.widest_round() > coarse_run.widest_round());
    assert!(fine_run.peak.gpu_bytes <= budget.gpu_bytes);

    // The sub-graph's own edges are scheduled like any other.
    assert!(fine_run.round_of("g0.join") > fine_run.round_of("g0.a"));
    assert!(fine_run.round_of("g1.join") > fine_run.round_of("g1.b"));
}
