//! The subdivision test.
//!
//! The next increment replaces one coarse GPU node with a sub-graph of finer
//! nodes carrying their own profiles and edges, while the work on either side of
//! it stays put. Both graphs below hold the same producer and the same consumer
//! fixed and are driven by the same unchanged `common::drive`: subdividing a node
//! is a change to graph data, never to an interface.

use common::{drive, Id, GB};
use openvm_scheduler::{Budget, Node, ResourceProfile};

mod common;

const CPU_BOUND: ResourceProfile = ResourceProfile::new(0, 4 * GB, 8);
const GPU_BOUND: ResourceProfile = ResourceProfile::new(16 * GB, 8 * GB, 2);

/// The producer upstream of the node being subdivided.
const UP: Id = "up";
/// The node that gets replaced.
const MID: Id = "mid";
/// The consumer downstream of it.
const DOWN: Id = "down";
/// The sub-graph that replaces [`MID`], ending in a join.
const MID_A: Id = "mid.a";
const MID_B: Id = "mid.b";
const MID_JOIN: Id = "mid.join";

/// The two nodes that must survive subdivision untouched. `produces` is the id
/// the consumer waits for — the only thing that differs between the two graphs,
/// and it is a dependency id: graph data the caller already owns, not part of any
/// interface.
fn boundary(produces: Id) -> (Node<Id>, Node<Id>) {
    (
        Node::new(UP, vec![], CPU_BOUND),
        Node::new(DOWN, vec![produces], CPU_BOUND),
    )
}

/// `UP -> MID -> DOWN`, one coarse GPU node in the middle.
fn coarse() -> Vec<Node<Id>> {
    let (up, down) = boundary(MID);
    vec![up, Node::new(MID, vec![UP], GPU_BOUND), down]
}

/// [`MID`] replaced by two heterogeneous stages and a join. Nothing else moves,
/// and the stages' GPU peaks sum to [`MID`]'s, so the same budget describes the
/// same machine.
fn subdivided() -> Vec<Node<Id>> {
    let (up, down) = boundary(MID_JOIN);
    vec![
        up,
        Node::new(MID_A, vec![UP], ResourceProfile::new(4 * GB, 2 * GB, 1)),
        Node::new(MID_B, vec![UP], ResourceProfile::new(6 * GB, 3 * GB, 1)),
        Node::new(
            MID_JOIN,
            vec![MID_A, MID_B],
            ResourceProfile::new(6 * GB, 3 * GB, 2),
        ),
        down,
    ]
}

#[test]
fn subdividing_one_node_keeps_the_boundary_around_it_intact() {
    let budget = Budget::new(16 * GB, 32 * GB, 16);
    let coarse_graph = coarse();
    let fine_graph = subdivided();

    // The surviving nodes are the same declarations on both sides: the producer
    // is identical down to its edges, and the consumer keeps its id and profile
    // while naming a different producer.
    let (coarse_up, fine_up) = (&coarse_graph[0], &fine_graph[0]);
    assert_eq!(coarse_up.id, fine_up.id);
    assert_eq!(coarse_up.profile, fine_up.profile);
    assert_eq!(coarse_up.dependencies, fine_up.dependencies);

    let coarse_down = coarse_graph.last().unwrap();
    let fine_down = fine_graph.last().unwrap();
    assert_eq!(coarse_down.id, fine_down.id);
    assert_eq!(coarse_down.profile, fine_down.profile);
    assert_eq!(coarse_down.dependencies, vec![MID]);
    assert_eq!(fine_down.dependencies, vec![MID_JOIN]);

    let coarse_run = drive(budget, coarse_graph);
    let fine_run = drive(budget, fine_graph);

    assert_eq!(coarse_run.rounds, vec![vec![UP], vec![MID], vec![DOWN]]);
    assert_eq!(
        fine_run.rounds,
        vec![vec![UP], vec![MID_B, MID_A], vec![MID_JOIN], vec![DOWN],]
    );

    // The external boundary still holds. The consumer waits for the whole
    // replacement sub-graph, not just the join it names, and the producer still
    // precedes every one of the finer nodes.
    assert!(fine_run.round_of(DOWN) > fine_run.round_of(MID_JOIN));
    assert!(fine_run.round_of(MID_JOIN) > fine_run.round_of(MID_A));
    assert!(fine_run.round_of(MID_JOIN) > fine_run.round_of(MID_B));
    assert!(fine_run.round_of(MID_A) > fine_run.round_of(UP));
    assert!(fine_run.round_of(MID_B) > fine_run.round_of(UP));
    assert_eq!(fine_run.rounds.first().unwrap(), &vec![UP]);
    assert_eq!(fine_run.rounds.last().unwrap(), &vec![DOWN]);

    // Subdivision buys concurrency the coarse node could not express, without
    // moving the boundary or exceeding the same budget.
    assert_eq!(coarse_run.widest_round(), 1);
    assert_eq!(fine_run.widest_round(), 2);
    assert!(fine_run.peak.gpu_bytes <= budget.gpu_bytes);
}
