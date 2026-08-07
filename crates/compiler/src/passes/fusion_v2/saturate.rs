//! Multi-round saturation driver bookkeeping.
//!
//! `detailed-fusion-plan-v2.md` §5.4 and §11. The [`SaturationState`]
//! sidecar tracks two pieces of state that live outside `GraphFuser` so
//! search bookkeeping does not mutate the alternative graph itself:
//!
//! - **`origins`**: for each alternative-graph node, the sorted set of seed `NodeId`s that node's
//!   execution represents. Seeds are their own origin (`origins[i] = {NodeId(i)}`); fused
//!   candidates take the union of their parents' origins. Generic composition rejects parents whose
//!   origin sets are not pairwise disjoint (§9.1 obligation / §11.2 origin check).
//! - **`seen_candidates`**: the persistent set of [`CandidateKey`] values accepted across all
//!   rounds. Two derivations that normalize to the same boundary + module hash (e.g. `(A+B)+C` and
//!   `A+(B+C)` after canonicalization) collide here and only the first is inserted.

use std::collections::{BTreeSet, HashSet};

use crate::passes::fusion_v2::{
    cost::ArtifactKey,
    model::{NodeId, ValueClassId},
};

/// Sidecar state for one `fuse_graph_v2` invocation. Owned by the
/// driver and threaded through every fusion-pass call in the current
/// round.
///
/// `origins` is dense — its length is equal to the current
/// `GraphFuser::nodes.len()`. `seen_candidates` grows monotonically
/// across rounds.
pub struct SaturationState {
    /// `origins[n.0]` = sorted set of seed `NodeId`s that node `n`
    /// represents. Empty for a freshly-allocated slot before the union
    /// is applied.
    pub origins: Vec<BTreeSet<NodeId>>,
    /// Cross-round dedup: candidate keys already inserted into the
    /// alternative graph.
    pub seen_candidates: HashSet<CandidateKey>,
}

impl SaturationState {
    /// Fresh state for a graph with `seed_node_count` seed nodes.
    /// Seeds are their own single-element origin set.
    pub fn new(seed_node_count: usize) -> Self {
        let origins = (0..seed_node_count)
            .map(|i| {
                let mut s = BTreeSet::new();
                s.insert(NodeId(i));
                s
            })
            .collect();
        Self {
            origins,
            seen_candidates: HashSet::new(),
        }
    }

    /// Union of the origins of every node in `nodes`. Empty when
    /// `nodes` is empty.
    pub fn union_origins(&self, nodes: &[NodeId]) -> BTreeSet<NodeId> {
        let mut out = BTreeSet::new();
        for n in nodes {
            out.extend(self.origins[n.0].iter().copied());
        }
        out
    }

    /// Whether the origin sets of `nodes` are pairwise disjoint.
    pub fn origins_disjoint(&self, nodes: &[NodeId]) -> bool {
        let mut seen = BTreeSet::new();
        for n in nodes {
            for seed in &self.origins[n.0] {
                if !seen.insert(*seed) {
                    return false;
                }
            }
        }
        true
    }

    /// Registers a new node's origins. `parents` is the list of parent
    /// nodes the candidate was synthesized from; the new node's
    /// origins are the union of theirs. Panics if `parents` have
    /// overlapping origins — callers must have called
    /// [`Self::origins_disjoint`] first.
    pub fn register_origins(&mut self, new_node: NodeId, parents: &[NodeId]) {
        debug_assert_eq!(
            new_node.0,
            self.origins.len(),
            "register_origins must be called in NodeId order"
        );
        debug_assert!(
            self.origins_disjoint(parents),
            "parents must have disjoint origins",
        );
        self.origins.push(self.union_origins(parents));
    }

    /// Records `key` in the seen set. Returns `true` if this is the
    /// first time the key has been seen.
    pub fn note_seen(&mut self, key: CandidateKey) -> bool {
        self.seen_candidates.insert(key)
    }
}

/// A candidate's persistent identity across saturation rounds.
///
/// `detailed-fusion-plan-v2.md` §9. Two candidates that normalize to
/// the same boundary values and the same compiled-artifact identity
/// are the same alternative; enumeration paths reaching the same
/// point (associative composition, redundant enumeration) collapse
/// here.
///
/// The key intentionally excludes origins — different derivations
/// producing the same normalized executable at the same boundary are
/// the same alternative regardless of which seed nodes they came from.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CandidateKey {
    pub inputs: Vec<ValueClassId>,
    pub outputs: Vec<ValueClassId>,
    pub artifact: ArtifactKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_have_singleton_origins() {
        let s = SaturationState::new(3);
        assert_eq!(s.origins.len(), 3);
        for i in 0..3 {
            let mut want = BTreeSet::new();
            want.insert(NodeId(i));
            assert_eq!(s.origins[i], want);
        }
    }

    #[test]
    fn disjoint_origins_check_catches_overlap() {
        let mut s = SaturationState::new(3);
        s.register_origins(NodeId(3), &[NodeId(0), NodeId(1)]);
        // A + AB has overlap on {A}.
        assert!(!s.origins_disjoint(&[NodeId(0), NodeId(3)]));
        // AB + C is disjoint.
        assert!(s.origins_disjoint(&[NodeId(3), NodeId(2)]));
    }

    #[test]
    fn note_seen_deduplicates() {
        let mut s = SaturationState::new(0);
        let key = CandidateKey {
            inputs: vec![ValueClassId(0)],
            outputs: vec![ValueClassId(1)],
            artifact: ArtifactKey {
                module_hash: [0; 32],
                target_arch: "test".into(),
                compiler_flags_hash: [0; 32],
            },
        };
        assert!(s.note_seen(key.clone()));
        assert!(!s.note_seen(key));
    }
}
