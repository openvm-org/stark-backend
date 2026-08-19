//! Restore SSA form for in-place mutations and buffer reuse.
//!
//! Two graph-building idioms break `verify_graph`'s single-writer / SSA
//! check:
//!
//! 1. `insert_blackbox_kernel(inputs, outputs, modifies)` lets a caller declare that some inputs
//!    are mutated in place — those buffers wind up in `KernelNode::carried_outputs` and share their
//!    `BufId` with the input side, so chains of mutations give the buffer several writers.
//! 2. Deliberate allocation reuse: a driver writes the same `BufId` repeatedly as a plain
//!    *declared* output (or memcpy/memset destination), e.g. ping-ponging fold results between a
//!    layer buffer and a work buffer round after round.
//!
//! This pass rewrites the graph into true SSA at the `BufId` level:
//! every repeated write (carried or declared) gets a fresh `BufId`,
//! downstream reads are rerouted to the new id, and the fresh id is
//! recorded as an *alias* of the original in [`GraphBuilder::aliases`]
//! so the planner packs both onto the same pool slot at compile time.
//! The planner derives WAW/WAR/RAW precedence edges over *canonical*
//! (alias-class) ids in insertion order (see `PlanCtx::edges`), so the
//! renamed writes stay correctly ordered against every other user of
//! the slot.
//!
//! # Invariants after the pass
//!
//! - Every `KernelNode::carried_outputs` is empty.
//! - Every `BufId` is written by at most one node — SSA restored, ready for `verify_graph`.
//! - `GraphBuilder::aliases[b] = Some(root)` for every SSA-renamed buffer, where `root` is the
//!   original canonical `BufId`.
//! - Registered inputs and outputs are unchanged.
//! - Node insertion order is preserved.
//!
//! # Aliasing contract
//!
//! Two buffers with the same `canonical_buf(b)` MUST map to the same
//! pool offset. A mutating blackbox closure receives its input pointer
//! and writes to that address; if the compiler mapped the fresh SSA
//! version to a different pool slot the mutation would land on the old
//! address and downstream readers of the new id would read stale data.
//! The alias-aware planner enforces this identity.

use std::collections::{HashMap, HashSet};

use crate::{
    graph_ir::{BufId, GraphBuilder, GraphNode},
    CompileError,
};

/// Summary of what the pass did. `renamed_carried` counts fresh
/// `BufId`s introduced for previously-carried outputs;
/// `renamed_writes` counts fresh ids introduced for repeated declared
/// writes (kernel outputs, memcpy/memset destinations).
#[derive(Debug, Default, Clone, Copy)]
pub struct RestoreSsaReport {
    pub renamed_carried: usize,
    pub renamed_writes: usize,
    pub aliases_added: usize,
}

/// Rewrite the graph so every repeated write (blackbox mutation or
/// declared re-write of an already-written buffer) produces a fresh
/// `BufId`, then update every downstream read to the new id.
///
/// Idempotent: on an already-SSA graph the pass returns a zero report.
pub fn restore_ssa(g: &mut GraphBuilder) -> Result<RestoreSsaReport, CompileError> {
    if g.nodes.is_empty() {
        return Ok(RestoreSsaReport::default());
    }

    if g.aliases.len() < g.bufs.len() {
        g.aliases.resize(g.bufs.len(), None);
    }

    // `current[original_buf] = latest SSA version` — populated lazily.
    // Lookup falls back to the original when no rename has happened yet.
    // We only ever key by the ORIGINAL `BufId` the caller used at graph
    // build time; that keeps remaps deterministic and avoids over-
    // remapping intermediate versions that some later node may
    // legitimately be reading.
    //
    // A previous version of this pass also inserted a
    // `current[latest] = fresh` fallback, which caused correctness
    // failures at layer 1 of the fractional-GKR e2e test. That fallback
    // over-remaps nodes whose input is a legitimate SSA-clean read of
    // `latest`, redirecting them to a version that has not executed yet.
    let mut current: HashMap<BufId, BufId> = HashMap::new();
    // Buffers (by authored id) that have been written by some earlier
    // node — a later write to the same authored id needs a fresh
    // version.
    let mut written: HashSet<BufId> = HashSet::new();
    let mut report = RestoreSsaReport::default();

    let node_count = g.nodes.len();
    for idx in 0..node_count {
        let mut node = std::mem::replace(
            &mut g.nodes[idx],
            GraphNode::Memset(crate::graph_ir::MemSetNode {
                node: BufId(usize::MAX),
                val: 0,
                offset: crate::quast::Quast::cst(0),
                num_bytes: crate::quast::Quast::cst(0),
            }),
        );

        match &mut node {
            GraphNode::BlackboxKernel(k) => {
                // Remap reads through `current` first so the node reads
                // the pre-mutation identity of every input.
                remap_slice(&mut k.inputs, &current);
                for out in k.outputs.iter_mut() {
                    rewrite_write(out, &mut current, &mut written, g, &mut report);
                }
                // Carried buffers are in-place mutations: always version
                // them (identified by ORIGINAL BufId) and move them to
                // the output side.
                let carried = std::mem::take(&mut k.carried_outputs);
                for original in carried {
                    let fresh = new_version(original, &mut current, g, &mut report);
                    report.renamed_carried += 1;
                    written.insert(original);
                    k.outputs.push(fresh);
                }
            }
            GraphNode::Kernel(k) => {
                remap_slice(&mut k.inputs, &current);
                for out in k.outputs.iter_mut() {
                    rewrite_write(out, &mut current, &mut written, g, &mut report);
                }
            }
            GraphNode::Memcpy(m) => {
                remap_one(&mut m.src, &current);
                rewrite_write(&mut m.dst, &mut current, &mut written, g, &mut report);
            }
            GraphNode::Memset(m) => {
                rewrite_write(&mut m.node, &mut current, &mut written, g, &mut report);
            }
            // Const data is materialized at graph load, not in node
            // order — never version it, just record the write so a
            // later writer of the same buffer gets a fresh id.
            GraphNode::Const(c) => {
                written.insert(c.buf);
            }
        }

        g.nodes[idx] = node;
    }

    g.plan = None;
    Ok(report)
}

/// First write of an authored id keeps the id; any later write gets a
/// fresh aliased version and reroutes subsequent reads to it.
fn rewrite_write(
    buf_ref: &mut BufId,
    current: &mut HashMap<BufId, BufId>,
    written: &mut HashSet<BufId>,
    g: &mut GraphBuilder,
    report: &mut RestoreSsaReport,
) {
    let original = *buf_ref;
    if written.insert(original) {
        return;
    }
    let fresh = new_version(original, current, g, report);
    report.renamed_writes += 1;
    *buf_ref = fresh;
}

/// Allocate a fresh SSA version of `original`, alias it to the
/// canonical, and update `current` so subsequent nodes reading the
/// ORIGINAL BufId route to the new version.
fn new_version(
    original: BufId,
    current: &mut HashMap<BufId, BufId>,
    g: &mut GraphBuilder,
    report: &mut RestoreSsaReport,
) -> BufId {
    let latest = current.get(&original).copied().unwrap_or(original);
    let info = g.bufs[latest.0].clone();
    let fresh = g.add_buf(info);
    let root = g.canonical_buf(latest);
    g.alias_bufs(fresh, root);
    report.aliases_added += 1;
    current.insert(original, fresh);
    fresh
}

fn remap_one(buf: &mut BufId, current: &HashMap<BufId, BufId>) {
    if let Some(&renamed) = current.get(buf) {
        *buf = renamed;
    }
}

fn remap_slice(bufs: &mut [BufId], current: &HashMap<BufId, BufId>) {
    for b in bufs.iter_mut() {
        remap_one(b, current);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        graph_ir::{BufInfo, DeviceType, GraphNode, KernelNode},
        quast::Quast,
    };

    fn buf(g: &mut GraphBuilder, name: &str, size: i64) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(size),
            concrete_size: size as usize,
            elem_size: 4,
        })
    }

    fn mut_kernel(inputs: Vec<BufId>, outputs: Vec<BufId>, carried: Vec<BufId>) -> GraphNode {
        GraphNode::BlackboxKernel(KernelNode {
            inputs,
            outputs,
            carried_outputs: carried,
            func: Arc::new(|_, _, _| {}),
            name: "test_mut".into(),
        })
    }

    #[test]
    fn empty_graph_is_noop() {
        let mut g = GraphBuilder::new();
        let report = restore_ssa(&mut g).unwrap();
        assert_eq!(report.renamed_carried, 0);
    }

    #[test]
    fn single_mutation_chain_produces_alias_class() {
        let mut g = GraphBuilder::new();
        let leaves = buf(&mut g, "leaves", 128);
        let layer = buf(&mut g, "layer", 128);
        g.insert_memcpy(leaves, layer);
        g.nodes.push(mut_kernel(vec![layer], vec![], vec![layer]));
        g.nodes.push(mut_kernel(vec![layer], vec![], vec![layer]));
        g.nodes.push(mut_kernel(vec![layer], vec![], vec![]));

        let report = restore_ssa(&mut g).unwrap();
        assert_eq!(report.renamed_carried, 2);

        let (writers, _) = crate::graph_ir::classify_buf_uses(&g.nodes, g.bufs.len());
        for (b, ws) in writers.iter().enumerate() {
            assert!(
                ws.len() <= 1,
                "buffer {b:?} has {} writers: {:?}",
                ws.len(),
                ws
            );
        }

        let mut alias_class: Vec<BufId> = (0..g.bufs.len())
            .map(BufId)
            .filter(|&b| g.canonical_buf(b) == layer)
            .collect();
        alias_class.sort_by_key(|b| b.0);
        assert_eq!(alias_class.len(), 3, "root + 2 fresh renames");
        let last = alias_class.last().copied().unwrap();
        match &g.nodes[3] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs, vec![last], "reader must read the last version");
                assert!(k.carried_outputs.is_empty());
            }
            other => panic!("expected blackbox at index 3, got {other:?}"),
        }
    }

    /// Intermediate reader between two mutations must see the pre-second
    /// mutation version — NOT the post-second version. This is the
    /// regression the deleted `current[latest] = fresh` fallback caused.
    #[test]
    fn intermediate_reader_sees_correct_version() {
        let mut g = GraphBuilder::new();
        let x = buf(&mut g, "x", 64);
        let out = buf(&mut g, "out", 64);
        // k0 mutates x
        g.nodes.push(mut_kernel(vec![x], vec![], vec![x]));
        // reader reads x (post-k0, pre-k1)
        g.nodes.push(mut_kernel(vec![x], vec![out], vec![]));
        // k1 mutates x
        g.nodes.push(mut_kernel(vec![x], vec![], vec![x]));

        let report = restore_ssa(&mut g).unwrap();
        assert_eq!(report.renamed_carried, 2);

        // k0's fresh output is the version 1.
        let k0_v1 = match &g.nodes[0] {
            GraphNode::BlackboxKernel(k) => *k.outputs.last().unwrap(),
            _ => panic!(),
        };
        // Reader must read k0's version.
        match &g.nodes[1] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(
                    k.inputs[0], k0_v1,
                    "intermediate reader must see k0's output, not k1's"
                );
            }
            _ => panic!(),
        }
        // k1 must read k0's version (and produce a fresh v2).
        match &g.nodes[2] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs[0], k0_v1, "k1 must read k0's output");
                assert_eq!(k.outputs.len(), 1);
                assert_ne!(k.outputs[0], k0_v1);
            }
            _ => panic!(),
        }
    }

    /// Deliberate allocation reuse: the same buffer written twice as a
    /// plain declared output (ping-pong idiom). The second write must
    /// get a fresh aliased version; readers between the writes keep the
    /// first version, readers after get the second.
    #[test]
    fn repeated_declared_output_writes_renamed() {
        let mut g = GraphBuilder::new();
        let work = buf(&mut g, "work", 64);
        let a = buf(&mut g, "a", 64);
        let b = buf(&mut g, "b", 64);
        g.nodes.push(mut_kernel(vec![a], vec![work], vec![]));
        g.nodes.push(mut_kernel(vec![work], vec![b], vec![]));
        g.nodes.push(mut_kernel(vec![b], vec![work], vec![]));
        g.nodes.push(mut_kernel(vec![work], vec![], vec![]));

        let report = restore_ssa(&mut g).unwrap();
        assert_eq!(report.renamed_carried, 0);
        assert_eq!(report.renamed_writes, 1);

        let (writers, _) = crate::graph_ir::classify_buf_uses(&g.nodes, g.bufs.len());
        for (b, ws) in writers.iter().enumerate() {
            assert!(ws.len() <= 1, "buffer {b} has {} writers", ws.len());
        }
        let v1 = match &g.nodes[2] {
            GraphNode::BlackboxKernel(k) => k.outputs[0],
            _ => panic!(),
        };
        assert_ne!(v1, work);
        assert_eq!(g.canonical_buf(v1), work);
        match &g.nodes[1] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs, vec![work], "intermediate reader keeps v0");
            }
            _ => panic!(),
        }
        match &g.nodes[3] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs, vec![v1], "later reader rerouted to v1");
            }
            _ => panic!(),
        }
    }

    /// A memcpy destination re-writing an already-written buffer gets a
    /// fresh aliased version, and later readers are rerouted to it.
    #[test]
    fn memcpy_dst_rewrite_renamed() {
        let mut g = GraphBuilder::new();
        let src = buf(&mut g, "src", 64);
        let layer = buf(&mut g, "layer", 64);
        g.nodes.push(mut_kernel(vec![], vec![layer], vec![]));
        g.nodes.push(mut_kernel(vec![layer], vec![src], vec![]));
        g.insert_memcpy(src, layer);
        g.nodes.push(mut_kernel(vec![layer], vec![], vec![]));

        let report = restore_ssa(&mut g).unwrap();
        assert_eq!(report.renamed_writes, 1);

        let v1 = match &g.nodes[2] {
            GraphNode::Memcpy(m) => m.dst,
            _ => panic!(),
        };
        assert_ne!(v1, layer);
        assert_eq!(g.canonical_buf(v1), layer);
        match &g.nodes[1] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs, vec![layer], "pre-copy reader keeps v0");
            }
            _ => panic!(),
        }
        match &g.nodes[3] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs, vec![v1], "post-copy reader rerouted to v1");
            }
            _ => panic!(),
        }
    }

    #[test]
    fn independent_buffers_stay_in_separate_classes() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 64);
        let b = buf(&mut g, "b", 64);
        g.nodes.push(mut_kernel(vec![a], vec![], vec![a]));
        g.nodes.push(mut_kernel(vec![b], vec![], vec![b]));

        let report = restore_ssa(&mut g).unwrap();
        assert_eq!(report.renamed_carried, 2);

        assert_eq!(g.canonical_buf(a), a);
        assert_eq!(g.canonical_buf(b), b);
    }

    #[test]
    fn idempotent_when_no_carried_outputs() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 64);
        let b = buf(&mut g, "b", 64);
        g.nodes.push(mut_kernel(vec![], vec![a], vec![]));
        g.nodes.push(mut_kernel(vec![a], vec![b], vec![]));

        let n_bufs_before = g.bufs.len();
        let report1 = restore_ssa(&mut g).unwrap();
        let n_bufs_mid = g.bufs.len();
        let report2 = restore_ssa(&mut g).unwrap();
        assert_eq!(report1.renamed_carried, 0);
        assert_eq!(report2.renamed_carried, 0);
        assert_eq!(n_bufs_before, n_bufs_mid);
        assert_eq!(g.bufs.len(), n_bufs_mid);
    }
}
