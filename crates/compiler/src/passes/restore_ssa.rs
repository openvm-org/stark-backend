//! Restore SSA form for blackbox in-place mutations.
//!
//! `insert_blackbox_kernel(inputs, outputs, modifies)` lets a caller
//! declare that some inputs are mutated in place — those buffers wind
//! up in `KernelNode::carried_outputs` and share their `BufId` with the
//! input side. This is convenient for callers, but `verify_graph`'s
//! single-writer / SSA check rejects chains of mutations outright
//! because each mutating kernel adds itself to the buffer's writer list.
//!
//! This pass rewrites the graph into true SSA at the `BufId` level:
//! every mutating blackbox kernel gets a fresh `BufId` for its
//! (previously carried) output, downstream reads are rerouted to the
//! new id, and the fresh id is recorded as an *alias* of the original
//! in [`GraphBuilder::aliases`] so the planner packs both onto the same
//! pool slot at compile time.
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

use std::collections::HashMap;

use crate::{
    graph_ir::{BufId, GraphBuilder, GraphNode, KernelNode},
    CompileError,
};

/// Summary of what the pass did. `renamed_carried` counts fresh
/// `BufId`s introduced for previously-carried outputs.
#[derive(Debug, Default, Clone, Copy)]
pub struct RestoreSsaReport {
    pub renamed_carried: usize,
    pub aliases_added: usize,
}

/// Rewrite the graph so every blackbox mutation produces a fresh
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
    let mut current: HashMap<BufId, BufId> = HashMap::new();
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
                rewrite_blackbox(k, &mut current, g, &mut report);
            }
            GraphNode::Kernel(k) => {
                remap_slice(&mut k.inputs, &current);
            }
            GraphNode::Memcpy(m) => {
                remap_one(&mut m.src, &current);
            }
            GraphNode::Memset(m) => {
                // Memset over an already-mutated buffer is a fresh write
                // of the canonical — allocate a fresh SSA version and
                // alias it.
                if let Some(&renamed) = current.get(&m.node) {
                    let root = g.canonical_buf(renamed);
                    let info = g.bufs[m.node.0].clone();
                    let fresh = g.add_buf(info);
                    g.alias_bufs(fresh, root);
                    report.aliases_added += 1;
                    current.insert(m.node, fresh);
                    m.node = fresh;
                }
            }
            GraphNode::Const(_) => {}
        }

        g.nodes[idx] = node;
    }

    g.plan = None;
    Ok(report)
}

/// Rewrite a single blackbox kernel node.
fn rewrite_blackbox(
    k: &mut KernelNode,
    current: &mut HashMap<BufId, BufId>,
    g: &mut GraphBuilder,
    report: &mut RestoreSsaReport,
) {
    let carried = std::mem::take(&mut k.carried_outputs);

    // Remap reads through `current` first so the mutation reads the
    // pre-mutation identity of every input.
    for input in k.inputs.iter_mut() {
        if let Some(&latest) = current.get(input) {
            *input = latest;
        }
    }

    // For every carried buffer (identified by ORIGINAL BufId), allocate
    // a fresh SSA output, alias it to the canonical, and update
    // `current` so subsequent nodes reading the ORIGINAL BufId route
    // to the new version.
    //
    // We deliberately key `current` ONLY by the original — a previous
    // version of this pass also inserted a `current[latest] = fresh`
    // fallback, which caused correctness failures at layer 1 of the
    // fractional-GKR e2e test. That fallback over-remaps nodes whose
    // input is a legitimate SSA-clean read of `latest`, redirecting
    // them to a version that has not executed yet.
    for original in carried {
        let latest = current.get(&original).copied().unwrap_or(original);
        let info = g.bufs[latest.0].clone();
        let fresh = g.add_buf(info);
        let root = g.canonical_buf(latest);
        g.alias_bufs(fresh, root);
        report.aliases_added += 1;
        report.renamed_carried += 1;

        current.insert(original, fresh);
        k.outputs.push(fresh);
    }
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
