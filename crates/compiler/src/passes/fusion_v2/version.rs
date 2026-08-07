//! Converts a [`GraphBuilder`] into a versioned seed alternative graph.
//!
//! `detailed-fusion-plan-v2.md` §6.
//!
//! The scan iterates the moved [`GraphNode`]s in insertion order, resolves
//! every operand against the current visible version of its physical
//! [`BufId`], and appends new logical value classes for every write. Each
//! node produces exactly one [`AltGraphNode`] with the positional bindings
//! defined by [`GraphNode::get_operands`]/[`GraphNode::get_results`].

use thiserror::Error;

use crate::{
    graph_ir::{BufId, GraphBuilder},
    passes::fusion_v2::model::{AltGraphNode, GraphFuser, NodeId, UseInfo, ValueClassId},
};

/// Failure modes of [`take_graph`]. Every variant is a hard structural
/// error: none of them should reach the pass entry after canonicalization
/// and the existing DCE.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TakeGraphError {
    /// A read of a physical buffer that has not been written yet and is not
    /// a registered graph input.
    #[error(
        "read-before-write: node {node} reads buffer {buf} which has no producer and is not a \
         registered graph input"
    )]
    ReadBeforeWrite { node: usize, buf: usize },
    /// A single node writes the same physical buffer at more than one
    /// result position. MVP does not define an ordering for this case
    /// (§6.2, last paragraph).
    #[error(
        "duplicate writes: node {node} writes buffer {buf} at result positions {first} and \
         {second}"
    )]
    DuplicateWrite {
        node: usize,
        buf: usize,
        first: usize,
        second: usize,
    },
}

/// Moves every [`GraphNode`] out of `g` and builds the versioned seed
/// alternative graph. `g.nodes` is emptied but `g.bufs`, `g.symbols`, and
/// the registered interface remain intact so the caller can restore the
/// original graph via [`crate::passes::fusion_v2::apply::apply_solution`]
/// (or its take-graph guard on error).
pub fn take_graph(g: &mut GraphBuilder) -> Result<GraphFuser, TakeGraphError> {
    let n_bufs = g.bufs.len();
    let mut gf = GraphFuser {
        // Seed one value class per original physical buffer; index equals
        // BufId, so `ValueClassId(b.0)` is the first instance of `BufId(b.0)`.
        bufs: g.bufs.clone(),
        nodes: Vec::new(),
        producers: vec![Vec::new(); n_bufs],
        consumers: vec![Vec::new(); n_bufs],
        access_relations: Vec::new(),
        re_exported: (0..n_bufs).map(ValueClassId).collect(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        seed_node_count: 0,
    };

    // Only registered inputs start with a visible current version.
    let mut current: Vec<Option<ValueClassId>> = vec![None; n_bufs];
    for &b in g.input_bufs() {
        current[b.0] = Some(ValueClassId(b.0));
        gf.inputs.push(ValueClassId(b.0));
    }

    // Move nodes out of the builder so ownership transfers into the fuser.
    let seed_nodes = std::mem::take(&mut g.nodes);
    for (idx, node) in seed_nodes.into_iter().enumerate() {
        let operand_bufs = node.get_operands(&gf.bufs);
        let result_bufs = node.get_results();
        // Reject duplicate writes to the same physical buffer within one
        // node — MVP has no ordering rule for that case.
        for (i, a) in result_bufs.iter().enumerate() {
            if let Some(j) = result_bufs[..i].iter().position(|b| b == a) {
                return Err(TakeGraphError::DuplicateWrite {
                    node: idx,
                    buf: a.0,
                    first: j,
                    second: i,
                });
            }
        }
        // Resolve inputs before allocating outputs so a same-BufId in/out
        // binds the old version to the input and the new version to the
        // output.
        let mut inputs = Vec::with_capacity(operand_bufs.len());
        for &b in &operand_bufs {
            let v = current[b.0].ok_or(TakeGraphError::ReadBeforeWrite {
                node: idx,
                buf: b.0,
            })?;
            inputs.push(v);
        }
        // Allocate one new value class per result BufId. Duplicate BufIds
        // among results were already rejected above, so this is
        // straightforward.
        let mut outputs = Vec::with_capacity(result_bufs.len());
        for &b in &result_bufs {
            let new_val = alloc_new_version(&mut gf, b);
            outputs.push(new_val);
        }
        // Publish the new versions as the visible current values only after
        // the inputs have been resolved.
        for &b in &result_bufs {
            // The last-allocated value with this BufId in this node is the
            // current version.
            for &v in outputs.iter().rev() {
                if gf.physical(v) == b {
                    current[b.0] = Some(v);
                    break;
                }
            }
        }
        // Register the node itself.
        let node_id = NodeId(gf.nodes.len());
        for (pos, &v) in inputs.iter().enumerate() {
            gf.consumers[v.0].push(UseInfo { node: node_id, pos });
        }
        for (pos, &v) in outputs.iter().enumerate() {
            gf.producers[v.0].push(UseInfo { node: node_id, pos });
        }
        gf.nodes.push(AltGraphNode {
            inputs,
            outputs,
            node,
        });
        // Access-relation extraction is deferred to M0-part 2; keep the
        // sidecar in lockstep with `nodes`.
        gf.access_relations.push(None);
    }
    gf.seed_node_count = gf.nodes.len();

    // Registered graph outputs resolve to their final visible version.
    for &b in g.output_bufs() {
        // Registered outputs are required to have a producer by the graph
        // interface's own validation. If a caller has registered an
        // input-passthrough output, that value class exists too (the
        // graph input was seeded with `ValueClassId(b.0)`).
        let v = current[b.0].expect(
            "registered graph output has no producer — validated by graph_exe before v2 runs",
        );
        gf.outputs.push(v);
    }

    Ok(gf)
}

/// Appends one new logical value class for a write of physical `buf`,
/// mirroring the buffer's metadata and pointing `re_exported` at the
/// first-instance value class (i.e. `ValueClassId(buf.0)`).
fn alloc_new_version(gf: &mut GraphFuser, buf: BufId) -> ValueClassId {
    let new_val = ValueClassId(gf.bufs.len());
    gf.bufs.push(gf.bufs[buf.0].clone());
    gf.producers.push(Vec::new());
    gf.consumers.push(Vec::new());
    gf.re_exported.push(ValueClassId(buf.0));
    new_val
}
