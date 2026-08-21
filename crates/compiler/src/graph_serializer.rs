//! Serializable snapshot of a [`GraphBuilder`].
//!
//! The snapshot is a full copy of the source graph — buffers, structured
//! [`crate::ir::Module`] payloads, memcpy / memset / const wiring,
//! blackbox kernel *shape* — with two carve-outs:
//!
//! 1. Blackbox `KernelFn` closures are omitted (they aren't serializable). A caller who wants to
//!    reconstruct the graph supplies an *existing* [`GraphBuilder`] to
//!    [`SerializableGraphBuilder::into_graph_builder`]; the closures are copied from that builder
//!    by node index.
//! 2. Constant-buffer bytes always land on the host side of the payload. A `ConstBuf::DeviceBuf`
//!    payload is streamed D2H on serialize and H2D on load, so callers hand a [`GpuDeviceCtx`] in
//!    both directions.
//!
//! # Surviving fusion
//!
//! The `existing` builder supplied on load may have been through
//! `compile()` (fusion / DCE / `restore_ssa`), so its post-pass content
//! hash differs from the snapshot's. Both sides therefore compare
//! [`GraphBuilder::original_hash`] — the pre-pass fingerprint cached the
//! first time `original_hash` is queried (before any pass runs).
//! Compile snapshots it on entry, so a builder that was compiled in
//! place still exposes its original fingerprint afterwards.
//!
//! Blackbox nodes' *positions* in the node list stay stable across
//! passes (fusion never removes or reorders them), which is what lets us
//! match a snapshot's blackbox at node index `i` to the existing
//! builder's blackbox at node index `i`. As a defensive check
//! [`SerializableGraphBuilder::into_graph_builder`] also verifies both
//! sides list identical `input_bufs` / `output_bufs` and identical
//! [`BufInfo`]s.

use std::{collections::BTreeMap, sync::Arc};

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    stream::GpuDeviceCtx,
};
use serde::{Deserialize, Serialize};

use crate::{
    graph_ir::{
        BufId, BufInfo, ConstBuf, ConstNode, GraphBuilder, GraphNode, KernelFn, KernelModuleNode,
        KernelNode, MemSetNode, MemcpyNode,
    },
    ir::VarId,
    CompileError,
};

/// Serialized form of a [`GraphBuilder`]. Round-trips through `bincode`
/// (or any `serde` format); a subsequent [`Self::into_graph_builder`]
/// yields a builder ready to hand to
/// [`crate::graph_compiler::GraphCompiler::compile`].
#[derive(Serialize, Deserialize)]
pub struct SerializableGraphBuilder {
    /// Pre-pass content hash captured from the source builder. Compared
    /// against the `existing` builder's `original_hash` on load.
    original_hash: [u8; 32],
    bufs: Vec<BufInfo>,
    symbols: BTreeMap<VarId, String>,
    next_var: u32,
    input_bufs: Vec<BufId>,
    output_bufs: Vec<BufId>,
    aliases: Vec<Option<BufId>>,
    nodes: Vec<SerializedGraphNode>,
}

#[derive(Serialize, Deserialize)]
enum SerializedGraphNode {
    Kernel(KernelModuleNode),
    /// Blackbox node without the closure. On load we re-attach the
    /// `func` from the caller-supplied `existing` builder at the same
    /// node index.
    BlackboxKernel(SerializedBlackbox),
    Const(SerializedConst),
    Memcpy(MemcpyNode),
    Memset(MemSetNode),
}

#[derive(Serialize, Deserialize)]
struct SerializedBlackbox {
    name: String,
    inputs: Vec<BufId>,
    outputs: Vec<BufId>,
    carried_outputs: Vec<BufId>,
}

#[derive(Serialize, Deserialize)]
struct SerializedConst {
    buf: BufId,
    /// Where the reconstructed [`ConstBuf`] should live: `true` means
    /// re-upload onto the GPU as [`ConstBuf::DeviceBuf`]; `false` means
    /// keep on the host as [`ConstBuf::HostBuf`].
    on_device: bool,
    bytes: Vec<u8>,
}

impl SerializableGraphBuilder {
    /// Snapshot a live [`GraphBuilder`]. Uses the builder's cached
    /// `original_hash` when present, else computes and returns the
    /// current `content_hash` (without caching it — the source builder
    /// stays `&`-borrowed). Callers who want the cache populated should
    /// call [`GraphBuilder::original_hash`] on the source builder before
    /// invoking this.
    ///
    /// `ctx` is only consulted for [`ConstBuf::DeviceBuf`] payloads
    /// (each is copied D2H into the snapshot). Passing a placeholder
    /// context is fine when the graph has no device-side constants.
    pub fn from_graph_builder(
        builder: &GraphBuilder,
        ctx: &GpuDeviceCtx,
    ) -> Result<Self, CompileError> {
        let original_hash = builder
            .original_hash
            .unwrap_or_else(|| builder.content_hash());
        let mut nodes = Vec::with_capacity(builder.nodes.len());
        for node in &builder.nodes {
            nodes.push(serialize_node(node, ctx)?);
        }
        Ok(Self {
            original_hash,
            bufs: builder.bufs.clone(),
            symbols: builder.symbols.clone(),
            next_var: builder.next_var(),
            input_bufs: builder.input_bufs().to_vec(),
            output_bufs: builder.output_bufs().to_vec(),
            aliases: builder.aliases.clone(),
            nodes,
        })
    }

    /// Pre-pass content hash of the source builder (see
    /// [`GraphBuilder::original_hash`]).
    pub fn original_hash(&self) -> &[u8; 32] {
        &self.original_hash
    }

    /// Whether the snapshot contains any blackbox nodes (i.e. whether
    /// [`Self::into_graph_builder`] will need a non-`None` `existing`).
    pub fn has_blackboxes(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n, SerializedGraphNode::BlackboxKernel(_)))
    }

    /// Reconstruct a [`GraphBuilder`] *without* touching the GPU: any
    /// `Const` node that was originally a `DeviceBuf` becomes a
    /// `HostBuf` of the same bytes, and every blackbox reconstructs
    /// with a panic-on-invoke placeholder closure. Suitable for
    /// analysis pipelines that only consume the graph's *shape*
    /// (buffer table, node topology, aliases) without ever running
    /// the exe — e.g. the abstract-timing scheduler test harness in
    /// [`crate::planner::abstract_timing`].
    pub fn into_graph_builder_offline(self) -> GraphBuilder {
        let Self {
            original_hash,
            bufs,
            symbols,
            next_var,
            input_bufs,
            output_bufs,
            aliases,
            nodes: ser_nodes,
        } = self;
        let nodes = ser_nodes
            .into_iter()
            .enumerate()
            .map(|(i, ser)| deserialize_node_offline(ser, i))
            .collect();
        GraphBuilder::from_serialized_parts(
            bufs,
            nodes,
            symbols,
            next_var,
            input_bufs,
            output_bufs,
            aliases,
            Some(original_hash),
        )
    }

    /// Reconstruct a [`GraphBuilder`]. When the snapshot contains any
    /// blackbox nodes, `existing` must be `Some` and its
    /// [`GraphBuilder::original_hash`] must match — the closures come
    /// from there.
    ///
    /// `ctx` is only consulted for `Const` nodes that were on the GPU
    /// at snapshot time; their bytes are re-uploaded via `ctx`.
    ///
    /// # Panics
    /// - `existing.original_hash()` differs from `self.original_hash`.
    /// - `existing`'s registered inputs / outputs, or their `BufInfo`s, differ from the snapshot's.
    /// - `existing.nodes[i]` isn't a blackbox where the snapshot has a blackbox at index `i` (would
    ///   indicate the "fusion preserves blackbox indices" invariant was violated).
    pub fn into_graph_builder(
        self,
        mut existing: Option<&mut GraphBuilder>,
        ctx: &GpuDeviceCtx,
    ) -> Result<GraphBuilder, CompileError> {
        if let Some(e) = existing.as_deref_mut() {
            self.check_matches_existing(e);
        }
        // When `existing` is `None` and the snapshot has blackboxes, each
        // reconstructed blackbox gets a *placeholder* closure that
        // panics if ever invoked. That's fine for read-only paths
        // (`GraphBuilder::print`, `to_cytoscape_json`, offline
        // inspection); anything that dispatches the graph (compile +
        // run, `capture_graph`, `collect_graph_info`) will trip the
        // placeholder. Callers that need a runnable builder must supply
        // an `existing` builder whose blackbox positions carry the
        // real closures.

        let Self {
            original_hash,
            bufs,
            symbols,
            next_var,
            input_bufs,
            output_bufs,
            aliases,
            nodes: ser_nodes,
        } = self;

        let existing_ref = existing.as_deref();
        let mut nodes = Vec::with_capacity(ser_nodes.len());
        for (i, ser) in ser_nodes.into_iter().enumerate() {
            nodes.push(deserialize_node(ser, i, existing_ref, ctx)?);
        }

        Ok(GraphBuilder::from_serialized_parts(
            bufs,
            nodes,
            symbols,
            next_var,
            input_bufs,
            output_bufs,
            aliases,
            Some(original_hash),
        ))
    }

    fn check_matches_existing(&self, existing: &mut GraphBuilder) {
        let existing_hash = existing.original_hash();
        if existing_hash != self.original_hash {
            panic!(
                "SerializableGraphBuilder: existing.original_hash() {} does not match snapshot \
                 (does not match snapshot.original_hash {}); the supplied builder does not \
                 correspond to this snapshot",
                hex::encode(existing_hash),
                hex::encode(self.original_hash),
            );
        }
        if existing.input_bufs() != self.input_bufs.as_slice() {
            panic!(
                "SerializableGraphBuilder: existing.input_bufs {:?} does not match snapshot \
                 input_bufs {:?}",
                existing.input_bufs(),
                self.input_bufs,
            );
        }
        if existing.output_bufs() != self.output_bufs.as_slice() {
            panic!(
                "SerializableGraphBuilder: existing.output_bufs {:?} does not match snapshot \
                 output_bufs {:?}",
                existing.output_bufs(),
                self.output_bufs,
            );
        }
        if existing.bufs.len() != self.bufs.len() {
            panic!(
                "SerializableGraphBuilder: existing.bufs has {} entries, snapshot has {}",
                existing.bufs.len(),
                self.bufs.len(),
            );
        }
        for (i, want) in self.bufs.iter().enumerate() {
            let have = &existing.bufs[i];
            if !buf_info_matches(want, have) {
                panic!(
                    "SerializableGraphBuilder: BufInfo mismatch at BufId({i}): snapshot {want:?} \
                     vs existing {have:?}",
                );
            }
        }
    }
}

fn buf_info_matches(a: &BufInfo, b: &BufInfo) -> bool {
    a.name == b.name
        && a.device_type == b.device_type
        && a.elem_size == b.elem_size
        && a.size == b.size
}

fn serialize_node(
    node: &GraphNode,
    ctx: &GpuDeviceCtx,
) -> Result<SerializedGraphNode, CompileError> {
    Ok(match node {
        GraphNode::Kernel(k) => SerializedGraphNode::Kernel(KernelModuleNode {
            module: Arc::clone(&k.module),
            param_bindings: k.param_bindings.clone(),
            inputs: k.inputs.clone(),
            outputs: k.outputs.clone(),
            types: None,
            hash: k.hash,
            canonical: k.canonical,
            fusion_history: None,
        }),
        GraphNode::BlackboxKernel(k) => SerializedGraphNode::BlackboxKernel(SerializedBlackbox {
            name: k.name.clone(),
            inputs: k.inputs.clone(),
            outputs: k.outputs.clone(),
            carried_outputs: k.carried_outputs.clone(),
        }),
        GraphNode::Const(c) => SerializedGraphNode::Const(serialize_const(c, ctx)?),
        GraphNode::Memcpy(m) => SerializedGraphNode::Memcpy(m.clone()),
        GraphNode::Memset(m) => SerializedGraphNode::Memset(m.clone()),
    })
}

fn serialize_const(c: &ConstNode, ctx: &GpuDeviceCtx) -> Result<SerializedConst, CompileError> {
    let (on_device, bytes) = match &c.data {
        ConstBuf::HostBuf(v) => (false, v.clone()),
        ConstBuf::DeviceBuf(d) => {
            let host = d.to_host_on(ctx).map_err(|e| {
                CompileError::Runtime(format!(
                    "graph_serializer: const D2H copy failed for buf {:?}: {e:?}",
                    c.buf
                ))
            })?;
            (true, host)
        }
    };
    Ok(SerializedConst {
        buf: c.buf,
        on_device,
        bytes,
    })
}

fn deserialize_node(
    ser: SerializedGraphNode,
    node_idx: usize,
    existing: Option<&GraphBuilder>,
    ctx: &GpuDeviceCtx,
) -> Result<GraphNode, CompileError> {
    Ok(match ser {
        SerializedGraphNode::Kernel(k) => GraphNode::Kernel(k),
        SerializedGraphNode::BlackboxKernel(bb) => {
            let func = match existing {
                Some(existing) => take_blackbox_func(existing, node_idx, &bb.name),
                None => placeholder_blackbox_func(&bb.name),
            };
            GraphNode::BlackboxKernel(KernelNode {
                inputs: bb.inputs,
                outputs: bb.outputs,
                carried_outputs: bb.carried_outputs,
                func,
                name: bb.name,
            })
        }
        SerializedGraphNode::Const(c) => {
            let SerializedConst {
                buf,
                on_device,
                bytes,
            } = c;
            let data = if on_device {
                let d = bytes.as_slice().to_device_on(ctx).map_err(|e| {
                    CompileError::Runtime(format!(
                        "graph_serializer: const H2D copy failed for buf {buf:?}: {e:?}",
                    ))
                })?;
                ConstBuf::DeviceBuf(d)
            } else {
                ConstBuf::HostBuf(bytes)
            };
            GraphNode::Const(ConstNode { buf, data })
        }
        SerializedGraphNode::Memcpy(m) => GraphNode::Memcpy(m),
        SerializedGraphNode::Memset(m) => GraphNode::Memset(m),
    })
}

/// Ctx-free variant of [`deserialize_node`]. `Const` nodes always come
/// back as `HostBuf` (no H2D upload); blackboxes get placeholder
/// closures. Used by [`SerializableGraphBuilder::into_graph_builder_offline`].
fn deserialize_node_offline(ser: SerializedGraphNode, _node_idx: usize) -> GraphNode {
    match ser {
        SerializedGraphNode::Kernel(k) => GraphNode::Kernel(k),
        SerializedGraphNode::BlackboxKernel(bb) => GraphNode::BlackboxKernel(KernelNode {
            inputs: bb.inputs,
            outputs: bb.outputs,
            carried_outputs: bb.carried_outputs,
            func: placeholder_blackbox_func(&bb.name),
            name: bb.name,
        }),
        SerializedGraphNode::Const(c) => GraphNode::Const(ConstNode {
            buf: c.buf,
            // Ignore the `on_device` flag: the bytes stay on the host,
            // since the caller is doing offline analysis and won't
            // dispatch this graph.
            data: ConstBuf::HostBuf(c.bytes),
        }),
        SerializedGraphNode::Memcpy(m) => GraphNode::Memcpy(m),
        SerializedGraphNode::Memset(m) => GraphNode::Memset(m),
    }
}

/// A closure that always panics with a descriptive message. Handed to
/// blackbox nodes reconstructed via `into_graph_builder(None, _)` so
/// read-only paths (printing, cytoscape emission) work without a source
/// builder while any attempt to dispatch the graph fails loudly.
fn placeholder_blackbox_func(name: &str) -> KernelFn {
    let name = name.to_string();
    Arc::new(move |_ins: &[*mut ()], _outs: &[*mut ()], _s| {
        panic!(
            "graph_serializer: blackbox `{name}` was reconstructed with a placeholder closure \
             (existing=None on into_graph_builder); rebuild the graph with a matching source \
             GraphBuilder to obtain a runnable builder"
        )
    })
}

/// Copy the `KernelFn` `Arc` for the blackbox at `existing.nodes[i]`.
///
/// # Panics
/// - `existing.nodes[i]` is out of bounds.
/// - `existing.nodes[i]` isn't a `BlackboxKernel`.
/// - Its `name` doesn't match `expected_name`.
fn take_blackbox_func(existing: &GraphBuilder, i: usize, expected_name: &str) -> KernelFn {
    let node = existing.nodes.get(i).unwrap_or_else(|| {
        panic!("into_graph_builder: existing builder is shorter than the snapshot at node {i}",)
    });
    let GraphNode::BlackboxKernel(k) = node else {
        panic!(
            "into_graph_builder: existing.nodes[{i}] is not a blackbox (variant: {node:?}); \
             fusion is expected to preserve blackbox positions but the existing builder does \
             not honor that invariant",
        );
    };
    if k.name != expected_name {
        panic!(
            "into_graph_builder: blackbox name mismatch at index {i}: snapshot expected \
             `{expected_name}`, existing has `{}`",
            k.name,
        );
    }
    Arc::clone(&k.func)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_ir::{BufInfo, DeviceType, GraphNode, KernelNode, MemSetNode, MemcpyNode},
        quast::Quast,
    };

    fn add_dev_buf(g: &mut GraphBuilder, name: &str, bytes: usize) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(bytes as i64),
            concrete_size: bytes,
            elem_size: 4,
        })
    }

    fn noop_kernel_fn() -> KernelFn {
        Arc::new(|_ins: &[*mut ()], _outs: &[*mut ()], _s| {})
    }

    fn make_ser(g: &mut GraphBuilder) -> SerializableGraphBuilder {
        let original_hash = g.original_hash();
        let mut nodes = Vec::with_capacity(g.nodes.len());
        for node in &g.nodes {
            nodes.push(match node {
                GraphNode::Memcpy(m) => SerializedGraphNode::Memcpy(m.clone()),
                GraphNode::Memset(m) => SerializedGraphNode::Memset(m.clone()),
                GraphNode::BlackboxKernel(k) => {
                    SerializedGraphNode::BlackboxKernel(SerializedBlackbox {
                        name: k.name.clone(),
                        inputs: k.inputs.clone(),
                        outputs: k.outputs.clone(),
                        carried_outputs: k.carried_outputs.clone(),
                    })
                }
                _ => unreachable!("only memcpy/memset/blackbox used in this test"),
            });
        }
        SerializableGraphBuilder {
            original_hash,
            bufs: g.bufs.clone(),
            symbols: g.symbols.clone(),
            next_var: g.next_var(),
            input_bufs: g.input_bufs().to_vec(),
            output_bufs: g.output_bufs().to_vec(),
            aliases: g.aliases.clone(),
            nodes,
        }
    }

    #[test]
    fn original_hash_caches_first_content_hash() {
        let mut g = GraphBuilder::new();
        let _a = add_dev_buf(&mut g, "a", 16);
        let h1 = g.original_hash();
        // Mutate the graph. The cached original_hash shouldn't change.
        let _b = add_dev_buf(&mut g, "b", 32);
        let h2 = g.original_hash();
        assert_eq!(h1, h2);
        // But content_hash *should* reflect the mutation.
        assert_ne!(g.content_hash(), h2);
    }

    #[test]
    fn memcpy_memset_round_trip_via_bincode() {
        let mut g = GraphBuilder::new();
        let a = add_dev_buf(&mut g, "a", 16);
        let b = add_dev_buf(&mut g, "b", 16);
        g.register_input(a);
        g.register_output(b);
        g.nodes.push(GraphNode::Memcpy(MemcpyNode {
            src: a,
            src_offset: Quast::cst(0),
            dst: b,
            dst_offset: Quast::cst(0),
            num_bytes: Quast::cst(16),
        }));
        g.nodes.push(GraphNode::Memset(MemSetNode {
            node: b,
            offset: Quast::cst(0),
            num_bytes: Quast::cst(16),
            val: 0,
        }));
        let ser = make_ser(&mut g);
        let bytes = bincode::serialize(&ser).unwrap();
        let round: SerializableGraphBuilder = bincode::deserialize(&bytes).unwrap();
        assert_eq!(round.original_hash, *ser.original_hash());
        assert_eq!(round.nodes.len(), 2);
        assert!(!round.has_blackboxes());
    }

    #[test]
    #[should_panic(expected = "does not match snapshot")]
    fn check_matches_existing_panics_on_hash_mismatch() {
        let mut g1 = GraphBuilder::new();
        let _a = add_dev_buf(&mut g1, "a", 16);
        let mut g2 = GraphBuilder::new();
        let _a = add_dev_buf(&mut g2, "a", 32);
        let ser = make_ser(&mut g1);
        ser.check_matches_existing(&mut g2);
    }

    #[test]
    fn check_matches_existing_accepts_identical_builders() {
        // Two builders constructed identically must have equal
        // original_hash and pass the sanity check.
        let build = |func: KernelFn| -> GraphBuilder {
            let mut g = GraphBuilder::new();
            let a = add_dev_buf(&mut g, "a", 16);
            let b = add_dev_buf(&mut g, "b", 16);
            g.register_input(a);
            g.register_output(b);
            g.nodes.push(GraphNode::BlackboxKernel(KernelNode {
                inputs: vec![a],
                outputs: vec![b],
                carried_outputs: vec![],
                func,
                name: "bb".into(),
            }));
            g
        };
        let mut g1 = build(noop_kernel_fn());
        let mut g2 = build(noop_kernel_fn());
        assert_eq!(g1.original_hash(), g2.original_hash());
        let ser = make_ser(&mut g1);
        ser.check_matches_existing(&mut g2);
        assert!(ser.has_blackboxes());
    }
}
