//! Graph IR: a raw computation graph over a GPU device.
//!
//! Nodes are kernel launches, memcpys and memsets operating on buffers
//! identified by [`BufId`]; buffers carry their device placement and sizes
//! in [`BufInfo`]. Buffer sizes are symbolic [`Quast`] expressions over
//! variables registered with [`GraphBuilder::register_symbol`]; concrete
//! byte counts are recovered at execution time by [`Quast::eval`] against
//! a binding of each symbol.
//!
//! Kernel launches come in two flavors: [`GraphNode::BlackboxKernel`] wraps
//! an opaque host closure ([`KernelNode`]) that receives raw input/output
//! pointers, and [`GraphNode::Kernel`] pairs a structured high-level
//! [`ir::Module`] with explicit input/output [`BufId`] bindings
//! ([`KernelModuleNode`]) so downstream passes know which graph buffers feed
//! which module inputs. Static data lives in [`GraphNode::Const`], which
//! carries either device or host bytes. See `graph-ir.md`.

use std::{collections::BTreeMap, fmt, rc::Rc};

use openvm_cuda_common::{d_buffer::DeviceBuffer, stream::cudaStream_t};

use crate::{
    ir::{self, VarId},
    quast::Quast,
};

/// Index of a buffer in the graph's buffer table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufId(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceType {
    /// CUDA device with the given ordinal.
    Cuda(usize),
    CpuPinned,
    CpuPaged,
}

#[derive(Clone, Debug)]
pub struct BufInfo {
    pub name: Option<String>,
    pub device_type: DeviceType,
    /// Symbolic size in bytes.
    pub size: Quast,
    pub elem_size: usize,
}

/// Type-erased kernel launch: receives raw pointers to the input buffers,
/// to the output buffers, and the CUDA stream to launch on. The closure
/// should launch its work *asynchronously* on `stream` (no in-closure
/// synchronization); [`crate::graph_exe::GraphExe::run`] issues launches in
/// planner-chosen order on that same stream, so intra-graph dependencies are
/// enforced by stream ordering.
pub type KernelFn = Box<dyn Fn(&[*mut ()], &[*mut ()], cudaStream_t)>;

pub struct KernelNode {
    pub inputs: Vec<BufId>,
    pub outputs: Vec<BufId>,
    /// Parallel to `inputs`: whether the kernel also writes the buffer
    /// in place (read-write dependency rather than read-only).
    pub modifies: Vec<bool>,
    pub func: KernelFn,
    pub name: String,
}

impl fmt::Debug for KernelNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelNode")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("modifies", &self.modifies)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// Device-to-device (or intra-device) byte copy: `dst[dst_offset ..
/// dst_offset + num_bytes] <- src[src_offset .. src_offset + num_bytes]`.
///
/// Offsets and length are [`Quast`] expressions so they can depend on the
/// same symbolic variables buffer sizes are built from. For a full-buffer
/// copy use [`GraphBuilder::insert_memcpy`], which sets both offsets to 0
/// and `num_bytes` to `dst`'s declared size; for partial copies use
/// [`GraphBuilder::insert_memcpy_range`].
#[derive(Clone, Debug)]
pub struct MemcpyNode {
    pub src: BufId,
    pub src_offset: Quast,
    pub dst: BufId,
    pub dst_offset: Quast,
    pub num_bytes: Quast,
}

/// Byte-pattern fill of `buf[offset .. offset + num_bytes]` with the
/// low-byte of `val` (see [`crate::graph_exe::GraphExe::run`] for the
/// byte-uniformity check).
#[derive(Clone, Debug)]
pub struct MemSetNode {
    pub node: BufId,
    pub offset: Quast,
    pub num_bytes: Quast,
    pub val: u32,
}

/// Structured kernel: a high-level [`ir::Module`] with explicit [`BufId`]
/// bindings for its module inputs and outputs. `inputs` is one-to-one and
/// order-aligned with `module.builder.inputs()`; `outputs` is one-to-one
/// with the module's top-level outputs (a scalar body binds one output; a
/// tuple body binds one BufId per element).
///
/// `module` is wrapped in an [`Rc`] so multiple `KernelModuleNode`s can
/// share the same source module by pointer identity; downstream compilation
/// (see `graph_exe`) deduplicates JIT builds keyed on that pointer.
pub struct KernelModuleNode {
    pub module: Rc<ir::Module>,
    pub inputs: Vec<BufId>,
    pub outputs: Vec<BufId>,
}

impl fmt::Debug for KernelModuleNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelModuleNode")
            .field("name", &self.module.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

/// Static data attached to a graph buffer.
pub enum ConstBuf {
    /// Bytes already resident on a CUDA device.
    DeviceBuf(DeviceBuffer<u8>),
    /// Bytes on the host.
    HostBuf(Vec<u8>),
}

impl fmt::Debug for ConstBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceBuf(_) => f.debug_tuple("DeviceBuf").finish(),
            Self::HostBuf(v) => f
                .debug_struct("HostBuf")
                .field("bytes", &v.len())
                .finish_non_exhaustive(),
        }
    }
}

/// Constant node: makes `buf` refer to statically-provided bytes in `data`.
/// Semantically the buffer is written by this node so downstream nodes can
/// consume it via ordinary read/write dependencies.
#[derive(Debug)]
pub struct ConstNode {
    pub buf: BufId,
    pub data: ConstBuf,
}

pub enum GraphNode {
    /// Structured kernel: an [`ir::Module`] with explicit BufId bindings.
    Kernel(KernelModuleNode),
    /// Opaque host closure carrying its own input/output pointer bindings.
    BlackboxKernel(KernelNode),
    /// Static data attached to a buffer (device- or host-resident).
    Const(ConstNode),
    Memcpy(MemcpyNode),
    Memset(MemSetNode),
}

impl fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kernel(k) => f.debug_tuple("Kernel").field(k).finish(),
            Self::BlackboxKernel(k) => f.debug_tuple("BlackboxKernel").field(k).finish(),
            Self::Const(c) => f.debug_tuple("Const").field(c).finish(),
            Self::Memcpy(m) => f.debug_tuple("Memcpy").field(m).finish(),
            Self::Memset(m) => f.debug_tuple("Memset").field(m).finish(),
        }
    }
}

#[derive(Default)]
pub struct GraphBuilder {
    pub bufs: Vec<BufInfo>,
    pub nodes: Vec<GraphNode>,
    /// Symbolic variables that may appear in buffer sizes. The value bound
    /// to each [`VarId`] is its printable name.
    pub symbols: BTreeMap<VarId, String>,
    next_var: u32,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a fresh symbolic variable usable inside buffer size
    /// [`Quast`] expressions, and remembers its printable name.
    pub fn register_symbol(&mut self, name: impl Into<String>) -> VarId {
        let v = VarId(self.next_var);
        self.next_var += 1;
        self.symbols.insert(v, name.into());
        v
    }

    pub fn add_buf(&mut self, info: BufInfo) -> BufId {
        let id = BufId(self.bufs.len());
        self.bufs.push(info);
        id
    }

    pub fn buf_info(&self, id: BufId) -> &BufInfo {
        &self.bufs[id.0]
    }

    /// Adds a structured kernel: an [`ir::Module`] to be lowered by the
    /// compiler pipeline, together with the graph buffers that feed its
    /// declared module inputs and receive its outputs.
    ///
    /// The module is passed as an `Rc<ir::Module>` (or anything convertible
    /// into one, so a bare `ir::Module` also works). Inserting two kernels
    /// with the *same* `Rc` clone lets [`crate::graph_exe::GraphCompiler`]
    /// JIT the module only once and reuse the compiled artifact across both
    /// nodes.
    ///
    /// `inputs.len()` must equal `module.builder.inputs().len()`.
    pub fn insert_kernel(
        &mut self,
        module: impl Into<Rc<ir::Module>>,
        inputs: impl IntoIterator<Item = BufId>,
        outputs: impl IntoIterator<Item = BufId>,
    ) {
        let module: Rc<ir::Module> = module.into();
        let inputs: Vec<BufId> = inputs.into_iter().collect();
        assert_eq!(
            inputs.len(),
            module.builder.inputs().len(),
            "insert_kernel: inputs.len() must match the number of module inputs \
             (module `{}` declares {})",
            module.name,
            module.builder.inputs().len(),
        );
        self.nodes.push(GraphNode::Kernel(KernelModuleNode {
            module,
            inputs,
            outputs: outputs.into_iter().collect(),
        }));
    }

    /// Adds a constant node: makes `buf` refer to static bytes carried in
    /// `data` (either device- or host-resident, see [`ConstBuf`]).
    pub fn insert_const(&mut self, buf: BufId, data: ConstBuf) {
        self.nodes.push(GraphNode::Const(ConstNode { buf, data }));
    }

    /// Adds a blackbox kernel: an opaque host closure with explicit input,
    /// output and in-place-modify buffer bindings.
    ///
    /// The closure receives the graph runner's [`cudaStream_t`] as its third
    /// argument and should launch its work *asynchronously* on that stream —
    /// do not synchronize inside the closure. Graph execution enforces
    /// intra-graph ordering by launching every node on the same stream in
    /// planner-chosen order.
    pub fn insert_blackbox_kernel(
        &mut self,
        name: impl Into<String>,
        inputs: impl Iterator<Item = BufId>,
        outputs: impl Iterator<Item = BufId>,
        modifies: impl Iterator<Item = bool>,
        f: impl Fn(&[*mut ()], &[*mut ()], cudaStream_t) + 'static,
    ) {
        let inputs: Vec<_> = inputs.collect();
        let modifies: Vec<_> = modifies.collect();
        assert_eq!(
            inputs.len(),
            modifies.len(),
            "modifies must have one flag per input"
        );
        self.nodes.push(GraphNode::BlackboxKernel(KernelNode {
            inputs,
            outputs: outputs.collect(),
            modifies,
            func: Box::new(f),
            name: name.into(),
        }));
    }

    /// Full-buffer copy: `dst[..] <- src[..dst_size]`. Offsets are 0 and
    /// `num_bytes` is `dst`'s declared byte size (matching this API's
    /// pre-offset semantics). Use [`Self::insert_memcpy_range`] for partial
    /// copies with explicit offsets.
    pub fn insert_memcpy(&mut self, src: BufId, dst: BufId) {
        let n = self.bufs[dst.0].size.clone();
        self.nodes.push(GraphNode::Memcpy(MemcpyNode {
            src,
            src_offset: Quast::cst(0),
            dst,
            dst_offset: Quast::cst(0),
            num_bytes: n,
        }));
    }

    /// Byte-range copy: `dst[dst_offset..dst_offset + num_bytes] <-
    /// src[src_offset..src_offset + num_bytes]`. All three quantities are
    /// symbolic [`Quast`] expressions; they're resolved to concrete byte
    /// counts against the same variable binding used for buffer sizes.
    pub fn insert_memcpy_range(
        &mut self,
        src: BufId,
        src_offset: Quast,
        dst: BufId,
        dst_offset: Quast,
        num_bytes: Quast,
    ) {
        self.nodes.push(GraphNode::Memcpy(MemcpyNode {
            src,
            src_offset,
            dst,
            dst_offset,
            num_bytes,
        }));
    }

    /// Full-buffer byte-pattern fill: `buf[..] <- low_byte(val)`. Offset is
    /// 0 and `num_bytes` is `buf`'s declared byte size. Use
    /// [`Self::insert_memset_range`] for partial fills.
    pub fn insert_memset(&mut self, node: BufId, val: u32) {
        let n = self.bufs[node.0].size.clone();
        self.nodes.push(GraphNode::Memset(MemSetNode {
            node,
            offset: Quast::cst(0),
            num_bytes: n,
            val,
        }));
    }

    /// Byte-range fill: `buf[offset..offset + num_bytes] <- low_byte(val)`.
    pub fn insert_memset_range(&mut self, node: BufId, offset: Quast, num_bytes: Quast, val: u32) {
        self.nodes.push(GraphNode::Memset(MemSetNode {
            node,
            offset,
            num_bytes,
            val,
        }));
    }

    /// SSA-form textual dump of the graph. Inputs are listed as header
    /// comments; each node prints as `let (out: T[size], ...) = Op(args,
    /// attrs);`. Buffer sizes are the symbolic `Quast` expressions declared
    /// on each `BufInfo`, with registered symbols shown by their names.
    /// Ordering follows the insertion order (a valid topological order for
    /// graphs built as write-before-read).
    pub fn print(&self) -> String {
        let mut out = String::new();
        let (writers, _readers) = classify_buf_uses(&self.nodes, self.bufs.len());
        out.push_str("// GraphBuilder IR dump\n");
        out.push_str(
            "// Buffer types: G[I]=CUDA device I, C=CpuPaged, CP=CpuPinned; \
             `T[expr]` = symbolic byte size.\n",
        );
        if !self.symbols.is_empty() {
            out.push_str("// Symbols:\n");
            for (v, name) in &self.symbols {
                out.push_str(&format!("//   VarId({}) = {name}\n", v.0));
            }
        }
        let mut input_bufs: Vec<usize> = (0..self.bufs.len())
            .filter(|&b| writers[b].is_empty())
            .collect();
        input_bufs.sort();
        if !input_bufs.is_empty() {
            out.push_str("// Graph inputs (no writer):\n");
            for b in input_bufs {
                out.push_str(&format!(
                    "//   {}: {}  // BufId({})\n",
                    self.buf_name(BufId(b)),
                    format_buf_type(&self.bufs[b], &self.symbols),
                    b,
                ));
            }
        }
        out.push('\n');
        for node in &self.nodes {
            out.push_str(&self.format_node_line(node));
            out.push('\n');
        }
        out
    }

    /// Cytoscape.js elements-JSON dump of the graph (`{"elements":
    /// {"nodes": [..], "edges": [..]}}`), for browser visualization via
    /// `scripts/serve_graph.py`.
    ///
    /// Graph nodes become cytoscape nodes labeled with the op name and node
    /// type; buffers with no writer get synthetic `Input` nodes on first
    /// read. Dataflow becomes directed edges from the buffer's most recent
    /// writer to the consumer, labeled `%name [size]`; edges are `black`
    /// when the consumer only reads the buffer and `red` when it also
    /// writes it (in-place modify or overwrite of previously written data).
    pub fn to_cytoscape_json(&self) -> String {
        let mut nodes_json: Vec<String> = Vec::new();
        let mut edges_json: Vec<String> = Vec::new();
        let mut last_writer: Vec<Option<String>> = vec![None; self.bufs.len()];

        let push_node = |nodes_json: &mut Vec<String>, id: &str, name: &str, ty: &str| {
            nodes_json.push(format!(
                "    {{\"data\":{{\"id\":\"{id}\",\"label\":\"{label}\",\"name\":\"{name}\",\
                 \"type\":\"{ty}\"}}}}",
                label = json_escape(&format!("{name}\n{ty}")),
                name = json_escape(name),
            ));
        };
        let buf_label = |b: BufId| {
            format!(
                "{} [{}]",
                self.buf_name(b),
                format_size(&self.bufs[b.0].size, &self.symbols)
            )
        };

        for (n, node) in self.nodes.iter().enumerate() {
            let id = format!("n{n}");
            let (name, ty) = match node {
                GraphNode::Kernel(k) => (k.module.name.clone(), "Kernel"),
                GraphNode::BlackboxKernel(k) => (k.name.clone(), "BlackboxKernel"),
                GraphNode::Const(c) => (self.buf_name(c.buf), "Const"),
                GraphNode::Memcpy(_) => ("memcpy".to_string(), "Memcpy"),
                GraphNode::Memset(m) => (format!("memset {:#x}", m.val), "Memset"),
            };
            push_node(&mut nodes_json, &id, &name, ty);

            // (buffer, consumer-also-writes-it)
            let mut reads: Vec<(BufId, bool)> = Vec::new();
            let mut writes: Vec<BufId> = Vec::new();
            match node {
                GraphNode::Kernel(k) => {
                    reads = k
                        .inputs
                        .iter()
                        .map(|b| (*b, k.outputs.contains(b)))
                        .collect();
                    writes = k.outputs.clone();
                }
                GraphNode::BlackboxKernel(k) => {
                    reads = k
                        .inputs
                        .iter()
                        .zip(&k.modifies)
                        .map(|(b, &m)| (*b, m || k.outputs.contains(b)))
                        .collect();
                    writes = k
                        .inputs
                        .iter()
                        .zip(&k.modifies)
                        .filter(|&(_, &m)| m)
                        .map(|(b, _)| *b)
                        .chain(k.outputs.iter().copied())
                        .collect();
                }
                GraphNode::Const(c) => writes.push(c.buf),
                GraphNode::Memcpy(m) => {
                    reads.push((m.src, false));
                    writes.push(m.dst);
                }
                GraphNode::Memset(m) => writes.push(m.node),
            }

            // Parallel edges (several buffers flowing between the same two
            // nodes) get merged into one edge with a combined label; a
            // single modifying buffer makes the merged edge red.
            let mut merged: Vec<(String, Vec<String>, bool)> = Vec::new();
            let mut add_edge = |src: String, buf: BufId, modifies: bool| {
                let label = buf_label(buf);
                match merged.iter_mut().find(|(s, ..)| *s == src) {
                    Some((_, labels, m)) => {
                        if !labels.contains(&label) {
                            labels.push(label);
                        }
                        *m |= modifies;
                    }
                    None => merged.push((src, vec![label], modifies)),
                }
            };
            for &(buf, modifies) in &reads {
                if last_writer[buf.0].is_none() {
                    let in_id = format!("in{}", buf.0);
                    push_node(&mut nodes_json, &in_id, &self.buf_name(buf), "Input");
                    last_writer[buf.0] = Some(in_id);
                }
                add_edge(last_writer[buf.0].clone().unwrap(), buf, modifies);
            }
            // Overwrites of previously written buffers that this node does
            // not read are still modifications (WAW ordering).
            for &buf in &writes {
                if reads.iter().any(|&(b, _)| b == buf) {
                    continue;
                }
                if let Some(src) = last_writer[buf.0].clone() {
                    add_edge(src, buf, true);
                }
            }
            for (src, labels, modifies) in merged {
                edges_json.push(format!(
                    "    {{\"data\":{{\"id\":\"e{eid}\",\"source\":\"{src}\",\"target\":\"{id}\",\
                     \"label\":\"{label}\",\"color\":\"{color}\"}}}}",
                    eid = edges_json.len(),
                    label = json_escape(&labels.join(", ")),
                    color = if modifies { "red" } else { "black" },
                ));
            }
            for buf in writes {
                last_writer[buf.0] = Some(id.clone());
            }
        }

        format!(
            "{{\"elements\":{{\n  \"nodes\":[\n{}\n  ],\n  \"edges\":[\n{}\n  ]\n}}}}\n",
            nodes_json.join(",\n"),
            edges_json.join(",\n"),
        )
    }

    fn buf_name(&self, id: BufId) -> String {
        match self.bufs[id.0].name.as_deref() {
            Some(n) => format!("%{n}"),
            None => format!("%b{}", id.0),
        }
    }

    fn buf_decl(&self, id: BufId) -> String {
        format!(
            "{}: {}",
            self.buf_name(id),
            format_buf_type(&self.bufs[id.0], &self.symbols)
        )
    }

    fn buf_ref_list(&self, ids: &[BufId]) -> String {
        ids.iter()
            .map(|&b| self.buf_name(b))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn buf_decl_list(&self, ids: &[BufId]) -> String {
        ids.iter()
            .map(|&b| self.buf_decl(b))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_node_line(&self, node: &GraphNode) -> String {
        match node {
            GraphNode::Kernel(k) => format!(
                "let ({}) = Kernel({}, name=\"{}\");",
                self.buf_decl_list(&k.outputs),
                self.buf_ref_list(&k.inputs),
                k.module.name,
            ),
            GraphNode::BlackboxKernel(k) => {
                let mut attrs = format!("name=\"{}\"", k.name);
                if k.modifies.iter().any(|&m| m) {
                    attrs.push_str(&format!(", modifies={:?}", k.modifies));
                }
                format!(
                    "let ({}) = BlackboxKernel({}, {});",
                    self.buf_decl_list(&k.outputs),
                    self.buf_ref_list(&k.inputs),
                    attrs,
                )
            }
            GraphNode::Const(c) => {
                let data = match &c.data {
                    ConstBuf::HostBuf(v) => format!("HostBuf(bytes={})", v.len()),
                    ConstBuf::DeviceBuf(_) => "DeviceBuf".to_string(),
                };
                format!("let ({}) = Const({data});", self.buf_decl(c.buf))
            }
            GraphNode::Memcpy(m) => format!(
                "let ({}) = Memcpy({}, src_off={}, dst_off={}, n={});",
                self.buf_decl(m.dst),
                self.buf_name(m.src),
                format_size(&m.src_offset, &self.symbols),
                format_size(&m.dst_offset, &self.symbols),
                format_size(&m.num_bytes, &self.symbols),
            ),
            GraphNode::Memset(m) => format!(
                "let ({}) = Memset(val={:#x}, off={}, n={});",
                self.buf_decl(m.node),
                m.val,
                format_size(&m.offset, &self.symbols),
                format_size(&m.num_bytes, &self.symbols),
            ),
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Renders `Quast` size expressions with named symbols and minimal
/// parentheses (uses `/` for [`Quast::FloorDiv`]).
pub(crate) fn format_size(q: &Quast, symbols: &BTreeMap<VarId, String>) -> String {
    format_quast_prec(q, symbols, 0)
}

pub(crate) fn format_buf_type(info: &BufInfo, symbols: &BTreeMap<VarId, String>) -> String {
    format!(
        "{}[{}]",
        device_ty_str(info.device_type),
        format_size(&info.size, symbols)
    )
}

pub(crate) fn device_ty_str(t: DeviceType) -> String {
    match t {
        DeviceType::Cuda(i) => format!("G[{i}]"),
        DeviceType::CpuPaged => "C".to_string(),
        DeviceType::CpuPinned => "CP".to_string(),
    }
}

/// Precedence: 0 = outermost, 1 = inside +/-, 2 = inside * / /.
fn format_quast_prec(q: &Quast, symbols: &BTreeMap<VarId, String>, prec: u8) -> String {
    match q {
        Quast::Sym(v) => symbols
            .get(v)
            .cloned()
            .unwrap_or_else(|| format!("v{}", v.0)),
        Quast::Const(c) => format!("{c}"),
        Quast::Add(a, b) => {
            let s = format!(
                "{} + {}",
                format_quast_prec(a, symbols, 1),
                format_quast_prec(b, symbols, 1)
            );
            if prec >= 2 {
                format!("({s})")
            } else {
                s
            }
        }
        Quast::Mul(a, c) => {
            let s = format!("{} * {c}", format_quast_prec(a, symbols, 2));
            if prec >= 2 {
                format!("({s})")
            } else {
                s
            }
        }
        Quast::FloorDiv(a, c) => {
            let s = format!("{} / {c}", format_quast_prec(a, symbols, 2));
            if prec >= 2 {
                format!("({s})")
            } else {
                s
            }
        }
        Quast::Neg(a) => format!("-{}", format_quast_prec(a, symbols, 2)),
    }
}

pub(crate) fn classify_buf_uses(
    nodes: &[GraphNode],
    n_bufs: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut writers = vec![vec![]; n_bufs];
    let mut readers = vec![vec![]; n_bufs];
    for (n, node) in nodes.iter().enumerate() {
        match node {
            GraphNode::Kernel(k) => {
                for b in &k.inputs {
                    readers[b.0].push(n);
                }
                for b in &k.outputs {
                    writers[b.0].push(n);
                }
            }
            GraphNode::BlackboxKernel(k) => {
                for (i, b) in k.inputs.iter().enumerate() {
                    readers[b.0].push(n);
                    if k.modifies[i] {
                        writers[b.0].push(n);
                    }
                }
                for b in &k.outputs {
                    writers[b.0].push(n);
                }
            }
            GraphNode::Const(c) => writers[c.buf.0].push(n),
            GraphNode::Memcpy(m) => {
                readers[m.src.0].push(n);
                writers[m.dst.0].push(n);
            }
            GraphNode::Memset(m) => writers[m.node.0].push(n),
        }
    }
    (writers, readers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRBuilder, ScalarType};

    fn buf(builder: &mut GraphBuilder, name: &str, device_type: DeviceType, size: Quast) -> BufId {
        builder.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type,
            size,
            elem_size: 4,
        })
    }

    #[test]
    fn builds_graph() {
        let mut b = GraphBuilder::new();
        let n = b.register_symbol("n");
        // 4 * n bytes: a symbolic size in terms of the registered variable.
        let sz = Quast::sym(n).mul_c(4);
        let host = buf(&mut b, "host", DeviceType::CpuPinned, sz.clone());
        let x = buf(&mut b, "x", DeviceType::Cuda(0), sz.clone());
        let y = buf(&mut b, "y", DeviceType::Cuda(0), sz.clone());

        b.insert_memcpy(host, x);
        b.insert_memset(y, 0);
        b.insert_blackbox_kernel(
            "add",
            [x].into_iter(),
            [y].into_iter(),
            [false].into_iter(),
            |_inputs, _outputs, _stream| {},
        );

        assert_eq!(b.bufs.len(), 3);
        assert_eq!(b.buf_info(x).name.as_deref(), Some("x"));
        assert_eq!(b.symbols.get(&n).map(String::as_str), Some("n"));
        // Symbolic size resolves once `n` is bound.
        let env = BTreeMap::from([(n, 256)]);
        assert_eq!(b.buf_info(x).size.eval(&env), 1024);
        assert_eq!(b.nodes.len(), 3);
        assert!(matches!(
            &b.nodes[0],
            GraphNode::Memcpy(MemcpyNode { src, dst, .. }) if *src == host && *dst == x
        ));
        assert!(matches!(
            &b.nodes[1],
            GraphNode::Memset(MemSetNode { node, val: 0, .. }) if *node == y
        ));
        match &b.nodes[2] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.name, "add");
                assert_eq!(k.inputs, vec![x]);
                assert_eq!(k.outputs, vec![y]);
                assert_eq!(k.modifies, vec![false]);
                (k.func)(&[], &[], std::ptr::null_mut());
            }
            other => panic!("expected blackbox kernel node, got {other:?}"),
        }
    }

    #[test]
    fn insert_kernel_module() {
        let mut ib = IRBuilder::new();
        let a = ib.input("a", ScalarType::BabyBear, vec![4]);
        let body = ib.compute(4, |ib, i| {
            let ai = ib.index(a, &[i]);
            let two = ib.const_field(2);
            ib.mul(ai, two)
        });
        let module = ib.finish("scale_by_two", body);

        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(16));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(16));
        b.insert_kernel(module, [a_buf], [out_buf]);

        assert_eq!(b.nodes.len(), 1);
        match &b.nodes[0] {
            GraphNode::Kernel(k) => {
                assert_eq!(k.module.name, "scale_by_two");
                assert_eq!(k.inputs, vec![a_buf]);
                assert_eq!(k.outputs, vec![out_buf]);
            }
            other => panic!("expected structured kernel node, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "inputs.len() must match the number of module inputs")]
    fn insert_kernel_wrong_input_count_panics() {
        let mut ib = IRBuilder::new();
        let _ = ib.input("a", ScalarType::BabyBear, vec![4]);
        let body = ib.compute(4, |ib, _i| ib.const_field(0));
        let module = ib.finish("bad", body);

        let mut b = GraphBuilder::new();
        b.insert_kernel(module, std::iter::empty(), std::iter::empty());
    }

    #[test]
    fn insert_const_host_and_device_buf() {
        let mut b = GraphBuilder::new();
        let x = buf(&mut b, "x", DeviceType::CpuPinned, Quast::cst(8));
        let y = buf(&mut b, "y", DeviceType::Cuda(0), Quast::cst(0));
        b.insert_const(x, ConstBuf::HostBuf(vec![0, 1, 2, 3, 4, 5, 6, 7]));
        b.insert_const(y, ConstBuf::DeviceBuf(DeviceBuffer::<u8>::new()));

        assert_eq!(b.nodes.len(), 2);
        match &b.nodes[0] {
            GraphNode::Const(ConstNode {
                buf,
                data: ConstBuf::HostBuf(bytes),
            }) => {
                assert_eq!(*buf, x);
                assert_eq!(bytes.len(), 8);
            }
            other => panic!("expected host Const node, got {other:?}"),
        }
        match &b.nodes[1] {
            GraphNode::Const(ConstNode {
                buf,
                data: ConstBuf::DeviceBuf(_),
            }) => {
                assert_eq!(*buf, y);
            }
            other => panic!("expected device Const node, got {other:?}"),
        }
    }

    #[test]
    fn insert_memcpy_full_buffer_uses_dst_size_and_zero_offsets() {
        let mut b = GraphBuilder::new();
        let n = b.register_symbol("n");
        let sz = Quast::sym(n).mul_c(4);
        let src = buf(&mut b, "src", DeviceType::Cuda(0), sz.clone());
        let dst = buf(&mut b, "dst", DeviceType::Cuda(0), sz.clone());
        b.insert_memcpy(src, dst);
        let env = BTreeMap::from([(n, 32)]);
        match &b.nodes[0] {
            GraphNode::Memcpy(m) => {
                assert_eq!(m.src, src);
                assert_eq!(m.dst, dst);
                assert_eq!(m.src_offset.eval(&env), 0);
                assert_eq!(m.dst_offset.eval(&env), 0);
                assert_eq!(m.num_bytes.eval(&env), 128);
            }
            other => panic!("expected memcpy, got {other:?}"),
        }
    }

    #[test]
    fn insert_memcpy_range_carries_offsets_and_length() {
        let mut b = GraphBuilder::new();
        let src = buf(&mut b, "src", DeviceType::Cuda(0), Quast::cst(64));
        let dst = buf(&mut b, "dst", DeviceType::Cuda(0), Quast::cst(64));
        b.insert_memcpy_range(src, Quast::cst(16), dst, Quast::cst(8), Quast::cst(32));
        match &b.nodes[0] {
            GraphNode::Memcpy(m) => {
                assert_eq!(m.src_offset.eval(&BTreeMap::new()), 16);
                assert_eq!(m.dst_offset.eval(&BTreeMap::new()), 8);
                assert_eq!(m.num_bytes.eval(&BTreeMap::new()), 32);
            }
            other => panic!("expected memcpy, got {other:?}"),
        }
    }

    #[test]
    fn insert_memset_range_carries_offset_and_length() {
        let mut b = GraphBuilder::new();
        let x = buf(&mut b, "x", DeviceType::Cuda(0), Quast::cst(128));
        b.insert_memset_range(x, Quast::cst(32), Quast::cst(64), 0xff);
        match &b.nodes[0] {
            GraphNode::Memset(m) => {
                assert_eq!(m.node, x);
                assert_eq!(m.val, 0xff);
                assert_eq!(m.offset.eval(&BTreeMap::new()), 32);
                assert_eq!(m.num_bytes.eval(&BTreeMap::new()), 64);
            }
            other => panic!("expected memset, got {other:?}"),
        }
    }

    #[test]
    fn register_symbol_allocates_distinct_ids() {
        let mut b = GraphBuilder::new();
        let n = b.register_symbol("n");
        let m = b.register_symbol("m");
        assert_ne!(n, m);
        assert_eq!(b.symbols.get(&n).map(String::as_str), Some("n"));
        assert_eq!(b.symbols.get(&m).map(String::as_str), Some("m"));
    }

    #[test]
    fn cytoscape_json_nodes_and_edge_colors() {
        let mut b = GraphBuilder::new();
        let n = b.register_symbol("n");
        let sz = Quast::sym(n).mul_c(4);
        let host = buf(&mut b, "host", DeviceType::CpuPinned, sz.clone());
        let x = buf(&mut b, "x", DeviceType::Cuda(0), sz.clone());
        let y = buf(&mut b, "y", DeviceType::Cuda(0), sz.clone());

        // host --(read)--> memcpy --(defines x)--> kernel "add" reads x,
        // defines y; kernel "fold" modifies y in place.
        b.insert_memcpy(host, x);
        b.insert_blackbox_kernel(
            "add",
            [x].into_iter(),
            [y].into_iter(),
            [false].into_iter(),
            |_, _, _| {},
        );
        b.insert_blackbox_kernel(
            "fold",
            [y].into_iter(),
            [].into_iter(),
            [true].into_iter(),
            |_, _, _| {},
        );
        // "pair" writes two buffers both read by "sum": the two parallel
        // edges must merge into one with a combined label.
        let u = buf(&mut b, "u", DeviceType::Cuda(0), sz.clone());
        let v = buf(&mut b, "v", DeviceType::Cuda(0), sz.clone());
        b.insert_blackbox_kernel(
            "pair",
            [].into_iter(),
            [u, v].into_iter(),
            [].into_iter(),
            |_, _, _| {},
        );
        b.insert_blackbox_kernel(
            "sum",
            [u, v].into_iter(),
            [].into_iter(),
            [false, false].into_iter(),
            |_, _, _| {},
        );

        let json = b.to_cytoscape_json();
        // `host` has no writer => synthetic Input node feeding the memcpy.
        assert!(json.contains(r#""id":"in0","label":"%host\nInput","name":"%host","type":"Input""#));
        assert!(json.contains(r#""id":"n0","label":"memcpy\nMemcpy""#));
        assert!(json.contains(r#""id":"n1","label":"add\nBlackboxKernel""#));
        // Pure reads are black, sized with the symbolic buffer size.
        assert!(json
            .contains(r#""source":"in0","target":"n0","label":"%host [n * 4]","color":"black""#));
        assert!(
            json.contains(r#""source":"n0","target":"n1","label":"%x [n * 4]","color":"black""#)
        );
        // In-place modify of y (written by n1) is red.
        assert!(json.contains(r#""source":"n1","target":"n2","label":"%y [n * 4]","color":"red""#));
        // Parallel edges pair->sum merged with a combined label.
        assert!(json.contains(
            r#""source":"n3","target":"n4","label":"%u [n * 4], %v [n * 4]","color":"black""#
        ));
        assert!(!json.contains(r#""label":"%u [n * 4]","#));
    }

    #[test]
    #[should_panic(expected = "modifies must have one flag per input")]
    fn kernel_modifies_length_mismatch_panics() {
        let mut b = GraphBuilder::new();
        let x = buf(&mut b, "x", DeviceType::Cuda(0), Quast::cst(1024));
        b.insert_blackbox_kernel(
            "bad",
            [x].into_iter(),
            [x].into_iter(),
            [].into_iter(),
            |_, _, _| {},
        );
    }
}
