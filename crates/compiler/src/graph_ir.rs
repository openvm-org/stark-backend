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

use std::{collections::BTreeMap, fmt};

use openvm_cuda_common::d_buffer::DeviceBuffer;

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

/// Type-erased kernel launch: receives raw pointers to the input buffers
/// and to the output buffers, in that order.
pub type KernelFn = Box<dyn Fn(&[*mut ()], &[*mut ()])>;

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

#[derive(Copy, Clone, Debug)]
pub struct MemcpyNode {
    pub src: BufId,
    pub dst: BufId,
}

#[derive(Copy, Clone, Debug)]
pub struct MemSetNode {
    pub node: BufId,
    pub val: u32,
}

/// Structured kernel: a high-level [`ir::Module`] with explicit [`BufId`]
/// bindings for its module inputs and outputs. `inputs` is one-to-one and
/// order-aligned with `module.builder.inputs()`; `outputs` is one-to-one
/// with the module's top-level outputs (a scalar body binds one output; a
/// tuple body binds one BufId per element).
pub struct KernelModuleNode {
    pub module: ir::Module,
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
    /// `inputs.len()` must equal `module.builder.inputs().len()`.
    pub fn insert_kernel(
        &mut self,
        module: ir::Module,
        inputs: impl IntoIterator<Item = BufId>,
        outputs: impl IntoIterator<Item = BufId>,
    ) {
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
    pub fn insert_blackbox_kernel(
        &mut self,
        name: impl Into<String>,
        inputs: impl Iterator<Item = BufId>,
        outputs: impl Iterator<Item = BufId>,
        modifies: impl Iterator<Item = bool>,
        f: impl Fn(&[*mut ()], &[*mut ()]) + 'static,
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

    pub fn insert_memcpy(&mut self, src: BufId, dst: BufId) {
        self.nodes.push(GraphNode::Memcpy(MemcpyNode { src, dst }));
    }

    pub fn insert_memset(&mut self, node: BufId, val: u32) {
        self.nodes.push(GraphNode::Memset(MemSetNode { node, val }));
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
                "let ({}) = Memcpy({});",
                self.buf_decl(m.dst),
                self.buf_name(m.src),
            ),
            GraphNode::Memset(m) => format!(
                "let ({}) = Memset(val={:#x});",
                self.buf_decl(m.node),
                m.val,
            ),
        }
    }
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
            |_inputs, _outputs| {},
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
            GraphNode::Memcpy(MemcpyNode { src, dst }) if *src == host && *dst == x
        ));
        assert!(matches!(
            &b.nodes[1],
            GraphNode::Memset(MemSetNode { node, val: 0 }) if *node == y
        ));
        match &b.nodes[2] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.name, "add");
                assert_eq!(k.inputs, vec![x]);
                assert_eq!(k.outputs, vec![y]);
                assert_eq!(k.modifies, vec![false]);
                (k.func)(&[], &[]);
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
    fn register_symbol_allocates_distinct_ids() {
        let mut b = GraphBuilder::new();
        let n = b.register_symbol("n");
        let m = b.register_symbol("m");
        assert_ne!(n, m);
        assert_eq!(b.symbols.get(&n).map(String::as_str), Some("n"));
        assert_eq!(b.symbols.get(&m).map(String::as_str), Some("m"));
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
            |_, _| {},
        );
    }
}
