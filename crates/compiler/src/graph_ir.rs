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

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::Arc,
};

use openvm_cuda_common::{d_buffer::DeviceBuffer, stream::cudaStream_t};

use crate::{
    ir::{self, VarId},
    module_hash::module_hash,
    passes::split_module::{split_module, ModuleSubgraph, SubgraphValue},
    quast::{Quast, SExpr, SymConst},
};

/// Index of a buffer in the graph's buffer table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufId(pub usize);

/// Per-node buffer accesses `(reads, writes)`, where each read carries a
/// flag indicating the node also writes the buffer (in-place modify). See
/// [`GraphBuilder::node_reads_writes`].
type NodeReadsWrites = (Vec<(BufId, bool)>, Vec<BufId>);

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
    /// Subset of `inputs` whose buffers the kernel also *writes* to
    /// (in-place). These are the "carried" inputs: they flow into the
    /// kernel as a read and out again as a fresh write of the same
    /// [`BufId`]. Buffers in `carried_outputs` are effectively both a
    /// read and a write of this node, so downstream nodes that touch
    /// them must sequence after this one. The union `outputs ∪
    /// carried_outputs` is the complete write set used to build data
    /// dependencies — any valid topological order over those edges
    /// respects the order of execution the caller intended.
    pub carried_outputs: Vec<BufId>,
    pub func: KernelFn,
    pub name: String,
}

impl fmt::Debug for KernelNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelNode")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("carried_outputs", &self.carried_outputs)
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
/// Invariant: `module` contains exactly one top-level kernel — its body is
/// a single `compute` (or bare `reduce`), never a `let` chain of several.
/// [`GraphBuilder::insert_kernel`] enforces this by splitting multi-kernel
/// modules on insertion (see [`crate::passes::split_module`]), one graph
/// node per kernel. The invariant is HIR-level only: JIT-time rewrites
/// (e.g. `rewrite_parallel_reduce` inside [`crate::compile_and_load`]) may
/// still expand one HIR kernel into several CUDA kernels within its
/// compiled [`crate::runtime::KernelModule`]. Callers that push
/// `GraphNode::Kernel` directly into [`GraphBuilder::nodes`] are
/// responsible for upholding it themselves.
///
/// `module` is wrapped in an [`Arc`] so multiple `KernelModuleNode`s can
/// share the same source module by pointer identity; downstream compilation
/// (see `graph_exe`) deduplicates JIT builds keyed on that pointer. `Arc`
/// (rather than `Rc`) also lets modules cross thread boundaries during
/// parallel JIT.
pub struct KernelModuleNode {
    pub module: Arc<ir::Module>,
    /// Concrete value of each `module.builder.params()` entry for *this*
    /// node, inferred at insertion by unifying the module's input shape
    /// expressions against the bound graph buffers (see
    /// [`GraphBuilder::infer_param_bindings`]). Empty iff the module
    /// declares no parameters. Order-aligned with the params registry.
    pub param_bindings: Vec<i64>,
    pub inputs: Vec<BufId>,
    pub outputs: Vec<BufId>,
    /// Trace of the fusion steps that produced this kernel. `None` on
    /// un-fused, freshly-lowered kernels; `Some` after
    /// [`crate::passes::fusion::apply_fusion`] merges two kernels. Metadata
    /// only — not part of [`module_hash`], and not read by the runtime.
    pub fusion_history: Option<Arc<FusionHistory>>,
}

impl fmt::Debug for KernelModuleNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelModuleNode")
            .field("name", &self.module.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("fused", &self.fusion_history.is_some())
            .finish_non_exhaustive()
    }
}

/// One original kernel captured before it was absorbed by a fusion.
/// Shape strings are computed at capture time via [`format_size`] because
/// the buffer table entries the leaf's `BufId`s point at may be rewritten
/// or removed by later fusion rounds.
pub struct KernelSnapshot {
    pub node: Arc<KernelModuleNode>,
    pub input_shapes: Vec<String>,
    pub output_shapes: Vec<String>,
}

impl fmt::Debug for KernelSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelSnapshot")
            .field("name", &self.node.module.name)
            .field("input_shapes", &self.input_shapes)
            .field("output_shapes", &self.output_shapes)
            .finish()
    }
}

/// Recursive record of how a fused kernel was built.
///
/// A [`FusionHistory::Kernel`] leaf is an original (un-fused) kernel that
/// went into some later fusion; a [`FusionHistory::Fused`] internal node
/// records the two operands of a single merge step, where `consumer` was
/// the destination kernel (the reader) and `producer` was the source (the
/// writer that got grafted into the consumer's read sites). Both variants
/// carry a [`KernelSnapshot`] — the leaf's is the original pre-fusion
/// kernel, the internal node's is the intermediate merged kernel at that
/// step — so the viewer can render module IR + shapes at any point in
/// the tree, not just at the leaves.
#[derive(Debug)]
pub enum FusionHistory {
    Kernel(Arc<KernelSnapshot>),
    Fused {
        snapshot: Arc<KernelSnapshot>,
        consumer: Arc<FusionHistory>,
        producer: Arc<FusionHistory>,
    },
}

impl FusionHistory {
    /// The [`KernelSnapshot`] associated with this node (leaf or fused).
    pub fn snapshot(&self) -> &KernelSnapshot {
        match self {
            FusionHistory::Kernel(s) => s,
            FusionHistory::Fused { snapshot, .. } => snapshot,
        }
    }

    /// Collects the names of every leaf original kernel in this tree, in
    /// left-to-right (consumer-first) traversal order.
    pub fn leaf_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_leaf_names(&mut out);
        out
    }

    fn collect_leaf_names(&self, out: &mut Vec<String>) {
        match self {
            FusionHistory::Kernel(s) => out.push(s.node.module.name.clone()),
            FusionHistory::Fused {
                consumer, producer, ..
            } => {
                consumer.collect_leaf_names(out);
                producer.collect_leaf_names(out);
            }
        }
    }
}

impl KernelModuleNode {
    /// Returns this node's fusion history, materializing a fresh leaf
    /// snapshot (with shape strings resolved via `g`) when the node has
    /// never been fused. Used by [`crate::passes::fusion::apply_fusion`]
    /// when assembling the merged tree.
    pub fn history_or_leaf(&self, g: &GraphBuilder) -> Arc<FusionHistory> {
        if let Some(h) = &self.fusion_history {
            return Arc::clone(h);
        }
        let node = Arc::new(KernelModuleNode {
            module: Arc::clone(&self.module),
            param_bindings: self.param_bindings.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            fusion_history: None,
        });
        let input_shapes = self
            .inputs
            .iter()
            .map(|b| format_size(&g.bufs[b.0].size, &g.symbols))
            .collect();
        let output_shapes = self
            .outputs
            .iter()
            .map(|b| format_size(&g.bufs[b.0].size, &g.symbols))
            .collect();
        Arc::new(FusionHistory::Kernel(Arc::new(KernelSnapshot {
            node,
            input_shapes,
            output_shapes,
        })))
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

/// Key of [`GraphBuilder::subgraph_cache`]: (module_hash, shape hint).
/// `module_hash` treats the hint as metadata, but the cached split kernels
/// carry hint *values* (sliced per kernel) that `insert_subgraph_impl` uses
/// for param-binding inference, so same-HIR modules with different hints
/// must not share an entry.
type SubgraphCacheKey = ([u8; 32], Option<Vec<i64>>);

#[derive(Default)]
pub struct GraphBuilder {
    pub bufs: Vec<BufInfo>,
    pub nodes: Vec<GraphNode>,
    /// Symbolic variables that may appear in buffer sizes. The value bound
    /// to each [`VarId`] is its printable name.
    pub symbols: BTreeMap<VarId, String>,
    next_var: u32,
    /// Content-hash → canonical `Arc<ir::Module>` for *split* (single-
    /// kernel) modules. Kernel insertion collapses structurally identical
    /// modules onto their first-seen `Arc` so downstream `Arc::as_ptr`
    /// identity checks (`GraphCompiler`'s per-module JIT dedup, cytoscape
    /// rendering, …) automatically see the merged set. This is skipped when
    /// a caller pushes `GraphNode::Kernel` directly into `nodes`, so the
    /// compiler still runs a hash-based dedup pass as a backstop.
    module_dedup: HashMap<[u8; 32], Arc<ir::Module>>,
    /// Content-hash of the *original* (possibly multi-kernel) module →
    /// its split subgraph. [`GraphBuilder::insert_kernel`] splits a module
    /// once per unique content and replays the cached subgraph on every
    /// later insertion.
    subgraph_cache: HashMap<SubgraphCacheKey, Arc<ModuleSubgraph>>,
    /// Buffers the caller declared as graph inputs, in registration order
    /// (which defines the `set_input` index order). See
    /// [`Self::register_input`].
    input_bufs: Vec<BufId>,
    /// Buffers the caller declared as graph outputs, in registration order
    /// (which defines the `get_output` index order). See
    /// [`Self::register_output`].
    output_bufs: Vec<BufId>,
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

    /// Declares `id` as a graph input: its contents are supplied by the
    /// caller at run time via `GraphExe::set_input`. Registration order
    /// defines the input index order. Inputs must not be written by any
    /// graph node (validated at compile time).
    pub fn register_input(&mut self, id: BufId) {
        self.input_bufs.push(id);
    }

    /// Declares `id` as a graph output: its final contents are observable
    /// by the caller via `GraphExe::get_output`. Registration order defines
    /// the output index order. Outputs must be written by at least one
    /// graph node (validated at compile time).
    pub fn register_output(&mut self, id: BufId) {
        self.output_bufs.push(id);
    }

    /// Registered graph inputs, in registration order.
    pub fn input_bufs(&self) -> &[BufId] {
        &self.input_bufs
    }

    /// Registered graph outputs, in registration order.
    pub fn output_bufs(&self) -> &[BufId] {
        &self.output_bufs
    }

    /// Whether `id` is part of the registered graph interface (input or
    /// output). Everything else is internal: graph rewrites (fusion, DCE)
    /// may freely eliminate it.
    pub fn buf_is_interface(&self, id: BufId) -> bool {
        self.input_bufs.contains(&id) || self.output_bufs.contains(&id)
    }

    /// Adds a structured kernel: an [`ir::Module`] to be lowered by the
    /// compiler pipeline, together with the graph buffers that feed its
    /// declared module inputs and receive its outputs.
    ///
    /// The module is type-checked, canonicalized and split into one
    /// single-kernel module per top-level compute/reduce (see
    /// [`crate::passes::split_module`]) the first time its content is seen;
    /// later insertions of a structurally identical module replay the
    /// cached split. One [`GraphNode::Kernel`] is pushed per split kernel,
    /// so a multi-kernel module produces multiple graph nodes, with
    /// intermediate buffers allocated automatically on the first input's
    /// device (each `GraphNode::Kernel` then upholds the single-kernel
    /// invariant documented on [`KernelModuleNode`]). Split kernels are
    /// content-deduped onto canonical `Arc`s so
    /// [`crate::graph_exe::GraphCompiler`] JITs each unique kernel once.
    ///
    /// The module is passed as an `Arc<ir::Module>` (or anything convertible
    /// into one, so a bare `ir::Module` also works).
    ///
    /// `inputs.len()` must equal `module.builder.inputs().len()` and
    /// `outputs` must have one BufId per module output (a scalar body binds
    /// one output; a tuple body binds one per element).
    ///
    /// `shape_hint` supplies a canonical concrete instantiation of the
    /// module's symbolic parameters, in declaration order: used for access
    /// checking and as the binding fallback for parameters that cannot be
    /// inferred from the input buffer shapes. Pass `&[]` for no hint;
    /// otherwise its length must equal the module's parameter count. The
    /// hint is attached to the module here — module authors never record
    /// hints themselves — and is part of the split cache key, so one module
    /// inserted under different hints splits (and JITs) per hint while
    /// structurally identical residuals still content-dedup downstream.
    ///
    /// Panics if the module fails type checking or canonicalization.
    pub fn insert_kernel(
        &mut self,
        module: impl Into<Arc<ir::Module>>,
        inputs: impl IntoIterator<Item = BufId>,
        outputs: impl IntoIterator<Item = BufId>,
        shape_hint: &[i64],
    ) {
        let mut module: Arc<ir::Module> = module.into();
        if !shape_hint.is_empty() {
            Arc::make_mut(&mut module)
                .builder
                .add_shape_hint(shape_hint);
        }
        let module = module;
        let inputs: Vec<BufId> = inputs.into_iter().collect();
        let outputs: Vec<BufId> = outputs.into_iter().collect();
        assert_eq!(
            inputs.len(),
            module.builder.inputs().len(),
            "insert_kernel: inputs.len() must match the number of module inputs \
             (module `{}` declares {})",
            module.name,
            module.builder.inputs().len(),
        );
        let key = (
            module_hash(&module),
            module.builder.shape_hint().map(<[i64]>::to_vec),
        );
        let subgraph = match self.subgraph_cache.get(&key) {
            Some(sg) => sg.clone(),
            None => {
                let sg = split_module(&module).unwrap_or_else(|e| {
                    panic!(
                        "insert_kernel: module `{}` failed to split: {e}",
                        module.name
                    )
                });
                let sg = Arc::new(sg);
                self.subgraph_cache.insert(key, sg.clone());
                sg
            }
        };
        assert_eq!(
            outputs.len(),
            subgraph.outputs.len(),
            "insert_kernel: outputs.len() must match the number of module outputs \
             (module `{}` produces {})",
            module.name,
            subgraph.outputs.len(),
        );
        self.insert_subgraph_impl(&subgraph, &inputs, Some(&outputs));
    }

    /// Adds every kernel of a split [`ModuleSubgraph`] (see
    /// [`crate::passes::split_module::split_module`]), wiring
    /// [`SubgraphValue::Input`]s to `inputs` and allocating a fresh buffer
    /// for every kernel output. Returns the buffers holding the subgraph's
    /// outputs, one per [`ModuleSubgraph::outputs`] entry.
    ///
    /// Intermediate and output buffers are allocated on the first input's
    /// device (`Cuda(0)` if the subgraph has no inputs) with byte sizes
    /// from the split kernels' [`crate::passes::OutputSpec`]s, evaluated
    /// under each kernel's inferred parameter bindings.
    ///
    /// `inputs.len()` must equal [`ModuleSubgraph::num_inputs`].
    pub fn insert_subgraph(&mut self, subgraph: &ModuleSubgraph, inputs: &[BufId]) -> Vec<BufId> {
        self.insert_subgraph_impl(subgraph, inputs, None)
    }

    /// Shared body of [`Self::insert_kernel`] / [`Self::insert_subgraph`]:
    /// pushes one [`GraphNode::Kernel`] per split kernel in dependency
    /// order. With `bound_outputs` set, the subgraph's outputs land in
    /// those buffers; otherwise fresh buffers are allocated for them like
    /// for any intermediate.
    fn insert_subgraph_impl(
        &mut self,
        subgraph: &ModuleSubgraph,
        inputs: &[BufId],
        bound_outputs: Option<&[BufId]>,
    ) -> Vec<BufId> {
        assert_eq!(
            inputs.len(),
            subgraph.num_inputs,
            "insert_subgraph: inputs.len() must match the subgraph's input count \
             (subgraph `{}` declares {})",
            subgraph.name,
            subgraph.num_inputs,
        );
        let device = inputs
            .first()
            .or_else(|| bound_outputs.and_then(|outs| outs.first()))
            .map(|b| self.bufs[b.0].device_type)
            .unwrap_or(DeviceType::Cuda(0));

        // Kernel outputs that are subgraph outputs bound by the caller.
        let mut bound: HashMap<SubgraphValue, BufId> = HashMap::new();
        if let Some(outs) = bound_outputs {
            assert_eq!(outs.len(), subgraph.outputs.len());
            for (val, &buf) in subgraph.outputs.iter().zip(outs) {
                match val {
                    SubgraphValue::KernelOutput { .. } => {
                        bound.insert(*val, buf);
                    }
                    SubgraphValue::Input(_) => panic!(
                        "insert_kernel: subgraph `{}` output {val:?} is an input \
                         passthrough and cannot be bound to an output buffer",
                        subgraph.name,
                    ),
                }
            }
        }

        // Per-kernel output buffers, in kernel order (producers precede
        // consumers, so back-references into `produced` always resolve).
        let mut produced: Vec<Vec<BufId>> = Vec::with_capacity(subgraph.kernels.len());
        for (ki, k) in subgraph.kernels.iter().enumerate() {
            let in_bufs: Vec<BufId> = k
                .inputs
                .iter()
                .map(|v| match *v {
                    SubgraphValue::Input(i) => inputs[i],
                    SubgraphValue::KernelOutput { kernel, out_idx } => produced[kernel][out_idx],
                })
                .collect();
            let param_bindings = self.infer_param_bindings(&k.module, &in_bufs);
            let env: BTreeMap<VarId, i64> = k
                .module
                .builder
                .params()
                .iter()
                .map(|(v, _)| *v)
                .zip(param_bindings.iter().copied())
                .collect();
            let out_bufs: Vec<BufId> = k
                .outputs
                .iter()
                .enumerate()
                .map(|(oi, spec)| {
                    let val = SubgraphValue::KernelOutput {
                        kernel: ki,
                        out_idx: oi,
                    };
                    if let Some(&b) = bound.get(&val) {
                        return b;
                    }
                    let num_elems =
                        spec.num_elems
                            .concretize(&env)
                            .as_const()
                            .unwrap_or_else(|| {
                                panic!(
                                    "insert_kernel: output {oi} of split kernel `{}` has element \
                                 count `{}` that stays symbolic after binding parameters",
                                    k.module.name, spec.num_elems,
                                )
                            });
                    self.add_buf(BufInfo {
                        name: Some(format!("{}.k{ki}.o{oi}", subgraph.name)),
                        device_type: device,
                        size: Quast::cst(num_elems * spec.elem.size_bytes() as i64),
                        elem_size: spec.elem.size_bytes(),
                    })
                })
                .collect();
            let module = self.dedup_module(k.module.clone());
            self.nodes.push(GraphNode::Kernel(KernelModuleNode {
                module,
                param_bindings,
                inputs: in_bufs,
                outputs: out_bufs.clone(),
                fusion_history: None,
            }));
            produced.push(out_bufs);
        }

        subgraph
            .outputs
            .iter()
            .map(|v| match *v {
                SubgraphValue::Input(i) => inputs[i],
                SubgraphValue::KernelOutput { kernel, out_idx } => produced[kernel][out_idx],
            })
            .collect()
    }

    /// Infers a concrete value for each of `module`'s parameters by
    /// unifying its input shape expressions against the bound graph
    /// buffers, one binding per `module.builder.params()` entry (in
    /// registry order). Returns an empty vec iff the module declares no
    /// parameters.
    ///
    /// Inference runs to a fixpoint over the inputs: an input whose shape
    /// mixes several parameters may only become solvable once another
    /// input has bound some of them. Per round, an input contributes when
    /// its buffer's byte size is concrete and, after substituting the
    /// bindings so far, at most one shape dimension remains symbolic; that
    /// dimension is then solved structurally against the buffer's element
    /// count (see [`solve_size_expr`]). Parameters still unbound at the
    /// fixpoint fall back to the module's shape hint. Panics if a
    /// parameter cannot be inferred, or if the bindings are inconsistent
    /// with any concrete input buffer size.
    fn infer_param_bindings(&self, module: &ir::Module, in_bufs: &[BufId]) -> Vec<i64> {
        let params = module.builder.params();
        if params.is_empty() {
            return Vec::new();
        }
        let decls = module.builder.inputs();

        // Element count of each input whose graph buffer has a concrete
        // byte size; symbolic-size buffers can't constrain parameters.
        let targets: Vec<Option<i64>> = decls
            .iter()
            .zip(in_bufs)
            .map(|(decl, b)| {
                let size = &self.bufs[b.0].size;
                let mut syms = BTreeSet::new();
                size.syms(&mut syms);
                if !syms.is_empty() {
                    return None;
                }
                let bytes = size.eval(&BTreeMap::new());
                let elem = decl.elem.size_bytes() as i64;
                assert!(
                    bytes % elem == 0,
                    "insert_kernel: module `{}` input `{}` is bound to a {bytes}-byte \
                     buffer, not a multiple of the {elem}-byte element size",
                    module.name,
                    decl.name,
                );
                Some(bytes / elem)
            })
            .collect();

        let mut env: BTreeMap<VarId, i64> = BTreeMap::new();
        loop {
            let before = env.len();
            for (decl, target) in decls.iter().zip(&targets) {
                let Some(elems) = *target else { continue };
                let mut known: i64 = 1;
                let mut unknown: Option<SExpr> = None;
                let mut skip = false;
                for dim in decl.shape.iter() {
                    let dim = dim.concretize(&env).fold_lits();
                    match dim.as_const() {
                        Some(c) => known *= c,
                        None if unknown.is_none() => unknown = Some(dim),
                        None => {
                            skip = true;
                            break;
                        }
                    }
                }
                let (Some(dim), false) = (unknown, skip) else {
                    continue;
                };
                if known == 0 || elems % known != 0 {
                    continue; // the final verification reports the mismatch
                }
                solve_size_expr(&dim, elems / known, &mut env);
            }
            if env.len() == before {
                break;
            }
        }

        // Parameters the input shapes don't pin down fall back to the
        // author's shape hint (aligned with the params registry).
        if env.len() < params.len() {
            if let Some(hint) = module.builder.shape_hint() {
                for ((v, _), &h) in params.iter().zip(hint) {
                    env.entry(*v).or_insert(h);
                }
            }
        }

        let bindings: Vec<i64> = params
            .iter()
            .map(|(v, name)| {
                *env.get(v).unwrap_or_else(|| {
                    panic!(
                        "insert_kernel: cannot infer parameter `{name}` of module `{}` \
                         from its input buffer sizes (add a shape hint or bind the \
                         parameter through an input shape)",
                        module.name,
                    )
                })
            })
            .collect();

        // Every concrete input buffer must agree with the bindings.
        for (decl, target) in decls.iter().zip(&targets) {
            let Some(elems) = *target else { continue };
            let product: i64 = decl
                .shape
                .iter()
                .map(|dim| {
                    dim.concretize(&env).as_const().unwrap_or_else(|| {
                        panic!(
                            "insert_kernel: module `{}` input `{}` dimension `{dim}` \
                             stays symbolic after binding all parameters",
                            module.name, decl.name,
                        )
                    })
                })
                .product();
            assert!(
                product == elems,
                "insert_kernel: module `{}` input `{}` has shape {:?} = {product} \
                 elements under the inferred parameter bindings, but its graph \
                 buffer holds {elems} elements",
                module.name,
                decl.name,
                decl.shape,
            );
        }

        bindings
    }

    /// Content-dedup: two callers can build structurally identical modules
    /// at different sites and pass distinct `Arc`s. Fold those onto the
    /// first-seen `Arc` so the graph carries a single canonical handle per
    /// unique kernel and downstream `Arc::as_ptr` identity checks see the
    /// merged set.
    pub(crate) fn dedup_module(&mut self, module: Arc<ir::Module>) -> Arc<ir::Module> {
        let hash = module_hash(&module);
        self.module_dedup
            .entry(hash)
            .or_insert_with(|| module.clone())
            .clone()
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
        // Fold the parallel `modifies` bool vector into `carried_outputs`
        // — the subset of inputs the kernel also writes. Everything
        // downstream (edge derivation, planner accesses, IR printer) is
        // written against `carried_outputs`, so a modified input becomes
        // a structural producer edge and any topological sort respects
        // it automatically.
        let carried_outputs: Vec<BufId> = inputs
            .iter()
            .zip(&modifies)
            .filter_map(|(b, &m)| m.then_some(*b))
            .collect();
        self.nodes.push(GraphNode::BlackboxKernel(KernelNode {
            inputs,
            outputs: outputs.collect(),
            carried_outputs,
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

    /// Cytoscape.js elements-JSON dump of the graph, for browser
    /// visualization via `scripts/serve_graph.py`.
    ///
    /// Graph nodes become cytoscape nodes labeled with the op name and node
    /// type; buffers with no writer get synthetic `Input` nodes on first
    /// read. Dataflow becomes directed edges from the buffer's most recent
    /// writer to the consumer, labeled `%name [size]`; edges are `black`
    /// when the consumer only reads the buffer and `red` when it also
    /// writes it (in-place modify or overwrite of previously written data).
    ///
    /// Each node carries an `ir` field (its `format_node_line` text) plus
    /// dataflow stats (`inputs`, `outputs`, `producers`, `consumers`).
    /// Kernel nodes additionally carry a `module` key pointing into the
    /// top-level `modules` map, which holds a single [`crate::dump::dump_hir`]
    /// entry per unique `Arc<ir::Module>` (deduped by pointer identity so
    /// two kernels sharing the same module share one dump). Overall shape:
    /// `{"elements": {"nodes": [..], "edges": [..]}, "modules": {..}}`.
    pub fn to_cytoscape_json(&self) -> String {
        // First pass: compute per-node reads/writes so we can reuse the
        // classification for edges and dataflow stats.
        let node_rw: Vec<NodeReadsWrites> = self
            .nodes
            .iter()
            .map(|n| self.node_reads_writes(n))
            .collect();

        // Second pass: replay the last-writer stream to derive edges, and
        // record (source_node_id, target_node_id) pairs for stats. Synthetic
        // Input nodes use ids `in{buf}` and count as producers.
        let buf_label = |b: BufId| {
            format!(
                "{} [{}]",
                self.buf_name(b),
                format_size(&self.bufs[b.0].size, &self.symbols)
            )
        };
        let mut last_writer: Vec<Option<String>> = vec![None; self.bufs.len()];
        let mut input_bufs: Vec<BufId> = Vec::new();
        // Which buffers were read *anywhere* — used together with
        // `last_writer` after this loop to identify terminal writes that
        // deserve a synthetic Output node (mirror of the Input synthetic
        // emitted for reads with no preceding writer).
        let mut was_read: Vec<bool> = vec![false; self.bufs.len()];
        // (src_id, tgt_id, merged_labels, any_modify)
        let mut edges: Vec<(String, String, Vec<String>, bool)> = Vec::new();
        for (n, (reads, writes)) in node_rw.iter().enumerate() {
            let id = format!("n{n}");
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
            for &(buf, modifies) in reads {
                was_read[buf.0] = true;
                if last_writer[buf.0].is_none() {
                    input_bufs.push(buf);
                    last_writer[buf.0] = Some(format!("in{}", buf.0));
                }
                add_edge(last_writer[buf.0].clone().unwrap(), buf, modifies);
            }
            // WAW: overwrites of previously written buffers this node does
            // not read are still modifications.
            for &buf in writes {
                if reads.iter().any(|&(b, _)| b == buf) {
                    continue;
                }
                if let Some(src) = last_writer[buf.0].clone() {
                    add_edge(src, buf, true);
                }
            }
            for (src, labels, modifies) in merged {
                edges.push((src, id.clone(), labels, modifies));
            }
            for &buf in writes {
                last_writer[buf.0] = Some(id.clone());
            }
        }

        // Synthetic Output nodes: one per buffer that ended the graph as a
        // sink (was written but never consumed) OR that the caller
        // declared as a graph output via `register_output`. Mirror of the
        // Input synthetics above — makes it clear where the graph's
        // observable results live, especially after fusion collapses
        // long producer chains down to a single terminal kernel.
        let mut output_bufs: Vec<BufId> = Vec::new();
        for b in 0..self.bufs.len() {
            let bid = BufId(b);
            let is_registered = self.output_bufs.contains(&bid);
            let is_natural_sink = !was_read[b];
            if let Some(writer) = last_writer[b].clone() {
                if is_natural_sink || is_registered {
                    output_bufs.push(bid);
                    edges.push((writer, format!("out{b}"), vec![buf_label(bid)], false));
                }
            }
        }
        // Registered graph inputs the emitter above didn't already add as
        // Input synthetics (either unused, or written before read — the
        // API forbids the latter but be defensive). Ensures the graph
        // interface is fully visible.
        for &bid in &self.input_bufs {
            if !input_bufs.contains(&bid) {
                input_bufs.push(bid);
            }
        }

        // Aggregate producers/consumers per node id from the edge list.
        let mut producers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut consumers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (src, tgt, _, _) in &edges {
            producers
                .entry(tgt.clone())
                .or_default()
                .insert(src.clone());
            consumers
                .entry(src.clone())
                .or_default()
                .insert(tgt.clone());
        }
        let count = |m: &BTreeMap<String, BTreeSet<String>>, id: &str| {
            m.get(id).map(|s| s.len()).unwrap_or(0)
        };

        // Display name per node id, including synthetic `in{buf}` inputs;
        // fed into `producer_names` / `consumer_names` so the viewer can
        // render clickable chips without a second lookup.
        let mut name_by_id: BTreeMap<String, String> = BTreeMap::new();
        for (n, node) in self.nodes.iter().enumerate() {
            let id = format!("n{n}");
            let name = match node {
                GraphNode::Kernel(k) => k.module.name.clone(),
                GraphNode::BlackboxKernel(k) => k.name.clone(),
                GraphNode::Const(c) => self.buf_name(c.buf),
                GraphNode::Memcpy(_) => "memcpy".to_string(),
                GraphNode::Memset(m) => format!("memset {:#x}", m.val),
            };
            name_by_id.insert(id, name);
        }
        for buf in &input_bufs {
            name_by_id.insert(format!("in{}", buf.0), self.buf_name(*buf));
        }
        for buf in &output_bufs {
            name_by_id.insert(format!("out{}", buf.0), self.buf_name(*buf));
        }

        // Emit an ordered ID array + parallel name array for the neighbors
        // of `id` in `m`. Empty when there are none.
        let neighbor_arrays = |m: &BTreeMap<String, BTreeSet<String>>, id: &str| {
            let set = match m.get(id) {
                Some(s) => s,
                None => return ("[]".to_string(), "[]".to_string()),
            };
            let ids: Vec<String> = set
                .iter()
                .map(|s| format!("\"{}\"", json_escape(s)))
                .collect();
            let names: Vec<String> = set
                .iter()
                .map(|s| {
                    format!(
                        "\"{}\"",
                        json_escape(name_by_id.get(s).map(String::as_str).unwrap_or(""))
                    )
                })
                .collect();
            (
                format!("[{}]", ids.join(",")),
                format!("[{}]", names.join(",")),
            )
        };

        // Shape strings (symbolic byte sizes) for a slice of buffers.
        let shape_array = |bufs: &[BufId]| -> String {
            let items: Vec<String> = bufs
                .iter()
                .map(|b| {
                    format!(
                        "\"{}\"",
                        json_escape(&format_size(&self.bufs[b.0].size, &self.symbols))
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        };

        // Dedupe modules by Arc pointer identity: two kernels sharing the
        // same Arc<ir::Module> share a single `modules` entry keyed by name
        // (with a suffix on pointer collisions where distinct modules
        // happen to have the same name).
        let mut module_key_by_ptr: HashMap<*const ir::Module, String> = HashMap::new();
        let mut modules_dump: BTreeMap<String, String> = BTreeMap::new();
        let mut module_key = |m: &Arc<ir::Module>| -> String {
            let ptr = Arc::as_ptr(m);
            if let Some(k) = module_key_by_ptr.get(&ptr) {
                return k.clone();
            }
            let mut key = m.name.clone();
            let mut n = 1;
            while modules_dump.contains_key(&key) {
                key = format!("{}#{n}", m.name);
                n += 1;
            }
            modules_dump.insert(key.clone(), crate::dump::dump_hir(m));
            module_key_by_ptr.insert(ptr, key.clone());
            key
        };

        // Now emit the JSON.
        let mut nodes_json: Vec<String> = Vec::new();
        let push_node = |nodes_json: &mut Vec<String>, fields: Vec<(&str, String)>| {
            let body = fields
                .iter()
                .map(|(k, v)| format!("\"{k}\":{v}"))
                .collect::<Vec<_>>()
                .join(",");
            nodes_json.push(format!("    {{\"data\":{{{body}}}}}"));
        };
        for (n, (reads, writes)) in node_rw.iter().enumerate() {
            let id = format!("n{n}");
            let node = &self.nodes[n];
            let (name, ty) = match node {
                GraphNode::Kernel(k) => (k.module.name.clone(), "Kernel"),
                GraphNode::BlackboxKernel(k) => (k.name.clone(), "BlackboxKernel"),
                GraphNode::Const(c) => (self.buf_name(c.buf), "Const"),
                GraphNode::Memcpy(_) => ("memcpy".to_string(), "Memcpy"),
                GraphNode::Memset(m) => (format!("memset {:#x}", m.val), "Memset"),
            };
            let ir_text = self.format_node_line(node);
            let read_bufs: Vec<BufId> = reads.iter().map(|&(b, _)| b).collect();
            let (prod_ids, prod_names) = neighbor_arrays(&producers, &id);
            let (cons_ids, cons_names) = neighbor_arrays(&consumers, &id);
            let mut fields = vec![
                ("id", format!("\"{id}\"")),
                (
                    "label",
                    format!("\"{}\"", json_escape(&format!("{name}\n{ty}"))),
                ),
                ("name", format!("\"{}\"", json_escape(&name))),
                ("type", format!("\"{ty}\"")),
                ("ir", format!("\"{}\"", json_escape(&ir_text))),
                ("inputs", reads.len().to_string()),
                ("outputs", writes.len().to_string()),
                ("producers", count(&producers, &id).to_string()),
                ("consumers", count(&consumers, &id).to_string()),
                ("input_shapes", shape_array(&read_bufs)),
                ("output_shapes", shape_array(writes)),
                ("producer_ids", prod_ids),
                ("producer_names", prod_names),
                ("consumer_ids", cons_ids),
                ("consumer_names", cons_names),
            ];
            if let GraphNode::Kernel(k) = node {
                let key = module_key(&k.module);
                fields.push(("module", format!("\"{}\"", json_escape(&key))));
                if let Some(history) = &k.fusion_history {
                    fields.push(("fusion_history", fusion_history_json(history)));
                }
            }
            push_node(&mut nodes_json, fields);
        }
        for buf in &input_bufs {
            let in_id = format!("in{}", buf.0);
            let name = self.buf_name(*buf);
            let (cons_ids, cons_names) = neighbor_arrays(&consumers, &in_id);
            push_node(
                &mut nodes_json,
                vec![
                    ("id", format!("\"{in_id}\"")),
                    (
                        "label",
                        format!("\"{}\"", json_escape(&format!("{name}\nInput"))),
                    ),
                    ("name", format!("\"{}\"", json_escape(&name))),
                    ("type", "\"Input\"".to_string()),
                    ("consumers", count(&consumers, &in_id).to_string()),
                    ("output_shapes", shape_array(std::slice::from_ref(buf))),
                    ("consumer_ids", cons_ids),
                    ("consumer_names", cons_names),
                ],
            );
        }
        for buf in &output_bufs {
            let out_id = format!("out{}", buf.0);
            let name = self.buf_name(*buf);
            let (prod_ids, prod_names) = neighbor_arrays(&producers, &out_id);
            push_node(
                &mut nodes_json,
                vec![
                    ("id", format!("\"{out_id}\"")),
                    (
                        "label",
                        format!("\"{}\"", json_escape(&format!("{name}\nOutput"))),
                    ),
                    ("name", format!("\"{}\"", json_escape(&name))),
                    ("type", "\"Output\"".to_string()),
                    ("producers", count(&producers, &out_id).to_string()),
                    ("input_shapes", shape_array(std::slice::from_ref(buf))),
                    ("producer_ids", prod_ids),
                    ("producer_names", prod_names),
                ],
            );
        }

        let mut edges_json: Vec<String> = Vec::with_capacity(edges.len());
        for (eid, (src, tgt, labels, modifies)) in edges.iter().enumerate() {
            edges_json.push(format!(
                "    {{\"data\":{{\"id\":\"e{eid}\",\"source\":\"{src}\",\"target\":\"{tgt}\",\
                 \"label\":\"{label}\",\"color\":\"{color}\"}}}}",
                label = json_escape(&labels.join(", ")),
                color = if *modifies { "red" } else { "black" },
            ));
        }

        let modules_json = modules_dump
            .iter()
            .map(|(k, v)| format!("    \"{}\":\"{}\"", json_escape(k), json_escape(v)))
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            "{{\"elements\":{{\n  \"nodes\":[\n{}\n  ],\n  \"edges\":[\n{}\n  ]\n}},\n\
             \"modules\":{{\n{}\n}}}}\n",
            nodes_json.join(",\n"),
            edges_json.join(",\n"),
            modules_json,
        )
    }

    /// Classifies a graph node's buffer accesses. Returns `(reads, writes)`
    /// where each read carries a flag indicating the node also modifies the
    /// buffer (in-place update). Matches the WAW-aware semantics used for
    /// building cytoscape edges.
    pub(crate) fn node_reads_writes(&self, node: &GraphNode) -> NodeReadsWrites {
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
                    .map(|b| (*b, k.carried_outputs.contains(b) || k.outputs.contains(b)))
                    .collect();
                writes = k
                    .carried_outputs
                    .iter()
                    .copied()
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
        (reads, writes)
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
                if !k.carried_outputs.is_empty() {
                    attrs.push_str(&format!(
                        ", carried_outputs=[{}]",
                        self.buf_ref_list(&k.carried_outputs),
                    ));
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

/// Serializes a [`FusionHistory`] tree into JSON. Leaves carry the
/// snapshot's IR (via [`crate::dump::dump_hir`]) plus captured shape
/// strings; `Fused` internal nodes carry the intermediate merged kernel's
/// snapshot (same IR + shape fields as a leaf) plus their two children
/// under `consumer` / `producer` keys.
fn fusion_history_json(h: &FusionHistory) -> String {
    let string_array = |ss: &[String]| -> String {
        let items: Vec<String> = ss
            .iter()
            .map(|s| format!("\"{}\"", json_escape(s)))
            .collect();
        format!("[{}]", items.join(","))
    };
    let snap = h.snapshot();
    let m = &snap.node.module;
    let ir = crate::dump::dump_hir(m);
    let base = format!(
        "\"name\":\"{name}\",\"ir\":\"{ir}\",\
         \"input_shapes\":{inputs},\"output_shapes\":{outputs}",
        name = json_escape(&m.name),
        ir = json_escape(&ir),
        inputs = string_array(&snap.input_shapes),
        outputs = string_array(&snap.output_shapes),
    );
    match h {
        FusionHistory::Kernel(_) => format!("{{\"kind\":\"leaf\",{base}}}"),
        FusionHistory::Fused {
            consumer, producer, ..
        } => format!(
            "{{\"kind\":\"fused\",{base},\
             \"consumer\":{consumer},\"producer\":{producer}}}",
            consumer = fusion_history_json(consumer),
            producer = fusion_history_json(producer),
        ),
    }
}

/// Structural solver for one symbolic shape dimension: binds parameters in
/// `env` so that `e` (already concretized against `env`, so any remaining
/// [`SymConst::Sym`] is unbound) evaluates to `target`.
///
/// Returns `false` when the expression's structure doesn't determine the
/// parameters (e.g. two unbound parameters multiplied together, or a
/// `FloorDiv`, which is not invertible) — the caller just tries again next
/// fixpoint round or falls back to the shape hint. Arithmetic
/// contradictions (a literal that doesn't match, a non-divisible
/// coefficient) panic: the module cannot fit the buffer it was bound to.
fn solve_size_expr(e: &SExpr, target: i64, env: &mut BTreeMap<VarId, i64>) -> bool {
    match e {
        SExpr::Const(SymConst::Lit(c)) => {
            assert!(
                *c == target,
                "insert_kernel: shape dimension `{e}` = {c} but the bound buffer \
                 requires {target}",
            );
            true
        }
        SExpr::Const(SymConst::Sym(v)) => {
            env.insert(*v, target);
            true
        }
        SExpr::Mul(inner, c) => {
            let k = match c {
                SymConst::Lit(k) => *k,
                SymConst::Sym(v) => match inner.as_const() {
                    // `const * p`: the coefficient is the unknown.
                    Some(k) => {
                        assert!(
                            k != 0 && target % k == 0,
                            "insert_kernel: cannot solve `{e}` = {target}: {target} \
                             is not a multiple of {k}",
                        );
                        env.insert(*v, target / k);
                        return true;
                    }
                    None => return false,
                },
            };
            assert!(
                k != 0 && target % k == 0,
                "insert_kernel: cannot solve `{e}` = {target}: {target} is not a \
                 multiple of {k}",
            );
            solve_size_expr(inner, target / k, env)
        }
        SExpr::Add(a, b) => match (a.as_const(), b.as_const()) {
            (Some(c), _) => solve_size_expr(b, target - c, env),
            (_, Some(c)) => solve_size_expr(a, target - c, env),
            _ => false,
        },
        SExpr::Neg(a) => solve_size_expr(a, -target, env),
        // Loop vars can't appear in input shapes; FloorDiv is lossy.
        SExpr::Sym(_) | SExpr::FloorDiv(..) => false,
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
                for b in &k.inputs {
                    readers[b.0].push(n);
                }
                for b in k.carried_outputs.iter().chain(k.outputs.iter()) {
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
    use crate::ir::{IRBuilder, ScalarType, SizeExpr};

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
                assert!(k.carried_outputs.is_empty());
                (k.func)(&[], &[], std::ptr::null_mut());
            }
            other => panic!("expected blackbox kernel node, got {other:?}"),
        }
    }

    #[test]
    fn subgraph_intermediates_are_not_interface() {
        let mut ib = IRBuilder::new();
        let a = ib.input("a", ScalarType::BabyBear, vec![4]);
        let stage1 = ib.compute(4, |ib, i| {
            let ai = ib.index(a, &[i]);
            let two = ib.const_field(2);
            ib.mul(ai, two)
        });
        let t = ib.let_bound(stage1);
        let body = ib.compute(4, |ib, i| {
            let ti = ib.index(t, &[i]);
            ib.add(ti, ti)
        });
        let module = ib.finish("two_stage", body);

        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(16));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(16));
        b.register_input(a_buf);
        b.register_output(out_buf);
        b.insert_kernel(module, [a_buf], [out_buf], &[]);

        assert_eq!(b.nodes.len(), 2);
        assert!(b.buf_is_interface(a_buf));
        assert!(b.buf_is_interface(out_buf));
        assert_eq!(b.input_bufs(), [a_buf]);
        assert_eq!(b.output_bufs(), [out_buf]);
        let mid = match &b.nodes[0] {
            GraphNode::Kernel(k) => k.outputs[0],
            other => panic!("expected kernel node, got {other:?}"),
        };
        assert_ne!(mid, a_buf);
        assert_ne!(mid, out_buf);
        assert!(!b.buf_is_interface(mid));
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
        b.insert_kernel(module, [a_buf], [out_buf], &[]);

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
        b.insert_kernel(module, std::iter::empty(), std::iter::empty(), &[]);
    }

    /// `let t = a * 2; out = t + a` as a two-kernel chain over one input.
    fn two_kernel_chain(name: &str, n: usize) -> ir::Module {
        let mut ib = IRBuilder::new();
        let a = ib.input("a", ScalarType::BabyBear, vec![n]);
        let t = ib.compute(n, |ib, i| {
            let ai = ib.index(a, &[i]);
            let two = ib.const_field(2);
            ib.mul(ai, two)
        });
        let t = ib.let_bound(t);
        let out = ib.compute(n, |ib, i| {
            let ti = ib.index(t, &[i]);
            let ai = ib.index(a, &[i]);
            ib.add(ti, ai)
        });
        ib.finish(name, out)
    }

    #[test]
    fn insert_kernel_splits_multi_kernel_module() {
        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(32));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(32));
        b.insert_kernel(two_kernel_chain("chain", 8), [a_buf], [out_buf], &[]);

        // One GraphNode::Kernel per split kernel, chained through an
        // auto-allocated intermediate buffer on the inputs' device.
        assert_eq!(b.nodes.len(), 2);
        let (k0, k1) = match (&b.nodes[0], &b.nodes[1]) {
            (GraphNode::Kernel(k0), GraphNode::Kernel(k1)) => (k0, k1),
            other => panic!("expected two structured kernel nodes, got {other:?}"),
        };
        assert_eq!(k0.module.name, "chain__k0");
        assert_eq!(k0.inputs, vec![a_buf]);
        assert_eq!(k0.outputs.len(), 1);
        let mid = k0.outputs[0];
        assert_ne!(mid, out_buf);
        assert_eq!(k1.module.name, "chain__k1");
        assert_eq!(k1.inputs, vec![mid, a_buf]);
        assert_eq!(k1.outputs, vec![out_buf]);

        let mid_info = b.buf_info(mid);
        assert_eq!(mid_info.name.as_deref(), Some("chain.k0.o0"));
        assert_eq!(mid_info.device_type, DeviceType::Cuda(0));
        assert_eq!(mid_info.size.eval(&BTreeMap::new()), 32);
        assert_eq!(mid_info.elem_size, 4);
    }

    #[test]
    fn insert_kernel_replays_cached_subgraph_with_shared_arcs() {
        let mut b = GraphBuilder::new();
        let a0 = buf(&mut b, "a0", DeviceType::Cuda(0), Quast::cst(32));
        let o0 = buf(&mut b, "o0", DeviceType::Cuda(0), Quast::cst(32));
        let a1 = buf(&mut b, "a1", DeviceType::Cuda(0), Quast::cst(32));
        let o1 = buf(&mut b, "o1", DeviceType::Cuda(0), Quast::cst(32));

        // Two structurally identical modules built independently: the
        // second insertion must hit the subgraph cache and reuse the same
        // canonical Arcs for the split kernels.
        b.insert_kernel(two_kernel_chain("chain", 8), [a0], [o0], &[]);
        b.insert_kernel(two_kernel_chain("chain", 8), [a1], [o1], &[]);

        assert_eq!(b.nodes.len(), 4);
        let kernels: Vec<_> = b
            .nodes
            .iter()
            .map(|n| match n {
                GraphNode::Kernel(k) => k,
                other => panic!("expected structured kernel node, got {other:?}"),
            })
            .collect();
        assert!(Arc::ptr_eq(&kernels[0].module, &kernels[2].module));
        assert!(Arc::ptr_eq(&kernels[1].module, &kernels[3].module));
        // Second insertion wires its own buffers.
        assert_eq!(kernels[2].inputs, vec![a1]);
        assert_eq!(kernels[3].outputs, vec![o1]);
    }

    #[test]
    fn insert_subgraph_allocates_and_returns_outputs() {
        let module = two_kernel_chain("chain", 8);
        let subgraph = crate::passes::split_module(&module).expect("split");

        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(32));
        let outs = b.insert_subgraph(&subgraph, &[a_buf]);

        assert_eq!(outs.len(), 1);
        assert_eq!(b.nodes.len(), 2);
        match &b.nodes[1] {
            GraphNode::Kernel(k) => assert_eq!(k.outputs, outs),
            other => panic!("expected structured kernel node, got {other:?}"),
        }
        let out_info = b.buf_info(outs[0]);
        assert_eq!(out_info.name.as_deref(), Some("chain.k1.o0"));
        assert_eq!(out_info.size.eval(&BTreeMap::new()), 32);
        assert_eq!(out_info.elem_size, 4);
    }

    /// `let t = a * 2; out = t + a` over a symbolic length `n`.
    fn symbolic_two_kernel_chain(name: &str) -> ir::Module {
        let mut ib = IRBuilder::new();
        let n = ib.symbol("n");
        let a = ib.input("a", ScalarType::BabyBear, vec![n]);
        let t = ib.compute(n, |ib, i| {
            let ai = ib.index(a, &[i]);
            let two = ib.const_field(2);
            ib.mul(ai, two)
        });
        let t = ib.let_bound(t);
        let out = ib.compute(n, |ib, i| {
            let ti = ib.index(t, &[i]);
            let ai = ib.index(a, &[i]);
            ib.add(ti, ai)
        });
        ib.finish(name, out)
    }

    fn kernel_nodes(b: &GraphBuilder) -> Vec<&KernelModuleNode> {
        b.nodes
            .iter()
            .map(|n| match n {
                GraphNode::Kernel(k) => k,
                other => panic!("expected structured kernel node, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn infers_binding_from_bare_input_dim() {
        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(32));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(32));
        b.insert_kernel(symbolic_two_kernel_chain("chain"), [a_buf], [out_buf], &[]);

        // Both split kernels bind `n = 8` from the 32-byte (8-element)
        // input buffer, and the intermediate is sized through the binding.
        let kernels = kernel_nodes(&b);
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0].param_bindings, vec![8]);
        assert_eq!(kernels[1].param_bindings, vec![8]);
        let mid = kernels[0].outputs[0];
        assert_eq!(b.buf_info(mid).size.eval(&BTreeMap::new()), 32);
    }

    #[test]
    fn infers_binding_from_compound_input_dim() {
        let mut ib = IRBuilder::new();
        let n = ib.symbol("n");
        let a = ib.input(
            "a",
            ScalarType::BabyBear,
            vec![SizeExpr::from(n), 64usize.into()],
        );
        let body = ib.compute(n, |ib, i| {
            let zero = ib.const_u32(0);
            let av = ib.index(a, &[i, zero]);
            ib.add(av, av)
        });
        let module = ib.finish("rows", body);

        let mut b = GraphBuilder::new();
        // 4 bytes * 64 columns * 32 rows: `n` must come out as 32.
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(4 * 64 * 32));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(4 * 32));
        b.insert_kernel(module, [a_buf], [out_buf], &[]);

        assert_eq!(kernel_nodes(&b)[0].param_bindings, vec![32]);
    }

    #[test]
    fn binding_falls_back_to_shape_hint() {
        let mut ib = IRBuilder::new();
        let n = ib.symbol("n");
        let m = ib.symbol("m");
        let a = ib.input("a", ScalarType::BabyBear, vec![n * m]);
        let body = ib.compute(n * m, |ib, i| {
            let av = ib.index(a, &[i]);
            ib.add(av, av)
        });
        let module = ib.finish("prod", body);

        let mut b = GraphBuilder::new();
        // `n * m` is structurally unsolvable (two unbound parameters), so
        // both come from the hint; the buffer still checks out (128 elems).
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(4 * 128));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(4 * 128));
        b.insert_kernel(module, [a_buf], [out_buf], &[8, 16]);

        assert_eq!(kernel_nodes(&b)[0].param_bindings, vec![8, 16]);
    }

    #[test]
    fn same_module_different_hints_bind_independently() {
        // Same HIR (hence same `module_hash`, which excludes the hint), but
        // different hint values. The subgraph cache must not replay the
        // first module's hint for the second insert.
        let build = || {
            let mut ib = IRBuilder::new();
            let n = ib.symbol("n");
            let m = ib.symbol("m");
            let a = ib.input("a", ScalarType::BabyBear, vec![n * m]);
            let body = ib.compute(n * m, |ib, i| {
                let av = ib.index(a, &[i]);
                ib.add(av, av)
            });
            ib.finish("prod", body)
        };

        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(4 * 128));
        let out1 = buf(&mut b, "out1", DeviceType::Cuda(0), Quast::cst(4 * 128));
        let out2 = buf(&mut b, "out2", DeviceType::Cuda(0), Quast::cst(4 * 128));
        b.insert_kernel(build(), [a_buf], [out1], &[8, 16]);
        b.insert_kernel(build(), [a_buf], [out2], &[4, 32]);

        let kernels = kernel_nodes(&b);
        assert_eq!(kernels[0].param_bindings, vec![8, 16]);
        assert_eq!(kernels[1].param_bindings, vec![4, 32]);
    }

    #[test]
    #[should_panic(expected = "cannot infer parameter")]
    fn unsolvable_binding_without_hint_panics() {
        let mut ib = IRBuilder::new();
        let n = ib.symbol("n");
        let m = ib.symbol("m");
        let a = ib.input("a", ScalarType::BabyBear, vec![n * m]);
        let body = ib.compute(n * m, |ib, i| {
            let av = ib.index(a, &[i]);
            ib.add(av, av)
        });
        let module = ib.finish("prod", body);

        let mut b = GraphBuilder::new();
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(4 * 128));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(4 * 128));
        b.insert_kernel(module, [a_buf], [out_buf], &[]);
    }

    #[test]
    #[should_panic(expected = "elements under the inferred parameter bindings")]
    fn inconsistent_binding_panics() {
        let mut ib = IRBuilder::new();
        let n = ib.symbol("n");
        let a = ib.input("a", ScalarType::BabyBear, vec![n]);
        let c = ib.input("c", ScalarType::BabyBear, vec![n]);
        let body = ib.compute(n, |ib, i| {
            let av = ib.index(a, &[i]);
            let cv = ib.index(c, &[i]);
            ib.add(av, cv)
        });
        let module = ib.finish("mismatch", body);

        let mut b = GraphBuilder::new();
        // `a` binds n = 8, but `c`'s buffer holds 16 elements.
        let a_buf = buf(&mut b, "a", DeviceType::Cuda(0), Quast::cst(32));
        let c_buf = buf(&mut b, "c", DeviceType::Cuda(0), Quast::cst(64));
        let out_buf = buf(&mut b, "out", DeviceType::Cuda(0), Quast::cst(32));
        b.insert_kernel(module, [a_buf, c_buf], [out_buf], &[]);
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

        // Per-node IR text and dataflow stats surface on each node.
        // n1 ("add") reads x, writes y => inputs=1, outputs=1.
        // Its producer is n0 (memcpy) and its consumer is n2 (fold).
        assert!(
            json.contains(r#""ir":"let (%y: G[0][n * 4]) = BlackboxKernel(%x, name=\"add\");""#)
        );
        assert!(json.contains(r#""inputs":1,"outputs":1,"producers":1,"consumers":1"#));
        // n3 ("pair") has 0 inputs, 2 outputs; both u and v flow into n4
        // ("sum") so consumers=1 (unique downstream node).
        assert!(json.contains(r#""inputs":0,"outputs":2,"producers":0,"consumers":1"#));
        // Synthetic Input node reports its downstream consumers.
        assert!(json.contains(
            r#""id":"in0","label":"%host\nInput","name":"%host","type":"Input","consumers":1"#
        ));

        // No Kernel (structured module) nodes here, so the modules map
        // exists but is empty.
        assert!(json.contains("\"modules\":{\n\n}"));
    }

    #[test]
    fn cytoscape_json_module_dedup_and_hir() {
        use crate::ir::{IRBuilder, ScalarType};
        let mut b = GraphBuilder::new();
        let sz = Quast::cst(1024);
        let inp = buf(&mut b, "inp", DeviceType::Cuda(0), sz.clone());
        let out_a = buf(&mut b, "out_a", DeviceType::Cuda(0), sz.clone());
        let out_b = buf(&mut b, "out_b", DeviceType::Cuda(0), sz.clone());

        // A minimal module used twice; the two insertions must share one
        // `modules` entry (Arc pointer identity preserved by dedup).
        let make_mod = || {
            let mut ib = IRBuilder::new();
            let x = ib.input("x", ScalarType::U32, vec![256]);
            let body = ib.compute(256, |ib, i| ib.index(x, &[i]));
            ib.finish("shared_mod", body)
        };
        let m = make_mod();
        b.insert_kernel(m.clone(), [inp], [out_a], &[]);
        b.insert_kernel(m, [inp], [out_b], &[]);

        let json = b.to_cytoscape_json();
        // Both kernel nodes reference the same module key.
        assert!(json.contains(r#""module":"shared_mod""#));
        // Module IR dump is under the modules map, with the module header.
        assert!(json.contains(r#""shared_mod":"module shared_mod"#));
        // The module name appears exactly once as a key in the map (dedup).
        let occurrences = json.matches(r#""shared_mod":"module"#).count();
        assert_eq!(occurrences, 1, "module dedup failed: {json}");
    }

    #[test]
    fn modifies_flags_lower_to_carried_outputs() {
        // Public API still takes a parallel `modifies` bool vector, but
        // the KernelNode stores the modified inputs as `carried_outputs`
        // (a Vec<BufId>). Downstream code reasons about the write set as
        // `carried_outputs ∪ outputs`, which makes the graph edges from
        // this node structurally complete.
        let mut b = GraphBuilder::new();
        let sz = Quast::cst(64);
        let x = buf(&mut b, "x", DeviceType::Cuda(0), sz.clone());
        let y = buf(&mut b, "y", DeviceType::Cuda(0), sz.clone());
        let z = buf(&mut b, "z", DeviceType::Cuda(0), sz.clone());
        b.insert_blackbox_kernel(
            "k",
            [x, y].into_iter(),
            [z].into_iter(),
            [false, true].into_iter(),
            |_, _, _| {},
        );
        match &b.nodes[0] {
            GraphNode::BlackboxKernel(k) => {
                assert_eq!(k.inputs, vec![x, y]);
                assert_eq!(k.outputs, vec![z]);
                assert_eq!(k.carried_outputs, vec![y]);
            }
            other => panic!("expected blackbox kernel, got {other:?}"),
        }
    }

    #[test]
    fn carried_outputs_close_the_topological_gap() {
        // Regression: with only a `modifies` flag on the middle node, a
        // downstream reader of the same buffer relied on insertion order
        // to sequence after the modifier. Since `carried_outputs` puts
        // the modified buffer into the node's write set, the writer /
        // reader chain (buffer x is written by k0, re-written by k1, read
        // by k2) shows up in `classify_buf_uses` and any topological sort
        // walks k0 → k1 → k2 without consulting insertion order.
        let mut b = GraphBuilder::new();
        let x = buf(&mut b, "x", DeviceType::Cuda(0), Quast::cst(32));
        b.insert_blackbox_kernel(
            "k0",
            std::iter::empty(),
            [x].into_iter(),
            std::iter::empty(),
            |_, _, _| {},
        );
        b.insert_blackbox_kernel(
            "k1",
            [x].into_iter(),
            std::iter::empty(),
            [true].into_iter(),
            |_, _, _| {},
        );
        b.insert_blackbox_kernel(
            "k2",
            [x].into_iter(),
            std::iter::empty(),
            [false].into_iter(),
            |_, _, _| {},
        );
        let (writers, readers) = classify_buf_uses(&b.nodes, b.bufs.len());
        assert_eq!(writers[x.0], vec![0, 1], "k1 must be a writer of x");
        assert_eq!(readers[x.0], vec![1, 2]);
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
