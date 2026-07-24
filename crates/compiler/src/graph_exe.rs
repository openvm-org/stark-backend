//! End-to-end graph compilation and execution.
//!
//! [`GraphCompiler`] consumes a [`GraphBuilder`], plans memory via
//! [`crate::planner::plan`], compiles every [`GraphNode::Kernel`]'s module
//! through [`crate::compile_and_load`], and packages the whole thing into a
//! [`GraphExe`]. `GraphExe::run` interprets the plan against a caller-owned
//! scratch buffer plus caller-owned input/output device buffers, mirroring
//! the shape of [`crate::runtime::KernelModule`]'s single-kernel API.
//!
//! Feature-gated behind `planner` (needs the CP-SAT planner + OR-Tools).

use std::{collections::BTreeMap, ffi::c_void, mem::ManuallyDrop};

use openvm_cuda_common::{
    copy::cuda_memcpy_on,
    d_buffer::{cudaMemsetAsync, DeviceBuffer},
    stream::GpuDeviceCtx,
};

use crate::{
    compile_and_load,
    graph_ir::{
        BufId, BufInfo, ConstBuf, ConstNode, DeviceType, GraphBuilder, GraphNode, KernelNode,
    },
    ir::VarId,
    planner::{self, access_from_node, MemoryPlan, NodeAccess, PlanError},
    quast::Quast,
    runtime::{CompileOptions, KernelModule},
    CompileError,
};

/// Builder-pattern compiler that plans a graph and JITs its structured
/// kernels.
///
/// ```text
/// let exe = GraphCompiler::new()
///     .device(DeviceType::Cuda(0))
///     .symbol(n_bytes_sym, 4096)
///     .compile_options(CompileOptions::default())
///     .compile(graph)?;
/// ```
pub struct GraphCompiler {
    device: DeviceType,
    env: BTreeMap<VarId, i64>,
    options: CompileOptions,
}

impl Default for GraphCompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCompiler {
    pub fn new() -> Self {
        Self {
            device: DeviceType::Cuda(0),
            env: BTreeMap::new(),
            options: CompileOptions::default(),
        }
    }

    /// Target device for the memory plan (all Kernel/BlackboxKernel buffer
    /// offsets are assigned within a single pool on this device).
    pub fn device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    /// Binds a symbolic size variable to a concrete value.
    pub fn symbol(mut self, sym: VarId, value: i64) -> Self {
        self.env.insert(sym, value);
        self
    }

    pub fn compile_options(mut self, options: CompileOptions) -> Self {
        self.options = options;
        self
    }

    /// Consumes the graph, plans it and compiles every structured kernel.
    ///
    /// The pipeline runs in three phases:
    ///
    /// 1. **Compile-first**: drain the graph's nodes and JIT every `Kernel(Module)` up front so we
    ///    know each module's own scratch requirement (`KernelModule::scratch_size()`).
    /// 2. **Fold scratch into the plan**: for every kernel that needs scratch, register a synthetic
    ///    device buffer of that size and mark it as both a read and a write of the kernel node —
    ///    the CP-SAT model then packs it into the graph pool with a single-time-step lifetime, so
    ///    scratches share memory with each other and with any graph buffer that's dead during that
    ///    step.
    /// 3. **Plan + assemble**: run [`planner::plan_raw`] on the augmented `(bufs, node accesses)`
    ///    pair and package everything into a [`GraphExe`], carrying each kernel's scratch BufId (if
    ///    any) so `run` can point the module at the pool offset via [`KernelModule::set_scratch`].
    pub fn compile(self, mut graph: GraphBuilder) -> Result<GraphExe, CompileError> {
        // Phase 1: drain and compile every structured kernel.
        //
        // `compiled` mirrors the original node order. For structured kernels
        // the module ownership is moved out and the compiled `KernelModule`
        // is stored here; non-kernel nodes are kept as-is.
        enum PreExe {
            Kernel {
                name: String,
                module: KernelModule,
                inputs: Vec<BufId>,
                outputs: Vec<BufId>,
                scratch: Option<BufId>,
            },
            Blackbox(KernelNode),
            Const(ConstNode),
            Memcpy {
                src: BufId,
                dst: BufId,
            },
            Memset {
                buf: BufId,
                val: u32,
            },
        }
        let nodes: Vec<GraphNode> = graph.nodes.drain(..).collect();
        let mut compiled: Vec<PreExe> = Vec::with_capacity(nodes.len());
        for node in nodes {
            compiled.push(match node {
                GraphNode::Kernel(k) => {
                    let name = k.module.name.clone();
                    let module = compile_and_load(k.module, &self.options)?;
                    // Allocate a synthetic scratch buffer alive during this
                    // node's step. `elem_size=1` because it's a byte pool.
                    let scratch = if module.scratch_size() > 0 {
                        let bid = graph.add_buf(BufInfo {
                            name: Some(format!("scratch<{name}>")),
                            device_type: self.device,
                            size: Quast::cst(module.scratch_size() as i64),
                            elem_size: 1,
                        });
                        Some(bid)
                    } else {
                        None
                    };
                    PreExe::Kernel {
                        name,
                        module,
                        inputs: k.inputs,
                        outputs: k.outputs,
                        scratch,
                    }
                }
                GraphNode::BlackboxKernel(k) => PreExe::Blackbox(k),
                GraphNode::Const(c) => PreExe::Const(c),
                GraphNode::Memcpy(m) => PreExe::Memcpy {
                    src: m.src,
                    dst: m.dst,
                },
                GraphNode::Memset(m) => PreExe::Memset {
                    buf: m.node,
                    val: m.val,
                },
            });
        }
        let bufs = graph.bufs.clone();

        // Phase 2: build per-node access sets, injecting scratch as both a
        // read and a write of the owning kernel (single-time-step lifetime).
        let accesses: Vec<NodeAccess> = compiled
            .iter()
            .map(|p| match p {
                PreExe::Kernel {
                    inputs,
                    outputs,
                    scratch,
                    ..
                } => {
                    let mut a = NodeAccess::default();
                    a.reads.extend(inputs.iter().copied());
                    a.writes.extend(outputs.iter().copied());
                    if let Some(sc) = scratch {
                        a.reads.push(*sc);
                        a.writes.push(*sc);
                    }
                    a
                }
                PreExe::Blackbox(k) => access_from_node(&GraphNode::BlackboxKernel(clone_kn(k))),
                PreExe::Const(c) => access_from_node(&GraphNode::Const(clone_cn(c))),
                PreExe::Memcpy { src, dst } => {
                    access_from_node(&GraphNode::Memcpy(crate::graph_ir::MemcpyNode {
                        src: *src,
                        dst: *dst,
                    }))
                }
                PreExe::Memset { buf, val } => {
                    access_from_node(&GraphNode::Memset(crate::graph_ir::MemSetNode {
                        node: *buf,
                        val: *val,
                    }))
                }
            })
            .collect();

        // Identify graph inputs / outputs from the augmented access sets,
        // excluding synthetic scratch buffers (they're single-node so they
        // have both a writer and a reader and would never qualify anyway).
        // We compute these before planning so the planner can skip them —
        // the caller supplies their storage at run time, so pool slots for
        // them would go unused and inflate the peak.
        let n_bufs = bufs.len();
        let (writers_per_buf, readers_per_buf) = writers_readers_from_accesses(&accesses, n_bufs);
        let mut input_bufs = Vec::new();
        let mut output_bufs = Vec::new();
        for (b, info) in bufs.iter().enumerate() {
            if info.device_type != self.device {
                continue;
            }
            let bid = BufId(b);
            if writers_per_buf[b].is_empty() {
                input_bufs.push(bid);
            }
            if readers_per_buf[b].is_empty() {
                output_bufs.push(bid);
            }
        }
        let exclude: Vec<BufId> = input_bufs
            .iter()
            .chain(output_bufs.iter())
            .copied()
            .collect();

        // Phase 3: plan and validate.
        let plan =
            planner::plan_raw(&bufs, &accesses, &self.env, self.device, &exclude).map_err(|e| {
                match e {
                    PlanError::UnboundSizeSymbol { .. } | PlanError::NegativeSize { .. } => {
                        CompileError::Type(format!("graph plan: {e}"))
                    }
                    PlanError::NoSolution(_) => CompileError::Runtime(format!("graph plan: {e}")),
                }
            })?;
        let sizes = evaluate_sizes_bufs(&bufs, &self.env)?;

        // Assemble ExeNodes.
        let mut nodes = Vec::with_capacity(compiled.len());
        for p in compiled {
            nodes.push(match p {
                PreExe::Kernel {
                    name,
                    module,
                    inputs,
                    outputs,
                    scratch,
                } => {
                    check_kernel_sizes(&module, &inputs, &outputs, &sizes)?;
                    if let Some(sc) = scratch {
                        // Scratch was sized from `module.scratch_size()` so
                        // this should always hold, but we double-check
                        // against the (possibly foreign-device) size
                        // resolution here.
                        if sizes[sc.0] < module.scratch_size() {
                            return Err(CompileError::Runtime(format!(
                                "kernel `{name}` scratch buffer sized {} bytes but module wants {}",
                                sizes[sc.0],
                                module.scratch_size()
                            )));
                        }
                    }
                    ExeNode::Kernel(ExeKernel {
                        name,
                        module,
                        inputs,
                        outputs,
                        scratch,
                    })
                }
                PreExe::Blackbox(k) => ExeNode::Blackbox(k),
                PreExe::Const(c) => ExeNode::Const(c),
                PreExe::Memcpy { src, dst } => ExeNode::Memcpy { src, dst },
                PreExe::Memset { buf, val } => ExeNode::Memset { buf, val },
            });
        }

        Ok(GraphExe {
            plan,
            sizes,
            nodes,
            input_bufs,
            output_bufs,
            device: self.device,
            bufs,
        })
    }
}

fn clone_kn(k: &KernelNode) -> KernelNode {
    // KernelNode's `func` is not Clone, but `access_from_node` only inspects
    // the buffer fields. Build a shallow copy with a placeholder closure.
    KernelNode {
        inputs: k.inputs.clone(),
        outputs: k.outputs.clone(),
        modifies: k.modifies.clone(),
        func: Box::new(|_, _| {}),
        name: k.name.clone(),
    }
}

fn clone_cn(c: &ConstNode) -> ConstNode {
    ConstNode {
        buf: c.buf,
        // Cheap placeholder — access_from_node only reads `buf`.
        data: ConstBuf::HostBuf(Vec::new()),
    }
}

fn writers_readers_from_accesses(
    accesses: &[NodeAccess],
    n_bufs: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut w = vec![vec![]; n_bufs];
    let mut r = vec![vec![]; n_bufs];
    for (n, a) in accesses.iter().enumerate() {
        for b in &a.writes {
            w[b.0].push(n);
        }
        for b in &a.reads {
            r[b.0].push(n);
        }
    }
    (w, r)
}

struct ExeKernel {
    name: String,
    module: KernelModule,
    inputs: Vec<BufId>,
    outputs: Vec<BufId>,
    /// Synthetic BufId for this module's private scratch, or `None` when
    /// the compiled kernel needs no scratch. Its offset in the graph pool
    /// is bound via [`KernelModule::set_scratch`] at run time.
    scratch: Option<BufId>,
}

enum ExeNode {
    Kernel(ExeKernel),
    Blackbox(KernelNode),
    Const(ConstNode),
    Memcpy { src: BufId, dst: BufId },
    Memset { buf: BufId, val: u32 },
}

/// A compiled, executable graph. Holds every JIT'd [`KernelModule`] and the
/// static memory plan; execution is stream-based against caller-provided
/// input, output and scratch buffers.
pub struct GraphExe {
    plan: MemoryPlan,
    sizes: Vec<usize>,
    nodes: Vec<ExeNode>,
    input_bufs: Vec<BufId>,
    output_bufs: Vec<BufId>,
    device: DeviceType,
    /// Preserved from the source graph for [`GraphExe::print`]: name and
    /// device_type per BufId.
    bufs: Vec<BufInfo>,
}

impl GraphExe {
    pub fn num_inputs(&self) -> usize {
        self.input_bufs.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.output_bufs.len()
    }

    /// Size in bytes of graph input `i`.
    pub fn input_size(&self, i: usize) -> usize {
        self.sizes[self.input_bufs[i].0]
    }

    /// Size in bytes of graph output `i`.
    pub fn output_size(&self, i: usize) -> usize {
        self.sizes[self.output_bufs[i].0]
    }

    pub fn input_buf_id(&self, i: usize) -> BufId {
        self.input_bufs[i]
    }

    pub fn output_buf_id(&self, i: usize) -> BufId {
        self.output_bufs[i]
    }

    /// Bytes needed in the graph-level scratch buffer passed to [`run`].
    pub fn scratch_bytes(&self) -> usize {
        self.plan.peak_bytes as usize
    }

    /// Target device the plan was built for.
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// SSA-form textual dump of the compiled graph. Nodes are printed in
    /// the planner-chosen execution order; intermediate buffer types
    /// include their concrete byte size and byte offset in the scratch
    /// pool, while graph inputs/outputs (whose storage the caller supplies
    /// at run time) show only their size.
    pub fn print(&self) -> String {
        let mut out = String::new();
        out.push_str("// GraphExe IR dump\n");
        out.push_str(&format!("// Device: {:?}\n", self.device));
        out.push_str("// Buffer types: G[I]=CUDA device I, C=CpuPaged, CP=CpuPinned;\n");
        out.push_str("//   intermediates: `T[N, offset=M]` (N bytes at M-byte scratch offset)\n");
        out.push_str("//   graph inputs/outputs: `T[N]` (caller-supplied storage)\n");
        out.push_str(&format!(
            "// Scratch pool peak: {} bytes\n",
            self.scratch_bytes()
        ));
        out.push_str(&format!("// Execution order: {:?}\n", self.plan.order));

        if !self.input_bufs.is_empty() {
            out.push_str("// Inputs (caller-supplied storage):\n");
            for &b in &self.input_bufs {
                out.push_str(&format!(
                    "//   {}: {}[{}]  // BufId({})\n",
                    self.buf_name(b),
                    crate::graph_ir::device_ty_str(self.bufs[b.0].device_type),
                    self.sizes[b.0],
                    b.0,
                ));
            }
        }
        if !self.output_bufs.is_empty() {
            out.push_str("// Outputs (caller-supplied storage):\n");
            for &b in &self.output_bufs {
                out.push_str(&format!(
                    "//   {}: {}[{}]  // BufId({})\n",
                    self.buf_name(b),
                    crate::graph_ir::device_ty_str(self.bufs[b.0].device_type),
                    self.sizes[b.0],
                    b.0,
                ));
            }
        }
        out.push('\n');

        for &node_idx in &self.plan.order {
            out.push_str(&self.format_exe_node_line(&self.nodes[node_idx]));
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

    /// Type annotation with concrete size and planned offset for
    /// intermediates, or size only for graph inputs / outputs.
    fn buf_decl(&self, id: BufId) -> String {
        let is_io = self.input_bufs.contains(&id) || self.output_bufs.contains(&id);
        let dev = crate::graph_ir::device_ty_str(self.bufs[id.0].device_type);
        let size = self.sizes[id.0];
        let ann = match (is_io, self.plan.offsets[id.0]) {
            (true, _) => format!("{size}"),
            (false, Some(off)) => format!("{size}, offset={off}"),
            (false, None) => format!("{size}"),
        };
        format!("{}: {dev}[{ann}]", self.buf_name(id))
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

    fn format_exe_node_line(&self, node: &ExeNode) -> String {
        match node {
            ExeNode::Kernel(k) => {
                let mut attrs = format!("name=\"{}\"", k.name);
                if let Some(sc) = k.scratch {
                    attrs.push_str(&format!(", scratch={}", self.buf_decl(sc)));
                }
                format!(
                    "let ({}) = Kernel({}, {});",
                    self.buf_decl_list(&k.outputs),
                    self.buf_ref_list(&k.inputs),
                    attrs,
                )
            }
            ExeNode::Blackbox(k) => {
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
            ExeNode::Const(c) => {
                let data = match &c.data {
                    ConstBuf::HostBuf(v) => format!("HostBuf(bytes={})", v.len()),
                    ConstBuf::DeviceBuf(_) => "DeviceBuf".to_string(),
                };
                format!("let ({}) = Const({data});", self.buf_decl(c.buf))
            }
            ExeNode::Memcpy { src, dst } => format!(
                "let ({}) = Memcpy({});",
                self.buf_decl(*dst),
                self.buf_name(*src),
            ),
            ExeNode::Memset { buf, val } => {
                format!("let ({}) = Memset(val={:#x});", self.buf_decl(*buf), val,)
            }
        }
    }

    /// Executes the graph on `ctx.stream`.
    ///
    /// - `inputs`: one [`DeviceBuffer<u8>`] per graph input, in the order given by
    ///   [`GraphExe::input_buf_id`].
    /// - `outputs`: one per graph output, similarly ordered; filled in by the execution.
    /// - `scratch`: byte pool of size >= [`scratch_bytes`]. Holds every intermediate device buffer
    ///   at its planned offset.
    ///
    /// The call is asynchronous on `ctx.stream`; synchronize the stream (or
    /// perform a D2H read) before reading `outputs` from the host.
    pub fn run(
        &mut self,
        ctx: &GpuDeviceCtx,
        inputs: &[DeviceBuffer<u8>],
        outputs: &mut [DeviceBuffer<u8>],
        scratch: &mut DeviceBuffer<u8>,
    ) -> Result<(), CompileError> {
        if inputs.len() != self.num_inputs() {
            return Err(CompileError::Runtime(format!(
                "graph exe: expected {} inputs, got {}",
                self.num_inputs(),
                inputs.len()
            )));
        }
        if outputs.len() != self.num_outputs() {
            return Err(CompileError::Runtime(format!(
                "graph exe: expected {} outputs, got {}",
                self.num_outputs(),
                outputs.len()
            )));
        }
        for (i, b) in inputs.iter().enumerate() {
            if b.len() < self.input_size(i) {
                return Err(CompileError::Runtime(format!(
                    "graph exe: input {i} device buffer is {} bytes, need {}",
                    b.len(),
                    self.input_size(i)
                )));
            }
        }
        for (i, b) in outputs.iter().enumerate() {
            if b.len() < self.output_size(i) {
                return Err(CompileError::Runtime(format!(
                    "graph exe: output {i} device buffer is {} bytes, need {}",
                    b.len(),
                    self.output_size(i)
                )));
            }
        }
        if scratch.len() < self.scratch_bytes() {
            return Err(CompileError::Runtime(format!(
                "graph exe: scratch is {} bytes, need {}",
                scratch.len(),
                self.scratch_bytes()
            )));
        }

        // BufId -> device pointer resolution. Graph inputs/outputs come from
        // the caller's buffers; everything else lives at scratch + offset.
        let scratch_base = scratch.as_mut_raw_ptr() as *mut u8;
        let bufid_ptr = |b: BufId| -> Result<*mut u8, CompileError> {
            if let Some(i) = self.input_bufs.iter().position(|&x| x == b) {
                return Ok(inputs[i].as_mut_raw_ptr() as *mut u8);
            }
            if let Some(i) = self.output_bufs.iter().position(|&x| x == b) {
                return Ok(outputs[i].as_mut_raw_ptr() as *mut u8);
            }
            match self.plan.offsets[b.0] {
                Some(off) => Ok(unsafe { scratch_base.add(off as usize) }),
                None => Err(CompileError::Runtime(format!(
                    "graph exe: buffer {b:?} is not on the plan's device ({:?}); \
                     cannot resolve its device pointer",
                    self.device
                ))),
            }
        };

        for &node_idx in &self.plan.order {
            match &mut self.nodes[node_idx] {
                ExeNode::Kernel(k) => {
                    for (i, &bid) in k.inputs.iter().enumerate() {
                        let ptr = bufid_ptr(bid)?;
                        let expected = k.module.input_size(i);
                        let fake = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<u8>::from_raw_parts(ptr, expected)
                        });
                        k.module.set_input(i, &fake)?;
                    }
                    for (i, &bid) in k.outputs.iter().enumerate() {
                        let ptr = bufid_ptr(bid)?;
                        let expected = k.module.output_size(i);
                        let fake = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<u8>::from_raw_parts(ptr, expected)
                        });
                        k.module.set_output(i, &fake)?;
                    }
                    if let Some(sc) = k.scratch {
                        let ptr = bufid_ptr(sc)?;
                        let want = k.module.scratch_size();
                        let fake = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<u8>::from_raw_parts(ptr, want)
                        });
                        k.module.set_scratch(&fake)?;
                    }
                    k.module.run(&ctx.stream)?;
                }
                ExeNode::Blackbox(k) => {
                    let ins: Vec<*mut ()> = k
                        .inputs
                        .iter()
                        .map(|&b| bufid_ptr(b).map(|p| p as *mut ()))
                        .collect::<Result<_, _>>()?;
                    let outs: Vec<*mut ()> = k
                        .outputs
                        .iter()
                        .map(|&b| bufid_ptr(b).map(|p| p as *mut ()))
                        .collect::<Result<_, _>>()?;
                    (k.func)(&ins, &outs);
                }
                ExeNode::Const(c) => {
                    let dst = bufid_ptr(c.buf)?;
                    let n = self.sizes[c.buf.0];
                    match &c.data {
                        ConstBuf::HostBuf(bytes) => {
                            if bytes.len() != n {
                                return Err(CompileError::Runtime(format!(
                                    "Const HostBuf for {:?} is {} bytes, buffer is {n}",
                                    c.buf,
                                    bytes.len()
                                )));
                            }
                            unsafe {
                                cuda_memcpy_on::<false, true>(
                                    dst as *mut c_void,
                                    bytes.as_ptr() as *const c_void,
                                    n,
                                    ctx,
                                )
                                .map_err(memcpy_err)?;
                            }
                        }
                        ConstBuf::DeviceBuf(src) => unsafe {
                            cuda_memcpy_on::<true, true>(
                                dst as *mut c_void,
                                src.as_raw_ptr(),
                                n,
                                ctx,
                            )
                            .map_err(memcpy_err)?;
                        },
                    }
                }
                ExeNode::Memcpy { src, dst } => {
                    let src_ptr = bufid_ptr(*src)?;
                    let dst_ptr = bufid_ptr(*dst)?;
                    let n = self.sizes[dst.0];
                    unsafe {
                        cuda_memcpy_on::<true, true>(
                            dst_ptr as *mut c_void,
                            src_ptr as *const c_void,
                            n,
                            ctx,
                        )
                        .map_err(memcpy_err)?;
                    }
                }
                ExeNode::Memset { buf, val } => {
                    let val_bytes = val.to_le_bytes();
                    if val_bytes[0] != val_bytes[1]
                        || val_bytes[0] != val_bytes[2]
                        || val_bytes[0] != val_bytes[3]
                    {
                        return Err(CompileError::Runtime(format!(
                            "Memset value {val:#x} is not byte-uniform; only byte-pattern \
                             fills are supported today"
                        )));
                    }
                    let n = self.sizes[buf.0];
                    let ptr = bufid_ptr(*buf)?;
                    let code = unsafe {
                        cudaMemsetAsync(
                            ptr as *mut c_void,
                            val_bytes[0] as i32,
                            n,
                            ctx.stream.as_raw(),
                        )
                    };
                    if code != 0 {
                        return Err(CompileError::Runtime(format!(
                            "cudaMemsetAsync failed with code {code}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

fn memcpy_err(e: openvm_cuda_common::error::MemCopyError) -> CompileError {
    CompileError::Runtime(format!("cudaMemcpy failed: {e:?}"))
}

fn evaluate_sizes_bufs(
    bufs: &[BufInfo],
    env: &BTreeMap<VarId, i64>,
) -> Result<Vec<usize>, CompileError> {
    let mut out = Vec::with_capacity(bufs.len());
    for (b, info) in bufs.iter().enumerate() {
        let mut syms = std::collections::BTreeSet::new();
        info.size.syms(&mut syms);
        for s in &syms {
            if !env.contains_key(s) {
                return Err(CompileError::Type(format!(
                    "buffer {b} references unbound symbol {s:?}"
                )));
            }
        }
        let v = info.size.eval(env);
        if v < 0 {
            return Err(CompileError::Type(format!(
                "buffer {b} evaluates to a negative size {v}"
            )));
        }
        out.push(v as usize);
    }
    Ok(out)
}

fn check_kernel_sizes(
    module: &KernelModule,
    inputs: &[BufId],
    outputs: &[BufId],
    sizes: &[usize],
) -> Result<(), CompileError> {
    if inputs.len() != module.num_inputs() {
        return Err(CompileError::Runtime(format!(
            "kernel `{}` declares {} module inputs, bound to {}",
            "<module>",
            module.num_inputs(),
            inputs.len()
        )));
    }
    if outputs.len() != module.num_outputs() {
        return Err(CompileError::Runtime(format!(
            "kernel declares {} module outputs, bound to {}",
            module.num_outputs(),
            outputs.len()
        )));
    }
    for (i, &b) in inputs.iter().enumerate() {
        if sizes[b.0] != module.input_size(i) {
            return Err(CompileError::Runtime(format!(
                "kernel input {i} size {} != bound buffer {:?} size {}",
                module.input_size(i),
                b,
                sizes[b.0]
            )));
        }
    }
    for (i, &b) in outputs.iter().enumerate() {
        if sizes[b.0] != module.output_size(i) {
            return Err(CompileError::Runtime(format!(
                "kernel output {i} size {} != bound buffer {:?} size {}",
                module.output_size(i),
                b,
                sizes[b.0]
            )));
        }
    }
    Ok(())
}
