//! Executable form of a compiled graph.
//!
//! [`GraphExe`] is the runtime output of
//! [`crate::graph_compiler::GraphCompiler::compile`]. It holds every JIT'd
//! [`KernelProgram`], the static memory plan, and the unified device pool
//! that backs every buffer at a fixed offset. Inputs are bound by eager
//! D2D copy via [`GraphExe::set_input`] or by writing directly into the
//! pool slot at the pointer returned by [`GraphExe::get_input_ptr`]
//! (skips the D2D staging buffer); outputs are read back through the
//! [`DevSlice`] views returned by [`GraphExe::get_output`]. Because every
//! node always resolves the same device addresses, a run is CUDA-graph
//! capturable and replayable.
//!
//! Feature-gated behind `planner`.

use std::{
    ffi::c_void,
    marker::PhantomData,
    mem::ManuallyDrop,
    sync::Arc,
};

use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H},
    d_buffer::{cudaMemsetAsync, DeviceBuffer},
    stream::{cudaStream_t, GpuDeviceCtx},
};

use crate::{
    graph_ir::{BufId, BufInfo, ConstBuf, ConstNode, DeviceType, KernelNode},
    passes::fusion_utils::FusionReport,
    planner::{StreamInstr, StreamMemoryPlan},
    runtime::KernelProgram,
    CompileError,
};

#[allow(non_camel_case_types)]
type cudaGraph_t = *mut c_void;
#[allow(non_camel_case_types)]
type cudaGraphExec_t = *mut c_void;

/// `cudaStreamCaptureModeThreadLocal` — capture is scoped to the calling
/// thread, so unrelated CUDA activity on other threads doesn't get pulled
/// into the graph.
const CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL: u32 = 1;

#[link(name = "cudart")]
extern "C" {
    fn cudaStreamBeginCapture(stream: cudaStream_t, mode: u32) -> i32;
    fn cudaStreamEndCapture(stream: cudaStream_t, graph: *mut cudaGraph_t) -> i32;
    fn cudaGraphInstantiateWithFlags(
        graph_exec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: u64,
    ) -> i32;
    fn cudaGraphLaunch(graph_exec: cudaGraphExec_t, stream: cudaStream_t) -> i32;
    fn cudaGraphDestroy(graph: cudaGraph_t) -> i32;
    fn cudaGraphExecDestroy(graph_exec: cudaGraphExec_t) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: cudaStream_t,
    ) -> i32;
}

/// `cudaMemcpyKind` values matching `crates/cuda-common/src/copy.rs`.
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

/// Thin raw-stream `cudaMemcpyAsync` wrapper. Prefer this in the multi-
/// stream launch loop; `cuda_memcpy_on` from cuda-common is bound to a
/// `GpuDeviceCtx` and always uses `ctx.stream`.
///
/// # Safety
/// Standard `cudaMemcpyAsync` rules: pointers must be valid for `count`
/// bytes in the appropriate memory space (host vs device) implied by
/// `kind`.
unsafe fn cuda_memcpy_async_on_raw(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: i32,
    stream: cudaStream_t,
) -> i32 {
    cudaMemcpyAsync(dst, src, count, kind, stream)
}

/// RAII owner for the pair of CUDA-graph handles returned by
/// `cudaStreamEndCapture` + `cudaGraphInstantiateWithFlags`. Both are
/// destroyed on drop.
struct CapturedGraph {
    graph: cudaGraph_t,
    graph_exec: cudaGraphExec_t,
}

unsafe impl Send for CapturedGraph {}
unsafe impl Sync for CapturedGraph {}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        if !self.graph_exec.is_null() {
            let err = unsafe { cudaGraphExecDestroy(self.graph_exec) };
            debug_assert_eq!(err, 0, "cudaGraphExecDestroy failed with code {err}");
            self.graph_exec = std::ptr::null_mut();
        }
        if !self.graph.is_null() {
            let err = unsafe { cudaGraphDestroy(self.graph) };
            debug_assert_eq!(err, 0, "cudaGraphDestroy failed with code {err}");
            self.graph = std::ptr::null_mut();
        }
    }
}

pub(crate) struct ExeKernel {
    pub(crate) name: String,
    /// Index into [`GraphExe::kernels`]: the compiled artifact this node
    /// launches. Multiple `ExeKernel`s can share a `kernel_idx` (when their
    /// source modules dedup to the same residual hash); execution is
    /// sequential so each node re-binds `set_params` / inputs / outputs on
    /// the shared program before its launch.
    pub(crate) kernel_idx: usize,
    pub(crate) inputs: Vec<BufId>,
    pub(crate) outputs: Vec<BufId>,
    /// Positional values for the kernel's runtime parameters, aligned with
    /// [`KernelProgram::params()`]'s name order. The launch loop pairs each
    /// value with its corresponding name and calls `set_symbol` before
    /// launching.
    pub(crate) set_params: Vec<i64>,
    /// Stream index this kernel launches on. `0` is `ctx.stream`; higher
    /// indices are internally-owned auxiliary streams.
    pub(crate) stream: u32,
    /// Original (post-fusion, pre-lower) source module for this node.
    /// Kept for debug tooling: the run-time trace can dump the module's
    /// HIR + the compiled `.cu` source when a specific instruction stalls,
    /// so we can identify which fused kernel is misbehaving without a
    /// separate lookup table.
    pub(crate) debug_module: Arc<crate::ir::Module>,
}

pub(crate) struct ExeBlackbox {
    pub(crate) kernel: KernelNode,
    pub(crate) stream: u32,
}

pub(crate) enum ExeNode {
    Kernel(ExeKernel),
    Blackbox(ExeBlackbox),
    Const(ConstNode),
    Memcpy {
        src: BufId,
        src_offset: usize,
        dst: BufId,
        dst_offset: usize,
        num_bytes: usize,
    },
    Memset {
        buf: BufId,
        offset: usize,
        num_bytes: usize,
        val: u32,
    },
}

/// A compiled, executable graph. Holds every JIT'd [`KernelProgram`], the
/// static memory plan and the unified device pool that backs every buffer
/// (inputs, outputs, intermediates and per-kernel scratch) at a fixed
/// offset. Bind inputs with [`Self::set_input`] (eager D2D copy into the
/// pool), execute with [`Self::run`], read outputs through
/// [`Self::get_output`]. Because every node always resolves the same
/// device addresses, a run is CUDA-graph capturable and replayable.
pub struct GraphExe {
    /// Deterministic fingerprint of the source [`GraphBuilder`] computed
    /// *before* any pass ran (see [`GraphBuilder::content_hash`]). The
    /// graph serializer stamps this on every payload and verifies it
    /// matches on the partial-restore path.
    pub(crate) graph_hash: [u8; 32],
    plan: StreamMemoryPlan,
    /// Auxiliary CUDA streams for `stream_idx >= 1`. `streams[0]` is the
    /// caller's `ctx.stream`, so it's stored as `None` and resolved at
    /// run-time. `streams[i]` for `i >= 1` is an owned non-blocking stream
    /// created once and reused for every `run()`.
    streams: Vec<Option<Arc<openvm_cuda_common::stream::CudaStream>>>,
    /// Cross-stream synchronization events. Indexed by the `event_idx`
    /// carried in `WaitOn`/`record_event`. Allocated once in `compile()`.
    events: Vec<openvm_cuda_common::stream::CudaEvent>,
    sizes: Vec<usize>,
    /// One compiled artifact per unique residual hash. `ExeKernel` nodes
    /// carry a `kernel_idx` into this vec; multiple nodes may share an
    /// entry (aliased modules that dedup to the same residual). Owned once
    /// here and borrowed `&mut` per-launch — no `Arc<Mutex<_>>` needed
    /// because execution is single-threaded.
    kernels: Vec<KernelProgram>,
    nodes: Vec<ExeNode>,
    input_bufs: Vec<BufId>,
    output_bufs: Vec<BufId>,
    /// Which inputs have been bound via [`Self::set_input`]; [`Self::run`]
    /// refuses to launch while any are missing. Bindings persist across
    /// runs (the bytes live in the pool).
    inputs_bound: Vec<bool>,
    /// The unified device pool. Allocated lazily on first
    /// [`Self::set_input`] / [`Self::run`], or supplied up front via
    /// [`Self::set_scratch`]; never reallocated afterwards so device
    /// addresses stay stable.
    pool: Option<DeviceBuffer<u8>>,
    device: DeviceType,
    /// Preserved from the source graph for [`GraphExe::print`]: name and
    /// device_type per BufId.
    bufs: Vec<BufInfo>,
    /// Number of distinct `KernelProgram`s in [`Self::kernels`]. Kernels
    /// are deduplicated first by (`Arc<ir::Module>` identity, parameter
    /// bindings) — monomorphizing parameterized modules to their residuals
    /// — and then by [`crate::module_hash::module_hash`] of the residual.
    num_unique_modules: usize,
    /// Subset of `num_unique_modules` served from the on-disk kernel cache
    /// (i.e. reused a persisted `.so` instead of running nvcc).
    num_cached_modules: usize,
    /// What the kernel-fusion pass did, `None` when it was disabled.
    fusion_report: Option<FusionReport>,
    /// Instantiated CUDA graph produced by [`Self::capture_graph`]. Reused
    /// by [`Self::launch_graph`] so we pay the capture + instantiate cost
    /// exactly once and every subsequent launch is a single `cudaGraphLaunch`.
    captured: Option<CapturedGraph>,
}

impl GraphExe {
    /// Assemble a `GraphExe` from the values produced by
    /// [`crate::graph_compiler::GraphCompiler::compile`]. Runtime-only
    /// fields (`pool`, `streams`, `events`, `inputs_bound`, `captured`)
    /// are initialized to their empty state; the pool and streams are
    /// allocated lazily on first `set_input` / `run`.
    pub(crate) fn from_compiled(
        graph_hash: [u8; 32],
        plan: StreamMemoryPlan,
        sizes: Vec<usize>,
        kernels: Vec<KernelProgram>,
        nodes: Vec<ExeNode>,
        input_bufs: Vec<BufId>,
        output_bufs: Vec<BufId>,
        device: DeviceType,
        bufs: Vec<BufInfo>,
        num_unique_modules: usize,
        num_cached_modules: usize,
        fusion_report: Option<FusionReport>,
    ) -> Self {
        let inputs_bound = vec![false; input_bufs.len()];
        Self {
            graph_hash,
            plan,
            sizes,
            kernels,
            nodes,
            input_bufs,
            output_bufs,
            inputs_bound,
            pool: None,
            device,
            bufs,
            num_unique_modules,
            num_cached_modules,
            fusion_report,
            captured: None,
            streams: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Structural fingerprint of the source [`GraphBuilder`], computed
    /// before any pass ran. Round-trips through the graph serializer;
    /// used to bind a partial payload to the [`GraphBuilder`] whose
    /// closures re-populate it on load.
    pub fn graph_hash(&self) -> &[u8; 32] {
        &self.graph_hash
    }

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

    /// Total bytes of the unified device pool: every graph buffer (inputs,
    /// outputs, intermediates and per-kernel scratch) lives at a planned
    /// offset inside it. The pool is allocated lazily on first
    /// [`Self::set_input`] / [`Self::run`]; use [`Self::set_scratch`] to
    /// supply a caller-owned arena of at least this size instead.
    pub fn scratch_bytes(&self) -> usize {
        self.plan.peak_bytes as usize
    }

    /// Target device the plan was built for.
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Planner output — stream/event assignment and per-buffer offsets.
    /// Consumed by
    /// [`crate::planner::abstract_timing::perf_est`] to estimate this
    /// plan's execution time under a captured [`crate::graph_info::GraphInfo`].
    pub fn plan(&self) -> &StreamMemoryPlan {
        &self.plan
    }

    /// Number of execution nodes in the planner-chosen order.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Whether execution node `node_idx` is a Kernel (as opposed to a
    /// Blackbox / Const / Memcpy / Memset).
    pub fn is_kernel_node(&self, node_idx: usize) -> bool {
        matches!(self.nodes[node_idx], ExeNode::Kernel(_))
    }

    /// Number of distinct compiled [`KernelProgram`]s held by this exe.
    ///
    /// Kernels are deduplicated in two passes: first by `Arc<ir::Module>`
    /// pointer identity, then by [`crate::module_hash::module_hash`].
    /// Structurally identical modules built at different call sites (so
    /// carrying different `Arc`s) share a single JIT'd artifact, which
    /// keeps this count aligned with the number of nvcc invocations the
    /// cold cache would trigger.
    pub fn num_unique_modules(&self) -> usize {
        self.num_unique_modules
    }

    /// Direct `&mut` access to the compiled artifact backing a graph
    /// kernel node. Callers can rebind a residual symbol via
    /// [`KernelProgram::set_symbol`], re-query buffer sizes, and re-run
    /// the artifact standalone — useful for driving one kernel at multiple
    /// sizes without re-JIT (the residual grid guard makes every value of
    /// the symbol correct; the block choice is perf-only).
    ///
    /// Panics if `node_idx` is out of range or does not refer to a Kernel
    /// node. Multiple kernel nodes may share one artifact; mutations
    /// through this handle affect every node that indexes the same slot.
    pub fn kernel_program(&mut self, node_idx: usize) -> &mut KernelProgram {
        let ExeNode::Kernel(k) = &self.nodes[node_idx] else {
            panic!("kernel_program: node {node_idx} is not a Kernel node");
        };
        &mut self.kernels[k.kernel_idx]
    }

    /// Scalar element type of graph input `i` — the element type of the
    /// underlying interface buffer. Delegates to the first kernel that
    /// reads this input (all readers must agree post-typecheck).
    pub fn input_type(&self, i: usize) -> crate::ir::ScalarType {
        let want = self.input_bufs[i];
        for node in &self.nodes {
            if let ExeNode::Kernel(k) = node {
                if let Some(pos) = k.inputs.iter().position(|&b| b == want) {
                    return self.kernels[k.kernel_idx].input_type(pos);
                }
            }
        }
        panic!("input_type: no kernel reads graph input {i}");
    }

    /// Scalar element type of graph output `i`. Delegates to the kernel
    /// that writes this output.
    pub fn output_type(&self, i: usize) -> crate::ir::ScalarType {
        let want = self.output_bufs[i];
        for node in &self.nodes {
            if let ExeNode::Kernel(k) = node {
                if let Some(pos) = k.outputs.iter().position(|&b| b == want) {
                    return self.kernels[k.kernel_idx].output_type(pos);
                }
            }
        }
        panic!("output_type: no kernel writes graph output {i}");
    }

    /// How many of [`Self::num_unique_modules`] were served from the on-disk
    /// [`crate::kernel_cache::KernelCache`] instead of re-running nvcc.
    pub fn num_cached_modules(&self) -> usize {
        self.num_cached_modules
    }

    /// What the kernel-fusion pass did during
    /// [`crate::graph_compiler::GraphCompiler::compile`];
    /// `None` when the pass was disabled via
    /// [`crate::graph_compiler::GraphCompiler::without_fusion`].
    pub fn fusion_report(&self) -> Option<&FusionReport> {
        self.fusion_report.as_ref()
    }

    /// SSA-form textual dump of the compiled graph. Nodes are printed in
    /// the planner-chosen execution order; every buffer type includes its
    /// concrete byte size and its byte offset in the unified device pool.
    pub fn print(&self) -> String {
        let mut out = String::new();
        out.push_str("// GraphExe IR dump\n");
        out.push_str(&format!("// Device: {:?}\n", self.device));
        out.push_str("// Buffer types: G[I]=CUDA device I, C=CpuPaged, CP=CpuPinned;\n");
        out.push_str("//   `T[N, offset=M]` = N bytes at M-byte offset in the device pool\n");
        out.push_str(&format!(
            "// Device pool size: {} bytes\n",
            self.scratch_bytes()
        ));
        out.push_str(&format!("// Execution order: {:?}\n", self.plan.order()));
        if self.plan.num_streams > 1 {
            out.push_str(&format!(
                "// Streams: {} (per-node: {:?})\n",
                self.plan.num_streams, self.plan.stream
            ));
        }

        if !self.input_bufs.is_empty() {
            out.push_str("// Inputs (registered; bound via set_input):\n");
            for &b in &self.input_bufs {
                out.push_str(&format!("//   {}  // BufId({})\n", self.buf_decl(b), b.0));
            }
        }
        if !self.output_bufs.is_empty() {
            out.push_str("// Outputs (registered; read via get_output):\n");
            for &b in &self.output_bufs {
                out.push_str(&format!("//   {}  // BufId({})\n", self.buf_decl(b), b.0));
            }
        }
        out.push('\n');

        for instr in &self.plan.instructions {
            match instr {
                StreamInstr::Node(node_idx) => {
                    out.push_str(&self.format_exe_node_line(&self.nodes[*node_idx]));
                    if self.plan.num_streams > 1 {
                        out.push_str(&format!(
                            "  // stream={}{}",
                            self.plan.stream[*node_idx],
                            match self.plan.record_event[*node_idx] {
                                Some(e) => format!(", record_event={e}"),
                                None => String::new(),
                            }
                        ));
                    }
                    out.push('\n');
                }
                StreamInstr::WaitOn(stream, event) => {
                    out.push_str(&format!("// WaitOn(stream={stream}, event={event})\n"));
                }
            }
        }
        out
    }

    fn buf_name(&self, id: BufId) -> String {
        match self.bufs[id.0].name.as_deref() {
            Some(n) => format!("%{n}"),
            None => format!("%b{}", id.0),
        }
    }

    /// Type annotation with concrete size and planned pool offset (offset
    /// omitted for off-device buffers, which have no pool slot).
    fn buf_decl(&self, id: BufId) -> String {
        let dev = crate::graph_ir::device_ty_str(self.bufs[id.0].device_type);
        let size = self.sizes[id.0];
        let ann = match self.plan.offsets[id.0] {
            Some(off) => format!("{size}, offset={off}"),
            None => format!("{size}"),
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
                let attrs = format!("name=\"{}\", kernel_idx={}", k.name, k.kernel_idx);
                format!(
                    "let ({}) = Kernel({}, {});",
                    self.buf_decl_list(&k.outputs),
                    self.buf_ref_list(&k.inputs),
                    attrs,
                )
            }
            ExeNode::Blackbox(bb) => {
                let k = &bb.kernel;
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
            ExeNode::Const(c) => {
                let data = match &c.data {
                    ConstBuf::HostBuf(v) => format!("HostBuf(bytes={})", v.len()),
                    ConstBuf::DeviceBuf(_) => "DeviceBuf".to_string(),
                };
                format!("let ({}) = Const({data});", self.buf_decl(c.buf))
            }
            ExeNode::Memcpy {
                src,
                src_offset,
                dst,
                dst_offset,
                num_bytes,
            } => format!(
                "let ({}) = Memcpy({}, src_off={src_offset}, dst_off={dst_offset}, n={num_bytes});",
                self.buf_decl(*dst),
                self.buf_name(*src),
            ),
            ExeNode::Memset {
                buf,
                offset,
                num_bytes,
                val,
            } => format!(
                "let ({}) = Memset(val={val:#x}, off={offset}, n={num_bytes});",
                self.buf_decl(*buf),
            ),
        }
    }

    /// Supplies a caller-allocated device pool of at least
    /// [`Self::scratch_bytes`] bytes instead of letting the exe allocate
    /// its own. Must precede the first [`Self::set_input`] / [`Self::run`]:
    /// the pool is what gives every buffer its stable device address, so
    /// swapping it later would break the capture-stability contract.
    ///
    /// Takes ownership — holding a borrow (or a stored raw pointer) would
    /// let the arena drop while the plan still references it.
    pub fn set_scratch(&mut self, pool: DeviceBuffer<u8>) -> Result<(), CompileError> {
        if self.pool.is_some() {
            return Err(CompileError::Runtime(
                "graph exe: pool already allocated; set_scratch must precede the first \
                 set_input/run"
                    .to_string(),
            ));
        }
        if pool.len() < self.scratch_bytes() {
            return Err(CompileError::Runtime(format!(
                "graph exe: supplied pool is {} bytes, need {}",
                pool.len(),
                self.scratch_bytes()
            )));
        }
        self.pool = Some(pool);
        Ok(())
    }

    fn ensure_pool(&mut self, ctx: &GpuDeviceCtx) {
        if self.pool.is_none() {
            self.pool = Some(DeviceBuffer::with_capacity_on(
                self.scratch_bytes().max(1),
                ctx,
            ));
        }
    }

    /// Binds graph input `i` by eagerly copying `buf`'s first
    /// [`Self::input_size`] bytes into the input's pinned pool slot
    /// (device-to-device). The exe never stores the caller's pointer, so
    /// `buf` may be dropped as soon as this returns; the bytes always land
    /// at the same device address, keeping replays capture-stable.
    ///
    /// Allocates the pool on first use (see [`Self::set_scratch`] to
    /// supply your own). The binding persists across [`Self::run`]s until
    /// overwritten by another `set_input`.
    pub fn set_input(
        &mut self,
        ctx: &GpuDeviceCtx,
        i: usize,
        buf: &DeviceBuffer<u8>,
    ) -> Result<(), CompileError> {
        if i >= self.num_inputs() {
            return Err(CompileError::Runtime(format!(
                "graph exe: set_input({i}) out of range, graph has {} inputs",
                self.num_inputs()
            )));
        }
        let need = self.input_size(i);
        if buf.len() < need {
            return Err(CompileError::Runtime(format!(
                "graph exe: input {i} device buffer is {} bytes, need {need}",
                buf.len()
            )));
        }
        self.ensure_pool(ctx);
        let dst = resolve_ptr(
            self.pool.as_ref().unwrap(),
            &self.plan.offsets,
            self.device,
            self.input_bufs[i],
        )?;
        unsafe {
            cuda_memcpy_on::<true, true>(dst as *mut c_void, buf.as_raw_ptr(), need, ctx)
                .map_err(memcpy_err)?;
        }
        self.inputs_bound[i] = true;
        Ok(())
    }

    /// Returns the device pointer to input `i`'s pool slot, allocating
    /// the pool if it hasn't been yet, and marks the input as bound.
    ///
    /// The caller is expected to write exactly [`Self::input_size`] bytes
    /// to this pointer — typically via `cudaMemcpyAsync` from a host or
    /// device source. Use this when the source lives on the host and you
    /// want to avoid the staging [`DeviceBuffer`] that [`Self::set_input`]
    /// requires: uploading directly into the pool slot saves peak device
    /// memory equal to `input_size(i)`. For device-side sources, prefer
    /// [`Self::set_input`].
    ///
    /// The returned pointer is stable for the lifetime of the pool (i.e.
    /// until the exe is dropped or [`Self::set_scratch`] replaces the
    /// pool, which is disallowed after the first alloc), so it stays
    /// valid across future `run` / `launch_graph` calls.
    pub fn get_input_ptr(
        &mut self,
        ctx: &GpuDeviceCtx,
        i: usize,
    ) -> Result<*mut c_void, CompileError> {
        if i >= self.num_inputs() {
            return Err(CompileError::Runtime(format!(
                "graph exe: get_input_ptr({i}) out of range, graph has {} inputs",
                self.num_inputs()
            )));
        }
        self.ensure_pool(ctx);
        let dst = resolve_ptr(
            self.pool.as_ref().unwrap(),
            &self.plan.offsets,
            self.device,
            self.input_bufs[i],
        )?;
        self.inputs_bound[i] = true;
        Ok(dst as *mut c_void)
    }

    /// Returns a view of graph output `i`'s pool slot. Meaningful after
    /// [`Self::run`]; the view borrows the exe, so the pool cannot be
    /// dropped while it is alive.
    ///
    /// # Panics
    ///
    /// Panics if `i` is out of range or the pool has not been allocated
    /// yet (no `set_scratch`/`set_input`/`run` has happened).
    pub fn get_output(&self, i: usize) -> DevSlice<'_> {
        assert!(
            i < self.num_outputs(),
            "graph exe: get_output({i}) out of range, graph has {} outputs",
            self.num_outputs()
        );
        let pool = self
            .pool
            .as_ref()
            .expect("graph exe: no device pool allocated yet; call set_input/run first");
        let b = self.output_bufs[i];
        let off = self.plan.offsets[b.0].expect("registered output always has a pool slot");
        DevSlice {
            ptr: unsafe { (pool.as_mut_raw_ptr() as *mut u8).add(off as usize) } as *mut c_void,
            len: self.sizes[b.0],
            _lt: PhantomData,
        }
    }

    /// Executes the graph on `ctx.stream` (stream index 0).
    ///
    /// For multi-stream plans, auxiliary streams (index >= 1) are used
    /// internally; cross-stream data dependencies are enforced by
    /// pre-recorded events (see `StreamInstr::WaitOn`). All work still
    /// completes before subsequent host-side syncs on `ctx.stream` because
    /// the plan issues implicit fork/join events around the multi-stream
    /// region (see [`Self::ensure_streams`]).
    ///
    /// Every buffer resolves to `pool + planned_offset`, so consecutive
    /// runs replay identical device addresses (CUDA-graph capturable). All
    /// inputs must have been bound via [`Self::set_input`]; bindings
    /// persist, so re-running without re-binding reuses the previous input
    /// bytes.
    ///
    /// The call is asynchronous on `ctx.stream`; synchronize the stream (or
    /// perform a D2H read such as [`DevSlice::to_host_on`]) before using
    /// the outputs on the host.
    pub fn run(&mut self, ctx: &GpuDeviceCtx) -> Result<(), CompileError> {
        if let Some(i) = self.inputs_bound.iter().position(|&bound| !bound) {
            return Err(CompileError::Runtime(format!(
                "graph exe: input {i} was never bound; call set_input first"
            )));
        }
        self.ensure_pool(ctx);
        self.ensure_streams()?;

        // Destructure so `nodes` and `kernels` can be borrowed mutably at
        // the same time (disjoint fields). `pool`/`plan`/`device` stay
        // immutable — the pool address is stable for the whole run.
        let GraphExe {
            nodes,
            kernels,
            plan,
            pool,
            device,
            sizes,
            streams,
            events,
            ..
        } = self;
        let pool = pool.as_ref().expect("pool ensured above");
        let device = *device;
        let bufid_ptr = |b: BufId| resolve_ptr(pool, &plan.offsets, device, b);

        // For multi-stream plans, fork from ctx.stream to every auxiliary
        // stream via a shared start event so their launches join the
        // CUDA-graph capture DAG (if capturing) and don't race with any
        // work still in flight on ctx.stream.
        let multi_stream = plan.num_streams > 1;
        let start_event = if multi_stream {
            let ev = openvm_cuda_common::stream::CudaEvent::new()
                .map_err(|e| CompileError::Runtime(format!("start event alloc failed: {e:?}")))?;
            ev.record_on(&ctx.stream)
                .map_err(|e| CompileError::Runtime(format!("start event record failed: {e:?}")))?;
            for aux in streams.iter().skip(1).filter_map(|s| s.as_ref()) {
                aux.wait(&ev)
                    .map_err(|e| CompileError::Runtime(format!("aux fork wait failed: {e:?}")))?;
            }
            Some(ev)
        } else {
            None
        };

        // Resolve a stream index to a raw handle, using ctx.stream for
        // index 0 and the owned auxiliary streams otherwise.
        let stream_raw = |idx: u32| -> cudaStream_t {
            if idx == 0 {
                ctx.stream.as_raw()
            } else {
                streams[idx as usize]
                    .as_ref()
                    .expect("aux stream ensured")
                    .as_raw()
            }
        };
        let stream_ref = |idx: u32| -> &openvm_cuda_common::stream::CudaStream {
            if idx == 0 {
                &ctx.stream
            } else {
                streams[idx as usize].as_ref().expect("aux stream ensured")
            }
        };

        // Optional per-instruction trace, gated by `GRAPH_EXE_TRACE=<N>`.
        // Prints one line every `N` instructions with elapsed wall time,
        // stream, and instruction kind — surfaces host-dispatch progress
        // when a large graph's warmup appears to hang.
        //
        // `GRAPH_EXE_SLOW_INSTR_MS=<ms>` additionally arms a per-instruction
        // timer: if a single dispatch takes longer than `<ms>` host-side,
        // dumps the offending instruction's kernel module (HIR + compiled
        // `.cu` source) to stderr and returns from `run` immediately. Used
        // to isolate a fused-kernel deadlock or a kernel that blocks on
        // the CUDA launch queue.
        let trace_stride: Option<usize> = std::env::var("GRAPH_EXE_TRACE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0);
        let slow_instr_ms: Option<u128> = std::env::var("GRAPH_EXE_SLOW_INSTR_MS")
            .ok()
            .and_then(|s| s.parse::<u128>().ok());
        // `GRAPH_EXE_STOP_AT_INSTR=<idx>` — dump the target instruction's
        // source module *before* dispatching it, then return. Use when a
        // dispatch is expected to block (e.g. CUDA launch queue back-pressure
        // from an earlier deadlocked kernel) so the post-dispatch
        // `SLOW_INSTR_MS` timer is unreachable.
        let stop_at_instr: Option<usize> = std::env::var("GRAPH_EXE_STOP_AT_INSTR")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        // `GRAPH_EXE_SYNC_EACH_INSTR=1` — synchronize the dispatch stream
        // after every instruction so the loop only advances once the GPU
        // has finished the previous kernel. Turns the async launch queue
        // into a serial one, so a deadlocked kernel blocks the sync — and
        // the last printed instr *is* the stuck one, unmasking the
        // upstream culprit behind launch-queue back-pressure.
        let sync_each: bool =
            std::env::var("GRAPH_EXE_SYNC_EACH_INSTR").ok().as_deref() == Some("1");
        // `GRAPH_EXE_DISPATCH_WATCHDOG_MS=<ms>` — run every instruction's
        // dispatch on a scoped worker thread and wait up to `<ms>` for it
        // to complete. If the worker doesn't finish in time, dump the
        // pre-prepared kernel module info for the stuck instruction and
        // `std::process::exit(101)` — the only safe way to abandon a
        // thread blocked inside a CUDA driver call. Only kicks in on the
        // instruction that actually blocks, so it doesn't slow the fast
        // path.
        let dispatch_watchdog_ms: Option<u64> = std::env::var("GRAPH_EXE_DISPATCH_WATCHDOG_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        let trace_t0 = std::time::Instant::now();
        let n_instr = plan.instructions.len();
        if trace_stride.is_some()
            || slow_instr_ms.is_some()
            || sync_each
            || dispatch_watchdog_ms.is_some()
        {
            eprintln!(
                "[graph_exe.run] dispatching {n_instr} stream instr(s) across {} stream(s)…",
                plan.num_streams,
            );
        }

        // Watchdog wiring. If `dispatch_watchdog_ms` is set, pre-materialize
        // one dump string per exe node (HIR + compiled CUDA source for
        // kernels; short label for Blackbox) so the watchdog can look up
        // the stuck node without racing the main thread for the borrow.
        // The main thread bumps the `Ordering::SeqCst` counters before each
        // dispatch; if the watchdog wakes up after `ms` and the counter
        // hasn't advanced, it prints the stashed dump for the currently
        // running node and `std::process::exit`s — the only safe way to
        // reap a thread blocked inside a CUDA driver call.
        let per_node_dump: std::sync::Arc<Vec<Option<String>>> = if dispatch_watchdog_ms.is_some() {
            let mut v = Vec::with_capacity(nodes.len());
            for (node_idx, exe_node) in nodes.iter().enumerate() {
                v.push(match exe_node {
                    ExeNode::Kernel(k) => Some(format!(
                        "instr Kernel node={node_idx} name=\"{}\" \
                             kernel_idx={}\n--- HIR ---\n{}\n\
                             --- compiled CUDA source ---\n{}",
                        k.name,
                        k.kernel_idx,
                        crate::dump::dump_hir(&k.debug_module),
                        kernels[k.kernel_idx].source(),
                    )),
                    ExeNode::Blackbox(bb) => Some(format!(
                        "instr Blackbox node={node_idx} name=\"{}\" \
                             (no HIR module)",
                        bb.kernel.name,
                    )),
                    _ => None,
                });
            }
            std::sync::Arc::new(v)
        } else {
            std::sync::Arc::new(Vec::new())
        };
        let current_instr_ai = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let current_node_ai = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
        let watchdog_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watchdog = dispatch_watchdog_ms.map(|ms| {
            let dump = std::sync::Arc::clone(&per_node_dump);
            let cur_instr = std::sync::Arc::clone(&current_instr_ai);
            let cur_node = std::sync::Arc::clone(&current_node_ai);
            let done = std::sync::Arc::clone(&watchdog_done);
            std::thread::spawn(move || {
                use std::sync::atomic::Ordering;
                loop {
                    if done.load(Ordering::SeqCst) {
                        return;
                    }
                    let baseline = cur_instr.load(Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(ms));
                    if done.load(Ordering::SeqCst) {
                        return;
                    }
                    let now = cur_instr.load(Ordering::SeqCst);
                    if now == baseline {
                        let nid = cur_node.load(Ordering::SeqCst);
                        eprintln!(
                            "\n[graph_exe.run] dispatch watchdog: instr {now} stuck \
                             for {ms}ms (node_idx={nid})",
                        );
                        if nid < dump.len() {
                            if let Some(d) = dump[nid].as_deref() {
                                eprintln!("{d}");
                            }
                        }
                        eprintln!("[graph_exe.run] dispatch watchdog: exit(101)");
                        std::process::exit(101);
                    }
                }
            })
        });

        for (i_idx, instr) in plan.instructions.iter().enumerate() {
            let desc = if trace_stride.is_some() || slow_instr_ms.is_some() {
                Some(match instr {
                    StreamInstr::WaitOn(s, e) => {
                        format!("WaitOn(stream={s}, event={e})")
                    }
                    StreamInstr::Node(node_idx) => match &nodes[*node_idx] {
                        ExeNode::Kernel(k) => format!(
                            "Kernel node={node_idx} stream={} name=\"{}\"",
                            plan.stream[*node_idx], k.name,
                        ),
                        ExeNode::Blackbox(bb) => format!(
                            "Blackbox node={node_idx} stream={} name=\"{}\"",
                            plan.stream[*node_idx], bb.kernel.name,
                        ),
                        ExeNode::Const(c) => format!(
                            "Const node={node_idx} stream={} buf={:?}",
                            plan.stream[*node_idx], c.buf,
                        ),
                        ExeNode::Memcpy {
                            src,
                            dst,
                            num_bytes,
                            ..
                        } => format!(
                            "Memcpy node={node_idx} stream={} {:?}->{:?} {num_bytes}B",
                            plan.stream[*node_idx], src, dst,
                        ),
                        ExeNode::Memset {
                            buf,
                            num_bytes,
                            val,
                            ..
                        } => format!(
                            "Memset node={node_idx} stream={} buf={:?} {num_bytes}B val={val:#x}",
                            plan.stream[*node_idx], buf,
                        ),
                    },
                })
            } else {
                None
            };
            if let (Some(stride), Some(d)) = (trace_stride, desc.as_ref()) {
                if i_idx % stride == 0 {
                    eprintln!(
                        "[graph_exe.run] instr {i_idx}/{n_instr} at {:>7.2}s: {d}",
                        trace_t0.elapsed().as_secs_f64(),
                    );
                }
            }
            if Some(i_idx) == stop_at_instr {
                eprintln!(
                    "\n[graph_exe.run] STOP_AT_INSTR fired at instr {i_idx}/{n_instr}: {}",
                    desc.as_deref().unwrap_or("(no description)"),
                );
                if let StreamInstr::Node(node_idx) = instr {
                    if let ExeNode::Kernel(k) = &nodes[*node_idx] {
                        eprintln!(
                            "\n--- HIR module `{}` (inputs={:?}, set_params={:?}) ---",
                            k.debug_module.name,
                            k.debug_module.builder.inputs(),
                            k.set_params,
                        );
                        eprintln!("{}", crate::dump::dump_hir(&k.debug_module));
                        eprintln!(
                            "\n--- compiled CUDA source (kernel_idx={}) ---",
                            k.kernel_idx,
                        );
                        eprintln!("{}", kernels[k.kernel_idx].source());
                    } else if let ExeNode::Blackbox(bb) = &nodes[*node_idx] {
                        eprintln!(
                            "(Blackbox node `{}` has no HIR module to dump)",
                            bb.kernel.name,
                        );
                    }
                }
                eprintln!("[graph_exe.run] returning early before dispatching instr {i_idx}");
                return Ok(());
            }
            let instr_t0 = if slow_instr_ms.is_some() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            if dispatch_watchdog_ms.is_some() {
                use std::sync::atomic::Ordering;
                current_instr_ai.store(i_idx, Ordering::SeqCst);
                current_node_ai.store(
                    match instr {
                        StreamInstr::Node(nid) => *nid,
                        _ => usize::MAX,
                    },
                    Ordering::SeqCst,
                );
            }
            match *instr {
                StreamInstr::WaitOn(s, e) => {
                    let ev = &events[e];
                    stream_ref(s as u32).wait(ev).map_err(|err| {
                        CompileError::Runtime(format!(
                            "WaitOn(stream={s}, event={e}) failed: {err:?}"
                        ))
                    })?;
                }
                StreamInstr::Node(node_idx) => {
                    let s_raw = stream_raw(plan.stream[node_idx]);
                    match &mut nodes[node_idx] {
                        ExeNode::Kernel(k) => {
                            let m = &mut kernels[k.kernel_idx];
                            let names: Vec<String> = m.params().to_vec();
                            for (name, &v) in names.iter().zip(k.set_params.iter()) {
                                m.set_symbol(name, v);
                            }
                            for (i, &bid) in k.inputs.iter().enumerate() {
                                let ptr = bufid_ptr(bid)?;
                                let expected = m.input_size(i);
                                let fake = ManuallyDrop::new(unsafe {
                                    DeviceBuffer::<u8>::from_raw_parts(ptr, expected)
                                });
                                m.set_input(i, &fake)?;
                            }
                            for (i, &bid) in k.outputs.iter().enumerate() {
                                let ptr = bufid_ptr(bid)?;
                                let expected = m.output_size(i);
                                let fake = ManuallyDrop::new(unsafe {
                                    DeviceBuffer::<u8>::from_raw_parts(ptr, expected)
                                });
                                m.set_output(i, &fake)?;
                            }
                            m.run(stream_ref(k.stream))?;
                        }
                        ExeNode::Blackbox(bb) => {
                            let k = &bb.kernel;
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
                            (k.func)(&ins, &outs, stream_raw(bb.stream));
                        }
                        ExeNode::Const(c) => {
                            let dst = bufid_ptr(c.buf)?;
                            let n = sizes[c.buf.0];
                            match &c.data {
                                ConstBuf::HostBuf(bytes) => {
                                    if bytes.len() != n {
                                        return Err(CompileError::Runtime(format!(
                                            "Const HostBuf for {:?} is {} bytes, buffer is {n}",
                                            c.buf,
                                            bytes.len()
                                        )));
                                    }
                                    let code = unsafe {
                                        cuda_memcpy_async_on_raw(
                                            dst as *mut c_void,
                                            bytes.as_ptr() as *const c_void,
                                            n,
                                            CUDA_MEMCPY_HOST_TO_DEVICE,
                                            s_raw,
                                        )
                                    };
                                    if code != 0 {
                                        return Err(CompileError::Runtime(format!(
                                            "cudaMemcpyAsync H2D failed with code {code}"
                                        )));
                                    }
                                }
                                ConstBuf::DeviceBuf(src) => {
                                    let code = unsafe {
                                        cuda_memcpy_async_on_raw(
                                            dst as *mut c_void,
                                            src.as_raw_ptr(),
                                            n,
                                            CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                            s_raw,
                                        )
                                    };
                                    if code != 0 {
                                        return Err(CompileError::Runtime(format!(
                                            "cudaMemcpyAsync D2D failed with code {code}"
                                        )));
                                    }
                                }
                            }
                        }
                        ExeNode::Memcpy {
                            src,
                            src_offset,
                            dst,
                            dst_offset,
                            num_bytes,
                        } => {
                            let src_ptr = bufid_ptr(*src)?;
                            let dst_ptr = bufid_ptr(*dst)?;
                            let code = unsafe {
                                openvm_cuda_common::error::check(cuda_memcpy_async_on_raw(
                                    dst_ptr.add(*dst_offset) as *mut c_void,
                                    src_ptr.add(*src_offset) as *const c_void,
                                    *num_bytes,
                                    CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                    s_raw,
                                ))
                            };
                            code.map_err(|e| {
                                CompileError::Runtime(format!("cudaMemcpyAsync D2D: {e:?}"))
                            })?;
                        }
                        ExeNode::Memset {
                            buf,
                            offset,
                            num_bytes,
                            val,
                        } => {
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
                            let ptr = bufid_ptr(*buf)?;
                            let code = unsafe {
                                cudaMemsetAsync(
                                    ptr.add(*offset) as *mut c_void,
                                    val_bytes[0] as i32,
                                    *num_bytes,
                                    s_raw,
                                )
                            };
                            if code != 0 {
                                return Err(CompileError::Runtime(format!(
                                    "cudaMemsetAsync failed with code {code}"
                                )));
                            }
                        }
                    }
                    if let Some(e) = plan.record_event[node_idx] {
                        let s = plan.stream[node_idx];
                        events[e as usize].record_on(stream_ref(s)).map_err(|err| {
                            CompileError::Runtime(format!(
                                "record_event({e}) on stream {s} failed: {err:?}"
                            ))
                        })?;
                    }
                }
            }
            if sync_each {
                // Sync every configured stream — a stuck kernel blocks here
                // and pins the last printed instr as the culprit.
                ctx.stream.synchronize().map_err(|e| {
                    CompileError::Runtime(format!("sync_each ctx.stream failed: {e:?}"))
                })?;
                for aux in streams.iter().skip(1).filter_map(|s| s.as_ref()) {
                    aux.synchronize().map_err(|e| {
                        CompileError::Runtime(format!("sync_each aux stream failed: {e:?}"))
                    })?;
                }
            }
            if let (Some(threshold_ms), Some(t0), Some(d)) =
                (slow_instr_ms, instr_t0, desc.as_ref())
            {
                let elapsed_ms = t0.elapsed().as_millis();
                if elapsed_ms > threshold_ms {
                    eprintln!(
                        "\n[graph_exe.run] SLOW instr {i_idx}/{n_instr} took {elapsed_ms} ms \
                         (threshold {threshold_ms} ms): {d}"
                    );
                    // For Kernel nodes, dump the source module (HIR) plus the
                    // compiled CUDA source of the deduped artifact this
                    // instruction launched. `debug_module` is the pre-lower
                    // HIR; kernels[kernel_idx].source() is the emitted .cu.
                    if let StreamInstr::Node(node_idx) = instr {
                        if let ExeNode::Kernel(k) = &nodes[*node_idx] {
                            eprintln!(
                                "\n--- HIR module `{}` (params={:?}, set_params={:?}) ---",
                                k.debug_module.name,
                                k.debug_module.builder.inputs(),
                                k.set_params,
                            );
                            eprintln!("{}", crate::dump::dump_hir(&k.debug_module));
                            eprintln!(
                                "\n--- compiled CUDA source (kernel_idx={}) ---",
                                k.kernel_idx,
                            );
                            eprintln!("{}", kernels[k.kernel_idx].source());
                        } else if let ExeNode::Blackbox(bb) = &nodes[*node_idx] {
                            eprintln!(
                                "(Blackbox node `{}` has no HIR module to dump)",
                                bb.kernel.name,
                            );
                        }
                    }
                    eprintln!("[graph_exe.run] returning early from run() after slow instr dump");
                    return Ok(());
                }
            }
        }

        if trace_stride.is_some() {
            eprintln!(
                "[graph_exe.run] all {n_instr} instr(s) dispatched in {:>7.2}s (host time)",
                trace_t0.elapsed().as_secs_f64(),
            );
        }
        // Retire the dispatch watchdog: signal done and join. The join is
        // trivial once the watchdog observes `done`.
        if let Some(h) = watchdog {
            use std::sync::atomic::Ordering;
            watchdog_done.store(true, Ordering::SeqCst);
            let _ = h.join();
        }

        // Join every auxiliary stream back into ctx.stream so callers can
        // sync on ctx.stream and observe all work.
        if let Some(_ev0) = start_event {
            for aux in streams.iter().skip(1).filter_map(|s| s.as_ref()) {
                let e = openvm_cuda_common::stream::CudaEvent::new().map_err(|err| {
                    CompileError::Runtime(format!("join event alloc failed: {err:?}"))
                })?;
                e.record_on(aux).map_err(|err| {
                    CompileError::Runtime(format!("join event record failed: {err:?}"))
                })?;
                ctx.stream.wait(&e).map_err(|err| {
                    CompileError::Runtime(format!("join wait on ctx.stream failed: {err:?}"))
                })?;
            }
        }
        Ok(())
    }

    /// Time each exe node's dispatch on `ctx.stream` and return a
    /// [`GraphInfo`] with per-node sample mean + sample standard
    /// deviation (in ms) across `num_iters` iterations. `num_warmup`
    /// un-timed iterations precede the timed pass so first-launch driver
    /// init doesn't leak into the samples.
    ///
    /// The `set_inputs` closure is called once before warmup so the
    /// caller can bind graph inputs (or leave them bound from a prior
    /// run). It receives the exe by mutable reference so it can call
    /// [`Self::set_input`] / [`Self::get_input_ptr`] etc.
    ///
    /// # Timing model
    ///
    /// Each node's cost is measured with a pair of CUDA events
    /// straddling its dispatch — `cudaEventElapsedTime` reports the
    /// device-side interval between them, so per-node numbers reflect
    /// GPU work exclusive of Rust host overhead. All dispatches happen
    /// on `ctx.stream` in [`crate::planner::StreamMemoryPlan::instructions`]
    /// order (with `WaitOn` entries skipped — the single serial stream
    /// makes cross-stream syncs unnecessary). Multi-stream plans still
    /// work, but the per-node timings reflect isolated cost, not the
    /// overlapped multi-stream schedule.
    ///
    /// The `total_ms_*` fields are host wall-clock per iteration, which
    /// includes event record overhead and the final `stream.synchronize`.
    pub fn collect_graph_info<F>(
        &mut self,
        ctx: &GpuDeviceCtx,
        mut set_inputs: F,
        num_warmup: usize,
        num_iters: usize,
    ) -> Result<crate::graph_info::GraphInfo, CompileError>
    where
        F: FnMut(&mut GraphExe, &GpuDeviceCtx) -> Result<(), CompileError>,
    {
        use openvm_cuda_common::stream::CudaEvent;

        use crate::graph_info::{mean_and_sample_std, GraphInfo, NodeKind, NodeTiming};

        set_inputs(self, ctx)?;
        if let Some(i) = self.inputs_bound.iter().position(|&b| !b) {
            return Err(CompileError::Runtime(format!(
                "graph exe: input {i} was never bound; set_inputs must bind every input"
            )));
        }
        self.ensure_pool(ctx);

        // Ordered list of (instr_pos, node_idx). Skip WaitOn: single-
        // stream dispatch subsumes cross-stream syncs.
        let node_order: Vec<usize> = self
            .plan
            .instructions
            .iter()
            .filter_map(|i| match i {
                StreamInstr::Node(idx) => Some(*idx),
                StreamInstr::WaitOn(_, _) => None,
            })
            .collect();
        let n_nodes = node_order.len();

        // Warmup — silence stream and driver init before the timed pass.
        // Re-run `set_inputs` before each iteration in case the graph
        // mutates any of its registered inputs in place (blackbox
        // `carried_outputs`); the closure is a no-op for pure inputs.
        for _ in 0..num_warmup {
            set_inputs(self, ctx)?;
            for &node_idx in &node_order {
                self.dispatch_node_on_stream(ctx, node_idx)?;
            }
            ctx.stream.synchronize().map_err(|e| {
                CompileError::Runtime(format!("collect_graph_info warmup sync: {e:?}"))
            })?;
        }

        // Timed pass. Per-node CUDA event pairs are freshly allocated
        // each iteration; samples are indexed by exe-node index (which
        // is also the source `GraphBuilder.nodes` index at compile
        // time), so a cytoscape dump can attach each `NodeTiming` by
        // index directly.
        let n_exe_nodes = self.nodes.len();
        let mut per_node_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(num_iters); n_exe_nodes];
        let mut total_samples: Vec<f64> = Vec::with_capacity(num_iters);
        for _iter in 0..num_iters {
            // Same rationale as the warmup loop above — refresh any
            // in-place-mutated inputs between iterations.
            set_inputs(self, ctx)?;
            let starts: Vec<CudaEvent> = (0..n_nodes)
                .map(|_| {
                    CudaEvent::new().map_err(|e| {
                        CompileError::Runtime(format!(
                            "collect_graph_info start event alloc: {e:?}"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            let ends: Vec<CudaEvent> = (0..n_nodes)
                .map(|_| {
                    CudaEvent::new().map_err(|e| {
                        CompileError::Runtime(format!("collect_graph_info end event alloc: {e:?}"))
                    })
                })
                .collect::<Result<_, _>>()?;

            let iter_t0 = std::time::Instant::now();
            for (i, &node_idx) in node_order.iter().enumerate() {
                starts[i]
                    .record_on(&ctx.stream)
                    .map_err(|e| CompileError::Runtime(format!("record start event: {e:?}")))?;
                self.dispatch_node_on_stream(ctx, node_idx)?;
                ends[i]
                    .record_on(&ctx.stream)
                    .map_err(|e| CompileError::Runtime(format!("record end event: {e:?}")))?;
            }
            ctx.stream.synchronize().map_err(|e| {
                CompileError::Runtime(format!("collect_graph_info timed sync: {e:?}"))
            })?;
            total_samples.push(iter_t0.elapsed().as_secs_f64() * 1e3);

            for (i, &node_idx) in node_order.iter().enumerate() {
                let ms = starts[i]
                    .elapsed_ms(&ends[i])
                    .map_err(|e| CompileError::Runtime(format!("cudaEventElapsedTime: {e:?}")))?;
                per_node_samples[node_idx].push(ms as f64);
            }
        }

        let (total_mean, total_std) = mean_and_sample_std(&total_samples);
        let nodes = (0..n_exe_nodes)
            .map(|node_idx| {
                let (kind, name) = match &self.nodes[node_idx] {
                    ExeNode::Kernel(k) => (NodeKind::Kernel, k.name.clone()),
                    ExeNode::Blackbox(bb) => (NodeKind::Blackbox, bb.kernel.name.clone()),
                    ExeNode::Const(_) => (NodeKind::Const, String::from("const")),
                    ExeNode::Memcpy { .. } => (NodeKind::Memcpy, String::from("memcpy")),
                    ExeNode::Memset { .. } => (NodeKind::Memset, String::from("memset")),
                };
                let (mean, std) = mean_and_sample_std(&per_node_samples[node_idx]);
                NodeTiming {
                    kind,
                    name,
                    mean_ms: mean,
                    std_ms: std,
                }
            })
            .collect();
        Ok(GraphInfo {
            graph_hash: self.graph_hash,
            num_warmup,
            num_iters,
            nodes,
            total_ms_mean: total_mean,
            total_ms_std: total_std,
        })
    }

    /// Dispatches exe node `node_idx` on `ctx.stream`. Mirrors the per-
    /// variant work in [`Self::run`] but ignores the plan's stream
    /// assignments so [`Self::collect_graph_info`] can measure each
    /// node's isolated cost.
    fn dispatch_node_on_stream(
        &mut self,
        ctx: &GpuDeviceCtx,
        node_idx: usize,
    ) -> Result<(), CompileError> {
        let GraphExe {
            nodes,
            kernels,
            plan,
            pool,
            device,
            sizes,
            ..
        } = self;
        let pool = pool.as_ref().expect("pool ensured by caller");
        let device = *device;
        let bufid_ptr = |b: BufId| resolve_ptr(pool, &plan.offsets, device, b);
        let s_raw = ctx.stream.as_raw();
        match &mut nodes[node_idx] {
            ExeNode::Kernel(k) => {
                let m = &mut kernels[k.kernel_idx];
                let names: Vec<String> = m.params().to_vec();
                for (name, &v) in names.iter().zip(k.set_params.iter()) {
                    m.set_symbol(name, v);
                }
                for (i, &bid) in k.inputs.iter().enumerate() {
                    let ptr = bufid_ptr(bid)?;
                    let expected = m.input_size(i);
                    let fake = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<u8>::from_raw_parts(ptr, expected)
                    });
                    m.set_input(i, &fake)?;
                }
                for (i, &bid) in k.outputs.iter().enumerate() {
                    let ptr = bufid_ptr(bid)?;
                    let expected = m.output_size(i);
                    let fake = ManuallyDrop::new(unsafe {
                        DeviceBuffer::<u8>::from_raw_parts(ptr, expected)
                    });
                    m.set_output(i, &fake)?;
                }
                m.run(&ctx.stream)?;
            }
            ExeNode::Blackbox(bb) => {
                let k = &bb.kernel;
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
                (k.func)(&ins, &outs, s_raw);
            }
            ExeNode::Const(c) => {
                let dst = bufid_ptr(c.buf)?;
                let n = sizes[c.buf.0];
                match &c.data {
                    ConstBuf::HostBuf(bytes) => {
                        if bytes.len() != n {
                            return Err(CompileError::Runtime(format!(
                                "Const HostBuf for {:?} is {} bytes, buffer is {n}",
                                c.buf,
                                bytes.len()
                            )));
                        }
                        let code = unsafe {
                            cuda_memcpy_async_on_raw(
                                dst as *mut c_void,
                                bytes.as_ptr() as *const c_void,
                                n,
                                CUDA_MEMCPY_HOST_TO_DEVICE,
                                s_raw,
                            )
                        };
                        if code != 0 {
                            return Err(CompileError::Runtime(format!(
                                "cudaMemcpyAsync H2D failed with code {code}"
                            )));
                        }
                    }
                    ConstBuf::DeviceBuf(src) => {
                        let code = unsafe {
                            cuda_memcpy_async_on_raw(
                                dst as *mut c_void,
                                src.as_raw_ptr(),
                                n,
                                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                s_raw,
                            )
                        };
                        if code != 0 {
                            return Err(CompileError::Runtime(format!(
                                "cudaMemcpyAsync D2D failed with code {code}"
                            )));
                        }
                    }
                }
            }
            ExeNode::Memcpy {
                src,
                src_offset,
                dst,
                dst_offset,
                num_bytes,
            } => {
                let src_ptr = bufid_ptr(*src)?;
                let dst_ptr = bufid_ptr(*dst)?;
                let code = unsafe {
                    openvm_cuda_common::error::check(cuda_memcpy_async_on_raw(
                        dst_ptr.add(*dst_offset) as *mut c_void,
                        src_ptr.add(*src_offset) as *const c_void,
                        *num_bytes,
                        CUDA_MEMCPY_DEVICE_TO_DEVICE,
                        s_raw,
                    ))
                };
                code.map_err(|e| CompileError::Runtime(format!("cudaMemcpyAsync D2D: {e:?}")))?;
            }
            ExeNode::Memset {
                buf,
                offset,
                num_bytes,
                val,
            } => {
                let val_bytes = val.to_le_bytes();
                if val_bytes[0] != val_bytes[1]
                    || val_bytes[0] != val_bytes[2]
                    || val_bytes[0] != val_bytes[3]
                {
                    return Err(CompileError::Runtime(format!(
                        "Memset value {val:#x} is not byte-uniform"
                    )));
                }
                let ptr = bufid_ptr(*buf)?;
                let code = unsafe {
                    cudaMemsetAsync(
                        ptr.add(*offset) as *mut c_void,
                        val_bytes[0] as i32,
                        *num_bytes,
                        s_raw,
                    )
                };
                if code != 0 {
                    return Err(CompileError::Runtime(format!(
                        "cudaMemsetAsync failed with code {code}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Allocates auxiliary streams and events lazily on the first
    /// [`Self::run`]. Slot 0 stays `None` and resolves to `ctx.stream` at
    /// run-time; slots >= 1 are internally-owned non-blocking streams.
    fn ensure_streams(&mut self) -> Result<(), CompileError> {
        if self.streams.len() != self.plan.num_streams as usize {
            let mut new = Vec::with_capacity(self.plan.num_streams as usize);
            new.push(None);
            for _ in 1..self.plan.num_streams {
                let s =
                    openvm_cuda_common::stream::CudaStream::new_non_blocking().map_err(|e| {
                        CompileError::Runtime(format!("aux stream alloc failed: {e:?}"))
                    })?;
                new.push(Some(Arc::new(s)));
            }
            self.streams = new;
        }
        if self.events.len() != self.plan.num_events as usize {
            let mut evs = Vec::with_capacity(self.plan.num_events as usize);
            for _ in 0..self.plan.num_events {
                evs.push(
                    openvm_cuda_common::stream::CudaEvent::new()
                        .map_err(|e| CompileError::Runtime(format!("event alloc failed: {e:?}")))?,
                );
            }
            self.events = evs;
        }
        Ok(())
    }

    /// Captures the CUDA work enqueued by [`Self::run`] into a replayable
    /// CUDA graph and caches it internally. Subsequent [`Self::launch_graph`]
    /// calls replay the cached graph in a single `cudaGraphLaunch` — no host
    /// per-node dispatch overhead.
    ///
    /// The pool must already be allocated and every input bound (both are
    /// enforced by the wrapped [`Self::run`]). The captured graph is bound
    /// to `ctx.stream` for the duration of capture, but any stream is legal
    /// at launch time. Re-capturing (calling again on an already-captured
    /// exe) replaces the cached graph.
    pub fn capture_graph(&mut self, ctx: &GpuDeviceCtx) -> Result<(), CompileError> {
        let stream = ctx.stream.as_raw();
        // The stream must be idle before `cudaStreamBeginCapture`.
        ctx.stream.synchronize().map_err(|e| {
            CompileError::Runtime(format!("stream sync before capture failed: {e:?}"))
        })?;
        let t_trace = std::time::Instant::now();
        let code = unsafe { cudaStreamBeginCapture(stream, CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL) };
        if code != 0 {
            return Err(CompileError::Runtime(format!(
                "cudaStreamBeginCapture failed with code {code}"
            )));
        }
        // Enqueue every node under capture. If this fails, we still need to
        // end capture to leave the stream in a valid (non-capturing) state.
        let run_result = self.run(ctx);
        let mut graph: cudaGraph_t = std::ptr::null_mut();
        let end_code = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        run_result?;
        if end_code != 0 {
            return Err(CompileError::Runtime(format!(
                "cudaStreamEndCapture failed with code {end_code}"
            )));
        }
        let trace_ms = t_trace.elapsed().as_secs_f64() * 1e3;
        let t_inst = std::time::Instant::now();
        let mut graph_exec: cudaGraphExec_t = std::ptr::null_mut();
        let inst_code = unsafe { cudaGraphInstantiateWithFlags(&mut graph_exec, graph, 0) };
        if inst_code != 0 {
            unsafe {
                let _ = cudaGraphDestroy(graph);
            }
            return Err(CompileError::Runtime(format!(
                "cudaGraphInstantiateWithFlags failed with code {inst_code}"
            )));
        }
        eprintln!(
            "[capture] traced {} node(s) on {} stream(s) in {trace_ms:.1} ms, \
             instantiated in {:.1} ms",
            self.nodes.len(),
            self.plan.num_streams,
            t_inst.elapsed().as_secs_f64() * 1e3,
        );
        // Drop any previously-cached graph before overwriting.
        self.captured = Some(CapturedGraph { graph, graph_exec });
        Ok(())
    }

    /// Replays the cached CUDA graph on `ctx.stream`. Captures it first via
    /// [`Self::capture_graph`] if no graph has been captured yet.
    pub fn launch_graph(&mut self, ctx: &GpuDeviceCtx) -> Result<(), CompileError> {
        if self.captured.is_none() {
            self.capture_graph(ctx)?;
        }
        let cg = self
            .captured
            .as_ref()
            .expect("captured graph populated above");
        let code = unsafe { cudaGraphLaunch(cg.graph_exec, ctx.stream.as_raw()) };
        if code != 0 {
            return Err(CompileError::Runtime(format!(
                "cudaGraphLaunch failed with code {code}"
            )));
        }
        Ok(())
    }
}

/// Resolves a buffer's device address inside the unified pool.
fn resolve_ptr(
    pool: &DeviceBuffer<u8>,
    offsets: &[Option<u64>],
    device: DeviceType,
    b: BufId,
) -> Result<*mut u8, CompileError> {
    match offsets[b.0] {
        Some(off) => Ok(unsafe { (pool.as_mut_raw_ptr() as *mut u8).add(off as usize) }),
        None => Err(CompileError::Runtime(format!(
            "graph exe: buffer {b:?} has no pool slot on the plan's device ({device:?}); \
             cannot resolve its device pointer"
        ))),
    }
}

/// A borrowed view of a device-memory range inside a [`GraphExe`]'s pool,
/// returned by [`GraphExe::get_output`]. The lifetime ties the raw device
/// pointer to the exe so the pool cannot be dropped or replaced while a
/// view is alive.
pub struct DevSlice<'a> {
    ptr: *mut c_void,
    len: usize,
    _lt: PhantomData<&'a ()>,
}

impl DevSlice<'_> {
    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer to the start of the range.
    pub fn as_raw_ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn as_mut_raw_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Synchronously copies the range back to the host.
    pub fn to_host_on(&self, ctx: &GpuDeviceCtx) -> Result<Vec<u8>, CompileError> {
        let fake = ManuallyDrop::new(unsafe {
            DeviceBuffer::<u8>::from_raw_parts(self.ptr as *mut u8, self.len)
        });
        fake.to_host_on(ctx).map_err(memcpy_err)
    }
}

fn memcpy_err(e: openvm_cuda_common::error::MemCopyError) -> CompileError {
    CompileError::Runtime(format!("cudaMemcpy failed: {e:?}"))
}
