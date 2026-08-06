//! End-to-end graph compilation and execution.
//!
//! [`GraphCompiler`] consumes a [`GraphBuilder`], validates its registered
//! input/output interface, plans memory via [`crate::planner::plan_raw`],
//! compiles every [`GraphNode::Kernel`]'s module through
//! [`crate::module_compiler::ModuleCompiler`], and packages the whole thing into a
//! [`GraphExe`]. Every buffer — inputs, outputs, and graph-level
//! intermediates — lives at a fixed offset inside one device pool, so
//! consecutive [`GraphExe::run`]s replay identical device addresses (the
//! CUDA-graph-capture contract). Inputs are bound by eager D2D copy via
//! [`GraphExe::set_input`]; outputs are read back through the [`DevSlice`]
//! views returned by [`GraphExe::get_output`].
//!
//! Feature-gated behind `planner` (needs the CP-SAT planner + OR-Tools).

use std::{
    collections::{BTreeMap, HashMap},
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

use crate::{
    graph_ir::{
        classify_buf_uses, for_each_unique_kernel, kernel_at, kernel_at_mut, BufId, BufInfo,
        ConstBuf, ConstNode, DeviceType, GraphBuilder, GraphNode, KernelNode,
    },
    ir::{self, VarId},
    kernel_cache::KernelCache,
    module_compiler::ModuleCompiler,
    module_hash::module_hash,
    passes::{
        check_accesses::check_module_accesses,
        fusion::{fuse_graph, renumber_module, FusionOptions, FusionReport},
        monomorphize::{block_size_policy, monomorphize_for_graph},
        type_infer,
    },
    planner::{self, access_from_node, MemoryPlan, NodeAccess, PlanError, SchedulerMode},
    quast::Quast,
    runtime::KernelProgram,
    CompileError,
};

/// Builder-pattern compiler that plans a graph and JITs its structured
/// kernels.
///
/// ```text
/// let exe = GraphCompiler::new()
///     .device(DeviceType::Cuda(0))
///     .symbol(n_bytes_sym, 4096)
///     .arch("sm_120")
///     .compile(graph)?;
/// ```
pub struct GraphCompiler {
    device: DeviceType,
    env: BTreeMap<VarId, i64>,
    /// Per-kernel backend used to lower + codegen every unique residual
    /// module. Kernel-compile knobs (`arch`, `nvcc`, `dump_dir`, ...) route
    /// through this — see the passthrough setters on `GraphCompiler`.
    module_compiler: ModuleCompiler,
    scheduler: SchedulerMode,
    /// On-disk cache queried before hitting nvcc; kernels found here skip
    /// compilation entirely. `None` disables the cache. Defaults to a shared
    /// `~/.openvm/kernel_cache` with the [`KernelCache`] defaults.
    kernel_cache: Option<Arc<KernelCache>>,
    /// Kernel-fusion pass tunables; `None` disables the pass. Defaults to
    /// [`FusionOptions::default`].
    fusion: Option<FusionOptions>,
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
            module_compiler: ModuleCompiler::new(),
            scheduler: SchedulerMode::default(),
            kernel_cache: Some(Arc::new(KernelCache::new())),
            fusion: Some(FusionOptions::default()),
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

    // ---------- ModuleCompiler passthrough setters ----------
    //
    // Every kernel-compile knob is exposed on the graph compiler for
    // ergonomics; each routes to the nested `ModuleCompiler`.

    /// GPU architecture, e.g. `sm_120` or `native`.
    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.module_compiler.set_arch(arch);
        self
    }

    /// Directory to write per-pass IR dumps into. `None` disables dumping.
    pub fn dump_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.module_compiler.set_dump_dir(dir);
        self
    }

    /// Amount of per-pass IR to dump when `dump_dir` is set.
    pub fn verbosity(mut self, v: crate::runtime::Verbosity) -> Self {
        self.module_compiler.set_verbosity(v);
        self
    }

    /// Enables the (expensive) exhaustive access-bound check per compiled
    /// `(module, bindings)` pair.
    pub fn check_accesses(mut self, on: bool) -> Self {
        self.module_compiler.set_check_accesses(on);
        self
    }

    /// Path to the nvcc binary.
    pub fn nvcc(mut self, nvcc: impl Into<String>) -> Self {
        self.module_compiler.set_nvcc(nvcc);
        self
    }

    /// Per-invocation wall-clock nvcc timeout. `None` disables the timeout.
    pub fn nvcc_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.module_compiler.set_nvcc_timeout(timeout);
        self
    }

    /// Appends an extra flag to every nvcc invocation.
    pub fn add_flag(mut self, flag: impl Into<String>) -> Self {
        self.module_compiler.add_flag(flag);
        self
    }

    /// Picks the memory scheduler backend. Default depends on features:
    /// with `planner-ortools`, [`SchedulerMode::CpSat`] with `max_secs =
    /// 30.0` (requires the OR-Tools install described in the compiler
    /// crate's `Cargo.toml`); without it, [`SchedulerMode::Heuristic`] —
    /// the OR-Tools-free fallback described in [`planner::plan_heuristic`].
    pub fn scheduler(mut self, scheduler: SchedulerMode) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// Overrides the on-disk kernel cache. Pass an
    /// [`Arc`] so the cache can be shared across compilations. See
    /// [`KernelCache`] for defaults.
    pub fn kernel_cache(mut self, cache: Arc<KernelCache>) -> Self {
        self.kernel_cache = Some(cache);
        self
    }

    /// Disables the kernel cache entirely — every structured kernel is
    /// re-JIT'd from scratch and nothing is written to disk.
    pub fn without_kernel_cache(mut self) -> Self {
        self.kernel_cache = None;
        self
    }

    /// Overrides the kernel-fusion tunables. The pass runs with
    /// [`FusionOptions::default`] unless disabled via [`Self::without_fusion`].
    pub fn fusion_options(mut self, opts: FusionOptions) -> Self {
        self.fusion = Some(opts);
        self
    }

    /// Disables the kernel-fusion pass: the graph is compiled exactly as
    /// built, one launch per inserted kernel.
    pub fn without_fusion(mut self) -> Self {
        self.fusion = None;
        self
    }

    // -----------------------------------------------------------------
    // Graph-level passes. Each pass is self-normalizing:
    // callers can invoke it directly on a freshly built graph without
    // running earlier passes explicitly (each pass calls its own
    // prerequisites, which are cheap no-ops when derived state is up to
    // date).
    // -----------------------------------------------------------------

    /// Fills every kernel node's `types` field, running `type_infer` once
    /// per unique module (post-`kernel_dedup`) and fanning the resulting
    /// `Arc<TypeMap>` out to all aliased nodes.
    ///
    /// Prerequisites: `kernel_dedup` (invoked by `for_each_unique_kernel`).
    pub fn typecheck(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        for_each_unique_kernel(g, |g, indices| -> Result<(), CompileError> {
            let already = kernel_at(g, indices[0]).types.clone();
            if already.is_some() && indices.iter().all(|&i| kernel_at(g, i).types.is_some()) {
                // Every alias already has cached types — nothing to do.
                return Ok(());
            }
            let types = match already {
                Some(t) => t,
                None => Arc::new(type_infer(&kernel_at(g, indices[0]).module)?),
            };
            for &i in indices {
                let n = kernel_at_mut(g, i);
                if n.types.is_none() {
                    n.types = Some(types.clone());
                }
            }
            Ok(())
        })
    }

    /// Rewrites every kernel node into canonical single-kernel form.
    ///
    /// Runs [`Self::typecheck`] first (self-normalizing), then per unique
    /// hash calls `canonicalize` once, producing a canonical [`Program`].
    /// A 1-kernel program has its module Arc swapped into every aliased
    /// node with refreshed types; a multi-kernel program is split via
    /// [`split_program`] and each aliased node is replaced by one child
    /// node per split kernel, with intermediate buffers allocated on the
    /// parent's device and child bindings name-projected from the parent.
    /// A final [`kernel_dedup`] collapses any newly-minted structurally
    /// identical modules.
    ///
    /// Post-condition: every `Kernel` node is single-kernel and
    /// `canonical = true`.
    pub fn canonicalize(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        self.typecheck(g)?;

        // First pass: per unique kernel, canonicalize + split once. Aliased
        // nodes share the resulting subgraph (so their child kernel Arcs
        // are the same objects). The parent's `TypeMap` is not propagated
        // to the child modules — their `NodeId`s are fresh, so the map
        // is not addressable; next `typecheck` re-fills types from each
        // child module.
        let mut results: HashMap<[u8; 32], Arc<crate::passes::ModuleSubgraph>> = HashMap::new();
        for_each_unique_kernel(g, |g, indices| -> Result<(), CompileError> {
            let node = kernel_at(g, indices[0]);
            if node.canonical {
                return Ok(());
            }
            let module = (*node.module).clone();
            let cached_types = node
                .types
                .as_ref()
                .expect("typecheck() ran first, so types is populated")
                .clone();
            let program = crate::passes::canonicalize(module, (*cached_types).clone())?;
            let subgraph = Arc::new(crate::passes::split_program(&program)?);
            let hash = node.hash.expect("kernel_dedup filled hashes");
            results.insert(hash, subgraph);
            Ok(())
        })?;

        // Second pass: apply results. Drain-and-rebuild so splits can
        // expand inline while preserving topological order — subgraph
        // kernels are listed in dependency order by `split_program`.
        let old_nodes = std::mem::take(&mut g.nodes);
        for node in old_nodes {
            match node {
                GraphNode::Kernel(kn) => {
                    let hash = kn.hash.expect("kernel_dedup filled hashes");
                    match results.get(&hash) {
                        None => {
                            // Group already canonical — pass through unchanged.
                            g.nodes.push(GraphNode::Kernel(kn));
                        }
                        Some(res) => {
                            // Even in the single-kernel case, `split_program`
                            // may reorder inputs (declarations follow the
                            // body's DFS first-use order in `extract_kernel`),
                            // so the graph node's input `BufId`s must be
                            // permuted via `SubgraphKernel.inputs` — exactly
                            // what `split_kernel_node` already does.
                            crate::graph_ir::split_kernel_node(g, kn, res)?;
                        }
                    }
                }
                other => g.nodes.push(other),
            }
        }
        g.plan = None;

        // Splits/rewrites minted new modules — refill hashes + collapse.
        crate::graph_ir::kernel_dedup(g);

        // Post-condition: every Kernel node is single-kernel and canonical.
        debug_assert!(g.nodes.iter().all(|n| match n {
            GraphNode::Kernel(k) => k.canonical,
            _ => true,
        }));
        Ok(())
    }

    /// Lowers `reduce` nodes with insufficient outer parallelism into
    /// block-shaped compute chains (see [`rewrite_parallel_reduce`]).
    ///
    /// Runs [`Self::typecheck`] first, then memoizes the rewrite per
    /// `(hash, relevant_bindings)`: `should_tree_lower` needs concrete
    /// `M`/`K`, so same-HIR nodes with different bindings intentionally
    /// diverge. Nodes whose gate says "leave untouched" (bounds still
    /// symbolic, or K/M below threshold) keep their original module
    /// `Arc`. A rewritten module carries a multi-kernel chain and stays
    /// as one graph node until [`Self::canonicalize`] splits it.
    pub fn lower_reduce(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        self.typecheck(g)?;
        // (hash, relevant bindings) → Some(rewritten module Arc) if the
        // gate fired, None if the gate said leave untouched. Cached
        // across nodes with matching keys.
        type MemoKey = ([u8; 32], BTreeMap<String, i64>);
        let mut memo: HashMap<MemoKey, Option<Arc<ir::Module>>> = HashMap::new();

        for idx in crate::graph_ir::kernel_node_indices(g) {
            let node = kernel_at(g, idx);
            let hash = node.hash.expect("typecheck ran kernel_dedup");
            let bindings = relevant_bindings(node);
            let key = (hash, bindings.clone());
            let rewritten = if let Some(cached) = memo.get(&key) {
                cached.clone()
            } else {
                let module = node.module.clone();
                let types = node.types.clone().expect("typecheck populated types");
                // Build a VarId-keyed env from the name-keyed relevant
                // bindings via the module's own param registry.
                let env: BTreeMap<VarId, i64> = module
                    .builder
                    .params()
                    .iter()
                    .filter_map(|(v, name)| bindings.get(name).map(|&val| (*v, val)))
                    .collect();
                let result =
                    crate::passes::rewrite_parallel_reduce(&module, &types, &env)?.map(Arc::new);
                memo.insert(key, result.clone());
                result
            };
            if let Some(m) = rewritten {
                kernel_at_mut(g, idx).replace_module(m);
            }
        }
        Ok(())
    }

    /// Bakes each kernel node's required-concrete parameters into its
    /// module and picks a per-template-group block hint for residuals
    /// that keep a symbolic outer bound.
    ///
    /// Runs [`kernel_dedup`] first (prerequisite per the pass table).
    /// The template group is transient — each node's pre-monomorphize
    /// `hash` snapshots into a local map; the pass picks
    /// `block_size_policy(max_outer[pre_hash])` per group; residuals
    /// with an author-set block hint are left alone
    /// ([`IRBuilder::set_block_hint_if_absent`]). A closing
    /// [`kernel_dedup`] collapses cross-size residuals that structurally
    /// agree onto a shared `Arc`.
    ///
    /// Idempotent: an already-baked residual produces
    /// `required_params.is_empty()`, so `monomorphize_for_graph` returns
    /// a structurally-equal residual and the second call is a no-op path.
    pub fn monomorphize(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        crate::graph_ir::kernel_dedup(g);

        // Template groups keyed by pre-monomorphize hash. Local to this
        // call — the grouping is transient and never stored on nodes.
        let mut groups: HashMap<[u8; 32], Vec<usize>> = HashMap::new();
        for idx in crate::graph_ir::kernel_node_indices(g) {
            let hash = kernel_at(g, idx)
                .hash
                .expect("kernel_dedup filled every hash");
            groups.entry(hash).or_default().push(idx);
        }

        // Per-node monomorphize result, and per-group max outer size
        // (Some iff any residual in the group keeps a symbolic outer
        // bound — that's the driver for block-hint selection).
        let mut gms: HashMap<usize, crate::passes::GraphMono> = HashMap::new();
        let mut max_outer: HashMap<[u8; 32], i64> = HashMap::new();
        for (&pre_hash, node_idxs) in &groups {
            for &idx in node_idxs {
                let node = kernel_at(g, idx);
                let gm = crate::passes::monomorphize_for_graph(&node.module, &node.param_bindings)?;
                if let Some(m) = gm.max_outer {
                    max_outer
                        .entry(pre_hash)
                        .and_modify(|v| *v = (*v).max(m))
                        .or_insert(m);
                }
                gms.insert(idx, gm);
            }
        }

        // Apply per-group block hint + swap in the residual module +
        // update bindings. Author-set hints win. If the residual is
        // structurally equal to the current module (idempotent path
        // when no params were required), skip the swap to preserve the
        // `Arc<ir::Module>` pointer and cached hash.
        for (pre_hash, node_idxs) in &groups {
            let block = max_outer
                .get(pre_hash)
                .copied()
                .map(|m| crate::passes::monomorphize::block_size_policy(m as usize));
            for &idx in node_idxs {
                let mut gm = gms.remove(&idx).expect("populated in the loop above");
                if let Some(b) = block {
                    gm.residual.builder.set_block_hint_if_absent(b);
                }
                let residual_hash = crate::module_hash::module_hash(&gm.residual);
                let node = kernel_at_mut(g, idx);
                if Some(residual_hash) == node.hash {
                    // Already-baked module: keep the Arc and cached
                    // hash; just refresh bindings (their key set may
                    // legitimately be a subset of the old one).
                    node.param_bindings = gm.residual_bindings;
                } else {
                    node.replace_module(gm.residual);
                    node.param_bindings = gm.residual_bindings;
                }
            }
        }

        // Post-pass dedup: residuals from different pre-mono groups
        // (e.g. two sizes of the same template producing the same
        // residual) collapse onto a shared Arc — the old phase-1b
        // content dedup, now the trailing normalization step.
        crate::graph_ir::kernel_dedup(g);
        Ok(())
    }

    /// Runs the pass driver's fusion pipeline:
    /// [`Self::lower_reduce`] → [`Self::monomorphize`] →
    /// [`Self::canonicalize`] → `fuse_graph` (skipped if
    /// [`Self::without_fusion`] disabled it). Every prerequisite pass
    /// self-normalizes, so calling `fuse` on a freshly built graph
    /// produces the same result as the explicit chain (per plan's
    /// any-order protocol).
    ///
    /// Returns the [`FusionReport`] from `fuse_graph`, or `None` when
    /// fusion is disabled.
    pub fn fuse(&self, g: &mut GraphBuilder) -> Result<Option<FusionReport>, CompileError> {
        self.lower_reduce(g)?;
        self.monomorphize(g)?;
        self.canonicalize(g)?;
        let report = self
            .fusion
            .as_ref()
            .map(|opts| fuse_graph(g, opts))
            .transpose()?;
        g.plan = None;
        Ok(report)
    }

    /// Drops kernel nodes whose outputs no live node reads. Wraps
    /// [`fusion::dce`](crate::passes::fusion::dce) and resets `g.plan`
    /// on removal (structural mutation invalidates the memory plan).
    /// Returns the number of nodes removed.
    pub fn dce(&self, g: &mut GraphBuilder) -> usize {
        let removed = crate::passes::fusion::dce(g);
        if removed > 0 {
            g.plan = None;
        }
        removed
    }

    /// Fills `g.plan` when unset. Reuses the cached plan on subsequent
    /// calls (any structural mutation resets it to `None`, so a stale
    /// plan is by construction impossible). No prerequisite passes —
    /// the planner reads the graph shape as it is.
    pub fn plan_memory(&self, g: &mut GraphBuilder) -> Result<(), CompileError> {
        if g.plan.is_some() {
            return Ok(());
        }
        let plan = crate::planner::plan_raw(
            &g.bufs,
            &g.nodes.iter().map(access_from_node).collect::<Vec<_>>(),
            &self.env,
            self.device,
            &g.input_bufs()
                .iter()
                .chain(g.output_bufs().iter())
                .copied()
                .collect::<Vec<_>>(),
            &self.scheduler,
        )
        .map_err(|e| match e {
            PlanError::UnboundSizeSymbol { .. } | PlanError::NegativeSize { .. } => {
                CompileError::Type(format!("graph plan: {e}"))
            }
            #[cfg(feature = "planner-ortools")]
            PlanError::NoSolution(_) => CompileError::Runtime(format!("graph plan: {e}")),
        })?;
        g.plan = Some(plan);
        Ok(())
    }

    /// Consumes the graph, validates its registered interface, plans it and
    /// compiles every structured kernel.
    ///
    /// The registered interface (see [`GraphBuilder::register_input`] /
    /// [`GraphBuilder::register_output`]) is validated against the raw graph
    /// first: inputs must exist on the target device, be distinct, never be
    /// written and be read at least once; outputs must exist on the target
    /// device, be distinct and be written at least once; any unregistered
    /// buffer that is read but never written is an error.
    ///
    /// Unless disabled via [`Self::without_fusion`], the kernel-fusion pass
    /// ([`fuse_graph`]) then rewrites the graph; the resulting
    /// [`FusionReport`] is available on the returned exe via
    /// [`GraphExe::fusion_report`]. The pipeline then runs in three phases:
    ///
    /// 1. **Compile-first**: drain the graph's nodes and JIT every `Kernel(Module)` up front. Each
    ///    graph kernel module is required to be single-kernel by the time compilation runs
    ///    (post-canonicalize + split_module invariant); intermediate buffers only exist as
    ///    graph-level buffers placed by the memory planner.
    /// 2. **Plan + assemble**: run [`planner::plan_raw`] on the `(bufs, node accesses)` pair with
    ///    the registered inputs/outputs pinned to whole-program lifetimes — every buffer (interface
    ///    or intermediate) gets a stable offset in one device pool — and package everything into a
    ///    [`GraphExe`].
    pub fn compile(self, mut graph: GraphBuilder) -> Result<GraphExe, CompileError> {
        validate_interface(&graph, self.device)?;
        let input_bufs = graph.input_bufs().to_vec();
        let output_bufs = graph.output_bufs().to_vec();

        // Pass driver: run the normalize + (optional) fuse chain before
        // compilation. Post-passes every Kernel node's `module` is the
        // residual, `param_bindings` are the residual bindings, and
        // `node.hash` is set to the residual hash — the inline
        // monomorphize in Phase 1a below is idempotent on this
        // already-processed input.
        let fusion_report = match &self.fusion {
            Some(_) => self.fuse(&mut graph)?,
            None => {
                self.lower_reduce(&mut graph)?;
                self.monomorphize(&mut graph)?;
                self.canonicalize(&mut graph)?;
                None
            }
        };
        self.dce(&mut graph);

        // Phase 1: drain and compile every structured kernel.
        //
        // `compiled` mirrors the original node order. For structured kernels
        // the compiled `KernelProgram` is wrapped in `Arc<Mutex<_>>` and
        // cached by (source `Arc<ir::Module>` pointer, parameter bindings),
        // so two `Kernel` nodes that carry the same module clone and the
        // same bindings share one JIT build.
        // `KernelProgram::set_symbol`/`set_input`/`set_output` take
        // `&mut self`, hence the `Mutex` for the shared handle.
        enum PreExe {
            Kernel {
                name: String,
                /// Compact index into `kernels: Vec<KernelProgram>`. Multiple
                /// `PreExe::Kernel`s may share a `kernel_idx` when their
                /// source modules dedup to the same residual hash.
                kernel_idx: usize,
                inputs: Vec<BufId>,
                outputs: Vec<BufId>,
                /// Positional values for the kernel's runtime parameters.
                /// Aligned with the kernel's declared param order; the
                /// launch loop pairs each with `KernelProgram::params()` to
                /// call `set_symbol`.
                set_params: Vec<i64>,
            },
            Blackbox(KernelNode),
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
        let nodes: Vec<GraphNode> = graph.nodes.drain(..).collect();

        // Phase 1a: partition Kernel nodes from the rest and identify the
        // unique (module, bindings) pairs to compile. Two `KernelModuleNode`s
        // sharing an `Arc` pointer and identical parameter bindings alias the
        // same compiled artifact. A parameterized module is monomorphized
        // against its bindings here: the *residual* (with its surviving
        // params) is what gets hashed, cached and compiled, so nodes whose
        // residuals agree share one JIT build even across different concrete
        // sizes. The α-normalized hash of the PRE-monomorphization module is
        // kept as the template grouping key for block selection below.
        enum ResidualSlot {
            Shared(Arc<ir::Module>),
            Owned(ir::Module),
        }
        struct PreUnique {
            residual: ResidualSlot,
            /// Binding values for the residual's surviving params (empty for
            /// concrete modules).
            vals: Vec<i64>,
            /// Max concrete outer compute size — `Some` iff the residual
            /// keeps a symbolic outer bound and needs a block hint.
            max_outer: Option<i64>,
            template_hash: [u8; 32],
        }
        let mut unique: Vec<PreUnique> = Vec::new();
        let mut key_index: HashMap<(*const ir::Module, BTreeMap<String, i64>), usize> =
            HashMap::new();
        // Per-node kernel slot: `Some(kernel_idx)` if this graph node is a
        // Kernel, referring to `unique[kernel_idx]`.
        let mut node_module_idx: Vec<Option<usize>> = Vec::with_capacity(nodes.len());
        for node in &nodes {
            match node {
                GraphNode::Kernel(k) => {
                    let key = (Arc::as_ptr(&k.module), k.param_bindings.clone());
                    let idx = match key_index.get(&key) {
                        Some(&idx) => idx,
                        None => {
                            if self.module_compiler.check_accesses {
                                check_module_accesses(&k.module, &k.param_bindings)?;
                            }
                            let template_hash = module_hash(&renumber_module(&k.module));
                            let pu = if k.param_bindings.is_empty() {
                                PreUnique {
                                    residual: ResidualSlot::Shared(k.module.clone()),
                                    vals: Vec::new(),
                                    max_outer: None,
                                    template_hash,
                                }
                            } else {
                                let gm = monomorphize_for_graph(&k.module, &k.param_bindings)?;
                                // Convert name-keyed bindings back to
                                // positional Vec<i64> aligned with the
                                // residual module's param order. The
                                // runtime `set_symbol(name, v)` ABI is
                                // name-based, but the launch loop pairs
                                // each value with its corresponding name
                                // from `KernelProgram::params()`.
                                let vals = pos_vec(&gm.residual, &gm.residual_bindings);
                                PreUnique {
                                    residual: ResidualSlot::Owned(gm.residual),
                                    vals,
                                    max_outer: gm.max_outer,
                                    template_hash,
                                }
                            };
                            unique.push(pu);
                            let idx = unique.len() - 1;
                            key_index.insert(key, idx);
                            idx
                        }
                    };
                    node_module_idx.push(Some(idx));
                }
                _ => node_module_idx.push(None),
            }
        }
        drop(key_index);

        // Phase 1a': per-template block selection. Variants of one
        // template must not fragment into different launch geometries,
        // so a residual that keeps a symbolic outer bound gets its
        // block hint from the max concrete compute size over ALL of
        // the template's instantiations, not just its own node's. An
        // author-set block hint wins.
        let mut group_max: HashMap<[u8; 32], i64> = HashMap::new();
        for pu in &unique {
            if let Some(m) = pu.max_outer {
                group_max
                    .entry(pu.template_hash)
                    .and_modify(|g| *g = (*g).max(m))
                    .or_insert(m);
            }
        }
        for pu in &mut unique {
            if pu.max_outer.is_none() {
                continue;
            }
            let ResidualSlot::Owned(m) = &mut pu.residual else {
                continue;
            };
            if m.builder.block_hint().is_none() {
                let group_m = group_max[&pu.template_hash];
                m.builder
                    .set_block_hint(block_size_policy(group_m as usize));
            }
        }
        drop(group_max);
        let unique_param_vals: Vec<Vec<i64>> = unique.iter().map(|pu| pu.vals.clone()).collect();
        let unique_modules: Vec<Arc<ir::Module>> = unique
            .into_iter()
            .map(|pu| match pu.residual {
                ResidualSlot::Shared(a) => a,
                ResidualSlot::Owned(m) => Arc::new(m),
            })
            .collect();

        // Phase 1b: content-based dedup. `Arc::as_ptr` identity picks up
        // callers who deliberately share a single module handle across
        // multiple `insert_kernel` calls, but a helper that rebuilds the
        // same `Module` at each call site produces distinct `Arc`s with
        // identical bytes. Group by `module_hash` so those still share one
        // JIT build. `hash_repr[i]` is the *representative* unique-module
        // index for `unique_modules[i]` (`i` itself if this module was the
        // first with its hash).
        let hashes: Vec<[u8; 32]> = unique_modules.iter().map(|m| module_hash(m)).collect();
        let mut hash_repr: Vec<usize> = Vec::with_capacity(unique_modules.len());
        let mut representative_of_hash: HashMap<[u8; 32], usize> = HashMap::new();
        let mut representatives: Vec<usize> = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            match representative_of_hash.get(h) {
                Some(&repr) => hash_repr.push(repr),
                None => {
                    representative_of_hash.insert(*h, i);
                    representatives.push(i);
                    hash_repr.push(i);
                }
            }
        }
        drop(representative_of_hash);
        let num_unique_content = representatives.len();

        // Each unique_module_idx maps to the *compact* index of its
        // representative in `kernels: Vec<KernelProgram>`. Aliased entries
        // share the same compact index.
        let compact_of_repr: HashMap<usize, usize> = representatives
            .iter()
            .enumerate()
            .map(|(compact, &repr)| (repr, compact))
            .collect();
        let unique_to_kernel_idx: Vec<usize> = (0..unique_modules.len())
            .map(|i| compact_of_repr[&hash_repr[i]])
            .collect();

        // Phase 1c: probe the on-disk cache, but only for representatives.
        // Cache hits are loaded on the main thread (cheap; just a dlopen).
        // `compiled_by_compact[compact_idx] = Some(km)` after Phase 1e.
        let mut compiled_by_compact: Vec<Option<KernelProgram>> =
            (0..num_unique_content).map(|_| None).collect();
        let mut cache_misses: Vec<usize> = Vec::new();
        let mut num_cached_modules = 0usize;
        for (compact, &repr) in representatives.iter().enumerate() {
            let module = &unique_modules[repr];
            let hit = match &self.kernel_cache {
                Some(cache) => cache.get(module)?,
                None => None,
            };
            match hit {
                Some(km) => {
                    compiled_by_compact[compact] = Some(km);
                    num_cached_modules += 1;
                }
                None => cache_misses.push(compact),
            }
        }

        // Phase 1d: JIT the misses in parallel — one compile per unique
        // content hash, not per unique Arc handle. Prints a progress line
        // every 5% completion so long compiles at large problem sizes are
        // observable.
        let module_compiler = &self.module_compiler;
        let cache = self.kernel_cache.clone();
        let n_to_compile = cache_misses.len();
        let compile_results: Vec<Result<(usize, KernelProgram), CompileError>> = {
            use std::{
                sync::atomic::{AtomicUsize, Ordering},
                time::Instant,
            };

            use rayon::prelude::*;
            let done = AtomicUsize::new(0);
            let next_tick = AtomicUsize::new(0);
            let start = Instant::now();
            if n_to_compile > 0 {
                let total = n_to_compile + num_cached_modules;
                eprintln!(
                    "[compile] {n_to_compile}/{total} unique kernel modules need fresh nvcc \
                     invocations ({num_cached_modules} served from on-disk cache)",
                );
            }
            let out: Vec<Result<(usize, KernelProgram), CompileError>> = cache_misses
                .par_iter()
                .map(|&compact| {
                    let repr = representatives[compact];
                    let module = &unique_modules[repr];
                    // The graph passes above leave `module` as a
                    // canonicalized, monomorphized, single-kernel
                    // residual — exactly what the strict per-kernel
                    // backend expects.
                    let km = module_compiler.compile((**module).clone())?;
                    if let Some(c) = &cache {
                        // Best-effort — a failed insert shouldn't fail the compile.
                        let _ = c.insert(module, &km);
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    // Advance the "next 5% tick" pointer past every ceiling we crossed;
                    // whoever performs the CAS that lands us at or past a tick prints it.
                    let pct = d * 100 / n_to_compile.max(1);
                    let tick = (pct / 5) * 5;
                    let mut cur = next_tick.load(Ordering::Relaxed);
                    while tick > cur
                        && next_tick
                            .compare_exchange_weak(cur, tick, Ordering::Relaxed, Ordering::Relaxed)
                            .is_err()
                    {
                        cur = next_tick.load(Ordering::Relaxed);
                    }
                    if tick > cur {
                        eprintln!(
                            "[compile] {tick:>3}% ({d}/{n_to_compile}, {:.1}s elapsed)",
                            start.elapsed().as_secs_f64(),
                        );
                    }
                    Ok((compact, km))
                })
                .collect();
            if n_to_compile > 0 {
                let succeeded = out.iter().filter(|r| r.is_ok()).count();
                let failed = n_to_compile - succeeded;
                if failed == 0 {
                    eprintln!(
                        "[compile] done ({succeeded}/{n_to_compile}, {:.1}s total)",
                        start.elapsed().as_secs_f64(),
                    );
                } else {
                    eprintln!(
                        "[compile] finished with errors ({succeeded}/{n_to_compile} succeeded, \
                         {failed} failed, {:.1}s total)",
                        start.elapsed().as_secs_f64(),
                    );
                    let mut timed_out: Vec<(&str, f64, f64)> = out
                        .iter()
                        .filter_map(|r| match r {
                            Err(CompileError::NvccTimeout {
                                name,
                                seconds,
                                limit,
                            }) => Some((name.as_str(), *seconds, *limit)),
                            _ => None,
                        })
                        .collect();
                    if !timed_out.is_empty() {
                        timed_out.sort_by(|a, b| b.1.total_cmp(&a.1));
                        eprintln!(
                            "[compile] {} nvcc invocation(s) timed out (limit {:.0}s):",
                            timed_out.len(),
                            timed_out[0].2,
                        );
                        for (name, secs, _) in &timed_out {
                            eprintln!("[compile]   [{secs:>7.1}s]  {name}");
                        }
                    }
                    let other_failures: Vec<&CompileError> = out
                        .iter()
                        .filter_map(|r| match r {
                            Err(e) if !matches!(e, CompileError::NvccTimeout { .. }) => Some(e),
                            _ => None,
                        })
                        .collect();
                    if !other_failures.is_empty() {
                        eprintln!("[compile] {} non-timeout failure(s):", other_failures.len(),);
                        for e in other_failures.iter().take(5) {
                            eprintln!("[compile]   {e}");
                        }
                        if other_failures.len() > 5 {
                            eprintln!("[compile]   ...and {} more", other_failures.len() - 5,);
                        }
                    }
                }
            }
            out
        };
        for res in compile_results {
            let (compact, km) = res?;
            compiled_by_compact[compact] = Some(km);
        }

        // Move the owned `KernelProgram`s into a compact `Vec` indexed by
        // the representative's compact index — this is `GraphExe.kernels`.
        // Aliased unique_module_idxs already share the same compact idx via
        // `unique_to_kernel_idx`, so no cloning needed. `mut` because the
        // ExeNode assembly below binds `set_params` on each kernel to
        // validate sizes.
        let mut kernels: Vec<KernelProgram> = compiled_by_compact
            .into_iter()
            .map(|slot| slot.expect("every compact slot filled"))
            .collect();

        // Phase 1d: assemble `PreExe`s in original node order. Each Kernel
        // node's `kernel_idx` points into `kernels`.
        let mut compiled: Vec<PreExe> = Vec::with_capacity(nodes.len());
        for (node, mod_idx) in nodes.into_iter().zip(node_module_idx.into_iter()) {
            compiled.push(match node {
                GraphNode::Kernel(k) => {
                    let idx = mod_idx.expect("Kernel node without module index");
                    let kernel_idx = unique_to_kernel_idx[idx];
                    let set_params = unique_param_vals[idx].clone();
                    let name = k.module.name.clone();
                    PreExe::Kernel {
                        name,
                        kernel_idx,
                        inputs: k.inputs,
                        outputs: k.outputs,
                        set_params,
                    }
                }
                GraphNode::BlackboxKernel(k) => PreExe::Blackbox(k),
                GraphNode::Const(c) => PreExe::Const(c),
                GraphNode::Memcpy(m) => PreExe::Memcpy {
                    src: m.src,
                    src_offset: eval_nonneg(&m.src_offset, &self.env, "memcpy src_offset")?,
                    dst: m.dst,
                    dst_offset: eval_nonneg(&m.dst_offset, &self.env, "memcpy dst_offset")?,
                    num_bytes: eval_nonneg(&m.num_bytes, &self.env, "memcpy num_bytes")?,
                },
                GraphNode::Memset(m) => PreExe::Memset {
                    buf: m.node,
                    offset: eval_nonneg(&m.offset, &self.env, "memset offset")?,
                    num_bytes: eval_nonneg(&m.num_bytes, &self.env, "memset num_bytes")?,
                    val: m.val,
                },
            });
        }
        // Report content-unique modules rather than Arc-unique ones. This
        // is the number of nvcc invocations a cold cache would trigger; the
        // Arc-level count would double-count structurally identical modules
        // built at different call sites.
        let num_unique_modules = num_unique_content;
        let bufs = graph.bufs.clone();

        // Phase 2: build per-node access sets.
        let accesses: Vec<NodeAccess> = compiled
            .iter()
            .map(|p| match p {
                PreExe::Kernel {
                    inputs, outputs, ..
                } => {
                    let mut a = NodeAccess::default();
                    a.reads.extend(inputs.iter().copied());
                    a.writes.extend(outputs.iter().copied());
                    a
                }
                PreExe::Blackbox(k) => access_from_node(&GraphNode::BlackboxKernel(clone_kn(k))),
                PreExe::Const(c) => access_from_node(&GraphNode::Const(clone_cn(c))),
                PreExe::Memcpy { src, dst, .. } => NodeAccess {
                    reads: vec![*src],
                    writes: vec![*dst],
                },
                PreExe::Memset { buf, .. } => NodeAccess {
                    reads: Vec::new(),
                    writes: vec![*buf],
                },
            })
            .collect();

        // Registered interface buffers are pinned in the plan: their pool
        // slots must stay stable and intact for the whole run (and across
        // replays that skip re-binding inputs), so no intermediate may ever
        // reuse their memory.
        let pin: Vec<BufId> = input_bufs
            .iter()
            .chain(output_bufs.iter())
            .copied()
            .collect();

        // Phase 3: plan and validate.
        let plan = planner::plan_raw(
            &bufs,
            &accesses,
            &self.env,
            self.device,
            &pin,
            &self.scheduler,
        )
        .map_err(|e| match e {
            PlanError::UnboundSizeSymbol { .. } | PlanError::NegativeSize { .. } => {
                CompileError::Type(format!("graph plan: {e}"))
            }
            #[cfg(feature = "planner-ortools")]
            PlanError::NoSolution(_) => CompileError::Runtime(format!("graph plan: {e}")),
        })?;
        let sizes = evaluate_sizes_bufs(&bufs, &self.env)?;

        // Assemble ExeNodes.
        let mut nodes = Vec::with_capacity(compiled.len());
        for p in compiled {
            nodes.push(match p {
                PreExe::Kernel {
                    name,
                    kernel_idx,
                    inputs,
                    outputs,
                    set_params,
                } => {
                    {
                        // Sizes on a shared handle are functions of the
                        // currently-bound params; bind this node's before
                        // querying.
                        let m = &mut kernels[kernel_idx];
                        let names: Vec<String> = m.params().to_vec();
                        for (name, &v) in names.iter().zip(set_params.iter()) {
                            m.set_symbol(name, v);
                        }
                        check_kernel_sizes(m, &inputs, &outputs, &sizes)?;
                    }
                    ExeNode::Kernel(ExeKernel {
                        name,
                        kernel_idx,
                        inputs,
                        outputs,
                        set_params,
                    })
                }
                PreExe::Blackbox(k) => ExeNode::Blackbox(k),
                PreExe::Const(c) => ExeNode::Const(c),
                PreExe::Memcpy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    num_bytes,
                } => {
                    let src_end = src_offset.checked_add(num_bytes).ok_or_else(|| {
                        CompileError::Runtime(format!(
                            "memcpy: src_offset+num_bytes overflows usize \
                             ({src_offset} + {num_bytes})"
                        ))
                    })?;
                    if src_end > sizes[src.0] {
                        return Err(CompileError::Runtime(format!(
                            "memcpy: src range [{src_offset}..{src_end}) exceeds \
                             buffer {src:?} size {}",
                            sizes[src.0]
                        )));
                    }
                    let dst_end = dst_offset.checked_add(num_bytes).ok_or_else(|| {
                        CompileError::Runtime(format!(
                            "memcpy: dst_offset+num_bytes overflows usize \
                             ({dst_offset} + {num_bytes})"
                        ))
                    })?;
                    if dst_end > sizes[dst.0] {
                        return Err(CompileError::Runtime(format!(
                            "memcpy: dst range [{dst_offset}..{dst_end}) exceeds \
                             buffer {dst:?} size {}",
                            sizes[dst.0]
                        )));
                    }
                    ExeNode::Memcpy {
                        src,
                        src_offset,
                        dst,
                        dst_offset,
                        num_bytes,
                    }
                }
                PreExe::Memset {
                    buf,
                    offset,
                    num_bytes,
                    val,
                } => {
                    let end = offset.checked_add(num_bytes).ok_or_else(|| {
                        CompileError::Runtime(format!(
                            "memset: offset+num_bytes overflows usize ({offset} + {num_bytes})"
                        ))
                    })?;
                    if end > sizes[buf.0] {
                        return Err(CompileError::Runtime(format!(
                            "memset: range [{offset}..{end}) exceeds buffer {buf:?} size {}",
                            sizes[buf.0]
                        )));
                    }
                    ExeNode::Memset {
                        buf,
                        offset,
                        num_bytes,
                        val,
                    }
                }
            });
        }

        Ok(GraphExe {
            plan,
            sizes,
            kernels,
            nodes,
            inputs_bound: vec![false; input_bufs.len()],
            input_bufs,
            output_bufs,
            pool: None,
            device: self.device,
            bufs,
            num_unique_modules,
            num_cached_modules,
            fusion_report,
            captured: None,
        })
    }
}

/// Validates the registered graph interface against the raw (pre-fusion)
/// graph. See [`GraphCompiler::compile`] for the rules enforced here.
fn validate_interface(graph: &GraphBuilder, device: DeviceType) -> Result<(), CompileError> {
    let n_bufs = graph.bufs.len();
    let (writers, readers) = classify_buf_uses(&graph.nodes, n_bufs);

    let name_of = |b: BufId| -> String {
        match graph.bufs.get(b.0).and_then(|i| i.name.as_deref()) {
            Some(n) => format!("{b:?} (`{n}`)"),
            None => format!("{b:?}"),
        }
    };

    let mut seen = vec![false; n_bufs];
    for &b in graph.input_bufs() {
        if b.0 >= n_bufs {
            return Err(CompileError::Type(format!(
                "graph interface: registered input {b:?} does not exist"
            )));
        }
        if graph.bufs[b.0].device_type != device {
            return Err(CompileError::Type(format!(
                "graph interface: input {} is on {:?}, but the graph compiles for {device:?}",
                name_of(b),
                graph.bufs[b.0].device_type
            )));
        }
        if seen[b.0] {
            return Err(CompileError::Type(format!(
                "graph interface: input {} registered twice",
                name_of(b)
            )));
        }
        seen[b.0] = true;
        if !writers[b.0].is_empty() {
            return Err(CompileError::Type(format!(
                "graph interface: input {} is written by node {}; inputs must only be read",
                name_of(b),
                writers[b.0][0]
            )));
        }
        if readers[b.0].is_empty() {
            return Err(CompileError::Type(format!(
                "graph interface: input {} is never read by any node",
                name_of(b)
            )));
        }
    }
    for &b in graph.output_bufs() {
        if b.0 >= n_bufs {
            return Err(CompileError::Type(format!(
                "graph interface: registered output {b:?} does not exist"
            )));
        }
        if graph.bufs[b.0].device_type != device {
            return Err(CompileError::Type(format!(
                "graph interface: output {} is on {:?}, but the graph compiles for {device:?}",
                name_of(b),
                graph.bufs[b.0].device_type
            )));
        }
        if seen[b.0] {
            return Err(CompileError::Type(format!(
                "graph interface: output {} registered twice (or also registered as an input)",
                name_of(b)
            )));
        }
        seen[b.0] = true;
        if writers[b.0].is_empty() {
            return Err(CompileError::Type(format!(
                "graph interface: output {} is never written by any node",
                name_of(b)
            )));
        }
    }
    // Unregistered buffers that are read but never written used to be
    // silently auto-classified as graph inputs; with explicit registration
    // they are uninitialized memory and therefore a graph bug.
    for b in 0..n_bufs {
        if !seen[b] && writers[b].is_empty() && !readers[b].is_empty() {
            return Err(CompileError::Type(format!(
                "graph buffer {} is read but never written and is not a registered input",
                name_of(BufId(b))
            )));
        }
    }
    // Topological-order hazard: the memory planner (see `planner.rs`
    // `PlanCtx::edges`) derives RAW/WAR/WAW precedence edges by versioning
    // each buffer along node insertion order and requires the graph to be
    // built write-before-read. Registered *inputs* are excepted by design
    // (they have no writer at all — the runtime binds them via
    // `set_input`), but for every other buffer with both writers and
    // readers the earliest writer must precede the earliest reader in
    // insertion order. Otherwise the derived edges never pin the reader
    // after its writer and the planner may schedule the reader against
    // uninitialized bytes.
    let mut is_input = vec![false; n_bufs];
    for &b in graph.input_bufs() {
        if b.0 < n_bufs {
            is_input[b.0] = true;
        }
    }
    for b in 0..n_bufs {
        if is_input[b] {
            continue;
        }
        if let (Some(&first_w), Some(&first_r)) = (writers[b].first(), readers[b].first()) {
            if first_w > first_r {
                return Err(CompileError::Type(format!(
                    "graph buffer {} is first read at node {first_r} but its earliest writer \
                     is at node {first_w} (>= first-read); insertion order is not a valid \
                     write-before-read topological order — the memory planner would derive \
                     ill-formed RAW/WAR/WAW edges and schedule the reader against \
                     uninitialized memory",
                    name_of(BufId(b))
                )));
            }
        }
    }
    Ok(())
}

fn clone_kn(k: &KernelNode) -> KernelNode {
    // KernelNode's `func` is not Clone, but `access_from_node` only inspects
    // the buffer fields. Build a shallow copy with a placeholder closure.
    KernelNode {
        inputs: k.inputs.clone(),
        outputs: k.outputs.clone(),
        carried_outputs: k.carried_outputs.clone(),
        func: Box::new(|_, _, _| {}),
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

struct ExeKernel {
    name: String,
    /// Index into [`GraphExe::kernels`]: the compiled artifact this node
    /// launches. Multiple `ExeKernel`s can share a `kernel_idx` (when their
    /// source modules dedup to the same residual hash); execution is
    /// sequential so each node re-binds `set_params` / inputs / outputs on
    /// the shared program before its launch.
    kernel_idx: usize,
    inputs: Vec<BufId>,
    outputs: Vec<BufId>,
    /// Positional values for the kernel's runtime parameters, aligned with
    /// [`KernelProgram::params()`]'s name order. The launch loop pairs each
    /// value with its corresponding name and calls `set_symbol` before
    /// launching.
    set_params: Vec<i64>,
}

enum ExeNode {
    Kernel(ExeKernel),
    Blackbox(KernelNode),
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
    plan: MemoryPlan,
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

    /// What the kernel-fusion pass did during [`GraphCompiler::compile`];
    /// `None` when the pass was disabled via
    /// [`GraphCompiler::without_fusion`].
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
        out.push_str(&format!("// Execution order: {:?}\n", self.plan.order));

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
            ExeNode::Blackbox(k) => {
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

    /// Executes the graph on `ctx.stream`.
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
            ..
        } = self;
        let pool = pool.as_ref().expect("pool ensured above");
        let device = *device;
        let bufid_ptr = |b: BufId| resolve_ptr(pool, &plan.offsets, device, b);

        for &node_idx in &plan.order {
            match &mut nodes[node_idx] {
                ExeNode::Kernel(k) => {
                    // Re-binding params / inputs / outputs just before the
                    // launch is safe even when this artifact is shared
                    // with other nodes (multiple `ExeKernel`s can point
                    // at the same `kernel_idx`), because execution is
                    // sequential. Params first: the sizes queried below
                    // are functions of the currently-bound params, which
                    // the previous node may have overwritten.
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
                    (k.func)(&ins, &outs, ctx.stream.as_raw());
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
                ExeNode::Memcpy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    num_bytes,
                } => {
                    let src_ptr = bufid_ptr(*src)?;
                    let dst_ptr = bufid_ptr(*dst)?;
                    unsafe {
                        cuda_memcpy_on::<true, true>(
                            dst_ptr.add(*dst_offset) as *mut c_void,
                            src_ptr.add(*src_offset) as *const c_void,
                            *num_bytes,
                            ctx,
                        )
                        .map_err(memcpy_err)?;
                    }
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

/// Subset of `node.param_bindings` that the module's Compute / Reduce
/// bounds actually read. Used by the [`GraphCompiler::lower_reduce`]
/// memoization key so same-HIR nodes with different K bindings diverge
/// while nodes with identical relevant bindings share the rewrite result.
fn relevant_bindings(node: &crate::graph_ir::KernelModuleNode) -> BTreeMap<String, i64> {
    use std::collections::BTreeSet;
    let b = &node.module.builder;
    let mut used: BTreeSet<VarId> = BTreeSet::new();
    let mut seen: BTreeSet<ir::NodeId> = BTreeSet::new();
    let mut stack = vec![node.module.body];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match b.node(id) {
            ir::Node::Compute { bound, .. } | ir::Node::Reduce { bound, .. } => {
                bound.param_syms(&mut used);
            }
            _ => {}
        }
        stack.extend(crate::module_hash::children_of(b.node(id)));
    }
    // Map back from VarId → name via the module's param registry.
    b.params()
        .iter()
        .filter_map(|(v, name)| {
            if used.contains(v) {
                node.param_bindings
                    .get(name)
                    .map(|&val| (name.clone(), val))
            } else {
                None
            }
        })
        .collect()
}

/// Positional binding values in a module's declared param order, keyed
/// by name. The runtime `set_symbol(name, v)` ABI is name-based, but
/// the launch loop pairs each positional value with its corresponding
/// name from the module's `params()` list. Values for names that don't
/// appear in the module (e.g. a `baked` entry keyed under the
/// pre-monomorphization module's names when queried against the residual)
/// are silently skipped; callers ensure the intended alignment.
fn pos_vec(m: &ir::Module, bindings: &BTreeMap<String, i64>) -> Vec<i64> {
    m.builder
        .params()
        .iter()
        .filter_map(|(_, name)| bindings.get(name).copied())
        .collect()
}

/// Evaluates a [`Quast`] to a non-negative `usize`, reporting `what` as
/// context on failure. Used for memcpy/memset offsets and lengths.
fn eval_nonneg(q: &Quast, env: &BTreeMap<VarId, i64>, what: &str) -> Result<usize, CompileError> {
    let mut syms = std::collections::BTreeSet::new();
    q.syms(&mut syms);
    for s in &syms {
        if !env.contains_key(s) {
            return Err(CompileError::Type(format!(
                "{what} references unbound symbol {s:?}"
            )));
        }
    }
    let v = q.eval(env);
    if v < 0 {
        return Err(CompileError::Type(format!(
            "{what} evaluates to a negative value {v}"
        )));
    }
    Ok(v as usize)
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
    module: &KernelProgram,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_cuda_common::{copy::MemCopyH2D, stream::GpuDeviceCtx};

    use super::*;
    use crate::{
        graph_ir::{BufInfo, GraphBuilder, KernelModuleNode},
        ir::{IRBuilder, ScalarType},
    };

    /// Two `Kernel` graph nodes that share the *same* `Arc<ir::Module>` are
    /// JIT'd exactly once, and each node still receives its own input/output
    /// bindings. The scaled outputs must equal 2 * input for both bindings.
    #[test]
    fn shared_module_compiled_once_and_runs_on_both_inputs() {
        const N: usize = 16;
        // Build one `scale_by_two` module and wrap it in an Rc; both graph
        // kernel nodes will share this clone.
        let module = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![N]);
            let body = b.compute(N, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                b.mul(ai, two)
            });
            Arc::new(b.finish("scale_by_two_shared", body))
        };

        let bytes = (N * 4) as i64;
        let mut g = GraphBuilder::new();
        let mk = |g: &mut GraphBuilder, name: &str| -> BufId {
            g.add_buf(BufInfo {
                name: Some(name.to_string()),
                device_type: DeviceType::Cuda(0),
                size: Quast::cst(bytes),
                elem_size: 4,
            })
        };
        let in0 = mk(&mut g, "in0");
        let in1 = mk(&mut g, "in1");
        let out0 = mk(&mut g, "out0");
        let out1 = mk(&mut g, "out1");
        g.register_input(in0);
        g.register_input(in1);
        g.register_output(out0);
        g.register_output(out1);
        g.insert_kernel(module.clone(), [in0], [out0], &[]);
        g.insert_kernel(module.clone(), [in1], [out1], &[]);

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile(g)
            .expect("graph compile");

        // The two kernel nodes shared a module, so only one JIT build happened.
        assert_eq!(exe.num_unique_modules(), 1);
        assert_eq!(exe.num_inputs(), 2);
        assert_eq!(exe.num_outputs(), 2);

        // Match caller-supplied buffer order to the graph exe's ordering.
        let ins_order: Vec<BufId> = (0..exe.num_inputs()).map(|i| exe.input_buf_id(i)).collect();
        let outs_order: Vec<BufId> = (0..exe.num_outputs())
            .map(|i| exe.output_buf_id(i))
            .collect();

        // Distinct inputs so we can check both outputs independently.
        let host0: Vec<u32> = (0..N as u32).map(|i| i + 1).collect();
        let host1: Vec<u32> = (0..N as u32).map(|i| 100 + i).collect();
        let host_for = |b: BufId| -> &Vec<u32> {
            if b == in0 {
                &host0
            } else if b == in1 {
                &host1
            } else {
                panic!("unexpected input BufId {b:?}")
            }
        };
        let want_for = |b: BufId| -> Vec<u32> {
            let src = if b == out0 {
                &host0
            } else if b == out1 {
                &host1
            } else {
                panic!("unexpected output BufId {b:?}")
            };
            src.iter()
                .map(|&x| {
                    // Multiplication happens in BabyBear; inputs stay well
                    // below p/2 so `x * 2` never wraps.
                    x * 2
                })
                .collect()
        };

        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        for (i, &b) in ins_order.iter().enumerate() {
            let bytes: Vec<u8> = host_for(b).iter().flat_map(|x| x.to_le_bytes()).collect();
            let dbuf = bytes.as_slice().to_device_on(&ctx).expect("H2D");
            exe.set_input(&ctx, i, &dbuf).expect("set_input");
        }
        exe.run(&ctx).expect("graph run");

        for (i, &out_bid) in outs_order.iter().enumerate() {
            let bytes: Vec<u8> = exe.get_output(i).to_host_on(&ctx).expect("D2H");
            let got: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let want = want_for(out_bid);
            assert_eq!(got, want, "output {i} mismatch (BufId {out_bid:?})");
        }
    }

    /// A multi-kernel module (`s = reduce_add(a); out = a[i] * s`) inserted
    /// through `insert_kernel` is split into one graph node per kernel, JIT'd
    /// as two unique modules, and runs end-to-end.
    ///
    /// TODO: a scalar reduce output flowing through `split_program`
    /// produces a child kernel that reads the scalar as a `[1]` tensor
    /// input, and re-type-inference of the consumer fails with
    /// "binary operand must be a scalar" for the `mul(ai, s)`. Fix
    /// belongs in `split_program`'s scalar-output handling.
    #[ignore = "scalar reduce → child kernel type-check fails through canonicalize-then-split"]
    #[test]
    fn multi_kernel_module_splits_and_runs() {
        const N: usize = 16;
        let module = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![N]);
            let s = b.reduce_add(N, |b, i| b.index(a, &[i]));
            let s = b.let_bound(s);
            let out = b.compute(N, |b, i| {
                let ai = b.index(a, &[i]);
                b.mul(ai, s)
            });
            b.finish("scale_by_sum", out)
        };

        let bytes = (N * 4) as i64;
        let mut g = GraphBuilder::new();
        let mk = |g: &mut GraphBuilder, name: &str| -> BufId {
            g.add_buf(BufInfo {
                name: Some(name.to_string()),
                device_type: DeviceType::Cuda(0),
                size: Quast::cst(bytes),
                elem_size: 4,
            })
        };
        let a_buf = mk(&mut g, "a");
        let out_buf = mk(&mut g, "out");
        g.register_input(a_buf);
        g.register_output(out_buf);
        g.insert_kernel(module, [a_buf], [out_buf], &[]);

        // `insert_kernel` pushes ONE node with the multi-kernel module.
        // `GraphCompiler::compile` canonicalizes and splits internally.
        assert_eq!(g.nodes.len(), 1);

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile(g)
            .expect("graph compile");

        // Two distinct split kernels → two unique JIT builds. The scalar
        // intermediate is unregistered, so it stays internal to the pool
        // and never appears in the registered interface.
        assert_eq!(exe.num_unique_modules(), 2);
        assert_eq!(exe.num_inputs(), 1);
        assert_eq!(exe.num_outputs(), 1);
        assert_eq!(exe.input_buf_id(0), a_buf);
        assert_eq!(exe.output_buf_id(0), out_buf);

        // BabyBear buffers hold Montgomery-form values (see passes::codegen):
        // encode inputs and compare Montgomery-encoded expectations. Sum and
        // products stay far below the modulus, so plain u32 arithmetic is
        // the canonical reference.
        use crate::passes::codegen::to_monty;
        let host: Vec<u32> = (1..=N as u32).collect();
        let sum: u32 = host.iter().sum();
        let want: Vec<u32> = host.iter().map(|&x| to_monty(x * sum)).collect();

        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let in_bytes: Vec<u8> = host
            .iter()
            .flat_map(|&x| to_monty(x).to_le_bytes())
            .collect();
        let in_buf = in_bytes.as_slice().to_device_on(&ctx).expect("H2D");
        exe.set_input(&ctx, 0, &in_buf).expect("set_input");
        exe.run(&ctx).expect("graph run");

        let bytes: Vec<u8> = exe.get_output(0).to_host_on(&ctx).expect("D2H");
        let got: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, want);
    }

    /// A CPU-only sanity check on the topological-order hazard guard in
    /// [`validate_interface`]: inserting a kernel that reads a buffer
    /// *before* its writer (a memcpy) must be rejected with a diagnostic
    /// pointing at the offending buffer, since the memory planner's
    /// insertion-order versioning would otherwise silently produce
    /// ill-formed RAW/WAR edges.
    #[test]
    fn topo_order_hazard_detected() {
        use crate::graph_ir::ConstBuf;

        const N: usize = 4;
        let module = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![N]);
            let body = b.compute(N, |b, i| b.index(a, &[i]));
            Arc::new(b.finish("copy_a", body))
        };

        let bytes = (N * 4) as i64;
        let mk = |g: &mut GraphBuilder, name: &str| -> BufId {
            g.add_buf(BufInfo {
                name: Some(name.to_string()),
                device_type: DeviceType::Cuda(0),
                size: Quast::cst(bytes),
                elem_size: 4,
            })
        };

        // Ill-ordered graph: insert the reading kernel FIRST, then a
        // const + memcpy that (supposedly) initializes `data`. Since
        // GraphExe replays in insertion order, at runtime the kernel
        // would fire against uninitialized bytes.
        let mut g = GraphBuilder::new();
        let init = mk(&mut g, "data_init");
        let data = mk(&mut g, "data");
        let out = mk(&mut g, "out");
        g.register_output(out);
        // Kernel first — reader of `data`.
        g.insert_kernel(module.clone(), [data], [out], &[]);
        // Then the const + memcpy that would fill `data`.
        g.insert_const(init, ConstBuf::HostBuf(vec![0u8; N * 4]));
        g.insert_memcpy(init, data);

        let err = validate_interface(&g, DeviceType::Cuda(0))
            .expect_err("topo-order hazard must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("data") && msg.contains("first read at node"),
            "diagnostic must name the offending buffer and its earliest reader/writer indices: {msg}"
        );

        // Sanity: swap the insertion order so writers come first, and
        // validate_interface accepts the graph.
        let mut g = GraphBuilder::new();
        let init = mk(&mut g, "data_init");
        let data = mk(&mut g, "data");
        let out = mk(&mut g, "out");
        g.register_output(out);
        g.insert_const(init, ConstBuf::HostBuf(vec![0u8; N * 4]));
        g.insert_memcpy(init, data);
        g.insert_kernel(module, [data], [out], &[]);
        validate_interface(&g, DeviceType::Cuda(0))
            .expect("write-before-read insertion order must validate");
    }

    // -----------------------------------------------------------------
    // CPU-only tests for typecheck + canonicalize graph passes.
    // Exercise them directly against a manually built GraphBuilder to
    // avoid any GPU dependency.
    // -----------------------------------------------------------------

    fn one_kernel_module() -> Arc<ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let body = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            let two = b.const_field(2);
            b.mul(ai, two)
        });
        Arc::new(b.finish("scale_by_two", body))
    }

    fn two_kernel_module() -> Arc<ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![8]);
        let t = b.compute(8, |b, i| {
            let ai = b.index(a, &[i]);
            let two = b.const_field(2);
            b.mul(ai, two)
        });
        let t = b.let_bound(t);
        let out = b.compute(8, |b, i| {
            let ti = b.index(t, &[i]);
            let ai = b.index(a, &[i]);
            b.add(ti, ai)
        });
        Arc::new(b.finish("chain", out))
    }

    fn buf_of(g: &mut GraphBuilder, name: &str, bytes: i64) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(bytes),
            elem_size: 4,
        })
    }

    /// `typecheck` fills every aliased node's `types` with the same
    /// `Arc<TypeMap>` pointer — one `type_infer` call per unique module.
    #[test]
    fn typecheck_fans_types_to_aliases() {
        let mut g = GraphBuilder::new();
        let a0 = buf_of(&mut g, "a0", 16);
        let a1 = buf_of(&mut g, "a1", 16);
        let o0 = buf_of(&mut g, "o0", 16);
        let o1 = buf_of(&mut g, "o1", 16);
        // Same HIR built twice — Arc-distinct but structurally identical.
        g.insert_kernel(one_kernel_module(), [a0], [o0], &[]);
        g.insert_kernel(one_kernel_module(), [a1], [o1], &[]);

        GraphCompiler::new().typecheck(&mut g).unwrap();

        let kernel_nodes = crate::graph_ir::kernel_node_indices(&g);
        assert_eq!(kernel_nodes.len(), 2);
        let t0 = crate::graph_ir::kernel_at(&g, kernel_nodes[0])
            .types
            .clone()
            .expect("typecheck populates types");
        let t1 = crate::graph_ir::kernel_at(&g, kernel_nodes[1])
            .types
            .clone()
            .expect("typecheck populates types");
        assert!(Arc::ptr_eq(&t0, &t1), "aliases must share one Arc<TypeMap>");
    }

    /// `canonicalize` on a graph containing a single-kernel module leaves
    /// the node count unchanged, marks it canonical, and gives it a
    /// refreshed types map. Calling it a second time is a no-op.
    #[test]
    fn canonicalize_single_kernel_is_idempotent() {
        let mut g = GraphBuilder::new();
        let a = buf_of(&mut g, "a", 16);
        let o = buf_of(&mut g, "o", 16);
        g.insert_kernel(one_kernel_module(), [a], [o], &[]);

        let gc = GraphCompiler::new();
        gc.canonicalize(&mut g).unwrap();
        let idxs = crate::graph_ir::kernel_node_indices(&g);
        assert_eq!(idxs.len(), 1);
        let hash_first = crate::graph_ir::kernel_at(&g, idxs[0])
            .hash
            .expect("kernel_dedup filled hash");
        assert!(crate::graph_ir::kernel_at(&g, idxs[0]).canonical);
        let module_first = crate::graph_ir::kernel_at(&g, idxs[0]).module.clone();

        // Second call: still one node, same hash, same Arc.
        gc.canonicalize(&mut g).unwrap();
        let idxs = crate::graph_ir::kernel_node_indices(&g);
        assert_eq!(idxs.len(), 1);
        let node = crate::graph_ir::kernel_at(&g, idxs[0]);
        assert_eq!(node.hash, Some(hash_first));
        assert!(Arc::ptr_eq(&node.module, &module_first));
    }

    /// Same symbolic HIR inserted twice at different `K` bindings: the
    /// smaller K is below `REDUCE_TREE_MIN` and stays untouched; the
    /// larger crosses the gate and gets rewritten. Together they
    /// exercise the memo's divergence rule (same HIR hash, different
    /// relevant bindings → distinct rewrite outcomes cached separately).
    #[test]
    fn lower_reduce_diverges_by_k_binding() {
        // `compute [2] |t| reduce [K] |r| x[r]`.
        // At K = 2, gate declines (K < REDUCE_TREE_MIN = 4).
        // At K = 128, gate fires (K power-of-two, M = 2 < 256).
        fn build() -> Arc<ir::Module> {
            let mut b = IRBuilder::new();
            let n = b.symbol("n");
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let body = b.compute(2, |b, _t| b.reduce_add(n, |b, r| b.index(x, &[r])));
            Arc::new(b.finish("row_reduce", body))
        }

        let mut g = GraphBuilder::new();
        let a0 = buf_of(&mut g, "a0", 2 * 4);
        let a1 = buf_of(&mut g, "a1", 128 * 4);
        let o0 = buf_of(&mut g, "o0", 2 * 4);
        let o1 = buf_of(&mut g, "o1", 2 * 4);
        for (module, ins, outs, n) in [
            (build(), vec![a0], vec![o0], 2),
            (build(), vec![a1], vec![o1], 128),
        ] {
            let bindings: BTreeMap<String, i64> = [("n".to_string(), n)].into();
            g.nodes
                .push(crate::graph_ir::GraphNode::Kernel(KernelModuleNode {
                    module,
                    param_bindings: bindings,
                    inputs: ins,
                    outputs: outs,
                    types: None,
                    hash: None,
                    canonical: false,
                    fusion_history: None,
                }));
        }

        let orig_hashes: Vec<[u8; 32]> = crate::graph_ir::kernel_node_indices(&g)
            .iter()
            .map(|&i| crate::module_hash::module_hash(&crate::graph_ir::kernel_at(&g, i).module))
            .collect();

        GraphCompiler::new().lower_reduce(&mut g).unwrap();

        let post: Vec<[u8; 32]> = crate::graph_ir::kernel_node_indices(&g)
            .iter()
            .map(|&i| crate::module_hash::module_hash(&crate::graph_ir::kernel_at(&g, i).module))
            .collect();
        assert_eq!(
            post[0], orig_hashes[0],
            "K=2 (< REDUCE_TREE_MIN): gate should decline and leave module untouched"
        );
        assert_ne!(
            post[1], orig_hashes[1],
            "K=128, M=2: rewrite should fire and mint a new module"
        );
    }

    /// Two sizes of one template: after monomorphize the residuals
    /// stay symbolic in the outer bound (which is the only place `n`
    /// appears), so they structurally agree; the trailing kernel_dedup
    /// collapses them onto one Arc, and the per-group block hint is
    /// picked from the larger size.
    #[test]
    fn monomorphize_collapses_cross_size_residuals() {
        // `compute [n] |i| x[i] * 2` — n survives (outer bound) so both
        // sizes share the same residual.
        fn build() -> Arc<ir::Module> {
            let mut b = IRBuilder::new();
            let n = b.symbol("n");
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                let two = b.const_field(2);
                b.mul(xi, two)
            });
            Arc::new(b.finish("outer_only", body))
        }

        let mut g = GraphBuilder::new();
        let a0 = buf_of(&mut g, "a0", 64 * 4);
        let a1 = buf_of(&mut g, "a1", 512 * 4);
        let o0 = buf_of(&mut g, "o0", 64 * 4);
        let o1 = buf_of(&mut g, "o1", 512 * 4);
        for (module, ins, outs, n) in [
            (build(), vec![a0], vec![o0], 64),
            (build(), vec![a1], vec![o1], 512),
        ] {
            let bindings: BTreeMap<String, i64> = [("n".to_string(), n)].into();
            g.nodes
                .push(crate::graph_ir::GraphNode::Kernel(KernelModuleNode {
                    module,
                    param_bindings: bindings,
                    inputs: ins,
                    outputs: outs,
                    types: None,
                    hash: None,
                    canonical: false,
                    fusion_history: None,
                }));
        }

        GraphCompiler::new().monomorphize(&mut g).unwrap();

        let idxs = crate::graph_ir::kernel_node_indices(&g);
        assert_eq!(idxs.len(), 2);
        let m0 = crate::graph_ir::kernel_at(&g, idxs[0]).module.clone();
        let m1 = crate::graph_ir::kernel_at(&g, idxs[1]).module.clone();
        assert!(
            Arc::ptr_eq(&m0, &m1),
            "cross-size residuals must collapse onto one Arc"
        );
        // Block hint is set from max_outer = 512, warp-rounded, capped
        // at 256.
        assert_eq!(m0.builder.block_hint(), Some(256));
        // Bindings survive by name.
        assert_eq!(
            crate::graph_ir::kernel_at(&g, idxs[0]).param_bindings,
            [("n".to_string(), 64)].into()
        );
        assert_eq!(
            crate::graph_ir::kernel_at(&g, idxs[1]).param_bindings,
            [("n".to_string(), 512)].into()
        );
    }

    /// A second call is a no-op: the module Arcs and hashes stay
    /// stable across the second monomorphize invocation.
    #[test]
    fn monomorphize_is_idempotent() {
        fn build() -> Arc<ir::Module> {
            let mut b = IRBuilder::new();
            let n = b.symbol("n");
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                let two = b.const_field(2);
                b.mul(xi, two)
            });
            Arc::new(b.finish("outer_only", body))
        }

        let mut g = GraphBuilder::new();
        let a = buf_of(&mut g, "a", 64 * 4);
        let o = buf_of(&mut g, "o", 64 * 4);
        g.nodes
            .push(crate::graph_ir::GraphNode::Kernel(KernelModuleNode {
                module: build(),
                param_bindings: [("n".to_string(), 64)].into(),
                inputs: vec![a],
                outputs: vec![o],
                types: None,
                hash: None,
                canonical: false,
                fusion_history: None,
            }));

        let gc = GraphCompiler::new();
        gc.monomorphize(&mut g).unwrap();
        let idxs = crate::graph_ir::kernel_node_indices(&g);
        let first_arc = crate::graph_ir::kernel_at(&g, idxs[0]).module.clone();
        let first_hash = crate::graph_ir::kernel_at(&g, idxs[0]).hash;

        gc.monomorphize(&mut g).unwrap();
        let idxs = crate::graph_ir::kernel_node_indices(&g);
        let node = crate::graph_ir::kernel_at(&g, idxs[0]);
        assert!(
            Arc::ptr_eq(&node.module, &first_arc),
            "second monomorphize should keep the same Arc"
        );
        assert_eq!(node.hash, first_hash);
    }

    /// `canonicalize` splits a multi-kernel module into one graph node
    /// per split kernel, in dependency order, with a fresh intermediate
    /// buffer between them and the parent's `outputs` binding to the
    /// last kernel's output.
    #[test]
    fn canonicalize_splits_multi_kernel_module() {
        let mut g = GraphBuilder::new();
        let a = buf_of(&mut g, "a", 32);
        let out = buf_of(&mut g, "out", 32);
        // Multi-kernel module inserted directly (skipping insert_kernel's
        // eager split so canonicalize gets to do the splitting).
        g.nodes
            .push(crate::graph_ir::GraphNode::Kernel(KernelModuleNode {
                module: two_kernel_module(),
                param_bindings: BTreeMap::new(),
                inputs: vec![a],
                outputs: vec![out],
                types: None,
                hash: None,
                canonical: false,
                fusion_history: None,
            }));

        GraphCompiler::new().canonicalize(&mut g).unwrap();

        let idxs = crate::graph_ir::kernel_node_indices(&g);
        assert_eq!(idxs.len(), 2, "multi-kernel module must split into 2 nodes");
        let k0 = crate::graph_ir::kernel_at(&g, idxs[0]);
        let k1 = crate::graph_ir::kernel_at(&g, idxs[1]);
        assert!(k0.canonical && k1.canonical);
        // k0's output feeds k1's input.
        assert_eq!(k0.outputs.len(), 1);
        assert_eq!(k1.inputs, vec![k0.outputs[0], a]);
        // Parent's `out` buffer is the terminal writer.
        assert_eq!(k1.outputs, vec![out]);
        // The intermediate is on the parent's device.
        let mid = k0.outputs[0];
        assert_eq!(g.buf_info(mid).device_type, DeviceType::Cuda(0));
    }
}
