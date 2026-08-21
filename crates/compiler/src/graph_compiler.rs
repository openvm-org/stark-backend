//! Graph compilation pipeline.
//!
//! [`GraphCompiler`] consumes a [`GraphBuilder`], validates its registered
//! input/output interface, plans memory via [`crate::planner::plan_raw`],
//! compiles every [`GraphNode::Kernel`]'s module through
//! [`crate::module_compiler::ModuleCompiler`], and packages the whole thing into a
//! [`GraphExe`]. Every buffer — inputs, outputs, and graph-level
//! intermediates — lives at a fixed offset inside one device pool, so
//! consecutive [`GraphExe::run`]s replay identical device addresses (the
//! CUDA-graph-capture contract).
//!
//! Feature-gated behind `planner` (needs the CP-SAT planner + OR-Tools).

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{
    graph_compiler_config::{ConfigError, FusionConfig, GraphCompilerConfig},
    graph_exe::{ExeBlackbox, ExeKernel, ExeNode, GraphExe},
    graph_ir::{
        classify_buf_uses, for_each_unique_kernel, kernel_at, kernel_at_mut, BufId, BufInfo,
        DeviceType, GraphBuilder, GraphNode,
    },
    ir::{self, VarId},
    kernel_cache::KernelCache,
    kernel_ir::KirProgram,
    module_compiler::ModuleCompiler,
    module_hash::module_hash,
    passes::{
        check_accesses::check_module_accesses,
        fusion::{fuse_graph, FusionOptions},
        fusion_utils::{dce, FusionReport},
        type_infer,
    },
    planner::{
        access_from_node, eval_size, AbstractTimingGraph, ListSchedulerV1, PlanError,
        SchedulerMode, StreamInstr, StreamMemoryPlan,
    },
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
    /// Optional per-node runtime timings (ms), indexed by
    /// post-fuse+dce graph node position. Threaded onto the
    /// [`AbstractTimingGraph`] the memory planner consumes; when
    /// `None` [`Self::plan_memory`] falls back to uniform `1.0` (the
    /// depth-only priority used on the first compile).
    node_times: Option<Vec<f64>>,
    /// On-disk cache queried before hitting nvcc; kernels found here skip
    /// compilation entirely. `None` disables the cache. Defaults to a shared
    /// `~/.openvm/kernel_cache` with the [`KernelCache`] defaults.
    kernel_cache: Option<Arc<KernelCache>>,
    /// Kernel-fusion options; `None` disables the pass. Defaults to
    /// [`FusionOptions::default`].
    fusion: Option<Box<FusionOptions>>,
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
            node_times: None,
            kernel_cache: Some(Arc::new(KernelCache::new())),
            fusion: Some(Box::new(FusionOptions::default())),
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
    /// Attach per-node runtime timings (ms), typically populated from a
    /// prior [`GraphExe::collect_graph_info`]. The vec length must match
    /// the post-fuse+dce node count at compile time — [`Self::plan_memory`]
    /// will panic in debug builds otherwise. Pass `None` to reset back to
    /// the default (uniform 1.0 / depth-only priority).
    pub fn node_times(mut self, times: Option<Vec<f64>>) -> Self {
        self.node_times = times;
        self
    }

    pub fn scheduler(mut self, scheduler: SchedulerMode) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// Convenience toggle: enables stream-aware list scheduling
    /// ([`SchedulerMode::ListV1`]) with the given tuning parameters.
    /// When `params.max_concurrency == 1`, this degenerates to a
    /// single-stream schedule identical in shape to the heuristic
    /// backend (no `WaitOn` instructions, no auxiliary streams).
    pub fn stream_planning(mut self, params: ListSchedulerV1) -> Self {
        self.scheduler = SchedulerMode::ListV1 { params };
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

    /// Overrides the kernel-fusion tunables. Symbol bindings registered
    /// via [`Self::symbol`] are merged into `opts.graph_symbols` at fuse
    /// time, overriding any caller-set bindings for the same symbol —
    /// the compiler's `env` is authoritative because memory planning
    /// and size evaluation already use it.
    ///
    /// Without the `planner-ortools` feature, the fusion pass still
    /// enumerates and cost-ranks candidates but extraction is limited to
    /// the brute-force extractor (small graphs) or the original graph with
    /// [`FallbackReason::SolverUnavailable`](crate::passes::fusion::FallbackReason).
    pub fn fusion_options(mut self, opts: FusionOptions) -> Self {
        self.fusion = Some(Box::new(opts));
        self
    }

    /// Disables the kernel-fusion pass: the graph is compiled exactly as
    /// built, one launch per inserted kernel.
    pub fn without_fusion(mut self) -> Self {
        self.fusion = None;
        self
    }

    /// Builds a `GraphCompiler` from a serialized [`GraphCompilerConfig`].
    /// Fields outside the TOML surface (symbol bindings, fusion
    /// estimator / artifact / graph_symbols) are left at their builder
    /// defaults; callers that need them keep using the builder setters on
    /// top of the returned compiler.
    pub fn from_config(cfg: GraphCompilerConfig) -> Self {
        let GraphCompilerConfig {
            device,
            module_compiler,
            scheduler,
            kernel_cache,
            fusion,
        } = cfg;

        let mut compiler = Self::new()
            .device(device.into())
            .scheduler(scheduler.into());

        // ModuleCompiler passthrough.
        compiler = compiler
            .arch(module_compiler.arch.clone())
            .nvcc(module_compiler.nvcc.clone())
            .verbosity(module_compiler.verbosity)
            .check_accesses(module_compiler.check_accesses)
            .nvcc_timeout(module_compiler.nvcc_timeout());
        if let Some(dir) = module_compiler.dump_dir.clone() {
            compiler = compiler.dump_dir(dir);
        }
        for flag in module_compiler.extra_nvcc_flags {
            compiler = compiler.add_flag(flag);
        }

        // Kernel cache.
        compiler = match kernel_cache.build() {
            Some(cache) => compiler.kernel_cache(cache),
            None => compiler.without_kernel_cache(),
        };

        // Fusion strategy.
        compiler = match fusion {
            FusionConfig::Off => compiler.without_fusion(),
            FusionConfig::On(cfg) => compiler.fusion_options(cfg.to_options()),
        };

        compiler
    }

    /// Reads a TOML config from `path` and builds a `GraphCompiler` via
    /// [`Self::from_config`]. Missing fields fall back to
    /// [`GraphCompilerConfig::default`].
    pub fn from_toml(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let cfg =
            GraphCompilerConfig::from_toml_str(&text).map_err(|source| ConfigError::Parse {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(Self::from_config(cfg))
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
    /// `apply_fusion` grafts producer bodies inline, which can denormalize
    /// the fused module and expose new must-be-concrete inner bounds; a
    /// fused kernel that keeps a symbolic outer bound also needs a block
    /// hint stamped. So on rounds that actually fused, we re-run
    /// `canonicalize` + `monomorphize` to re-normalize and block-hint the
    /// fused outputs.
    ///
    /// Returns the [`FusionReport`] from the fusion pass, or `None`
    /// when fusion is disabled.
    pub fn fuse(&self, g: &mut GraphBuilder) -> Result<Option<FusionReport>, CompileError> {
        self.lower_reduce(g)?;
        self.monomorphize(g)?;
        self.canonicalize(g)?;
        let report = match &self.fusion {
            None => None,
            Some(opts) => {
                let mut opts = opts.as_ref().clone();
                opts.graph_symbols
                    .extend(self.env.iter().map(|(k, v)| (*k, *v)));
                Some(
                    fuse_graph(g, &opts)
                        .map_err(|e| CompileError::Verify(format!("fusion: {e}")))?,
                )
            }
        };

        self.canonicalize(g)?;
        self.monomorphize(g)?;
        g.plan = None;
        Ok(report)
    }

    /// Drops kernel nodes whose outputs no live node reads. Wraps
    /// [`fusion_utils::dce`](crate::passes::fusion_utils::dce) and
    /// resets `g.plan` on removal (structural mutation invalidates the
    /// memory plan). Returns the number of nodes removed.
    pub fn dce(&self, g: &mut GraphBuilder) -> usize {
        let removed = dce(g);
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
        // Build an ATG (with synthetic ordering-edge buffers) and hand
        // it to the planner. Sizes are evaluated against `self.env`.
        let mut bufs = g.bufs.clone();
        for (i, info) in bufs.iter_mut().enumerate() {
            if info.device_type == self.device {
                let s = eval_size(BufId(i), &info.size, &self.env).map_err(|e| {
                    CompileError::Type(format!("graph plan: {e}"))
                })?;
                info.concrete_size = s.max(0) as usize;
            }
        }
        let reads: Vec<Vec<BufId>> = g
            .nodes
            .iter()
            .map(|n| access_from_node(n).reads)
            .collect();
        let writes: Vec<Vec<BufId>> = g
            .nodes
            .iter()
            .map(|n| access_from_node(n).writes)
            .collect();
        let node_times = self
            .node_times
            .clone()
            .filter(|t| t.len() == g.nodes.len())
            .unwrap_or_else(|| vec![1.0; g.nodes.len()]);
        let atg = AbstractTimingGraph::from_accesses(
            bufs,
            &reads,
            &writes,
            node_times,
            self.device,
            g.input_bufs().to_vec(),
            g.output_bufs().to_vec(),
        );
        let plan = crate::planner::plan(&atg, &self.scheduler).map_err(|e| match e {
            PlanError::UnboundSizeSymbol { .. } | PlanError::NegativeSize { .. } => {
                CompileError::Type(format!("graph plan: {e}"))
            }
            PlanError::Infeasible(_) => CompileError::Runtime(format!("graph plan: {e}")),
        })?;
        g.plan = Some(plan);
        Ok(())
    }

    /// Consumes the graph, validates its registered interface, plans it and
    /// compiles every structured kernel. Runs in four stages:
    ///
    /// 1. **Normalize**: [`Self::fuse`] (or the explicit `lower_reduce` → `monomorphize` →
    ///    `canonicalize` chain when fusion is disabled) + [`Self::dce`]. Post-passes every Kernel
    ///    node's module is a canonical, monomorphized, single-kernel residual and `kernel_dedup`
    ///    has collapsed structurally identical modules onto one `Arc<ir::Module>`.
    /// 2. **Compile kernels**: [`Self::compile_unique_kernels`] JITs one artifact per unique module
    ///    Arc in parallel (cache hits skip nvcc).
    /// 3. **Plan memory**: [`Self::plan_memory`] fills `g.plan`.
    /// 4. **Assemble**: build one [`ExeNode`] per graph node in insertion order (execution order is
    ///    chosen by the plan at runtime), evaluating memcpy/memset offsets and validating kernel
    ///    input/output sizes against the plan.
    ///
    /// The registered interface (see [`GraphBuilder::register_input`] /
    /// [`GraphBuilder::register_output`]) is validated up front: inputs must
    /// exist on the target device, be distinct, never be written and be
    /// read at least once; outputs must exist on the target device, be
    /// distinct and be written at least once; any unregistered buffer that
    /// is read but never written is an error.
    pub fn compile(self, graph: GraphBuilder) -> Result<GraphExe, CompileError> {
        self.compile_with_post_fuse_hook(graph, |_| Ok(()))
    }

    /// Same as [`Self::compile`] but invokes `hook` on the graph state
    /// *after* `restore_ssa → fuse → dce`, right before the compile
    /// pipeline drains nodes into the exe. That handoff point is the
    /// same graph the exe's `ExeNode`s mirror index-by-index: node `i`
    /// on the hook side becomes exe node `i`, which is also the
    /// [`crate::graph_info::GraphInfo::nodes`] index a subsequent
    /// [`GraphExe::collect_graph_info`] populates.
    ///
    /// Callers therefore use the hook to serialize the post-fuse+dce
    /// `GraphBuilder` (see [`crate::graph_serializer`]) without having
    /// to re-run fusion themselves — the fusion solver isn't
    /// deterministic under multi-worker CP-SAT, so a re-fused graph
    /// may pick a different solution and shift node indices.
    pub fn compile_with_post_fuse_hook<F>(
        self,
        mut graph: GraphBuilder,
        hook: F,
    ) -> Result<GraphExe, CompileError>
    where
        F: FnOnce(&GraphBuilder) -> Result<(), CompileError>,
    {
        validate_interface(&graph, self.device)?;
        // Snapshot the caller-visible content hash *before* any pass rewrites
        // the graph — the graph serializer pairs a payload with a
        // structurally-equivalent `GraphBuilder` (possibly one that has
        // since gone through fusion), so both sides must expose the same
        // pre-pass fingerprint via `original_hash`.
        let graph_hash = graph.original_hash();
        let nodes_before = graph.nodes.len();

        // Stage 1: normalize the graph. Post-passes every Kernel node's
        // module is a canonical, monomorphized, single-kernel residual
        // with `hash` set; `kernel_dedup` has collapsed structurally
        // identical modules onto one `Arc<ir::Module>`, so the compile
        // stage below can dedup by Arc pointer alone.
        let t_passes = std::time::Instant::now();
        let fusion_report = match &self.fusion {
            Some(_) => self.fuse(&mut graph)?,
            None => {
                self.lower_reduce(&mut graph)?;
                self.monomorphize(&mut graph)?;
                self.canonicalize(&mut graph)?;
                // canonicalize + split_program rebuild each kernel through
                // a fresh IRBuilder that drops the block hint, and the new
                // outer bounds (post-flatten of nested computes) may differ
                // from the pre-canonicalize ones anyway. Re-run monomorphize
                // so split children get a block hint sized from their own
                // `max_outer`.
                self.monomorphize(&mut graph)?;
                None
            }
        };
        self.dce(&mut graph);
        eprintln!(
            "[compile] passes: {:.1} ms ({nodes_before} -> {} nodes)",
            t_passes.elapsed().as_secs_f64() * 1e3,
            graph.nodes.len(),
        );

        // Hand the caller a peek at the post-fuse+dce graph state
        // before any kernel compilation or planning happens. This is
        // the exact graph shape that maps 1:1 into `exe.nodes` further
        // down (`graph.nodes.drain(..)` preserves order).
        hook(&graph)?;

        // Stage 2: compile every unique kernel module in parallel.
        let t_kernels = std::time::Instant::now();
        let CompiledKernels {
            mut kernels,
            kirs,
            module_hashes,
            kernel_of_ptr,
            num_unique_modules,
            num_cached_modules,
        } = self.compile_unique_kernels(&graph)?;
        eprintln!(
            "[compile] kernels: {:.1} ms ({num_unique_modules} unique modules, \
             {num_cached_modules} from cache)",
            t_kernels.elapsed().as_secs_f64() * 1e3,
        );

        // Stage 3: plan graph memory.
        let t_plan = std::time::Instant::now();
        self.plan_memory(&mut graph)?;
        let plan = graph.plan.take().expect("plan_memory populated g.plan");
        let sizes = evaluate_sizes_bufs(&graph.bufs, &self.env)?;
        eprintln!(
            "[compile] memory plan: {:.1} ms ({} stream(s), {} event(s), peak pool {} bytes)",
            t_plan.elapsed().as_secs_f64() * 1e3,
            plan.num_streams,
            plan.num_events,
            plan.peak_bytes,
        );
        if std::env::var("GRAPH_EXE_DUMP_SCHEDULE").ok().as_deref() == Some("1") {
            dump_schedule(&graph, &plan);
        }

        // Stage 4: derive `ExeNode`s in the graph's insertion order.
        // Execution order comes from `plan.order` at runtime.
        let t_assemble = std::time::Instant::now();
        let input_bufs = graph.input_bufs().to_vec();
        let output_bufs = graph.output_bufs().to_vec();
        let bufs = graph.bufs.clone();
        let mut exe_nodes: Vec<ExeNode> = Vec::with_capacity(graph.nodes.len());
        for (idx, node) in graph.nodes.drain(..).enumerate() {
            let stream = plan.stream.get(idx).copied().unwrap_or(0);
            exe_nodes.push(build_exe_node(
                node,
                &kernel_of_ptr,
                &mut kernels,
                &sizes,
                &self.env,
                stream,
            )?);
        }
        eprintln!(
            "[compile] assemble: {:.1} ms ({} exe nodes)",
            t_assemble.elapsed().as_secs_f64() * 1e3,
            exe_nodes.len(),
        );

        // KIR and module hashes are no longer retained on the exe — the
        // graph serializer snapshots the pre-pass `GraphBuilder` instead.
        drop((kirs, module_hashes));

        Ok(GraphExe::from_compiled(
            graph_hash,
            plan,
            sizes,
            kernels,
            exe_nodes,
            input_bufs,
            output_bufs,
            self.device,
            bufs,
            num_unique_modules,
            num_cached_modules,
            fusion_report,
        ))
    }

    /// Compiles one [`KernelProgram`] per unique `Arc<ir::Module>` seen
    /// in `graph.nodes`. The graph passes have already dedup'd
    /// structurally identical modules onto one Arc, so pointer equality
    /// is the full deduplication key. Cache hits skip nvcc; misses are
    /// JIT'd in parallel (see [`jit_kernel_misses`]). `check_accesses`
    /// runs graph-side per unique `(module ptr, bindings)` pair since it
    /// needs concrete binding values.
    fn compile_unique_kernels(
        &self,
        graph: &GraphBuilder,
    ) -> Result<CompiledKernels, CompileError> {
        use rayon::prelude::*;

        // Collect unique modules by Arc pointer in first-seen order.
        let mut unique_modules: Vec<Arc<ir::Module>> = Vec::new();
        let mut kernel_of_ptr: HashMap<*const ir::Module, usize> = HashMap::new();
        let mut checked: HashSet<(*const ir::Module, BTreeMap<String, i64>)> = HashSet::new();
        for node in &graph.nodes {
            let GraphNode::Kernel(k) = node else { continue };
            let ptr = Arc::as_ptr(&k.module);
            kernel_of_ptr.entry(ptr).or_insert_with(|| {
                unique_modules.push(k.module.clone());
                unique_modules.len() - 1
            });
            if self.module_compiler.check_accesses
                && checked.insert((ptr, k.param_bindings.clone()))
            {
                check_module_accesses(&k.module, &k.param_bindings)?;
            }
        }

        // Per-module structural hashes: cache key today, serialized-payload
        // key tomorrow (the graph serializer emits these so a loaded payload
        // can hit the same on-disk cache entry).
        let module_hashes: Vec<[u8; 32]> = unique_modules.iter().map(|m| module_hash(m)).collect();

        if std::env::var("GRAPH_EXE_DUMP_MODULES").ok().as_deref() == Some("1") {
            let mut by_name: BTreeMap<&str, usize> = BTreeMap::new();
            for m in &unique_modules {
                *by_name.entry(m.name.as_str()).or_default() += 1;
            }
            eprintln!("[compile] {} unique modules by name:", unique_modules.len());
            for (name, count) in by_name {
                eprintln!("[compile]   {count:4}  {name}");
            }
        }

        // Probe the on-disk cache before spawning nvcc.
        let mut compiled: Vec<Option<KernelProgram>> =
            (0..unique_modules.len()).map(|_| None).collect();
        let mut misses: Vec<usize> = Vec::new();
        let mut num_cached_modules = 0usize;
        for (i, hash) in module_hashes.iter().enumerate() {
            let hit = match &self.kernel_cache {
                Some(cache) => cache.get_by_hash(hash)?,
                None => None,
            };
            match hit {
                Some(km) => {
                    compiled[i] = Some(km);
                    num_cached_modules += 1;
                }
                None => misses.push(i),
            }
        }

        // Lower every unique module to KIR in parallel. Cache hits pay a
        // small lower cost here (no codegen/nvcc) so the graph serializer
        // has KIR ready to emit without ever revisiting the HIR module.
        let kirs: Vec<KirProgram> = unique_modules
            .par_iter()
            .map(|m| self.module_compiler.lower((**m).clone()))
            .collect::<Result<Vec<_>, _>>()?;

        // Codegen the miss set in parallel, using each miss's already-lowered KIR.
        for res in codegen_kernel_misses(
            &misses,
            &kirs,
            &module_hashes,
            &self.module_compiler,
            self.kernel_cache.as_deref(),
            num_cached_modules,
        ) {
            let (i, km) = res?;
            compiled[i] = Some(km);
        }

        let kernels: Vec<KernelProgram> = compiled
            .into_iter()
            .map(|slot| slot.expect("every unique-module slot filled"))
            .collect();
        let num_unique_modules = kernels.len();
        Ok(CompiledKernels {
            kernels,
            kirs,
            module_hashes,
            kernel_of_ptr,
            num_unique_modules,
            num_cached_modules,
        })
    }
}

/// Return value of [`GraphCompiler::compile_unique_kernels`].
struct CompiledKernels {
    /// One artifact per unique `Arc<ir::Module>`, indexed by first-seen
    /// order in `graph.nodes`.
    kernels: Vec<KernelProgram>,
    /// KIR for each unique module, parallel to [`Self::kernels`]. Preserved
    /// on [`GraphExe`] for the graph serializer to emit; a fresh load runs
    /// [`ModuleCompiler::codegen`] on the KIR when the on-disk kernel cache
    /// misses.
    kirs: Vec<KirProgram>,
    /// Stable structural hash of each unique HIR module, parallel to
    /// [`Self::kernels`]. Used as the on-disk kernel-cache key and preserved
    /// through the graph serializer.
    module_hashes: Vec<[u8; 32]>,
    /// Maps a kernel node's `Arc::as_ptr(&module)` to its index in
    /// [`Self::kernels`]. Every kernel node in the graph appears here.
    kernel_of_ptr: HashMap<*const ir::Module, usize>,
    num_unique_modules: usize,
    num_cached_modules: usize,
}

/// Parallel codegen for the given cache-miss indices, with periodic progress
/// output and a post-run summary of failures/timeouts. Uses each miss's
/// already-lowered KIR (produced up-front so cache hits and misses share
/// the same lowering path) plus its precomputed HIR `module_hash` for the
/// cache insert. Best-effort insert into `cache` when supplied — an insert
/// failure never fails the compile.
fn codegen_kernel_misses(
    misses: &[usize],
    kirs: &[KirProgram],
    module_hashes: &[[u8; 32]],
    module_compiler: &ModuleCompiler,
    cache: Option<&KernelCache>,
    num_cached_modules: usize,
) -> Vec<Result<(usize, KernelProgram), CompileError>> {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    use rayon::prelude::*;

    let n_to_compile = misses.len();
    if n_to_compile > 0 {
        let total = n_to_compile + num_cached_modules;
        eprintln!(
            "[compile] {n_to_compile}/{total} unique kernel modules need fresh nvcc \
             invocations ({num_cached_modules} served from on-disk cache)",
        );
    }
    let done = AtomicUsize::new(0);
    let next_tick = AtomicUsize::new(0);
    let start = Instant::now();
    let out: Vec<Result<(usize, KernelProgram), CompileError>> = misses
        .par_iter()
        .map(|&i| {
            let km = module_compiler.codegen(kirs[i].clone())?;
            if let Some(c) = cache {
                let _ = c.insert_by_hash(&module_hashes[i], &km);
            }
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            // Advance past every 5% tick we crossed; whoever CAS's past a
            // tick prints it.
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
            Ok((i, km))
        })
        .collect();
    if n_to_compile == 0 {
        return out;
    }
    let succeeded = out.iter().filter(|r| r.is_ok()).count();
    let failed = n_to_compile - succeeded;
    if failed == 0 {
        eprintln!(
            "[compile] done ({succeeded}/{n_to_compile}, {:.1}s total)",
            start.elapsed().as_secs_f64(),
        );
        return out;
    }
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
        eprintln!("[compile] {} non-timeout failure(s):", other_failures.len());
        for e in other_failures.iter().take(5) {
            eprintln!("[compile]   {e}");
        }
        if other_failures.len() > 5 {
            eprintln!("[compile]   ...and {} more", other_failures.len() - 5);
        }
    }
    out
}

/// Turns one `GraphNode` into its `ExeNode`. Evaluates memcpy/memset
/// offsets under the graph compiler's env, validates range bounds, and
/// for kernel nodes: looks up the compiled artifact by module Arc
/// pointer, binds `param_bindings` on the shared handle, and validates
/// kernel input/output sizes against the plan.
fn build_exe_node(
    node: GraphNode,
    kernel_of_ptr: &HashMap<*const ir::Module, usize>,
    kernels: &mut [KernelProgram],
    sizes: &[usize],
    env: &BTreeMap<VarId, i64>,
    stream: u32,
) -> Result<ExeNode, CompileError> {
    Ok(match node {
        GraphNode::Kernel(k) => {
            let kernel_idx = *kernel_of_ptr
                .get(&Arc::as_ptr(&k.module))
                .expect("compile_unique_kernels visited every Kernel node");
            let set_params = pos_vec(&k.module, &k.param_bindings);
            // Kernel sizes are a function of the currently-bound symbols;
            // bind this node's before validating.
            let m = &mut kernels[kernel_idx];
            let names: Vec<String> = m.params().to_vec();
            for (name, &v) in names.iter().zip(set_params.iter()) {
                m.set_symbol(name, v);
            }
            check_kernel_sizes(m, &k.inputs, &k.outputs, sizes)?;
            ExeNode::Kernel(ExeKernel {
                name: k.module.name.clone(),
                kernel_idx,
                inputs: k.inputs,
                outputs: k.outputs,
                set_params,
                stream,
                debug_module: Arc::clone(&k.module),
            })
        }
        GraphNode::BlackboxKernel(k) => ExeNode::Blackbox(ExeBlackbox { kernel: k, stream }),
        GraphNode::Const(c) => ExeNode::Const(c),
        GraphNode::Memcpy(m) => {
            let src_offset = eval_nonneg(&m.src_offset, env, "memcpy src_offset")?;
            let dst_offset = eval_nonneg(&m.dst_offset, env, "memcpy dst_offset")?;
            let num_bytes = eval_nonneg(&m.num_bytes, env, "memcpy num_bytes")?;
            let src_end = src_offset.checked_add(num_bytes).ok_or_else(|| {
                CompileError::Runtime(format!(
                    "memcpy: src_offset+num_bytes overflows usize \
                     ({src_offset} + {num_bytes})"
                ))
            })?;
            if src_end > sizes[m.src.0] {
                return Err(CompileError::Runtime(format!(
                    "memcpy: src range [{src_offset}..{src_end}) exceeds \
                     buffer {:?} size {}",
                    m.src, sizes[m.src.0]
                )));
            }
            let dst_end = dst_offset.checked_add(num_bytes).ok_or_else(|| {
                CompileError::Runtime(format!(
                    "memcpy: dst_offset+num_bytes overflows usize \
                     ({dst_offset} + {num_bytes})"
                ))
            })?;
            if dst_end > sizes[m.dst.0] {
                return Err(CompileError::Runtime(format!(
                    "memcpy: dst range [{dst_offset}..{dst_end}) exceeds \
                     buffer {:?} size {}",
                    m.dst, sizes[m.dst.0]
                )));
            }
            ExeNode::Memcpy {
                src: m.src,
                src_offset,
                dst: m.dst,
                dst_offset,
                num_bytes,
            }
        }
        GraphNode::Memset(m) => {
            let offset = eval_nonneg(&m.offset, env, "memset offset")?;
            let num_bytes = eval_nonneg(&m.num_bytes, env, "memset num_bytes")?;
            let end = offset.checked_add(num_bytes).ok_or_else(|| {
                CompileError::Runtime(format!(
                    "memset: offset+num_bytes overflows usize ({offset} + {num_bytes})"
                ))
            })?;
            if end > sizes[m.node.0] {
                return Err(CompileError::Runtime(format!(
                    "memset: range [{offset}..{end}) exceeds buffer {:?} size {}",
                    m.node, sizes[m.node.0]
                )));
            }
            ExeNode::Memset {
                buf: m.node,
                offset,
                num_bytes,
                val: m.val,
            }
        }
    })
}

/// `GRAPH_EXE_DUMP_SCHEDULE=1`: prints the planned instruction stream —
/// per-node stream assignment, alias-resolved buffer reads/writes, the
/// event each node records, and every cross-stream wait with the event's
/// producer. This is the ground truth for "why didn't X overlap Y":
/// same-stream program order and `wait` lines are the only two ways the
/// plan serializes nodes.
fn dump_schedule(g: &GraphBuilder, plan: &StreamMemoryPlan) {
    let node_name = |i: usize| -> &str {
        match &g.nodes[i] {
            GraphNode::Kernel(k) => &k.module.name,
            GraphNode::BlackboxKernel(k) => &k.name,
            GraphNode::Const(_) => "const",
            GraphNode::Memcpy(_) => "memcpy",
            GraphNode::Memset(_) => "memset",
        }
    };
    let buf_name = |b: BufId| -> String {
        let mut canon = b;
        while let Some(next) = g.aliases[canon.0] {
            canon = next;
        }
        let name = g.bufs[canon.0].name.as_deref().unwrap_or("?");
        if canon == b {
            format!("b{}:{name}", b.0)
        } else {
            format!("b{}~b{}:{name}", b.0, canon.0)
        }
    };
    let mut event_owner: Vec<usize> = vec![usize::MAX; plan.num_events as usize];
    for (i, ev) in plan.record_event.iter().enumerate() {
        if let Some(e) = ev {
            event_owner[*e as usize] = i;
        }
    }
    eprintln!(
        "[schedule] {} instruction(s), {} stream(s), {} event(s):",
        plan.instructions.len(),
        plan.num_streams,
        plan.num_events,
    );
    for instr in &plan.instructions {
        match *instr {
            StreamInstr::Node(i) => {
                let acc = access_from_node(&g.nodes[i]);
                let reads: Vec<String> = acc.reads.iter().map(|b| buf_name(*b)).collect();
                let writes: Vec<String> = acc.writes.iter().map(|b| buf_name(*b)).collect();
                let ev = match plan.record_event[i] {
                    Some(e) => format!(" -> ev{e}"),
                    None => String::new(),
                };
                eprintln!(
                    "[schedule] s{}: #{i} {} r[{}] w[{}]{ev}",
                    plan.stream[i],
                    node_name(i),
                    reads.join(","),
                    writes.join(","),
                );
            }
            StreamInstr::WaitOn(s, e) => {
                let owner = event_owner[e];
                eprintln!(
                    "[schedule] s{s}: wait ev{e} (#{owner} {} on s{})",
                    node_name(owner),
                    plan.stream[owner],
                );
            }
        }
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
        // Inputs may be written in-place by graph kernels (blackbox
        // `carried_outputs`): the caller populates the initial value
        // before each launch, kernels then read + mutate it. This is
        // the "graph launch skips the initial H2D" pattern — H2D goes
        // to the input's pool slot outside the captured CUDA graph.
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
                concrete_size: bytes as usize,
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
                let ti = b.index(a, &[i]);
                b.mul(ti, s)
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
                concrete_size: bytes as usize,
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
                concrete_size: bytes as usize,
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
            concrete_size: bytes as usize,
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

    /// `collect_graph_info` runs the graph `num_iters` times, records
    /// per-node CUDA-event timings, and returns a snapshot whose
    /// `graph_hash` matches the source exe and whose per-node kinds line
    /// up with the plan's Node instructions.
    #[test]
    fn collect_graph_info_produces_per_node_timings() {
        use crate::graph_info::NodeKind;

        const N: usize = 64;
        let bytes = (N * 4) as i64;
        let mut g = GraphBuilder::new();
        let mk = |g: &mut GraphBuilder, name: &str| -> BufId {
            g.add_buf(BufInfo {
                name: Some(name.into()),
                device_type: DeviceType::Cuda(0),
                size: Quast::cst(bytes),
                concrete_size: bytes as usize,
                elem_size: 4,
            })
        };
        let x = mk(&mut g, "x");
        let y = mk(&mut g, "y");
        let out = mk(&mut g, "out");
        g.register_input(x);
        g.register_output(out);
        // Memcpy x -> y, then scale-by-two kernel: y -> out.
        g.insert_memcpy(x, y);
        let module = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![N]);
            let body = b.compute(N, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                b.mul(ai, two)
            });
            Arc::new(b.finish("scale_by_two_info", body))
        };
        g.insert_kernel(module, [y], [out], &[]);

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile(g)
            .expect("graph compile");
        let expected_hash = *exe.graph_hash();

        let ctx = GpuDeviceCtx::for_current_device().expect("GPU ctx");
        let host: Vec<u32> = (0..N as u32).map(|i| i + 1).collect();
        let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_input = host_bytes.as_slice().to_device_on(&ctx).unwrap();

        let info = exe
            .collect_graph_info(
                &ctx,
                |exe, ctx| exe.set_input(ctx, 0, &d_input),
                /* num_warmup= */ 2,
                /* num_iters= */ 5,
            )
            .expect("collect_graph_info");

        assert_eq!(info.graph_hash, expected_hash);
        assert_eq!(info.num_warmup, 2);
        assert_eq!(info.num_iters, 5);
        assert_eq!(info.nodes.len(), exe.num_nodes());
        // We inserted memcpy first, kernel second.
        assert_eq!(info.nodes[0].kind, NodeKind::Memcpy);
        assert_eq!(info.nodes[1].kind, NodeKind::Kernel);
        assert_eq!(info.nodes[1].name, "scale_by_two_info");
        for nt in &info.nodes {
            // GPU work is strictly positive; sample std is non-negative.
            assert!(nt.mean_ms >= 0.0, "mean must be non-negative");
            assert!(nt.std_ms >= 0.0, "std must be non-negative");
        }
        assert!(info.total_ms_mean > 0.0);
        assert!(info.total_ms_std >= 0.0);

        // Snapshot survives a bincode round-trip.
        let bytes = bincode::serialize(&info).unwrap();
        let round: crate::graph_info::GraphInfo = bincode::deserialize(&bytes).unwrap();
        assert_eq!(round.graph_hash, info.graph_hash);
        assert_eq!(round.nodes.len(), info.nodes.len());
    }
}
