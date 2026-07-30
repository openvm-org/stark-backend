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

use std::{
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

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
    ir::{self, VarId},
    kernel_cache::KernelCache,
    module_hash::module_hash,
    planner::{self, access_from_node, MemoryPlan, NodeAccess, PlanError, SchedulerMode},
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
    scheduler: SchedulerMode,
    /// On-disk cache queried before hitting nvcc; kernels found here skip
    /// compilation entirely. `None` disables the cache. Defaults to a shared
    /// `~/.openvm/kernel_cache` with the [`KernelCache`] defaults.
    kernel_cache: Option<Arc<KernelCache>>,
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
            scheduler: SchedulerMode::default(),
            kernel_cache: Some(Arc::new(KernelCache::new())),
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
        // the compiled `KernelModule` is wrapped in `Rc<RefCell<_>>` and
        // cached by the source `Arc<ir::Module>`'s pointer identity, so two
        // `Kernel` nodes that carry the same module clone share one JIT
        // build. `KernelModule::set_input`/`set_output`/`set_scratch` take
        // `&mut self`, hence the `RefCell` for the shared handle.
        enum PreExe {
            Kernel {
                name: String,
                module: Arc<Mutex<KernelModule>>,
                inputs: Vec<BufId>,
                outputs: Vec<BufId>,
                scratch: Option<BufId>,
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
        // unique modules to compile. Two `KernelModuleNode`s sharing an `Arc`
        // pointer alias the same source module, so they share a compiled
        // artifact. `unique_keys` holds those pointers in first-seen order so
        // parallel compilation can index into it.
        let mut unique_keys: Vec<*const ir::Module> = Vec::new();
        let mut unique_modules: Vec<Arc<ir::Module>> = Vec::new();
        let mut key_index: HashMap<*const ir::Module, usize> = HashMap::new();
        // Per-node kernel slot: `Some(kernel_idx)` if this graph node is a
        // Kernel, referring to `unique_modules[kernel_idx]`.
        let mut node_module_idx: Vec<Option<usize>> = Vec::with_capacity(nodes.len());
        for node in &nodes {
            match node {
                GraphNode::Kernel(k) => {
                    let key = Arc::as_ptr(&k.module);
                    let idx = *key_index.entry(key).or_insert_with(|| {
                        unique_keys.push(key);
                        unique_modules.push(k.module.clone());
                        unique_modules.len() - 1
                    });
                    node_module_idx.push(Some(idx));
                }
                _ => node_module_idx.push(None),
            }
        }
        drop(key_index);

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

        // Phase 1c: probe the on-disk cache, but only for representatives.
        // Cache hits are loaded on the main thread (cheap; just a dlopen).
        let mut compiled_slots: Vec<Option<(Arc<Mutex<KernelModule>>, usize)>> =
            (0..unique_modules.len()).map(|_| None).collect();
        let mut cache_misses: Vec<usize> = Vec::new();
        let mut num_cached_modules = 0usize;
        for &repr in &representatives {
            let module = &unique_modules[repr];
            let hit = match &self.kernel_cache {
                Some(cache) => cache.get(module)?,
                None => None,
            };
            match hit {
                Some(km) => {
                    let scratch_size = km.scratch_size();
                    compiled_slots[repr] = Some((Arc::new(Mutex::new(km)), scratch_size));
                    num_cached_modules += 1;
                }
                None => cache_misses.push(repr),
            }
        }

        // Phase 1d: JIT the misses in parallel — one compile per unique
        // content hash, not per unique Arc handle.
        let options = self.options.clone();
        let cache = self.kernel_cache.clone();
        let compile_results: Vec<Result<(usize, KernelModule), CompileError>> = {
            use rayon::prelude::*;
            cache_misses
                .par_iter()
                .map(|&idx| {
                    let module = &unique_modules[idx];
                    let km = compile_and_load(module, &options)?;
                    if let Some(c) = &cache {
                        // Best-effort — a failed insert shouldn't fail the compile.
                        let _ = c.insert(module, &km);
                    }
                    Ok((idx, km))
                })
                .collect()
        };
        for res in compile_results {
            let (idx, km) = res?;
            let scratch_size = km.scratch_size();
            compiled_slots[idx] = Some((Arc::new(Mutex::new(km)), scratch_size));
        }

        // Phase 1e: fan the representative's compiled slot out to every
        // aliased unique-module index. Two `Kernel` nodes with distinct
        // `Arc`s but identical hashes now share the same `Arc<Mutex<
        // KernelModule>>` — and each still gets its own scratch BufId
        // downstream, since the planner needs disjoint scratch lifetimes.
        for i in 0..unique_modules.len() {
            let repr = hash_repr[i];
            if repr != i {
                compiled_slots[i] = compiled_slots[repr].clone();
            }
        }

        // Phase 1d: assemble `PreExe`s in original node order, allocating a
        // per-node scratch BufId where the compiled module reports scratch.
        let mut compiled: Vec<PreExe> = Vec::with_capacity(nodes.len());
        for (node, mod_idx) in nodes.into_iter().zip(node_module_idx.into_iter()) {
            compiled.push(match node {
                GraphNode::Kernel(k) => {
                    let idx = mod_idx.expect("Kernel node without module index");
                    let (module_handle, scratch_size) = compiled_slots[idx]
                        .as_ref()
                        .expect("missing compiled slot")
                        .clone();
                    let name = k.module.name.clone();
                    let scratch = if scratch_size > 0 {
                        let bid = graph.add_buf(BufInfo {
                            name: Some(format!("scratch<{name}>")),
                            device_type: self.device,
                            size: Quast::cst(scratch_size as i64),
                            elem_size: 1,
                        });
                        Some(bid)
                    } else {
                        None
                    };
                    PreExe::Kernel {
                        name,
                        module: module_handle,
                        inputs: k.inputs,
                        outputs: k.outputs,
                        scratch,
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
        let plan = planner::plan_raw(
            &bufs,
            &accesses,
            &self.env,
            self.device,
            &exclude,
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
                    module,
                    inputs,
                    outputs,
                    scratch,
                } => {
                    {
                        let m = module.lock().unwrap();
                        check_kernel_sizes(&m, &inputs, &outputs, &sizes)?;
                        if let Some(sc) = scratch {
                            // Scratch was sized from `module.scratch_size()`
                            // so this should always hold, but we double-check
                            // against the (possibly foreign-device) size
                            // resolution here.
                            if sizes[sc.0] < m.scratch_size() {
                                return Err(CompileError::Runtime(format!(
                                    "kernel `{name}` scratch buffer sized {} bytes but module wants {}",
                                    sizes[sc.0],
                                    m.scratch_size()
                                )));
                            }
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
            nodes,
            input_bufs,
            output_bufs,
            device: self.device,
            bufs,
            num_unique_modules,
            num_cached_modules,
        })
    }
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
    /// Shared handle to the JIT'd module. Multiple `ExeKernel`s can point at
    /// the same underlying `KernelModule` when their source `Arc<ir::Module>`
    /// was shared at graph construction time; execution is sequential so
    /// re-binding inputs / outputs / scratch on the shared handle before
    /// each launch is safe.
    module: Arc<Mutex<KernelModule>>,
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
    /// Number of distinct `KernelModule`s that were compiled or loaded from
    /// cache. Kernels are deduplicated first by `Arc<ir::Module>` identity
    /// and then by [`crate::module_hash::module_hash`], so two `Kernel`
    /// nodes with structurally identical modules count once even when they
    /// carry distinct `Arc` handles.
    num_unique_modules: usize,
    /// Subset of `num_unique_modules` served from the on-disk kernel cache
    /// (i.e. reused a persisted `.so` instead of running nvcc).
    num_cached_modules: usize,
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

    /// Number of distinct compiled [`KernelModule`]s held by this exe.
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

    /// How many of [`Self::num_unique_modules`] were served from the on-disk
    /// [`crate::kernel_cache::KernelCache`] instead of re-running nvcc.
    pub fn num_cached_modules(&self) -> usize {
        self.num_cached_modules
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
                    // Re-binding inputs / outputs / scratch just before the
                    // launch is safe even when this handle is shared with
                    // another `ExeKernel`, because execution is sequential.
                    let mut m = k.module.lock().unwrap();
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
                    if let Some(sc) = k.scratch {
                        let ptr = bufid_ptr(sc)?;
                        let want = m.scratch_size();
                        let fake = ManuallyDrop::new(unsafe {
                            DeviceBuffer::<u8>::from_raw_parts(ptr, want)
                        });
                        m.set_scratch(&fake)?;
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
}

fn memcpy_err(e: openvm_cuda_common::error::MemCopyError) -> CompileError {
    CompileError::Runtime(format!("cudaMemcpy failed: {e:?}"))
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
        stream::GpuDeviceCtx,
    };

    use super::*;
    use crate::{
        graph_ir::{BufInfo, GraphBuilder},
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
        g.insert_kernel(module.clone(), [in0], [out0]);
        g.insert_kernel(module.clone(), [in1], [out1]);

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile_options(CompileOptions::default())
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
        let in_bufs: Vec<DeviceBuffer<u8>> = ins_order
            .iter()
            .map(|&b| {
                let bytes: Vec<u8> = host_for(b).iter().flat_map(|x| x.to_le_bytes()).collect();
                bytes.as_slice().to_device_on(&ctx).expect("H2D")
            })
            .collect();
        let mut out_bufs: Vec<DeviceBuffer<u8>> = (0..exe.num_outputs())
            .map(|i| DeviceBuffer::with_capacity_on(exe.output_size(i), &ctx))
            .collect();
        let mut scratch: DeviceBuffer<u8> =
            DeviceBuffer::with_capacity_on(exe.scratch_bytes().max(1), &ctx);

        exe.run(&ctx, &in_bufs, &mut out_bufs, &mut scratch)
            .expect("graph run");

        for (i, &out_bid) in outs_order.iter().enumerate() {
            let bytes: Vec<u8> = out_bufs[i].to_host_on(&ctx).expect("D2H");
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
    /// as two unique modules, and runs end-to-end: the reduce's scalar result
    /// flows to the consumer through a planner-managed intermediate buffer.
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
        g.insert_kernel(module, [a_buf], [out_buf]);

        // Split on insertion: one graph node per top-level kernel.
        assert_eq!(g.nodes.len(), 2);

        let mut exe = GraphCompiler::new()
            .device(DeviceType::Cuda(0))
            .compile_options(CompileOptions::default())
            .compile(g)
            .expect("graph compile");

        // Two distinct split kernels → two unique JIT builds. The scalar
        // intermediate has both a writer and a reader, so it lives in the
        // planner's scratch pool, not in the caller-facing I/O sets.
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
        let in_bufs = vec![in_bytes.as_slice().to_device_on(&ctx).expect("H2D")];
        let mut out_bufs = vec![DeviceBuffer::<u8>::with_capacity_on(
            exe.output_size(0),
            &ctx,
        )];
        let mut scratch: DeviceBuffer<u8> =
            DeviceBuffer::with_capacity_on(exe.scratch_bytes().max(1), &ctx);

        exe.run(&ctx, &in_bufs, &mut out_bufs, &mut scratch)
            .expect("graph run");

        let bytes: Vec<u8> = out_bufs[0].to_host_on(&ctx).expect("D2H");
        let got: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, want);
    }
}
