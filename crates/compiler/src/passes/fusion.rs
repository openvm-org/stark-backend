//! Kernel fusion over the buffer graph (see `crates/compiler/fusion-plan.md`).
//!
//! Stage 1: per-kernel access-relation extraction + cost estimates. Each
//! `GraphNode::Kernel` is canonicalized in isolation and summarized as an
//! [`AccessRelation`]: which module inputs it reads (and through which
//! quasi-affine index expressions), what it writes, a kernel class (how
//! fusable it is), and per-iteration-point cost estimates.
//!
//! Stage 2: candidate enumeration over internal single-writer buffers with
//! ≥ 1 kernel readers ([`enumerate_candidates`]) — one `(src, dst_i)`
//! candidate per reader —, case dispatch + substitution building
//! ([`dispatch`]), scoring against the plan section 7 cost model (with the
//! producer's write and launch costs amortized across the readers), and
//! greedy dst-disjoint selection ([`select_candidates`]) that lets a
//! producer participate in multiple fusions per round.
//!
//! Stage 3: fusion application ([`apply_fusion`]): the producer's result
//! expression is grafted into every consumer read site of the fused buffer
//! and the consumer graph node is replaced by the merged kernel.
//!
//! Stage 4: dead-node elimination ([`dce`]): drops graph nodes whose
//! writes can never be observed — in particular producers whose only
//! reader was fused away by stage 3.
//!
//! Stage 5: fixpoint driver ([`fuse_graph`]): DCE, then rounds of
//! extract / enumerate / select / apply until no profitable candidate
//! remains (or `max_iterations`), with a final DCE per round.
//!
//! Stage 6: post-fusion module deduplication ([`dedup_modules`]):
//! α-normalizes every kernel module ([`renumber_module`]) and folds
//! structurally identical ones onto a single `Arc<Module>` so each
//! distinct fused pattern JIT-compiles once.
//!
//! Cost convention: `KernelCost` quantities are *per outer iteration point*
//! (per value of the outer compute variable), not per output element. For
//! `Simple` kernels with scalar results the two coincide; for pack results
//! or a trailing inner compute the per-point cost folds in the pack width /
//! inner bound (matching `bytes_out_per_elem = bytes(elem) * pack_k` in the
//! plan).

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use sha3::{Digest, Sha3_256};

use crate::{
    graph_ir::{
        classify_buf_uses, BufId, FusionHistory, GraphBuilder, GraphNode, KernelModuleNode,
    },
    ir::{BinOp, IRBuilder, Module, Node, NodeId, ReduceOp, ScalarType, SizeExpr, VarId},
    module_hash::{children_of, module_hash},
    passes::{
        canonicalize::{canonicalize, CanonValue, Program, ResultExpr, TensorRef},
        parallel_reduce_rewrite::should_tree_lower,
        type_infer::{type_infer, TypeMap},
        utils::{hir_to_quast, hir_to_sexpr, replace_nodes, resolve_tensor_ref},
    },
    quast::{self, NodeEmitter, Quast, SymConst},
    CompileError,
};

/// Access summary of one canonical kernel: for all `i` with
/// `0 <= i < outer_bound`, the kernel reads the sets described by `reads`
/// and writes the buffers described by `writes`.
#[derive(Clone, Debug)]
pub struct AccessRelation {
    pub outer_var: VarId,
    pub outer_bound: usize,
    pub reads: Vec<ReadAccess>,
    pub writes: Vec<WriteAccess>,
    pub cost: KernelCost,
    pub class: KernelClass,
    /// Physical output index -> iteration point, when the write map is
    /// invertible (single output, no scatter). Required for case-B fusion
    /// (kernel as producer into a `Simple` consumer).
    pub write_inverse: Option<InverseMap>,
    /// Binder of the trailing inner compute, when present. Case-B
    /// substitutions map it alongside `outer_var`.
    pub inner_var: Option<VarId>,
    /// The kernel body is a plain scalar expression per iteration point: no
    /// tiles, no `#[grid]`, no inner `par`, no reduce or nested compute in
    /// the results. Producers must be scalar-body so their result can be
    /// grafted at consumer read sites.
    pub scalar_body: bool,
}

/// How fusable a kernel is (see plan section 3).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelClass {
    /// Flat elementwise compute: no inner compute, no tiles, no scatter,
    /// no `#[grid]`, no reduce anywhere in the body. Can act as producer
    /// or consumer in every supported fusion case.
    Simple,
    /// Structured single kernel (inner compute, tiles, scatter store,
    /// small sequential reduce, ...). Consumer in case A; producer in
    /// case B only via `write_inverse`.
    General,
    /// Expands to several CUDA kernels at JIT time (tree-lowered reduce).
    /// Only usable as a case-A consumer.
    MultiKernel,
}

/// Inverse of a kernel's write map: recovers the iteration point that wrote
/// physical flat index `f` of the output buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InverseMap {
    /// No scatter: `i = f / (tile * pack)`, `j = (f / pack) % tile`,
    /// `elem = f % pack` where `tile` is the trailing inner-compute bound
    /// and `pack` the pack width (1 when absent).
    DivMod { tile: usize, pack: usize },
}

/// One global-memory read site.
#[derive(Clone, Debug)]
pub struct ReadAccess {
    /// Which module input is read.
    pub input_pos: usize,
    /// The `Node::Index` site (hash-consed, so unique per index shape).
    pub site: NodeId,
    /// Linearized quasi-affine index expression over the outer variable and
    /// the binders in `inner`. `None` means the site is non-affine (e.g. a
    /// data-dependent gather): the relation is still usable, but fusion
    /// through this site is vetoed.
    pub expr: Option<Quast>,
    /// Binders in scope at the site with their bounds (trailing inner
    /// compute var, tile iteration var, reduce vars), outermost first.
    pub inner: Vec<(VarId, usize)>,
}

/// One output buffer write.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WriteAccess {
    pub out_pos: usize,
    /// Total elements written (`outer_bound * inner_factor`).
    pub len_elems: usize,
    /// Elements written per outer iteration point (inner bound x pack width).
    pub inner_factor: usize,
    /// Byte size of one element.
    pub elem_bytes: usize,
}

/// Per-outer-iteration-point cost estimates (see module docs for the
/// convention). Flops are in equivalent-u32-op units per the weight table in
/// plan section 7; address arithmetic inside index expressions is free.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KernelCost {
    pub flops_per_elem: f64,
    pub bytes_in_per_elem: f64,
    pub bytes_out_per_elem: f64,
    /// Raw count of reachable HIR nodes; guards code-size blowup.
    pub body_ops: usize,
}

/// `(module_hash, param_bindings)` -> extraction result, surviving fusion
/// rounds. The same symbolic module yields a different *concrete* relation
/// per binding vector, hence the compound key. `None` caches an extraction
/// failure so it is not retried every round.
pub type RelationCache = HashMap<([u8; 32], Vec<i64>), Option<Arc<AccessRelation>>>;

/// The node's parameter environment: module param `VarId` -> bound value.
fn binding_env(kn: &KernelModuleNode) -> BTreeMap<VarId, i64> {
    kn.module
        .builder
        .params()
        .iter()
        .zip(&kn.param_bindings)
        .map(|((v, _), &val)| (*v, val))
        .collect()
}

/// Extracts an [`AccessRelation`] for every `GraphNode::Kernel` in the
/// graph, keyed by node index. Kernels whose extraction fails (unsupported
/// constructs) are simply absent: they never participate in fusion.
pub fn extract_relations(
    g: &GraphBuilder,
    cache: &mut RelationCache,
) -> HashMap<usize, Arc<AccessRelation>> {
    let mut out = HashMap::new();
    for (idx, node) in g.nodes.iter().enumerate() {
        let GraphNode::Kernel(kn) = node else {
            continue;
        };
        let rel = cache
            .entry((module_hash(&kn.module), kn.param_bindings.clone()))
            .or_insert_with(|| extract_one(&kn.module, &binding_env(kn)).ok().map(Arc::new));
        if let Some(rel) = rel {
            out.insert(idx, Arc::clone(rel));
        }
    }
    out
}

/// Extracts the access relation of a split (single-kernel) module. The
/// relation is fully concrete: symbolic bounds and shapes are resolved
/// through `params` (the node's param bindings), so a symbolic module
/// participates in fusion with per-node concrete geometry while its HIR
/// stays symbolic.
pub fn extract_one(
    module: &Module,
    params: &BTreeMap<VarId, i64>,
) -> Result<AccessRelation, CompileError> {
    // `rewrite_parallel_reduce` runs on the raw module at JIT time (before
    // canonicalize), so multi-kernel detection must mirror its walk on the
    // raw body.
    let multi = has_tree_lowered_reduce(module, params);

    let types = type_infer(module)?;
    let program = canonicalize(module.clone(), types)?;
    if program.kernels.len() != 1 {
        return Err(CompileError::Lower(format!(
            "fusion: expected a split single-kernel module, `{}` has {} kernels",
            program.module.name,
            program.kernels.len()
        )));
    }
    let k = &program.kernels[0];
    let sym_err = || {
        CompileError::Lower(
            "fusion: kernel bound is not concrete under the node's param bindings".into(),
        )
    };
    let resolve = |e: &SizeExpr| e.concretize(params).as_const().map(|c| c as usize);
    let outer_bound = resolve(&k.outer_bound).ok_or_else(sym_err)?;
    let inner = match &k.inner {
        None => None,
        Some((mb, mv)) => Some((resolve(mb).ok_or_else(sym_err)?, *mv)),
    };
    let m = inner.map_or(1, |(mb, _)| mb);

    let mut w = Walker {
        b: &program.module.builder,
        types: &program.types,
        env: &program.env,
        params,
        inline_lets: &k.inline_lets,
        tile_vars: k.inner_lets.iter().map(|l| l.var).collect(),
        let_overlay: HashMap::new(),
        outer_var: k.outer_var,
        visited: HashSet::new(),
        scope: Vec::new(),
        addr_depth: 0,
        reads: Vec::new(),
        flops: 0.0,
        bytes_in: 0.0,
        saw_reduce: false,
        saw_compute: false,
        resolve_error: None,
    };

    // Shared-memory tiles run once per outer iteration point.
    for l in &k.inner_lets {
        let lb = resolve(&l.bound).ok_or_else(sym_err)?;
        w.scope.push((l.iter_var, lb));
        w.walk_result(&l.result, lb as f64);
        w.scope.pop();
    }
    // Result expressions run once per (outer, inner) point.
    if let Some((mb, mv)) = inner {
        w.scope.push((mv, mb));
    }
    for r in &k.results {
        w.walk_result(r, m as f64);
    }
    w.scope.clear();

    if let Some(e) = w.resolve_error.take() {
        return Err(e);
    }

    let mut writes = Vec::with_capacity(k.results.len());
    let mut bytes_out = 0.0;
    let mut pack_k = 1;
    for (j, r) in k.results.iter().enumerate() {
        pack_k = match r {
            ResultExpr::Scalar(_) => 1,
            ResultExpr::Pack(es) => es.len(),
        };
        let inner_factor = m * pack_k;
        let elem_bytes = k.member_types[j]
            .scalar_type()
            .map_or(0, |st| st.size_bytes());
        bytes_out += (inner_factor * elem_bytes) as f64;
        writes.push(WriteAccess {
            out_pos: j,
            len_elems: outer_bound * inner_factor,
            inner_factor,
            elem_bytes,
        });
    }

    let write_inverse =
        (k.results.len() == 1 && k.scatter_store.is_none()).then_some(InverseMap::DivMod {
            tile: m,
            pack: pack_k,
        });

    let scalar_body = k.inner_lets.is_empty()
        && k.threads.is_none()
        && k.inner_par.is_none()
        && !w.saw_reduce
        && !w.saw_compute;
    let class = if multi {
        KernelClass::MultiKernel
    } else if scalar_body && k.inner.is_none() && k.scatter_store.is_none() {
        KernelClass::Simple
    } else {
        KernelClass::General
    };

    Ok(AccessRelation {
        outer_var: k.outer_var,
        outer_bound,
        reads: w.reads,
        writes,
        cost: KernelCost {
            flops_per_elem: w.flops,
            bytes_in_per_elem: w.bytes_in,
            bytes_out_per_elem: bytes_out,
            body_ops: w.visited.len(),
        },
        class,
        write_inverse,
        inner_var: inner.map(|(_, v)| v),
        scalar_body,
    })
}

/// Whether `rewrite_parallel_reduce` would expand this module into several
/// CUDA kernels. Mirrors its top-level walk: bare reduces and
/// attribute-free `compute { reduce }` values trigger tree lowering per
/// [`should_tree_lower`]. Bounds resolve through `params`.
fn has_tree_lowered_reduce(module: &Module, params: &BTreeMap<VarId, i64>) -> bool {
    let b = &module.builder;
    let mut cur = module.body;
    loop {
        match b.node(cur) {
            Node::Let { value, body, .. } => {
                if top_value_tree_lowers(b, *value, params) {
                    return true;
                }
                cur = *body;
            }
            _ => return top_value_tree_lowers(b, cur, params),
        }
    }
}

fn top_value_tree_lowers(b: &IRBuilder, id: NodeId, params: &BTreeMap<VarId, i64>) -> bool {
    let resolve = |e: &SizeExpr| e.concretize(params).as_const();
    match b.node(id) {
        Node::Reduce { bound, .. } => {
            resolve(bound).is_some_and(|k| should_tree_lower(k as usize, 1))
        }
        Node::Compute {
            bound,
            body,
            scatter: None,
            par: None,
            threads: None,
            ..
        } => match (b.node(*body), resolve(bound)) {
            (Node::Reduce { bound: k, .. }, Some(m)) => {
                resolve(k).is_some_and(|k| should_tree_lower(k as usize, m as usize))
            }
            _ => false,
        },
        Node::Tuple(elems) => elems.iter().any(|&e| top_value_tree_lowers(b, e, params)),
        _ => false,
    }
}

fn scalar_op_weight(is_mul: bool, st: Option<ScalarType>) -> f64 {
    match (is_mul, st) {
        (false, Some(ScalarType::BabyBear)) => 2.0,
        (true, Some(ScalarType::BabyBear)) => 8.0,
        (false, Some(ScalarType::FpExt)) => 8.0,
        (true, Some(ScalarType::FpExt)) => 50.0,
        _ => 1.0,
    }
}

fn bin_weight(op: BinOp, st: Option<ScalarType>) -> f64 {
    match op {
        BinOp::Add | BinOp::Sub => scalar_op_weight(false, st),
        BinOp::Mul => scalar_op_weight(true, st),
        BinOp::Div | BinOp::Rem | BinOp::Lt | BinOp::Le | BinOp::Eq => 1.0,
    }
}

/// Single DFS over the canonical kernel body that simultaneously collects
/// [`ReadAccess`]es and accumulates the cost estimates. `scale` is the
/// number of executions of the current subtree per outer iteration point;
/// `Reduce` (and defensively `Compute`) multiply it by their bound.
///
/// The `visited` set is global across all roots: hash-consed shared
/// subexpressions are counted once (matching downstream CSE), with the
/// first-visited scope/scale winning — an accepted approximation.
struct Walker<'a> {
    b: &'a IRBuilder,
    types: &'a TypeMap,
    env: &'a HashMap<VarId, CanonValue>,
    /// Module param -> bound value (the node's `param_bindings`).
    params: &'a BTreeMap<VarId, i64>,
    inline_lets: &'a HashMap<VarId, NodeId>,
    /// Tile (`InnerLet`) variables: reads from these are shared memory,
    /// not global accesses.
    tile_vars: HashSet<VarId>,
    /// Scalar lets encountered mid-walk (canonical bodies are normally
    /// let-free, but reduce bodies may retain them).
    let_overlay: HashMap<VarId, NodeId>,
    outer_var: VarId,
    visited: HashSet<NodeId>,
    scope: Vec<(VarId, usize)>,
    /// Positive while walking index expressions: flops are suppressed
    /// (address arithmetic is free) but nested global loads still count.
    addr_depth: usize,
    reads: Vec<ReadAccess>,
    flops: f64,
    bytes_in: f64,
    saw_reduce: bool,
    saw_compute: bool,
    resolve_error: Option<CompileError>,
}

impl Walker<'_> {
    fn walk_result(&mut self, r: &ResultExpr, scale: f64) {
        match r {
            ResultExpr::Scalar(n) => self.walk(*n, scale),
            ResultExpr::Pack(es) => {
                for &e in es {
                    self.walk(e, scale);
                }
            }
        }
    }

    fn charge(&mut self, flops: f64) {
        if self.addr_depth == 0 {
            self.flops += flops;
        }
    }

    fn walk(&mut self, id: NodeId, scale: f64) {
        if !self.visited.insert(id) {
            return;
        }
        match self.b.node(id).clone() {
            Node::Input(_)
            | Node::ConstU32(_)
            | Node::ConstField(_)
            | Node::ConstFpExt(_)
            | Node::ConstSym(_) => {}
            Node::Var(v) => {
                if let Some(&n) = self.inline_lets.get(&v) {
                    self.walk(n, scale);
                } else if let Some(&n) = self.let_overlay.get(&v) {
                    self.walk(n, scale);
                } else if let Some(CanonValue::Scalar(n)) = self.env.get(&v) {
                    self.walk(*n, scale);
                }
            }
            Node::LiftFpExt(x) => {
                self.charge(scale);
                self.walk(x, scale);
            }
            Node::Bin(op, a, c) => {
                let st = self.types.try_get(id).and_then(|t| t.scalar_type());
                self.charge(scale * bin_weight(op, st));
                self.walk(a, scale);
                self.walk(c, scale);
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                self.charge(scale * 2.0);
                self.walk(cond, scale);
                self.walk(then_val, scale);
                self.walk(else_val, scale);
            }
            Node::Index { tensor, indices } => {
                self.addr_depth += 1;
                for &ix in &indices {
                    self.walk(ix, scale);
                }
                self.addr_depth -= 1;
                self.record_read(id, tensor, &indices, scale);
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                self.saw_reduce = true;
                let Some(bc) = bound.concretize(self.params).as_const().map(|c| c as usize) else {
                    self.resolve_error = Some(CompileError::Lower(
                        "fusion: kernel bound is not concrete under the node's param bindings"
                            .into(),
                    ));
                    return;
                };
                let st = self.types.try_get(id).and_then(|t| t.scalar_type());
                self.charge(scale * bc as f64 * scalar_op_weight(op == ReduceOp::Mul, st));
                self.scope.push((var, bc));
                self.walk(body, scale * bc as f64);
                self.scope.pop();
            }
            Node::Compute {
                bound, var, body, ..
            } => {
                // Canonical result exprs contain no nested computes (they
                // are peeled into `inner` / `inner_lets`); handled
                // defensively so the relation still carries a conservative
                // cost and the kernel is kept out of the Simple class.
                self.saw_compute = true;
                let Some(bc) = bound.concretize(self.params).as_const().map(|c| c as usize) else {
                    self.resolve_error = Some(CompileError::Lower(
                        "fusion: kernel bound is not concrete under the node's param bindings"
                            .into(),
                    ));
                    return;
                };
                self.scope.push((var, bc));
                self.walk(body, scale * bc as f64);
                self.scope.pop();
            }
            Node::Let { var, value, body } => {
                self.let_overlay.insert(var, value);
                self.walk(value, scale);
                self.walk(body, scale);
            }
            Node::Tuple(es) | Node::Pack(es) => {
                for &e in &es {
                    self.walk(e, scale);
                }
            }
            Node::Proj(t, _) => self.walk(t, scale),
        }
    }

    fn record_read(&mut self, site: NodeId, tensor: NodeId, indices: &[NodeId], scale: f64) {
        if let Node::Var(v) = self.b.node(tensor) {
            if self.tile_vars.contains(v) {
                return;
            }
        }
        let input_pos = match resolve_tensor_ref(self.b, self.env, tensor) {
            Ok(TensorRef::Input(p)) => p,
            Ok(TensorRef::Let { .. }) => {
                self.resolve_error = Some(CompileError::Lower(
                    "fusion: split single-kernel module reads a top-level let output".into(),
                ));
                return;
            }
            Err(e) => {
                self.resolve_error = Some(e);
                return;
            }
        };
        let decl = &self.b.inputs()[input_pos];
        self.bytes_in += scale * decl.elem.size_bytes() as f64;
        let expr = self.site_quast(indices, &decl.shape);
        self.reads.push(ReadAccess {
            input_pos,
            site,
            expr,
            inner: self.scope.clone(),
        });
    }

    fn site_quast(&self, indices: &[NodeId], shape: &[SizeExpr]) -> Option<Quast> {
        // Shape dims and index expressions resolve through the node's param
        // bindings; a dim that stays symbolic means the linearized site
        // expression is unknown and the read is recorded without one
        // (conservative).
        let shape: Vec<usize> = shape
            .iter()
            .map(|d| d.concretize(self.params).as_const().map(|c| c as usize))
            .collect::<Option<_>>()?;
        let in_scope = |v: VarId| v == self.outer_var || self.scope.iter().any(|&(sv, _)| sv == v);
        let q_syms = |v: VarId| in_scope(v).then(|| Quast::sym(v));
        let s_syms = |v: VarId| in_scope(v).then(|| SizeExpr::sym(v));
        let lets = |v: VarId| {
            self.inline_lets
                .get(&v)
                .or_else(|| self.let_overlay.get(&v))
                .copied()
                .or_else(|| match self.env.get(&v) {
                    Some(CanonValue::Scalar(n)) => Some(*n),
                    _ => None,
                })
        };
        let exprs: Option<Vec<Quast>> = indices
            .iter()
            .map(|&ix| {
                // Param-bearing sites (`Node::ConstSym`) are not quasi-affine
                // as written; extract symbolically and concretize instead.
                hir_to_quast(self.b, ix, &q_syms, &lets).ok().or_else(|| {
                    hir_to_sexpr(self.b, ix, &s_syms, &lets)
                        .ok()?
                        .concretize(self.params)
                        .try_to_quast()
                })
            })
            .collect();
        exprs.map(|e| quast::linearize(&e, &shape))
    }
}

// ---------------------------------------------------------------------------
// Stage 2: candidate enumeration, case dispatch, scoring, greedy selection
// (plan sections 3, 4 and 7).
// ---------------------------------------------------------------------------

/// Tunables for the fusion pass (plan section 10).
#[derive(Clone, Debug)]
pub struct FusionOptions {
    /// Maximum fusion rounds of the stage-5 driver.
    pub max_iterations: usize,
    /// Reject pairs whose combined reachable-HIR-node count exceeds this
    /// (guards code-size / register-pressure blowup).
    pub max_body_ops: usize,
    /// Fixed cost of one kernel launch, in seconds.
    pub launch_cost: f64,
    /// Effective global-memory bandwidth, bytes per second.
    pub bw_eff: f64,
    /// Effective compute throughput, u32-equivalent ops per second.
    pub flop_eff: f64,
    /// Discount on re-read producer inputs (L2 catches part of the extra
    /// traffic when consumer sites overlap).
    pub gamma: f64,
    /// Effective per-thread memory throughput, bytes per second. Used to
    /// price the *sequential* memory ops added inside a consumer thread by
    /// inlining a producer with `k` loads per output at each of `m`
    /// consumer read sites — this cost cannot be amortized across the
    /// grid because it's on the consumer thread's critical path. Roughly
    /// one order of magnitude below [`Self::bw_eff`] to reflect the
    /// limited memory-level parallelism available inside one warp/thread.
    /// Default 1e9 (~1 GB/s per thread, ~3 orders of magnitude below
    /// aggregate DRAM BW), tuned so that folding a `k=4`-load producer
    /// into a consumer site under a `reduce[16]` loop tips the cost above
    /// the launch-cost savings — matches the observed catastrophic
    /// fold-into-reduce pattern.
    pub per_thread_bw: f64,
    /// When true, [`fuse_graph`] prints one line per round with the round
    /// stats (also recorded on [`FusionReport::rounds_detail`]).
    pub verbose: bool,
}

impl Default for FusionOptions {
    fn default() -> Self {
        FusionOptions {
            max_iterations: 10,
            max_body_ops: 1024,
            launch_cost: 5e-6,
            bw_eff: 2e12,
            flop_eff: 1e13,
            gamma: 0.5,
            per_thread_bw: 1e9,
            verbose: false,
        }
    }
}

/// A profitable src -> dst fusion through internal buffer `buf`.
#[derive(Clone, Debug)]
pub struct FusionCandidate {
    /// Producer graph-node index.
    pub src: usize,
    /// Consumer graph-node index.
    pub dst: usize,
    /// The intermediate buffer the fusion eliminates.
    pub buf: BufId,
    /// Estimated net saving in seconds; only positive scores are emitted.
    pub score: f64,
    pub info: FusionInfo,
}

#[derive(Clone, Debug)]
pub struct FusionInfo {
    /// One rewrite per consumer read site of the fused buffer.
    pub rewrites: Vec<SiteRewrite>,
}

/// Stage-3 instruction: replace the consumer `Node::Index` at `index_node`
/// with the producer's (α-renamed) result expression under `sigma`.
#[derive(Clone, Debug)]
pub struct SiteRewrite {
    pub index_node: NodeId,
    /// Producer binder (outer var, and trailing inner var for case B) ->
    /// quasi-affine expression over the consumer's binders.
    pub sigma: BTreeMap<VarId, Quast>,
    /// For pack producers: which pack member this site selects.
    pub pack_elem: Option<usize>,
    /// Bounds of the consumer binders that may appear in `sigma`'s
    /// expressions (the consumer's outer var plus the site's `inner`
    /// binders); required by [`Quast::emit`] when the substitution is
    /// materialized as HIR.
    pub bounds: BTreeMap<VarId, u64>,
}

/// Enumerates fusion candidates (plan sections 3 and 7): for every
/// non-interface buffer with a unique kernel writer, emits one candidate
/// per kernel reader that survives the fusion preconditions (WAR safety,
/// no output aliasing, body-ops guard, case A/B dispatch). Each pair is
/// scored under the amortized cost model (see [`score_candidate`]) with
/// the total reader count as an input. Candidates may share graph nodes;
/// [`select_candidates`] picks a dst-disjoint subset.
pub fn enumerate_candidates(
    g: &GraphBuilder,
    rels: &HashMap<usize, Arc<AccessRelation>>,
    opts: &FusionOptions,
) -> Vec<FusionCandidate> {
    let (writers, readers) = classify_buf_uses(&g.nodes, g.bufs.len());
    let mut out = Vec::new();
    let diag = std::env::var_os("FUSION_DIAG").is_some();
    let log = |src_name: &str, dst_name: &str, reason: &str| {
        if diag {
            eprintln!("[fusion-diag] reject {src_name} -> {dst_name}: {reason}");
        }
    };
    for bi in 0..g.bufs.len() {
        let buf = BufId(bi);
        if g.buf_is_interface(buf) {
            continue;
        }
        let &[src] = writers[bi].as_slice() else {
            continue;
        };
        let dsts = &readers[bi];
        if dsts.is_empty() {
            continue;
        }
        let n_readers = dsts.len();
        let src_name = match &g.nodes[src] {
            GraphNode::Kernel(k) => k.module.name.clone(),
            _ => String::from("<non-kernel>"),
        };
        let Some(src_rel) = rels.get(&src) else {
            continue;
        };
        let GraphNode::Kernel(kn_src) = &g.nodes[src] else {
            continue;
        };
        for &dst in dsts {
            let dst_name = match &g.nodes[dst] {
                GraphNode::Kernel(k) => k.module.name.clone(),
                GraphNode::Memcpy(_) => "<memcpy>".to_string(),
                _ => "<non-kernel>".to_string(),
            };
            if src >= dst {
                log(&src_name, &dst_name, "topo: src >= dst");
                continue;
            }
            let Some(dst_rel) = rels.get(&dst) else {
                log(
                    &src_name,
                    &dst_name,
                    "no dst_rel (extract failed or non-kernel)",
                );
                continue;
            };
            let GraphNode::Kernel(kn_dst) = &g.nodes[dst] else {
                log(&src_name, &dst_name, "dst is not a Kernel node");
                continue;
            };
            if kn_dst.outputs.iter().any(|ob| kn_src.inputs.contains(ob)) {
                log(
                    &src_name,
                    &dst_name,
                    "output aliasing (dst writes a src input)",
                );
                continue;
            }
            if kn_src
                .inputs
                .iter()
                .any(|ib| writers[ib.0].iter().any(|&wn| wn > src && wn < dst))
            {
                log(
                    &src_name,
                    &dst_name,
                    "WAR: writer of a src input between src and dst",
                );
                continue;
            }
            if src_rel.cost.body_ops + dst_rel.cost.body_ops > opts.max_body_ops {
                log(
                    &src_name,
                    &dst_name,
                    &format!(
                        "body_ops guard: {} + {} > {}",
                        src_rel.cost.body_ops, dst_rel.cost.body_ops, opts.max_body_ops
                    ),
                );
                continue;
            }
            let sites: Vec<&ReadAccess> = dst_rel
                .reads
                .iter()
                .filter(|ra| kn_dst.inputs[ra.input_pos] == buf)
                .collect();
            if sites.is_empty() {
                log(&src_name, &dst_name, "no read sites for this buffer");
                continue;
            }
            let Some(info) = dispatch(src_rel, dst_rel, &sites) else {
                log(
                    &src_name,
                    &dst_name,
                    &format!(
                        "dispatch: src_class={:?}, dst_class={:?}, scalar_body={}, \
                         write_inverse={:?}, sites={} (some may be non-affine or \
                         non-const pack members)",
                        src_rel.class,
                        dst_rel.class,
                        src_rel.scalar_body,
                        src_rel.write_inverse,
                        sites.len(),
                    ),
                );
                continue;
            };
            let score = score_candidate(src_rel, dst_rel, &sites, n_readers, opts);
            if score <= 0.0 {
                let evals_after: f64 = sites
                    .iter()
                    .map(|ra| {
                        dst_rel.outer_bound as f64
                            * ra.inner.iter().map(|&(_, b)| b as f64).product::<f64>()
                    })
                    .sum();
                log(
                    &src_name,
                    &dst_name,
                    &format!(
                        "score={score:.3e} <= 0 (b_len={}, evals_after={}, \
                         n_readers={n_readers}, flops/elem={:.1}, bytes_in/elem={:.1})",
                        src_rel.writes[0].len_elems,
                        evals_after,
                        src_rel.cost.flops_per_elem,
                        src_rel.cost.bytes_in_per_elem,
                    ),
                );
                continue;
            }
            out.push(FusionCandidate {
                src,
                dst,
                buf,
                score,
                info,
            });
        }
    }
    out
}

/// Case dispatch (plan section 4): `(Simple, _)` fuses via case A (map the
/// producer's outer var through the site expression); `(General, Simple)`
/// fuses via case B (invert the producer's write map with
/// [`InverseMap::DivMod`]). Returns `None` when the pair is not fusable
/// under the v1 restrictions.
fn dispatch(
    src: &AccessRelation,
    dst: &AccessRelation,
    sites: &[&ReadAccess],
) -> Option<FusionInfo> {
    if !src.scalar_body {
        return None;
    }
    let InverseMap::DivMod { tile, pack } = src.write_inverse?;
    let case_b_inner = match (src.class, dst.class) {
        (KernelClass::Simple, _) => None,
        (KernelClass::General, KernelClass::Simple) => Some(src.inner_var?),
        _ => return None,
    };
    let mut rewrites = Vec::with_capacity(sites.len());
    for ra in sites {
        let expr = ra.expr.as_ref()?;
        let (q, r) = split_div_rem(expr, pack as i64)?;
        let mut sigma = BTreeMap::new();
        match case_b_inner {
            None => {
                sigma.insert(src.outer_var, q);
            }
            Some(iv) => {
                sigma.insert(src.outer_var, q.floordiv(tile as i64));
                sigma.insert(iv, q.rem_c(tile as i64));
            }
        }
        let mut bounds = BTreeMap::new();
        bounds.insert(dst.outer_var, dst.outer_bound as u64);
        for &(v, b) in &ra.inner {
            bounds.insert(v, b as u64);
        }
        rewrites.push(SiteRewrite {
            index_node: ra.site,
            sigma,
            pack_elem: (pack > 1).then_some(r as usize),
            bounds,
        });
    }
    Some(FusionInfo { rewrites })
}

/// Cost model (plan section 7), in estimated seconds saved. Producer
/// per-outer-point costs are normalized to per-element via `inner_factor`.
///
/// `n_readers` is the total number of kernel readers of the fused buffer
/// in the current graph, from which this pair fuses one. The producer's
/// write bytes and launch overhead are shared by all `n_readers` — this
/// pair is attributed a `1/n_readers` slice of them, so summing the
/// per-pair scores over all readers reconstructs the whole-fusion savings
/// (when the last reader fuses and DCE removes the producer). With
/// `n_readers = 1` the formula collapses to the sole-reader case.
///
/// Beyond the aggregate bandwidth/flop terms (which assume perfect grid
/// parallelism and coalesced access), an explicit *sequential-read*
/// penalty is subtracted: after fusion, each of the `m = sites.len()`
/// consumer read sites of the fused buffer is inlined into the producer's
/// full body, which reads `bytes_in_per_elem` bytes per graft. That is
/// `m * bytes_in_per_elem` bytes newly on the consumer thread's critical
/// path per outer iteration, at each site multiplied by its per-thread
/// iteration volume (inner-compute / reduce binders in scope at the
/// site). Priced at [`FusionOptions::per_thread_bw`] rather than the
/// aggregate `bw_eff`, this captures the "trading parallel producer
/// launches for serial consumer-thread work" cost that plain bandwidth
/// accounting misses — the specific failure mode for chains like
/// `fold_ef_frac_columns_dsl_* → frac_compute_round_dsl_*`.
fn score_candidate(
    src: &AccessRelation,
    dst: &AccessRelation,
    sites: &[&ReadAccess],
    n_readers: usize,
    opts: &FusionOptions,
) -> f64 {
    let wa = &src.writes[0];
    let b_len = wa.len_elems as f64;
    // Producer evaluations after fusion: one per dynamic execution of each
    // consumer read site.
    let evals_after: f64 = sites
        .iter()
        .map(|ra| dst.outer_bound as f64 * ra.inner.iter().map(|&(_, b)| b as f64).product::<f64>())
        .sum();
    let share = 1.0 / n_readers as f64;
    let b_share = b_len * share;
    let excess = evals_after - b_share;
    let per_elem = 1.0 / wa.inner_factor as f64;
    // This reader's B-loads always die on fusion; the producer's B-store
    // and its launch only die when DCE catches up (i.e. once all readers
    // have fused), so both are amortized over `n_readers`.
    let mem_saved = (b_share + evals_after) * wa.elem_bytes as f64;
    let mem_added = opts.gamma * excess * src.cost.bytes_in_per_elem * per_elem;
    let extra_flops = excess * src.cost.flops_per_elem * per_elem;

    // Sequential-read penalty. Per consumer read site of `buf`, fusion
    // replaces one `elem_bytes` B-load with a full producer-body evaluation
    // that loads `bytes_in_per_elem` bytes; the delta stays on the consumer
    // thread's critical path. Multiply by the site's per-thread iteration
    // volume (`ra.inner`'s bounds) — the outer var is grid-parallelized so
    // it does not contribute.
    let elem_bytes = wa.elem_bytes as f64;
    let extra_seq_bytes_per_thread: f64 = sites
        .iter()
        .map(|ra| {
            let iters_per_thread: f64 = ra.inner.iter().map(|&(_, b)| b as f64).product();
            (src.cost.bytes_in_per_elem - elem_bytes).max(0.0) * iters_per_thread
        })
        .sum();
    let seq_read_cost = extra_seq_bytes_per_thread / opts.per_thread_bw;

    (mem_saved - mem_added) / opts.bw_eff + share * opts.launch_cost
        - extra_flops / opts.flop_eff
        - seq_read_cost
}

/// Greedy dst-disjoint selection: by descending score (ties broken on
/// `(src, dst)` for determinism), keep a candidate iff its `dst` has not
/// been picked and neither endpoint conflicts with an earlier pick's role.
///
/// A producer node may participate in several picks in one round (each
/// pick rewrites only its `dst`), but a node picked as a producer must
/// not simultaneously be picked as a consumer (or vice versa) — that
/// would apply one fusion against a module the other fusion had just
/// rewritten and invalidate its precomputed rewrites.
pub fn select_candidates(mut cands: Vec<FusionCandidate>) -> Vec<FusionCandidate> {
    cands.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| (a.src, a.dst).cmp(&(b.src, b.dst)))
    });
    let mut used_dst = HashSet::new();
    let mut used_src = HashSet::new();
    cands.retain(|c| {
        if used_dst.contains(&c.dst) || used_dst.contains(&c.src) || used_src.contains(&c.dst) {
            return false;
        }
        used_dst.insert(c.dst);
        used_src.insert(c.src);
        true
    });
    cands
}

/// Structurally splits `q` into `(d, r)` such that `q = k*d + r` with
/// `0 <= r < k` and `r` provably constant. Returns `None` when the
/// remainder cannot be proven constant (e.g. `i % 2` for symbolic `i`).
/// Resolves which pack member a consumer site selects (case A) and peels
/// the pack part off case-B inverses.
fn split_div_rem(q: &Quast, k: i64) -> Option<(Quast, i64)> {
    debug_assert!(k >= 1);
    if k == 1 {
        return Some((q.clone(), 0));
    }
    match q {
        Quast::Const(c) => Some((Quast::cst(c.div_euclid(k)), c.rem_euclid(k))),
        Quast::Sym(_) | Quast::FloorDiv(..) => None,
        Quast::Add(a, b) => {
            let (da, ra) = split_div_rem(a, k)?;
            let (db, rb) = split_div_rem(b, k)?;
            let r = ra + rb;
            Some((add_cst(q_add(da, db), r.div_euclid(k)), r.rem_euclid(k)))
        }
        Quast::Mul(a, m) => {
            if m % k == 0 {
                Some((q_mul((**a).clone(), m / k), 0))
            } else {
                let (da, ra) = split_div_rem(a, k)?;
                let r = m * ra;
                Some((add_cst(q_mul(da, *m), r.div_euclid(k)), r.rem_euclid(k)))
            }
        }
        Quast::Neg(a) => {
            let (da, ra) = split_div_rem(a, k)?;
            if ra == 0 {
                Some((q_neg(da), 0))
            } else {
                // -(k*d + r) = k*(-d - 1) + (k - r) for 0 < r < k.
                Some((add_cst(q_neg(da), -1), k - ra))
            }
        }
    }
}

fn q_add(a: Quast, b: Quast) -> Quast {
    if a == Quast::Const(0) {
        b
    } else if b == Quast::Const(0) {
        a
    } else {
        a.add(&b)
    }
}

fn add_cst(a: Quast, c: i64) -> Quast {
    if c == 0 {
        a
    } else {
        q_add(a, Quast::cst(c))
    }
}

fn q_mul(a: Quast, m: i64) -> Quast {
    match (&a, m) {
        (_, 0) | (Quast::Const(0), _) => Quast::cst(0),
        (_, 1) => a,
        _ => a.mul_c(m),
    }
}

fn q_neg(a: Quast) -> Quast {
    if a == Quast::Const(0) {
        a
    } else {
        a.neg()
    }
}

// ---------------------------------------------------------------------------
// Stage 3: fusion application (plan section 8).
// ---------------------------------------------------------------------------

fn ferr(msg: impl std::fmt::Display) -> CompileError {
    CompileError::Lower(format!("fusion: {msg}"))
}

/// Counts reachable HIR nodes from `m.body`. Same shape as the walker used
/// during access-relation extraction, so it's a stand-in for "kernel size"
/// when choosing whose base name wins in [`fused_kernel_name`].
fn reachable_node_count(m: &Module) -> usize {
    let mut seen = HashSet::new();
    let mut stack = vec![m.body];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        stack.extend(children_of(m.builder.node(id)));
    }
    seen.len()
}

/// Strips a `fused_<base>_<8-hex-chars>` decoration off a kernel name, if
/// present, so cascading fusions don't stack the `fused_` prefix nor
/// concatenate old hashes. Anything not matching that shape is returned
/// as-is.
fn strip_fused_decoration(name: &str) -> &str {
    let Some(rest) = name.strip_prefix("fused_") else {
        return name;
    };
    let bytes = rest.as_bytes();
    if bytes.len() < 10 || bytes[bytes.len() - 9] != b'_' {
        return name;
    }
    let tail = &rest[rest.len() - 8..];
    if tail.bytes().all(|b| b.is_ascii_hexdigit()) {
        &rest[..rest.len() - 9]
    } else {
        name
    }
}

/// Chooses the name of a merged kernel from the two operands' modules and
/// the fusion histories of both. The base is whichever operand has more
/// reachable HIR nodes (ties go to the consumer); the suffix is 8 hex
/// digits of SHA3-256 over the *sorted* union of leaf original names in
/// the two histories, so re-ordered fusions of the same leaf set collapse
/// to the same name.
fn fused_kernel_name(
    consumer_module: &Module,
    producer_module: &Module,
    consumer_history: &FusionHistory,
    producer_history: &FusionHistory,
) -> String {
    let c_ops = reachable_node_count(consumer_module);
    let p_ops = reachable_node_count(producer_module);
    let base = if c_ops >= p_ops {
        strip_fused_decoration(&consumer_module.name)
    } else {
        strip_fused_decoration(&producer_module.name)
    };
    let mut names = consumer_history.leaf_names();
    names.extend(producer_history.leaf_names());
    names.sort();
    let mut hasher = Sha3_256::new();
    for n in &names {
        hasher.update((n.len() as u64).to_le_bytes());
        hasher.update(n.as_bytes());
    }
    let digest = hasher.finalize();
    format!("fused_{base}_{}", hex::encode(&digest[..4]))
}

/// Applies a selected candidate to the graph: grafts the producer's result
/// expression into every consumer read site of the fused buffer and replaces
/// the consumer graph node with the merged kernel. The producer node is left
/// untouched; stage-4 DCE removes it once its outputs are unread.
///
/// The merged module is rebuilt from the consumer's *canonical kernel view*
/// (result expressions with peeled scalar lets re-inlined, re-wrapped in the
/// inner/outer computes) rather than patched into the raw module body: the
/// site `NodeId`s recorded in stage 1 live in canonicalization-rewritten
/// nodes (peeled lets, absorbed nests) that the raw body may not contain.
///
/// Fallible by design: on any error the graph is left unchanged and the
/// caller simply skips the candidate.
pub fn apply_fusion(g: &mut GraphBuilder, cand: &FusionCandidate) -> Result<(), CompileError> {
    let (GraphNode::Kernel(kn_src), GraphNode::Kernel(kn_dst)) =
        (&g.nodes[cand.src], &g.nodes[cand.dst])
    else {
        return Err(ferr("candidate endpoints are not kernel nodes"));
    };
    let src_module = Arc::clone(&kn_src.module);
    let dst_module = Arc::clone(&kn_dst.module);
    let src_bufs = kn_src.inputs.clone();
    let dst_bufs = kn_dst.inputs.clone();
    let dst_outputs = kn_dst.outputs.clone();
    let src_param_bindings = kn_src.param_bindings.clone();
    let dst_param_bindings = kn_dst.param_bindings.clone();
    // Capture the pre-merge history now — after the graft below the src
    // BufIds may no longer resolve to live buffer-table entries, so
    // shape-string snapshots must happen against the current graph.
    let consumer_history = kn_dst.history_or_leaf(g);
    let producer_history = kn_src.history_or_leaf(g);

    // Re-canonicalize both modules. Canonicalization is deterministic, so
    // the NodeIds/VarIds recorded in `cand.info` (extracted from an
    // identical canonicalization in stage 1) address the same nodes here;
    // the residual-read check below backstops any mismatch.
    let src_prog = canonicalize((*src_module).clone(), type_infer(&src_module)?)?;
    let dst_prog = canonicalize((*dst_module).clone(), type_infer(&dst_module)?)?;
    if src_prog.kernels.len() != 1 || dst_prog.kernels.len() != 1 {
        return Err(ferr("fusion endpoints must be split single-kernel modules"));
    }
    let ks = &src_prog.kernels[0];
    if ks.results.len() != 1 {
        return Err(ferr("producer must have a single output"));
    }

    let Program {
        module: dst_mod,
        types: mut scratch_types,
        kernels: dst_kernels,
        env: dst_env,
        ..
    } = dst_prog;
    let kd = &dst_kernels[0];
    // v1 surgery restrictions (plan section 12): the consumer view must be
    // rebuildable from `results` + `inner` alone.
    if !kd.inner_lets.is_empty() {
        return Err(ferr(
            "consumer with shared-memory tiles is not fusable in v1",
        ));
    }
    if kd.scatter_store.is_some() {
        return Err(ferr("consumer with a scatter store is not fusable in v1"));
    }
    let Module {
        builder: mut mb, ..
    } = dst_mod;

    // Consumer input positions bound to the fused buffer: needed for seam
    // unification here and for the residual-read check after grafting.
    let removed: Vec<usize> = dst_bufs
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b == cand.buf)
        .map(|(p, _)| p)
        .collect();
    if removed.is_empty() {
        return Err(ferr("fused buffer is not a consumer input"));
    }

    // Seam-based param unification (structural, not value-based): a flat
    // scalar-body pack-1 producer writes exactly `outer_bound` elements,
    // and the consumer states that same element count as the removed
    // input's shape — both sides were checked against the buffer at
    // insertion, so the two expressions agree under their bindings. When
    // the producer bound is a bare param it is remapped onto the
    // consumer-side product expression; every other producer param is
    // appended to the merged module on first use (its bound value travels
    // with the merged bindings, so the kernel stays correct either way —
    // un-unified params only cost dedup opportunities).
    let mut param_remap: BTreeMap<VarId, SizeExpr> = BTreeMap::new();
    if ks.inner.is_none() && matches!(ks.results[0], ResultExpr::Scalar(_)) {
        if let SizeExpr::Const(SymConst::Sym(p)) = ks.outer_bound.fold_lits() {
            if let Some(prod) = SizeExpr::product(&mb.inputs()[removed[0]].shape) {
                param_remap.insert(p, prod);
            }
        }
    }

    // Phase 1: prepare one grafted producer expression per read site.
    let mut cx = GraftCx {
        src_b: &src_prog.module.builder,
        src_env: &src_prog.env,
        src_inline_lets: &ks.inline_lets,
        src_bufs: &src_bufs,
        dst_bufs: &dst_bufs,
        src_param_bindings: &src_param_bindings,
        param_remap,
        appended_params: Vec::new(),
        input_map: HashMap::new(),
        appended: Vec::new(),
    };
    let mut grafts: HashMap<NodeId, NodeId> = HashMap::new();
    for rw in &cand.info.rewrites {
        let site_env: BTreeMap<VarId, NodeId> = rw
            .bounds
            .keys()
            .map(|&v| (v, mb.intern(Node::Var(v))))
            .collect();
        let mut subst = BTreeMap::new();
        let mut em = NodeEmitter {
            b: &mut mb,
            types: &mut scratch_types,
            env: &site_env,
        };
        for (&sv, q) in &rw.sigma {
            subst.insert(sv, q.emit(&rw.bounds, &mut em)?);
        }
        let member = match (&ks.results[0], rw.pack_elem) {
            (ResultExpr::Scalar(n), None) => *n,
            (ResultExpr::Pack(es), Some(e)) => *es
                .get(e)
                .ok_or_else(|| ferr(format!("pack member {e} out of bounds")))?,
            (ResultExpr::Pack(es), None) if es.len() == 1 => es[0],
            _ => return Err(ferr("producer result does not match the pack selection")),
        };
        let grafted = cx.copy(
            &mut mb,
            member,
            &subst,
            &mut HashMap::new(),
            &mut HashMap::new(),
        )?;
        grafts.insert(rw.index_node, grafted);
    }
    let appended = cx.appended;
    // Merged bindings: consumer params keep their positions (mb is the
    // consumer's builder), producer params appended by the graft follow in
    // declaration order.
    let merged_bindings: Vec<i64> = dst_param_bindings
        .into_iter()
        .chain(cx.appended_params)
        .collect();

    // Phase 2: rebuild the consumer body from the canonical view with the
    // grafts applied and every peeled scalar let inlined back.
    let mut memo = HashMap::new();
    let mut member_nodes = Vec::with_capacity(kd.results.len());
    for r in &kd.results {
        let n = match r {
            ResultExpr::Scalar(n) => {
                rewrite_body(&mut mb, *n, &grafts, &kd.inline_lets, &dst_env, &mut memo)
            }
            ResultExpr::Pack(es) => {
                let es2: Vec<NodeId> = es
                    .iter()
                    .map(|&e| {
                        rewrite_body(&mut mb, e, &grafts, &kd.inline_lets, &dst_env, &mut memo)
                    })
                    .collect();
                mb.intern(Node::Pack(es2))
            }
        };
        member_nodes.push(n);
    }
    let mut root = if member_nodes.len() == 1 {
        member_nodes[0]
    } else {
        mb.intern(Node::Tuple(member_nodes))
    };
    if let Some((m, j)) = &kd.inner {
        root = mb.intern(Node::Compute {
            bound: m.clone(),
            var: *j,
            body: root,
            scatter: None,
            par: kd.inner_par.clone().map(Box::new),
            threads: None,
        });
    }
    let body = mb.intern(Node::Compute {
        bound: kd.outer_bound.clone(),
        var: kd.outer_var,
        body: root,
        scatter: None,
        par: None,
        threads: kd.threads,
    });

    // Residual-read check: no reference to the fused buffer's input
    // positions may survive grafting (a missed site would keep the data
    // dependency while DCE later removes the producer).
    let mut stack = vec![body];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Node::Input(p) = mb.node(id) {
            if removed.contains(p) {
                return Err(ferr("a read of the fused buffer survived grafting"));
            }
        }
        stack.extend(children_of(mb.node(id)));
    }

    // Drop the fused input declarations, renumbering the survivors with one
    // simultaneous replacement (map images are used verbatim, so shifted
    // positions cannot cascade).
    let n_inputs = mb.inputs().len();
    let mut renumber = HashMap::new();
    let mut new_pos = 0usize;
    for old in 0..n_inputs {
        if removed.contains(&old) {
            continue;
        }
        if old != new_pos {
            let from = mb.intern(Node::Input(old));
            let to = mb.intern(Node::Input(new_pos));
            renumber.insert(from, to);
        }
        new_pos += 1;
    }
    let body = replace_nodes(&mut mb, body, &renumber);
    for &p in removed.iter().rev() {
        mb.remove_input_decl(p);
    }

    let merged_name = fused_kernel_name(
        &dst_module,
        &src_module,
        &consumer_history,
        &producer_history,
    );
    let merged = Module {
        name: merged_name,
        builder: mb,
        body,
    };
    let mut inputs: Vec<BufId> = dst_bufs
        .iter()
        .enumerate()
        .filter(|&(p, _)| !removed.contains(&p))
        .map(|(_, &b)| b)
        .collect();
    inputs.extend(appended);
    if inputs.len() != merged.builder.inputs().len() {
        return Err(ferr("merged input binding count mismatch"));
    }

    // Full validation before touching the graph: the merged module must
    // type-check and canonicalize back into the consumer's kernel shape.
    let t = type_infer(&merged)?;
    let vprog = canonicalize(merged.clone(), t)?;
    if vprog.kernels.len() != 1 || vprog.outputs.len() != dst_outputs.len() {
        return Err(ferr(
            "merged module did not canonicalize to a single kernel with the consumer's outputs",
        ));
    }

    // Optional debug hook: when `FUSION_DUMP_STEPS=<dir>` is set, write the
    // grafted HIR of every successful merge to `<dir>/<step>.<name>.hir`
    // before the shared-Arc dedup swaps it. Lets an inspector see every
    // intermediate fused module in the order they were produced without
    // touching apply_fusion's return type.
    if let Some(dir) = std::env::var_os("FUSION_DUMP_STEPS") {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static STEP: AtomicUsize = AtomicUsize::new(0);
        let step = STEP.fetch_add(1, Ordering::Relaxed);
        let dir = std::path::PathBuf::from(dir);
        if std::fs::create_dir_all(&dir).is_ok() {
            let hir = crate::dump::dump_hir(&merged);
            let short = &merged.name[..merged.name.len().min(180)];
            let path = dir.join(format!("{step:03}.{short}.hir"));
            let _ = std::fs::write(path, hir);
        }
    }

    let module = g.dedup_module(Arc::new(merged));
    // Snapshot the intermediate merged kernel for the fusion-history
    // tree entry. Shape strings must be computed now (before further
    // fusion rewrites the buffer table). The snapshot's inner node has
    // its own `fusion_history: None` — it represents the kernel *at
    // this fusion step*, independent of what happens next.
    let input_shapes: Vec<String> = inputs
        .iter()
        .map(|b| crate::graph_ir::format_size(&g.bufs[b.0].size, &g.symbols))
        .collect();
    let output_shapes: Vec<String> = dst_outputs
        .iter()
        .map(|b| crate::graph_ir::format_size(&g.bufs[b.0].size, &g.symbols))
        .collect();
    let snapshot = Arc::new(crate::graph_ir::KernelSnapshot {
        node: Arc::new(KernelModuleNode {
            module: Arc::clone(&module),
            param_bindings: merged_bindings.clone(),
            inputs: inputs.clone(),
            outputs: dst_outputs.clone(),
            fusion_history: None,
        }),
        input_shapes,
        output_shapes,
    });
    let fusion_history = Arc::new(FusionHistory::Fused {
        snapshot,
        consumer: consumer_history,
        producer: producer_history,
    });
    g.nodes[cand.dst] = GraphNode::Kernel(KernelModuleNode {
        module,
        param_bindings: merged_bindings,
        inputs,
        outputs: dst_outputs,
        fusion_history: Some(fusion_history),
    });
    Ok(())
}

/// Cross-builder copier for the producer's scalar result expression:
/// rebuilds src-builder nodes inside the consumer's builder with the
/// producer binders substituted (`subst`), producer-side scalar lets
/// inlined, and producer inputs remapped onto consumer input positions
/// (reusing a consumer position when it binds the same buffer with an
/// identical declaration, appending a new declaration otherwise).
struct GraftCx<'a> {
    src_b: &'a IRBuilder,
    src_env: &'a HashMap<VarId, CanonValue>,
    src_inline_lets: &'a HashMap<VarId, NodeId>,
    src_bufs: &'a [BufId],
    dst_bufs: &'a [BufId],
    /// Producer param bindings, parallel to `src_b.params()`.
    src_param_bindings: &'a [i64],
    /// Producer param -> merged-module size expression. Seeded with the
    /// seam-unified params; the rest are appended lazily on first use
    /// ([`Self::map_params`]).
    param_remap: BTreeMap<VarId, SizeExpr>,
    /// Bound values of the lazily appended params, in append order.
    appended_params: Vec<i64>,
    /// src input position -> consumer-builder node, shared across sites.
    input_map: HashMap<usize, NodeId>,
    /// Buffers of the inputs appended to the consumer, in append order.
    appended: Vec<BufId>,
}

impl GraftCx<'_> {
    /// Rewrites a producer-side size expression into the merged module's
    /// param namespace. Producer params without a seam mapping are
    /// re-declared as fresh params (fresh `VarId`, so no collision with
    /// consumer vars) with their bound value recorded for the merged
    /// bindings.
    fn map_params(&mut self, mb: &mut IRBuilder, e: &SizeExpr) -> Result<SizeExpr, CompileError> {
        let mut ps = BTreeSet::new();
        e.param_syms(&mut ps);
        for v in ps {
            if self.param_remap.contains_key(&v) {
                continue;
            }
            let pos = self
                .src_b
                .params()
                .iter()
                .position(|(pv, _)| *pv == v)
                .ok_or_else(|| ferr(format!("undeclared producer param {v:?}")))?;
            let base = &self.src_b.params()[pos].1;
            let mut name = base.clone();
            let mut k = 0;
            while mb.params().iter().any(|(_, n)| *n == name) {
                k += 1;
                name = format!("{base}__f{k}");
            }
            let value = self.src_param_bindings[pos];
            let fresh = mb.symbol(name);
            mb.extend_shape_hint(value);
            self.param_remap
                .insert(v, SizeExpr::cst(SymConst::Sym(fresh.0)));
            self.appended_params.push(value);
        }
        e.subst_params(&self.param_remap).ok_or_else(|| {
            ferr("a producer param in a coefficient position maps to a compound expression")
        })
    }

    fn map_input(&mut self, mb: &mut IRBuilder, p: usize) -> Result<NodeId, CompileError> {
        if let Some(&n) = self.input_map.get(&p) {
            return Ok(n);
        }
        let buf = self.src_bufs[p];
        let sdecl = self.src_b.inputs()[p].clone();
        let shape: Vec<SizeExpr> = sdecl
            .shape
            .iter()
            .map(|d| self.map_params(mb, d))
            .collect::<Result<_, _>>()?;
        let reuse = (0..self.dst_bufs.len()).find(|&q| {
            let d = &mb.inputs()[q];
            self.dst_bufs[q] == buf && d.elem == sdecl.elem && d.shape == shape
        });
        let n = match reuse {
            Some(q) => mb.intern(Node::Input(q)),
            None => {
                let name = if mb.inputs().iter().any(|d| d.name == sdecl.name) {
                    format!("{}__f{p}", sdecl.name)
                } else {
                    sdecl.name
                };
                self.appended.push(buf);
                mb.input(name, sdecl.elem, shape)
            }
        };
        self.input_map.insert(p, n);
        Ok(n)
    }

    fn copy(
        &mut self,
        mb: &mut IRBuilder,
        id: NodeId,
        subst: &BTreeMap<VarId, NodeId>,
        local: &mut HashMap<VarId, NodeId>,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> Result<NodeId, CompileError> {
        if let Some(&r) = memo.get(&id) {
            return Ok(r);
        }
        let new = match self.src_b.node(id).clone() {
            Node::ConstU32(c) => mb.intern(Node::ConstU32(c)),
            Node::ConstField(c) => mb.intern(Node::ConstField(c)),
            Node::ConstFpExt(c) => mb.intern(Node::ConstFpExt(c)),
            Node::ConstSym(e) => {
                // σ substitution rewrites producer binders at `Node::Var`
                // sites; it cannot reach a loop var embedded inside an
                // `SExpr`, so such an expression cannot be grafted soundly.
                let mut loop_vars = BTreeSet::new();
                e.syms(&mut loop_vars);
                if !loop_vars.is_empty() {
                    return Err(ferr("producer param expression references loop variables"));
                }
                let e2 = self.map_params(mb, &e)?;
                mb.intern(Node::ConstSym(e2))
            }
            Node::Input(p) => self.map_input(mb, p)?,
            Node::Var(v) => {
                if let Some(&n) = subst.get(&v) {
                    n
                } else if let Some(&n) = local.get(&v) {
                    n
                } else if let Some(&n) = self.src_inline_lets.get(&v) {
                    self.copy(mb, n, subst, local, memo)?
                } else if let Some(CanonValue::Scalar(n)) = self.src_env.get(&v) {
                    self.copy(mb, *n, subst, local, memo)?
                } else {
                    return Err(ferr(format!(
                        "unbound variable {v:?} in producer result expression"
                    )));
                }
            }
            Node::LiftFpExt(x) => {
                let x2 = self.copy(mb, x, subst, local, memo)?;
                mb.intern(Node::LiftFpExt(x2))
            }
            Node::Bin(op, a, b) => {
                let a2 = self.copy(mb, a, subst, local, memo)?;
                let b2 = self.copy(mb, b, subst, local, memo)?;
                mb.intern(Node::Bin(op, a2, b2))
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let c2 = self.copy(mb, cond, subst, local, memo)?;
                let t2 = self.copy(mb, then_val, subst, local, memo)?;
                let e2 = self.copy(mb, else_val, subst, local, memo)?;
                mb.intern(Node::Select {
                    cond: c2,
                    then_val: t2,
                    else_val: e2,
                })
            }
            Node::Index { tensor, indices } => {
                let t = match resolve_tensor_ref(self.src_b, self.src_env, tensor)? {
                    TensorRef::Input(p) => self.map_input(mb, p)?,
                    TensorRef::Let { .. } => {
                        return Err(ferr("producer reads a top-level let output"))
                    }
                };
                let ix: Vec<NodeId> = indices
                    .iter()
                    .map(|&i| self.copy(mb, i, subst, local, memo))
                    .collect::<Result<_, _>>()?;
                mb.intern(Node::Index {
                    tensor: t,
                    indices: ix,
                })
            }
            // Scalar lets inside the expression are inlined on copy.
            Node::Let { var, value, body } => {
                let v2 = self.copy(mb, value, subst, local, memo)?;
                local.insert(var, v2);
                self.copy(mb, body, subst, local, memo)?
            }
            n @ (Node::Reduce { .. }
            | Node::Compute { .. }
            | Node::Tuple(_)
            | Node::Pack(_)
            | Node::Proj(..)) => {
                return Err(ferr(format!(
                    "non-scalar construct in producer result expression: {n:?}"
                )))
            }
        };
        memo.insert(id, new);
        Ok(new)
    }
}

/// Rewrites one canonical result expression of the consumer: graft sites
/// are replaced by their prepared producer expressions, peeled scalar lets
/// (`inline_lets`) and top-level scalar bindings (`env`) are inlined (the
/// rebuilt body no longer carries the original `let` wrappers), everything
/// else is rebuilt structurally with binders kept verbatim. Tensor-typed
/// variables resolve to `CanonValue::Tensors` and are deliberately left
/// untouched (a split module has none reachable).
fn rewrite_body(
    mb: &mut IRBuilder,
    id: NodeId,
    grafts: &HashMap<NodeId, NodeId>,
    inline_lets: &HashMap<VarId, NodeId>,
    env: &HashMap<VarId, CanonValue>,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&r) = grafts.get(&id) {
        return r;
    }
    if let Some(&r) = memo.get(&id) {
        return r;
    }
    let new = match mb.node(id).clone() {
        Node::Input(_)
        | Node::ConstU32(_)
        | Node::ConstField(_)
        | Node::ConstFpExt(_)
        | Node::ConstSym(_) => id,
        Node::Var(v) => {
            if let Some(&n) = inline_lets.get(&v) {
                rewrite_body(mb, n, grafts, inline_lets, env, memo)
            } else if let Some(CanonValue::Scalar(n)) = env.get(&v) {
                rewrite_body(mb, *n, grafts, inline_lets, env, memo)
            } else {
                id
            }
        }
        Node::LiftFpExt(x) => {
            let x2 = rewrite_body(mb, x, grafts, inline_lets, env, memo);
            if x2 == x {
                id
            } else {
                mb.intern(Node::LiftFpExt(x2))
            }
        }
        Node::Bin(op, a, b) => {
            let a2 = rewrite_body(mb, a, grafts, inline_lets, env, memo);
            let b2 = rewrite_body(mb, b, grafts, inline_lets, env, memo);
            if (a2, b2) == (a, b) {
                id
            } else {
                mb.intern(Node::Bin(op, a2, b2))
            }
        }
        Node::Select {
            cond,
            then_val,
            else_val,
        } => {
            let c2 = rewrite_body(mb, cond, grafts, inline_lets, env, memo);
            let t2 = rewrite_body(mb, then_val, grafts, inline_lets, env, memo);
            let e2 = rewrite_body(mb, else_val, grafts, inline_lets, env, memo);
            if (c2, t2, e2) == (cond, then_val, else_val) {
                id
            } else {
                mb.intern(Node::Select {
                    cond: c2,
                    then_val: t2,
                    else_val: e2,
                })
            }
        }
        Node::Index { tensor, indices } => {
            let t2 = rewrite_body(mb, tensor, grafts, inline_lets, env, memo);
            let ix2: Vec<NodeId> = indices
                .iter()
                .map(|&ix| rewrite_body(mb, ix, grafts, inline_lets, env, memo))
                .collect();
            if t2 == tensor && ix2 == indices {
                id
            } else {
                mb.intern(Node::Index {
                    tensor: t2,
                    indices: ix2,
                })
            }
        }
        Node::Compute {
            bound,
            var,
            body,
            scatter,
            par,
            threads,
        } => {
            let b2 = rewrite_body(mb, body, grafts, inline_lets, env, memo);
            if b2 == body {
                id
            } else {
                mb.intern(Node::Compute {
                    bound,
                    var,
                    body: b2,
                    scatter,
                    par,
                    threads,
                })
            }
        }
        Node::Reduce {
            op,
            bound,
            var,
            body,
        } => {
            let b2 = rewrite_body(mb, body, grafts, inline_lets, env, memo);
            if b2 == body {
                id
            } else {
                mb.intern(Node::Reduce {
                    op,
                    bound,
                    var,
                    body: b2,
                })
            }
        }
        Node::Let { var, value, body } => {
            let v2 = rewrite_body(mb, value, grafts, inline_lets, env, memo);
            let b2 = rewrite_body(mb, body, grafts, inline_lets, env, memo);
            if (v2, b2) == (value, body) {
                id
            } else {
                mb.intern(Node::Let {
                    var,
                    value: v2,
                    body: b2,
                })
            }
        }
        Node::Tuple(es) => {
            let es2: Vec<NodeId> = es
                .iter()
                .map(|&e| rewrite_body(mb, e, grafts, inline_lets, env, memo))
                .collect();
            if es2 == es {
                id
            } else {
                mb.intern(Node::Tuple(es2))
            }
        }
        Node::Pack(es) => {
            let es2: Vec<NodeId> = es
                .iter()
                .map(|&e| rewrite_body(mb, e, grafts, inline_lets, env, memo))
                .collect();
            if es2 == es {
                id
            } else {
                mb.intern(Node::Pack(es2))
            }
        }
        Node::Proj(t, k) => {
            let t2 = rewrite_body(mb, t, grafts, inline_lets, env, memo);
            if t2 == t {
                id
            } else {
                mb.intern(Node::Proj(t2, k))
            }
        }
    };
    memo.insert(id, new);
    new
}

// ---------------------------------------------------------------------
// Stage 4: dead-node elimination (plan section 9).
// ---------------------------------------------------------------------

/// Removes graph nodes whose writes can never be observed.
///
/// A node is live iff it is a [`GraphNode::BlackboxKernel`] (opaque
/// effects) or writes at least one *needed* buffer. `needed` starts as
/// the set of registered interface buffers (their contents are
/// observable by the caller, so they escape the graph) and grows with
/// the reads of every live node during a reverse walk over `g.nodes`
/// (nodes are inserted write-before-read, so reverse insertion order is
/// a reverse topological order). Needed bits are never cleared: writes
/// may be partial, so an earlier writer of a needed buffer stays live
/// even when a later live node overwrites it.
///
/// Dead nodes are dropped from `g.nodes`; the bufs table is left
/// untouched — orphaned buffers get no pool slot (the planner skips
/// buffers with no readers and no writers).
///
/// Returns the number of nodes removed.
pub fn dce(g: &mut GraphBuilder) -> usize {
    let mut needed: Vec<bool> = (0..g.bufs.len())
        .map(|b| g.buf_is_interface(BufId(b)))
        .collect();
    let mut live = vec![false; g.nodes.len()];
    for (n, node) in g.nodes.iter().enumerate().rev() {
        let (reads, writes) = g.node_reads_writes(node);
        live[n] =
            matches!(node, GraphNode::BlackboxKernel(_)) || writes.iter().any(|b| needed[b.0]);
        if live[n] {
            for (b, _) in reads {
                needed[b.0] = true;
            }
        }
    }
    let before = g.nodes.len();
    let mut it = live.iter();
    g.nodes.retain(|_| *it.next().unwrap());
    before - g.nodes.len()
}

// ---------------------------------------------------------------------
// Stage 5: fixpoint driver (plan section 10).
// ---------------------------------------------------------------------

/// What [`fuse_graph`] did to the graph.
#[derive(Clone, Debug, Default)]
pub struct FusionReport {
    /// Rounds that had at least one selected candidate.
    pub rounds: usize,
    /// Successfully applied fusions, as (producer, consumer) module names.
    pub fused: Vec<(String, String)>,
    /// Node count on entry (before the initial DCE).
    pub nodes_before: usize,
    /// Node count after the final DCE.
    pub nodes_after: usize,
    /// Drop in the number of distinct `Arc<Module>`s achieved by the
    /// final α-normalization sweep ([`dedup_modules`]).
    pub deduped: usize,
    /// One entry per fusion round (post-initial-DCE), in order.
    pub rounds_detail: Vec<RoundStats>,
}

/// Per-round counters recorded by [`fuse_graph`].
#[derive(Copy, Clone, Debug, Default)]
pub struct RoundStats {
    /// 1-based round index.
    pub round: usize,
    /// Successful `apply_fusion` calls in this round.
    pub fused: usize,
    /// Nodes removed by the round's post-apply DCE.
    pub dce_removed: usize,
    /// Kernel nodes remaining after DCE.
    pub nodes_after: usize,
    /// Distinct `Arc<ir::Module>`s across the surviving kernel nodes.
    pub unique_modules_after: usize,
}

/// Count distinct `Arc<ir::Module>` pointers across the graph's kernel
/// nodes — matches how [`crate::graph_ir::GraphBuilder::dedup_module`] and
/// downstream Arc-identity caches see "one module".
fn distinct_kernel_modules(g: &GraphBuilder) -> usize {
    g.nodes
        .iter()
        .filter_map(|n| match n {
            GraphNode::Kernel(k) => Some(Arc::as_ptr(&k.module)),
            _ => None,
        })
        .collect::<HashSet<_>>()
        .len()
}

/// Runs the fusion pipeline to a fixpoint (plan section 10).
///
/// DCE runs first — dead readers of a fusable buffer would inflate
/// `n_readers` in the cost model (shrinking each pair's amortized share
/// of the launch/write savings) and dead producers would waste
/// extraction work. Each round then extracts access relations (cached
/// across rounds by module hash), enumerates every `(producer, reader)`
/// candidate, greedily selects a dst-disjoint subset (a producer may be
/// picked into multiple readers per round), applies each candidate
/// (validation failures skip that candidate and leave the graph
/// untouched), and DCEs any producers whose last reader was just fused
/// away. Rounds stop when no candidate is selected, when a round applies
/// nothing, or after `max_iterations`. A final [`dedup_modules`] sweep
/// folds α-equivalent kernel modules onto shared `Arc`s so identical
/// fused patterns JIT-compile once.
pub fn fuse_graph(g: &mut GraphBuilder, opts: &FusionOptions) -> FusionReport {
    let mut report = FusionReport {
        nodes_before: g.nodes.len(),
        ..FusionReport::default()
    };
    let initial_dce = dce(g);
    if opts.verbose {
        eprintln!(
            "[fusion] round 0 (initial DCE): removed={initial_dce}, nodes_after={}, \
             unique_modules_after={}",
            g.nodes.len(),
            distinct_kernel_modules(g),
        );
    }
    let mut cache = RelationCache::new();
    for _ in 0..opts.max_iterations {
        let rels = extract_relations(g, &mut cache);
        let selected = select_candidates(enumerate_candidates(g, &rels, opts));
        if selected.is_empty() {
            break;
        }
        report.rounds += 1;
        let mut fused_this_round = 0usize;
        for cand in &selected {
            let (GraphNode::Kernel(src), GraphNode::Kernel(dst)) =
                (&g.nodes[cand.src], &g.nodes[cand.dst])
            else {
                unreachable!("candidates only pair Kernel nodes");
            };
            let names = (src.module.name.clone(), dst.module.name.clone());
            if apply_fusion(g, cand).is_ok() {
                report.fused.push(names);
                fused_this_round += 1;
            }
        }
        let dce_removed = dce(g);
        let stats = RoundStats {
            round: report.rounds,
            fused: fused_this_round,
            dce_removed,
            nodes_after: g.nodes.len(),
            unique_modules_after: distinct_kernel_modules(g),
        };
        if opts.verbose {
            eprintln!(
                "[fusion] round {}: fused={}, dce_removed={}, nodes_after={}, \
                 unique_modules_after={}",
                stats.round,
                stats.fused,
                stats.dce_removed,
                stats.nodes_after,
                stats.unique_modules_after,
            );
        }
        report.rounds_detail.push(stats);
        if fused_this_round == 0 {
            // Every selected candidate failed validation; re-enumerating
            // would just select the same set again.
            break;
        }
    }
    report.deduped = dedup_modules(g);
    report.nodes_after = g.nodes.len();
    report
}

// ---------------------------------------------------------------------
// Stage 6: post-fusion module deduplication (plan section 11).
// ---------------------------------------------------------------------

/// α-normalizes every kernel module in the graph and folds structurally
/// identical ones onto a single `Arc<Module>`, so each distinct fused
/// pattern JIT-compiles once.
///
/// [`module_hash`] canonicalizes `NodeId`s but hashes raw `VarId`s, and
/// fusion surgery α-renames producer binders with fresh variables from
/// the consumer's builder — so two fusions of the same logical pattern
/// at different graph locations produce α-equivalent modules with
/// different hashes, which both the in-memory module dedup and the
/// on-disk JIT cache miss. [`renumber_module`] rewrites `VarId`s into a
/// canonical numbering, after which α-equivalent modules hash
/// identically and [`GraphBuilder::dedup_module`] collapses them onto
/// one `Arc`. The sweep covers *all* kernel nodes, so pre-existing
/// α-variant duplicates in the input graph are deduped too.
///
/// Returns the drop in the number of distinct module `Arc`s.
pub fn dedup_modules(g: &mut GraphBuilder) -> usize {
    let before = distinct_kernel_modules(g);
    for i in 0..g.nodes.len() {
        let GraphNode::Kernel(k) = &g.nodes[i] else {
            continue;
        };
        let renumbered = Arc::new(renumber_module(&k.module));
        let module = g.dedup_module(renumbered);
        let GraphNode::Kernel(k) = &mut g.nodes[i] else {
            unreachable!();
        };
        k.module = module;
    }
    before - distinct_kernel_modules(g)
}

/// Rebuilds `m` with a canonical `VarId` numbering: reachable nodes are
/// walked in the same canonical post-order [`module_hash`] uses, and
/// every `VarId` slot — `Var` references, `Compute`/`Reduce`/`Let`
/// binders, scatter params / exprs / bounds and `ParSpec` symbols — is
/// assigned a new id in first-occurrence order over that walk. Modules
/// that differ only by a `VarId` bijection renumber to hash-identical
/// modules; renumbering an already-canonical module is a no-op (up to
/// arena garbage, which the hash never sees).
pub(crate) fn renumber_module(m: &Module) -> Module {
    let b = &m.builder;

    // Canonical post-order over reachable nodes (mirrors module_hash's
    // WalkCtx: children depth-first, left-to-right; DAG, so no cycles).
    fn number(b: &IRBuilder, id: NodeId, seen: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) {
        if seen.contains(&id) {
            return;
        }
        for child in children_of(b.node(id)) {
            number(b, child, seen, order);
        }
        seen.insert(id);
        order.push(id);
    }
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    number(b, m.body, &mut seen, &mut order);

    // Pass 1: assign new VarIds in first-occurrence order, visiting each
    // node's VarId slots in the same order module_hash feeds them. Uses
    // inside a compute body come before the binder (post-order emits the
    // body first); that's fine — the order only needs to be structural.
    // Scatter bounds keys are always covered by the params vec, so the
    // BTreeMap's raw-id iteration order never decides an assignment.
    fn visit_var(vmap: &mut HashMap<VarId, VarId>, v: VarId) {
        let next = VarId(vmap.len() as u32);
        vmap.entry(v).or_insert(next);
    }
    fn visit_quast(vmap: &mut HashMap<VarId, VarId>, q: &Quast) {
        match q {
            Quast::Sym(v) => visit_var(vmap, *v),
            Quast::Const(_) => {}
            Quast::Add(a, b) => {
                visit_quast(vmap, a);
                visit_quast(vmap, b);
            }
            Quast::Mul(a, _) | Quast::FloorDiv(a, _) | Quast::Neg(a) => visit_quast(vmap, a),
        }
    }
    let mut vmap: HashMap<VarId, VarId> = HashMap::new();
    // Module parameters are declarations, not walk-reachable nodes: number
    // them first, in declaration order, so bounds and input shapes that
    // mention them renumber deterministically.
    for (v, _) in b.params() {
        visit_var(&mut vmap, *v);
    }
    for &id in &order {
        match b.node(id) {
            Node::Var(v) => visit_var(&mut vmap, *v),
            Node::Compute {
                var, scatter, par, ..
            } => {
                visit_var(&mut vmap, *var);
                if let Some(s) = scatter {
                    for &p in &s.params {
                        visit_var(&mut vmap, p);
                    }
                    for e in &s.exprs {
                        visit_quast(&mut vmap, e);
                    }
                    for &p in &s.inv_params {
                        visit_var(&mut vmap, p);
                    }
                    for e in &s.inv_exprs {
                        visit_quast(&mut vmap, e);
                    }
                    for &v in s.bounds.keys() {
                        visit_var(&mut vmap, v);
                    }
                }
                if let Some(p) = par {
                    visit_var(&mut vmap, p.thread);
                    visit_var(&mut vmap, p.seq);
                    visit_quast(&mut vmap, &p.expr);
                }
            }
            Node::Reduce { var, .. } | Node::Let { var, .. } => visit_var(&mut vmap, *var),
            _ => {}
        }
    }

    // Pass 2: rebuild into a fresh builder in canonical order, remapping
    // every VarId; children precede parents in `order`, so operand ids
    // are always already mapped.
    let mut nb = IRBuilder::new();
    for (v, name) in b.params() {
        nb.inherit_param(vmap[v], name.clone());
    }
    if let Some(h) = b.shape_hint() {
        nb.add_shape_hint(h);
    }
    if let Some(block) = b.block_hint() {
        nb.set_block_hint(block);
    }
    for d in b.inputs() {
        let shape: Vec<SizeExpr> = d.shape.iter().map(|e| remap_sexpr(e, &vmap)).collect();
        nb.input(d.name.clone(), d.elem, shape);
    }
    let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
    for &id in &order {
        let new = match b.node(id) {
            Node::Input(k) => nb.intern(Node::Input(*k)),
            Node::Var(v) => nb.intern(Node::Var(vmap[v])),
            Node::ConstU32(c) => nb.const_u32(*c),
            Node::ConstField(c) => nb.const_field(*c),
            Node::ConstFpExt(c) => nb.const_fpext(*c),
            Node::ConstSym(e) => nb.intern(Node::ConstSym(remap_sexpr(e, &vmap))),
            Node::LiftFpExt(x) => nb.lift_fpext(node_map[x]),
            Node::Bin(op, x, y) => nb.bin(*op, node_map[x], node_map[y]),
            Node::Select {
                cond,
                then_val,
                else_val,
            } => nb.select(node_map[cond], node_map[then_val], node_map[else_val]),
            Node::Index { tensor, indices } => {
                let idx: Vec<NodeId> = indices.iter().map(|i| node_map[i]).collect();
                nb.index(node_map[tensor], &idx)
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
                par,
                threads,
            } => nb.intern(Node::Compute {
                bound: remap_sexpr(bound, &vmap),
                var: vmap[var],
                body: node_map[body],
                scatter: scatter.as_ref().map(|s| Box::new(remap_scatter(s, &vmap))),
                par: par.as_ref().map(|p| Box::new(remap_par(p, &vmap))),
                threads: *threads,
            }),
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => nb.intern(Node::Reduce {
                op: *op,
                bound: remap_sexpr(bound, &vmap),
                var: vmap[var],
                body: node_map[body],
            }),
            Node::Let { var, value, body } => nb.intern(Node::Let {
                var: vmap[var],
                value: node_map[value],
                body: node_map[body],
            }),
            Node::Tuple(es) => {
                let es: Vec<NodeId> = es.iter().map(|e| node_map[e]).collect();
                nb.tuple(&es)
            }
            Node::Proj(t, k) => nb.proj(node_map[t], *k),
            Node::Pack(es) => {
                let es: Vec<NodeId> = es.iter().map(|e| node_map[e]).collect();
                nb.pack(&es)
            }
        };
        node_map.insert(id, new);
    }
    nb.raise_var_watermark(vmap.len() as u32);
    Module {
        name: m.name.clone(),
        builder: nb,
        body: node_map[&m.body],
    }
}

/// Remaps module-parameter `VarId`s (`SymConst` positions) in a bound or
/// shape expression. Sizes never contain loop-var `Sym` nodes, but those are
/// remapped too for uniformity.
fn remap_sexpr(e: &SizeExpr, vmap: &HashMap<VarId, VarId>) -> SizeExpr {
    use quast::SymConst;
    let c = |k: &SymConst| match k {
        SymConst::Sym(v) => SymConst::Sym(vmap[v]),
        lit => *lit,
    };
    match e {
        SizeExpr::Sym(v) => SizeExpr::Sym(vmap[v]),
        SizeExpr::Const(k) => SizeExpr::Const(c(k)),
        SizeExpr::Add(a, b) => SizeExpr::Add(
            Arc::new(remap_sexpr(a, vmap)),
            Arc::new(remap_sexpr(b, vmap)),
        ),
        SizeExpr::Mul(a, k) => SizeExpr::Mul(Arc::new(remap_sexpr(a, vmap)), c(k)),
        SizeExpr::FloorDiv(a, k) => SizeExpr::FloorDiv(Arc::new(remap_sexpr(a, vmap)), c(k)),
        SizeExpr::Neg(a) => SizeExpr::Neg(Arc::new(remap_sexpr(a, vmap))),
    }
}

fn remap_quast(q: &Quast, vmap: &HashMap<VarId, VarId>) -> Quast {
    match q {
        Quast::Sym(v) => Quast::Sym(vmap[v]),
        Quast::Const(c) => Quast::Const(*c),
        Quast::Add(a, b) => Quast::Add(
            Arc::new(remap_quast(a, vmap)),
            Arc::new(remap_quast(b, vmap)),
        ),
        Quast::Mul(a, c) => Quast::Mul(Arc::new(remap_quast(a, vmap)), *c),
        Quast::FloorDiv(a, c) => Quast::FloorDiv(Arc::new(remap_quast(a, vmap)), *c),
        Quast::Neg(a) => Quast::Neg(Arc::new(remap_quast(a, vmap))),
    }
}

fn remap_scatter(s: &quast::Scatter, vmap: &HashMap<VarId, VarId>) -> quast::Scatter {
    quast::Scatter {
        params: s.params.iter().map(|p| vmap[p]).collect(),
        exprs: s.exprs.iter().map(|e| remap_quast(e, vmap)).collect(),
        inv_params: s.inv_params.iter().map(|p| vmap[p]).collect(),
        inv_exprs: s.inv_exprs.iter().map(|e| remap_quast(e, vmap)).collect(),
        out_shape: s.out_shape.clone(),
        bounds: s.bounds.iter().map(|(v, &bnd)| (vmap[v], bnd)).collect(),
    }
}

fn remap_par(p: &quast::ParSpec, vmap: &HashMap<VarId, VarId>) -> quast::ParSpec {
    quast::ParSpec {
        thread: vmap[&p.thread],
        seq: vmap[&p.seq],
        expr: remap_quast(&p.expr, vmap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ScalarType;

    fn extract(b: IRBuilder, body: NodeId) -> AccessRelation {
        let module = b.finish("k", body);
        extract_one(&module, &BTreeMap::new()).unwrap()
    }

    #[test]
    fn simple_elementwise() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let body = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(r.outer_bound, 16);
        assert_eq!(r.reads.len(), 1);
        assert_eq!(r.reads[0].input_pos, 0);
        assert!(r.reads[0].inner.is_empty());
        assert_eq!(r.reads[0].expr, Some(Quast::sym(r.outer_var)));
        assert_eq!(
            r.writes,
            vec![WriteAccess {
                out_pos: 0,
                len_elems: 16,
                inner_factor: 1,
                elem_bytes: 4
            }]
        );
        assert_eq!(
            r.write_inverse,
            Some(InverseMap::DivMod { tile: 1, pack: 1 })
        );
        // One BabyBear mul; address math is free.
        assert_eq!(r.cost.flops_per_elem, 8.0);
        assert_eq!(r.cost.bytes_in_per_elem, 4.0);
        assert_eq!(r.cost.bytes_out_per_elem, 4.0);
    }

    /// `a[2*i]`, `a[0]`, `a[i/2]`, `a[i % 4]` all extract to the expected
    /// quasi-affine forms.
    #[test]
    fn affine_read_forms() {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![32]);
        let body = b.compute(8, |b, i| {
            let c0 = b.const_u32(0);
            let c2 = b.const_u32(2);
            let c4 = b.const_u32(4);
            let i2 = b.mul(i, c2);
            let l1 = b.index(a, &[i2]);
            let l2 = b.index(a, &[c0]);
            let idiv = b.div(i, c2);
            let l3 = b.index(a, &[idiv]);
            let irem = b.rem(i, c4);
            let l4 = b.index(a, &[irem]);
            let s1 = b.add(l1, l2);
            let s2 = b.add(s1, l3);
            b.add(s2, l4)
        });
        let r = extract(b, body);
        assert_eq!(r.reads.len(), 4);
        let i = Quast::sym(r.outer_var);
        let mut got: Vec<Quast> = r
            .reads
            .iter()
            .map(|ra| ra.expr.clone().expect("affine site"))
            .collect();
        let mut want = vec![i.mul_c(2), Quast::cst(0), i.floordiv(2), i.rem_c(4)];
        got.sort();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn pack_write_and_inverse() {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![8]);
        let one = b.const_field(1);
        let body = b.compute(8, |b, i| {
            let ai = b.index(a, &[i]);
            let ai1 = b.add(ai, one);
            b.pack(&[ai, ai1])
        });
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(
            r.writes,
            vec![WriteAccess {
                out_pos: 0,
                len_elems: 16,
                inner_factor: 2,
                elem_bytes: 4
            }]
        );
        assert_eq!(
            r.write_inverse,
            Some(InverseMap::DivMod { tile: 1, pack: 2 })
        );
        assert_eq!(r.cost.bytes_out_per_elem, 8.0);
    }

    /// A small sequential reduce keeps the kernel single-kernel but makes
    /// it General; the site records the reduce binder and the cost scales
    /// by the reduce bound.
    #[test]
    fn small_reduce_is_general() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![24]);
        let body = b.compute(8, |b, i| {
            // K = 3: non-power-of-two, so never tree-lowered.
            b.reduce(ReduceOp::Add, 3, |b, r| {
                let c3 = b.const_u32(3);
                let row = b.mul(i, c3);
                let ix = b.add(row, r);
                b.index(x, &[ix])
            })
        });
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::General);
        assert_eq!(r.reads.len(), 1);
        let ra = &r.reads[0];
        assert_eq!(ra.inner.len(), 1);
        assert_eq!(ra.inner[0].1, 3);
        let want = Quast::sym(r.outer_var)
            .mul_c(3)
            .add(&Quast::sym(ra.inner[0].0));
        assert_eq!(ra.expr, Some(want));
        // 3 accumulator adds at BabyBear weight 2; address math free.
        assert_eq!(r.cost.flops_per_elem, 6.0);
        assert_eq!(r.cost.bytes_in_per_elem, 12.0);
    }

    #[test]
    fn tree_lowered_reduce_is_multikernel() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![4096]);
        let body = b.compute(4, |b, i| {
            b.reduce(ReduceOp::Add, 1024, |b, r| {
                let c = b.const_u32(1024);
                let row = b.mul(i, c);
                let ix = b.add(row, r);
                b.index(x, &[ix])
            })
        });
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::MultiKernel);
    }

    #[test]
    fn scatter_store_is_general_without_inverse() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let sc = b.scatter_map(
            1,
            None,
            |p, _c| vec![p[0].rem_c(8).mul_c(2).add(&p[0].floordiv(8))],
            |q, _c| vec![q[0].rem_c(2).mul_c(8).add(&q[0].floordiv(2))],
        );
        let body = b.compute_scatter(16, sc, |b, i| b.index(x, &[i]));
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::General);
        assert!(r.write_inverse.is_none());
    }

    /// `a[idx[i]]`: the gather site is non-affine (`expr == None`) but the
    /// inner `idx[i]` load is still discovered and both loads are charged.
    #[test]
    fn nonaffine_gather_read() {
        let mut b = IRBuilder::new();
        let idx = b.input("idx", ScalarType::U32, vec![8]);
        let a = b.input("a", ScalarType::BabyBear, vec![64]);
        let body = b.compute(8, |b, i| {
            let ii = b.index(idx, &[i]);
            b.index(a, &[ii])
        });
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(r.reads.len(), 2);
        let gather = r.reads.iter().find(|ra| ra.input_pos == 1).unwrap();
        assert!(gather.expr.is_none());
        let index_load = r.reads.iter().find(|ra| ra.input_pos == 0).unwrap();
        assert_eq!(index_load.expr, Some(Quast::sym(r.outer_var)));
        assert_eq!(r.cost.bytes_in_per_elem, 8.0);
    }

    /// A let-bound tile makes the kernel General; the tile read is shared
    /// memory (no ReadAccess), while the global load inside the tile body
    /// records the tile binder.
    #[test]
    fn tiled_kernel_is_general_with_tile_reads_excluded() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let body = b.compute(2, |b, i| {
            let tile = b.compute(8, |b, j| {
                let c8 = b.const_u32(8);
                let row = b.mul(i, c8);
                let ix = b.add(row, j);
                b.index(x, &[ix])
            });
            b.bind(tile, |b, tv| {
                b.compute(8, |b, j| {
                    let tj = b.index(tv, &[j]);
                    b.add(tj, tj)
                })
            })
        });
        let r = extract(b, body);
        assert_eq!(r.class, KernelClass::General);
        assert_eq!(r.reads.len(), 1);
        let ra = &r.reads[0];
        assert_eq!(ra.input_pos, 0);
        assert_eq!(ra.inner.len(), 1);
        assert_eq!(ra.inner[0].1, 8);
        let want = Quast::sym(r.outer_var)
            .mul_c(8)
            .add(&Quast::sym(ra.inner[0].0));
        assert_eq!(ra.expr, Some(want));
    }

    // ------------------------------------------------------------------
    // Stage 2: enumeration, dispatch, scoring, selection.
    // ------------------------------------------------------------------

    use crate::graph_ir::{BufInfo, DeviceType};

    fn graph_of(module: Module, n_in: usize, n_out: usize) -> (GraphBuilder, Vec<BufId>) {
        let mut g = GraphBuilder::new();
        let buf = |g: &mut GraphBuilder, name: String| {
            g.add_buf(BufInfo {
                name: Some(name),
                device_type: DeviceType::Cuda(0),
                size: Quast::cst(1 << 20),
                elem_size: 4,
            })
        };
        let ins: Vec<BufId> = (0..n_in).map(|k| buf(&mut g, format!("in{k}"))).collect();
        let outs: Vec<BufId> = (0..n_out).map(|k| buf(&mut g, format!("out{k}"))).collect();
        for &b in &ins {
            g.register_input(b);
        }
        for &b in &outs {
            g.register_output(b);
        }
        g.insert_kernel(module, ins.clone(), outs, &[]);
        (g, ins)
    }

    fn candidates(g: &GraphBuilder, opts: &FusionOptions) -> Vec<FusionCandidate> {
        let mut cache = RelationCache::new();
        let rels = extract_relations(g, &mut cache);
        enumerate_candidates(g, &rels, opts)
    }

    /// `a = x * 2; out[i] = a[i] + a[(i + 1) % 16]`.
    fn chain_module() -> Module {
        chain_module_shifted(0)
    }

    /// [`chain_module`] built after burning `shift` fresh `VarId`s: same
    /// structure, disjoint variable numbering — an α-variant with a
    /// different [`module_hash`].
    fn chain_module_shifted(shift: u32) -> Module {
        let mut b = IRBuilder::new();
        for _ in 0..shift {
            b.fresh_var();
        }
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                let c1 = b.const_u32(1);
                let c16 = b.const_u32(16);
                let i1 = b.add(i, c1);
                let iw = b.rem(i1, c16);
                let ai1 = b.index(av, &[iw]);
                b.add(ai, ai1)
            })
        });
        b.finish("chain", body)
    }

    #[test]
    fn elementwise_chain_yields_case_a_candidate() {
        let (g, _) = graph_of(chain_module(), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        let c = &cands[0];
        assert_eq!((c.src, c.dst), (0, 1));
        assert!(c.score > 0.0);
        assert_eq!(c.info.rewrites.len(), 2);
        for rw in &c.info.rewrites {
            assert_eq!(rw.pack_elem, None);
            assert_eq!(rw.sigma.len(), 1);
        }
    }

    /// `b[i] = a[2i] + a[2i]` reads a strict subsample of the producer:
    /// still fusable, with sigma composing the producer var through `2i`.
    #[test]
    fn subsample_consumer_fuses_with_composed_index() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            b.compute(8, |b, i| {
                let c2 = b.const_u32(2);
                let i2 = b.mul(i, c2);
                let ai = b.index(av, &[i2]);
                b.add(ai, ai)
            })
        });
        let (g, _) = graph_of(b.finish("sub", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        let rw = &cands[0].info.rewrites[0];
        assert_eq!(rw.sigma.len(), 1);
        assert!(matches!(
            rw.sigma.values().next().unwrap(),
            Quast::Mul(_, 2)
        ));
    }

    /// A pack producer consumed at constant members resolves `pack_elem`.
    #[test]
    fn pack_producer_resolves_members() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![8]);
        let one = b.const_field(1);
        let a = b.compute(8, |b, i| {
            let xi = b.index(x, &[i]);
            let xi1 = b.add(xi, one);
            b.pack(&[xi, xi1])
        });
        let body = b.bind(a, |b, av| {
            b.compute(8, |b, i| {
                let c0 = b.const_u32(0);
                let c1 = b.const_u32(1);
                let e0 = b.index(av, &[i, c0]);
                let e1 = b.index(av, &[i, c1]);
                b.add(e0, e1)
            })
        });
        let (g, _) = graph_of(b.finish("pk", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        let mut elems: Vec<Option<usize>> = cands[0]
            .info
            .rewrites
            .iter()
            .map(|rw| rw.pack_elem)
            .collect();
        elems.sort();
        assert_eq!(elems, vec![Some(0), Some(1)]);
    }

    /// A pack member selected by a symbolic expression (`a[i, i % 2]`)
    /// cannot be proven constant: the pair is rejected.
    #[test]
    fn symbolic_pack_member_is_rejected() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![8]);
        let one = b.const_field(1);
        let a = b.compute(8, |b, i| {
            let xi = b.index(x, &[i]);
            let xi1 = b.add(xi, one);
            b.pack(&[xi, xi1])
        });
        let body = b.bind(a, |b, av| {
            b.compute(8, |b, i| {
                let c2 = b.const_u32(2);
                let e = b.rem(i, c2);
                let v = b.index(av, &[i, e]);
                b.add(v, v)
            })
        });
        let (g, _) = graph_of(b.finish("pkr", body), 1, 1);
        assert!(candidates(&g, &FusionOptions::default()).is_empty());
    }

    /// A reduce in the producer body is not scalar-body: rejected in v1.
    #[test]
    fn reduce_producer_is_rejected() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![24]);
        let two = b.const_field(2);
        let a = b.compute(8, |b, i| {
            b.reduce(ReduceOp::Add, 3, |b, r| {
                let c3 = b.const_u32(3);
                let row = b.mul(i, c3);
                let ix = b.add(row, r);
                b.index(x, &[ix])
            })
        });
        let body = b.bind(a, |b, av| {
            b.compute(8, |b, i| {
                let ai = b.index(av, &[i]);
                b.mul(ai, two)
            })
        });
        let (g, _) = graph_of(b.finish("red", body), 1, 1);
        assert!(candidates(&g, &FusionOptions::default()).is_empty());
    }

    /// A writer of the producer's input between producer and consumer is a
    /// WAR hazard: recompute at the consumer would read the new value.
    #[test]
    fn war_hazard_blocks_fusion() {
        let (mut g, ins) = graph_of(chain_module(), 1, 1);
        g.insert_memset(ins[0], 0);
        let last = g.nodes.len() - 1;
        g.nodes.swap(1, last);
        assert!(candidates(&g, &FusionOptions::default()).is_empty());
    }

    /// Fan-out from a single producer: enumeration emits one candidate
    /// per reader, each amortizing the producer's write and launch costs
    /// over the two readers; greedy accepts both (dst-disjoint even
    /// though they share the src), and the round applies both fusions
    /// so DCE drops the producer.
    #[test]
    fn multiple_readers_fuse_pairwise() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let one = b.const_field(1);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            let b1 = b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                b.add(ai, one)
            });
            b.bind(b1, |b, v1| {
                let b2 = b.compute(16, |b, i| {
                    let ai = b.index(av, &[i]);
                    b.mul(ai, two)
                });
                b.bind(b2, |b, v2| b.tuple(&[v1, v2]))
            })
        });
        let (mut g, _) = graph_of(b.finish("fan", body), 1, 2);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 2, "one candidate per reader of `a`");
        assert!(cands.iter().all(|c| c.src == 0), "shared producer");
        let dsts: HashSet<usize> = cands.iter().map(|c| c.dst).collect();
        assert_eq!(dsts, HashSet::from([1, 2]));
        let picked = select_candidates(cands);
        assert_eq!(picked.len(), 2, "dst-disjoint greedy takes both");
        let report = fuse_graph(&mut g, &FusionOptions::default());
        assert_eq!(report.rounds, 1);
        assert_eq!(report.fused.len(), 2);
        assert_eq!(report.nodes_after, 2, "producer DCE'd; two fused consumers");
    }

    /// General producer (kept nest from the 0e probe rollback) into a
    /// Simple consumer: case B builds a two-binder sigma through the
    /// DivMod write inverse.
    #[test]
    fn general_producer_fuses_via_case_b() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![32]);
        // a[p][t] = x[((p % 4) * 8 + (t % 5) * 2 + 1) % 32]: the absorbed
        // `(v % 12) % 5` (5 does not divide 12) keeps the index unprovable
        // and the nest un-absorbed, so the producer is General with a
        // trailing inner compute.
        let a = b.compute(12, |b, p| {
            b.compute(12, |b, t| {
                let c2 = b.const_u32(2);
                let c4 = b.const_u32(4);
                let c5 = b.const_u32(5);
                let c8 = b.const_u32(8);
                let c1 = b.const_u32(1);
                let c32 = b.const_u32(32);
                let pm = b.rem(p, c4);
                let row = b.mul(pm, c8);
                let tm = b.rem(t, c5);
                let col = b.mul(tm, c2);
                let s = b.add(row, col);
                let s1 = b.add(s, c1);
                let ix = b.rem(s1, c32);
                b.index(x, &[ix])
            })
        });
        let two = b.const_field(2);
        let body = b.bind(a, |b, av| {
            b.compute(144, |b, i| {
                let c12 = b.const_u32(12);
                let d = b.div(i, c12);
                let r = b.rem(i, c12);
                let ai = b.index(av, &[d, r]);
                b.mul(ai, two)
            })
        });
        let (g, _) = graph_of(b.finish("caseb", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        let rw = &cands[0].info.rewrites[0];
        assert_eq!(rw.pack_elem, None);
        assert_eq!(rw.sigma.len(), 2);
        assert!(rw
            .sigma
            .values()
            .any(|q| matches!(q, Quast::FloorDiv(_, 12))));
    }

    #[test]
    fn body_ops_guard_rejects_large_pairs() {
        let (g, _) = graph_of(chain_module(), 1, 1);
        let opts = FusionOptions {
            max_body_ops: 1,
            ..Default::default()
        };
        assert!(candidates(&g, &opts).is_empty());
    }

    /// a -> b -> c yields two candidates sharing the middle node; greedy
    /// selection keeps exactly one.
    #[test]
    fn greedy_selection_is_node_disjoint() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let one = b.const_field(1);
        let three = b.const_field(3);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            let m = b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                b.add(ai, one)
            });
            b.bind(m, |b, mv| {
                b.compute(16, |b, i| {
                    let mi = b.index(mv, &[i]);
                    b.mul(mi, three)
                })
            })
        });
        let (g, _) = graph_of(b.finish("abc", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 2);
        let picked = select_candidates(cands);
        assert_eq!(picked.len(), 1);
        assert_eq!((picked[0].src, picked[0].dst), (0, 1));
    }

    #[test]
    fn split_div_rem_forms() {
        let i = Quast::sym(VarId(900));
        // constants
        assert_eq!(split_div_rem(&Quast::cst(7), 2), Some((Quast::cst(3), 1)));
        // 2i by 2 -> (i, 0)
        assert_eq!(split_div_rem(&i.mul_c(2), 2), Some((i.clone(), 0)));
        // 2i + 1 by 2 -> (i, 1)
        assert_eq!(
            split_div_rem(&i.mul_c(2).add(&Quast::cst(1)), 2),
            Some((i.clone(), 1))
        );
        // 4i + 6 by 4 -> (i + 1, 2)
        assert_eq!(
            split_div_rem(&i.mul_c(4).add(&Quast::cst(6)), 4),
            Some((i.add(&Quast::cst(1)), 2))
        );
        // 6i by 4: 6 = 2 (mod 4), remainder not constant
        assert_eq!(split_div_rem(&i.mul_c(6), 4), None);
        // -(2i + 1) by 2 -> (-i - 1, 1)
        assert_eq!(
            split_div_rem(&i.mul_c(2).add(&Quast::cst(1)).neg(), 2),
            Some((i.neg().add(&Quast::cst(-1)), 1))
        );
        // bare symbol by 2: unknown remainder
        assert_eq!(split_div_rem(&i, 2), None);
        // floordiv is opaque
        assert_eq!(split_div_rem(&i.floordiv(2).mul_c(3), 2), None);
    }

    // ------------------------------------------------------------------
    // Stage 3: fusion application.
    // ------------------------------------------------------------------

    fn kernel_at(g: &GraphBuilder, idx: usize) -> &crate::graph_ir::KernelModuleNode {
        let GraphNode::Kernel(kn) = &g.nodes[idx] else {
            panic!("expected kernel node at {idx}");
        };
        kn
    }

    #[test]
    fn apply_fuses_elementwise_chain() {
        let (mut g, ins) = graph_of(chain_module(), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![ins[0]]);
        let r = extract_one(&kn.module, &BTreeMap::new()).unwrap();
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(r.outer_bound, 16);
        // Both grafted sites read `x` directly through the composed indices.
        assert_eq!(r.reads.len(), 2);
        assert!(r.reads.iter().all(|ra| ra.input_pos == 0));
        let i = Quast::sym(r.outer_var);
        let mut got: Vec<Quast> = r
            .reads
            .iter()
            .map(|ra| ra.expr.clone().expect("affine site"))
            .collect();
        let mut want = vec![i.clone(), i.add(&Quast::cst(1)).rem_c(16)];
        got.sort();
        want.sort();
        assert_eq!(got, want);
    }

    /// Pack producer: each consumer site grafts its selected member; both
    /// members share the single `x[i]` load after hash-consing.
    #[test]
    fn apply_grafts_pack_members() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![8]);
        let one = b.const_field(1);
        let a = b.compute(8, |b, i| {
            let xi = b.index(x, &[i]);
            let xi1 = b.add(xi, one);
            b.pack(&[xi, xi1])
        });
        let body = b.bind(a, |b, av| {
            b.compute(8, |b, i| {
                let c0 = b.const_u32(0);
                let c1 = b.const_u32(1);
                let e0 = b.index(av, &[i, c0]);
                let e1 = b.index(av, &[i, c1]);
                b.add(e0, e1)
            })
        });
        let (mut g, ins) = graph_of(b.finish("pk", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![ins[0]]);
        let r = extract_one(&kn.module, &BTreeMap::new()).unwrap();
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(r.reads.len(), 1);
        assert_eq!(r.reads[0].expr, Some(Quast::sym(r.outer_var)));
    }

    /// Case B: the General producer's body is grafted with its two binders
    /// mapped through the DivMod write inverse.
    #[test]
    fn apply_fuses_case_b_producer() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![32]);
        let a = b.compute(12, |b, p| {
            b.compute(12, |b, t| {
                let c4 = b.const_u32(4);
                let c8 = b.const_u32(8);
                let c1 = b.const_u32(1);
                let c32 = b.const_u32(32);
                let pm = b.rem(p, c4);
                let row = b.mul(pm, c8);
                let s = b.add(row, t);
                let s1 = b.add(s, c1);
                let ix = b.rem(s1, c32);
                b.index(x, &[ix])
            })
        });
        let two = b.const_field(2);
        let body = b.bind(a, |b, av| {
            b.compute(144, |b, i| {
                let c12 = b.const_u32(12);
                let d = b.div(i, c12);
                let r = b.rem(i, c12);
                let ai = b.index(av, &[d, r]);
                b.mul(ai, two)
            })
        });
        let (mut g, ins) = graph_of(b.finish("caseb", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![ins[0]]);
        let r = extract_one(&kn.module, &BTreeMap::new()).unwrap();
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(r.outer_bound, 144);
        assert_eq!(r.reads.len(), 1);
        assert_eq!(r.reads[0].input_pos, 0);
        assert!(r.reads[0].expr.is_some());
    }

    /// Case A into a General consumer: the graft lands inside the reduce
    /// body, with the reduce binder flowing through sigma.
    #[test]
    fn apply_grafts_inside_reduce_consumer() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![24]);
        let two = b.const_field(2);
        let a = b.compute(24, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            b.compute(8, |b, i| {
                b.reduce(ReduceOp::Add, 3, |b, r| {
                    let c3 = b.const_u32(3);
                    let row = b.mul(i, c3);
                    let ix = b.add(row, r);
                    b.index(av, &[ix])
                })
            })
        });
        let (mut g, ins) = graph_of(b.finish("redc", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![ins[0]]);
        let r = extract_one(&kn.module, &BTreeMap::new()).unwrap();
        assert_eq!(r.class, KernelClass::General);
        assert_eq!(r.reads.len(), 1);
        assert_eq!(r.reads[0].inner.len(), 1);
        assert_eq!(r.reads[0].inner[0].1, 3);
    }

    /// The producer's input is not among the consumer's: it is appended to
    /// the merged module and bound to the producer's buffer.
    #[test]
    fn apply_appends_missing_producer_input() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let y = b.input("y", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                let yi = b.index(y, &[i]);
                b.add(ai, yi)
            })
        });
        let (mut g, ins) = graph_of(b.finish("app", body), 2, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.module.builder.inputs().len(), 2);
        let mut got = kn.inputs.clone();
        got.sort_by_key(|b| b.0);
        let mut want = vec![ins[0], ins[1]];
        want.sort_by_key(|b| b.0);
        assert_eq!(got, want);
        let r = extract_one(&kn.module, &BTreeMap::new()).unwrap();
        assert_eq!(r.reads.len(), 2);
    }

    /// The consumer already reads the producer's input: the merged kernel
    /// reuses the existing declaration instead of appending a duplicate.
    #[test]
    fn apply_reuses_shared_consumer_input() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                let xi = b.index(x, &[i]);
                b.add(ai, xi)
            })
        });
        let (mut g, ins) = graph_of(b.finish("share", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![ins[0]]);
        assert_eq!(kn.module.builder.inputs().len(), 1);
        let r = extract_one(&kn.module, &BTreeMap::new()).unwrap();
        // x[i]*2 + x[i]: a single hash-consed load site feeds both terms.
        assert_eq!(r.reads.len(), 1);
    }

    /// A graph buffer holding `elems` 4-byte elements: binding inference on
    /// symbolic modules solves params from these byte sizes.
    fn sized_buf(g: &mut GraphBuilder, name: &str, elems: i64) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(elems * 4),
            elem_size: 4,
        })
    }

    /// Seam unification: a flat symbolic producer whose bound is the bare
    /// param `n` fuses into the symbolic consumer with `n` remapped onto
    /// the consumer's param — the merged module keeps a single param and
    /// the consumer's bindings.
    #[test]
    fn apply_unifies_bare_param_seam() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::BabyBear, vec![n]);
        let two = b.const_field(2);
        let a = b.compute(n, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            b.compute(n, |b, i| {
                let ai = b.index(av, &[i]);
                let c1 = b.const_u32(1);
                let cn = b.const_sym(n);
                let i1 = b.add(i, c1);
                let iw = b.rem(i1, cn);
                let ai1 = b.index(av, &[iw]);
                b.add(ai, ai1)
            })
        });
        let module = b.finish("symchain", body);

        let mut g = GraphBuilder::new();
        let xin = sized_buf(&mut g, "x", 32);
        let out = sized_buf(&mut g, "out", 32);
        g.register_input(xin);
        g.register_output(out);
        g.insert_kernel(module, vec![xin], vec![out], &[]);

        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![xin]);
        assert_eq!(kn.module.builder.params().len(), 1);
        assert_eq!(kn.param_bindings, vec![32]);
        let r = extract_one(&kn.module, &binding_env(kn)).unwrap();
        assert_eq!(r.class, KernelClass::Simple);
        assert_eq!(r.outer_bound, 32);
        assert_eq!(r.reads.len(), 2);
        assert!(r.reads.iter().all(|ra| ra.input_pos == 0));
    }

    /// A producer param the seam cannot unify (compound producer bound) is
    /// appended to the merged module as a fresh param, with its bound value
    /// recorded on the merged node's bindings.
    #[test]
    fn apply_appends_ununified_producer_param() {
        let prod = {
            let mut b = IRBuilder::new();
            let m = b.symbol("m");
            let x = b.input("x", ScalarType::BabyBear, vec![m * 2]);
            let two = b.const_field(2);
            let body = b.compute(m * 2, |b, i| {
                let xi = b.index(x, &[i]);
                b.mul(xi, two)
            });
            b.finish("symprod", body)
        };
        let cons = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![32]);
            let body = b.compute(32, |b, i| {
                let ai = b.index(a, &[i]);
                b.add(ai, ai)
            });
            b.finish("cons", body)
        };
        let mut g = GraphBuilder::new();
        let xin = sized_buf(&mut g, "x", 32);
        let mid = sized_buf(&mut g, "a", 32);
        let out = sized_buf(&mut g, "out", 32);
        g.register_input(xin);
        g.register_output(out);
        g.insert_kernel(prod, vec![xin], vec![mid], &[]);
        g.insert_kernel(cons, vec![mid], vec![out], &[]);

        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        apply_fusion(&mut g, &cands[0]).unwrap();
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(kn.inputs, vec![xin]);
        let params = kn.module.builder.params();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].1, "m");
        assert_eq!(kn.param_bindings, vec![16]);
        // The appended input decl still states its shape through the param.
        assert_eq!(
            kn.module.builder.inputs()[0].shape,
            vec![SizeExpr::cst(SymConst::Sym(params[0].0)).mul_c(SymConst::Lit(2))]
        );
        let r = extract_one(&kn.module, &binding_env(kn)).unwrap();
        assert_eq!(r.outer_bound, 32);
        assert_eq!(r.reads.len(), 1);
        assert_eq!(r.reads[0].expr, Some(Quast::sym(r.outer_var)));
    }

    /// A producer whose `ConstSym` embeds a loop var cannot be grafted (σ
    /// substitution only reaches `Node::Var` sites): apply_fusion vetoes
    /// the candidate.
    #[test]
    fn apply_vetoes_loop_var_in_producer_const_sym() {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::U32, vec![16]);
        let a = b.compute(16, |b, i| {
            let Node::Var(iv) = *b.node(i) else {
                panic!("compute binder is not a var");
            };
            let cs = b.const_sym(SizeExpr::sym(iv));
            let xi = b.index(x, &[i]);
            b.add(xi, cs)
        });
        let body = b.bind(a, |b, av| {
            b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                b.add(ai, ai)
            })
        });
        let (mut g, _) = graph_of(b.finish("loopsym", body), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        let err = apply_fusion(&mut g, &cands[0]).unwrap_err();
        assert!(err.to_string().contains("loop variables"), "{err}");
    }

    /// A candidate whose site ids don't land (simulating a stale relation)
    /// trips the residual-read check and leaves the graph untouched.
    #[test]
    fn apply_failure_leaves_graph_unchanged() {
        let (mut g, _) = graph_of(chain_module(), 1, 1);
        let mut cands = candidates(&g, &FusionOptions::default());
        assert_eq!(cands.len(), 1);
        cands[0].info.rewrites[0].index_node = NodeId(0);
        let before = Arc::as_ptr(&kernel_at(&g, cands[0].dst).module);
        assert!(apply_fusion(&mut g, &cands[0]).is_err());
        let kn = kernel_at(&g, cands[0].dst);
        assert_eq!(Arc::as_ptr(&kn.module), before);
        assert_eq!(kn.inputs.len(), 1);
    }

    // ------------------------------------------------------------------
    // Stage 4: dead-node elimination.
    // ------------------------------------------------------------------

    use crate::graph_ir::{ConstBuf, ConstNode, KernelNode, MemSetNode, MemcpyNode};

    /// Unregistered (hence internal / optimizable) 64-byte device buffer.
    fn buf64(g: &mut GraphBuilder, name: &str) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(64),
            elem_size: 4,
        })
    }

    fn memset(buf: BufId) -> GraphNode {
        GraphNode::Memset(MemSetNode {
            node: buf,
            offset: Quast::cst(0),
            num_bytes: Quast::cst(64),
            val: 0,
        })
    }

    /// After fusing the chain, the producer's only reader is gone: DCE
    /// drops it and keeps just the merged kernel.
    #[test]
    fn dce_removes_fused_away_producer() {
        let (mut g, ins) = graph_of(chain_module(), 1, 1);
        let cands = candidates(&g, &FusionOptions::default());
        apply_fusion(&mut g, &cands[0]).unwrap();
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(dce(&mut g), 1);
        assert_eq!(g.nodes.len(), 1);
        let kn = kernel_at(&g, 0);
        assert_eq!(kn.inputs, vec![ins[0]]);
        assert!(kn.module.name.starts_with("fused_"));
        // Fusion history: one Fused root wrapping two Kernel leaves that
        // preserve the pre-fusion module names.
        let history = kn
            .fusion_history
            .as_ref()
            .expect("fused kernel has history");
        assert_eq!(history.leaf_names().len(), 2);
    }

    /// Before fusion the producer's internal buffer is still read: nothing
    /// is dead.
    #[test]
    fn dce_keeps_live_chain() {
        let (mut g, _) = graph_of(chain_module(), 1, 1);
        assert_eq!(g.nodes.len(), 2);
        assert_eq!(dce(&mut g), 0);
        assert_eq!(g.nodes.len(), 2);
    }

    /// Blackbox kernels have opaque effects: live even when no written
    /// buffer is needed, and their reads keep upstream writers alive.
    #[test]
    fn dce_keeps_blackbox_and_its_upstream() {
        let mut g = GraphBuilder::new();
        let a = buf64(&mut g, "a");
        g.nodes.push(memset(a));
        g.nodes.push(GraphNode::BlackboxKernel(KernelNode {
            inputs: vec![a],
            outputs: vec![],
            carried_outputs: vec![],
            func: Box::new(|_, _, _| {}),
            name: "bb".into(),
        }));
        assert_eq!(dce(&mut g), 0);
    }

    /// A write chain ending in an unread internal buffer dies wholesale:
    /// the dead memcpy never marks its source needed.
    #[test]
    fn dce_removes_dead_write_chain() {
        let mut g = GraphBuilder::new();
        let a = buf64(&mut g, "a");
        let b = buf64(&mut g, "b");
        g.nodes.push(GraphNode::Const(ConstNode {
            buf: a,
            data: ConstBuf::HostBuf(vec![0; 64]),
        }));
        g.nodes.push(GraphNode::Memcpy(MemcpyNode {
            src: a,
            src_offset: Quast::cst(0),
            dst: b,
            dst_offset: Quast::cst(0),
            num_bytes: Quast::cst(64),
        }));
        assert_eq!(dce(&mut g), 2);
        assert!(g.nodes.is_empty());
    }

    /// Needed bits never clear: every writer of a needed buffer stays live
    /// (a later write may only cover part of the buffer).
    #[test]
    fn dce_keeps_all_writers_of_needed_buf() {
        let mut g = GraphBuilder::new();
        let a = buf64(&mut g, "a");
        let out = buf64(&mut g, "out");
        g.register_output(out);
        g.nodes.push(memset(a));
        g.nodes.push(memset(a));
        g.nodes.push(GraphNode::Memcpy(MemcpyNode {
            src: a,
            src_offset: Quast::cst(0),
            dst: out,
            dst_offset: Quast::cst(0),
            num_bytes: Quast::cst(64),
        }));
        assert_eq!(dce(&mut g), 0);
        assert_eq!(g.nodes.len(), 3);
    }

    /// Writers of registered interface buffers are always live: the caller
    /// observes those contents after the graph runs. The same buffer left
    /// unregistered is internal, so its writer dies.
    #[test]
    fn dce_keeps_writer_of_registered_output() {
        let mut g = GraphBuilder::new();
        let out = buf64(&mut g, "out");
        g.register_output(out);
        g.nodes.push(memset(out));
        assert_eq!(dce(&mut g), 0);

        let mut g = GraphBuilder::new();
        let out = buf64(&mut g, "out");
        g.nodes.push(memset(out));
        assert_eq!(dce(&mut g), 1);
    }

    // ------------------------------------------------------------------
    // Stage 5: fixpoint driver.
    // ------------------------------------------------------------------

    /// `a = x*2; b = a+1; out = b*3` — three chained elementwise kernels.
    fn triple_chain_module() -> Module {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let a = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        let body = b.bind(a, |b, av| {
            let one = b.const_field(1);
            let mid = b.compute(16, |b, i| {
                let ai = b.index(av, &[i]);
                b.add(ai, one)
            });
            b.bind(mid, |b, mv| {
                let three = b.const_field(3);
                b.compute(16, |b, i| {
                    let mi = b.index(mv, &[i]);
                    b.mul(mi, three)
                })
            })
        });
        b.finish("triple", body)
    }

    #[test]
    fn fuse_graph_fuses_chain() {
        let (mut g, ins) = graph_of(chain_module(), 1, 1);
        let report = fuse_graph(&mut g, &FusionOptions::default());
        assert_eq!(report.rounds, 1);
        assert_eq!(report.fused.len(), 1);
        assert_eq!(report.nodes_before, 2);
        assert_eq!(report.nodes_after, 1);
        let kn = kernel_at(&g, 0);
        assert_eq!(kn.inputs, vec![ins[0]]);
        assert!(kn.module.name.starts_with("fused_"));
        // Fusion history: one Fused root wrapping two Kernel leaves that
        // preserve the pre-fusion module names.
        let history = kn
            .fusion_history
            .as_ref()
            .expect("fused kernel has history");
        assert_eq!(history.leaf_names().len(), 2);
    }

    /// Fused kernels surface their fusion history in the cytoscape dump:
    /// a `Fused` root with `consumer` / `producer` children, each a leaf
    /// carrying the original pre-fusion module name and shape strings.
    /// Registered graph inputs/outputs also get synthetic Input/Output
    /// endpoints so the dump makes the graph interface visible.
    #[test]
    fn cytoscape_dump_includes_fusion_history_and_neighbors() {
        let (mut g, _) = graph_of(triple_chain_module(), 1, 1);
        let report = fuse_graph(&mut g, &FusionOptions::default());
        assert_eq!(report.fused.len(), 2);
        assert_eq!(g.nodes.len(), 1);
        let json = g.to_cytoscape_json();
        assert!(
            json.contains(r#""fusion_history":{"kind":"fused""#),
            "expected fused root in dump: {json}"
        );
        assert!(
            json.contains(r#""kind":"leaf""#),
            "expected leaf entries in dump: {json}"
        );
        assert!(
            json.contains(r#""consumer":{"kind":"#) && json.contains(r#""producer":{"kind":"#),
            "expected consumer/producer children in dump: {json}"
        );
        // Every kernel/blackbox node carries the new metadata arrays.
        assert!(json.contains(r#""input_shapes":["#));
        assert!(json.contains(r#""output_shapes":["#));
        assert!(json.contains(r#""producer_ids":["#));
        assert!(json.contains(r#""consumer_ids":["#));
        assert!(json.contains(r#""producer_names":["#));
        assert!(json.contains(r#""consumer_names":["#));
        // Registered graph interface: one Input synthetic (the sole `in0`
        // buffer) and one Output synthetic (the sole `out0` buffer).
        assert!(
            json.contains(r#""type":"Input""#),
            "expected an Input synthetic node in dump: {json}"
        );
        assert!(
            json.contains(r#""type":"Output""#),
            "expected an Output synthetic node in dump: {json}"
        );
    }

    /// A three-kernel chain needs two rounds: the two candidates share the
    /// middle node, so greedy selection applies one per round.
    #[test]
    fn fuse_graph_three_kernel_chain_two_rounds() {
        let (mut g, _) = graph_of(triple_chain_module(), 1, 1);
        assert_eq!(g.nodes.len(), 3);
        let report = fuse_graph(&mut g, &FusionOptions::default());
        assert_eq!(report.rounds, 2);
        assert_eq!(report.fused.len(), 2);
        assert_eq!(report.nodes_after, 1);
        assert_eq!(g.nodes.len(), 1);
    }

    /// `max_iterations` caps the rounds even when candidates remain.
    #[test]
    fn fuse_graph_honors_max_iterations() {
        let (mut g, _) = graph_of(triple_chain_module(), 1, 1);
        let opts = FusionOptions {
            max_iterations: 1,
            ..FusionOptions::default()
        };
        let report = fuse_graph(&mut g, &opts);
        assert_eq!(report.rounds, 1);
        assert_eq!(report.fused.len(), 1);
        assert_eq!(g.nodes.len(), 2);
    }

    /// The initial DCE unblocks fusion: with a dead second reader still
    /// present, enumeration would inflate `n_readers` (shrinking each
    /// candidate's amortized share) and waste work on a producer whose
    /// only real consumer is the live one.
    #[test]
    fn fuse_graph_dces_dead_reader_first() {
        let (mut g, _) = graph_of(chain_module(), 1, 1);
        let mid = (0..g.bufs.len())
            .map(BufId)
            .find(|&b| !g.buf_is_interface(b))
            .unwrap();
        let dead = buf64(&mut g, "dead");
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![16]);
        let three = b.const_field(3);
        let body = b.compute(16, |b, i| {
            let ai = b.index(a, &[i]);
            b.mul(ai, three)
        });
        g.insert_kernel(b.finish("dead_reader", body), [mid], [dead], &[]);
        // Sanity: with the dead reader present, `mid` has two readers, so
        // enumeration emits a candidate per reader — including one whose
        // dst is itself unreachable. `fuse_graph`'s initial DCE avoids all
        // of that by dropping the dead reader before enumeration runs.
        assert_eq!(candidates(&g, &FusionOptions::default()).len(), 2);
        let report = fuse_graph(&mut g, &FusionOptions::default());
        assert_eq!(report.nodes_before, 3);
        assert_eq!(report.fused.len(), 1);
        assert_eq!(g.nodes.len(), 1);
    }

    // ------------------------------------------------------------------
    // Stage 6: post-fusion module deduplication.
    // ------------------------------------------------------------------

    /// One-kernel module `out = x * 2` with a shifted var counter.
    fn scale_module_shifted(shift: u32) -> Module {
        let mut b = IRBuilder::new();
        for _ in 0..shift {
            b.fresh_var();
        }
        let x = b.input("x", ScalarType::BabyBear, vec![16]);
        let two = b.const_field(2);
        let body = b.compute(16, |b, i| {
            let xi = b.index(x, &[i]);
            b.mul(xi, two)
        });
        b.finish("scale", body)
    }

    #[test]
    fn renumber_normalizes_alpha_variants() {
        let a = chain_module();
        let b = chain_module_shifted(7);
        assert_ne!(module_hash(&a), module_hash(&b));
        let (ra, rb) = (renumber_module(&a), renumber_module(&b));
        assert_eq!(module_hash(&ra), module_hash(&rb));
        // Idempotent: renumbering a canonical module changes nothing.
        assert_eq!(module_hash(&ra), module_hash(&renumber_module(&ra)));
    }

    /// Scatter params, exprs and the compute binder renumber too.
    #[test]
    fn renumber_covers_scatter_modules() {
        let build = |shift: u32| {
            let mut b = IRBuilder::new();
            for _ in 0..shift {
                b.fresh_var();
            }
            let x = b.input("x", ScalarType::BabyBear, vec![16]);
            let sc = b.scatter_map(
                1,
                None,
                |p, _c| vec![p[0].rem_c(8).mul_c(2).add(&p[0].floordiv(8))],
                |q, _c| vec![q[0].rem_c(2).mul_c(8).add(&q[0].floordiv(2))],
            );
            let body = b.compute_scatter(16, sc, |b, i| b.index(x, &[i]));
            b.finish("scat", body)
        };
        let (a, b) = (build(0), build(11));
        assert_ne!(module_hash(&a), module_hash(&b));
        assert_eq!(
            module_hash(&renumber_module(&a)),
            module_hash(&renumber_module(&b))
        );
    }

    /// Module parameters (in bounds and input shapes) renumber too.
    #[test]
    fn renumber_covers_symbolic_params() {
        let build = |shift: u32| {
            let mut b = IRBuilder::new();
            for _ in 0..shift {
                b.fresh_var();
            }
            let n = b.symbol("n");
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                b.mul(xi, xi)
            });
            b.finish("sym_scale", body)
        };
        let (a, b) = (build(0), build(9));
        assert_ne!(module_hash(&a), module_hash(&b));
        let (ra, rb) = (renumber_module(&a), renumber_module(&b));
        assert_eq!(module_hash(&ra), module_hash(&rb));
        assert_eq!(module_hash(&ra), module_hash(&renumber_module(&ra)));
        // The remapped param is still declared and still referenced by the
        // input shape.
        assert_eq!(ra.builder.params().len(), 1);
        let pv = ra.builder.params()[0].0;
        let mut syms = std::collections::BTreeSet::new();
        ra.builder.inputs()[0].shape[0].param_syms(&mut syms);
        assert_eq!(syms.into_iter().collect::<Vec<_>>(), vec![pv]);
    }

    /// `Node::ConstSym` participates in α-renumbering and hashing: the
    /// embedded parameter syms remap together with the module params, so
    /// shifted α-variants converge to one hash.
    #[test]
    fn renumber_covers_const_sym() {
        let build = |shift: u32| {
            let mut b = IRBuilder::new();
            for _ in 0..shift {
                b.fresh_var();
            }
            let n = b.symbol("n");
            let x = b.input("x", ScalarType::U32, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                let c = b.const_sym(n - 1);
                b.mul(xi, c)
            });
            b.finish("sym_splice", body)
        };
        let (a, b) = (build(0), build(9));
        assert_ne!(module_hash(&a), module_hash(&b));
        let (ra, rb) = (renumber_module(&a), renumber_module(&b));
        assert_eq!(module_hash(&ra), module_hash(&rb));
        assert_eq!(module_hash(&ra), module_hash(&renumber_module(&ra)));
    }

    #[test]
    fn dedup_modules_folds_alpha_variants() {
        let mut g = GraphBuilder::new();
        let (i0, o0) = (buf64(&mut g, "i0"), buf64(&mut g, "o0"));
        let (i1, o1) = (buf64(&mut g, "i1"), buf64(&mut g, "o1"));
        g.register_input(i0);
        g.register_input(i1);
        g.register_output(o0);
        g.register_output(o1);
        g.insert_kernel(scale_module_shifted(0), [i0], [o0], &[]);
        g.insert_kernel(scale_module_shifted(5), [i1], [o1], &[]);
        // α-variants hash differently, so insertion-time dedup missed.
        assert!(!Arc::ptr_eq(
            &kernel_at(&g, 0).module,
            &kernel_at(&g, 1).module
        ));
        assert_eq!(dedup_modules(&mut g), 1);
        assert!(Arc::ptr_eq(
            &kernel_at(&g, 0).module,
            &kernel_at(&g, 1).module
        ));
        // Idempotent.
        assert_eq!(dedup_modules(&mut g), 0);
    }

    /// Two α-variant chains fused at different graph locations produce
    /// α-equivalent fused modules; the final sweep folds them onto one
    /// `Arc` and reports it.
    #[test]
    fn fuse_graph_dedups_alpha_variant_fusions() {
        let mut g = GraphBuilder::new();
        let (i0, o0) = (buf64(&mut g, "i0"), buf64(&mut g, "o0"));
        let (i1, o1) = (buf64(&mut g, "i1"), buf64(&mut g, "o1"));
        g.register_input(i0);
        g.register_input(i1);
        g.register_output(o0);
        g.register_output(o1);
        g.insert_kernel(chain_module(), [i0], [o0], &[]);
        g.insert_kernel(chain_module_shifted(9), [i1], [o1], &[]);
        let report = fuse_graph(&mut g, &FusionOptions::default());
        assert_eq!(report.nodes_before, 4);
        assert_eq!(report.fused.len(), 2);
        assert_eq!(report.nodes_after, 2);
        assert_eq!(report.deduped, 1);
        assert!(Arc::ptr_eq(
            &kernel_at(&g, 0).module,
            &kernel_at(&g, 1).module
        ));
    }
}
