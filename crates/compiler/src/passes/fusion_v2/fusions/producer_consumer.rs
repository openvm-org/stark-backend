//! Producer-consumer fusion — drop-seams (M3) and keep-seams (M5) variants.
//!
//! `detailed-fusion-plan-v2.md` §10.1 (drop) and §10.2 (keep). The pass
//! merges a producer kernel into a consumer kernel by inlining the
//! producer's result expression at every consumer read of the seam.
//!
//! **Drop variant** (§10.1) — `outputs = consumer.outputs`. The seam is
//! no longer materialized; if some other node still needs the seam, the
//! ILP must select a separate producer for it.
//!
//! **Keep variant** (§10.2) — `outputs = consumer.outputs ++ [seam]`.
//! The fused module returns the producer expression as an additional
//! top-level output alongside the consumer's outputs. Both share the
//! consumer's compute domain, so one single `Compute` with a `Tuple`
//! body produces both — no second launch. The keep candidate is
//! emitted only when at least one of the following holds (§10.2):
//!
//! - the seam is a registered graph output;
//! - the seed graph has another original consumer of the seam that is not the current consumer;
//! - the caller requests all keep variants via `FusionOptionsV2::enable_all_keep_variants`.
//!
//! Supported cases (both variants):
//!
//! - both producer and consumer are pure structured kernels;
//! - both are single-`compute` modules with a scalar body (no scatter, no `par`, no `threads`
//!   hint);
//! - the producer body may be a plain scalar expression, an inline `Let` chain, or a `Reduce`
//!   (which evaluates to one scalar per outer iteration — the "reduction producer" case);
//! - the consumer body may be flat or contain an inner `Compute` / `Reduce` — reads of the seam are
//!   detected regardless of their scope, as long as their index expression is a quasi-affine
//!   function of the enclosing loop variables (**identity, affine permutation, and nested-index**
//!   cases);
//! - the two kernels share the same concrete outer bound;
//! - the seam value has exactly one producer output position (single output on the producer
//!   kernel).
//!
//! Synthesis is a capture-free HIR clone. The consumer's compute body is
//! cloned into a fresh module. At every seam-read site the producer body
//! is *re-cloned* with the producer's outer variable substituted by the
//! quasi-affine index expression that appeared at that consumer read
//! (evaluated as a fresh HIR expression in the fused builder).
//!
//! The multi-seam case (§10.1 grouping) is deferred: it requires
//! multi-output producers, which live outside this M3 slice.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    graph_ir::{GraphNode, KernelModuleNode},
    ir::{IRBuilder, Module, Node, NodeId as HirNodeId, SizeExpr, VarId},
    module_hash::children_of,
    passes::{
        fusion_utils::{clone_expr, clone_expr_with_hook, CloneError},
        fusion_v2::model::{AltGraphNode, GraphFuser, NodeId, ValueClassId},
        utils::hir_to_quast,
    },
    quast::{Quast, QuastEmitter},
    CompileError,
};

/// Whether the fused module materializes the seam value (§10.2) or
/// drops it entirely (§10.1).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FusionVariant {
    /// `outputs = consumer.outputs`; seam is not materialized.
    Drop,
    /// `outputs = consumer.outputs ++ [seam]`; seam is emitted as an
    /// additional top-level output.
    Keep,
}

/// A candidate produced by a fusion pass, not yet registered in the
/// alternative graph.
///
/// The saturation driver validates the candidate, allocates a
/// [`NodeId`], appends it to the fuser, and updates the sidecars. The
/// pass itself must NOT hold or allocate any [`NodeId`]s.
pub struct CandidateDraft {
    /// The alternative-graph nodes that this candidate consumes. Used
    /// to enforce disjoint origins in the saturation driver.
    pub parents: Vec<NodeId>,
    /// Which variant the candidate represents. Diagnostic only; the ILP
    /// sees the same alternative-graph shape regardless.
    pub variant: FusionVariant,
    /// The finalized alternative-graph entry, ready to be pushed.
    pub alt: AltGraphNode,
}

/// Structural facts about a producer or consumer kernel module used to
/// synthesize a fused module.
///
/// The recognizer covers the M3 exit-gate cases:
///
/// - **identity**: `y[outer_var]`;
/// - **affine permutation**: `y[a*outer_var + b]`, `y[N-1-outer_var]`, etc. — any quasi-affine
///   function of the enclosing loop variables;
/// - **nested-index consumers**: consumer body may contain an inner `Compute` or `Reduce`; reads
///   inside that inner scope are captured with their index-scope (`inner_vars`) recorded;
/// - **reduction producers**: the producer body is a `Reduce` (single scalar per outer iteration),
///   which we detect but do not treat specially — the fused synthesis clones the reduce
///   sub-expression at every seam read like any other body expression.
#[derive(Debug, Clone)]
struct KernelShape {
    /// Fresh [`VarId`] of the outer `compute` iteration variable.
    outer_var: VarId,
    /// Symbolic outer bound.
    outer_bound: SizeExpr,
    /// The scalar body expression (per-outer-iteration return value).
    body_root: HirNodeId,
    /// Every `Node::Index` site reachable from `body_root` whose
    /// `tensor` is a `Node::Input(_)` and whose indices form a
    /// quasi-affine expression in `outer_var` and any inner-scope
    /// binders.
    reads: Vec<ReadSite>,
}

#[derive(Debug, Clone)]
struct ReadSite {
    /// Module input position this site reads.
    input_pos: usize,
    /// [`Node::Index`] site.
    index_node: HirNodeId,
    /// The Quast that names the read index (single-index tensors only).
    /// `None` for multi-index reads, which are conservatively skipped:
    /// they can only be a seam if the read is not inside an inner loop
    /// scope, and the fused synthesis path currently emits scalars.
    index_expr: Option<Quast>,
    /// In-scope binders at this read site, outer-first. Each entry is
    /// the source module's [`VarId`] and its concrete bound.
    inner_vars: Vec<VarId>,
}

/// Recognizes a producer- or consumer-shaped kernel module. Returns
/// `None` if the module does not fit any of the supported shapes.
fn identify_kernel_shape(module: &Module) -> Option<KernelShape> {
    let (outer_var, outer_bound, body_root) = match module.builder.node(module.body) {
        Node::Compute {
            bound,
            var,
            body,
            scatter,
            par,
            threads,
        } => {
            if scatter.is_some() || par.is_some() || threads.is_some() {
                return None;
            }
            (*var, bound.clone(), *body)
        }
        _ => return None,
    };

    // Walk the body, recording read sites and the scope they occur in.
    // We track scope as a stack of (VarId, ...) — only Compute/Reduce
    // bodies push. Let bindings do not extend the index scope for
    // quasi-affine purposes but their local values are chased inline
    // via `hir_to_quast`'s `lets` argument.
    let mut reads: Vec<ReadSite> = Vec::new();
    let mut input_reached_via_index: HashMap<HirNodeId, ()> = HashMap::new();
    let mut seen: std::collections::HashSet<HirNodeId> = std::collections::HashSet::new();
    // (node, scope) — scope is the current stack of in-scope loop
    // variables, outer-first.
    let mut work: Vec<(HirNodeId, Vec<VarId>)> = vec![(body_root, vec![outer_var])];
    while let Some((id, scope)) = work.pop() {
        // Occurrence-based visit — we do NOT dedup on NodeId because a
        // hash-consed sub-expression could appear under different scopes.
        // But we still guard against infinite recursion on malformed IR
        // via a per-frame seen set at the *same* scope; for well-formed
        // HIR this collapses to a normal DAG walk.
        let node = module.builder.node(id);
        if let Node::Index { tensor, indices } = node {
            if let Node::Input(k) = module.builder.node(*tensor) {
                input_reached_via_index.insert(*tensor, ());
                // Attempt to convert the single-index expression to a
                // Quast in the source module's variable namespace.
                let (index_expr, ok) = if indices.len() == 1 {
                    let syms =
                        |v: VarId| -> Option<Quast> { scope.contains(&v).then(|| Quast::sym(v)) };
                    let lets = |_v: VarId| -> Option<HirNodeId> { None };
                    let expr = hir_to_quast(&module.builder, indices[0], &syms, &lets).ok();
                    (expr, true)
                } else {
                    // Multi-index reads (tensor with rank > 1) — reject
                    // by declining to record an index expression. The
                    // synthesis path treats absent index_expr as
                    // "cannot fuse through this read".
                    (None, false)
                };
                if ok {
                    reads.push(ReadSite {
                        input_pos: *k,
                        index_node: id,
                        index_expr,
                        inner_vars: scope.clone(),
                    });
                }
            }
        }
        if !seen.insert(id) {
            continue;
        }
        // Descend into children. For Compute/Reduce push their bound
        // variable onto the scope of the child; other nodes keep the
        // outer scope.
        match node {
            Node::Compute { var, body, .. } | Node::Reduce { var, body, .. } => {
                let mut inner = scope.clone();
                inner.push(*var);
                work.push((*body, inner));
                // The bound expression is a size, not an HIR NodeId,
                // so no further descent needed.
            }
            _ => {
                for c in children_of(node) {
                    work.push((c, scope.clone()));
                }
            }
        }
    }
    // Every reachable Input(_) must occur as the tensor operand of at
    // least one Index; otherwise the module uses the tensor in a way
    // the fused synthesis does not know how to rewrite (e.g. Proj,
    // Tuple, passed through Let).
    for id in &seen {
        let node = module.builder.node(*id);
        if matches!(node, Node::Input(_)) && !input_reached_via_index.contains_key(id) {
            return None;
        }
    }

    Some(KernelShape {
        outer_var,
        outer_bound,
        body_root,
        reads,
    })
}

/// Failure modes when trying to synthesize a fused module from
/// `(producer, consumer)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynthesisFailure {
    /// One of the parents was not a `GraphNode::Kernel`.
    NotAKernel,
    /// One of the parent modules did not fit the supported shape.
    UnsupportedShape,
    /// Producer and consumer have different outer bounds.
    OuterBoundMismatch,
    /// The producer produces zero or several outputs. The M3 slice
    /// only supports single-output producers.
    ProducerNotSingleOutput,
    /// No consumer read site references the seam value.
    NoSeamReadInConsumer,
    /// A consumer read of the seam has an index expression that is not
    /// a quasi-affine function of the enclosing scope.
    SeamIndexNotAffine,
    /// Clone-time failure — internal compiler error.
    CloneError(String),
    /// Post-synthesis type inference rejected the module.
    TypeCheckFailed(String),
}

/// Synthesizes a fused `Kernel` node that replaces `producer_node` and
/// `consumer_node` on a single seam value.
///
/// `seam_val` is the seam [`ValueClassId`] produced by `producer_node`
/// and read by `consumer_node` at at least one input position. The
/// synthesized module's boundary is: producer inputs first (unique
/// positional order), then the consumer's non-seam inputs. Outputs are
/// determined by `variant`:
///
/// - [`FusionVariant::Drop`] — outputs are the consumer's outputs only (§10.1);
/// - [`FusionVariant::Keep`] — outputs are consumer outputs followed by the seam value; the fused
///   compute body wraps the consumer and producer expressions in a `Tuple` so both are emitted from
///   the same iteration (§10.2).
///
/// The producer body is *re-cloned* at each seam-read site with
/// the producer's outer variable substituted by that read's index
/// expression, so identity, affine-permutation, and nested-index
/// consumers all reduce to the same synthesis loop.
pub fn synthesize_producer_consumer(
    gf: &GraphFuser,
    producer_node: NodeId,
    consumer_node: NodeId,
    seam_val: ValueClassId,
    variant: FusionVariant,
) -> Result<CandidateDraft, SynthesisFailure> {
    let p_alt = &gf.nodes[producer_node.0];
    let c_alt = &gf.nodes[consumer_node.0];
    let (p_module, p_binding) = match &p_alt.node {
        GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
        _ => return Err(SynthesisFailure::NotAKernel),
    };
    let (c_module, c_binding) = match &c_alt.node {
        GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
        _ => return Err(SynthesisFailure::NotAKernel),
    };

    let p_shape = identify_kernel_shape(&p_module).ok_or(SynthesisFailure::UnsupportedShape)?;
    let c_shape = identify_kernel_shape(&c_module).ok_or(SynthesisFailure::UnsupportedShape)?;

    if p_alt.outputs.len() != 1 {
        return Err(SynthesisFailure::ProducerNotSingleOutput);
    }
    if p_shape.outer_bound != c_shape.outer_bound {
        return Err(SynthesisFailure::OuterBoundMismatch);
    }

    // Consumer input position(s) at which the seam is bound.
    let seam_positions: Vec<usize> = c_alt
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == seam_val).then_some(i))
        .collect();
    if seam_positions.is_empty() {
        return Err(SynthesisFailure::NoSeamReadInConsumer);
    }
    // Every seam-read site in the consumer must have an analyzable
    // (quasi-affine) index expression, or fusion through it isn't
    // supported.
    let seam_reads: Vec<&ReadSite> = c_shape
        .reads
        .iter()
        .filter(|r| seam_positions.contains(&r.input_pos))
        .collect();
    if seam_reads.iter().any(|r| r.index_expr.is_none()) {
        return Err(SynthesisFailure::SeamIndexNotAffine);
    }

    // Fused-module boundary: producer inputs first, followed by consumer
    // inputs whose positions are not seam positions.
    let mut fb = IRBuilder::new();
    let mut merged_bindings = p_binding.clone();
    for (name, val) in &c_binding {
        match merged_bindings.get(name) {
            Some(existing) if existing != val => {
                return Err(SynthesisFailure::UnsupportedShape);
            }
            _ => {
                merged_bindings.insert(name.clone(), *val);
            }
        }
    }
    // Track producer/consumer param VarIds mapped into fb, so
    // ConstSym / shape expressions can be alpha-renamed.
    let mut param_map: HashMap<VarId, VarId> = HashMap::new();
    let mut seen_names: HashMap<String, VarId> = HashMap::new();
    for (v, name) in p_module.builder.params() {
        let fresh = *seen_names.entry(name.clone()).or_insert_with(|| {
            let n = fb.var_watermark();
            fb.raise_var_watermark(n + 1);
            fb.inherit_param(VarId(n), name.clone());
            VarId(n)
        });
        param_map.insert(*v, fresh);
    }
    for (v, name) in c_module.builder.params() {
        let fresh = *seen_names.entry(name.clone()).or_insert_with(|| {
            let n = fb.var_watermark();
            fb.raise_var_watermark(n + 1);
            fb.inherit_param(VarId(n), name.clone());
            VarId(n)
        });
        param_map.insert(*v, fresh);
    }

    // Declare inputs and remember their Input NodeIds in fb.
    let mut fused_p_input_nodes: Vec<HirNodeId> = Vec::new();
    let mut fused_c_input_nodes: Vec<Option<HirNodeId>> =
        vec![None; c_module.builder.inputs().len()];

    for decl in p_module.builder.inputs().iter() {
        let shape: Vec<SizeExpr> = decl
            .shape
            .iter()
            .map(|d| remap_size_expr(d, &param_map))
            .collect();
        let n = fb.input(decl.name.clone(), decl.elem, shape);
        fused_p_input_nodes.push(n);
    }
    for (i, decl) in c_module.builder.inputs().iter().enumerate() {
        if seam_positions.contains(&i) {
            continue;
        }
        let shape: Vec<SizeExpr> = decl
            .shape
            .iter()
            .map(|d| remap_size_expr(d, &param_map))
            .collect();
        let n = fb.input(decl.name.clone(), decl.elem, shape);
        fused_c_input_nodes[i] = Some(n);
    }

    // Pre-compute the maps needed by clone_expr, keyed on the *source*
    // modules' Input NodeIds. These maps are stable across
    // producer-body inlines because each seam read only differs in the
    // *index expression* substituted for `p_shape.outer_var`.
    let consumer_input_nodes: HashMap<usize, HirNodeId> =
        find_input_nodes(&c_module, c_shape.body_root);
    let producer_input_nodes: HashMap<usize, HirNodeId> =
        find_input_nodes(&p_module, p_shape.body_root);

    let mut producer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
    for (i, &p_input_node) in &producer_input_nodes {
        producer_subst.insert(p_input_node, fused_p_input_nodes[*i]);
    }

    // Build the fused compute by hand so error propagation from the
    // body-construction closure is straightforward. `k_var` is the
    // fused module's outer iteration variable; `k_var_node` its
    // `Node::Var` NodeId.
    let outer_bound_fb = remap_size_expr(&c_shape.outer_bound, &param_map);
    let k_var = {
        let n = fb.var_watermark();
        fb.raise_var_watermark(n + 1);
        VarId(n)
    };
    let k_var_node = fb.intern(Node::Var(k_var));

    // Consumer non-seam Input NodeIds → their fused Input NodeIds.
    let mut consumer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
    for (i, &c_input_node) in &consumer_input_nodes {
        if let Some(fused_input) = fused_c_input_nodes[*i] {
            consumer_subst.insert(c_input_node, fused_input);
        }
    }

    // Consumer VarId → fused NodeId. Outer var fuses to `k_var_node`;
    // inner Compute/Reduce bound vars are introduced by `clone_expr`
    // itself when it descends. Parameter VarIds map to their remapped
    // fused-module Sym nodes.
    let mut consumer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
    consumer_vars.insert(c_shape.outer_var, k_var_node);
    for (from, to) in &param_map {
        let dst = fb.intern(Node::Var(*to));
        consumer_vars.insert(*from, dst);
    }

    // Build a fast lookup from seam-read NodeId to (index_expr, scope).
    let seam_read_by_node: HashMap<HirNodeId, &ReadSite> =
        seam_reads.iter().map(|r| (r.index_node, *r)).collect();

    // Substitute at every seam read via a clone-time hook. The hook
    // fires when clone_expr visits the seam Index NodeId, with the
    // current `vars` snapshot — which includes the fresh identities
    // clone_expr has already allocated for any inner Compute/Reduce
    // binders. That means nested-index seam reads are handled the
    // same way as top-level ones.
    let hook = |dst: &mut IRBuilder,
                src_id: HirNodeId,
                vars: &HashMap<VarId, HirNodeId>|
     -> Result<Option<HirNodeId>, CloneError> {
        let Some(read) = seam_read_by_node.get(&src_id) else {
            return Ok(None);
        };
        let sigma = read
            .index_expr
            .as_ref()
            .expect("seam_reads was filtered to Some(index_expr)");
        // Build the emit env: every in-scope source VarId (outer +
        // inner binders currently on `vars`) maps to its
        // destination NodeId.
        let mut env: HashMap<VarId, HirNodeId> = HashMap::new();
        for &v in &read.inner_vars {
            let n = *vars.get(&v).ok_or(CloneError::UnboundVar { var: v })?;
            env.insert(v, n);
        }
        let bounds: BTreeMap<VarId, u64> = read.inner_vars.iter().map(|&v| (v, u64::MAX)).collect();
        let sigma_node = {
            let mut em = IndexEmitter { b: dst, env: &env };
            sigma
                .emit(&bounds, &mut em)
                .expect("index emit succeeds on quasi-affine expressions")
        };
        // Clone the producer body with producer.outer_var → sigma_node.
        let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
        producer_vars.insert(p_shape.outer_var, sigma_node);
        for (from, to) in &param_map {
            let n = dst.intern(Node::Var(*to));
            producer_vars.insert(*from, n);
        }
        let inlined = clone_expr(
            &p_module,
            p_shape.body_root,
            dst,
            &producer_subst,
            &producer_vars,
        )?;
        Ok(Some(inlined))
    };

    let cloned_body = clone_expr_with_hook(
        &c_module,
        c_shape.body_root,
        &mut fb,
        &consumer_subst,
        &consumer_vars,
        hook,
    )
    .map_err(|e| SynthesisFailure::CloneError(format!("{e:?}")))?;

    // For the keep variant, materialize the seam alongside the consumer
    // output. Both share the fused compute's outer var, so a single
    // Compute with a `Tuple` body suffices (§10.2). Producer inputs and
    // producer-var mapping match the hook's inline substitution, but the
    // outer-var image is `k_var_node` directly (identity access at the
    // materialized index).
    let compute_body = match variant {
        FusionVariant::Drop => cloned_body,
        FusionVariant::Keep => {
            let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
            producer_vars.insert(p_shape.outer_var, k_var_node);
            for (from, to) in &param_map {
                let dst = fb.intern(Node::Var(*to));
                producer_vars.insert(*from, dst);
            }
            let seam_body = clone_expr(
                &p_module,
                p_shape.body_root,
                &mut fb,
                &producer_subst,
                &producer_vars,
            )
            .map_err(|e| SynthesisFailure::CloneError(format!("{e:?}")))?;
            fb.intern(Node::Tuple(vec![cloned_body, seam_body]))
        }
    };

    let fused_body_id = fb.intern(Node::Compute {
        bound: outer_bound_fb,
        var: k_var,
        body: compute_body,
        scatter: None,
        par: None,
        threads: None,
    });
    // Use a canonical name for fused modules so structurally-identical
    // compositions (e.g. `(A+B)+C` and `A+(B+C)`) hash to the same
    // `module_hash` and dedup at `CandidateKey` time in M6 saturation.
    // The variant is part of the name because drop and keep have
    // different bodies and correctly hash to different artifacts.
    let name = match variant {
        FusionVariant::Drop => "fused_drop",
        FusionVariant::Keep => "fused_keep",
    };
    let fused_module = fb.finish(name.to_string(), fused_body_id);

    // Type-check the synthesized module before wrapping it into an
    // AltGraphNode. Any failure here is a synthesis bug in this pass.
    crate::passes::type_infer(&fused_module).map_err(|e| match e {
        CompileError::Type(m) => SynthesisFailure::TypeCheckFailed(m),
        other => SynthesisFailure::TypeCheckFailed(other.to_string()),
    })?;

    // Boundary value bindings for the AltGraphNode: producer inputs
    // (mapped through p_alt.inputs) followed by consumer non-seam
    // inputs (mapped through c_alt.inputs).
    let mut fused_inputs: Vec<ValueClassId> = p_alt.inputs.clone();
    for (i, &v) in c_alt.inputs.iter().enumerate() {
        if !seam_positions.contains(&i) {
            fused_inputs.push(v);
        }
    }
    let mut fused_outputs: Vec<ValueClassId> = c_alt.outputs.clone();
    if variant == FusionVariant::Keep {
        // Keep variant appends the seam value as an additional output
        // (§10.2). The extractor now sees this candidate as a valid
        // producer of the seam, so selecting keep alone satisfies the
        // producer equation for both the seam and the consumer's outputs.
        fused_outputs.push(seam_val);
    }

    // Buffer bindings for the underlying KernelModuleNode: the
    // node's inputs/outputs are BufIds keyed off the *original*
    // graph's physical buffers, one per positional port. Reconstruction
    // will map value classes back to BufIds via `gf.physical`, but the
    // KernelModuleNode still needs a plausible set of BufIds. We use
    // the physical BufIds of the fused_inputs/outputs directly.
    let input_bufs: Vec<crate::graph_ir::BufId> =
        fused_inputs.iter().map(|&v| gf.physical(v)).collect();
    let output_bufs: Vec<crate::graph_ir::BufId> =
        fused_outputs.iter().map(|&v| gf.physical(v)).collect();

    let node = GraphNode::Kernel(KernelModuleNode {
        module: std::sync::Arc::new(fused_module),
        param_bindings: merged_bindings,
        inputs: input_bufs,
        outputs: output_bufs,
        types: None,
        hash: None,
        canonical: false,
        fusion_history: None,
    });

    Ok(CandidateDraft {
        parents: vec![producer_node, consumer_node],
        variant,
        alt: AltGraphNode {
            inputs: fused_inputs,
            outputs: fused_outputs,
            node,
        },
    })
}

/// `QuastEmitter` that lowers a Quast into HIR nodes on a mutable
/// `IRBuilder`. `env` maps every symbol appearing in the Quast to a
/// pre-interned NodeId in the destination builder.
struct IndexEmitter<'a> {
    b: &'a mut IRBuilder,
    env: &'a HashMap<VarId, HirNodeId>,
}

impl QuastEmitter for IndexEmitter<'_> {
    type Val = HirNodeId;
    fn sym(&mut self, v: VarId) -> HirNodeId {
        *self
            .env
            .get(&v)
            .unwrap_or_else(|| panic!("unbound symbol {v:?} in IndexEmitter"))
    }
    fn cst(&mut self, c: u32) -> HirNodeId {
        self.b.const_u32(c)
    }
    fn add(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.b.add(a, b)
    }
    fn sub(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.b.sub(a, b)
    }
    fn mul(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.b.mul(a, b)
    }
    fn div(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.b.div(a, b)
    }
    fn rem(&mut self, a: HirNodeId, b: HirNodeId) -> HirNodeId {
        self.b.rem(a, b)
    }
}

/// Alpha-renames every [`VarId`] mentioned in `e` through `map`. Used
/// to remap shape expressions and `ConstSym`-carried symbols when
/// cloning parent module input declarations.
fn remap_size_expr(e: &SizeExpr, map: &HashMap<VarId, VarId>) -> SizeExpr {
    use crate::quast::{Expr, SymConst};
    match e {
        Expr::Sym(v) => Expr::Sym(*map.get(v).unwrap_or(v)),
        Expr::Const(SymConst::Lit(_)) => e.clone(),
        Expr::Const(SymConst::Sym(v)) => Expr::Const(SymConst::Sym(*map.get(v).unwrap_or(v))),
        Expr::Add(a, b) => Expr::Add(
            std::sync::Arc::new(remap_size_expr(a, map)),
            std::sync::Arc::new(remap_size_expr(b, map)),
        ),
        Expr::Mul(a, c) => {
            let c = match c {
                SymConst::Lit(_) => *c,
                SymConst::Sym(v) => SymConst::Sym(*map.get(v).unwrap_or(v)),
            };
            Expr::Mul(std::sync::Arc::new(remap_size_expr(a, map)), c)
        }
        Expr::FloorDiv(a, c) => {
            let c = match c {
                SymConst::Lit(_) => *c,
                SymConst::Sym(v) => SymConst::Sym(*map.get(v).unwrap_or(v)),
            };
            Expr::FloorDiv(std::sync::Arc::new(remap_size_expr(a, map)), c)
        }
        Expr::Neg(a) => Expr::Neg(std::sync::Arc::new(remap_size_expr(a, map))),
    }
}

/// Locates the [`Node::Input`] NodeIds reachable from `root`. Because
/// hash-consing interns one NodeId per distinct `Node::Input(k)`, each
/// referenced input position appears exactly once in the returned map.
fn find_input_nodes(module: &Module, root: HirNodeId) -> HashMap<usize, HirNodeId> {
    let mut out = HashMap::new();
    let mut work = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = work.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = module.builder.node(id);
        if let Node::Input(k) = node {
            out.insert(*k, id);
        }
        for c in children_of(node) {
            work.push(c);
        }
    }
    out
}

// -------------------------------------------------------------------------
// Enumeration
// -------------------------------------------------------------------------

/// Options passed to [`enumerate`] to control which variants are emitted.
///
/// The M3 drop variant is always enumerated; the M5 keep variant is
/// gated by the per-seam trigger conditions in §10.2 and can be forced
/// on for every drop candidate via
/// [`EnumerateOptions::enable_all_keep_variants`].
#[derive(Copy, Clone, Debug, Default)]
pub struct EnumerateOptions {
    /// If `true`, every legal drop candidate also emits its keep sibling
    /// regardless of the §10.2 trigger conditions. Useful as a
    /// diagnostic and to test the extractor's ability to choose between
    /// materialize/duplicate/keep. Off by default because it inflates
    /// enumeration for the common "seam feeds one consumer only" case
    /// where keep is strictly worse than drop.
    pub enable_all_keep_variants: bool,
}

/// Enumeration context (M6): the frozen prefix of nodes eligible as
/// parents this round, and the origins of every such node used for the
/// disjoint-origins check (§9.1 obligation).
///
/// The saturation driver freezes the alternative-graph node count at the
/// start of each round and passes it here; new candidates inserted mid-
/// round never enter enumeration until the next round.
pub struct EnumerateContext<'a> {
    /// Only nodes `0..frozen_node_count` are eligible as parents.
    pub frozen_node_count: usize,
    /// `origins[n.0]` for each node `n < frozen_node_count`. Length
    /// equals `frozen_node_count`.
    pub origins: &'a [BTreeSet<NodeId>],
    /// The saturation driver skips pairs whose parents were both
    /// enumerated in an earlier round by requiring at least one parent
    /// to have `NodeId >= min_new_parent_id`. Round 1 sets this to
    /// `0` (all pairs eligible); round `r+1` sets it to the
    /// alternative-graph node count at the end of round `r`.
    pub min_new_parent_id: usize,
    pub options: EnumerateOptions,
}

/// Owned context wrapper for callers that want an
/// [`EnumerateContext`] over the full node set with seed-like origins
/// ({NodeId(i)}). Tests and pass-standalone code paths use this.
pub struct OwnedEnumerateContext {
    pub frozen_node_count: usize,
    pub origins: Vec<BTreeSet<NodeId>>,
    pub options: EnumerateOptions,
}

impl OwnedEnumerateContext {
    /// Treats every node in `gf` as its own seed origin and freezes at
    /// `gf.nodes.len()`. Suitable for unit tests that don't run
    /// bounded saturation.
    pub fn all_seed(gf: &GraphFuser, options: EnumerateOptions) -> Self {
        let n = gf.nodes.len();
        let origins = (0..n)
            .map(|i| {
                let mut s = BTreeSet::new();
                s.insert(NodeId(i));
                s
            })
            .collect();
        Self {
            frozen_node_count: n,
            origins,
            options,
        }
    }

    /// Borrowed view suitable for passing to [`enumerate`].
    pub fn as_ref(&self) -> EnumerateContext<'_> {
        EnumerateContext {
            frozen_node_count: self.frozen_node_count,
            origins: &self.origins,
            min_new_parent_id: 0,
            options: self.options,
        }
    }
}

/// Enumerates producer-consumer candidates in a deterministic order.
///
/// For each single-writer seam feeding a consumer within the frozen
/// prefix, the pass emits a drop candidate (§10.1) and — when the seam
/// has legitimate other users — a keep candidate (§10.2) that also
/// materializes the seam. Composition of already-fused candidates is
/// allowed as long as origin sets stay disjoint.
///
/// The keep-trigger predicate (`should_emit_keep`) fires when:
///
/// - the seam is a registered graph output (`gf.outputs`);
/// - the seam has another eligible consumer distinct from the current one;
/// - `options.enable_all_keep_variants`.
pub fn enumerate(gf: &GraphFuser, ctx: &EnumerateContext) -> Vec<CandidateDraft> {
    let frozen = ctx.frozen_node_count.min(gf.nodes.len());
    let min_new = ctx.min_new_parent_id;
    let mut out = Vec::new();
    for (v, producers) in gf.producers.iter().enumerate() {
        // Iterate every producer of `v` (§10.1). A value can have more
        // than one producer once fused candidates that materialize it
        // land — e.g. B and drop(A,B) both produce B's output value
        // class. We consider each `(producer, consumer)` pair
        // independently; deduplication is handled downstream by
        // `CandidateKey`.
        for pu in producers {
            let p_node = pu.node;
            if p_node.0 >= frozen {
                continue;
            }
            let consumers = &gf.consumers[v];
            for cu in consumers {
                let c_node = cu.node;
                if c_node.0 >= frozen {
                    continue;
                }
                if c_node == p_node {
                    continue;
                }
                // At least one parent must be new since the last round;
                // pairs where both were seen in previous rounds have
                // already been emitted.
                if p_node.0 < min_new && c_node.0 < min_new {
                    continue;
                }
                if !disjoint_origins(&ctx.origins[p_node.0], &ctx.origins[c_node.0]) {
                    continue;
                }
                let seam = ValueClassId(v);
                if let Ok(draft) =
                    synthesize_producer_consumer(gf, p_node, c_node, seam, FusionVariant::Drop)
                {
                    out.push(draft);
                }
                if should_emit_keep(gf, seam, c_node, frozen, &ctx.options) {
                    if let Ok(draft) =
                        synthesize_producer_consumer(gf, p_node, c_node, seam, FusionVariant::Keep)
                    {
                        out.push(draft);
                    }
                }
            }
        }
    }
    out
}

fn disjoint_origins(a: &BTreeSet<NodeId>, b: &BTreeSet<NodeId>) -> bool {
    a.is_disjoint(b)
}

/// Predicate for §10.2's keep-trigger conditions.
fn should_emit_keep(
    gf: &GraphFuser,
    seam: ValueClassId,
    current_consumer: NodeId,
    frozen: usize,
    options: &EnumerateOptions,
) -> bool {
    if options.enable_all_keep_variants {
        return true;
    }
    if gf.outputs.contains(&seam) {
        return true;
    }
    // Any other consumer of the seam within the frozen prefix.
    for u in &gf.consumers[seam.0] {
        if u.node == current_consumer {
            continue;
        }
        if u.node.0 < frozen {
            return true;
        }
    }
    false
}
