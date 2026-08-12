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
//! Supported cases (drop variant):
//!
//! - both producer and consumer are single-`compute` structured kernels;
//! - the consumer may carry `scatter` / `par` / `threads` attributes — they transfer verbatim onto
//!   the fused compute; the producer may carry a `scatter`, whose *provided inverse* is composed
//!   into every seam read (the inverse is trusted, never verified);
//! - seam reads may have arbitrary rank: read coordinates are linearized over the consumer's
//!   declared seam view and delinearized over the producer's physical output shape whenever the two
//!   shapes do not agree axis-wise (the "reshape view" case);
//! - the producer body may be a scalar expression, an inline `Let` chain, a `Reduce`, a `Pack`
//!   (rank-extending), or a nest of inner `Compute`s — reads that drill into a `Pack` must resolve
//!   to literal component indices;
//! - producer and consumer outer bounds may differ; the drop gate is seam *element-count* equality,
//!   proven symbolically when possible and otherwise certified against the concrete parameter
//!   bindings of the candidate pair;
//! - module parameters unify by (name, bound value); a name bound to two different values is split
//!   (`n`, `n#1`, …), so a chain of the *same* symbolic kernel invoked at shrinking sizes fuses
//!   into a single symbolic artifact shared by every level;
//! - the producer may have several outputs (a `Tuple` body): the seam is one designated output
//!   element, inlined at consumer reads; the sibling elements remain materialized, which requires
//!   plain kernels and equal outer bounds (the siblings are emitted at the fused domain's index).
//!
//! The keep variant additionally requires plain kernels (no attributes),
//! a scalar seam body (no rank-extending spine), and equal outer
//! bounds, because the seam is materialized at the fused domain's index.
//!
//! Synthesis is a capture-free HIR clone. The consumer's compute body is
//! cloned into a fresh module. At every seam-read site the producer body
//! is *re-cloned* with the producer's outer variable substituted by the
//! composed coordinate σ (read coords → linearize over the seam view →
//! delinearize over the producer's physical shape → provided scatter
//! inverse), evaluated as a fresh HIR expression in the fused builder.
//!
//! Multi-output producers are enumerated per `(producer, output)` pair:
//! `gf.producers` lists a node once per produced value class, so the
//! seam value alone designates the output element to fuse through.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

pub(super) use crate::passes::fusion_utils::remap_size_expr;
use crate::{
    graph_ir::{GraphNode, KernelModuleNode},
    ir::{IRBuilder, Module, Node, NodeId as HirNodeId, SizeExpr, VarId},
    module_hash::children_of,
    passes::{
        fusion_utils::{clone_expr_with_params, CloneError},
        fusion_v2::model::{AltGraphNode, GraphFuser, NodeId, ValueClassId},
        utils::hir_to_sexpr,
    },
    quast::{ParSpec, SExpr, Scatter, SymConst},
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
/// The recognizer covers:
///
/// - **identity**: `y[outer_var]`;
/// - **affine permutation**: `y[a*outer_var + b]`, `y[N-1-outer_var]`, etc. — any quasi-affine
///   function of the enclosing loop variables and module parameters;
/// - **rank-k reads**: every axis of a multi-index read is captured as its own [`SExpr`];
/// - **nested-index consumers**: consumer body may contain an inner `Compute` or `Reduce`; reads
///   inside that inner scope are captured with their index-scope (`inner_vars`) recorded;
/// - **reduction producers**: the producer body is a `Reduce` (single scalar per outer iteration),
///   which we detect but do not treat specially — the fused synthesis clones the reduce
///   sub-expression at every seam read like any other body expression.
///
/// Attributes on the outer compute (`scatter` / `par` / `threads`) are
/// recorded, not rejected; callers that only support plain kernels must
/// filter with [`KernelShape::is_plain`].
#[derive(Debug, Clone)]
pub(super) struct KernelShape {
    /// Fresh [`VarId`] of the outer `compute` iteration variable.
    pub(super) outer_var: VarId,
    /// Symbolic outer bound.
    pub(super) outer_bound: SizeExpr,
    /// The scalar body expression (per-outer-iteration return value).
    pub(super) body_root: HirNodeId,
    /// Every `Node::Index` site reachable from `body_root` whose
    /// `tensor` is a `Node::Input(_)`.
    pub(super) reads: Vec<ReadSite>,
    /// Write map of the outer compute, if any.
    pub(super) scatter: Option<Box<Scatter>>,
    /// Compute layout of the outer compute, if any.
    pub(super) par: Option<Box<ParSpec>>,
    /// Thread-count hint of the outer compute, if any.
    pub(super) threads: Option<usize>,
}

impl KernelShape {
    /// No `scatter` / `par` / `threads` attributes on the outer compute.
    pub(super) fn is_plain(&self) -> bool {
        self.scatter.is_none() && self.par.is_none() && self.threads.is_none()
    }
}

#[derive(Debug, Clone)]
pub(super) struct ReadSite {
    /// Module input position this site reads.
    pub(super) input_pos: usize,
    /// [`Node::Index`] site.
    pub(super) index_node: HirNodeId,
    /// Per-axis index expressions in the source module's variable
    /// namespace: loop vars appear in `Sym` position, module parameters
    /// in `SymConst::Sym` position. `None` when any axis is not
    /// expressible as an [`SExpr`] over the enclosing scope — such a
    /// read can never be a fused seam.
    pub(super) index_exprs: Option<Vec<SExpr>>,
    /// In-scope binders at this read site, outer-first. Each entry is
    /// the source module's [`VarId`].
    pub(super) inner_vars: Vec<VarId>,
}

/// Recognizes a producer- or consumer-shaped kernel module. Returns
/// `None` if the module does not fit any of the supported shapes.
pub(super) fn identify_kernel_shape(module: &Module) -> Option<KernelShape> {
    let (outer_var, outer_bound, body_root, scatter, par, threads) =
        match module.builder.node(module.body) {
            Node::Compute {
                bound,
                var,
                body,
                scatter,
                par,
                threads,
            } => (
                *var,
                bound.clone(),
                *body,
                scatter.clone(),
                par.clone(),
                *threads,
            ),
            _ => return None,
        };

    // Walk the body, recording read sites and the scope they occur in.
    // We track scope as a stack of (VarId, ...) — only Compute/Reduce
    // bodies push.
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
                // Convert every index axis to an SExpr in the source
                // module's variable namespace; any non-convertible axis
                // voids the whole site's expression vector.
                let syms =
                    |v: VarId| -> Option<SExpr> { scope.contains(&v).then(|| SExpr::sym(v)) };
                let lets = |_v: VarId| -> Option<HirNodeId> { None };
                let index_exprs: Option<Vec<SExpr>> = indices
                    .iter()
                    .map(|&ix| hir_to_sexpr(&module.builder, ix, &syms, &lets).ok())
                    .collect();
                reads.push(ReadSite {
                    input_pos: *k,
                    index_node: id,
                    index_exprs,
                    inner_vars: scope.clone(),
                });
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
        scatter,
        par,
        threads,
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
    /// Keep variant only: producer and consumer have different outer
    /// bounds (drop tolerates this via the element-count gate).
    OuterBoundMismatch,
    /// No consumer read site references the seam value.
    NoSeamReadInConsumer,
    /// A consumer read of the seam has an index expression that is not
    /// a quasi-affine function of the enclosing scope.
    SeamIndexNotAffine,
    /// The consumer's declared seam view and the producer's output
    /// shape disagree in total element count, or a shape axis needed by
    /// the linearize/delinearize roundtrip is not a positive literal.
    SeamShapeMismatch,
    /// A drill coordinate into a producer `Pack` component did not fold
    /// to an in-range literal.
    SeamComponentNotConst,
    /// The composed logical coordinate rank disagrees with the producer
    /// body's rank-extending spine.
    ProducerBodyRankMismatch,
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
/// The producer's seam element is *re-cloned* at each seam-read site
/// with the producer's outer variable substituted by that read's
/// composed coordinate σ, so identity, affine-permutation, nested-index,
/// reshape-view and scattered-producer consumers all reduce to the same
/// synthesis loop. Rank-extending producer bodies (`Pack`, inner
/// `Compute` nests) are drilled per read using the trailing composed
/// coordinates. For a multi-output producer the seam is
/// `p_alt.outputs.position(seam_val)`'s tuple element; the sibling
/// elements stay materialized (drop appends them to the outputs, keep
/// appends every producer output).
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

    // Seam selection for multi-output producers: the seam is one
    // designated output; a multi-output producer's body is a `Tuple`
    // whose elements line up with `p_alt.outputs`. The seam element is
    // inlined at consumer reads; every other element stays materialized.
    let Some(seam_out_idx) = p_alt.outputs.iter().position(|&v| v == seam_val) else {
        debug_assert!(false, "seam value is not an output of the producer node");
        return Err(SynthesisFailure::UnsupportedShape);
    };
    let p_elems: Vec<HirNodeId> = match p_module.builder.node(p_shape.body_root) {
        Node::Tuple(es) if p_alt.outputs.len() > 1 => es.clone(),
        _ => vec![p_shape.body_root],
    };
    if p_elems.len() != p_alt.outputs.len() {
        return Err(SynthesisFailure::UnsupportedShape);
    }
    let p_seam_root = p_elems[seam_out_idx];

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
    if seam_reads.iter().any(|r| r.index_exprs.is_none()) {
        return Err(SynthesisFailure::SeamIndexNotAffine);
    }

    // Fused-module boundary: producer inputs first, followed by consumer
    // inputs whose positions are not seam positions.
    //
    // Parameters unify by (name, bound value): a name already claimed at
    // a *different* value re-probes as `name#1`, `name#2`, … until an
    // unclaimed or equal-valued slot is found, so a chain of the same
    // symbolic kernel invoked at shrinking sizes fuses without
    // collapsing its sizes. Fused param VarIds are allocated first
    // (producer side, then consumer, in first-appearance order) because
    // `module_hash` hashes param VarIds raw — reference modules built
    // by hand allocate params at the lowest VarIds, and fused modules
    // must match.
    let mut fb = IRBuilder::new();
    let mut claimed: HashMap<String, (VarId, Option<i64>)> = HashMap::new();
    let mut merged_bindings: BTreeMap<String, i64> = BTreeMap::new();
    let p_param_map = normalize_params(
        &mut fb,
        &mut claimed,
        &mut merged_bindings,
        &p_module,
        &p_binding,
    );
    let c_param_map = normalize_params(
        &mut fb,
        &mut claimed,
        &mut merged_bindings,
        &c_module,
        &c_binding,
    );

    let p_ctx = SideCtx {
        param_map: &p_param_map,
        env: side_env(&p_module, &p_binding),
    };
    let c_ctx = SideCtx {
        param_map: &c_param_map,
        env: side_env(&c_module, &c_binding),
    };

    // Producer output geometry for the seam element. `spine` is the
    // rank-extending chain of Pack / inner-Compute steps under the outer
    // compute; `p_logical` is the seam's logical output shape (outer
    // bound ++ spine dims); `p_phys` is what the seam buffer actually
    // stores — equal to `p_logical` unless a scatter rewrites the
    // layout.
    let spine = producer_spine(&p_module, p_seam_root)?;
    let mut p_logical: Vec<SExpr> = vec![p_shape.outer_bound.clone()];
    p_logical.extend(spine.iter().map(|s| s.dim.clone()));
    let p_phys: Vec<SExpr> = match &p_shape.scatter {
        None => p_logical.clone(),
        Some(sc) => {
            let conc: Vec<usize> = p_logical
                .iter()
                .map(|d| {
                    d.concretize(&p_ctx.env)
                        .as_const()
                        .and_then(|c| usize::try_from(c).ok())
                })
                .collect::<Option<Vec<usize>>>()
                .ok_or(SynthesisFailure::UnsupportedShape)?;
            sc.out_shape_for(&conc)
                .map_err(|_| SynthesisFailure::UnsupportedShape)?
                .into_iter()
                .map(|d| SExpr::cst(SymConst::Lit(d as i64)))
                .collect()
        }
    };

    // Materialized producer elements: every element for keep, the
    // non-seam elements for drop. Materialization emits the element at
    // the fused domain's index, so it requires plain kernels and equal
    // outer bounds; the keep seam additionally needs a scalar body (no
    // rank-extending spine), and a materialized element may not be a
    // bare inner compute (canonicalize has no binder to hoist it under).
    let materializes_producer_elems = variant == FusionVariant::Keep || p_alt.outputs.len() > 1;
    if materializes_producer_elems {
        if !p_shape.is_plain() || !c_shape.is_plain() {
            return Err(SynthesisFailure::UnsupportedShape);
        }
        if variant == FusionVariant::Keep && !spine.is_empty() {
            return Err(SynthesisFailure::UnsupportedShape);
        }
        for (i, &e) in p_elems.iter().enumerate() {
            let materialized = variant == FusionVariant::Keep || i != seam_out_idx;
            if materialized && matches!(p_module.builder.node(e), Node::Compute { .. }) {
                return Err(SynthesisFailure::UnsupportedShape);
            }
        }
        if !two_tier_eq(&p_shape.outer_bound, &p_ctx, &c_shape.outer_bound, &c_ctx) {
            return Err(SynthesisFailure::OuterBoundMismatch);
        }
    }

    // Drop gate: the consumer's declared seam view must cover exactly
    // as many elements as the producer writes — proven symbolically
    // when possible, otherwise certified against the concrete bindings
    // of this candidate pair.
    for &pos in &seam_positions {
        let decl = &c_module.builder.inputs()[pos].shape;
        if !counts_eq(decl, &c_ctx, &p_logical, &p_ctx) {
            return Err(SynthesisFailure::SeamShapeMismatch);
        }
    }

    // Compose σ per seam read: read coords → linearize over the
    // declared seam view → delinearize over the producer's physical
    // shape → provided scatter inverse → drill down the producer's
    // rank-extending spine.
    let mut read_plans: HashMap<HirNodeId, ReadPlan> = HashMap::new();
    for read in &seam_reads {
        let decl = &c_module.builder.inputs()[read.input_pos].shape;
        let plan = plan_read(
            read,
            decl,
            &p_phys,
            &spine,
            p_shape.scatter.as_deref(),
            &p_ctx,
            &c_ctx,
        )?;
        read_plans.insert(read.index_node, plan);
    }

    // Declare inputs and remember their Input NodeIds in fb.
    let mut fused_p_input_nodes: Vec<HirNodeId> = Vec::new();
    let mut fused_c_input_nodes: Vec<Option<HirNodeId>> =
        vec![None; c_module.builder.inputs().len()];

    for decl in p_module.builder.inputs().iter() {
        let shape: Vec<SizeExpr> = decl
            .shape
            .iter()
            .map(|d| remap_size_expr(d, &p_param_map))
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
            .map(|d| remap_size_expr(d, &c_param_map))
            .collect();
        let n = fb.input(decl.name.clone(), decl.elem, shape);
        fused_c_input_nodes[i] = Some(n);
    }

    // Pre-compute the maps needed by the clone, keyed on the *source*
    // modules' Input NodeIds. These maps are stable across
    // producer-body inlines because each seam read only differs in the
    // coordinate σ substituted for `p_shape.outer_var`.
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
    let outer_bound_fb = remap_size_expr(&c_shape.outer_bound, &c_param_map);
    let k_var = fb.fresh_var();
    let k_var_node = fb.intern(Node::Var(k_var));

    // Consumer non-seam Input NodeIds → their fused Input NodeIds.
    let mut consumer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
    for (i, &c_input_node) in &consumer_input_nodes {
        if let Some(fused_input) = fused_c_input_nodes[*i] {
            consumer_subst.insert(c_input_node, fused_input);
        }
    }

    // Consumer VarId → fused NodeId. Outer var fuses to `k_var_node`;
    // inner Compute/Reduce bound vars are introduced by the clone
    // itself when it descends. Params never appear as `Node::Var` in a
    // well-formed module — they ride in `ConstSym` / shape positions,
    // which the clone remaps through the param map directly.
    let mut consumer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
    consumer_vars.insert(c_shape.outer_var, k_var_node);

    // Substitute at every seam read via a clone-time hook. The hook
    // fires when the clone visits the seam Index NodeId, with the
    // current `vars` snapshot — which includes the fresh identities
    // the clone has already allocated for any inner Compute/Reduce
    // binders. That means nested-index seam reads are handled the
    // same way as top-level ones.
    let hook = |dst: &mut IRBuilder,
                src_id: HirNodeId,
                vars: &HashMap<VarId, HirNodeId>|
     -> Result<Option<HirNodeId>, CloneError> {
        let Some(plan) = read_plans.get(&src_id) else {
            return Ok(None);
        };
        // Emit env: every source VarId appearing in σ or in a drill
        // coordinate maps to its destination NodeId from the clone's
        // current scope.
        let mut needed: BTreeSet<VarId> = BTreeSet::new();
        plan.sigma.syms(&mut needed);
        for (_, step) in &plan.drill {
            if let DrillStepPlan::BindInner(e) = step {
                e.syms(&mut needed);
            }
        }
        let mut env: HashMap<VarId, HirNodeId> = HashMap::new();
        for v in needed {
            let n = *vars.get(&v).ok_or(CloneError::UnboundVar { var: v })?;
            env.insert(v, n);
        }
        let sigma_node = emit_sexpr(dst, &plan.sigma, &env, &c_param_map)?;
        let mut drill: HashMap<HirNodeId, DrillAction> = HashMap::new();
        for (node, step) in &plan.drill {
            let action = match step {
                DrillStepPlan::PackElem(k) => DrillAction::PackElem(*k),
                DrillStepPlan::BindInner(e) => {
                    DrillAction::BindInner(emit_sexpr(dst, e, &env, &c_param_map)?)
                }
            };
            drill.insert(*node, action);
        }
        // Clone the seam element of the producer body with
        // producer.outer_var → sigma_node, drilling Pack / inner-Compute
        // spine steps per this read's plan.
        let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
        producer_vars.insert(p_shape.outer_var, sigma_node);
        let inlined = clone_with_drill(
            &p_module,
            p_seam_root,
            dst,
            &producer_subst,
            &producer_vars,
            &p_param_map,
            &drill,
        )?;
        Ok(Some(inlined))
    };

    let cloned_body = clone_expr_with_params(
        &c_module,
        c_shape.body_root,
        &mut fb,
        &consumer_subst,
        &consumer_vars,
        &c_param_map,
        hook,
    )
    .map_err(|e| SynthesisFailure::CloneError(format!("{e:?}")))?;

    // Materialize producer elements alongside the consumer outputs: the
    // seam (and, for a multi-output producer, its siblings) for keep,
    // the non-seam siblings only for drop. All share the fused compute's
    // outer var, so a single Compute with a `Tuple` body suffices
    // (§10.2). Producer inputs and producer-var mapping match the hook's
    // inline substitution, but the outer-var image is `k_var_node`
    // directly (identity access at the materialized index).
    let mut body_elems: Vec<HirNodeId> = if c_alt.outputs.len() > 1 {
        match fb.node(cloned_body) {
            Node::Tuple(es) => es.clone(),
            _ => {
                return Err(SynthesisFailure::CloneError(
                    "multi-output consumer body is not a tuple".into(),
                ))
            }
        }
    } else {
        vec![cloned_body]
    };
    for (i, &elem_root) in p_elems.iter().enumerate() {
        let materialized = variant == FusionVariant::Keep || i != seam_out_idx;
        if !materialized {
            continue;
        }
        let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
        producer_vars.insert(p_shape.outer_var, k_var_node);
        let cloned = clone_expr_with_params(
            &p_module,
            elem_root,
            &mut fb,
            &producer_subst,
            &producer_vars,
            &p_param_map,
            |_, _, _| Ok(None),
        )
        .map_err(|e| SynthesisFailure::CloneError(format!("{e:?}")))?;
        body_elems.push(cloned);
    }
    let compute_body = if body_elems.len() == 1 {
        body_elems[0]
    } else {
        fb.intern(Node::Tuple(body_elems))
    };

    // The consumer's launch attributes transfer verbatim: their exprs
    // reference only the consumer's own binders (which keep their
    // meaning on the fused compute). Producer par / threads are launch
    // hints that vanish under inlining; only the producer scatter
    // affects stored layout, and that is already composed into every
    // seam read via the provided inverse.
    let fused_body_id = fb.intern(Node::Compute {
        bound: outer_bound_fb,
        var: k_var,
        body: compute_body,
        scatter: c_shape.scatter.clone(),
        par: c_shape.par.clone(),
        threads: c_shape.threads,
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
    // Output order mirrors the body tuple: consumer outputs first, then
    // the materialized producer elements in producer output order. Keep
    // appends every producer output — the seam included, so the
    // extractor sees this candidate as a valid producer of the seam
    // (§10.2) — while drop appends only the seam's siblings.
    let mut fused_outputs: Vec<ValueClassId> = c_alt.outputs.clone();
    for (i, &v) in p_alt.outputs.iter().enumerate() {
        if variant == FusionVariant::Keep || i != seam_out_idx {
            fused_outputs.push(v);
        }
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

// -------------------------------------------------------------------------
// Synthesis helpers
// -------------------------------------------------------------------------

/// One rank-extending step under the producer's outer compute.
struct SpineStep {
    /// The producer-module HIR node of the step.
    node: HirNodeId,
    kind: SpineKind,
    /// Extent of the dimension this step introduces.
    dim: SExpr,
}

enum SpineKind {
    Pack { len: usize },
    Compute,
}

/// Walks the producer body from `body_root`, skipping `Let`s, and
/// collects the rank-extending spine: inner `Compute` nests terminated
/// by an optional `Pack`. Any other node ends the spine (scalar body).
/// Inner computes must be attribute-free — a scatter/par/threads there
/// has no meaning once the loop is unrolled into the consumer.
fn producer_spine(
    module: &Module,
    body_root: HirNodeId,
) -> Result<Vec<SpineStep>, SynthesisFailure> {
    let mut spine = Vec::new();
    let mut cur = body_root;
    loop {
        match module.builder.node(cur) {
            Node::Let { body, .. } => cur = *body,
            Node::Pack(elems) => {
                spine.push(SpineStep {
                    node: cur,
                    kind: SpineKind::Pack { len: elems.len() },
                    dim: SExpr::cst(SymConst::Lit(elems.len() as i64)),
                });
                break;
            }
            Node::Compute {
                bound,
                body,
                scatter,
                par,
                threads,
                ..
            } => {
                if scatter.is_some() || par.is_some() || threads.is_some() {
                    return Err(SynthesisFailure::UnsupportedShape);
                }
                spine.push(SpineStep {
                    node: cur,
                    kind: SpineKind::Compute,
                    dim: bound.clone(),
                });
                cur = *body;
            }
            _ => break,
        }
    }
    Ok(spine)
}

/// Strips a `#k` split suffix from a parameter name (`n#1` → `n`).
/// Returns the name unchanged when there is no well-formed suffix.
fn strip_split_suffix(name: &str) -> &str {
    match name.rsplit_once('#') {
        Some((base, suffix))
            if !base.is_empty()
                && !suffix.is_empty()
                && suffix.bytes().all(|b| b.is_ascii_digit()) =>
        {
            base
        }
        _ => name,
    }
}

/// Allocates fused-module params for one side. Each source param claims
/// the first `base`, `base#1`, `base#2`, … slot whose bound value
/// matches (an unbound param only unifies with another unbound one).
/// Returns the side's source-VarId → fused-VarId map.
fn normalize_params(
    fb: &mut IRBuilder,
    claimed: &mut HashMap<String, (VarId, Option<i64>)>,
    merged_bindings: &mut BTreeMap<String, i64>,
    module: &Module,
    binding: &BTreeMap<String, i64>,
) -> HashMap<VarId, VarId> {
    let mut map = HashMap::new();
    for (v, name) in module.builder.params() {
        let value = binding.get(name).copied();
        let base = strip_split_suffix(name);
        let mut k = 0usize;
        let fused = loop {
            let probe = if k == 0 {
                base.to_string()
            } else {
                format!("{base}#{k}")
            };
            match claimed.get(&probe) {
                None => {
                    let fresh = fb.fresh_var();
                    fb.inherit_param(fresh, probe.clone());
                    if let Some(val) = value {
                        merged_bindings.insert(probe.clone(), val);
                    }
                    claimed.insert(probe, (fresh, value));
                    break fresh;
                }
                Some((existing, ev)) if *ev == value => break *existing,
                Some(_) => k += 1,
            }
        };
        map.insert(*v, fused);
    }
    map
}

/// Per-side context for symbolic/concrete shape comparisons.
struct SideCtx<'a> {
    /// Source param VarId → fused param VarId.
    param_map: &'a HashMap<VarId, VarId>,
    /// Source param VarId → concrete value from the candidate's
    /// `param_bindings`.
    env: BTreeMap<VarId, i64>,
}

fn side_env(module: &Module, binding: &BTreeMap<String, i64>) -> BTreeMap<VarId, i64> {
    module
        .builder
        .params()
        .iter()
        .filter_map(|(v, name)| binding.get(name).map(|&val| (*v, val)))
        .collect()
}

/// Two-tier size equality. Tier 1 proves symbolically in the fused
/// param namespace (`n == n` after unification). Tier 2 certifies
/// against the concrete bindings of this candidate pair (`n@8 == m@8`).
/// Tier 2 is sound because the driver re-runs synthesis per candidate,
/// so a relation that only holds concretely is re-checked at every
/// pair; the module text itself stays symbolic either way.
fn two_tier_eq(a: &SExpr, a_ctx: &SideCtx, b: &SExpr, b_ctx: &SideCtx) -> bool {
    if remap_size_expr(a, a_ctx.param_map).fold_lits()
        == remap_size_expr(b, b_ctx.param_map).fold_lits()
    {
        return true;
    }
    match (
        a.concretize(&a_ctx.env).as_const(),
        b.concretize(&b_ctx.env).as_const(),
    ) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Element-count equality of two shapes: [`two_tier_eq`] on symbolic
/// products when both are representable, otherwise concrete per-dim
/// products.
fn counts_eq(a: &[SExpr], a_ctx: &SideCtx, b: &[SExpr], b_ctx: &SideCtx) -> bool {
    if let (Some(pa), Some(pb)) = (SExpr::product(a), SExpr::product(b)) {
        if two_tier_eq(&pa, a_ctx, &pb, b_ctx) {
            return true;
        }
    }
    let conc = |dims: &[SExpr], ctx: &SideCtx| -> Option<i64> {
        dims.iter().try_fold(1i64, |acc, d| {
            Some(acc * d.concretize(&ctx.env).as_const()?)
        })
    };
    match (conc(a, a_ctx), conc(b, b_ctx)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// σ and per-spine-step drill actions for one seam read, in the
/// consumer's variable namespace (loop vars in `Sym` position, consumer
/// params in `SymConst::Sym` position — remapped at emit time).
struct ReadPlan {
    /// Coordinate substituted for the producer's outer var.
    sigma: SExpr,
    /// Drill steps, outermost first, keyed by the producer spine node
    /// they apply to.
    drill: Vec<(HirNodeId, DrillStepPlan)>,
}

enum DrillStepPlan {
    /// Take component `k` of a producer `Pack`.
    PackElem(usize),
    /// Bind a producer inner `Compute` var to this coordinate.
    BindInner(SExpr),
}

/// Emitted form of a [`DrillStepPlan`] within one hook invocation.
enum DrillAction {
    PackElem(usize),
    BindInner(HirNodeId),
}

/// Composes the coordinate pipeline for one seam read: read coords →
/// linearize over the consumer's declared seam view → delinearize over
/// the producer's physical shape → provided scatter inverse → split
/// into the outer-var image (σ) and per-spine drill coordinates.
fn plan_read(
    read: &ReadSite,
    decl_dims: &[SExpr],
    p_phys: &[SExpr],
    spine: &[SpineStep],
    scatter: Option<&Scatter>,
    p_ctx: &SideCtx,
    c_ctx: &SideCtx,
) -> Result<ReadPlan, SynthesisFailure> {
    let coords = read
        .index_exprs
        .as_ref()
        .expect("caller filtered index_exprs to Some");

    // Fast path: the declared view agrees with the physical shape
    // axis-wise, so the read coordinates are already physical.
    let axiswise = decl_dims.len() == p_phys.len()
        && coords.len() == p_phys.len()
        && decl_dims
            .iter()
            .zip(p_phys)
            .all(|(a, b)| two_tier_eq(a, c_ctx, b, p_ctx));
    let phys: Vec<SExpr> = if axiswise {
        coords.clone()
    } else {
        // Reshape view: linearize row-major over the declared view
        // (Horner), then delinearize over the physical shape. All inner
        // extents must be positive literals for strides to be
        // expressible.
        if coords.len() != decl_dims.len() {
            return Err(SynthesisFailure::SeamShapeMismatch);
        }
        let mut flat = coords[0].clone();
        for (c, d) in coords[1..].iter().zip(&decl_dims[1..]) {
            let d = d
                .as_const()
                .filter(|&d| d > 0)
                .ok_or(SynthesisFailure::SeamShapeMismatch)?;
            flat = flat.mul_c(SymConst::Lit(d)).add(c);
        }
        if p_phys.len() == 1 {
            vec![flat]
        } else {
            let trailing: Vec<i64> = p_phys[1..]
                .iter()
                .map(|d| {
                    d.as_const()
                        .filter(|&d| d > 0)
                        .ok_or(SynthesisFailure::SeamShapeMismatch)
                })
                .collect::<Result<_, _>>()?;
            let mut strides = vec![1i64; p_phys.len()];
            for j in (0..p_phys.len() - 1).rev() {
                strides[j] = strides[j + 1] * trailing[j];
            }
            (0..p_phys.len())
                .map(|j| {
                    let q = if strides[j] == 1 {
                        flat.clone()
                    } else {
                        flat.floordiv(SymConst::Lit(strides[j]))
                    };
                    if j > 0 {
                        q.rem_c(SymConst::Lit(trailing[j - 1]))
                    } else {
                        q
                    }
                })
                .collect()
        }
    };

    // Compose the provided scatter inverse (physical → logical). The
    // inverse is trusted, never verified against the forward map.
    let logical: Vec<SExpr> = match scatter {
        None => phys,
        Some(sc) => {
            if sc.inv_params.len() != phys.len() {
                return Err(SynthesisFailure::SeamShapeMismatch);
            }
            let map: BTreeMap<VarId, SExpr> = sc
                .inv_params
                .iter()
                .copied()
                .zip(phys.iter().cloned())
                .collect();
            sc.inv_exprs
                .iter()
                .map(|q| SExpr::from(q).substitute(&map))
                .collect()
        }
    };
    let logical: Vec<SExpr> = logical.iter().map(SExpr::fold_lits).collect();

    if logical.len() != 1 + spine.len() {
        return Err(SynthesisFailure::ProducerBodyRankMismatch);
    }

    let mut drill = Vec::with_capacity(spine.len());
    for (step, coord) in spine.iter().zip(&logical[1..]) {
        let plan = match step.kind {
            SpineKind::Pack { len } => {
                // Pack components are positional, so the coordinate
                // must fold to an in-range literal. Deliberately NOT
                // certified via `concretize`: a binding-dependent
                // component pick would bake this candidate's sizes into
                // the module text and break one-artifact-per-chain.
                let k = coord
                    .as_const()
                    .filter(|&k| k >= 0 && (k as usize) < len)
                    .ok_or(SynthesisFailure::SeamComponentNotConst)?;
                DrillStepPlan::PackElem(k as usize)
            }
            SpineKind::Compute => DrillStepPlan::BindInner(coord.clone()),
        };
        drill.push((step.node, plan));
    }

    Ok(ReadPlan {
        sigma: logical[0].clone(),
        drill,
    })
}

/// Lowers an [`SExpr`] into HIR arithmetic on `b`. `env` maps every
/// loop-var `Sym` to a pre-interned NodeId; `param_map` alpha-renames
/// parameter symbols into the fused namespace. Loop-var-free
/// sub-expressions become `const_u32` / `const_sym` leaves; `a - b` is
/// recovered from `Add(x, Neg(y))` after literal folding.
fn emit_sexpr(
    b: &mut IRBuilder,
    e: &SExpr,
    env: &HashMap<VarId, HirNodeId>,
    param_map: &HashMap<VarId, VarId>,
) -> Result<HirNodeId, CloneError> {
    let e = push_negs(&e.fold_lits());
    emit_sexpr_rec(b, &e, env, param_map)
}

fn emit_sexpr_rec(
    b: &mut IRBuilder,
    e: &SExpr,
    env: &HashMap<VarId, HirNodeId>,
    param_map: &HashMap<VarId, VarId>,
) -> Result<HirNodeId, CloneError> {
    use crate::quast::Expr;
    let mut vars = BTreeSet::new();
    e.syms(&mut vars);
    if vars.is_empty() {
        // Loop-var-free: a literal or a parameter expression.
        return Ok(match e.as_const() {
            Some(c) if c >= 0 => b.const_u32(c as u32),
            _ => b.const_sym(remap_size_expr(e, param_map)),
        });
    }
    Ok(match e {
        Expr::Sym(v) => *env.get(v).ok_or(CloneError::UnboundVar { var: *v })?,
        Expr::Add(x, y) => match (x.as_ref(), y.as_ref()) {
            (_, Expr::Neg(y2)) => {
                let xa = emit_sexpr_rec(b, x, env, param_map)?;
                let ya = emit_sexpr_rec(b, y2, env, param_map)?;
                b.sub(xa, ya)
            }
            (Expr::Neg(x2), _) => {
                let ya = emit_sexpr_rec(b, y, env, param_map)?;
                let xa = emit_sexpr_rec(b, x2, env, param_map)?;
                b.sub(ya, xa)
            }
            _ => {
                let xa = emit_sexpr_rec(b, x, env, param_map)?;
                let ya = emit_sexpr_rec(b, y, env, param_map)?;
                b.add(xa, ya)
            }
        },
        Expr::Mul(x, c) => {
            let xa = emit_sexpr_rec(b, x, env, param_map)?;
            let ca = emit_symconst(b, c, param_map);
            b.mul(xa, ca)
        }
        Expr::FloorDiv(x, c) => {
            let xa = emit_sexpr_rec(b, x, env, param_map)?;
            let ca = emit_symconst(b, c, param_map);
            b.div(xa, ca)
        }
        Expr::Neg(x) => {
            let zero = b.const_u32(0);
            let xa = emit_sexpr_rec(b, x, env, param_map)?;
            b.sub(zero, xa)
        }
        // `syms` returned non-empty, so this cannot be a Const leaf.
        Expr::Const(_) => unreachable!("const leaf has no syms"),
    })
}

fn emit_symconst(b: &mut IRBuilder, c: &SymConst, param_map: &HashMap<VarId, VarId>) -> HirNodeId {
    match c {
        SymConst::Lit(x) => b.const_u32(*x as u32),
        SymConst::Sym(v) => {
            let mapped = *param_map.get(v).unwrap_or(v);
            b.const_sym(SExpr::cst(SymConst::Sym(mapped)))
        }
    }
}

/// Rewrites `Mul(a, -c)` as `Neg(Mul(a, c))` so subtraction recovery in
/// [`emit_sexpr_rec`] sees the negation at the `Add` level.
fn push_negs(e: &SExpr) -> SExpr {
    use crate::quast::Expr;
    match e {
        Expr::Mul(a, SymConst::Lit(c)) if *c < 0 => Expr::Neg(Arc::new(Expr::Mul(
            Arc::new(push_negs(a)),
            SymConst::Lit(-c),
        ))),
        Expr::Add(a, b) => Expr::Add(Arc::new(push_negs(a)), Arc::new(push_negs(b))),
        Expr::Mul(a, c) => Expr::Mul(Arc::new(push_negs(a)), *c),
        Expr::FloorDiv(a, c) => Expr::FloorDiv(Arc::new(push_negs(a)), *c),
        Expr::Neg(a) => Expr::Neg(Arc::new(push_negs(a))),
        Expr::Sym(_) | Expr::Const(_) => e.clone(),
    }
}

/// [`clone_expr_with_params`] wrapper whose hook drills the producer's
/// rank-extending spine: `Pack` steps collapse to the planned
/// component, inner `Compute` steps are dissolved by binding their var
/// to the planned coordinate.
fn clone_with_drill(
    src: &Module,
    root: HirNodeId,
    dst: &mut IRBuilder,
    subst: &HashMap<HirNodeId, HirNodeId>,
    vars: &HashMap<VarId, HirNodeId>,
    param_map: &HashMap<VarId, VarId>,
    drill: &HashMap<HirNodeId, DrillAction>,
) -> Result<HirNodeId, CloneError> {
    clone_expr_with_params(
        src,
        root,
        dst,
        subst,
        vars,
        param_map,
        |dst, id, vars_now| {
            let Some(action) = drill.get(&id) else {
                return Ok(None);
            };
            let node = src.builder.node(id).clone();
            match (node, action) {
                (Node::Pack(elems), DrillAction::PackElem(k)) => {
                    clone_with_drill(src, elems[*k], dst, subst, vars_now, param_map, drill)
                        .map(Some)
                }
                (Node::Compute { var, body, .. }, DrillAction::BindInner(coord)) => {
                    let mut inner = vars_now.clone();
                    inner.insert(var, *coord);
                    clone_with_drill(src, body, dst, subst, &inner, param_map, drill).map(Some)
                }
                _ => Err(CloneError::MissingSubst { node: id }),
            }
        },
    )
}

/// Locates the [`Node::Input`] NodeIds reachable from `root`. Because
/// hash-consing interns one NodeId per distinct `Node::Input(k)`, each
/// referenced input position appears exactly once in the returned map.
pub(super) fn find_input_nodes(module: &Module, root: HirNodeId) -> HashMap<usize, HirNodeId> {
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
    /// Optional wall-time deadline for the enumerator. When set,
    /// passes check it in the site-collection loop and between
    /// synthesis chunks and return early once the current instant
    /// passes it. `None` disables the deadline.
    pub deadline: Option<std::time::Instant>,
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
            deadline: None,
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
    let debug = super::debug_reject_level();
    // Site collection is cheap (index scans + origin checks); synthesis
    // dominates, so it runs in parallel over the collected sites.
    //
    // Every producer of `v` is considered (§10.1). A value can have more
    // than one producer once fused candidates that materialize it
    // land — e.g. B and drop(A,B) both produce B's output value
    // class. We consider each `(producer, consumer)` pair
    // independently; deduplication is handled downstream by
    // `CandidateKey`.
    let mut sites: Vec<(NodeId, NodeId, ValueClassId)> = Vec::new();
    'outer: for (v, producers) in gf.producers.iter().enumerate() {
        if let Some(t) = ctx.deadline {
            if std::time::Instant::now() >= t {
                break 'outer;
            }
        }
        for pu in producers {
            let p_node = pu.node;
            if p_node.0 >= frozen {
                continue;
            }
            for cu in &gf.consumers[v] {
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
                sites.push((p_node, c_node, ValueClassId(v)));
            }
        }
    }
    let (out, rejects) = super::par_enumerate(sites, ctx.deadline, |(p_node, c_node, seam)| {
        let mut drafts = Vec::new();
        let mut rejects: Vec<(String, u64)> = Vec::new();
        match synthesize_producer_consumer(gf, p_node, c_node, seam, FusionVariant::Drop) {
            Ok(draft) => drafts.push(draft),
            Err(e) => {
                if debug >= 2 {
                    eprintln!(
                        "[fusion-v2-debug] pc drop p={} c={} seam={}: {:?}",
                        p_node.0, c_node.0, seam.0, e
                    );
                }
                if debug >= 1 {
                    rejects.push((super::variant_name(&e), 1));
                }
            }
        }
        if should_emit_keep(gf, seam, c_node, frozen, &ctx.options) {
            match synthesize_producer_consumer(gf, p_node, c_node, seam, FusionVariant::Keep) {
                Ok(draft) => drafts.push(draft),
                Err(e) => {
                    if debug >= 1 {
                        rejects.push((format!("keep:{}", super::variant_name(&e)), 1));
                    }
                }
            }
        }
        (drafts, rejects)
    });
    if debug >= 1 {
        super::dump_rejects("producer-consumer", &rejects);
    }
    out
}

fn disjoint_origins(a: &BTreeSet<NodeId>, b: &BTreeSet<NodeId>) -> bool {
    a.is_disjoint(b)
}

/// Predicate for §10.2's keep-trigger conditions.
pub(super) fn should_emit_keep(
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
