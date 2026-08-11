//! Epilogue fusion — §10.4 (M10).
//!
//! Fuses a pointwise consumer *into its producer's launch schedule*.
//! Producer-consumer fusion (§10.1/§10.2) rebuilds the fused kernel on
//! the *consumer's* schedule and only accepts plain sequential
//! producers; epilogue fusion is its dual for producers that carry a
//! deliberate schedule — a `par`/`threads` mapping, a block hint, or a
//! body shape the producer-consumer recognizer rejects. The producer's
//! top-level `Compute` is retained verbatim (bound, `par`, `threads`,
//! block hint) and the consumer's scalar expression is substituted into
//! the result path right before the store.
//!
//! Legality (§10.4):
//!
//! - the producer is a single-output kernel whose top-level node is a `Compute` with no `scatter`
//!   and a non-`Tuple` body — `par`, `threads`, and block hints are allowed and retained;
//! - the consumer is a single-output *flat pointwise* kernel: recognizable by
//!   [`identify_kernel_shape`], no inner `Compute`/`Reduce`, no block hint of its own, and every
//!   seam read at the identity index `y[k]` (affine-permutation seams are deferred);
//! - producer and consumer share the same symbolic outer bound (the schedule is reused as-is, so no
//!   concrete-bound requirement applies).
//!
//! Both §10.1-style drop and §10.2-style keep variants are emitted; the
//! keep variant materializes the seam as an extra `Tuple` element from
//! the same iteration.
//!
//! Producers that the producer-consumer pass already covers — flat
//! recognizable shape and no block hint — are skipped in [`enumerate`]:
//! for those, both passes would synthesize identical HIR up to the
//! module name, and the duplicate would not dedup at `CandidateKey`
//! time (the name participates in `module_hash`).

use std::collections::HashMap;

use super::{
    horizontal::body_is_flat,
    producer_consumer::{
        find_input_nodes, identify_kernel_shape, remap_size_expr, should_emit_keep, CandidateDraft,
        EnumerateContext, FusionVariant, ReadSite,
    },
};
use crate::{
    graph_ir::{GraphNode, KernelModuleNode},
    ir::{IRBuilder, Module, Node, NodeId as HirNodeId, SizeExpr, VarId},
    passes::{
        fusion_utils::clone_expr,
        fusion_v2::model::{AltGraphNode, GraphFuser, NodeId, ValueClassId},
    },
    quast::{ParSpec, SExpr},
    CompileError,
};

/// Failure modes when trying to synthesize an epilogue candidate from
/// `(producer, consumer)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpilogueFailure {
    /// One of the parents was not a `GraphNode::Kernel`.
    NotAKernel,
    /// The producer's top-level node is not a `Compute`, or it carries a
    /// `scatter`, or its body is a `Tuple`.
    ProducerUnsupportedShape,
    /// The producer produces zero or several outputs.
    ProducerNotSingleOutput,
    /// The consumer module did not fit [`identify_kernel_shape`].
    ConsumerUnsupportedShape,
    /// The consumer body contains an inner `Compute`/`Reduce`.
    ConsumerNotPointwise,
    /// The consumer produces zero or several outputs, or its body is a
    /// `Tuple`.
    ConsumerNotSingleOutput,
    /// The consumer carries its own block hint, which the retained
    /// producer schedule would silently discard.
    ConsumerHasBlockHint,
    /// Producer and consumer have different outer bounds.
    OuterBoundMismatch,
    /// No consumer input position is bound to the seam value.
    NoSeamReadInConsumer,
    /// A consumer read of the seam is not the identity access `y[k]`.
    SeamReadNotIdentity,
    /// Producer and consumer bind the same parameter name to different
    /// values.
    ParamConflict,
    /// Clone-time failure — internal compiler error.
    CloneError(String),
    /// Post-synthesis type inference rejected the module.
    TypeCheckFailed(String),
}

/// Structural facts about an epilogue-eligible producer: a top-level
/// `Compute` whose schedule (bound, `par`, `threads`) is retained
/// verbatim in the fused module.
struct EpilogueProducerShape {
    outer_var: VarId,
    outer_bound: SizeExpr,
    body_root: HirNodeId,
    /// Copied verbatim — a [`ParSpec`]'s `expr` only references its own
    /// `thread`/`seq` binders, so no alpha-renaming is needed.
    par: Option<Box<ParSpec>>,
    threads: Option<usize>,
}

/// Recognizes an epilogue-shaped producer. Unlike
/// [`identify_kernel_shape`] this accepts `par`/`threads` and performs
/// no read-site analysis: the producer body is cloned wholesale, so
/// inputs may be used in any form, not only under `Index`.
fn identify_epilogue_producer(module: &Module) -> Option<EpilogueProducerShape> {
    match module.builder.node(module.body) {
        Node::Compute {
            bound,
            var,
            body,
            scatter,
            par,
            threads,
        } => {
            if scatter.is_some() {
                return None;
            }
            if matches!(module.builder.node(*body), Node::Tuple(_)) {
                return None;
            }
            Some(EpilogueProducerShape {
                outer_var: *var,
                outer_bound: bound.clone(),
                body_root: *body,
                par: par.clone(),
                threads: *threads,
            })
        }
        _ => None,
    }
}

/// Whether the producer-consumer pass would synthesize the same fused
/// HIR for any pair rooted at this producer: flat recognizable shape
/// with no block hint. [`enumerate`] skips such producers to avoid
/// near-duplicate candidates differing only in module name.
fn producer_consumer_covers(module: &Module) -> bool {
    identify_kernel_shape(module).is_some_and(|s| s.is_plain())
        && module.builder.block_hint().is_none()
}

/// Enumerates epilogue candidates in a deterministic order, mirroring
/// the producer-consumer seam loop: for every single-writer seam whose
/// producer is *not* covered by the producer-consumer pass, emit a drop
/// candidate per consumer and — under the §10.2 trigger conditions — a
/// keep sibling.
pub fn enumerate(gf: &GraphFuser, ctx: &EnumerateContext) -> Vec<CandidateDraft> {
    let frozen = ctx.frozen_node_count.min(gf.nodes.len());
    let min_new = ctx.min_new_parent_id;
    let debug = super::debug_reject_level();
    // Sequential site collection (cheap filters + covers check), then
    // parallel synthesis over the collected sites.
    let mut pre_rejects: std::collections::BTreeMap<String, u64> = Default::default();
    let mut sites: Vec<(NodeId, NodeId, ValueClassId)> = Vec::new();
    for (v, producers) in gf.producers.iter().enumerate() {
        for pu in producers {
            let p_node = pu.node;
            if p_node.0 >= frozen {
                continue;
            }
            let GraphNode::Kernel(k) = &gf.nodes[p_node.0].node else {
                continue;
            };
            if producer_consumer_covers(&k.module) {
                if debug >= 1 {
                    *pre_rejects
                        .entry("CoveredByProducerConsumer".into())
                        .or_default() += 1;
                }
                continue;
            }
            for cu in &gf.consumers[v] {
                let c_node = cu.node;
                if c_node.0 >= frozen || c_node == p_node {
                    continue;
                }
                if p_node.0 < min_new && c_node.0 < min_new {
                    continue;
                }
                if !ctx.origins[p_node.0].is_disjoint(&ctx.origins[c_node.0]) {
                    continue;
                }
                sites.push((p_node, c_node, ValueClassId(v)));
            }
        }
    }
    let (out, mut rejects) = super::par_enumerate(sites, |(p_node, c_node, seam)| {
        let mut drafts = Vec::new();
        let mut rejects: Vec<(String, u64)> = Vec::new();
        match synthesize_epilogue(gf, p_node, c_node, seam, FusionVariant::Drop) {
            Ok(draft) => drafts.push(draft),
            Err(e) => {
                if debug >= 1 {
                    rejects.push((super::variant_name(&e), 1));
                }
            }
        }
        if should_emit_keep(gf, seam, c_node, frozen, &ctx.options) {
            match synthesize_epilogue(gf, p_node, c_node, seam, FusionVariant::Keep) {
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
        for (k, n) in pre_rejects {
            *rejects.entry(k).or_default() += n;
        }
        super::dump_rejects("epilogue", &rejects);
    }
    out
}

/// Synthesizes a fused kernel that retains `producer_node`'s launch
/// schedule and substitutes `consumer_node`'s pointwise expression into
/// the result path (§10.4).
///
/// The fused module's boundary follows the producer-consumer
/// convention: producer inputs first, then the consumer's non-seam
/// inputs. Outputs are the consumer's outputs (drop) or the consumer's
/// outputs followed by the seam (keep, `Tuple` body).
pub fn synthesize_epilogue(
    gf: &GraphFuser,
    producer_node: NodeId,
    consumer_node: NodeId,
    seam_val: ValueClassId,
    variant: FusionVariant,
) -> Result<CandidateDraft, EpilogueFailure> {
    let p_alt = &gf.nodes[producer_node.0];
    let c_alt = &gf.nodes[consumer_node.0];
    let (p_module, p_binding) = match &p_alt.node {
        GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
        _ => return Err(EpilogueFailure::NotAKernel),
    };
    let (c_module, c_binding) = match &c_alt.node {
        GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
        _ => return Err(EpilogueFailure::NotAKernel),
    };

    let p_shape =
        identify_epilogue_producer(&p_module).ok_or(EpilogueFailure::ProducerUnsupportedShape)?;
    if p_alt.outputs.len() != 1 {
        return Err(EpilogueFailure::ProducerNotSingleOutput);
    }

    let c_shape = identify_kernel_shape(&c_module)
        .filter(|s| s.is_plain())
        .ok_or(EpilogueFailure::ConsumerUnsupportedShape)?;
    if c_alt.outputs.len() != 1
        || matches!(c_module.builder.node(c_shape.body_root), Node::Tuple(_))
    {
        return Err(EpilogueFailure::ConsumerNotSingleOutput);
    }
    if !body_is_flat(&c_module, c_shape.body_root) {
        return Err(EpilogueFailure::ConsumerNotPointwise);
    }
    if c_module.builder.block_hint().is_some() {
        return Err(EpilogueFailure::ConsumerHasBlockHint);
    }
    if p_shape.outer_bound != c_shape.outer_bound {
        return Err(EpilogueFailure::OuterBoundMismatch);
    }

    // Consumer input position(s) at which the seam is bound, and their
    // read sites. Every seam read must be the identity access `y[k]`.
    let seam_positions: Vec<usize> = c_alt
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (*v == seam_val).then_some(i))
        .collect();
    if seam_positions.is_empty() {
        return Err(EpilogueFailure::NoSeamReadInConsumer);
    }
    let seam_reads: Vec<&ReadSite> = c_shape
        .reads
        .iter()
        .filter(|r| seam_positions.contains(&r.input_pos))
        .collect();
    let identity = [SExpr::sym(c_shape.outer_var)];
    if seam_reads
        .iter()
        .any(|r| r.index_exprs.as_deref() != Some(&identity[..]))
    {
        return Err(EpilogueFailure::SeamReadNotIdentity);
    }

    // Fused-module boundary: producer inputs first, followed by consumer
    // inputs whose positions are not seam positions.
    let mut fb = IRBuilder::new();
    let mut merged_bindings = p_binding.clone();
    for (name, val) in &c_binding {
        match merged_bindings.get(name) {
            Some(existing) if existing != val => {
                return Err(EpilogueFailure::ParamConflict);
            }
            _ => {
                merged_bindings.insert(name.clone(), *val);
            }
        }
    }
    let mut param_map: HashMap<VarId, VarId> = HashMap::new();
    let mut seen_names: HashMap<String, VarId> = HashMap::new();
    for (v, name) in p_module
        .builder
        .params()
        .iter()
        .chain(c_module.builder.params())
    {
        let fresh = *seen_names.entry(name.clone()).or_insert_with(|| {
            let n = fb.var_watermark();
            fb.raise_var_watermark(n + 1);
            fb.inherit_param(VarId(n), name.clone());
            VarId(n)
        });
        param_map.insert(*v, fresh);
    }

    let mut fused_p_input_nodes: Vec<HirNodeId> = Vec::new();
    let mut fused_c_input_nodes: Vec<Option<HirNodeId>> =
        vec![None; c_module.builder.inputs().len()];
    for decl in p_module.builder.inputs().iter() {
        let shape: Vec<SizeExpr> = decl
            .shape
            .iter()
            .map(|d| remap_size_expr(d, &param_map))
            .collect();
        fused_p_input_nodes.push(fb.input(decl.name.clone(), decl.elem, shape));
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
        fused_c_input_nodes[i] = Some(fb.input(decl.name.clone(), decl.elem, shape));
    }

    // The fused outer iteration variable. Both bodies are cloned with
    // their respective outer vars mapped here.
    let k_var = {
        let n = fb.var_watermark();
        fb.raise_var_watermark(n + 1);
        VarId(n)
    };
    let k_var_node = fb.intern(Node::Var(k_var));

    // Clone the producer body once at the identity index. All seam
    // reads are identity, so this single clone serves every read site.
    let producer_input_nodes = find_input_nodes(&p_module, p_shape.body_root);
    let mut producer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
    for (i, &p_input_node) in &producer_input_nodes {
        producer_subst.insert(p_input_node, fused_p_input_nodes[*i]);
    }
    let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
    producer_vars.insert(p_shape.outer_var, k_var_node);
    for (from, to) in &param_map {
        let dst = fb.intern(Node::Var(*to));
        producer_vars.insert(*from, dst);
    }
    let cloned_p = clone_expr(
        &p_module,
        p_shape.body_root,
        &mut fb,
        &producer_subst,
        &producer_vars,
    )
    .map_err(|e| EpilogueFailure::CloneError(format!("{e:?}")))?;

    // Clone the consumer body with every seam `Index` site substituted
    // by the producer clone. `clone_expr` consults `subst` before
    // descending, so a plain NodeId mapping suffices for identity reads
    // — no clone-time hook needed.
    let consumer_input_nodes = find_input_nodes(&c_module, c_shape.body_root);
    let mut consumer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
    for (i, &c_input_node) in &consumer_input_nodes {
        if let Some(fused_input) = fused_c_input_nodes[*i] {
            consumer_subst.insert(c_input_node, fused_input);
        }
    }
    for r in &seam_reads {
        consumer_subst.insert(r.index_node, cloned_p);
    }
    let mut consumer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
    consumer_vars.insert(c_shape.outer_var, k_var_node);
    for (from, to) in &param_map {
        let dst = fb.intern(Node::Var(*to));
        consumer_vars.insert(*from, dst);
    }
    let cloned_c = clone_expr(
        &c_module,
        c_shape.body_root,
        &mut fb,
        &consumer_subst,
        &consumer_vars,
    )
    .map_err(|e| EpilogueFailure::CloneError(format!("{e:?}")))?;

    let compute_body = match variant {
        FusionVariant::Drop => cloned_c,
        FusionVariant::Keep => fb.intern(Node::Tuple(vec![cloned_c, cloned_p])),
    };

    // Retain the producer's schedule: bound, `par`, `threads`, and the
    // block hint all carry over verbatim.
    let fused_body_id = fb.intern(Node::Compute {
        bound: remap_size_expr(&p_shape.outer_bound, &param_map),
        var: k_var,
        body: compute_body,
        scatter: None,
        par: p_shape.par,
        threads: p_shape.threads,
    });
    if let Some(hint) = p_module.builder.block_hint() {
        fb.set_block_hint(hint);
    }
    let name = match variant {
        FusionVariant::Drop => "epilogue_drop",
        FusionVariant::Keep => "epilogue_keep",
    };
    let fused_module = fb.finish(name.to_string(), fused_body_id);

    crate::passes::type_infer(&fused_module).map_err(|e| match e {
        CompileError::Type(m) => EpilogueFailure::TypeCheckFailed(m),
        other => EpilogueFailure::TypeCheckFailed(other.to_string()),
    })?;

    let mut fused_inputs: Vec<ValueClassId> = p_alt.inputs.clone();
    for (i, &v) in c_alt.inputs.iter().enumerate() {
        if !seam_positions.contains(&i) {
            fused_inputs.push(v);
        }
    }
    let mut fused_outputs: Vec<ValueClassId> = c_alt.outputs.clone();
    if variant == FusionVariant::Keep {
        fused_outputs.push(seam_val);
    }

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
