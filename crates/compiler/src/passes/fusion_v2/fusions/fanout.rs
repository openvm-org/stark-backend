//! Fanout fusion (M7 — §10.5).
//!
//! A fanout targets a producer whose seam is read by two or more
//! consumers. The synthesized kernel evaluates the producer expression
//! **once per outer index** and threads that value into every consumer
//! body as a shared HIR NodeId — hash-consing folds every reference to
//! a single `NodeId`, so the store-and-reload chain each consumer would
//! otherwise pay disappears and the producer expression appears exactly
//! once in the fused HIR.
//!
//! Composition (§10.5):
//!
//! ```text
//! inputs  = stable_unique(producer.inputs ++ each c_i.inputs \ seam)
//! outputs = Drop:  (producer.outputs \ seam) ++ concat(c_1.outputs, .., c_k.outputs)
//!           Keep:  producer.outputs ++ concat(c_1.outputs, .., c_k.outputs)
//! ```
//!
//! Multi-output producers (e.g. keep-variant fused kernels) are handled
//! by grouping consumers per `(producer, output)` pair: each output is
//! an independent fanout seam. The seam element of the producer's body
//! tuple feeds the consumers; sibling elements stay materialized in the
//! fused tuple under both variants.
//!
//! The M7 slice restricts to:
//!
//! - **identity** seam reads at every consumer — every reachable `y[j]` in a consumer must resolve
//!   to `outer_var(j)`. Non-identity permutations force per-site producer re-evaluation and defeat
//!   fanout's "compute once" invariant; those cases fall back to producer-consumer for individual
//!   `(p, c)` pairs.
//!
//! Legality (§10.5):
//!
//! - producer and every consumer identify as a valid [`KernelShape`];
//! - every consumer's outer bound matches the producer's outer bound;
//! - every consumer reads the seam only through identity access;
//! - no consumer reads another consumer's output value class (direct dependency check);
//! - origins of producer and all consumers are pairwise disjoint (enforced by
//!   [`super::producer_consumer::EnumerateContext::origins`]).
//!
//! Anti-pattern rejection (§10.5): if two producer-consumer candidates
//! sharing a producer are horizontally merged, the merged HIR contains
//! the producer expression twice. Fanout construction clones the
//! producer body once and hands the resulting NodeId to every consumer
//! read via a `clone_expr_with_hook`, so hash-consing (and the child
//! walk `module_hash` performs) finds a single occurrence. The M7 tests
//! assert this by comparing to a hand-authored dual-consumer reference
//! module.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{
    graph_ir::{BufId, GraphNode, KernelModuleNode},
    ir::{IRBuilder, Module, Node, NodeId as HirNodeId, SizeExpr, VarId},
    passes::{
        fusion_utils::{clone_expr_with_params, remap_size_expr, CloneError},
        fusion_v2::{
            fusions::producer_consumer::{
                identify_kernel_shape, CandidateDraft, EnumerateContext, FusionVariant,
                KernelShape, ReadSite,
            },
            model::{AltGraphNode, GraphFuser, NodeId, UseInfo, ValueClassId},
        },
    },
    quast::SExpr,
    CompileError,
};

/// Failure modes when trying to synthesize a fanout candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FanoutFailure {
    NotAKernel,
    UnsupportedShape,
    NotEnoughConsumers,
    OuterBoundMismatch,
    SeamReadNotIdentity,
    ConsumerToConsumerDependency,
    ParamNameConflict,
    CloneError(String),
    TypeCheckFailed(String),
}

/// Runs the fanout pass over the frozen prefix in `ctx`, emitting one
/// candidate per legal `(producer, consumer set)` grouping, plus a keep
/// variant when the seam has another user outside the fanout (§10.5).
pub fn enumerate(gf: &GraphFuser, ctx: &EnumerateContext) -> Vec<CandidateDraft> {
    let frozen = ctx.frozen_node_count.min(gf.nodes.len());
    let debug = super::debug_reject_level();
    // Sequential group collection (cheap consumer scans + origin
    // checks), then parallel synthesis over the collected groups.
    // A site is `(producer, sorted consumer set, seam)`.
    type FanoutSite = (NodeId, Vec<(NodeId, Vec<usize>)>, ValueClassId);
    let mut pre_rejects: std::collections::BTreeMap<String, u64> = Default::default();
    let mut sites: Vec<FanoutSite> = Vec::new();
    'outer: for p_id in 0..frozen {
        if let Some(t) = ctx.deadline {
            if std::time::Instant::now() >= t {
                break 'outer;
            }
        }
        let p_node = NodeId(p_id);
        // Only kernels can be producers in fanout.
        if !matches!(&gf.nodes[p_id].node, GraphNode::Kernel(_)) {
            continue;
        }

        // Consumers are grouped per `(producer, output)` pair: each
        // output of a multi-output producer (e.g. a keep-variant fused
        // kernel) is an independent fanout seam.
        for &seam in &gf.nodes[p_id].outputs {
            // Collect eligible consumers of `seam` within the frozen prefix.
            let mut candidates: Vec<(NodeId, Vec<usize>)> = Vec::new();
            for cu in &gf.consumers[seam.0] {
                if cu.node.0 >= frozen {
                    continue;
                }
                if cu.node == p_node {
                    continue;
                }
                if !matches!(&gf.nodes[cu.node.0].node, GraphNode::Kernel(_)) {
                    continue;
                }
                match candidates.iter_mut().find(|(id, _)| *id == cu.node) {
                    Some((_, positions)) => positions.push(cu.pos),
                    None => candidates.push((cu.node, vec![cu.pos])),
                }
            }
            if candidates.len() < 2 {
                continue;
            }
            // Only consider this group if at least one participant is new
            // since the last saturation round; otherwise the same fanout
            // was already emitted.
            let all_old = candidates.iter().all(|(c, _)| c.0 < ctx.min_new_parent_id)
                && p_id < ctx.min_new_parent_id;
            if all_old {
                continue;
            }
            // Origin-disjointness across the full parent set.
            let mut origin_union: std::collections::BTreeSet<NodeId> =
                std::collections::BTreeSet::new();
            let mut origins_ok = true;
            for seed in &ctx.origins[p_id] {
                if !origin_union.insert(*seed) {
                    origins_ok = false;
                    break;
                }
            }
            if origins_ok {
                for (c, _) in &candidates {
                    for seed in &ctx.origins[c.0] {
                        if !origin_union.insert(*seed) {
                            origins_ok = false;
                            break;
                        }
                    }
                    if !origins_ok {
                        break;
                    }
                }
            }
            if !origins_ok {
                if debug >= 1 {
                    *pre_rejects.entry("OriginOverlap".into()).or_default() += 1;
                }
                continue;
            }
            // Deterministic order: consumers sorted by NodeId.
            candidates.sort_by_key(|(c, _)| c.0);
            sites.push((p_node, candidates, seam));
        }
    }
    let (out, mut rejects) =
        super::par_enumerate(sites, ctx.deadline, |(p_node, candidates, seam)| {
            let mut drafts = Vec::new();
            let mut rejects: Vec<(String, u64)> = Vec::new();
            match synthesize_fanout(gf, p_node, &candidates, seam, FusionVariant::Drop) {
                Ok(draft) => drafts.push(draft),
                Err(e) => {
                    if debug >= 1 {
                        rejects.push((super::variant_name(&e), 1));
                    }
                }
            }
            if should_emit_keep(gf, seam, &candidates, frozen) {
                match synthesize_fanout(gf, p_node, &candidates, seam, FusionVariant::Keep) {
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
        super::dump_rejects("fanout", &rejects);
    }
    out
}

/// §10.2 keep trigger, restated for fanout: emit keep when the seam has
/// consumers outside the fanout, or is a registered graph output.
fn should_emit_keep(
    gf: &GraphFuser,
    seam: ValueClassId,
    fanout_consumers: &[(NodeId, Vec<usize>)],
    frozen: usize,
) -> bool {
    if gf.outputs.contains(&seam) {
        return true;
    }
    let fanout_set: HashSet<NodeId> = fanout_consumers.iter().map(|(c, _)| *c).collect();
    for u in &gf.consumers[seam.0] {
        if u.node.0 >= frozen {
            continue;
        }
        if !fanout_set.contains(&u.node) {
            return true;
        }
    }
    false
}

/// Builds the fanout candidate. See the module docs for structure.
pub fn synthesize_fanout(
    gf: &GraphFuser,
    producer_node: NodeId,
    consumers: &[(NodeId, Vec<usize>)],
    seam: ValueClassId,
    variant: FusionVariant,
) -> Result<CandidateDraft, FanoutFailure> {
    if consumers.len() < 2 {
        return Err(FanoutFailure::NotEnoughConsumers);
    }
    // Producer must be a Kernel with recognizable shape.
    let p_alt = &gf.nodes[producer_node.0];
    let (p_module, p_binding) = match &p_alt.node {
        GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
        _ => return Err(FanoutFailure::NotAKernel),
    };
    let Some(seam_out_idx) = p_alt.outputs.iter().position(|&v| v == seam) else {
        debug_assert!(false, "seam value is not an output of the producer node");
        return Err(FanoutFailure::UnsupportedShape);
    };
    let p_shape = identify_kernel_shape(&p_module)
        .filter(|s| s.is_plain())
        .ok_or(FanoutFailure::UnsupportedShape)?;
    // Per-output tuple elements of the producer body. Multi-output
    // producers (keep-variant fused kernels) expose one element per
    // output; the seam element feeds the consumers and the sibling
    // elements stay materialized in the fused tuple.
    let p_elems: Vec<HirNodeId> = match p_module.builder.node(p_shape.body_root) {
        Node::Tuple(es) if p_alt.outputs.len() > 1 => es.clone(),
        _ => vec![p_shape.body_root],
    };
    if p_elems.len() != p_alt.outputs.len() {
        return Err(FanoutFailure::UnsupportedShape);
    }
    for (i, &e) in p_elems.iter().enumerate() {
        let materialized = variant == FusionVariant::Keep || i != seam_out_idx;
        if materialized && matches!(p_module.builder.node(e), Node::Compute { .. }) {
            // A bare tile-valued element cannot be re-materialized
            // through the fused scalar tuple.
            return Err(FanoutFailure::UnsupportedShape);
        }
    }

    // Every consumer must recognize + share bound + read seam by identity.
    struct ConsumerRec {
        alt_idx: usize,
        module: Arc<crate::ir::Module>,
        bindings: BTreeMap<String, i64>,
        shape: KernelShape,
        seam_positions: Vec<usize>,
    }
    let mut consumer_recs = Vec::with_capacity(consumers.len());
    for (c_id, seam_positions) in consumers {
        let c_alt = &gf.nodes[c_id.0];
        let (c_module, c_binding) = match &c_alt.node {
            GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
            _ => return Err(FanoutFailure::NotAKernel),
        };
        let c_shape = identify_kernel_shape(&c_module)
            .filter(|s| s.is_plain())
            .ok_or(FanoutFailure::UnsupportedShape)?;
        if c_shape.outer_bound != p_shape.outer_bound {
            return Err(FanoutFailure::OuterBoundMismatch);
        }
        // Every reachable seam read must be identity access.
        for r in &c_shape.reads {
            if !seam_positions.contains(&r.input_pos) {
                continue;
            }
            if !read_is_identity(r, c_shape.outer_var) {
                return Err(FanoutFailure::SeamReadNotIdentity);
            }
        }
        consumer_recs.push(ConsumerRec {
            alt_idx: c_id.0,
            module: c_module,
            bindings: c_binding,
            shape: c_shape,
            seam_positions: seam_positions.clone(),
        });
    }

    // No consumer's outputs may appear in another consumer's inputs
    // (direct dependency check).
    let consumer_output_sets: Vec<HashSet<ValueClassId>> = consumer_recs
        .iter()
        .map(|r| gf.nodes[r.alt_idx].outputs.iter().copied().collect())
        .collect();
    for (i, r_i) in consumer_recs.iter().enumerate() {
        for v in &gf.nodes[r_i.alt_idx].inputs {
            for (j, out_set_j) in consumer_output_sets.iter().enumerate() {
                if j == i {
                    continue;
                }
                if out_set_j.contains(v) {
                    return Err(FanoutFailure::ConsumerToConsumerDependency);
                }
            }
        }
    }

    // Build the fused module.
    let mut fb = IRBuilder::new();

    // Merged param bindings + name-keyed var remap. Same as
    // producer_consumer. The remap is per parent module: each has its own
    // VarId namespace, so a single map keyed by source VarId would let
    // one parent's param clobber another's when the same numeric id
    // names different params.
    let mut merged_bindings = p_binding.clone();
    let mut seen_names: HashMap<String, VarId> = HashMap::new();
    let mut normalize = |fb: &mut IRBuilder, module: &Module| {
        let mut map: HashMap<VarId, VarId> = HashMap::new();
        for (v, name) in module.builder.params() {
            let fresh = *seen_names.entry(name.clone()).or_insert_with(|| {
                let n = fb.var_watermark();
                fb.raise_var_watermark(n + 1);
                fb.inherit_param(VarId(n), name.clone());
                VarId(n)
            });
            map.insert(*v, fresh);
        }
        map
    };
    let p_param_map = normalize(&mut fb, &p_module);
    let mut c_param_maps: Vec<HashMap<VarId, VarId>> = Vec::with_capacity(consumer_recs.len());
    for rec in &consumer_recs {
        for (name, val) in &rec.bindings {
            match merged_bindings.get(name) {
                Some(existing) if existing != val => {
                    return Err(FanoutFailure::ParamNameConflict);
                }
                _ => {
                    merged_bindings.insert(name.clone(), *val);
                }
            }
        }
        c_param_maps.push(normalize(&mut fb, &rec.module));
    }

    // Boundary: producer inputs first, then each consumer's non-seam inputs,
    // stable-unique by ValueClassId. We also record, per parent local
    // input position, which fused input position it maps to.
    let mut fused_input_values: Vec<ValueClassId> = Vec::new();
    let mut fused_input_present: HashSet<ValueClassId> = HashSet::new();
    let mut producer_local_input_map: HashMap<usize, usize> = HashMap::new();
    let mut consumer_local_input_map: Vec<HashMap<usize, usize>> =
        vec![HashMap::new(); consumer_recs.len()];
    for (pi, &v) in p_alt.inputs.iter().enumerate() {
        if fused_input_present.insert(v) {
            producer_local_input_map.insert(pi, fused_input_values.len());
            fused_input_values.push(v);
        } else {
            let existing = fused_input_values.iter().position(|&x| x == v).unwrap();
            producer_local_input_map.insert(pi, existing);
        }
    }
    for (ci, rec) in consumer_recs.iter().enumerate() {
        for (pos, &v) in gf.nodes[rec.alt_idx].inputs.iter().enumerate() {
            if rec.seam_positions.contains(&pos) {
                continue;
            }
            if fused_input_present.insert(v) {
                consumer_local_input_map[ci].insert(pos, fused_input_values.len());
                fused_input_values.push(v);
            } else {
                let existing = fused_input_values.iter().position(|&x| x == v).unwrap();
                consumer_local_input_map[ci].insert(pos, existing);
            }
        }
    }

    // Declare inputs in the fused module and record the Input NodeIds.
    // For each fused position, find the first parent (producer or a
    // consumer) that maps to it and use that parent's InputDecl.
    let mut fused_input_nodes: Vec<HirNodeId> = Vec::with_capacity(fused_input_values.len());
    for fused_pos in 0..fused_input_values.len() {
        let mut decl: Option<(&str, crate::ir::ScalarType, Vec<SizeExpr>)> = None;
        for (pi, &fp) in &producer_local_input_map {
            if fp == fused_pos {
                let d = &p_module.builder.inputs()[*pi];
                let shape: Vec<SizeExpr> = d
                    .shape
                    .iter()
                    .map(|s| remap_size_expr(s, &p_param_map))
                    .collect();
                decl = Some((d.name.as_str(), d.elem, shape));
                break;
            }
        }
        if decl.is_none() {
            for (ci, cm) in consumer_local_input_map.iter().enumerate() {
                let mut done = false;
                for (pos, &fp) in cm {
                    if fp == fused_pos {
                        let d = &consumer_recs[ci].module.builder.inputs()[*pos];
                        let shape: Vec<SizeExpr> = d
                            .shape
                            .iter()
                            .map(|s| remap_size_expr(s, &c_param_maps[ci]))
                            .collect();
                        decl = Some((d.name.as_str(), d.elem, shape));
                        done = true;
                        break;
                    }
                }
                if done {
                    break;
                }
            }
        }
        let (name, elem, shape) =
            decl.expect("every fused input position must come from at least one parent");
        let n = fb.input(name.to_string(), elem, shape);
        fused_input_nodes.push(n);
    }

    // Fused compute's outer var + Node::Var.
    let outer_bound_fb = remap_size_expr(&p_shape.outer_bound, &p_param_map);
    let k_var = {
        let n = fb.var_watermark();
        fb.raise_var_watermark(n + 1);
        VarId(n)
    };
    let k_var_node = fb.intern(Node::Var(k_var));

    // ---- Step A: clone the producer body once with outer_var -> k_var_node.
    let producer_input_nodes: HashMap<usize, HirNodeId> =
        find_input_nodes(&p_module, p_shape.body_root);
    let mut producer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
    for (pi, &p_input_node) in &producer_input_nodes {
        let fused_pos = producer_local_input_map[pi];
        producer_subst.insert(p_input_node, fused_input_nodes[fused_pos]);
    }
    let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
    producer_vars.insert(p_shape.outer_var, k_var_node);
    // The whole body is cloned once (not per element) so intra-producer
    // subexpression sharing between the seam and its siblings survives
    // into the fused HIR. `clone_expr_with_params` alpha-renames the
    // source module's param VarIds inside `ConstSym` payloads and
    // `Compute`/`Reduce` bounds; a plain clone would leave the source
    // VarIds in place, silently aliasing them to whichever fused param
    // got that numeric id.
    let cloned_p_root = clone_expr_with_params(
        &p_module,
        p_shape.body_root,
        &mut fb,
        &producer_subst,
        &producer_vars,
        &p_param_map,
        |_, _, _| Ok(None),
    )
    .map_err(|e| FanoutFailure::CloneError(format!("{e:?}")))?;
    let cloned_p_elems: Vec<HirNodeId> = if p_alt.outputs.len() > 1 {
        match fb.node(cloned_p_root) {
            Node::Tuple(es) => es.clone(),
            _ => {
                return Err(FanoutFailure::CloneError(
                    "multi-output producer body did not clone to a tuple".into(),
                ))
            }
        }
    } else {
        vec![cloned_p_root]
    };
    let seam_body_node = cloned_p_elems[seam_out_idx];

    // ---- Step B: for each consumer, clone its body with the seam-read
    // hook returning `seam_body_node` directly. Hash-consing means every
    // seam read across every consumer collapses to the same NodeId, so
    // the producer expression appears exactly once in the fused HIR
    // (§10.5 anti-pattern rejection): downstream canonicalize sees one
    // instance shared by all consumers.
    let mut cloned_consumer_bodies: Vec<HirNodeId> = Vec::with_capacity(consumer_recs.len());
    for (ci, rec) in consumer_recs.iter().enumerate() {
        let consumer_input_nodes: HashMap<usize, HirNodeId> =
            find_input_nodes(&rec.module, rec.shape.body_root);
        let mut consumer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
        for (pos, &c_input_node) in &consumer_input_nodes {
            if let Some(&fused_pos) = consumer_local_input_map[ci].get(pos) {
                consumer_subst.insert(c_input_node, fused_input_nodes[fused_pos]);
            }
        }
        let mut consumer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
        consumer_vars.insert(rec.shape.outer_var, k_var_node);
        // Fast lookup for this consumer's seam-read Index NodeIds.
        let seam_read_nodes: HashSet<HirNodeId> = rec
            .shape
            .reads
            .iter()
            .filter(|r| rec.seam_positions.contains(&r.input_pos))
            .map(|r| r.index_node)
            .collect();

        let hook = |_dst: &mut IRBuilder,
                    src_id: HirNodeId,
                    _vars: &HashMap<VarId, HirNodeId>|
         -> Result<Option<HirNodeId>, CloneError> {
            if seam_read_nodes.contains(&src_id) {
                Ok(Some(seam_body_node))
            } else {
                Ok(None)
            }
        };
        let cloned = clone_expr_with_params(
            &rec.module,
            rec.shape.body_root,
            &mut fb,
            &consumer_subst,
            &consumer_vars,
            &c_param_maps[ci],
            hook,
        )
        .map_err(|e| FanoutFailure::CloneError(format!("{e:?}")))?;
        cloned_consumer_bodies.push(cloned);
    }

    // ---- Step C: assemble the fused compute body.
    //
    // Body:
    //     Tuple([p_elems \ seam, c_i(k, seam)...])   — drop
    //     Tuple([p_elems, c_i(k, seam)...])          — keep
    //     compute[N] |k| { <body> }
    //
    // Sibling outputs of a multi-output producer stay materialized in
    // both variants; only the seam element is dropped under Drop.
    let mut tuple_elems: Vec<HirNodeId> = Vec::new();
    for (i, &e) in cloned_p_elems.iter().enumerate() {
        if variant == FusionVariant::Keep || i != seam_out_idx {
            tuple_elems.push(e);
        }
    }
    tuple_elems.extend(cloned_consumer_bodies.iter().copied());
    let compute_body = if tuple_elems.len() == 1 {
        tuple_elems[0]
    } else {
        fb.intern(Node::Tuple(tuple_elems))
    };
    let fused_body_id = fb.intern(Node::Compute {
        bound: outer_bound_fb,
        var: k_var,
        body: compute_body,
        scatter: None,
        par: None,
        threads: None,
    });
    let name = match variant {
        FusionVariant::Drop => "fanout_drop",
        FusionVariant::Keep => "fanout_keep",
    };
    let fused_module = fb.finish(name.to_string(), fused_body_id);

    crate::passes::type_infer(&fused_module).map_err(|e| match e {
        CompileError::Type(m) => FanoutFailure::TypeCheckFailed(m),
        other => FanoutFailure::TypeCheckFailed(other.to_string()),
    })?;

    // Boundary value bindings.
    let fused_inputs = fused_input_values.clone();
    let mut fused_outputs: Vec<ValueClassId> = Vec::new();
    for (i, &v) in p_alt.outputs.iter().enumerate() {
        if variant == FusionVariant::Keep || i != seam_out_idx {
            fused_outputs.push(v);
        }
    }
    for rec in &consumer_recs {
        fused_outputs.extend(gf.nodes[rec.alt_idx].outputs.iter().copied());
    }

    // Physical BufIds.
    let input_bufs: Vec<BufId> = fused_inputs.iter().map(|&v| gf.physical(v)).collect();
    let output_bufs: Vec<BufId> = fused_outputs.iter().map(|&v| gf.physical(v)).collect();

    let node = GraphNode::Kernel(KernelModuleNode {
        module: Arc::new(fused_module),
        param_bindings: merged_bindings,
        inputs: input_bufs,
        outputs: output_bufs,
        types: None,
        hash: None,
        canonical: false,
        fusion_history: None,
    });

    let mut parents = vec![producer_node];
    parents.extend(consumer_recs.iter().map(|r| NodeId(r.alt_idx)));

    Ok(CandidateDraft {
        parents,
        variant,
        alt: AltGraphNode {
            inputs: fused_inputs,
            outputs: fused_outputs,
            node,
        },
    })
}

// -----------------------------------------------------------------------
// Helpers.
// -----------------------------------------------------------------------

/// Whether a read's index expression is exactly the enclosing outer
/// var (rank-1 identity access at the outer scope).
fn read_is_identity(read: &ReadSite, outer_var: VarId) -> bool {
    // Nested-scope reads (inside inner Compute/Reduce) are not identity
    // at the outer var; reject those too.
    if read.inner_vars.len() != 1 || read.inner_vars[0] != outer_var {
        return false;
    }
    read.index_exprs.as_deref() == Some(&[SExpr::sym(outer_var)][..])
}

fn find_input_nodes(module: &crate::ir::Module, root: HirNodeId) -> HashMap<usize, HirNodeId> {
    use crate::module_hash::children_of;
    let mut out = HashMap::new();
    let mut work = vec![root];
    let mut seen = HashSet::new();
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

// Silence unused-import warnings when the trait system doesn't need them.
#[allow(dead_code)]
type _MarkerUseInfo = UseInfo;
