//! Same-domain horizontal fusion (M9 — §10.6).
//!
//! Merges two kernels with **no dataflow relation** into one kernel
//! that executes both bodies at the same logical index and returns the
//! concatenated tuple of outputs:
//!
//! ```text
//! inputs  = stable_unique(a.inputs ++ b.inputs)
//! outputs = concat(a.outputs, b.outputs)
//! body    = compute[N] |k| Tuple(elems(a) ++ elems(b))
//! ```
//!
//! There is no seam and therefore no drop/keep distinction; the draft
//! carries [`FusionVariant::Drop`] as a diagnostic placeholder.
//!
//! The pass is deliberately narrow (§10.6):
//!
//! - **equal concrete outer domain** — both kernels iterate the same constant bound. Symbolic
//!   bounds and `compute[max(Na, Nb)]` masking are rejected: dense output lowering would write the
//!   smaller buffer out of bounds (§10.6 last paragraph).
//! - **equal block hint and thread geometry** — [`identify_kernel_shape`] already rejects `scatter`
//!   / `par` / `threads` on the outer compute, so geometry reduces to the modules' builder-level
//!   block hints, which must be equal (and are propagated to the fused module).
//! - **flat structured kernels** — no inner `Compute`/`Reduce` anywhere in either body. This also
//!   excludes shared-memory tiles (`Let`-bound inner computes) and the syncs they lower to.
//! - **no dataflow path in either direction** — checked with the §9.1 reachability primitive
//!   ([`would_create_cycle`]) over the union boundary: any directed path from one kernel's outputs
//!   to the other's inputs (direct or transitive, through any alternative) rejects the pair.
//!   Extraction has no acyclicity constraints (§13.7), so this must be conservative.
//! - **no storage hazard between regions** — the two kernels' physical read/write footprints must
//!   not conflict (write∩write, write∩read on physical `BufId`s). A fused node executes both
//!   regions concurrently, so cross-region WAW/WAR ordering cannot be recovered by the §7 hazard
//!   sort.
//! - **disjoint origins** — enforced by the enumeration context (§9.1 obligation).
//!
//! Identical normalized loads are shared only through hash-consing:
//! when both bodies read the same fused input at the same index, the
//! `Index` node interns to one `NodeId` and cost extraction observes
//! the sharing (§10.6). No explicit CSE is performed.
//!
//! Multi-output parents (keep-variant or previously horizontally-fused
//! kernels) participate: a parent with `k > 1` outputs must have a
//! `Tuple` body of arity `k`, whose elements are spliced positionally
//! into the fused tuple. This is what lets bounded saturation compose
//! three-way merges across rounds.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use crate::{
    graph_ir::{BufId, GraphNode, KernelModuleNode},
    ir::{IRBuilder, Module, Node, NodeId as HirNodeId, ScalarType, SizeExpr, VarId},
    module_hash::children_of,
    passes::{
        fusion_utils::clone_expr,
        fusion_v2::{
            fusions::producer_consumer::{
                find_input_nodes, identify_kernel_shape, remap_size_expr, CandidateDraft,
                EnumerateContext, FusionVariant, KernelShape,
            },
            model::{AltGraphNode, GraphFuser, NodeId, ValueClassId},
            validate::would_create_cycle,
        },
    },
    CompileError,
};

/// Failure modes when trying to synthesize a horizontal candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HorizontalFailure {
    NotAKernel,
    UnsupportedShape,
    /// The body contains an inner `Compute` or `Reduce` (§10.6 requires
    /// flat structured kernels; this also excludes shared-memory tiles).
    NotFlat,
    NonConstantBound,
    OuterBoundMismatch,
    BlockHintMismatch,
    /// A parent with `k > 1` outputs did not have a `Tuple` body of
    /// arity `k` (or a single-output parent had a `Tuple` body).
    TupleArityMismatch,
    /// A directed dataflow path exists between the two kernels.
    DataflowPath,
    /// The kernels' physical read/write footprints conflict.
    StorageHazard,
    ParamNameConflict,
    CloneError(String),
    TypeCheckFailed(String),
}

/// Per-node eligibility facts computed once per enumeration round.
struct Eligibility {
    outer_bound: i64,
    block_hint: Option<usize>,
}

/// Runs the horizontal pass over the frozen prefix in `ctx`, emitting
/// one candidate per legal unordered pair `(a, b)` with `a < b`.
pub fn enumerate(gf: &GraphFuser, ctx: &EnumerateContext) -> Vec<CandidateDraft> {
    use rayon::prelude::*;
    let frozen = ctx.frozen_node_count.min(gf.nodes.len());
    let debug = super::debug_reject_level();
    let elig: Vec<Option<Eligibility>> = (0..frozen)
        .into_par_iter()
        .map(|i| node_eligibility(gf, NodeId(i)))
        .collect();
    if debug >= 1 {
        let eligible = elig.iter().filter(|e| e.is_some()).count();
        eprintln!("[fusion-v2-debug] horizontal: eligible={eligible}/{frozen} nodes");
    }
    // One parallel task per row `a`; the inner `b` loop stays
    // sequential within the task (rayon work-stealing balances the
    // triangular row lengths).
    let rows: Vec<usize> = (0..frozen).filter(|&a| elig[a].is_some()).collect();
    let elig = &elig;
    let (out, rejects) = super::par_enumerate(rows, ctx.deadline, |a| {
        let ea = elig[a].as_ref().expect("rows filtered to Some");
        let mut drafts = Vec::new();
        let mut row_rejects: std::collections::BTreeMap<String, u64> = Default::default();
        for (b, eb) in elig.iter().enumerate().skip(a + 1) {
            let Some(eb) = eb else { continue };
            // At least one parent must be new since the last round.
            if a < ctx.min_new_parent_id && b < ctx.min_new_parent_id {
                continue;
            }
            if ea.outer_bound != eb.outer_bound || ea.block_hint != eb.block_hint {
                continue;
            }
            if !ctx.origins[a].is_disjoint(&ctx.origins[b]) {
                continue;
            }
            match synthesize_horizontal(gf, NodeId(a), NodeId(b)) {
                Ok(draft) => drafts.push(draft),
                Err(e) => {
                    if debug >= 1 {
                        *row_rejects.entry(super::variant_name(&e)).or_default() += 1;
                    }
                }
            }
        }
        (drafts, row_rejects.into_iter().collect())
    });
    if debug >= 1 {
        super::dump_rejects("horizontal", &rejects);
    }
    out
}

/// Cheap per-node prefilter: kernel, recognizable shape, flat body,
/// concrete outer bound, and output arity consistent with the body.
fn node_eligibility(gf: &GraphFuser, node: NodeId) -> Option<Eligibility> {
    let alt = &gf.nodes[node.0];
    let module = match &alt.node {
        GraphNode::Kernel(k) => &k.module,
        _ => return None,
    };
    let shape = identify_kernel_shape(module).filter(|s| s.is_plain())?;
    let outer_bound = shape.outer_bound.as_const()?;
    if !body_is_flat(module, shape.body_root) {
        return None;
    }
    tuple_elem_count(module, shape.body_root, alt.outputs.len())?;
    Some(Eligibility {
        outer_bound,
        block_hint: module.builder.block_hint(),
    })
}

/// `Some(arity)` when the body root is consistent with `n_outputs`:
/// single-output kernels must have a non-`Tuple` body; multi-output
/// kernels must have a `Tuple` body of matching arity.
fn tuple_elem_count(module: &Module, body_root: HirNodeId, n_outputs: usize) -> Option<usize> {
    if n_outputs == 0 {
        return None;
    }
    match module.builder.node(body_root) {
        Node::Tuple(elems) => (n_outputs > 1 && elems.len() == n_outputs).then_some(elems.len()),
        _ => (n_outputs == 1).then_some(1),
    }
}

/// Whether the body contains no inner `Compute` / `Reduce`.
pub(super) fn body_is_flat(module: &Module, root: HirNodeId) -> bool {
    let mut work = vec![root];
    let mut seen: HashSet<HirNodeId> = HashSet::new();
    while let Some(id) = work.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = module.builder.node(id);
        if matches!(node, Node::Compute { .. } | Node::Reduce { .. }) {
            return false;
        }
        work.extend(children_of(node));
    }
    true
}

/// Whether the two nodes' physical storage footprints conflict:
/// write∩write (WAW) or write∩read in either direction (WAR/RAW at
/// differing versions). Value-class-level RAW (same version) shows up
/// as a dataflow path instead and is caught by [`has_dataflow_path`].
fn has_storage_hazard(gf: &GraphFuser, a: NodeId, b: NodeId) -> bool {
    let phys =
        |vs: &[ValueClassId]| -> HashSet<usize> { vs.iter().map(|&v| gf.physical(v).0).collect() };
    let writes_a = phys(&gf.nodes[a.0].outputs);
    let writes_b = phys(&gf.nodes[b.0].outputs);
    let reads_a = phys(&gf.nodes[a.0].inputs);
    let reads_b = phys(&gf.nodes[b.0].inputs);
    !writes_a.is_disjoint(&writes_b)
        || !writes_a.is_disjoint(&reads_b)
        || !writes_b.is_disjoint(&reads_a)
}

/// Whether a directed dataflow path exists between `a` and `b` in
/// either direction. Reuses the §9.1 reachability primitive on the
/// union boundary: a path from any of `{a,b}`'s outputs to any of
/// `{a,b}`'s inputs is exactly a path `a → b` or `b → a` (a node cannot
/// reach its own inputs in a DAG).
fn has_dataflow_path(gf: &GraphFuser, a: NodeId, b: NodeId) -> bool {
    let mut inputs = gf.nodes[a.0].inputs.clone();
    inputs.extend(gf.nodes[b.0].inputs.iter().copied());
    let mut outputs = gf.nodes[a.0].outputs.clone();
    outputs.extend(gf.nodes[b.0].outputs.iter().copied());
    would_create_cycle(gf, &inputs, &outputs)
}

/// Builds the horizontal candidate for the pair `(a_node, b_node)`.
/// Performs the full §10.6 legality check; [`enumerate`] prefilters
/// only the cheap per-node facts.
pub fn synthesize_horizontal(
    gf: &GraphFuser,
    a_node: NodeId,
    b_node: NodeId,
) -> Result<CandidateDraft, HorizontalFailure> {
    struct Part {
        alt_idx: usize,
        module: Arc<Module>,
        bindings: BTreeMap<String, i64>,
        shape: KernelShape,
    }

    let mut parts: Vec<Part> = Vec::with_capacity(2);
    for n in [a_node, b_node] {
        let alt = &gf.nodes[n.0];
        let (module, bindings) = match &alt.node {
            GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
            _ => return Err(HorizontalFailure::NotAKernel),
        };
        let shape = identify_kernel_shape(&module)
            .filter(|s| s.is_plain())
            .ok_or(HorizontalFailure::UnsupportedShape)?;
        if shape.outer_bound.as_const().is_none() {
            return Err(HorizontalFailure::NonConstantBound);
        }
        if !body_is_flat(&module, shape.body_root) {
            return Err(HorizontalFailure::NotFlat);
        }
        if tuple_elem_count(&module, shape.body_root, alt.outputs.len()).is_none() {
            return Err(HorizontalFailure::TupleArityMismatch);
        }
        parts.push(Part {
            alt_idx: n.0,
            module,
            bindings,
            shape,
        });
    }
    if parts[0].shape.outer_bound != parts[1].shape.outer_bound {
        return Err(HorizontalFailure::OuterBoundMismatch);
    }
    if parts[0].module.builder.block_hint() != parts[1].module.builder.block_hint() {
        return Err(HorizontalFailure::BlockHintMismatch);
    }
    // Dataflow first: a direct producer→consumer pair also overlaps on
    // physical storage, and DataflowPath is the more precise diagnosis.
    if has_dataflow_path(gf, a_node, b_node) {
        return Err(HorizontalFailure::DataflowPath);
    }
    if has_storage_hazard(gf, a_node, b_node) {
        return Err(HorizontalFailure::StorageHazard);
    }

    // Build the fused module.
    let mut fb = IRBuilder::new();
    if let Some(hint) = parts[0].module.builder.block_hint() {
        fb.set_block_hint(hint);
    }

    // Merged param bindings + name-keyed var remap. Same pattern as
    // producer_consumer / fanout.
    let mut merged_bindings: BTreeMap<String, i64> = BTreeMap::new();
    let mut param_map: HashMap<VarId, VarId> = HashMap::new();
    let mut seen_names: HashMap<String, VarId> = HashMap::new();
    for part in &parts {
        for (name, val) in &part.bindings {
            match merged_bindings.get(name) {
                Some(existing) if existing != val => {
                    return Err(HorizontalFailure::ParamNameConflict);
                }
                _ => {
                    merged_bindings.insert(name.clone(), *val);
                }
            }
        }
        for (v, name) in part.module.builder.params() {
            let fresh = *seen_names.entry(name.clone()).or_insert_with(|| {
                let n = fb.var_watermark();
                fb.raise_var_watermark(n + 1);
                fb.inherit_param(VarId(n), name.clone());
                VarId(n)
            });
            param_map.insert(*v, fresh);
        }
    }

    // Boundary: stable-unique(a.inputs ++ b.inputs), with a per-part
    // map from local input position to fused input position.
    let mut fused_input_values: Vec<ValueClassId> = Vec::new();
    let mut local_input_maps: Vec<HashMap<usize, usize>> = Vec::with_capacity(parts.len());
    for part in &parts {
        let mut m = HashMap::new();
        for (pos, &v) in gf.nodes[part.alt_idx].inputs.iter().enumerate() {
            let fused_pos = match fused_input_values.iter().position(|&x| x == v) {
                Some(p) => p,
                None => {
                    fused_input_values.push(v);
                    fused_input_values.len() - 1
                }
            };
            m.insert(pos, fused_pos);
        }
        local_input_maps.push(m);
    }

    // Declare fused inputs. For each fused position, the declaring
    // parent is the first part (then lowest local position) mapping to
    // it — deterministic by construction.
    let mut fused_input_nodes: Vec<HirNodeId> = Vec::with_capacity(fused_input_values.len());
    for fused_pos in 0..fused_input_values.len() {
        let mut chosen: Option<(String, ScalarType, Vec<SizeExpr>)> = None;
        'parts: for (pi, part) in parts.iter().enumerate() {
            let n_local = gf.nodes[part.alt_idx].inputs.len();
            for pos in 0..n_local {
                if local_input_maps[pi].get(&pos) == Some(&fused_pos) {
                    let d = &part.module.builder.inputs()[pos];
                    let shape: Vec<SizeExpr> = d
                        .shape
                        .iter()
                        .map(|s| remap_size_expr(s, &param_map))
                        .collect();
                    chosen = Some((d.name.clone(), d.elem, shape));
                    break 'parts;
                }
            }
        }
        let (name, elem, shape) = chosen.expect("every fused input has a source");
        fused_input_nodes.push(fb.input(name, elem, shape));
    }

    // Fused compute's outer var.
    let outer_bound_fb = remap_size_expr(&parts[0].shape.outer_bound, &param_map);
    let k_var = {
        let n = fb.var_watermark();
        fb.raise_var_watermark(n + 1);
        VarId(n)
    };
    let k_var_node = fb.intern(Node::Var(k_var));

    // Clone each part's body with its outer var mapped to the shared
    // fused var, splicing multi-output Tuple bodies positionally.
    // Hash-consing shares identical loads across the two bodies.
    let mut tuple_elems: Vec<HirNodeId> = Vec::new();
    for (pi, part) in parts.iter().enumerate() {
        let input_nodes = find_input_nodes(&part.module, part.shape.body_root);
        let mut subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
        for (pos, &input_node) in &input_nodes {
            let fused_pos = *local_input_maps[pi]
                .get(pos)
                .expect("kernel input positions align with alt-node inputs");
            subst.insert(input_node, fused_input_nodes[fused_pos]);
        }
        let mut vars: HashMap<VarId, HirNodeId> = HashMap::new();
        vars.insert(part.shape.outer_var, k_var_node);
        for (from, to) in &param_map {
            let dst = fb.intern(Node::Var(*to));
            vars.insert(*from, dst);
        }
        let cloned = clone_expr(&part.module, part.shape.body_root, &mut fb, &subst, &vars)
            .map_err(|e| HorizontalFailure::CloneError(format!("{e:?}")))?;
        let n_outputs = gf.nodes[part.alt_idx].outputs.len();
        if n_outputs == 1 {
            tuple_elems.push(cloned);
        } else {
            let elems = match fb.node(cloned) {
                Node::Tuple(elems) if elems.len() == n_outputs => elems.clone(),
                _ => return Err(HorizontalFailure::TupleArityMismatch),
            };
            tuple_elems.extend(elems);
        }
    }

    let compute_body = fb.intern(Node::Tuple(tuple_elems));
    let fused_body_id = fb.intern(Node::Compute {
        bound: outer_bound_fb,
        var: k_var,
        body: compute_body,
        scatter: None,
        par: None,
        threads: None,
    });
    let fused_module = fb.finish("horizontal".to_string(), fused_body_id);

    crate::passes::type_infer(&fused_module).map_err(|e| match e {
        CompileError::Type(m) => HorizontalFailure::TypeCheckFailed(m),
        other => HorizontalFailure::TypeCheckFailed(other.to_string()),
    })?;

    // Boundary value bindings.
    let fused_inputs = fused_input_values;
    let mut fused_outputs: Vec<ValueClassId> = gf.nodes[a_node.0].outputs.clone();
    fused_outputs.extend(gf.nodes[b_node.0].outputs.iter().copied());

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

    Ok(CandidateDraft {
        parents: vec![a_node, b_node],
        variant: FusionVariant::Drop,
        alt: AltGraphNode {
            inputs: fused_inputs,
            outputs: fused_outputs,
            node,
        },
    })
}
