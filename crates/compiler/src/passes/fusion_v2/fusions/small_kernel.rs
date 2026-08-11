//! Small-kernel block fusion (M8 — §10.7).
//!
//! Fuses a **linear chain** of pure-Kernel nodes with concrete outer
//! bounds into a single kernel whose grid launches once and whose
//! block runs each source kernel as a shared-memory-backed
//! `let`-tile — reusing the DSL's existing inner-let tile pattern.
//!
//! Unlike M7 fanout (which requires all consumers to share the
//! producer's outer domain), small-kernel fusion **explicitly handles
//! different domain sizes**: every kernel in the chain keeps its own
//! iteration count, and the resulting KIR block sizes itself to the
//! largest per-layer domain via `lower_to_kir`'s `max_par` policy.
//!
//! Synthesized module shape for a chain `[k_1, k_2, ..., k_L]` with
//! iteration counts `[N_1, N_2, ..., N_L]` (all concrete):
//!
//! ```text
//! compute[N_L] |i| {
//!     let tile_1 = compute[N_1] |j| body_of_k_1[j]
//!     let tile_2 = compute[N_2] |j| body_of_k_2[j, tile_1]
//!     ...
//!     body_of_k_L[i, tile_{L-1}]
//! }
//! ```
//!
//! `canonicalize::peel_body_lets` folds each `let tile_i = compute[N_i]
//! |j| ...` into an [`InnerLet`], which `lower_to_kir` lowers as a
//! shared-memory buffer written by its own `Par` before the outer
//! compute runs. The grid launches with `outer_bound = N_L`, block
//! sized to `max(N_1..N_L)` (clamped to `BLOCK_SIZE`).
//!
//! Legality (M8 first slice):
//!
//! - Every kernel in the chain identifies as a valid [`KernelShape`] with a concrete outer bound.
//! - Every consumer reads its immediate producer's seam via a supported access (`hir_to_quast` must
//!   accept the index expression — identity, affine permutation, etc.). Non-identity accesses are
//!   fine because the tile is a full-shape shared-memory allocation, so any bounded index
//!   expression reads a valid entry.
//! - Every intermediate kernel `k_i` (`1 ≤ i < L`) has **exactly one** downstream consumer inside
//!   the chain: the next link `k_{i+1}`. Chains through fanout points don't compose here — the
//!   fanout pass owns that case.
//! - Origins are pairwise disjoint across the chain (§9.1 obligation).
//! - Combined per-tile byte footprint does not exceed [`SmallKernelOptions::max_shared_bytes`]
//!   (§10.7 shared-memory budget).
//!
//! Deviations from the plan's ideal §10.7 shape:
//!
//! - The plan describes a layered `compute[B_l]` chain with layer-boundary syncs and if/else
//!   dispatch. That structure requires HIR extensions (multiple top-level computes inside one
//!   kernel) not currently supported. The tile approach lowers through the same DSL surface as
//!   existing kernels — no compiler changes required — and preserves the launch-overhead saving
//!   because each tile is computed once per block rather than per outer iteration.
//! - Grid launches with `outer_bound = N_L` blocks, not `1`. If the last layer has a small enough
//!   `N_L` (≤ `BLOCK_SIZE`) and no `threads` hint, the block runs it inline. The launch overhead is
//!   still `launch_cycles_fit_sm` when the grid fits on one SM, giving the M8 win over `L` separate
//!   launches.
//! - **Parallel siblings within a layer** (§10.7's Σ iteration counts) are deferred; only linear
//!   chains for now.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    graph_ir::{BufId, GraphNode, KernelModuleNode},
    ir::{IRBuilder, Node, NodeId as HirNodeId, ScalarType, SizeExpr, VarId},
    passes::{
        fusion_utils::{clone_expr, remap_size_expr},
        fusion_v2::{
            fusions::producer_consumer::{
                identify_kernel_shape, CandidateDraft, EnumerateContext, FusionVariant, KernelShape,
            },
            model::{AltGraphNode, GraphFuser, NodeId, ValueClassId},
        },
    },
    CompileError,
};

/// Options controlling small-kernel enumeration.
#[derive(Copy, Clone, Debug)]
pub struct SmallKernelOptions {
    /// Maximum combined tile bytes allowed in a fused candidate
    /// (§10.7 shared-memory budget). Candidates exceeding this are
    /// rejected before synthesis. Default: 48 KiB.
    pub max_shared_bytes: usize,
    /// Maximum number of kernels a chain may contain (§11
    /// `max_region_seed_nodes`). Default: 6.
    pub max_chain_length: usize,
}

impl Default for SmallKernelOptions {
    fn default() -> Self {
        Self {
            max_shared_bytes: 48 * 1024,
            max_chain_length: 6,
        }
    }
}

/// Failure modes for small-kernel synthesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmallKernelFailure {
    ChainTooShort,
    ChainTooLong,
    NotAKernel,
    UnsupportedShape,
    NonConstantBound,
    SharedMemoryBudgetExceeded,
    ParamNameConflict,
    CloneError(String),
    TypeCheckFailed(String),
    ChainedKernelHasMultipleOutputs,
    IntermediateKernelHasBranchingConsumers,
}

/// Enumerates small-kernel candidates. Iterates possible chain starts
/// in seed-`NodeId` order, extending each maximal linear chain of
/// concrete-bound kernels before synthesizing one candidate.
pub fn enumerate(
    gf: &GraphFuser,
    ctx: &EnumerateContext,
    options: SmallKernelOptions,
) -> Vec<CandidateDraft> {
    let frozen = ctx.frozen_node_count.min(gf.nodes.len());
    let debug = super::debug_reject_level();
    // Sequential chain identification + dedup, then parallel synthesis
    // over the unique chains.
    let mut chains: Vec<Vec<NodeId>> = Vec::new();
    let mut emitted_chains: HashSet<Vec<NodeId>> = HashSet::new();
    for p_id in 0..frozen {
        let start = NodeId(p_id);
        let chain = match identify_chain(gf, ctx, start, &options) {
            Some(c) => c,
            None => continue,
        };
        if chain.len() < 2 {
            continue;
        }
        // Skip chains that don't include at least one node newer than
        // the previous round's watermark (saturation efficiency).
        if chain.iter().all(|n| n.0 < ctx.min_new_parent_id) {
            continue;
        }
        if !emitted_chains.insert(chain.clone()) {
            continue;
        }
        chains.push(chain);
    }
    let options = &options;
    let (out, rejects) = super::par_enumerate(chains, |chain| {
        match synthesize_small_kernel(gf, &chain, FusionVariant::Drop, options) {
            Ok(draft) => (vec![draft], Vec::new()),
            Err(e) => {
                let rejects = if debug >= 1 {
                    vec![(super::variant_name(&e), 1)]
                } else {
                    Vec::new()
                };
                (Vec::new(), rejects)
            }
        }
    });
    if debug >= 1 {
        super::dump_rejects("small-kernel", &rejects);
    }
    out
}

/// Returns the longest linear chain starting at `start` that meets the
/// M8 legality requirements, or `None` if no chain of length ≥ 2 fits.
///
/// A chain step `k_i → k_{i+1}` requires:
/// - `k_i` is a pure Kernel with concrete bound;
/// - `k_i` produces exactly one output value class;
/// - within the frozen prefix that value has exactly one consumer, `k_{i+1}`;
/// - `k_{i+1}` is a pure Kernel with concrete bound.
fn identify_chain(
    gf: &GraphFuser,
    ctx: &EnumerateContext,
    start: NodeId,
    options: &SmallKernelOptions,
) -> Option<Vec<NodeId>> {
    // Head eligibility.
    if !is_eligible_kernel_node(gf, start) {
        return None;
    }
    // Head must not have been the middle of a chain (otherwise we'd
    // double-count). We accept a head with any predecessor structure —
    // duplicates are filtered by the emitted_chains set.
    let mut chain = vec![start];
    let mut cur = start;
    loop {
        if chain.len() >= options.max_chain_length {
            break;
        }
        // cur must produce exactly one value.
        let outputs = &gf.nodes[cur.0].outputs;
        if outputs.len() != 1 {
            break;
        }
        let seam = outputs[0];
        // The seam must have exactly one consumer within the frozen
        // prefix. Multiple consumers => fanout territory, deferred.
        let consumers: Vec<NodeId> = gf.consumers[seam.0]
            .iter()
            .filter(|u| u.node.0 < ctx.frozen_node_count)
            .map(|u| u.node)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if consumers.len() != 1 {
            break;
        }
        let next = consumers[0];
        if next == cur {
            break;
        }
        if !is_eligible_kernel_node(gf, next) {
            break;
        }
        // Origin-disjointness with the accumulated chain.
        let mut origins_ok = true;
        let mut union: std::collections::BTreeSet<NodeId> = std::collections::BTreeSet::new();
        for n in chain.iter().copied().chain(std::iter::once(next)) {
            for seed in &ctx.origins[n.0] {
                if !union.insert(*seed) {
                    origins_ok = false;
                    break;
                }
            }
            if !origins_ok {
                break;
            }
        }
        if !origins_ok {
            break;
        }
        chain.push(next);
        cur = next;
    }
    if chain.len() < 2 {
        None
    } else {
        Some(chain)
    }
}

fn is_eligible_kernel_node(gf: &GraphFuser, node: NodeId) -> bool {
    let alt = &gf.nodes[node.0];
    let module = match &alt.node {
        GraphNode::Kernel(k) => &k.module,
        _ => return false,
    };
    let shape = match identify_kernel_shape(module).filter(|s| s.is_plain()) {
        Some(s) => s,
        None => return false,
    };
    // Concrete outer bound.
    if shape.outer_bound.as_const().is_none() {
        return false;
    }
    // Single output.
    if alt.outputs.len() != 1 {
        return false;
    }
    true
}

/// Byte size of a tensor of shape `[bound]` with element type `elem`
/// (concrete-shape only — panics on symbolic bounds, which callers
/// have already filtered out).
fn tile_bytes(bound: usize, elem: ScalarType) -> usize {
    bound * elem.size_bytes()
}

/// Builds the fused small-kernel candidate.
pub fn synthesize_small_kernel(
    gf: &GraphFuser,
    chain: &[NodeId],
    variant: FusionVariant,
    options: &SmallKernelOptions,
) -> Result<CandidateDraft, SmallKernelFailure> {
    if variant != FusionVariant::Drop {
        // Keep variant deferred (would require materializing every
        // internal seam alongside the last-layer outputs).
        return Err(SmallKernelFailure::UnsupportedShape);
    }
    if chain.len() < 2 {
        return Err(SmallKernelFailure::ChainTooShort);
    }
    if chain.len() > options.max_chain_length {
        return Err(SmallKernelFailure::ChainTooLong);
    }

    // Gather each kernel's module + shape + input seam positions.
    struct Link {
        alt_idx: usize,
        module: Arc<crate::ir::Module>,
        bindings: std::collections::BTreeMap<String, i64>,
        shape: KernelShape,
        outer_bound: i64,
        /// Input positions where the immediate predecessor's seam is
        /// bound. `Vec<usize>` because a kernel could read the seam at
        /// multiple positional operands, though M8's first slice only
        /// expects one.
        seam_positions_of_prev: Vec<usize>,
    }

    let mut links: Vec<Link> = Vec::with_capacity(chain.len());
    for (i, node) in chain.iter().enumerate() {
        let alt = &gf.nodes[node.0];
        let (module, bindings) = match &alt.node {
            GraphNode::Kernel(k) => (k.module.clone(), k.param_bindings.clone()),
            _ => return Err(SmallKernelFailure::NotAKernel),
        };
        let shape = identify_kernel_shape(&module)
            .filter(|s| s.is_plain())
            .ok_or(SmallKernelFailure::UnsupportedShape)?;
        let outer_bound = shape
            .outer_bound
            .as_const()
            .ok_or(SmallKernelFailure::NonConstantBound)?;
        if alt.outputs.len() != 1 {
            return Err(SmallKernelFailure::ChainedKernelHasMultipleOutputs);
        }
        let seam_positions_of_prev = if i == 0 {
            Vec::new()
        } else {
            let prev_seam = gf.nodes[chain[i - 1].0].outputs[0];
            let mut ps: Vec<usize> = alt
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(pos, v)| (*v == prev_seam).then_some(pos))
                .collect();
            ps.sort_unstable();
            if ps.is_empty() {
                return Err(SmallKernelFailure::IntermediateKernelHasBranchingConsumers);
            }
            ps
        };
        links.push(Link {
            alt_idx: node.0,
            module,
            bindings,
            shape,
            outer_bound,
            seam_positions_of_prev,
        });
    }

    // Shared-memory budget: sum tile bytes for links 0..L-1.
    let mut total_tile_bytes = 0usize;
    for link in &links[..links.len() - 1] {
        let elem = link.module.builder.inputs()[0].elem; // seam has same elem as producer's output
                                                         // Actually, the tile's element type matches the producer body's scalar type. Since the
                                                         // producer's output has the same scalar type as `link.module.builder.inputs()[0].elem` only
                                                         // when the body preserves that type, prefer the *output* type — but we don't have an
                                                         // explicit output-type getter; look it up via the module type map.
        let _ = elem;
        let bound = usize::try_from(link.outer_bound).unwrap_or(usize::MAX);
        let tile_elem = tile_element_type(&link.module)?;
        total_tile_bytes = total_tile_bytes.saturating_add(tile_bytes(bound, tile_elem));
    }
    if total_tile_bytes > options.max_shared_bytes {
        return Err(SmallKernelFailure::SharedMemoryBudgetExceeded);
    }

    // Build the fused module.
    let mut fb = IRBuilder::new();

    // Merge param bindings and remap params. Same pattern as producer_consumer.
    let mut merged_bindings = std::collections::BTreeMap::new();
    let mut param_map: HashMap<VarId, VarId> = HashMap::new();
    let mut seen_names: HashMap<String, VarId> = HashMap::new();
    for link in &links {
        for (name, val) in &link.bindings {
            match merged_bindings.get(name) {
                Some(existing) if existing != val => {
                    return Err(SmallKernelFailure::ParamNameConflict);
                }
                _ => {
                    merged_bindings.insert(name.clone(), *val);
                }
            }
        }
        for (v, name) in link.module.builder.params() {
            let fresh = *seen_names.entry(name.clone()).or_insert_with(|| {
                let n = fb.var_watermark();
                fb.raise_var_watermark(n + 1);
                fb.inherit_param(VarId(n), name.clone());
                VarId(n)
            });
            param_map.insert(*v, fresh);
        }
    }

    // Boundary: for each link, its non-seam inputs (i.e., not the
    // immediate predecessor's seam). Stable-uniqued by ValueClassId.
    let mut fused_input_values: Vec<ValueClassId> = Vec::new();
    let mut fused_input_present: HashSet<ValueClassId> = HashSet::new();
    // For each link, positional-input map from the source kernel to
    // the fused module.
    let mut link_input_map: Vec<HashMap<usize, usize>> = Vec::new();
    for link in &links {
        let mut m = HashMap::new();
        for (pos, &v) in gf.nodes[link.alt_idx].inputs.iter().enumerate() {
            if link.seam_positions_of_prev.contains(&pos) {
                // Seam — will be substituted with the previous tile
                // Var, not declared as a fused input.
                continue;
            }
            if fused_input_present.insert(v) {
                m.insert(pos, fused_input_values.len());
                fused_input_values.push(v);
            } else {
                let existing = fused_input_values.iter().position(|&x| x == v).unwrap();
                m.insert(pos, existing);
            }
        }
        link_input_map.push(m);
    }

    // Declare the fused module inputs. For each fused position, find
    // the first link whose local input maps to it and use that link's
    // module's InputDecl (name / elem / shape) as the fused input.
    let mut fused_input_nodes: Vec<HirNodeId> = Vec::with_capacity(fused_input_values.len());
    for fused_pos in 0..fused_input_values.len() {
        let mut chosen: Option<(String, ScalarType, Vec<SizeExpr>)> = None;
        'outer: for (li, m) in link_input_map.iter().enumerate() {
            for (pos, &fp) in m {
                if fp == fused_pos {
                    let d = &links[li].module.builder.inputs()[*pos];
                    let shape: Vec<SizeExpr> = d
                        .shape
                        .iter()
                        .map(|s| remap_size_expr(s, &param_map))
                        .collect();
                    chosen = Some((d.name.clone(), d.elem, shape));
                    break 'outer;
                }
            }
        }
        let (name, elem, shape) = chosen.expect("every fused input has a source");
        let n = fb.input(name, elem, shape);
        fused_input_nodes.push(n);
    }

    // Fused compute's outer var (using the last link's domain).
    let last = &links[links.len() - 1];
    let outer_bound_fb = remap_size_expr(&last.shape.outer_bound, &param_map);
    let k_var = {
        let n = fb.var_watermark();
        fb.raise_var_watermark(n + 1);
        VarId(n)
    };

    // Build the body inside-out: clone each link's body with its
    // Input NodeIds substituted for either the fused inputs or the
    // preceding tile Var. Wrap tiles for links 0..L-2 as let-bound
    // inner computes.
    //
    // tile_vars[i] = the Var NodeId (in fb) bound to link i's tile.
    // We build these vars up front so downstream links can reference them.
    let tile_vars: Vec<Option<VarId>> = links
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i < links.len() - 1 {
                let v = fb.var_watermark();
                fb.raise_var_watermark(v + 1);
                Some(VarId(v))
            } else {
                None
            }
        })
        .collect();

    // Clone each link's compute body, producing the compute NodeId
    // for links 0..L-1 (the tiles) and the scalar body-expression for
    // link L-1 (the outer body).
    let mut tile_computes: Vec<HirNodeId> = Vec::with_capacity(links.len() - 1);
    let mut last_body: Option<HirNodeId> = None;
    for (i, link) in links.iter().enumerate() {
        // Iteration variable of this link's compute:
        //   - inner tile (i < L-1) uses a fresh VarId `j_i`;
        //   - the outer (i = L-1) uses `k_var`.
        let iter_var = if i == links.len() - 1 {
            k_var
        } else {
            let v = fb.var_watermark();
            fb.raise_var_watermark(v + 1);
            VarId(v)
        };
        let iter_var_node = fb.intern(Node::Var(iter_var));

        // Consumer_subst: input NodeIds in `link.module`'s HIR mapped
        // into `fb`. Non-seam inputs -> fused input nodes; seam
        // inputs -> tile Var of previous link.
        let module_input_nodes: HashMap<usize, HirNodeId> =
            find_input_nodes(&link.module, link.shape.body_root);
        let mut consumer_subst: HashMap<HirNodeId, HirNodeId> = HashMap::new();
        for (pos, &input_node) in &module_input_nodes {
            if link.seam_positions_of_prev.contains(pos) {
                let prev = i
                    .checked_sub(1)
                    .ok_or(SmallKernelFailure::UnsupportedShape)?;
                let tile_var = tile_vars[prev].ok_or(SmallKernelFailure::UnsupportedShape)?;
                let tile_node = fb.intern(Node::Var(tile_var));
                consumer_subst.insert(input_node, tile_node);
            } else if let Some(&fused_pos) = link_input_map[i].get(pos) {
                consumer_subst.insert(input_node, fused_input_nodes[fused_pos]);
            }
        }

        // Vars: link's outer var -> the fused iter var; params via param_map.
        let mut vars: HashMap<VarId, HirNodeId> = HashMap::new();
        vars.insert(link.shape.outer_var, iter_var_node);
        for (from, to) in &param_map {
            let dst = fb.intern(Node::Var(*to));
            vars.insert(*from, dst);
        }

        let cloned = clone_expr(
            &link.module,
            link.shape.body_root,
            &mut fb,
            &consumer_subst,
            &vars,
        )
        .map_err(|e| SmallKernelFailure::CloneError(format!("{e:?}")))?;

        if i < links.len() - 1 {
            // Wrap as a compute[N_i] |j| cloned.
            let bound_fb = remap_size_expr(&link.shape.outer_bound, &param_map);
            let tile_compute = fb.intern(Node::Compute {
                bound: bound_fb,
                var: iter_var,
                body: cloned,
                scatter: None,
                par: None,
                threads: None,
            });
            tile_computes.push(tile_compute);
        } else {
            last_body = Some(cloned);
        }
    }
    let last_body = last_body.expect("chain length ≥ 2 => last link exists");

    // Wrap the last body in nested Let bindings for each tile.
    // Innermost-first: iterate tiles in reverse so tile_0 wraps
    // the outermost Let, tile_{L-2} wraps closest to the body.
    let mut body_node = last_body;
    for (i, tile_compute) in tile_computes.iter().enumerate().rev() {
        let var = tile_vars[i].expect("tile var allocated");
        body_node = fb.intern(Node::Let {
            var,
            value: *tile_compute,
            body: body_node,
        });
    }

    // Wrap in the outer compute.
    let fused_body_id = fb.intern(Node::Compute {
        bound: outer_bound_fb,
        var: k_var,
        body: body_node,
        scatter: None,
        par: None,
        threads: None,
    });
    let fused_module = fb.finish("small_kernel".to_string(), fused_body_id);

    let types = crate::passes::type_infer(&fused_module).map_err(|e| match e {
        CompileError::Type(m) => SmallKernelFailure::TypeCheckFailed(m),
        other => SmallKernelFailure::TypeCheckFailed(other.to_string()),
    })?;
    // Verify the module can canonicalize (catches synthesis bugs
    // before the estimator hits a `debug_assert!` on invalid IR).
    let _ = crate::passes::canonicalize(fused_module.clone(), types)
        .map_err(|e| SmallKernelFailure::TypeCheckFailed(format!("canonicalize failed: {e:?}")))?;

    // Boundary bindings.
    let fused_inputs = fused_input_values.clone();
    let fused_outputs: Vec<ValueClassId> = gf.nodes[last.alt_idx].outputs.clone();

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

    let parents: Vec<NodeId> = chain.to_vec();

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

/// Locates the element type produced by a link's compute body. This
/// walks the module's type inference results.
fn tile_element_type(module: &crate::ir::Module) -> Result<ScalarType, SmallKernelFailure> {
    let types = crate::passes::type_infer(module).map_err(|e| match e {
        CompileError::Type(m) => SmallKernelFailure::TypeCheckFailed(m),
        other => SmallKernelFailure::TypeCheckFailed(other.to_string()),
    })?;
    let root_ty = types
        .try_get(module.body)
        .ok_or_else(|| {
            SmallKernelFailure::TypeCheckFailed(format!(
                "module `{}` body has no type",
                module.name
            ))
        })?
        .clone();
    match root_ty {
        crate::ir::Type::Tensor(elem, _) => Ok(elem),
        other => Err(SmallKernelFailure::TypeCheckFailed(format!(
            "expected tensor type at module body, got {other:?}"
        ))),
    }
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
