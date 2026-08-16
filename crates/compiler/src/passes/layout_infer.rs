//! Layout inference over [`KirProgram`]s.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ir::{ScalarType, SizeExpr, VarId},
    kernel_ir::{
        maps_agree, AddressSpace, BufId, BufferDecl, BufferKind, IndexMap, Kernel, KirProgram,
        LinearLayout, ParAttr, SSABlock, SSANode, SSAOp, SSAOpCode,
    },
    passes::{
        layout_cost::ConversionCostModel,
        par_attr_infer::{infer_par_attr, ReadDemand},
        utils::ceil_log2,
    },
};

/// Allocate a placeholder shared scratch buffer for a `ConvertLayout`.
/// Shape stays at `[0]` here; `passes::allocate_convert_scratch` walks each
/// `ConvertLayout` post-inference and resizes the scratch from the (src,
/// dst) layout pair (Phase B.2). A 0-byte scratch that never gets written
/// stays filtered out of `plan_shared_mem`'s liveness, so the pure
/// shuffle / mirror paths pay nothing today. `disc` is a small
/// discriminator that keeps names unique when multiple ConvertLayouts
/// share the same source tile (view vs. mirror, or several views).
fn new_scratch_buffer(p: &mut KirProgram, src_name: &str, elem: ScalarType, disc: usize) -> BufId {
    let id = BufId(p.buffers.len() as u32);
    let suffix = if disc == usize::MAX {
        "_cs".to_string()
    } else {
        format!("_cs{disc}")
    };
    p.buffers.push(BufferDecl {
        name: format!("{src_name}{suffix}"),
        elem,
        shape: vec![SizeExpr::from(0usize)],
        kind: BufferKind::Shared,
        space: AddressSpace::Shared,
        layout: None,
    });
    id
}

/// Fills in the layout attributes left empty by lowering — the greedy
/// chaining algorithm (B.7):
///
/// 1. **Linearize XOR-linear access maps** ([`linearize_accesses`]): any [`IndexMap::Affine`] that
///    depends only on its par's own power-of-two index is rewritten to [`IndexMap::Linear`].
/// 2. **Greedy chain walk** ([`greedy_chain`]): a single program-order walk. For each par `P`: (a)
///    if `P.attr` is `None`, infer via [`infer_par_attr`] (B.6.1) using the producer layouts of
///    `P`'s reads *resolved through the chain tips* — a tile that was converted to layout `B` feeds
///    `B` (not its original layout) to downstream demand scoring; (b) rewire each read through the
///    source tile's chain ([`chain_par_reads`]): a read whose effective demand matches the tip
///    reads it in place, a mismatched linear read emits one [`SSAOpCode::ConvertLayout`] from the
///    tip into a fresh register version (which becomes the new tip), a linear read of a shared tip
///    loads it into a register version (linear readers always land on registers), a non-analyzable
///    read of a register tip mirrors it to shared, and a non-analyzable read of a shared buffer
///    reads it in place; (c) decide the space and layout of `P`'s writes ([`decide_target`] —
///    registers unless the write map isn't convertible to an invertible linear layout) and reset
///    the written tiles' chains.
///
/// The walk is greedy: each reader converts from whatever the current
/// tip is, in program order, so consecutive readers with equal demands
/// share one version while a demand that repeats after an intervening
/// conversion pays a fresh one. A search over conversion plans is a
/// future step.
pub fn layout_infer(p: &mut KirProgram) {
    for ki in 0..p.kernels.len() {
        linearize_accesses(&p.buffers, &mut p.kernels[ki]);
        greedy_chain(p, ki);
    }
    // Safety net: any Shared buffer still without a layout (e.g. a
    // buffer no par ever wrote to) gets identity. Global buffers
    // (Input/Output) keep whatever layout lowering supplied — their
    // shapes may be symbolic, so `.len()` would panic. Register buffers
    // keep `None` as "no explicit layout, default to identity" for
    // codegen.
    for buf in &mut p.buffers {
        if buf.layout.is_none() && buf.space == AddressSpace::Shared {
            buf.layout = Some(LinearLayout::identity(ceil_log2(buf.len().max(1))));
        }
    }
}

/// Statement position: enclosing loop (`None` = grid block) and index.
type StmtLoc = (Option<SSANode>, usize);

/// Intra-par version dedup entry:
/// `(original tile, space, layout, shape, version)`.
type CreatedVersion = (
    BufId,
    AddressSpace,
    Option<LinearLayout>,
    Vec<SizeExpr>,
    BufId,
);

/// Walk state for [`greedy_chain`].
#[derive(Default)]
struct GreedyState {
    /// Original tile → chain tip (the latest chainable version). Absent
    /// means the original buffer itself is the tip.
    tip: BTreeMap<BufId, BufId>,
    /// Original tile → (register versions, shared versions, total
    /// versions) counters for version and scratch naming.
    counts: BTreeMap<BufId, (usize, usize, usize)>,
    /// Version buffer → the insertion group holding its defining
    /// `ConvertLayout`. Converts *from* a version append to the same
    /// group, so they execute right after it and share its frequency
    /// (once per execution of the version's own definition).
    def_group: BTreeMap<BufId, StmtLoc>,
    /// Pending op insertions grouped by (parent, pre-insertion index).
    /// Vec order within a group is execution order.
    groups: BTreeMap<StmtLoc, Vec<NewOp>>,
    /// Read rewires: `(par, read index, new buffer, new index map)`.
    rewrites: Vec<(SSANode, usize, BufId, Option<IndexMap>)>,
}

/// The single program-order walk: par-attr inference, write-target
/// decisions, and chained `ConvertLayout` emission, all interleaved so
/// each step sees the state the previous statements left behind.
fn greedy_chain(p: &mut KirProgram, ki: usize) {
    let block = p.kernels[ki].block;
    let block_bits = ceil_log2(block);
    let cost = ConversionCostModel::default();
    // Kernel-wide analysis: which buffers have single writers, whether
    // any reader precedes the writer (back-edge), and whether the writer
    // has a mismatched self-read. Used by the register-eligibility check.
    let buf_info = analyze_buffers(&p.kernels[ki], &p.buffers);
    let mut tiles: BTreeMap<BufId, TileUse> = BTreeMap::new();
    {
        let k = &p.kernels[ki];
        walk_stmts(k, &p.buffers, None, &k.grid.block.body, &mut 0, &mut tiles);
    }
    let mut st = GreedyState::default();
    walk_chain_block(
        p, ki, None, block, block_bits, &cost, &buf_info, &tiles, &mut st,
    );

    // Apply read rewires first (doesn't shift statement indices).
    {
        let kernel = &mut p.kernels[ki];
        for (rnode, rai, nb, nix) in std::mem::take(&mut st.rewrites) {
            let op = &mut kernel.ops_mut()[rnode.0 as usize];
            let SSAOpCode::Par { reads, .. } = &mut op.opcode else {
                unreachable!("reader is a par")
            };
            reads[rai].buf = nb;
            if let Some(ix) = nix {
                reads[rai].index = ix;
            }
        }
    }

    // Per-block indices were recorded before any insertion; applying in
    // descending index order keeps every recorded index valid (groups
    // under different parents don't shift each other).
    let mut groups: Vec<(StmtLoc, Vec<NewOp>)> = st.groups.into_iter().collect();
    groups.sort_by(|a, b| b.0 .1.cmp(&a.0 .1));
    let kernel = &mut p.kernels[ki];
    for ((parent, at), ops) in groups {
        let nodes: Vec<SSANode> = ops
            .into_iter()
            .map(|o| {
                let opcode = match o {
                    NewOp::Alloc(buf) => SSAOpCode::Alloc { buf },
                    NewOp::Convert {
                        dst,
                        src,
                        scratch,
                        map,
                    } => SSAOpCode::ConvertLayout {
                        dst,
                        src,
                        scratch,
                        map,
                    },
                };
                kernel.push_op(SSAOp {
                    operands: Default::default(),
                    results: Default::default(),
                    opcode,
                    block: SSABlock::default(),
                })
            })
            .collect();
        for n in nodes.into_iter().rev() {
            match parent {
                None => kernel.grid.block.body.insert(at, n),
                Some(sid) => kernel.ops_mut()[sid.0 as usize].block.body.insert(at, n),
            }
        }
    }
}

/// Recursive statement walk for [`greedy_chain`]. Loops reset the
/// chains of every buffer their body writes, both at entry and at exit:
/// the back edge lets iteration `i+1` read what iteration `i` wrote, so
/// versions converted before (or inside) the loop may predate the
/// latest write and can't be trusted across the boundary.
#[allow(clippy::too_many_arguments)]
fn walk_chain_block(
    p: &mut KirProgram,
    ki: usize,
    parent: Option<SSANode>,
    block: usize,
    block_bits: usize,
    cost: &ConversionCostModel,
    buf_info: &BTreeMap<BufId, BufAnalysis>,
    tiles: &BTreeMap<BufId, TileUse>,
    st: &mut GreedyState,
) {
    let body: Vec<SSANode> = match parent {
        None => p.kernels[ki].grid.block.body.to_vec(),
        Some(sid) => p.kernels[ki].op(sid).block.body.to_vec(),
    };
    for (idx, &nid) in body.iter().enumerate() {
        let is_loop = matches!(p.kernels[ki].op(nid).opcode, SSAOpCode::Loop { .. });
        if is_loop {
            let mut written = BTreeSet::new();
            collect_par_writes(&p.kernels[ki], nid, &mut written);
            for b in &written {
                st.tip.remove(b);
            }
            let entry_tip = st.tip.clone();
            walk_chain_block(
                p,
                ki,
                Some(nid),
                block,
                block_bits,
                cost,
                buf_info,
                tiles,
                st,
            );
            // Restore the entry snapshot: versions created inside the
            // loop don't dominate the statements after it (the loop may
            // execute zero times), and tips for body-written buffers may
            // predate the last iteration's write.
            st.tip = entry_tip;
            continue;
        }
        if !matches!(p.kernels[ki].op(nid).opcode, SSAOpCode::Par { .. }) {
            continue;
        }
        infer_and_set_par_attr(p, ki, nid, block, block_bits, cost, &st.tip);
        chain_par_reads(p, ki, nid, (parent, idx), tiles, st);
        assign_layouts_for_par_writes(p, ki, nid, buf_info);
        // A write invalidates every version converted from the tile:
        // reset the chain to the original buffer.
        let wbufs: Vec<BufId> = {
            let SSAOpCode::Par { writes, .. } = &p.kernels[ki].op(nid).opcode else {
                unreachable!("checked above")
            };
            writes.iter().map(|w| w.buf).collect()
        };
        for b in wbufs {
            st.tip.remove(&b);
        }
    }
}

/// Buffers written by any par under `node` (inclusive), recursing into
/// nested loops.
fn collect_par_writes(kernel: &Kernel, node: SSANode, out: &mut BTreeSet<BufId>) {
    let op = kernel.op(node);
    match &op.opcode {
        SSAOpCode::Par { writes, .. } => {
            for w in writes.iter() {
                out.insert(w.buf);
            }
        }
        SSAOpCode::Loop { .. } => {
            for &c in &op.block.body {
                collect_par_writes(kernel, c, out);
            }
        }
        _ => {}
    }
}

/// Per-buffer analysis needed for the register-eligibility check.
#[derive(Default)]
struct BufAnalysis {
    /// Number of distinct par writes into this buffer (across the whole
    /// kernel). Register promotion needs exactly one writer.
    writer_count: usize,
    /// Whether any reader precedes the writer in program order — a
    /// loop-back-edge read from a register buffer isn't representable.
    has_back_edge_reader: bool,
    /// Whether the writer has a self-read of this buffer whose index map
    /// disagrees with the write map. Under a register buffer, the writer
    /// can only read its own slot; a mismatched self-read forces Shared.
    writer_self_read_mismatch: bool,
}

fn analyze_buffers(kernel: &Kernel, buffers: &[BufferDecl]) -> BTreeMap<BufId, BufAnalysis> {
    #[derive(Default)]
    struct Raw {
        writer: Option<(SSANode, usize)>,
        writer_seq: usize,
        writer_count: usize,
        readers: Vec<(SSANode, usize)>, // par + seq
    }
    fn recurse(kernel: &Kernel, body: &[SSANode], seq: &mut usize, raw: &mut BTreeMap<BufId, Raw>) {
        for &nid in body {
            let op = kernel.op(nid);
            *seq += 1;
            let my_seq = *seq;
            match &op.opcode {
                SSAOpCode::Par { reads, writes, .. } => {
                    for (wai, w) in writes.iter().enumerate() {
                        let r = raw.entry(w.buf).or_default();
                        r.writer_count += 1;
                        if r.writer.is_none() {
                            r.writer = Some((nid, wai));
                            r.writer_seq = my_seq;
                        }
                    }
                    for a in reads.iter() {
                        let r = raw.entry(a.buf).or_default();
                        r.readers.push((nid, my_seq));
                    }
                }
                SSAOpCode::Loop { .. } => recurse(kernel, &op.block.body, seq, raw),
                _ => {}
            }
        }
    }
    let mut raw: BTreeMap<BufId, Raw> = BTreeMap::new();
    let mut seq = 0;
    recurse(kernel, &kernel.grid.block.body, &mut seq, &mut raw);

    let mut out = BTreeMap::new();
    for (buf, r) in raw {
        let mut a = BufAnalysis {
            writer_count: r.writer_count,
            ..Default::default()
        };
        let writer_seq = r.writer_seq;
        for &(_, rseq) in &r.readers {
            if rseq < writer_seq {
                a.has_back_edge_reader = true;
            }
        }
        // Writer self-read mismatch check.
        if let Some((wnode, wai)) = r.writer {
            let op = kernel.op(wnode);
            if let SSAOpCode::Par { reads, writes, .. } = &op.opcode {
                let write_map = &writes[wai].index;
                for read in reads.iter().filter(|r| r.buf == buf) {
                    match (&read.index, write_map) {
                        (IndexMap::Linear(g), IndexMap::Linear(f)) if g == f => {}
                        _ => {
                            a.writer_self_read_mismatch = true;
                            break;
                        }
                    }
                }
            }
        }
        let _ = buffers; // reserved for future writer-kind checks
        out.insert(buf, a);
    }
    out
}

/// (a) of the interleaved pass: if `par_nid` lacks an attr, infer one
/// via [`infer_par_attr`] using real producer layouts, resolved through
/// the chain tips — a read of tile `A` whose chain tip is version `B`
/// will actually consume `B`, so `B`'s layout is the one to score.
#[allow(clippy::too_many_arguments)]
fn infer_and_set_par_attr(
    p: &mut KirProgram,
    ki: usize,
    par_nid: SSANode,
    block: usize,
    block_bits: usize,
    cost: &ConversionCostModel,
    tip: &BTreeMap<BufId, BufId>,
) {
    let op = p.kernels[ki].op(par_nid);
    let SSAOpCode::Par {
        attr,
        bound,
        spans_grid,
        reads,
        ..
    } = &op.opcode
    else {
        return;
    };
    if attr.is_some() {
        return;
    }
    let seq_size = if *spans_grid {
        1
    } else {
        bound
            .as_const()
            .expect("non-grid-spanning par bounds are concrete")
            .div_ceil(block)
    };
    let phys_bits = ceil_log2(seq_size) + block_bits;
    let default = LinearLayout::identity(phys_bits);
    let mut demands: Vec<ReadDemand> = Vec::new();
    for a in reads.iter() {
        let IndexMap::Linear(g) = &a.index else {
            continue;
        };
        let cur = tip.get(&a.buf).copied().unwrap_or(a.buf);
        let buf = &p.buffers[cur.0 as usize];
        // Use the real producer layout if set. Register buffers already
        // have `phys → logical`; shared/global have `logical → phys`
        // (not scoreable by B.6.1 which expects phys → logical), so
        // for those we fall back to identity — that ties with the
        // default and the tie-breaker picks the default.
        let logical_bits = ceil_log2(buf.len().max(1));
        let producer_layout = if buf.space == AddressSpace::Register {
            buf.layout
                .clone()
                .unwrap_or_else(|| LinearLayout::identity(logical_bits))
        } else {
            LinearLayout::identity(logical_bits)
        };
        let g_padded = pad_layout_identity(g.clone(), phys_bits);
        demands.push(ReadDemand {
            g: g_padded,
            producer_layout,
            logical_bits,
            loop_iters: Vec::new(),
        });
    }
    let picked = if demands.is_empty() {
        default.clone()
    } else {
        infer_par_attr(cost, &demands, default.clone(), block, phys_bits)
    };
    let op_mut = &mut p.kernels[ki].ops_mut()[par_nid.0 as usize];
    if let SSAOpCode::Par { attr, .. } = &mut op_mut.opcode {
        *attr = Some(ParAttr {
            seq_size,
            layout: picked,
        });
    }
}

/// (b) of the interleaved pass: for each buffer `par_nid` writes to,
/// decide `target_space` and `target_layout` and update the buffer
/// declaration in place. Buffers whose layouts were set at lowering
/// (Input/Output) or by an earlier writer are left alone.
fn assign_layouts_for_par_writes(
    p: &mut KirProgram,
    ki: usize,
    par_nid: SSANode,
    buf_info: &BTreeMap<BufId, BufAnalysis>,
) {
    let op = p.kernels[ki].op(par_nid);
    let SSAOpCode::Par {
        writes,
        attr,
        spans_grid,
        ..
    } = &op.opcode
    else {
        return;
    };
    let par_attr = attr
        .as_ref()
        .expect("par-attr assigned before writes")
        .clone();
    let par_spans_grid = *spans_grid;

    let decisions: Vec<(BufId, AddressSpace, BufferKind, LinearLayout)> = writes
        .iter()
        .filter_map(|w| {
            let buf_id = w.buf;
            let buf = &p.buffers[buf_id.0 as usize];
            // Only intermediate tiles (`BufferKind::Shared` at lowering
            // time) get their space/layout picked here. Input/Output
            // keep their lowering-set layouts.
            if !matches!(buf.kind, BufferKind::Shared) {
                return None;
            }
            if buf.layout.is_some() {
                return None;
            }
            let (space, kind, layout) =
                decide_target(buf, par_spans_grid, &par_attr, w, buf_info.get(&buf_id));
            Some((buf_id, space, kind, layout))
        })
        .collect();

    for (buf_id, space, kind, layout) in decisions {
        let buf = &mut p.buffers[buf_id.0 as usize];
        buf.space = space;
        buf.kind = kind;
        buf.layout = Some(layout);
    }
}

/// The register-eligibility check + layout derivation for one write.
/// Buffers that pass every gate become Register with layout
/// `write_map ∘ par_attr.layout`; others fall to Shared with identity
/// (a placeholder that B.6.2's `choose_shared_layout` will replace).
fn decide_target(
    buf: &BufferDecl,
    par_spans_grid: bool,
    par_attr: &ParAttr,
    write: &crate::kernel_ir::Access,
    info: Option<&BufAnalysis>,
) -> (AddressSpace, BufferKind, LinearLayout) {
    let n = buf.len();
    let kb = ceil_log2(n.max(1));
    let shared_identity = (
        AddressSpace::Shared,
        BufferKind::Shared,
        LinearLayout::identity(kb),
    );

    if par_spans_grid {
        return shared_identity;
    }
    if !n.is_power_of_two() {
        return shared_identity;
    }
    let Some(info) = info else {
        return shared_identity;
    };
    if info.writer_count != 1 {
        return shared_identity;
    }
    if info.has_back_edge_reader {
        return shared_identity;
    }
    if info.writer_self_read_mismatch {
        return shared_identity;
    }
    let IndexMap::Linear(f) = &write.index else {
        return shared_identity;
    };
    if f.bases.len() != kb {
        return shared_identity;
    }
    let l = f.compose(&par_attr.layout);
    if l.right_inverse(kb).is_none() {
        return shared_identity;
    }
    (AddressSpace::Register, BufferKind::Register, l)
}

/// Extend `l`'s bases with identity bases at positions
/// `l.bases.len()..target_bits`. Semantically: the padded input bits
/// don't exist in the caller's phys space (they're always zero for
/// actual inputs), but padding with identity keeps the resulting square
/// map bijective — a prerequisite for [`classify_convert`]'s lane-block
/// invertibility check to yield `Shuffle` rather than `Bounce`.
fn pad_layout_identity(mut l: LinearLayout, target_bits: usize) -> LinearLayout {
    for i in l.bases.len()..target_bits {
        l.bases.push(1u64 << i);
    }
    l
}

/// Rewrites `Affine` access maps that depend only on the par's own index
/// over a power-of-two domain into [`IndexMap::Linear`]. Accesses to
/// register accumulators keep their `Affine` own-index form.
fn linearize_accesses(buffers: &[BufferDecl], kernel: &mut Kernel) {
    for op in kernel.ops_mut() {
        let vid = op.block.operands.first().copied();
        let SSAOpCode::Par { reads, writes, .. } = &mut op.opcode else {
            continue;
        };
        let sym = VarId(vid.expect("par block binds its index").0);
        for a in reads.iter_mut().chain(writes.iter_mut()) {
            if buffers[a.buf.0 as usize].kind == BufferKind::Register {
                continue;
            }
            let IndexMap::Affine { expr, bounds } = &a.index else {
                continue;
            };
            let mut syms = BTreeSet::new();
            expr.syms(&mut syms);
            if syms.len() != 1 || !syms.contains(&sym) {
                continue;
            }
            if let Some(ll) = expr.to_linear_layout(bounds) {
                a.index = IndexMap::Linear(ll);
            }
        }
    }
}

/// Rewire one par's reads through the greedy chains (step (b) of
/// [`greedy_chain`]). Every read classifies against the chain tip *at
/// par entry* — a par's reads are simultaneous, so chaining applies
/// across statements, not between the reads of one statement — and
/// versions created for this par dedup by `(space, layout, shape)`.
///
/// Per read of tile `A` with tip `T`:
///
/// - **Writer self-read**: left in place. Register self-reads passed [`decide_target`]'s
///   matching-map gate; shared self-reads see the buffer's current contents.
/// - **Scoreable read** (linear index, non-grid-spanning par, power-of-two reader bound within a
///   power-of-two tile), `T` on registers: demand `E = g ∘ attr` matches `T`'s layout → rewire to
///   `T`; else emit `ConvertLayout` from `T` into a fresh register version with layout `E` (the new
///   tip). `gen_convert` handles every `best_decomposition` outcome (Copy / Slot / Shuffle
///   including the multi-round pull / Bounce via scratch), and `allocate_convert_scratch` runs the
///   same decomposition to size the scratch.
/// - **Scoreable read, `T` shared**: linear readers always land on registers — emit a
///   shared→register `ConvertLayout` into a fresh register version with layout `E`.
/// - **Non-scoreable read, `A` shared**: read the original in place (its data is never mutated by
///   converts, and non-linear indices address shared directly).
/// - **Non-scoreable read, `A` on registers**: mirror the tip to a shared version and read that
///   (register→shared `ConvertLayout`), or rewire to an existing shared mirror tip.
fn chain_par_reads(
    p: &mut KirProgram,
    ki: usize,
    par_nid: SSANode,
    par_loc: StmtLoc,
    tiles: &BTreeMap<BufId, TileUse>,
    st: &mut GreedyState,
) {
    let (par_reads, par_wbufs, par_attr, par_bound, par_grid) = {
        let op = p.kernels[ki].op(par_nid);
        let SSAOpCode::Par {
            reads,
            writes,
            attr,
            bound,
            spans_grid,
            ..
        } = &op.opcode
        else {
            return;
        };
        (
            reads.clone(),
            writes.iter().map(|w| w.buf).collect::<BTreeSet<_>>(),
            attr.clone(),
            bound.clone(),
            *spans_grid,
        )
    };
    let entry_tip = st.tip.clone();
    // Versions created for this par, for intra-par dedup:
    // (orig, space, layout, shape, id).
    let mut created: Vec<CreatedVersion> = Vec::new();
    let mut new_tip: BTreeMap<BufId, BufId> = BTreeMap::new();

    for (rai, read) in par_reads.iter().enumerate() {
        let orig = read.buf;
        if !matches!(
            p.buffers[orig.0 as usize].kind,
            BufferKind::Shared | BufferKind::Register
        ) {
            continue;
        }
        if par_wbufs.contains(&orig) {
            continue;
        }
        let Some(tu) = tiles.get(&orig) else {
            continue;
        };
        if tu.alloc.is_none() {
            continue;
        }
        let cur = entry_tip.get(&orig).copied().unwrap_or(orig);
        let (cur_space, cur_layout) = {
            let d = &p.buffers[cur.0 as usize];
            (d.space, d.layout.clone())
        };
        let (orig_space, orig_shape, shape_concrete) = {
            let d = &p.buffers[orig.0 as usize];
            (
                d.space,
                d.shape.clone(),
                d.shape.iter().all(|s| s.as_const().is_some()),
            )
        };
        let n = if shape_concrete {
            p.buffers[orig.0 as usize].len()
        } else {
            0
        };

        // F₂-scoreable read: linear index, non-grid-spanning par, and a
        // power-of-two reader bound within a power-of-two tile.
        let linear_g = match &read.index {
            IndexMap::Linear(g) => Some(g.clone()),
            _ => None,
        };
        let scoreable = linear_g.is_some()
            && !par_grid
            && n.is_power_of_two()
            && par_bound
                .as_const()
                .is_some_and(|rb| rb <= n && rb.is_power_of_two());

        if !scoreable {
            if orig_space == AddressSpace::Shared {
                // The original shared tile is always a valid source for a
                // non-analyzable read: converts never mutate it and the
                // walk resets its chain on every write. The read already
                // points at it — nothing to do.
                continue;
            }
            match cur_space {
                AddressSpace::Register => {
                    if cur_layout.is_none() {
                        // Register accumulators without a layout are
                        // accessed only at the par's own slot.
                        continue;
                    }
                    let kb = n.trailing_zeros() as usize;
                    let vid = version_for(
                        p,
                        st,
                        &mut created,
                        &mut new_tip,
                        orig,
                        cur,
                        AddressSpace::Shared,
                        None,
                        orig_shape,
                        kb,
                        par_loc,
                        tu,
                    );
                    st.rewrites.push((par_nid, rai, vid, None));
                }
                // The tip is already a shared mirror: read it in place.
                AddressSpace::Shared => st.rewrites.push((par_nid, rai, cur, None)),
                AddressSpace::Global => unreachable!("chain tips are kernel-local"),
            }
            continue;
        }

        // For readers whose par bound is smaller than the writer's tile
        // (halving-tree pattern), extend the reader's linear-layout
        // inputs with identity bases so the composed map is square over
        // the tile's kb bits — a lane permutation that crosses only the
        // reader's active range still classifies as `Shuffle`.
        let kb = n.trailing_zeros() as usize;
        let g_padded = pad_layout_identity(linear_g.expect("scoreable implies linear"), kb);
        let eff = match &par_attr {
            Some(a) => {
                let a_padded = pad_layout_identity(a.layout.clone(), kb);
                g_padded.compose(&a_padded)
            }
            None => g_padded.clone(),
        };
        match cur_space {
            AddressSpace::Register => {
                let Some(l) = cur_layout else {
                    continue;
                };
                if maps_agree(&eff, &l) {
                    st.rewrites
                        .push((par_nid, rai, cur, Some(IndexMap::Linear(g_padded))));
                    continue;
                }
            }
            // Linear readers always land on registers: fall through to
            // the shared→register load below.
            AddressSpace::Shared => {}
            AddressSpace::Global => unreachable!("chain tips are kernel-local"),
        }
        let vid = version_for(
            p,
            st,
            &mut created,
            &mut new_tip,
            orig,
            cur,
            AddressSpace::Register,
            Some(eff),
            vec![SizeExpr::from(1usize << kb)],
            kb,
            par_loc,
            tu,
        );
        st.rewrites
            .push((par_nid, rai, vid, Some(IndexMap::Linear(g_padded))));
    }
    st.tip.append(&mut new_tip);
}

/// Find or create a version of `orig` converted from `cur`. Dedup is
/// per-par (the `created` list): reads of one par with the same demand
/// share a version. The convert is placed:
///
/// - right after `cur`'s defining statement when that position is safe — register originals have a
///   single dominating writer (decide_target gates) and versions are written exactly once by their
///   own convert, so "right after the def" both dominates every later reader and executes at the
///   def's own frequency (a loop reader of a top-level tile pays the convert once, not per
///   iteration);
/// - right before the reader par when `cur` is a *shared original*, whose writers may be multiple
///   or follow the reader on a loop back edge — re-executing per reader execution is the only
///   generally safe frequency.
///
/// Version buffers alloc at the original tile's alloc site, which
/// dominates every possible convert position.
#[allow(clippy::too_many_arguments)]
fn version_for(
    p: &mut KirProgram,
    st: &mut GreedyState,
    created: &mut Vec<CreatedVersion>,
    new_tip: &mut BTreeMap<BufId, BufId>,
    orig: BufId,
    cur: BufId,
    space: AddressSpace,
    layout: Option<LinearLayout>,
    shape: Vec<SizeExpr>,
    kb: usize,
    par_loc: StmtLoc,
    tu: &TileUse,
) -> BufId {
    if let Some(&(.., id)) = created
        .iter()
        .find(|(o, sp, ll, sh, _)| *o == orig && *sp == space && *ll == layout && *sh == shape)
    {
        return id;
    }
    let (orig_name, orig_elem) = {
        let d = &p.buffers[orig.0 as usize];
        (d.name.clone(), d.elem)
    };
    let (regs, sms, total) = *st.counts.entry(orig).or_default();
    let (kind, suffix) = match space {
        AddressSpace::Register => (BufferKind::Register, format!("_v{regs}")),
        AddressSpace::Shared => (
            BufferKind::Shared,
            if sms == 0 {
                "_sm".to_string()
            } else {
                format!("_sm{sms}")
            },
        ),
        AddressSpace::Global => unreachable!("layout_infer never inserts a Global version"),
    };
    let vid = BufId(p.buffers.len() as u32);
    p.buffers.push(BufferDecl {
        name: format!("{orig_name}{suffix}"),
        elem: orig_elem,
        shape: shape.clone(),
        kind,
        space,
        layout: layout.clone(),
    });
    let scratch = new_scratch_buffer(p, &orig_name, orig_elem, total);
    {
        let c = st.counts.entry(orig).or_default();
        match space {
            AddressSpace::Register => c.0 += 1,
            AddressSpace::Shared => c.1 += 1,
            AddressSpace::Global => {}
        }
        c.2 += 1;
    }
    let alloc_at = tu.alloc.expect("caller checked the tile has an alloc");
    st.groups
        .entry((alloc_at.0, alloc_at.1 + 1))
        .or_default()
        .extend([NewOp::Alloc(vid), NewOp::Alloc(scratch)]);
    let conv_group = if cur != orig {
        st.def_group[&cur]
    } else if p.buffers[orig.0 as usize].space == AddressSpace::Register {
        tu.writer_loc.map(|w| (w.0, w.1 + 1)).unwrap_or(par_loc)
    } else {
        par_loc
    };
    st.groups
        .entry(conv_group)
        .or_default()
        .push(NewOp::Convert {
            dst: vid,
            src: cur,
            scratch,
            map: LinearLayout::identity(kb),
        });
    st.def_group.insert(vid, conv_group);
    created.push((orig, space, layout.clone(), shape, vid));
    // Chain advance: the version becomes the tile's tip for subsequent
    // statements — unless it's a register version with a non-surjective
    // (folding) layout, which can't serve as a convert source (no right
    // inverse). Those stay dead-end leaves; the tip is unchanged.
    let chainable = match (space, &layout) {
        (AddressSpace::Shared, _) => true,
        (_, Some(l)) => l.right_inverse(kb).is_some(),
        (_, None) => false,
    };
    if chainable {
        new_tip.insert(orig, vid);
    }
    vid
}

enum NewOp {
    Alloc(BufId),
    Convert {
        dst: BufId,
        src: BufId,
        scratch: BufId,
        map: LinearLayout,
    },
}

#[derive(Default)]
struct TileUse {
    alloc: Option<StmtLoc>,
    /// `(par, write access index)` pairs.
    writes: Vec<(SSANode, usize)>,
    writer_loc: Option<StmtLoc>,
    writer_seq: usize,
    /// `(par, read access index, statement walk order)` triples.
    reads: Vec<(SSANode, usize, usize)>,
}

/// Pre-order walk of the statement blocks, recording kernel-local
/// buffer allocs, writers and readers. `seq` numbers statements in
/// program order so that reader-before-writer (back-edge) uses can be
/// detected. Kind filter: `Shared` and `Register` (both are kernel-local
/// Alloc buffers — `assign_par_attrs_and_writes` may have promoted the
/// former into the latter).
fn walk_stmts(
    k: &Kernel,
    buffers: &[BufferDecl],
    parent: Option<SSANode>,
    body: &[SSANode],
    seq: &mut usize,
    tiles: &mut BTreeMap<BufId, TileUse>,
) {
    let is_kernel_local = |buf: BufId| {
        matches!(
            buffers[buf.0 as usize].kind,
            BufferKind::Shared | BufferKind::Register
        )
    };
    for (i, &nid) in body.iter().enumerate() {
        let op = k.op(nid);
        *seq += 1;
        let my_seq = *seq;
        match &op.opcode {
            SSAOpCode::Alloc { buf } if is_kernel_local(*buf) => {
                tiles.entry(*buf).or_default().alloc = Some((parent, i));
            }
            SSAOpCode::Par { reads, writes, .. } => {
                for (ai, a) in writes.iter().enumerate() {
                    if is_kernel_local(a.buf) {
                        let t = tiles.entry(a.buf).or_default();
                        t.writes.push((nid, ai));
                        t.writer_loc = Some((parent, i));
                        t.writer_seq = my_seq;
                    }
                }
                for (ai, a) in reads.iter().enumerate() {
                    if is_kernel_local(a.buf) {
                        tiles
                            .entry(a.buf)
                            .or_default()
                            .reads
                            .push((nid, ai, my_seq));
                    }
                }
            }
            SSAOpCode::Loop { .. } => walk_stmts(k, buffers, Some(nid), &op.block.body, seq, tiles),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{IRBuilder, NodeId, ScalarType},
        passes::{
            codegen, insert_sync,
            utils::test_util::{lowered, stmt_kinds},
            verify,
        },
    };

    fn own_index_tile(b: &mut IRBuilder, src: NodeId, i: NodeId, t: usize) -> NodeId {
        b.compute(t, |b, j| {
            let tc = b.const_u32(t as u32);
            let base = b.mul(i, tc);
            let ix = b.add(base, j);
            b.index(src, &[ix])
        })
    }

    /// A tile written and read at the par's own index is promoted to
    /// registers: no shared memory, no barrier, direct slot reads.
    #[test]
    fn own_index_tile_promotes_to_registers() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = own_index_tile(b, a, i, t);
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let x = b.index(tile, &[j]);
                    b.mul(x, x)
                })
            })
        });
        let module = b.finish("reg_tile", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        assert_eq!(stmt_kinds(&kprog.kernels[0]), ["alloc", "par", "par"]);
        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].shape, vec![SizeExpr::from(t)]);
        // Under Phase B.1, the register layout is `f_w ∘ par_attr` on the
        // full phys width; with `bound=8 < block=32` and identity par-attr,
        // the composite is `[1,2,4,0,0]` (identity on 3 low bits, zero
        // columns above — Gap 2's replicated layout).
        assert_eq!(
            regs[0].layout.as_ref().unwrap(),
            &LinearLayout {
                bases: vec![1, 2, 4, 0, 0],
                offset: 0
            }
        );
        assert!(!kprog.buffers.iter().any(|b| b.kind == BufferKind::Shared));

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("uint32_t r"), "{source}");
        assert!(!source.contains("__shared__"), "{source}");
        assert!(!source.contains("__syncthreads"), "{source}");
    }

    /// A lane-permuting linear reader gets an identity-layout register view
    /// filled by a warp-shuffle `ConvertLayout`.
    #[test]
    fn lane_rotated_reader_gets_shuffled_view() {
        let (blocks, t) = (2usize, 64usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = own_index_tile(b, a, i, t);
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    // Rotate the five lane bits: j/32*32 + j%16*2 + j%32/16.
                    let (c2, c16, c32) = (b.const_u32(2), b.const_u32(16), b.const_u32(32));
                    let hi = b.div(j, c32);
                    let hi = b.mul(hi, c32);
                    let lo = b.rem(j, c16);
                    let lo = b.mul(lo, c2);
                    let mid = b.rem(j, c32);
                    let mid = b.div(mid, c16);
                    let ix = b.add(hi, lo);
                    let ix = b.add(ix, mid);
                    let x = b.index(tile, &[ix]);
                    let y = b.index(tile, &[j]);
                    b.add(x, y)
                })
            })
        });
        let module = b.finish("shuffle_tile", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        assert_eq!(
            stmt_kinds(&kprog.kernels[0]),
            ["alloc", "alloc", "alloc", "par", "convert", "par"]
        );
        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        assert_eq!(regs.len(), 2, "tile and its view");
        // Phase B.2: each ConvertLayout carries a 0-byte shared scratch
        // buffer as a placeholder for future bounce routes; today's
        // shuffle-only path leaves it 0-byte, so filter by phys_bytes,
        // not kind, to check no real shared allocation happens.
        for b in &kprog.buffers {
            if b.kind == BufferKind::Shared {
                assert_eq!(
                    b.len(),
                    0,
                    "scratch {} should be 0-byte for this shuffle-only case",
                    b.name
                );
            }
        }

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shfl_sync"), "{source}");
        assert!(!source.contains("__syncthreads"), "{source}");
    }

    /// A many-to-one reader of a promoted tile (`tile[j % (t/2)]` folds
    /// pairs of lanes onto one slot) becomes a `__shfl_sync` broadcast
    /// under Gap 4 — the singular lane block still classifies as
    /// `Shuffle` when the sender-slot function is constant across
    /// receivers. No shared mirror, no barrier.
    #[test]
    fn non_linear_reader_gets_shuffle_view() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = own_index_tile(b, a, i, t);
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let half = b.const_u32((t / 2) as u32);
                    let fold = b.rem(j, half);
                    let x = b.index(tile, &[fold]);
                    let y = b.index(tile, &[j]);
                    b.add(x, y)
                })
            })
        });
        let module = b.finish("mirror_tile", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        assert_eq!(
            stmt_kinds(&kprog.kernels[0]),
            ["alloc", "alloc", "alloc", "par", "convert", "par"]
        );
        assert!(!kprog.buffers.iter().any(|b| b.name.ends_with("_sm")));

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shfl_sync"), "{source}");
        // The scratch buffer is 0-byte, so codegen emits no shared array
        // declaration for it (`plan_shared_mem` filters unwritten shared
        // buffers out of the liveness pool).
        assert!(!source.contains("__shared__"), "{source}");
        assert!(!source.contains("__syncthreads"), "{source}");
    }

    /// A warp-crossing composite that isn't warp-column-identity: the
    /// reader's index permutes bits across the lane/warp boundary, so
    /// [`best_decomposition`] picks `Strategy::Bounce` and codegen emits
    /// a store→sync→load through a shared scratch buffer (paper §5.4
    /// Optimal Swizzling, Phase B.4-full).
    ///
    /// Setup: t=128 with no `#[grid(threads=…)]` gives `block = 128`
    /// (max_par.min(BLOCK_SIZE).max(32)), so the physical partition has
    /// 5 lane bits and 2 warp bits (positions 5, 6). The reader carries
    /// an explicit `#[par((th, s) -> th)]` par-attr — identity, so
    /// [`infer_par_attr`] (B.6.1) can't rotate bits to eliminate the
    /// mismatch. The reader's index `(j & 31) * 4 + (j / 32)` then
    /// moves warp bits 5, 6 into positions 0, 1 (lane), failing
    /// `is_warp_column_identity` and triggering the Bounce strategy.
    #[test]
    fn warp_crossing_reader_bounces_via_scratch() {
        use crate::passes::allocate_convert_scratch;
        let (blocks, t) = (2usize, 128usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = crate::kernel!(b,
            compute [blocks] |i| {
                let tile = compute [t] |j| { a[i * #t + j] };
                #[par((th, s) -> th + s * 128)]
                compute [t] |j| { tile[j % 32 * 4 + j / 32] }
            }
        );
        let module = b.finish("bounce_test", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        allocate_convert_scratch(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        // Block is 128 (warp bits = 2), not the usual 32.
        assert_eq!(kprog.kernels[0].block, 128);

        // The Bounce reg→reg conversion routes through a scratch shared
        // buffer sized to the full tile.
        let scratch: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.name.contains("_cs") && !b.is_empty())
            .collect();
        assert_eq!(scratch.len(), 1, "one non-zero scratch buffer");
        assert_eq!(scratch[0].len(), t);
        // The scratch buffer's layout must be a bijection (all bases
        // linearly independent) — `pick_scratch_layout` projects a
        // valid `SharedLayout` partition via `to_linear_layout`. For
        // this test's identical-shape accesses the swizzle degenerates
        // to identity; the layout is nonetheless *set* and is a
        // well-formed bijection.
        let scratch_ll = scratch[0].layout.as_ref().unwrap();
        assert!(
            scratch_ll.inverse().is_some(),
            "scratch layout must be a bijection: {:?}",
            scratch_ll
        );

        // Codegen emits __syncthreads() from the internal bounce
        // (store→sync→load).
        let source = codegen(&kprog).unwrap();
        assert!(
            source.contains("__syncthreads"),
            "bounce should emit __syncthreads: {source}"
        );
        // A shared pool must exist since the scratch is non-zero.
        assert!(source.contains("_sh_pool"), "shared pool: {source}");
        // No `__shfl_sync(0xFFFFFFFFu` — this is a Bounce path, not
        // Shuffle. The FpExt helper's `__shfl_sync` definition (a static
        // inline overload for the extension-field type) always appears
        // in the output, so we match on the specific full-mask
        // invocation pattern instead of substring `__shfl_sync`.
        assert!(
            !source.contains("__shfl_sync(0xFFFFFFFFu"),
            "no shuffles in kernel body: {source}"
        );
    }

    /// A tile read whose index contains a symbolic constant lowers to an
    /// [`IndexMap::SExpr`] access; layout inference is not able to analyze
    /// it, so the read is routed through a shared-memory mirror (the
    /// non-linear catch-all) while the tile itself is still promoted.
    #[test]
    fn sexpr_reader_gets_shared_mirror() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let m = b.symbol("m");
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = own_index_tile(b, a, i, t);
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    // `(j + #m) % t`: symbolic, closed over binders + params.
                    let mv = b.const_sym(m);
                    let off = b.add(j, mv);
                    let tc = b.const_u32(t as u32);
                    let ix = b.rem(off, tc);
                    let x = b.index(tile, &[ix]);
                    let y = b.index(tile, &[j]);
                    b.add(x, y)
                })
            })
        });
        let module = b.finish("sexpr_mirror_tile", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        assert_eq!(
            stmt_kinds(&kprog.kernels[0]),
            ["alloc", "alloc", "alloc", "par", "convert", "sync", "par"]
        );
        let mirror_id = kprog
            .buffers
            .iter()
            .position(|b| b.name.ends_with("_sm"))
            .expect("mirror buffer");
        assert_eq!(kprog.buffers[mirror_id].kind, BufferKind::Shared);

        // The symbolic read kept its SExpr index and was rewired to the
        // mirror.
        let sexpr_read = kprog.kernels[0]
            .ops()
            .iter()
            .filter_map(|op| match &op.opcode {
                SSAOpCode::Par { reads, .. } => Some(reads),
                _ => None,
            })
            .flatten()
            .find(|a| matches!(a.index, IndexMap::SExpr(_)))
            .expect("SExpr read access");
        assert_eq!(sexpr_read.buf, BufId(mirror_id as u32));

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("p_m"), "{source}");
        assert_eq!(source.matches("__syncthreads();").count(), 1, "{source}");
    }

    /// Par layouts compose into promotion. Under `#[grid(threads = 32)]`
    /// with tiles of 512, a `#[par((th, s) -> th*16 + s)]` writer makes the
    /// register layout `g_w ∘ f_w`; its same-layout reader consumes the own
    /// slot directly for `c1[j]` and a warp-shuffle view for `c1[j ^ 64]`.
    /// The identity-scheduled gather tile is still promoted to a register
    /// buffer with identity layout — the register-first default — and gets
    /// one shared mirror through which the mismatched-layout reader picks
    /// up its data.
    #[test]
    fn par_layouts_compose_in_promotion() {
        let (blocks, t) = (2usize, 512usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = crate::kernel!(b,
            #[grid(threads = 32)]
            compute [blocks] |i| {
                let buf = compute [t] |j| { a[i * #t + j] };
                let c1 =
                    #[par((th, s) -> th * 16 + s)]
                    compute [t] |j| { buf[j] + buf[j + 8 - j % 16 / 8 * 16] };
                #[par((th, s) -> th * 16 + s)]
                compute [t] |j| { c1[j] * c1[j + 64 - j % 128 / 64 * 128] }
            }
        );
        let module = b.finish("par_compose", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        crate::passes::allocate_convert_scratch(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        // Under B.4-full, Shuffle-not-emittable composites (like the
        // gather↔`(th,s)→th*16+s` par-attr conversion here — a lane↔slot
        // swap with warp block trivially identity) route through
        // `gen_reg_bounce` via a shared scratch buffer instead of the
        // old Plan::Mirror path. So no Shared *alias* is emitted; the
        // scratch buffer is sized to a full tile (128 elements) and is
        // named `_cs<n>`.
        let mirror: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.name.ends_with("_sm"))
            .collect();
        assert_eq!(mirror.len(), 0, "no mirror under B.4-full");
        let scratch: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.name.contains("_cs") && !b.is_empty())
            .collect();
        assert!(!scratch.is_empty(), "at least one non-zero scratch buffer");
        // For the identity gather ↔ `(th,s)→th*16+s` conversion, the two
        // accesses touching scratch have distinct thread column subspaces
        // (I = span([16]), E and F are 4-dim complements), so B.6.2's
        // `choose_shared_layout` picks a non-trivial swizzle: bank bits
        // 5..9 get XOR-mixed with the lane bits, matching paper §5.4's
        // paired-XOR pattern. Verify by checking the scratch layout has
        // at least one non-power-of-two basis (evidence of XOR mixing).
        let scratch_ll = scratch[0].layout.as_ref().unwrap();
        assert!(
            scratch_ll.inverse().is_some(),
            "scratch layout is a bijection: {:?}",
            scratch_ll
        );
        let has_swizzle = scratch_ll.bases.iter().any(|&b| b.count_ones() > 1);
        assert!(
            has_swizzle,
            "expected an XOR-swizzled scratch layout (paper §5.4 paired-XOR), \
             got {:?}",
            scratch_ll
        );

        // Register buffers: at least the gather (identity) and the
        // promoted tile with the composed par-attr layout. Views for
        // mismatched readers are additional; each Bounce-fallback
        // conversion adds one Register alias (was Shared Mirror
        // pre-B.4-full).
        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        assert!(regs.len() >= 3);
        assert!(regs[0].layout.as_ref().unwrap().is_identity());
        let composed = LinearLayout {
            bases: vec![16, 32, 64, 128, 256, 1, 2, 4, 8],
            offset: 0,
        };
        assert_eq!(regs[1].layout.as_ref().unwrap(), &composed);

        let source = codegen(&kprog).unwrap();
        // __syncthreads still appears (now from gen_reg_bounce's internal
        // store→sync→load, not from the Mirror-consumer path).
        assert!(
            source.contains("__syncthreads();"),
            "bounce sync in output: {source}"
        );
    }

    /// Multiple tile writes (e.g. butterfly stages) block promotion: the
    /// buffer stays in shared memory with the identity layout. Under B.7
    /// greedy chaining its linear reader still lands on registers via a
    /// Shared→Register ConvertLayout placed right before the reader.
    #[test]
    fn multi_writer_tile_stays_shared() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            // A pack body writes the tile twice per point, like a butterfly.
            let tile = b.compute(t / 2, |b, j| {
                let tc = b.const_u32(t as u32 / 2);
                let base = b.mul(i, tc);
                let ix = b.add(base, j);
                let lo = b.index(a, &[ix]);
                b.pack(&[lo, lo])
            });
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let c2 = b.const_u32(2);
                    let row = b.div(j, c2);
                    let col = b.rem(j, c2);
                    b.index(tile, &[row, col])
                })
            })
        });
        let module = b.finish("multi_write_tile", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        // 0-byte convert scratch placeholders are also BufferKind::Shared;
        // filter by non-empty to check the tile itself.
        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Shared && !b.is_empty())
            .collect();
        assert_eq!(shared.len(), 1);
        assert!(shared[0].layout.as_ref().unwrap().is_identity());

        // The reader's access is a linear layout, so it reads through a
        // Shared→Register version; insert_sync fences the dirty shared
        // source before the convert.
        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        assert_eq!(regs.len(), 1, "one sh→reg version for the linear reader");
        assert_eq!(
            stmt_kinds(&kprog.kernels[0]),
            ["alloc", "alloc", "alloc", "par", "sync", "convert", "par"]
        );
    }
}
