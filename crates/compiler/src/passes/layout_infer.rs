//! Layout inference over [`KernelProgram`]s.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ir::VarId,
    kernel_ir::{
        classify_convert, AddressSpace, BufId, BufferDecl, BufferKind, ConvertKind, IndexMap,
        Kernel, KernelProgram, LinearLayout, ParAttr, SSABlock, SSANode, SSAOp, SSAOpCode,
    },
    passes::utils::ceil_log2,
};

/// Fills in the layout attributes left empty by lowering and promotes shared
/// tiles into registers by default (the design.md "Layout infer pass"):
///
/// 1. every access map that depends only on the par's own power-of-two index and is XOR-linear is
///    rewritten to [`IndexMap::Linear`];
/// 2. **every** shared tile written exactly once by a non-grid-spanning par over the tile's own
///    domain through an invertible linear map `g_w` is promoted to a register buffer with layout `L
///    = g_w ∘ f_w` (the writer par's own layout `f_w` applied first). Registers are the default; a
///    tile stays in shared only when the writer's write isn't XOR-linear, when the writer is
///    grid-spanning or multi-writer, or when the resulting `L` isn't invertible;
/// 3. each reader of a promoted tile with linear access map `g` under par layout `f` has the
///    effective map `E = g ∘ f` and is classified by `C = L^-1 ∘ E` ([`classify_convert`]): an
///    identity `C` reads the writer's registers in place; a slot permutation or warp shuffle reads
///    a register view with layout `E` filled by a [`SSAOpCode::ConvertLayout`]; anything else falls
///    back through one shared-memory mirror per tile, shared across every Bounce reader. View and
///    mirror allocs go right after the tile's own alloc (whose scope dominates every reader), the
///    conversions right after the writer par;
/// 4. remaining shared buffers (those that failed the promotion preconditions in step 2) get the
///    identity layout, and every per-block `par [N]` without a lowering-provided attr (`#[par]`)
///    gets the identity (strided) factorization `i = s * blockDim + t` with `seq_size = ceil(N /
///    block)` (grid-spanning pars have one point per thread, `seq_size = 1`).
pub fn layout_infer(p: &mut KernelProgram) {
    for ki in 0..p.kernels.len() {
        linearize_accesses(&p.buffers, &mut p.kernels[ki]);
        promote_tiles(p, ki);
    }
    for buf in &mut p.buffers {
        if buf.space == AddressSpace::Shared {
            buf.layout = Some(LinearLayout::identity(ceil_log2(buf.len())));
        }
    }
    for kernel in &mut p.kernels {
        let block = kernel.block;
        for op in kernel.ops_mut() {
            if let SSAOpCode::Par {
                bound,
                spans_grid,
                attr,
                ..
            } = &mut op.opcode
            {
                if attr.is_none() {
                    let seq_size = if *spans_grid {
                        1
                    } else {
                        bound.div_ceil(block)
                    };
                    *attr = Some(ParAttr {
                        seq_size,
                        layout: LinearLayout::identity(ceil_log2(seq_size) + ceil_log2(block)),
                    });
                }
            }
        }
    }
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

/// Statement position: enclosing loop (`None` = grid block) and index.
type StmtLoc = (Option<SSANode>, usize);

/// How a reader of a promoted tile consumes it.
enum Plan {
    /// `C` is the identity: the effective map equals the tile layout, so
    /// the read stays on the thread's own register slot.
    Direct,
    /// Slot permutation or warp shuffle into a register view with this
    /// effective map `g ∘ f` as its layout.
    View(LinearLayout),
    /// The read goes through a shared-memory mirror of the tile.
    Mirror,
}

struct Promotion {
    buf: BufId,
    /// Register layout `g_w ∘ f_w`: physical index to buffer index.
    l: LinearLayout,
    readers: Vec<(SSANode, usize, Plan)>,
    alloc_at: StmtLoc,
    convert_at: StmtLoc,
}

enum NewOp {
    Alloc(BufId),
    Convert {
        dst: BufId,
        src: BufId,
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

/// Design steps 2–3: register promotion of single-writer shared tiles.
fn promote_tiles(p: &mut KernelProgram, ki: usize) {
    let k = &p.kernels[ki];
    let mut tiles: BTreeMap<BufId, TileUse> = BTreeMap::new();
    walk_stmts(k, &p.buffers, None, &k.grid.block.body, &mut 0, &mut tiles);

    let mut promotions = Vec::new();
    'tiles: for (&buf, tile) in &tiles {
        let (Some(alloc_at), Some(convert_at), [(wnode, wai)]) =
            (tile.alloc, tile.writer_loc, tile.writes.as_slice())
        else {
            continue;
        };
        let SSAOpCode::Par {
            bound,
            spans_grid,
            attr: wattr,
            writes,
            ..
        } = &k.op(*wnode).opcode
        else {
            unreachable!("tile writer is a par")
        };
        let n = p.buffers[buf.0 as usize].len();
        if *spans_grid || *bound != n || !n.is_power_of_two() {
            continue;
        }
        let kb = n.trailing_zeros() as usize;
        let IndexMap::Linear(f) = &writes[*wai].index else {
            continue;
        };
        if f.bases.len() != kb {
            continue;
        }
        // Register layout: physical x runs logical f_w(x) and writes buffer
        // index g_w(f_w(x)). Explicit par layouts are lowering-validated
        // bijections, so `l` is invertible iff the access map is.
        let l = match wattr {
            Some(a) => f.compose(&a.layout),
            None => f.clone(),
        };
        let Some(l_inv) = l.inverse() else {
            continue;
        };

        let mut readers = Vec::new();
        for &(rnode, rai, rseq) in &tile.reads {
            let SSAOpCode::Par {
                bound: rbound,
                spans_grid: rgrid,
                attr: rattr,
                reads,
                ..
            } = &k.op(rnode).opcode
            else {
                unreachable!("tile reader is a par")
            };
            let a = &reads[rai];
            if rnode == *wnode {
                // The writer reads its own tile: only the exact map keeps
                // the read on the thread's own slot.
                match &a.index {
                    IndexMap::Linear(g) if g == f => readers.push((rnode, rai, Plan::Direct)),
                    _ => continue 'tiles,
                }
                continue;
            }
            if rseq < tile.writer_seq {
                // A loop back-edge read: the view/mirror would be consumed
                // before it is defined.
                continue 'tiles;
            }
            let plan = match &a.index {
                IndexMap::Linear(g) if !*rgrid && *rbound == n => {
                    let eff = match rattr {
                        Some(a) => g.compose(&a.layout),
                        None => g.clone(),
                    };
                    match classify_convert(&l_inv.compose(&eff), k.block) {
                        ConvertKind::Copy => Plan::Direct,
                        ConvertKind::Slot | ConvertKind::Shuffle => Plan::View(eff),
                        ConvertKind::Bounce => Plan::Mirror,
                    }
                }
                _ => Plan::Mirror,
            };
            readers.push((rnode, rai, plan));
        }
        promotions.push(Promotion {
            buf,
            l,
            readers,
            alloc_at,
            convert_at,
        });
    }

    let mut inserts: Vec<(Option<SSANode>, usize, Vec<NewOp>)> = Vec::new();
    for pr in promotions {
        let decl = &mut p.buffers[pr.buf.0 as usize];
        let (name, elem, shape) = (decl.name.clone(), decl.elem, decl.shape.clone());
        decl.kind = BufferKind::Register;
        decl.space = AddressSpace::Register;
        decl.layout = Some(pr.l.clone());
        let kb = pr.l.bases.len();

        let mut allocs = Vec::new();
        let mut converts = Vec::new();
        let mut views: Vec<(LinearLayout, BufId)> = Vec::new();
        let mut mirror = None;
        let mut rewrites: Vec<(SSANode, usize, BufId, Option<IndexMap>)> = Vec::new();
        for (rnode, rai, plan) in pr.readers {
            match plan {
                Plan::Direct => {}
                Plan::View(eff) => {
                    // The view's layout is the reader's effective map, so its
                    // original access resolves to the thread's own slot; the
                    // identity-map conversion re-lays the tile out as `E`.
                    let vb = match views.iter().find(|(ve, _)| *ve == eff) {
                        Some(&(_, id)) => id,
                        None => {
                            let id = BufId(p.buffers.len() as u32);
                            p.buffers.push(BufferDecl {
                                name: format!("{name}_v{}", views.len()),
                                elem,
                                shape: vec![1 << kb],
                                kind: BufferKind::Register,
                                space: AddressSpace::Register,
                                layout: Some(eff.clone()),
                            });
                            allocs.push(NewOp::Alloc(id));
                            converts.push(NewOp::Convert {
                                dst: id,
                                src: pr.buf,
                                map: LinearLayout::identity(kb),
                            });
                            views.push((eff, id));
                            id
                        }
                    };
                    rewrites.push((rnode, rai, vb, None));
                }
                Plan::Mirror => {
                    let mb = *mirror.get_or_insert_with(|| {
                        let id = BufId(p.buffers.len() as u32);
                        p.buffers.push(BufferDecl {
                            name: format!("{name}_sm"),
                            elem,
                            shape: shape.clone(),
                            kind: BufferKind::Shared,
                            space: AddressSpace::Shared,
                            layout: None,
                        });
                        allocs.push(NewOp::Alloc(id));
                        converts.push(NewOp::Convert {
                            dst: id,
                            src: pr.buf,
                            map: LinearLayout::identity(kb),
                        });
                        id
                    });
                    rewrites.push((rnode, rai, mb, None));
                }
            }
        }

        let kernel = &mut p.kernels[ki];
        for (rnode, rai, nb, nix) in rewrites {
            let op = &mut kernel.ops_mut()[rnode.0 as usize];
            let SSAOpCode::Par { reads, .. } = &mut op.opcode else {
                unreachable!("reader is a par")
            };
            reads[rai].buf = nb;
            if let Some(ix) = nix {
                reads[rai].index = ix;
            }
        }
        inserts.push((pr.alloc_at.0, pr.alloc_at.1 + 1, allocs));
        inserts.push((pr.convert_at.0, pr.convert_at.1 + 1, converts));
    }

    // Per-block indices were recorded before any insertion; applying in
    // descending index order keeps every recorded index valid.
    inserts.sort_by(|a, b| b.1.cmp(&a.1));
    let kernel = &mut p.kernels[ki];
    for (parent, at, ops) in inserts {
        let nodes: Vec<SSANode> = ops
            .into_iter()
            .map(|o| {
                let opcode = match o {
                    NewOp::Alloc(buf) => SSAOpCode::Alloc { buf },
                    NewOp::Convert { dst, src, map } => SSAOpCode::ConvertLayout { dst, src, map },
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

/// Pre-order walk of the statement blocks, recording shared-tile allocs,
/// writers and readers. `seq` numbers statements in program order so that
/// reader-before-writer (back-edge) uses can be detected.
fn walk_stmts(
    k: &Kernel,
    buffers: &[BufferDecl],
    parent: Option<SSANode>,
    body: &[SSANode],
    seq: &mut usize,
    tiles: &mut BTreeMap<BufId, TileUse>,
) {
    for (i, &nid) in body.iter().enumerate() {
        let op = k.op(nid);
        *seq += 1;
        let my_seq = *seq;
        match &op.opcode {
            SSAOpCode::Alloc { buf } if buffers[buf.0 as usize].kind == BufferKind::Shared => {
                tiles.entry(*buf).or_default().alloc = Some((parent, i));
            }
            SSAOpCode::Par { reads, writes, .. } => {
                for (ai, a) in writes.iter().enumerate() {
                    if buffers[a.buf.0 as usize].kind == BufferKind::Shared {
                        let t = tiles.entry(a.buf).or_default();
                        t.writes.push((nid, ai));
                        t.writer_loc = Some((parent, i));
                        t.writer_seq = my_seq;
                    }
                }
                for (ai, a) in reads.iter().enumerate() {
                    if buffers[a.buf.0 as usize].kind == BufferKind::Shared {
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
        assert_eq!(regs[0].shape, vec![t]);
        assert!(regs[0].layout.as_ref().unwrap().is_identity());
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
            ["alloc", "alloc", "par", "convert", "par"]
        );
        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        assert_eq!(regs.len(), 2, "tile and its view");
        assert!(!kprog.buffers.iter().any(|b| b.kind == BufferKind::Shared));

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shfl_sync"), "{source}");
        assert!(!source.contains("__syncthreads"), "{source}");
    }

    /// A non-linear reader of a promoted tile goes through a shared-memory
    /// mirror, and the mirror write still gets a barrier before the reader.
    #[test]
    fn non_linear_reader_gets_shared_mirror() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = own_index_tile(b, a, i, t);
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let last = b.const_u32(t as u32 - 1);
                    let rev = b.sub(last, j);
                    let x = b.index(tile, &[rev]);
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
            ["alloc", "alloc", "par", "convert", "sync", "par"]
        );
        let tile = kprog.buffers.iter().find(|b| b.name.ends_with("_sm"));
        let mirror = tile.expect("mirror buffer");
        assert_eq!(mirror.kind, BufferKind::Shared);
        assert_eq!(mirror.shape, vec![t]);

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shared__"), "{source}");
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
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Shared)
            .collect();
        assert_eq!(shared.len(), 1, "one mirror for the identity gather");
        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        // gather (identity), promoted tile (f_par), and its shuffle view.
        assert_eq!(regs.len(), 3);
        assert!(regs[0].layout.as_ref().unwrap().is_identity());
        // (th, s) -> th*16 + s over th < 32, s < 16: thread bits land in
        // logical bits 4..9, seq bits in 0..4.
        let composed = LinearLayout {
            bases: vec![16, 32, 64, 128, 256, 1, 2, 4, 8],
            offset: 0,
        };
        assert_eq!(regs[1].layout.as_ref().unwrap(), &composed);
        let view = LinearLayout {
            offset: 64,
            ..composed
        };
        assert_eq!(regs[2].layout.as_ref().unwrap(), &view);

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shfl_sync"), "{source}");
        assert_eq!(source.matches("__syncthreads();").count(), 1, "{source}");
    }

    /// Multiple tile writes (e.g. butterfly stages) block promotion: the
    /// buffer stays in shared memory with the identity layout.
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

        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Shared)
            .collect();
        assert_eq!(shared.len(), 1);
        assert!(shared[0].layout.as_ref().unwrap().is_identity());
    }
}
