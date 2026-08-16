//! Barrier insertion over [`KirProgram`]s.

use std::collections::BTreeSet;

use crate::kernel_ir::{
    AddressSpace, BufId, Kernel, KirProgram, SSABlock, SSANode, SSAOp, SSAOpCode,
};

/// Inserts [`SSAOpCode::Sync`] barriers: a par that reads a shared buffer
/// written since the last barrier gets a `Sync` right before it, and a
/// write to a shared buffer whose memory may still be read by a concurrent
/// thread also gets one — this second case matters when
/// [`crate::passes::plan_shared_mem`] aliases disjoint-lifetime shared
/// buffers into the same physical region. A sequential loop marks its
/// body's shared writes dirty up front, since iteration i+1 may read what
/// iteration i wrote (the back edge).
pub fn insert_sync(p: &mut KirProgram) {
    /// Walks a statement block; `sid` is its owning loop (`None` for the
    /// grid block). Records `(block, index)` sync insertion points.
    fn walk(
        p: &KirProgram,
        k: &Kernel,
        sid: Option<SSANode>,
        body: &[SSANode],
        dirty: &mut BTreeSet<BufId>,
        reads_since_sync: &mut bool,
        points: &mut Vec<(Option<SSANode>, usize)>,
    ) {
        for (i, &nid) in body.iter().enumerate() {
            let op = k.op(nid);
            let mut synced = false;
            let mut sync_here = |dirty: &mut BTreeSet<BufId>, reads_since_sync: &mut bool| {
                if !synced {
                    points.push((sid, i));
                    dirty.clear();
                    *reads_since_sync = false;
                }
                synced = true;
            };
            match &op.opcode {
                SSAOpCode::Loop { .. } => {
                    collect_shared_writes(p, k, &op.block.body, dirty);
                    walk(
                        p,
                        k,
                        Some(nid),
                        &op.block.body,
                        dirty,
                        reads_since_sync,
                        points,
                    );
                }
                SSAOpCode::Par { reads, writes, .. } => {
                    if reads.iter().any(|a| dirty.contains(&a.buf)) {
                        sync_here(dirty, reads_since_sync);
                    }
                    if writes
                        .iter()
                        .any(|a| p.buffer(a.buf).space == AddressSpace::Shared)
                        && *reads_since_sync
                    {
                        sync_here(dirty, reads_since_sync);
                    }
                    for a in reads {
                        if p.buffer(a.buf).space == AddressSpace::Shared {
                            *reads_since_sync = true;
                        }
                    }
                    for a in writes {
                        if p.buffer(a.buf).space == AddressSpace::Shared {
                            dirty.insert(a.buf);
                        }
                    }
                }
                SSAOpCode::ConvertLayout {
                    dst, src, scratch, ..
                } => {
                    if dirty.contains(src) {
                        sync_here(dirty, reads_since_sync);
                    }
                    if p.buffer(*dst).space == AddressSpace::Shared && *reads_since_sync {
                        sync_here(dirty, reads_since_sync);
                    }
                    // Scratch is a shared buffer used strictly inside this
                    // op (Phase B.4-full's bounce: store → __syncthreads →
                    // load). The inner sync is codegen-emitted; the outer
                    // walk still marks scratch as both read and written so
                    // the packer's aliasing (`plan_shared_mem`) can't
                    // schedule an unrelated shared buffer to overlap this
                    // op's scratch region without a proper barrier.
                    if p.buffer(*scratch).space == AddressSpace::Shared
                        && !p.buffer(*scratch).is_empty()
                    {
                        if *reads_since_sync {
                            sync_here(dirty, reads_since_sync);
                        }
                        *reads_since_sync = true;
                        dirty.insert(*scratch);
                    }
                    if p.buffer(*src).space == AddressSpace::Shared {
                        *reads_since_sync = true;
                    }
                    if p.buffer(*dst).space == AddressSpace::Shared {
                        dirty.insert(*dst);
                    }
                }
                _ => {}
            }
        }
    }

    for ki in 0..p.kernels.len() {
        let mut points = Vec::new();
        let k = &p.kernels[ki];
        let mut dirty = BTreeSet::new();
        let mut reads_since_sync = false;
        walk(
            p,
            k,
            None,
            &k.grid.block.body,
            &mut dirty,
            &mut reads_since_sync,
            &mut points,
        );
        // Per-block indices ascend in walk order, so applying in reverse
        // keeps every recorded index valid.
        let k = &mut p.kernels[ki];
        for (sid, i) in points.into_iter().rev() {
            let sync = k.push_op(SSAOp {
                operands: Default::default(),
                results: Default::default(),
                opcode: SSAOpCode::Sync,
                block: SSABlock::default(),
            });
            match sid {
                None => k.grid.block.body.insert(i, sync),
                Some(sid) => k.ops_mut()[sid.0 as usize].block.body.insert(i, sync),
            }
        }
    }
}

/// Shared buffers written by any par under `stmts`.
fn collect_shared_writes(p: &KirProgram, k: &Kernel, stmts: &[SSANode], out: &mut BTreeSet<BufId>) {
    for &sid in stmts {
        let op = k.op(sid);
        match &op.opcode {
            SSAOpCode::Par { writes, .. } => {
                for a in writes {
                    if p.buffer(a.buf).space == AddressSpace::Shared {
                        out.insert(a.buf);
                    }
                }
            }
            SSAOpCode::ConvertLayout { dst, scratch, .. } => {
                if p.buffer(*dst).space == AddressSpace::Shared {
                    out.insert(*dst);
                }
                if p.buffer(*scratch).space == AddressSpace::Shared
                    && !p.buffer(*scratch).is_empty()
                {
                    out.insert(*scratch);
                }
            }
            SSAOpCode::Loop { .. } => collect_shared_writes(p, k, &op.block.body, out),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{IRBuilder, ScalarType},
        passes::{
            codegen, layout_infer,
            utils::test_util::{lowered, stmt_kinds},
            verify,
        },
    };

    /// The shared-memory tile pattern lowers to a grid kernel where the
    /// producer par writes to a promoted register tile, a
    /// [`SSAOpCode::ConvertLayout`] mirrors it to shared for the bouncing
    /// consumer, and `insert_sync` places a barrier before the consumer.
    ///
    /// The consumer's read index folds through a symbolic constant
    /// (`(j + #m) % t`) so `linearize_accesses` can't turn it into a
    /// `Linear` map — the read stays `SExpr`, layout inference routes
    /// it through the shared-memory mirror, and `insert_sync` places a
    /// barrier before the consumer par.
    #[test]
    fn shared_tile_gets_alloc_and_sync() {
        let (blocks, t) = (4usize, 8usize);
        let mut b = IRBuilder::new();
        let m = b.symbol("m");
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = b.compute(t, |b, j| {
                let tc = b.const_u32(t as u32);
                let base = b.mul(i, tc);
                let ix = b.add(base, j);
                b.index(a, &[ix])
            });
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let mv = b.const_sym(m);
                    let off = b.add(j, mv);
                    let tc = b.const_u32(t as u32);
                    let ix = b.rem(off, tc);
                    b.index(tile, &[ix])
                })
            })
        });
        let module = b.finish("tile", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        assert_eq!(kprog.kernels.len(), 1);
        let kernel = &kprog.kernels[0];
        assert_eq!(kernel.grid.bound.as_const(), Some(blocks));
        // Warp-aligned launch (Gap 1): sub-warp compute[t] runs on 32
        // threads with a replicated par-attr.
        assert_eq!(kernel.block, 32);
        assert_eq!(
            stmt_kinds(kernel),
            ["alloc", "alloc", "alloc", "par", "convert", "sync", "par"]
        );

        // The producer par writes the register tile; its grid-var capture
        // and result count still match one write.
        let g = kernel.grid_var();
        for &sid in &kernel.grid.block.body {
            let op = kernel.op(sid);
            let SSAOpCode::Par { writes, .. } = &op.opcode else {
                continue;
            };
            assert_eq!(op.operands.as_slice(), [g]);
            assert_eq!(op.results.len(), writes.len());
            assert_eq!(op.results.len(), op.block.yields.len());
        }

        // Phase B.2 attaches a 0-byte scratch buffer to every
        // ConvertLayout; filter to the real mirror by non-zero size.
        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.space == AddressSpace::Shared && !b.is_empty())
            .collect();
        assert_eq!(shared.len(), 1, "the bouncing consumer gets one mirror");
        assert_eq!(shared[0].shape, vec![crate::ir::SizeExpr::from(t)]);
        assert!(shared[0].layout.as_ref().unwrap().is_identity());

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shared__"));
        assert!(source.contains("__syncthreads();"));
    }

    /// Back-to-back tile producers do not need a barrier between them; the
    /// barrier goes before the first consumer that reads their mirrors.
    /// Both consumer reads go through symbolic-modulus indices so
    /// `linearize_accesses` can't linearize them — they stay `SExpr`
    /// and route through the shared mirror, exercising the "shared read"
    /// path in `insert_sync`.
    #[test]
    fn independent_tiles_share_one_sync() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let m = b.symbol("m");
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let t1 = b.compute(t, |b, j| {
                let tc = b.const_u32(t as u32);
                let base = b.mul(i, tc);
                let ix = b.add(base, j);
                b.index(a, &[ix])
            });
            b.bind(t1, |b, t1| {
                let t2 = b.compute(t, |b, j| {
                    let tc = b.const_u32(t as u32);
                    let base = b.mul(i, tc);
                    let ix = b.add(base, j);
                    let v = b.index(a, &[ix]);
                    b.mul(v, v)
                });
                b.bind(t2, |b, t2| {
                    b.compute(t, |b, j| {
                        // `(j + #m) % t` keeps the read `SExpr` so both
                        // tiles bounce through shared memory (the only
                        // reliably-Mirror-planning path post-Gap-4).
                        let mv = b.const_sym(m);
                        let off = b.add(j, mv);
                        let tc = b.const_u32(t as u32);
                        let ix = b.rem(off, tc);
                        let x = b.index(t1, &[ix]);
                        let y = b.index(t2, &[ix]);
                        b.add(x, y)
                    })
                })
            })
        });
        let module = b.finish("two_tiles", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();

        // Each tile becomes register + mirror; only one barrier is needed
        // before the reader that bounces both mirrors. Each ConvertLayout
        // gets an extra 0-byte scratch alloc (Phase B.2 placeholder).
        assert_eq!(
            stmt_kinds(&kprog.kernels[0]),
            [
                "alloc", "alloc", "alloc", "par", "convert", "alloc", "alloc", "alloc", "par",
                "convert", "sync", "par"
            ]
        );
        let source = codegen(&kprog).unwrap();
        assert_eq!(source.matches("__syncthreads();").count(), 1);
    }

    /// Phase B.3: a ConvertLayout with a non-zero scratch buffer is
    /// treated by the dirty walk as both reading and writing shared
    /// memory. Today's layout_infer never sizes scratch > 0 (every
    /// emitted convert is a pure shuffle or a direct mirror), so we
    /// simulate the future B.4-full bounce path by force-resizing the
    /// scratch buffer after layout_infer.
    ///
    /// The check is that the walk terminates cleanly, verify passes,
    /// and codegen still succeeds — a smoke test that the extended
    /// dirty walk handles the non-zero-scratch case without regressing
    /// today's shuffle-view output.
    #[test]
    fn nonzero_scratch_survives_dirty_walk() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            // Own-index producer promotes to registers, `tile[j % (t/2)]`
            // reader forces a ConvertLayout emission (View path).
            let tile = b.compute(t, |b, j| {
                let tc = b.const_u32(t as u32);
                let base = b.mul(i, tc);
                let ix = b.add(base, j);
                b.index(a, &[ix])
            });
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
        let module = b.finish("nonzero_scratch", body);

        let mut kprog = lowered(module);
        layout_infer(&mut kprog);
        // Force the scratch buffer(s) non-zero to exercise the B.3
        // dirty-walk extension. In real B.4-full runs
        // `allocate_convert_scratch` sizes them.
        for buf in kprog.buffers.iter_mut() {
            if buf.name.contains("_cs") {
                buf.shape = vec![crate::ir::SizeExpr::from(t)];
                buf.layout = Some(crate::kernel_ir::LinearLayout::identity(
                    t.trailing_zeros() as usize
                ));
            }
        }
        insert_sync(&mut kprog);
        verify(&kprog).unwrap();
        // Convert is still a pure shuffle at emission time, but the
        // dirty walk must not have added spurious syncs from the
        // scratch handling.
        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shfl_sync"));
    }
}
