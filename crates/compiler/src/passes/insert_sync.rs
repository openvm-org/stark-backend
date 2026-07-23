//! Barrier insertion over [`KernelProgram`]s.

use std::collections::BTreeSet;

use crate::kernel_ir::{
    AddressSpace, BufId, Kernel, KernelProgram, SSABlock, SSANode, SSAOp, SSAOpCode,
};

/// Inserts [`SSAOpCode::Sync`] barriers: a par that reads a shared buffer
/// written since the last barrier gets a `Sync` right before it, and a
/// write to a shared buffer whose memory may still be read by a concurrent
/// thread also gets one — this second case matters when
/// [`crate::passes::plan_shared_mem`] aliases disjoint-lifetime shared
/// buffers into the same physical region. A sequential loop marks its
/// body's shared writes dirty up front, since iteration i+1 may read what
/// iteration i wrote (the back edge).
pub fn insert_sync(p: &mut KernelProgram) {
    /// Walks a statement block; `sid` is its owning loop (`None` for the
    /// grid block). Records `(block, index)` sync insertion points.
    fn walk(
        p: &KernelProgram,
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
                SSAOpCode::ConvertLayout { dst, src, .. } => {
                    if dirty.contains(src) {
                        sync_here(dirty, reads_since_sync);
                    }
                    if p.buffer(*dst).space == AddressSpace::Shared && *reads_since_sync {
                        sync_here(dirty, reads_since_sync);
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
fn collect_shared_writes(
    p: &KernelProgram,
    k: &Kernel,
    stmts: &[SSANode],
    out: &mut BTreeSet<BufId>,
) {
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
            SSAOpCode::ConvertLayout { dst, .. } => {
                if p.buffer(*dst).space == AddressSpace::Shared {
                    out.insert(*dst);
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
    #[test]
    fn shared_tile_gets_alloc_and_sync() {
        let (blocks, t) = (4usize, 8usize);
        let mut b = IRBuilder::new();
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
                    let last = b.const_u32(t as u32 - 1);
                    let rev = b.sub(last, j);
                    b.index(tile, &[rev])
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
        assert_eq!(kernel.grid.bound, blocks);
        assert_eq!(kernel.block, t);
        assert_eq!(
            stmt_kinds(kernel),
            ["alloc", "alloc", "par", "convert", "sync", "par"]
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

        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.space == AddressSpace::Shared)
            .collect();
        assert_eq!(shared.len(), 1, "the bouncing consumer gets one mirror");
        assert_eq!(shared[0].shape, vec![t]);
        assert!(shared[0].layout.as_ref().unwrap().is_identity());

        let source = codegen(&kprog).unwrap();
        assert!(source.contains("__shared__ uint32_t"));
        assert!(source.contains("__syncthreads();"));
    }

    /// Back-to-back tile producers do not need a barrier between them; the
    /// barrier goes before the first consumer.
    #[test]
    fn independent_tiles_share_one_sync() {
        let (blocks, t) = (2usize, 8usize);
        let mut b = IRBuilder::new();
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
                        // Reversed reads keep both tiles in shared memory
                        // (`t - 1 - j` is affine-XOR, not linear, so the
                        // readers cannot be served from registers).
                        let last = b.const_u32(t as u32 - 1);
                        let rev = b.sub(last, j);
                        let x = b.index(t1, &[rev]);
                        let y = b.index(t2, &[rev]);
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
        // before the reader that bounces both mirrors.
        assert_eq!(
            stmt_kinds(&kprog.kernels[0]),
            [
                "alloc", "alloc", "par", "convert", "alloc", "alloc", "par", "convert", "sync",
                "par"
            ]
        );
        let source = codegen(&kprog).unwrap();
        assert_eq!(source.matches("__syncthreads();").count(), 1);
    }
}
