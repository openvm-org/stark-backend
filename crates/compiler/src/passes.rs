//! KernelIR passes: layout inference and synchronization insertion.

use std::collections::BTreeSet;

use crate::kernel_ir::{AddressSpace, KExpr, KernelProgram, LinearLayout, ParAttr, Stmt};

fn ceil_log2(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// Fills in the layout attributes left empty by lowering. The initial
/// implementation assigns every `par [N]` the identity (strided)
/// factorization `i = s * blockDim + t` with `seq_size = ceil(N / block)`,
/// and every shared buffer the identity map `i -> i`.
pub fn layout_infer(p: &mut KernelProgram) {
    for buf in &mut p.buffers {
        if buf.space == AddressSpace::Shared {
            buf.layout = Some(LinearLayout::identity(ceil_log2(buf.len())));
        }
    }
    for kernel in &mut p.kernels {
        infer_par_attrs(&mut kernel.body, kernel.block);
    }
}

fn infer_par_attrs(stmts: &mut [Stmt], block: usize) {
    for stmt in stmts {
        match stmt {
            Stmt::Par {
                bound, attr, body, ..
            } => {
                let seq_size = bound.div_ceil(block);
                *attr = Some(ParAttr {
                    seq_size,
                    layout: LinearLayout::identity(ceil_log2(seq_size) + ceil_log2(block)),
                });
                infer_par_attrs(body, block);
            }
            Stmt::Loop { body, .. } => infer_par_attrs(body, block),
            _ => {}
        }
    }
}

/// Inserts `Sync` barriers between `Par` statements where a par reads a
/// shared buffer written by an earlier par since the last barrier (RAW).
/// Each shared buffer is written by exactly one par (single assignment), so
/// WAW/WAR hazards cannot occur.
pub fn insert_sync(p: &mut KernelProgram) {
    let shared: BTreeSet<usize> = p
        .buffers
        .iter()
        .enumerate()
        .filter(|(_, b)| b.space == AddressSpace::Shared)
        .map(|(i, _)| i)
        .collect();
    if shared.is_empty() {
        return;
    }
    for kernel in &mut p.kernels {
        let mut dirty: BTreeSet<usize> = BTreeSet::new();
        let mut body = Vec::with_capacity(kernel.body.len());
        for stmt in kernel.body.drain(..) {
            if let Stmt::Par { body: pbody, .. } = &stmt {
                let mut reads = BTreeSet::new();
                let mut writes = BTreeSet::new();
                collect_shared_accesses(pbody, &shared, &mut reads, &mut writes);
                if !reads.is_disjoint(&dirty) {
                    body.push(Stmt::Sync);
                    dirty.clear();
                }
                dirty.extend(writes);
            }
            body.push(stmt);
        }
        kernel.body = body;
    }
}

fn collect_shared_accesses(
    stmts: &[Stmt],
    shared: &BTreeSet<usize>,
    reads: &mut BTreeSet<usize>,
    writes: &mut BTreeSet<usize>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Def {
                expr: KExpr::Load { buf, .. },
                ..
            } if shared.contains(buf) => {
                reads.insert(*buf);
            }
            Stmt::Store { buf, .. } if shared.contains(buf) => {
                writes.insert(*buf);
            }
            Stmt::Loop { body, .. } | Stmt::Par { body, .. } => {
                collect_shared_accesses(body, shared, reads, writes);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        canonicalize::canonicalize,
        ir::{IRBuilder, ScalarType},
        kernel_ir::LaunchShape,
        lower::lower,
    };

    /// The shared-memory tile pattern lowers to a Grid kernel with an
    /// `Alloc`ed shared buffer and a `Sync` between the producing and the
    /// consuming `Par`.
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

        let program = canonicalize(module).unwrap();
        let mut kprog = lower(&program).unwrap();
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);

        assert_eq!(kprog.kernels.len(), 1);
        let kernel = &kprog.kernels[0];
        assert!(matches!(kernel.launch, LaunchShape::Grid { n } if n == blocks));

        let kinds: Vec<&'static str> = kernel
            .body
            .iter()
            .map(|s| match s {
                Stmt::Alloc { .. } => "alloc",
                Stmt::Par { .. } => "par",
                Stmt::Sync => "sync",
                _ => "other",
            })
            .filter(|&k| k != "other")
            .collect();
        assert_eq!(kinds, ["alloc", "par", "sync", "par"]);

        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.space == AddressSpace::Shared)
            .collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].shape, vec![t]);
        assert!(shared[0].layout.as_ref().unwrap().is_identity());

        let source = crate::codegen::generate_cuda(&kprog);
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
                        let x = b.index(t1, &[j]);
                        let y = b.index(t2, &[j]);
                        b.add(x, y)
                    })
                })
            })
        });
        let module = b.finish("two_tiles", body);

        let program = canonicalize(module).unwrap();
        let mut kprog = lower(&program).unwrap();
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);

        let syncs = kprog.kernels[0]
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::Sync))
            .count();
        assert_eq!(syncs, 1);
    }
}
