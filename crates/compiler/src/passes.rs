//! KernelIR passes: layout inference.
//!
//! Synchronization is not a pass anymore: codegen derives `__syncthreads()`
//! barriers from the pars' declared reads and writes.

use crate::kernel_ir::{AddressSpace, KernelProgram, LinearLayout, ParAttr, Stmt};

fn ceil_log2(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// Fills in the layout attributes left empty by lowering. The initial
/// implementation assigns every per-block `par [N]` the identity (strided)
/// factorization `i = s * blockDim + t` with `seq_size = ceil(N / block)`
/// (grid-spanning pars have one point per thread, `seq_size = 1`), and every
/// shared buffer the identity map `i -> i`.
pub fn layout_infer(p: &mut KernelProgram) {
    for buf in &mut p.buffers {
        if buf.space == AddressSpace::Shared {
            buf.layout = Some(LinearLayout::identity(ceil_log2(buf.len())));
        }
    }
    for kernel in &mut p.kernels {
        let block = kernel.block;
        for stmt in kernel.stmts_mut() {
            if let Stmt::Par {
                bound,
                spans_grid,
                attr,
                ..
            } = stmt
            {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        canonicalize::canonicalize,
        ir::{IRBuilder, ReduceOp, ScalarType},
        kernel_ir::{BufferKind, Kernel},
        lower::lower,
    };

    fn stmt_kinds(kernel: &Kernel) -> Vec<&'static str> {
        kernel
            .grid
            .body
            .iter()
            .map(|&id| match kernel.stmt(id) {
                Stmt::BufDecl { .. } => "bufdecl",
                Stmt::Par { .. } => "par",
                Stmt::Loop { .. } => "loop",
            })
            .collect()
    }

    /// The shared-memory tile pattern lowers to a grid kernel with a
    /// `BufDecl`ed shared buffer, and codegen derives a barrier between the
    /// producing and the consuming `Par`.
    #[test]
    fn shared_tile_gets_bufdecl_and_sync() {
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

        assert_eq!(kprog.kernels.len(), 1);
        let kernel = &kprog.kernels[0];
        assert_eq!(kernel.grid.bound, blocks);
        assert_eq!(kernel.block, t);
        assert_eq!(stmt_kinds(kernel), ["bufdecl", "par", "par"]);

        let shared: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.space == AddressSpace::Shared)
            .collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].shape, vec![t]);
        assert!(shared[0].layout.as_ref().unwrap().is_identity());

        let source = crate::codegen::generate_cuda(&kprog).unwrap();
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

        let source = crate::codegen::generate_cuda(&kprog).unwrap();
        assert_eq!(source.matches("__syncthreads();").count(), 1);
    }

    /// A reduce whose body loads from memory is hoisted out of its par: a
    /// register accumulator, an init par, a sequential loop around an
    /// accumulate par, and the consumer par reading the accumulator.
    #[test]
    fn reduce_with_loads_hoists_to_register_loop() {
        let (n, k) = (8usize, 4usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![k * n]);
        let body = b.compute(n, |b, i| {
            b.reduce(ReduceOp::Add, k, |b, j| {
                let nc = b.const_u32(n as u32);
                let base = b.mul(j, nc);
                let ix = b.add(base, i);
                b.index(a, &[ix])
            })
        });
        let module = b.finish("colsum", body);

        let program = canonicalize(module).unwrap();
        let mut kprog = lower(&program).unwrap();
        layout_infer(&mut kprog);

        assert_eq!(kprog.kernels.len(), 1);
        let kernel = &kprog.kernels[0];
        assert_eq!(stmt_kinds(kernel), ["bufdecl", "par", "loop", "par"]);

        let regs: Vec<_> = kprog
            .buffers
            .iter()
            .filter(|b| b.kind == BufferKind::Register)
            .collect();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].shape, vec![n]);

        let loop_id = kernel.grid.body[2];
        let Stmt::Loop { bound, body, .. } = kernel.stmt(loop_id) else {
            panic!("expected loop");
        };
        assert_eq!(*bound, k);
        assert_eq!(body.len(), 1);
        assert!(matches!(kernel.stmt(body[0]), Stmt::Par { .. }));
    }
}
