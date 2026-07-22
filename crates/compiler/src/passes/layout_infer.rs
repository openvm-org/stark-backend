//! Layout inference over [`KernelProgram`]s.

use crate::{
    kernel_ir::{AddressSpace, KernelProgram, LinearLayout, ParAttr, SSAOpCode},
    passes::utils::ceil_log2,
};

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
        for op in kernel.ops_mut() {
            if let SSAOpCode::Par {
                bound,
                spans_grid,
                attr,
                ..
            } = &mut op.opcode
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
