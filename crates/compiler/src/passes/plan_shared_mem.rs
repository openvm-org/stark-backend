//! Shared-memory placement over [`KernelProgram`]s.

use std::collections::HashMap;

use crate::kernel_ir::{BufId, BufferKind, Kernel, KernelProgram, SSANode, SSAOpCode};

/// Packed placement of shared buffers within each kernel's shared-memory
/// allocation, in alloc order (mirroring the declaration order codegen
/// emits).
pub struct SharedMemPlan {
    /// Byte offset of each shared buffer within its kernel's allocation.
    pub offsets: HashMap<BufId, usize>,
    /// Total shared bytes of each kernel, indexed like
    /// [`KernelProgram::kernels`].
    pub per_kernel: Vec<usize>,
}

/// Lays out each kernel's shared buffers back to back in alloc order.
pub fn plan_shared_mem(p: &KernelProgram) -> SharedMemPlan {
    fn walk(
        p: &KernelProgram,
        k: &Kernel,
        stmts: &[SSANode],
        total: &mut usize,
        offsets: &mut HashMap<BufId, usize>,
    ) {
        for &sid in stmts {
            let op = k.op(sid);
            match &op.opcode {
                SSAOpCode::Alloc { buf } => {
                    let decl = p.buffer(*buf);
                    if decl.kind == BufferKind::Shared {
                        offsets.insert(*buf, *total);
                        *total += decl.phys_len() * 4;
                    }
                }
                SSAOpCode::Loop { .. } => walk(p, k, &op.block.body, total, offsets),
                _ => {}
            }
        }
    }
    let mut offsets = HashMap::new();
    let mut per_kernel = Vec::with_capacity(p.kernels.len());
    for k in &p.kernels {
        let mut total = 0;
        walk(p, k, &k.grid.block.body, &mut total, &mut offsets);
        per_kernel.push(total);
    }
    SharedMemPlan {
        offsets,
        per_kernel,
    }
}
