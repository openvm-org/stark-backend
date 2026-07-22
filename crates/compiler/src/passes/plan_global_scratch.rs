//! Global scratch placement over canonical [`Program`]s.

use std::collections::{HashMap, HashSet};

use crate::{
    ir::{IRBuilder, Node, NodeId, Type},
    passes::{
        canonicalize::{CanonKernel, CanonValue, Program, ResultExpr, TensorRef},
        utils::resolve_tensor_ref,
    },
    CompileError,
};

const SCRATCH_ALIGN: usize = 256;

/// Placement of the intermediate (non-output) kernel results in one global
/// scratch arena.
pub struct GlobalScratchPlan {
    /// Byte offset of each intermediate tensor within the arena.
    pub offsets: HashMap<TensorRef, usize>,
    /// Arena size in bytes.
    pub total_bytes: usize,
}

/// Assigns scratch offsets with an interval-liveness first-fit allocator:
/// each intermediate tensor is live from the kernel that writes it through
/// the last kernel that reads it, and dead tensors' storage is reused.
pub fn plan_global_scratch(program: &Program) -> Result<GlobalScratchPlan, CompileError> {
    // Liveness intervals `(tref, def, last_read, size)` in def order.
    let mut intervals: Vec<(TensorRef, usize, usize, usize)> = Vec::new();
    let mut index_of: HashMap<TensorRef, usize> = HashMap::new();
    for (let_id, ck) in program.kernels.iter().enumerate() {
        for (out_idx, member_ty) in ck.member_types.iter().enumerate() {
            let tref = TensorRef::Let { let_id, out_idx };
            if program.outputs.contains(&tref) {
                continue;
            }
            // Non-tensor members are reported by lowering.
            let Type::Tensor(_, shape) = member_ty else {
                continue;
            };
            let size = (shape.iter().product::<usize>() * 4).next_multiple_of(SCRATCH_ALIGN);
            index_of.insert(tref, intervals.len());
            intervals.push((tref, let_id, let_id, size));
        }
    }
    for (k_idx, ck) in program.kernels.iter().enumerate() {
        for tref in read_refs(program, ck)? {
            if let Some(&i) = index_of.get(&tref) {
                intervals[i].2 = intervals[i].2.max(k_idx);
            }
        }
    }

    let mut free: Vec<(usize, usize)> = Vec::new(); // (offset, size), sorted
    let mut active: Vec<(usize, usize, usize)> = Vec::new(); // (end, offset, size)
    let mut high = 0usize;
    let mut offsets = HashMap::new();

    for &(tref, def, end, size) in &intervals {
        // Release allocations that died before this kernel.
        let mut still_active = Vec::new();
        for (a_end, a_off, a_size) in active.drain(..) {
            if a_end < def {
                free_insert(&mut free, a_off, a_size);
            } else {
                still_active.push((a_end, a_off, a_size));
            }
        }
        active = still_active;

        let offset = match free.iter().position(|&(_, hole_size)| hole_size >= size) {
            Some(i) => {
                let (off, hole_size) = free[i];
                if hole_size == size {
                    free.remove(i);
                } else {
                    free[i] = (off + size, hole_size - size);
                }
                off
            }
            None => {
                let off = high;
                high += size;
                off
            }
        };
        offsets.insert(tref, offset);
        active.push((end, offset, size));
    }

    Ok(GlobalScratchPlan {
        offsets,
        total_bytes: high,
    })
}

fn free_insert(free: &mut Vec<(usize, usize)>, offset: usize, size: usize) {
    let pos = free.partition_point(|&(o, _)| o < offset);
    free.insert(pos, (offset, size));
    // Coalesce with the next hole, then with the previous one.
    if pos + 1 < free.len() && free[pos].0 + free[pos].1 == free[pos + 1].0 {
        free[pos].1 += free[pos + 1].1;
        free.remove(pos + 1);
    }
    if pos > 0 && free[pos - 1].0 + free[pos - 1].1 == free[pos].0 {
        free[pos - 1].1 += free[pos].1;
        free.remove(pos);
    }
}

/// Top-level tensors read by `ck`'s result and tile expressions.
fn read_refs(program: &Program, ck: &CanonKernel) -> Result<HashSet<TensorRef>, CompileError> {
    let b = &program.module.builder;
    let mut out = HashSet::new();
    let mut visited = HashSet::new();
    let roots = ck
        .results
        .iter()
        .chain(ck.inner_lets.iter().map(|il| &il.result));
    for r in roots {
        let scalars: &[NodeId] = match r {
            ResultExpr::Scalar(n) => std::slice::from_ref(n),
            ResultExpr::Pack(elems) => elems,
        };
        for &n in scalars {
            collect_reads(b, program, ck, n, &mut visited, &mut out)?;
        }
    }
    Ok(out)
}

fn collect_reads(
    b: &IRBuilder,
    program: &Program,
    ck: &CanonKernel,
    id: NodeId,
    visited: &mut HashSet<NodeId>,
    out: &mut HashSet<TensorRef>,
) -> Result<(), CompileError> {
    if !visited.insert(id) {
        return Ok(());
    }
    match b.node(id).clone() {
        Node::Index { tensor, indices } => {
            // Inner-let tiles live in shared memory, not the arena.
            let is_tile = matches!(
                b.node(tensor),
                Node::Var(v) if ck.inner_lets.iter().any(|il| il.var == *v)
            );
            if !is_tile {
                out.insert(resolve_tensor_ref(b, &program.env, tensor)?);
            }
            for &ix in &indices {
                collect_reads(b, program, ck, ix, visited, out)?;
            }
        }
        // A scalar module input (declared with empty shape).
        Node::Input(k) => {
            out.insert(TensorRef::Input(k));
        }
        Node::Var(v) => {
            if let Some(&n) = ck.inline_lets.get(&v) {
                collect_reads(b, program, ck, n, visited, out)?;
            } else if let Some(CanonValue::Scalar(n)) = program.env.get(&v) {
                collect_reads(b, program, ck, *n, visited, out)?;
            }
        }
        Node::Bin(_, x, y) => {
            collect_reads(b, program, ck, x, visited, out)?;
            collect_reads(b, program, ck, y, visited, out)?;
        }
        Node::Select {
            cond,
            then_val,
            else_val,
        } => {
            collect_reads(b, program, ck, cond, visited, out)?;
            collect_reads(b, program, ck, then_val, visited, out)?;
            collect_reads(b, program, ck, else_val, visited, out)?;
        }
        Node::Let { value, body, .. } => {
            collect_reads(b, program, ck, value, visited, out)?;
            collect_reads(b, program, ck, body, visited, out)?;
        }
        Node::Reduce { body, .. } => {
            collect_reads(b, program, ck, body, visited, out)?;
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{IRBuilder, ScalarType},
        passes::{canonicalize, type_infer},
    };

    /// In a four-kernel chain `t1 -> t2 -> t3 -> out`, `t1` dies once `t2`
    /// is produced, so `t3` reuses its offset and the arena holds two slots.
    #[test]
    fn chain_reuses_dead_scratch() {
        let n = 64usize; // 64 * 4 bytes = one 256-byte slot each
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![n]);
        let t1 = b.compute(n, |b, i| {
            let v = b.index(x, &[i]);
            b.mul(v, v)
        });
        let body = b.bind(t1, |b, t1| {
            let t2 = b.compute(n, |b, i| {
                let v = b.index(t1, &[i]);
                b.add(v, v)
            });
            b.bind(t2, |b, t2| {
                let t3 = b.compute(n, |b, i| {
                    let v = b.index(t2, &[i]);
                    b.mul(v, v)
                });
                b.bind(t3, |b, t3| {
                    b.compute(n, |b, i| {
                        let v = b.index(t3, &[i]);
                        b.add(v, v)
                    })
                })
            })
        });
        let module = b.finish("chain", body);

        let types = type_infer(&module).unwrap();
        let program = canonicalize(module, types).unwrap();
        let plan = plan_global_scratch(&program).unwrap();

        let off = |let_id| plan.offsets[&TensorRef::Let { let_id, out_idx: 0 }];
        assert_eq!(off(0), 0);
        assert_eq!(off(1), 256);
        assert_eq!(off(2), 0);
        // The module output is not scratch.
        assert!(!plan.offsets.contains_key(&TensorRef::Let {
            let_id: 3,
            out_idx: 0
        }));
        assert_eq!(plan.total_bytes, 512);
    }
}
