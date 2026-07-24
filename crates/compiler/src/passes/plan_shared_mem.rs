//! Shared-memory placement over [`KernelProgram`]s.

use std::collections::HashMap;

use crate::kernel_ir::{BufId, BufferKind, Kernel, KernelProgram, SSANode, SSAOpCode};

/// Liveness-based placement of shared buffers within each kernel's shared-
/// memory allocation: buffers whose live ranges do not overlap share the
/// same byte offset, so the peak footprint equals the maximum concurrently-
/// live shared bytes.
pub struct SharedMemPlan {
    /// Byte offset of each shared buffer within its kernel's allocation.
    pub offsets: HashMap<BufId, usize>,
    /// Peak concurrent shared bytes of each kernel, indexed like
    /// [`KernelProgram::kernels`].
    pub per_kernel: Vec<usize>,
}

/// `(buf, alloc_pos, last_use_pos, size_bytes, align_bytes)` intervals
/// sorted by `alloc_pos`.
type Interval = (BufId, usize, usize, usize, usize);

/// Records the program-order range in which each shared buffer's memory is
/// in use: an interval starts at its first write (`Par` store or
/// `ConvertLayout` dst) and extends to its last read or write; the `Alloc`
/// itself is only a pointer declaration, so buffers whose writes are
/// separated by another buffer's whole lifetime can share the same slot.
/// Accesses inside a loop are pinned to the loop's end so the buffer
/// survives every iteration.
fn compute_liveness(p: &KernelProgram, k: &Kernel) -> Vec<Interval> {
    fn note(
        p: &KernelProgram,
        buf: BufId,
        at: usize,
        write: bool,
        ranges: &mut HashMap<BufId, (Option<usize>, usize)>,
    ) {
        if p.buffer(buf).kind != BufferKind::Shared {
            return;
        }
        let r = ranges.entry(buf).or_insert((None, at));
        if write && r.0.is_none() {
            r.0 = Some(at);
        }
        r.1 = r.1.max(at);
    }
    fn walk(
        p: &KernelProgram,
        k: &Kernel,
        stmts: &[SSANode],
        pos: &mut usize,
        ranges: &mut HashMap<BufId, (Option<usize>, usize)>,
    ) {
        for &sid in stmts {
            let op = k.op(sid);
            let at = *pos;
            *pos += 1;
            match &op.opcode {
                SSAOpCode::Alloc { .. } | SSAOpCode::Sync => {}
                SSAOpCode::ConvertLayout { dst, src, .. } => {
                    note(p, *dst, at, true, ranges);
                    note(p, *src, at, false, ranges);
                }
                SSAOpCode::Par { reads, writes, .. } => {
                    for a in reads.iter() {
                        note(p, a.buf, at, false, ranges);
                    }
                    for a in writes.iter() {
                        note(p, a.buf, at, true, ranges);
                    }
                }
                SSAOpCode::Loop { .. } => {
                    let start = *pos;
                    walk(p, k, &op.block.body, pos, ranges);
                    let end = *pos - 1;
                    for r in ranges.values_mut() {
                        if r.1 >= start {
                            r.1 = r.1.max(end);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut ranges: HashMap<BufId, (Option<usize>, usize)> = HashMap::new();
    let mut pos = 0usize;
    walk(p, k, &k.grid.block.body, &mut pos, &mut ranges);

    let mut intervals: Vec<Interval> = ranges
        .into_iter()
        .filter_map(|(buf, (start, end))| {
            start.map(|s| {
                let decl = p.buffer(buf);
                let elem_bytes = decl.elem.size_bytes();
                (buf, s, end, decl.phys_len() * elem_bytes, elem_bytes)
            })
        })
        .collect();
    intervals.sort_by_key(|&(_, start, _, _, _)| start);
    intervals
}

/// Assigns each buffer the smallest byte offset such that its live range
/// does not overlap any other buffer currently occupying that region —
/// standard first-fit interval graph coloring for offline linear scan.
/// Each buffer's chosen offset is rounded up to a multiple of its element
/// size, so `FpExt` (16-byte) buffers stay 16-byte aligned for LDS.128.
fn assign_offsets(intervals: &[Interval]) -> (HashMap<BufId, usize>, usize) {
    let mut offsets: HashMap<BufId, usize> = HashMap::new();
    // Active buffers as (offset, size, end_pos), sorted by offset.
    let mut active: Vec<(usize, usize, usize)> = Vec::new();
    let mut peak = 0usize;
    for &(buf, start, end, size, align) in intervals {
        active.retain(|&(_, _, e)| e >= start);
        active.sort_by_key(|&(o, _, _)| o);
        let mut cursor = 0usize;
        let mut chosen = None;
        for &(o, sz, _) in &active {
            let aligned = cursor.next_multiple_of(align);
            if o >= aligned + size {
                chosen = Some(aligned);
                break;
            }
            cursor = cursor.max(o + sz);
        }
        let offset = chosen.unwrap_or_else(|| cursor.next_multiple_of(align));
        offsets.insert(buf, offset);
        active.push((offset, size, end));
        peak = peak.max(offset + size);
    }
    (offsets, peak)
}

pub fn plan_shared_mem(p: &KernelProgram) -> SharedMemPlan {
    let mut offsets = HashMap::new();
    let mut per_kernel = Vec::with_capacity(p.kernels.len());
    for k in &p.kernels {
        let intervals = compute_liveness(p, k);
        let (kern_offsets, peak) = assign_offsets(&intervals);
        offsets.extend(kern_offsets);
        per_kernel.push(peak);
    }
    SharedMemPlan {
        offsets,
        per_kernel,
    }
}
