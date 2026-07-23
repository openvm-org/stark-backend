//! CUDA C++ code generation from [`KernelProgram`].
//!
//! The generated translation unit is self-contained and exports the C
//! interface from `design.md`:
//!
//! ```c
//! Prog* make_module();
//! void destroy_module(Prog*);
//! uint64_t scratch_size(Prog*);
//! uint64_t num_outputs(Prog*);   uint64_t output_size(Prog*, uint64_t);
//! uint64_t num_inputs(Prog*);    uint64_t input_size(Prog*, uint64_t);
//! void set_input(Prog*, uint64_t, void*);
//! void set_output(Prog*, uint64_t, void*);
//! void set_scratch_buf(Prog*, void*);
//! cudaError_t run(Prog*, cudaStream_t);
//! ```
//!
//! Every SSA value is a `uint32_t` named `v{n}`; buffers are `b{n}` (global
//! pointers and shared arrays) or `r{n}` (per-thread register arrays, one
//! slot per sequential step of the accessing pars). Barriers are `Sync` ops
//! placed by `passes::insert_sync`; codegen emits a `__syncthreads()` for
//! each.
//!
//! BabyBear arithmetic operates on the canonical `u32` representation; the
//! modulus is a compile-time constant so nvcc strength-reduces the `%`.

use std::{collections::HashMap, fmt::Write};

use crate::{
    ir::{BinOp, ScalarType, VarId},
    kernel_ir::{
        classify_convert, Access, BufId, BufferKind, ConvertKind, IndexMap, Kernel, KernelProgram,
        LinearLayout, ParAttr, SSABlock, SSANode, SSAOpCode, SSARes,
    },
    passes::plan_shared_mem::{plan_shared_mem, SharedMemPlan},
    quast::{CStrEmitter, Quast},
    CompileError,
};

/// Per-kernel names for register buffers: [`assign_register_colors`] maps
/// each register `BufId` to a color and codegen emits arrays named after
/// the color, so buffers with disjoint live ranges share a single array
/// and ptxas can reuse the underlying registers.
type RegNames = HashMap<BufId, usize>;

fn reg_name(names: &RegNames, buf: BufId) -> String {
    format!("r{}", names.get(&buf).copied().unwrap_or(buf.0 as usize))
}

/// First-fit interval graph coloring for register buffers: two buffers
/// share a color iff their `[first_write, last_use]` ranges are disjoint.
/// The resulting number of colors bounds the concurrently-live register
/// arrays, which is what ptxas turns into physical registers.
fn assign_register_colors(p: &KernelProgram, k: &Kernel) -> (RegNames, HashMap<usize, usize>) {
    fn note(
        p: &KernelProgram,
        buf: BufId,
        at: usize,
        write: bool,
        ranges: &mut HashMap<BufId, (Option<usize>, usize)>,
    ) {
        if p.buffer(buf).kind != BufferKind::Register {
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

    let mut items: Vec<(BufId, usize, usize)> = ranges
        .into_iter()
        .filter_map(|(buf, (start, end))| start.map(|s| (buf, s, end)))
        .collect();
    items.sort_by_key(|&(_, start, _)| start);

    // Each color tracks (last_end, max_size); assign the smallest color
    // whose last interval ends before the incoming buffer's start.
    let mut colors: Vec<(usize, usize)> = Vec::new();
    let mut names: RegNames = HashMap::new();
    for (buf, start, end) in items {
        let size = register_slots(p, k, buf);
        let color = colors
            .iter()
            .position(|&(last_end, _)| last_end < start)
            .unwrap_or_else(|| {
                colors.push((0, 0));
                colors.len() - 1
            });
        colors[color].0 = end;
        colors[color].1 = colors[color].1.max(size);
        names.insert(buf, color);
    }
    let sizes: HashMap<usize, usize> = colors
        .into_iter()
        .enumerate()
        .map(|(c, (_, sz))| (c, sz))
        .collect();
    (names, sizes)
}

const PRELUDE: &str = r#"// Auto-generated by crypto-compiler. Do not edit.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#define BB_P 2013265921u

static __device__ __forceinline__ uint32_t bb_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}
static __device__ __forceinline__ uint32_t bb_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + (BB_P - b);
}
static __device__ __forceinline__ uint32_t bb_mul(uint32_t a, uint32_t b) {
    return (uint32_t)(((unsigned long long)a * (unsigned long long)b) % BB_P);
}
"#;

/// Generates the CUDA C++ translation unit for a [`KernelProgram`].
pub fn codegen(p: &KernelProgram) -> Result<String, CompileError> {
    let plan = plan_shared_mem(p);
    let mut s = String::new();
    s.push_str(PRELUDE);
    s.push('\n');

    for (ki, k) in p.kernels.iter().enumerate() {
        gen_kernel(&mut s, p, k, &plan, ki)?;
        s.push('\n');
    }
    gen_host(&mut s, p);
    Ok(s)
}

fn val(v: SSARes) -> String {
    format!("v{}", v.0)
}

fn gen_kernel(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    plan: &SharedMemPlan,
    ki: usize,
) -> Result<(), CompileError> {
    let params = k
        .params
        .iter()
        .map(|&(buf, writable)| {
            let cq = if writable { "" } else { "const " };
            format!("{cq}uint32_t* __restrict__ b{}", buf.0)
        })
        .collect::<Vec<_>>()
        .join(", ");
    // The launch-bounds hint lets ptxas budget registers against the
    // actual thread count so a fixed 512-thread block doesn't blow past
    // the 65,536-register per-SM cap.
    writeln!(
        s,
        "__global__ void __launch_bounds__({}) {}({params}) {{",
        k.block, k.name
    )
    .unwrap();
    // Kernel-wide shared pool; each shared `Alloc` becomes a pointer into
    // it at the buffer's planned byte offset, so buffers with disjoint live
    // ranges share the same memory.
    let shared_bytes = plan.per_kernel[ki];
    if shared_bytes > 0 {
        writeln!(
            s,
            "    __shared__ uint32_t _sh_pool[{}];",
            shared_bytes.div_ceil(4)
        )
        .unwrap();
    }
    // Per-color register arrays: buffers with disjoint live ranges share
    // an array name so ptxas reuses their registers instead of holding
    // them all live at once.
    let (reg_names, color_sizes) = assign_register_colors(p, k);
    let mut colors: Vec<usize> = color_sizes.keys().copied().collect();
    colors.sort();
    for c in colors {
        writeln!(s, "    uint32_t r{c}[{}];", color_sizes[&c]).unwrap();
    }
    if kernel_uses(k, k.grid_var()) {
        writeln!(s, "    const uint32_t {} = blockIdx.x;", val(k.grid_var())).unwrap();
    }
    gen_stmts(s, p, k, plan, &reg_names, &k.grid.block.body, 1)?;
    writeln!(s, "}}").unwrap();
    Ok(())
}

/// Whether the kernel references `res` in any op or access index.
fn kernel_uses(k: &Kernel, res: SSARes) -> bool {
    let sym = VarId(res.0);
    k.ops().iter().any(|op| {
        op.operands.contains(&res)
            || match &op.opcode {
                SSAOpCode::Par { reads, writes, .. } => {
                    reads.iter().chain(writes).any(|a| match &a.index {
                        IndexMap::Affine { expr, .. } => quast_uses(expr, sym),
                        IndexMap::Linear(_) => false,
                    })
                }
                _ => false,
            }
    })
}

fn quast_uses(q: &Quast, v: VarId) -> bool {
    match q {
        Quast::Sym(s) => *s == v,
        Quast::Const(_) => false,
        Quast::Add(a, b) => quast_uses(a, v) || quast_uses(b, v),
        Quast::Mul(a, _) | Quast::FloorDiv(a, _) | Quast::Neg(a) => quast_uses(a, v),
    }
}

/// Emits a statement-level block body: `Alloc`, `Sync`, sequential `Loop`
/// and `Par` ops.
fn gen_stmts(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    plan: &SharedMemPlan,
    reg_names: &RegNames,
    stmts: &[SSANode],
    depth: usize,
) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    for &sid in stmts {
        let op = k.op(sid);
        match &op.opcode {
            SSAOpCode::Alloc { buf } => {
                let decl = p.buffer(*buf);
                match decl.kind {
                    BufferKind::Shared => {
                        // Unused shared buffers still emit a pointer for
                        // symmetry; they're pinned to offset 0 since their
                        // memory is never touched.
                        let offset = plan.offsets.get(buf).copied().unwrap_or(0);
                        writeln!(s, "{pad}uint32_t *b{} = &_sh_pool[{}u];", buf.0, offset / 4)
                            .unwrap();
                    }
                    // Register arrays are pre-declared at the kernel top.
                    BufferKind::Register => {}
                    _ => {
                        return Err(CompileError::Codegen(format!(
                            "buffer {} with kind {:?} cannot be declared in a kernel",
                            decl.name, decl.kind
                        )))
                    }
                }
            }
            SSAOpCode::Sync => {
                writeln!(s, "{pad}__syncthreads();").unwrap();
            }
            SSAOpCode::ConvertLayout { dst, src, map } => {
                gen_convert(s, p, k, reg_names, *dst, *src, map, depth)?;
            }
            SSAOpCode::Loop { bound } => {
                let v = val(op.block.operands[0]);
                writeln!(s, "{pad}for (uint32_t {v} = 0u; {v} < {bound}u; ++{v}) {{").unwrap();
                gen_stmts(s, p, k, plan, reg_names, &op.block.body, depth + 1)?;
                writeln!(s, "{pad}}}").unwrap();
            }
            SSAOpCode::Par {
                bound,
                spans_grid,
                attr,
                reads,
                writes,
            } => {
                gen_par(
                    s,
                    p,
                    k,
                    reg_names,
                    *bound,
                    *spans_grid,
                    attr,
                    reads,
                    writes,
                    &op.block,
                    depth,
                )?;
            }
            other => unreachable!("scalar op {other:?} at statement level"),
        }
    }
    Ok(())
}

/// Register slots a buffer needs per thread. A laid-out register buffer
/// (promoted tile or conversion view) spreads its power-of-two domain over
/// the block; an accumulator (no layout) needs one slot per sequential step
/// of the pars accessing it.
fn register_slots(p: &KernelProgram, k: &Kernel, buf: BufId) -> usize {
    let decl = p.buffer(buf);
    if decl.layout.is_some() {
        return decl.len().div_ceil(k.block);
    }
    k.ops()
        .iter()
        .filter_map(|op| match &op.opcode {
            SSAOpCode::Par {
                attr,
                reads,
                writes,
                ..
            } if reads.iter().chain(writes).any(|a| a.buf == buf) => Some(
                attr.as_ref()
                    .expect("layout_infer must run before codegen")
                    .seq_size,
            ),
            _ => None,
        })
        .max()
        .unwrap_or(1)
}

#[allow(clippy::too_many_arguments)]
fn gen_par(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    reg_names: &RegNames,
    bound: usize,
    spans_grid: bool,
    attr: &Option<ParAttr>,
    reads: &[Access],
    writes: &[Access],
    block: &SSABlock,
    depth: usize,
) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    let attr = attr.as_ref().expect("layout_infer must run before codegen");
    let vid = block.operands[0];
    let v = val(vid);
    let uses_reg = reads
        .iter()
        .chain(writes)
        .any(|a| p.buffer(a.buf).kind == BufferKind::Register);

    if spans_grid {
        // The grid covers the whole domain: one point per thread.
        writeln!(
            s,
            "{pad}const uint32_t {v} = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        let guard = k.grid.bound * k.block > bound;
        if guard {
            writeln!(s, "{pad}if ({v} < {bound}u) {{").unwrap();
        }
        let d = depth + usize::from(guard);
        gen_par_body(s, p, k, reg_names, attr, reads, writes, block, "0u", d)?;
        if guard {
            writeln!(s, "{pad}}}").unwrap();
        }
    } else if attr.layout.is_identity() {
        // Identity layout is the strided factorization `i = s * blockDim +
        // t`, realized as a strided loop whose condition doubles as the
        // bounds guard. The `_s` counter tracks the register slot.
        if uses_reg {
            writeln!(
                s,
                "{pad}for (uint32_t {v} = threadIdx.x, {v}_s = 0u; {v} < {bound}u; \
                 {v} += blockDim.x, ++{v}_s) {{"
            )
            .unwrap();
        } else {
            writeln!(
                s,
                "{pad}for (uint32_t {v} = threadIdx.x; {v} < {bound}u; {v} += blockDim.x) {{"
            )
            .unwrap();
        }
        gen_par_body(
            s,
            p,
            k,
            reg_names,
            attr,
            reads,
            writes,
            block,
            &format!("{v}_s"),
            depth + 1,
        )?;
        writeln!(s, "{pad}}}").unwrap();
    } else {
        let seq = attr.seq_size;
        let phys = format!("({v}_s * blockDim.x + threadIdx.x)");
        writeln!(
            s,
            "{pad}for (uint32_t {v}_s = 0u; {v}_s < {seq}u; ++{v}_s) {{"
        )
        .unwrap();
        writeln!(
            s,
            "{pad}    const uint32_t {v} = {};",
            ll_apply_str(&attr.layout, &phys)
        )
        .unwrap();
        writeln!(s, "{pad}    if ({v} >= {bound}u) continue;").unwrap();
        gen_par_body(
            s,
            p,
            k,
            reg_names,
            attr,
            reads,
            writes,
            block,
            &format!("{v}_s"),
            depth + 1,
        )?;
        writeln!(s, "{pad}}}").unwrap();
    }
    Ok(())
}

/// The body of one par point: bind the loaded block operands, run the SSA
/// ops, store the yields.
#[allow(clippy::too_many_arguments)]
fn gen_par_body(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    reg_names: &RegNames,
    attr: &ParAttr,
    reads: &[Access],
    writes: &[Access],
    block: &SSABlock,
    slot: &str,
    depth: usize,
) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    let vid = block.operands[0];
    for (i, access) in reads.iter().enumerate() {
        let operand = block.operands[1 + i];
        let src = access_str(p, access, reg_names, attr, vid, slot)?;
        writeln!(s, "{pad}const uint32_t {} = {src};", val(operand)).unwrap();
    }
    gen_ops(s, k, &block.body, depth)?;
    for (i, access) in writes.iter().enumerate() {
        let dst = access_str(p, access, reg_names, attr, vid, slot)?;
        writeln!(s, "{pad}{dst} = {};", val(block.yields[i])).unwrap();
    }
    Ok(())
}

/// Emits a [`SSAOpCode::ConvertLayout`]: `dst[i] = src[map(i)]` over `dst`'s
/// logical domain. Register-to-register conversions reduce to the map
/// `C = f_src^-1 ∘ map ∘ f_dst` from dst physical to src physical index and
/// become slot permutations or warp shuffles per [`classify_convert`];
/// register-to-shared stages the registers out through shared memory.
#[allow(clippy::too_many_arguments)]
fn gen_convert(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    reg_names: &RegNames,
    dst: BufId,
    src: BufId,
    map: &LinearLayout,
    depth: usize,
) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    let dd = p.buffer(dst);
    let sd = p.buffer(src);
    let kb = map.bases.len();
    let n = 1usize << kb;
    let id = || LinearLayout::identity(kb);
    let (dst_reg, src_reg) = (reg_name(reg_names, dst), reg_name(reg_names, src));
    match (dd.kind, sd.kind) {
        (BufferKind::Register, BufferKind::Register) => {
            let ld = dd.layout.clone().unwrap_or_else(id);
            let f = sd.layout.clone().unwrap_or_else(id);
            let f_inv = f.inverse().ok_or_else(|| {
                CompileError::Codegen(format!("register buffer {} has a singular layout", sd.name))
            })?;
            let c = f_inv.compose(&map.compose(&ld));
            match classify_convert(&c, k.block) {
                ConvertKind::Copy => {
                    for i in 0..n.div_ceil(k.block) {
                        writeln!(s, "{pad}{dst_reg}[{i}] = {src_reg}[{i}];").unwrap();
                    }
                }
                ConvertKind::Slot => {
                    let tb = kb.min(k.block.trailing_zeros() as usize);
                    for i in 0..(1usize << (kb - tb)) {
                        let from = c.apply((i as u64) << tb) >> tb;
                        writeln!(s, "{pad}{dst_reg}[{i}] = {src_reg}[{from}];").unwrap();
                    }
                }
                ConvertKind::Shuffle => {
                    gen_shuffle(s, &c, &dst_reg, &src_reg, dst.0, kb, k.block, depth)
                }
                ConvertKind::Bounce => {
                    return Err(CompileError::Codegen(format!(
                        "register conversion {} <- {} needs a shared-memory bounce",
                        dd.name, sd.name
                    )))
                }
            }
        }
        (BufferKind::Shared, BufferKind::Register) => {
            // Iterate src physical indices: slot `s` of thread `t` (physical
            // `x = s * blockDim + t`) holds src's logical `f(x)`, which lands
            // at dst logical `map^-1(f(x))`, i.e. address `Ld(map^-1(f(x)))`.
            let ld = dd.layout.clone().unwrap_or_else(id);
            let f = sd.layout.clone().unwrap_or_else(id);
            let map_inv = map.inverse().ok_or_else(|| {
                CompileError::Codegen(format!(
                    "convert_layout {} <- {}: map is singular",
                    dd.name, sd.name
                ))
            })?;
            let g = ld.compose(&map_inv.compose(&f));
            let x = format!("_cv{}_x", dst.0);
            let sl = format!("_cv{}_s", dst.0);
            writeln!(
                s,
                "{pad}for (uint32_t {x} = threadIdx.x, {sl} = 0u; {x} < {n}u; \
                 {x} += blockDim.x, ++{sl}) {{"
            )
            .unwrap();
            writeln!(
                s,
                "{pad}    b{}[{}] = {src_reg}[{sl}];",
                dst.0,
                ll_apply_str(&g, &x),
            )
            .unwrap();
            writeln!(s, "{pad}}}").unwrap();
        }
        _ => {
            return Err(CompileError::Codegen(format!(
                "unsupported convert_layout between buffer kinds {:?} <- {:?}",
                dd.kind, sd.kind
            )))
        }
    }
    Ok(())
}

/// One `__shfl_sync` per destination slot. `c` maps dst physical to src
/// physical over `kb` bits; [`classify_convert`] guaranteed its warp bits
/// are fixed and its lane-to-lane block `M` is invertible.
///
/// **Sender-slot handling.** At dst slot `s'` this thread needs src physical
/// `C(s' << tb ^ tid)`, living in lane `(C(s' << tb) ^ C(tid)) & 31`. The
/// sender's own slot to provide, `(C(s' << tb ^ tid)) >> tb`, is a
/// compile-time constant iff no lane- or warp-input base of `C` has slot
/// output bits — i.e. no `C.bases[i] & slot_mask` for `i < tb` — since
/// then `C(x) >> tb` collapses to `C(x & slot_mask) >> tb` and the tid
/// contribution vanishes. That fast path — the whole butterfly-partner
/// family, including every stage of the register NTT — emits one
/// `__shfl_sync` per slot with a constant source-slot index.
///
/// **General path.** When the sender-slot does depend on tid (e.g. a
/// laned-in transpose), we synthesize it: as a sender the thread computes
/// which receiver lane needs it, `l = M^-1(lane ^ (C(s' << tb) ^ C(warp)) &
/// 31)`, and offers `(C(s' << tb) ^ C(warp) ^ C(l)) >> tb`. That slot index
/// varies per lane, so a nested `?:` chain over the `slots`-many
/// possibilities keeps every read at a compile-time-constant index into the
/// register array.
#[allow(clippy::too_many_arguments)]
fn gen_shuffle(
    s: &mut String,
    c: &LinearLayout,
    dst_reg: &str,
    src_reg: &str,
    dst_id: u32,
    kb: usize,
    block: usize,
    depth: usize,
) {
    let mut pad = "    ".repeat(depth);
    let tb = kb.min(block.trailing_zeros() as usize);
    let slots = 1usize << (kb - tb);
    let n = 1usize << kb;
    let pre = format!("_cv{dst_id}");
    // `classify_convert` guarantees kb >= 5, so `n` covers whole warps and
    // the full shuffle mask stays valid under the guard.
    let guard = n < block;
    if guard {
        writeln!(s, "{pad}if (threadIdx.x < {n}u) {{").unwrap();
        pad.push_str("    ");
    }
    let c_lin = LinearLayout {
        bases: c.bases.clone(),
        offset: 0,
    };
    // Sender-slot constness: no thread-input bit (i < tb) may cross into
    // slot output (bit ≥ tb). Warp-input bases are pinned to their input by
    // classify_convert's `warp_fixed`, so this is really just a check on
    // the lane-input rows of `C`.
    let slot_mask: u64 = if tb >= u64::BITS as usize {
        0
    } else {
        !((1u64 << tb) - 1)
    };
    let const_src_slot = c.bases[..tb].iter().all(|&b| b & slot_mask == 0);
    writeln!(
        s,
        "{pad}const uint32_t {pre}_ct = {};",
        ll_apply_str(&c_lin, "threadIdx.x")
    )
    .unwrap();
    // The lane-block and its inverse only matter for the sender-slot
    // ternary; skip them entirely on the fast path.
    let c_lane = LinearLayout {
        bases: c.bases[..5.min(kb)].to_vec(),
        offset: 0,
    };
    let m_inv = if slots > 1 && !const_src_slot {
        writeln!(
            s,
            "{pad}const uint32_t {pre}_cw = {pre}_ct ^ {};",
            ll_apply_str(&c_lane, "(threadIdx.x & 31u)")
        )
        .unwrap();
        Some(
            LinearLayout {
                bases: c_lane.bases.iter().map(|&b| b & 31).collect(),
                offset: 0,
            }
            .inverse()
            .expect("classify_convert checked the lane block"),
        )
    } else {
        None
    };
    for sp in 0..slots {
        let cs = c.apply((sp as u64) << tb);
        let val = if slots == 1 || const_src_slot {
            // `cs` has no bits below tb from `sp << tb`, and no bits at or
            // above tb from any lane/warp input under the const-slot
            // condition, so the sender slot is exactly `cs >> tb`.
            format!("{src_reg}[{}]", cs >> tb)
        } else {
            let m_inv = m_inv.as_ref().unwrap();
            let l = format!("{pre}_l{sp}");
            writeln!(
                s,
                "{pad}const uint32_t {l} = {};",
                ll_apply_str(m_inv, &format!("((threadIdx.x ^ {cs}u ^ {pre}_cw) & 31u)"))
            )
            .unwrap();
            writeln!(
                s,
                "{pad}const uint32_t {pre}_s{sp} = ({cs}u ^ {pre}_cw ^ {}) >> {tb};",
                ll_apply_str(&c_lane, &l)
            )
            .unwrap();
            let mut v = format!("{src_reg}[{}]", slots - 1);
            for q in (0..slots - 1).rev() {
                v = format!("{pre}_s{sp} == {q}u ? {src_reg}[{q}] : ({v})");
            }
            v
        };
        writeln!(
            s,
            "{pad}{dst_reg}[{sp}] = __shfl_sync(0xffffffffu, {val}, ({cs}u ^ {pre}_ct) & 31u);"
        )
        .unwrap();
    }
    if guard {
        pad.truncate(pad.len() - 4);
        writeln!(s, "{pad}}}").unwrap();
    }
}

fn gen_ops(s: &mut String, k: &Kernel, body: &[SSANode], depth: usize) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    for &nid in body {
        let op = k.op(nid);
        match &op.opcode {
            SSAOpCode::ConstU32(c) | SSAOpCode::ConstField(c) => {
                writeln!(s, "{pad}const uint32_t {} = {c}u;", val(op.results[0])).unwrap();
            }
            SSAOpCode::Bin(bop, ty) => {
                let a = val(op.operands[0]);
                let b = val(op.operands[1]);
                writeln!(
                    s,
                    "{pad}const uint32_t {} = {};",
                    val(op.results[0]),
                    bin_str(*bop, *ty, &a, &b)
                )
                .unwrap();
            }
            SSAOpCode::Select => {
                writeln!(
                    s,
                    "{pad}const uint32_t {} = {} ? {} : {};",
                    val(op.results[0]),
                    val(op.operands[0]),
                    val(op.operands[1]),
                    val(op.operands[2])
                )
                .unwrap();
            }
            SSAOpCode::Loop { bound } => {
                // The results double as the mutable loop-carried slots.
                for (i, res) in op.results.iter().enumerate() {
                    writeln!(s, "{pad}uint32_t {} = {};", val(*res), val(op.operands[i])).unwrap();
                }
                let iv = val(op.block.operands[0]);
                writeln!(
                    s,
                    "{pad}for (uint32_t {iv} = 0u; {iv} < {bound}u; ++{iv}) {{"
                )
                .unwrap();
                for (i, carried) in op.block.operands[1..].iter().enumerate() {
                    writeln!(
                        s,
                        "{pad}    const uint32_t {} = {};",
                        val(*carried),
                        val(op.results[i])
                    )
                    .unwrap();
                }
                gen_ops(s, k, &op.block.body, depth + 1)?;
                for (i, y) in op.block.yields.iter().enumerate() {
                    writeln!(s, "{pad}    {} = {};", val(op.results[i]), val(*y)).unwrap();
                }
                writeln!(s, "{pad}}}").unwrap();
            }
            other @ (SSAOpCode::Par { .. }
            | SSAOpCode::Alloc { .. }
            | SSAOpCode::Sync
            | SSAOpCode::ConvertLayout { .. }) => {
                unreachable!("statement-level op {other:?} inside a par block")
            }
        }
    }
    Ok(())
}

fn bin_str(op: BinOp, ty: ScalarType, a: &str, b: &str) -> String {
    if ty == ScalarType::BabyBear {
        match op {
            BinOp::Add => format!("bb_add({a}, {b})"),
            BinOp::Sub => format!("bb_sub({a}, {b})"),
            BinOp::Mul => format!("bb_mul({a}, {b})"),
            _ => unreachable!("op {op:?} is not defined on BabyBear"),
        }
    } else {
        let c_op = match op {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Eq => "==",
        };
        match op {
            BinOp::Lt | BinOp::Le | BinOp::Eq => format!("(uint32_t)({a} {c_op} {b})"),
            _ => format!("{a} {c_op} {b}"),
        }
    }
}

/// Whether two XOR-affine maps agree as functions, treating missing high
/// bases as zero (inputs past either domain are masked by bounds guards).
fn maps_agree(a: &LinearLayout, b: &LinearLayout) -> bool {
    a.offset == b.offset
        && (0..a.bases.len().max(b.bases.len()))
            .all(|i| a.bases.get(i).copied().unwrap_or(0) == b.bases.get(i).copied().unwrap_or(0))
}

/// C lvalue for one access: the buffer element at the access's index. The
/// buffer's linear layout (logical -> physical) is applied when non-trivial.
fn access_str(
    p: &KernelProgram,
    access: &Access,
    reg_names: &RegNames,
    attr: &ParAttr,
    vid: SSARes,
    slot: &str,
) -> Result<String, CompileError> {
    let decl = p.buffer(access.buf);
    if decl.kind == BufferKind::Register {
        // Each par point owns one element, held in the thread's slot for the
        // point's sequential step: accumulators are only touched at the
        // par's own index; laid-out register buffers only through an
        // effective map `g ∘ f_par` equal to their layout (so at physical
        // index x the access lands on the element the layout puts there).
        let own = match (&decl.layout, &access.index) {
            (None, IndexMap::Affine { expr, .. }) => *expr == Quast::sym(VarId(vid.0)),
            (Some(l), IndexMap::Linear(g)) => maps_agree(&g.compose(&attr.layout), l),
            _ => false,
        };
        if !own {
            return Err(CompileError::Codegen(format!(
                "register buffer {} accessed at an index other than the par's own",
                decl.name
            )));
        }
        return Ok(format!("{}[{slot}]", reg_name(reg_names, access.buf)));
    }
    let logical = match &access.index {
        IndexMap::Linear(ll) => ll_apply_str(ll, &val(vid)),
        IndexMap::Affine { expr, bounds } => expr.emit(bounds, &mut CStrEmitter)?,
    };
    Ok(match &decl.layout {
        Some(l) if !l.is_identity() => {
            format!(
                "b{}[{}]",
                access.buf.0,
                ll_apply_str(l, &format!("({logical})"))
            )
        }
        _ => format!("b{}[{logical}]", access.buf.0),
    })
}

/// C expression applying a [`LinearLayout`] to `x`: the XOR of the offset
/// and the bases selected by the bits of `x`. The identity map is `x`
/// itself; maximal runs of single-bit bases shifting consecutive input bits
/// by a common amount collapse to one mask-and-shift term; any other
/// non-zero base falls back to a per-bit select.
fn ll_apply_str(layout: &LinearLayout, x: &str) -> String {
    if layout.is_identity() {
        return x.to_string();
    }
    let bases = &layout.bases;
    let mut terms = Vec::new();
    let mut i = 0;
    while i < bases.len() {
        let b = bases[i];
        if b == 0 {
            i += 1;
        } else if b.is_power_of_two() {
            let mut j = i + 1;
            while j < bases.len()
                && b.checked_shl((j - i) as u32)
                    .is_some_and(|shifted| shifted == bases[j])
            {
                j += 1;
            }
            let mask = ((1u64 << (j - i)) - 1) << i;
            let masked = format!("({x} & {mask}u)");
            let shift = b.trailing_zeros() as i64 - i as i64;
            terms.push(match shift {
                0 => masked,
                s if s > 0 => format!("({masked} << {s})"),
                s => format!("({masked} >> {})", -s),
            });
            i = j;
        } else {
            terms.push(format!("((({x} >> {i}) & 1u) * {b}u)"));
            i += 1;
        }
    }
    if layout.offset != 0 {
        terms.push(format!("{}u", layout.offset));
    }
    if terms.is_empty() {
        "0u".into()
    } else {
        terms.join(" ^ ")
    }
}

/// C expression for a buffer pointer inside `run`, respecting constness.
fn buf_arg(p: &KernelProgram, buf: BufId, writable: bool) -> String {
    let cq = if writable { "" } else { "const " };
    match p.buffer(buf).kind {
        BufferKind::Input(k) => format!("({cq}uint32_t*)p->inputs[{k}]"),
        BufferKind::Output(k) => format!("({cq}uint32_t*)p->outputs[{k}]"),
        BufferKind::Scratch { offset } => {
            format!("({cq}uint32_t*)((char*)p->scratch + {offset})")
        }
        BufferKind::Shared | BufferKind::Register => {
            unreachable!("kernel-local buffers are never kernel parameters")
        }
    }
}

fn gen_host(s: &mut String, p: &KernelProgram) {
    let n_in = p.input_bufs.len();
    let n_out = p.output_bufs.len();
    let in_sizes = p
        .input_bufs
        .iter()
        .map(|&b| format!("{}ull", p.buffer(b).size_bytes()))
        .collect::<Vec<_>>()
        .join(", ");
    let out_sizes = p
        .output_bufs
        .iter()
        .map(|&b| format!("{}ull", p.buffer(b).size_bytes()))
        .collect::<Vec<_>>()
        .join(", ");

    writeln!(
        s,
        r#"struct Prog {{
    void* inputs[{in_cap}];
    void* outputs[{out_cap}];
    void* scratch;
}};

static const uint64_t kInputSizes[{in_cap}] = {{{in_sizes}}};
static const uint64_t kOutputSizes[{out_cap}] = {{{out_sizes}}};

extern "C" Prog* make_module() {{
    Prog* p = new Prog;
    std::memset(p, 0, sizeof(Prog));
    return p;
}}
extern "C" void destroy_module(Prog* p) {{ delete p; }}
extern "C" uint64_t scratch_size(Prog*) {{ return {scratch}ull; }}
extern "C" uint64_t num_outputs(Prog*) {{ return {n_out}ull; }}
extern "C" uint64_t output_size(Prog*, uint64_t i) {{ return kOutputSizes[i]; }}
extern "C" uint64_t num_inputs(Prog*) {{ return {n_in}ull; }}
extern "C" uint64_t input_size(Prog*, uint64_t i) {{ return kInputSizes[i]; }}
extern "C" void set_input(Prog* p, uint64_t i, void* ptr) {{ p->inputs[i] = ptr; }}
extern "C" void set_output(Prog* p, uint64_t i, void* ptr) {{ p->outputs[i] = ptr; }}
extern "C" void set_scratch_buf(Prog* p, void* ptr) {{ p->scratch = ptr; }}
"#,
        in_cap = n_in.max(1),
        out_cap = n_out.max(1),
        in_sizes = if n_in == 0 { "0ull".into() } else { in_sizes },
        out_sizes = if n_out == 0 { "0ull".into() } else { out_sizes },
        scratch = p.scratch_bytes,
        n_out = n_out,
        n_in = n_in,
    )
    .unwrap();

    writeln!(
        s,
        "extern \"C\" cudaError_t run(Prog* p, cudaStream_t stream) {{"
    )
    .unwrap();
    for k in &p.kernels {
        let args = k
            .params
            .iter()
            .map(|&(buf, writable)| buf_arg(p, buf, writable))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            s,
            "    {}<<<dim3({}u), dim3({}u), 0, stream>>>({args});",
            k.name, k.grid.bound, k.block
        )
        .unwrap();
        writeln!(
            s,
            "    {{ cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) return err; }}"
        )
        .unwrap();
    }
    writeln!(s, "    return cudaSuccess;\n}}").unwrap();
}
