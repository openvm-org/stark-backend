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
//! void set_param(Prog*, uint64_t, int64_t);
//! cudaError_t run(Prog*, cudaStream_t);
//! ```
//!
//! Every SSA value is a `uint32_t` named `v{n}`; buffers are `b{n}` (global
//! pointers and shared arrays) or `r{n}` (per-thread register arrays, one
//! slot per sequential step of the accessing pars). Barriers are `Sync` ops
//! placed by `passes::insert_sync`; codegen emits a `__syncthreads()` for
//! each.
//!
//! BabyBear arithmetic operates on the **Montgomery** `u32` representation:
//! every field element `x` is stored as `x * R mod P` with `R = 2^32`. Add
//! and sub are unchanged (Montgomery preserves them), but mul is a
//! Montgomery reduction (`mul.lo` + `mad.lo.cc` + `madc.hi.cc` inline PTX
//! following sppark's `mont32_t::mul`). External callers must Montgomery-
//! encode inputs and decode outputs; `ConstField`s in the IR are canonical
//! and converted at emission time. See [`to_monty`].

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fmt::Write,
};

use crate::{
    ir::{BinOp, ScalarType, SizeExpr, VarId},
    kernel_ir::{
        classify_convert, Access, BufId, BufferKind, ConvertKind, IndexMap, KBound, Kernel,
        KernelProgram, LinearLayout, ParAttr, SSABlock, SSANode, SSAOpCode, SSARes,
    },
    passes::plan_shared_mem::{plan_shared_mem, SharedMemPlan},
    quast::{CStrEmitter, Expr, Quast, SymConst},
    CompileError,
};

fn reg_name(buf: BufId) -> String {
    format!("r{}", buf.0)
}

/// Device-code C identifier for a module parameter: `p_{name}` with
/// non-alphanumeric characters mapped to `_`.
fn param_ident(name: &str) -> String {
    let ident: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("p_{ident}")
}

fn device_param(p: &KernelProgram, v: VarId) -> String {
    let (_, name) = p
        .params
        .iter()
        .find(|(pv, _)| *pv == v)
        .expect("size expression references an undeclared module parameter");
    param_ident(name)
}

/// C expression for a [`SizeExpr`], fully parenthesized; `var_sym` renders
/// `Expr::Sym` positions (kernel SSA values in access indices, host / device
/// symbols elsewhere) and `const_sym` renders `SymConst::Sym` positions
/// (module parameters). C integer division on non-negative sizes matches
/// floor division.
fn sexpr_str(
    e: &SizeExpr,
    var_sym: &dyn Fn(VarId) -> String,
    const_sym: &dyn Fn(VarId) -> String,
) -> String {
    let sc = |c: &SymConst| match c {
        SymConst::Lit(v) => format!("{v}"),
        SymConst::Sym(v) => const_sym(*v),
    };
    let rec = |e: &SizeExpr| sexpr_str(e, var_sym, const_sym);
    match e {
        Expr::Sym(v) => var_sym(*v),
        Expr::Const(c) => sc(c),
        Expr::Add(a, b) => format!("({} + {})", rec(a), rec(b)),
        Expr::Mul(a, c) => format!("({} * {})", rec(a), sc(c)),
        Expr::FloorDiv(a, c) => format!("({} / {})", rec(a), sc(c)),
        Expr::Neg(a) => format!("(-{})", rec(a)),
    }
}

/// BabyBear prime.
const BB_P: u64 = 2_013_265_921;

/// Convert a canonical BabyBear `u32` `x ∈ [0, P)` to Montgomery form
/// `x * R mod P` with `R = 2^32`. `mont(0) = 0`; `mont(1) = 0x0ffffffe`.
pub(crate) fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % BB_P) as u32
}

/// C++ type name for a [`ScalarType`] — every 32-bit type shares the
/// same underlying `uint32_t` storage; `FpExt` is a 16-byte struct that
/// nvcc will pick up as `LDG.128`/`STG.128` when reading from a
/// `FpExt*`-typed pointer.
fn c_type(ty: ScalarType) -> &'static str {
    match ty {
        ScalarType::FpExt => "FpExt",
        ScalarType::BabyBear | ScalarType::U32 | ScalarType::Bool => "uint32_t",
    }
}

/// Precomputed scalar type of every SSA value defined in a kernel:
/// constants and `Bin` results are known from their opcodes; `Select`
/// takes its then/else operand's type; `Loop` carried slots inherit
/// their initial operand's type; par block operands are the par index
/// (U32) followed by the loaded values, each typed to its access
/// buffer's element type.
type ValTypes = HashMap<SSARes, ScalarType>;

fn compute_val_types(p: &KernelProgram, k: &Kernel) -> ValTypes {
    fn walk(p: &KernelProgram, k: &Kernel, body: &[SSANode], types: &mut ValTypes) {
        for &nid in body {
            let op = k.op(nid);
            match &op.opcode {
                SSAOpCode::ConstU32(_) | SSAOpCode::ConstSym(_) => {
                    types.insert(op.results[0], ScalarType::U32);
                }
                SSAOpCode::ConstField(_) => {
                    types.insert(op.results[0], ScalarType::BabyBear);
                }
                SSAOpCode::ConstFpExt(_) | SSAOpCode::LiftFpExt => {
                    types.insert(op.results[0], ScalarType::FpExt);
                }
                SSAOpCode::Bin(bop, ty) => {
                    let result_ty = match bop {
                        BinOp::Lt | BinOp::Le | BinOp::Eq => ScalarType::Bool,
                        _ => *ty,
                    };
                    types.insert(op.results[0], result_ty);
                }
                SSAOpCode::Select { else_block } => {
                    // Both branches yield a value of the merged result
                    // type; walk each block's body first so its yield's
                    // producer is typed, then read the type off the
                    // yield.
                    walk(p, k, &op.block.body, types);
                    walk(p, k, &else_block.body, types);
                    let t = types[&op.block.yields[0]];
                    types.insert(op.results[0], t);
                }
                SSAOpCode::Loop { .. } => {
                    types.insert(op.block.operands[0], ScalarType::U32);
                    for (i, res) in op.results.iter().enumerate() {
                        let t = types[&op.operands[i]];
                        types.insert(*res, t);
                        types.insert(op.block.operands[1 + i], t);
                    }
                    walk(p, k, &op.block.body, types);
                }
                SSAOpCode::Par { reads, .. } => {
                    types.insert(op.block.operands[0], ScalarType::U32);
                    for (i, access) in reads.iter().enumerate() {
                        let t = p.buffer(access.buf).elem;
                        types.insert(op.block.operands[1 + i], t);
                    }
                    walk(p, k, &op.block.body, types);
                }
                SSAOpCode::Alloc { .. } | SSAOpCode::Sync | SSAOpCode::ConvertLayout { .. } => {}
            }
        }
    }
    let mut types = ValTypes::new();
    types.insert(k.grid_var(), ScalarType::U32);
    walk(p, k, &k.grid.block.body, &mut types);
    types
}

const PRELUDE: &str = r#"// Auto-generated by crypto-compiler. Do not edit.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// BabyBear modulus and Montgomery constants (matching sppark's bb31_t):
//   BB_P  = 0x78000001   = 2013265921 = P
//   BB_M0 = 0x77ffffff   = -P^{-1} mod 2^32
//   R     = 2^32         (Montgomery radix; not needed as a runtime constant)
// Field elements are stored as `x * R mod P` (Montgomery form). The N=32
// carry logic in mont32_t applies here since P sits in the top half of u32.
#define BB_P  2013265921u
#define BB_M0 0x77ffffffu

// Montgomery-form addition: same u32 arithmetic as canonical (a+b) mod P.
// The `s < b` guard handles the wrap-around case for N=32: a + b can
// overflow u32 before we compare against P, so a raw compare against P
// would miss the carried value.
static __device__ uint32_t bb_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    if (s < b || s >= BB_P) s -= BB_P;
    return s;
}
// Montgomery-form subtraction: same u32 arithmetic as canonical.
static __device__ uint32_t bb_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + (BB_P - b);
}
// Montgomery multiplication: given a = x*R, b = y*R (both mod P), returns
// (x*y)*R mod P via a single 32-bit reduction step matching the PTX in
// sppark's mont32_t::mul (crates/cuda-common/include/ff/mont32_t.cuh).
//
// Layout mirrors mont32_t exactly:
//   tmp = a * b            (64-bit, split into (lo, hi))
//   red = (tmp.lo * M0) mod 2^32
//   tmp += red * MOD        (32-bit carry chain: mad.lo.cc + madc.hi.cc)
//   final subtraction if tmp.hi >= MOD or the add carried out
// The final_sub uses the N=32 variant (`addc + setp` on both value and
// carry) so the wrap case is handled by the extra `@!%p sub` predicate.
// Emitting the carry-consuming `addc` + `final_sub` inside the same PTX
// scope as the `madc.hi.cc.u32` keeps ptxas from scheduling other
// carry-touching instructions between them.
static __device__ uint32_t bb_mul(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
    uint32_t lo, hi, red;
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(lo), "=r"(hi) : "r"(a), "r"(b));
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(lo), "r"(BB_M0));
    asm("{ .reg.pred %p; .reg.b32 %c;\n"
        "  mad.lo.cc.u32 %0, %2, %3, %0;\n"
        "  madc.hi.cc.u32 %1, %2, %3, %1;\n"
        "  addc.u32 %c, 0, 0;\n"
        "  setp.lt.u32 %p, %1, %3;\n"
        "  @%p setp.eq.u32 %p, %c, 0;\n"
        "  @!%p sub.u32 %1, %1, %3;\n"
        "}"
        : "+r"(lo), "+r"(hi)
        : "r"(red), "r"(BB_P));
    return hi;
#else
    // Host / analyzer fallback: same result via plain 64-bit math.
    unsigned long long t = (unsigned long long)a * (unsigned long long)b;
    uint32_t lo = (uint32_t)t, hi = (uint32_t)(t >> 32);
    uint32_t red = lo * BB_M0;
    unsigned long long u = (unsigned long long)lo + (unsigned long long)red * (unsigned long long)BB_P;
    uint32_t carry = (uint32_t)(u >> 32);
    unsigned long long v = (unsigned long long)hi + (unsigned long long)carry;
    uint32_t out_hi = (uint32_t)v;
    uint32_t out_carry = (uint32_t)(v >> 32);
    if (out_carry != 0 || out_hi >= BB_P) out_hi -= BB_P;
    return out_hi;
#endif
}

// Degree-4 binomial extension of BabyBear over `x^4 - 11`. The 16-byte
// alignment lets nvcc pick LDG.128 / STG.128 (and LDS.128 / STS.128 for
// shared) for a whole element in a single instruction.
struct __align__(16) FpExt {
    uint32_t v[4];
};
static __device__ FpExt fpext_add(FpExt a, FpExt b) {
    return FpExt{{
        bb_add(a.v[0], b.v[0]), bb_add(a.v[1], b.v[1]),
        bb_add(a.v[2], b.v[2]), bb_add(a.v[3], b.v[3]),
    }};
}
static __device__ FpExt fpext_sub(FpExt a, FpExt b) {
    return FpExt{{
        bb_sub(a.v[0], b.v[0]), bb_sub(a.v[1], b.v[1]),
        bb_sub(a.v[2], b.v[2]), bb_sub(a.v[3], b.v[3]),
    }};
}
// Karatsuba over `x^4 - 11` (schoolbook is clearer than any 3-mul trick
// at this size, and ptxas schedules it well):
// (a0..a3) * (b0..b3) = c0..c3 where
//   c0 = a0 b0 + 11*(a1 b3 + a2 b2 + a3 b1)
//   c1 = a0 b1 + a1 b0 + 11*(a2 b3 + a3 b2)
//   c2 = a0 b2 + a1 b1 + a2 b0 + 11*(a3 b3)
//   c3 = a0 b3 + a1 b2 + a2 b1 + a3 b0
// `B` is 11 in Montgomery form (`(11 << 32) mod P`), matching the
// `BETA` constant in sppark's `bb31_4_t`.
//
// `__noinline__` is deliberate: each `fpext_mul` inlines 16 `bb_mul` calls
// (each with 3 inline PTX asm blocks for the Montgomery reduction), so a
// kernel body with N `fpext_mul` sites otherwise expands into ~48N inline
// asm blocks in one function. nvcc's `cicc` middle-end does at least
// quadratic-in-asm-block work per function on liveness/reg-alloc across
// predicate-scoped asm; a kernel with N ≥ ~15 (e.g. `frac_compute_round`
// on n ≥ 128) hits a compile-time wall (measured: 3+ min to timeout).
// Keeping `fpext_mul` non-inline localizes those asm blocks to one
// function, cutting compile time by ~3× at a runtime cost of one `bl`
// instruction per FpExt multiply.
static __device__ __noinline__ FpExt fpext_mul(FpExt a, FpExt b) {
    const uint32_t B = 0x37ffffe9u;
    uint32_t high0 = bb_mul(B, bb_add(bb_add(bb_mul(a.v[1], b.v[3]),
                                              bb_mul(a.v[2], b.v[2])),
                                       bb_mul(a.v[3], b.v[1])));
    uint32_t high1 = bb_mul(B, bb_add(bb_mul(a.v[2], b.v[3]),
                                       bb_mul(a.v[3], b.v[2])));
    uint32_t high2 = bb_mul(B, bb_mul(a.v[3], b.v[3]));
    return FpExt{{
        bb_add(bb_mul(a.v[0], b.v[0]), high0),
        bb_add(bb_add(bb_mul(a.v[0], b.v[1]), bb_mul(a.v[1], b.v[0])), high1),
        bb_add(bb_add(bb_add(bb_mul(a.v[0], b.v[2]), bb_mul(a.v[1], b.v[1])),
                       bb_mul(a.v[2], b.v[0])), high2),
        bb_add(bb_add(bb_add(bb_mul(a.v[0], b.v[3]), bb_mul(a.v[1], b.v[2])),
                       bb_mul(a.v[2], b.v[1])), bb_mul(a.v[3], b.v[0])),
    }};
}
// FpExt warp shuffle: `__shfl_sync` doesn't take struct arguments, so
// we shuffle each 32-bit coefficient separately. Overloading on the
// value's type lets `gen_shuffle` emit a single `__shfl_sync(mask, val,
// srcLane)` call regardless of `T`.
static __device__ FpExt __shfl_sync(unsigned mask, FpExt val, int src_lane) {
    return FpExt{{
        __shfl_sync(mask, val.v[0], src_lane),
        __shfl_sync(mask, val.v[1], src_lane),
        __shfl_sync(mask, val.v[2], src_lane),
        __shfl_sync(mask, val.v[3], src_lane),
    }};
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
    let mut params = k
        .params
        .iter()
        .map(|&(buf, writable)| {
            let cq = if writable { "" } else { "const " };
            format!(
                "{cq}{}* __restrict__ b{}",
                c_type(p.buffer(buf).elem),
                buf.0
            )
        })
        .collect::<Vec<_>>();
    for (_, name) in &p.params {
        params.push(format!("const uint32_t {}", param_ident(name)));
    }
    let params = params.join(", ");
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
    // ranges share the same memory. The pool is 16-byte aligned so any
    // `FpExt*` alias hits `LDS.128`/`STS.128`.
    let shared_bytes = plan.per_kernel[ki];
    if shared_bytes > 0 {
        writeln!(
            s,
            "    __shared__ __align__(16) uint32_t _sh_pool[{}];",
            shared_bytes.div_ceil(4)
        )
        .unwrap();
    }
    // Every register buffer gets its own top-level array. ptxas does
    // inter-array liveness under `__launch_bounds__` and reuses physical
    // registers across arrays whose live ranges don't overlap, so we
    // don't need to hand-color them.
    for op in k.ops().iter() {
        if let SSAOpCode::Alloc { buf } = &op.opcode {
            if p.buffer(*buf).kind == BufferKind::Register {
                writeln!(
                    s,
                    "    {} r{}[{}];",
                    c_type(p.buffer(*buf).elem),
                    buf.0,
                    register_slots(p, k, *buf)
                )
                .unwrap();
            }
        }
    }
    if kernel_uses(k, k.grid_var()) {
        writeln!(s, "    const uint32_t {} = blockIdx.x;", val(k.grid_var())).unwrap();
    }
    let types = compute_val_types(p, k);
    gen_stmts(s, p, k, plan, &types, &k.grid.block.body, 1)?;
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
                        IndexMap::SExpr(e) | IndexMap::Blackbox(e) => {
                            let mut syms = std::collections::BTreeSet::new();
                            e.syms(&mut syms);
                            syms.contains(&sym)
                        }
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
#[allow(clippy::too_many_arguments)]
fn gen_stmts(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    plan: &SharedMemPlan,
    types: &ValTypes,
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
                        // memory is never touched. FpExt buffers alias
                        // through `reinterpret_cast` since the pool is
                        // typed `uint32_t` for word-level packing.
                        let offset = plan.offsets.get(buf).copied().unwrap_or(0);
                        let ct = c_type(decl.elem);
                        writeln!(
                            s,
                            "{pad}{ct} *b{} = reinterpret_cast<{ct}*>(&_sh_pool[{}u]);",
                            buf.0,
                            offset / 4
                        )
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
                gen_convert(s, p, k, *dst, *src, map, depth)?;
            }
            SSAOpCode::Loop { bound } => {
                let v = val(op.block.operands[0]);
                writeln!(s, "{pad}for (uint32_t {v} = 0u; {v} < {bound}u; ++{v}) {{").unwrap();
                gen_stmts(s, p, k, plan, types, &op.block.body, depth + 1)?;
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
                    types,
                    bound,
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
    types: &ValTypes,
    bound: &KBound,
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
        let guard = match bound {
            KBound::Const(b) => {
                // Skip the guard only when the grid provably has no slack.
                let needed = k.grid.bound.as_const().is_none_or(|g| g * k.block > *b);
                needed.then(|| format!("{b}u"))
            }
            KBound::Expr(e) => {
                let dp = |v: VarId| device_param(p, v);
                Some(format!("(uint32_t)({})", sexpr_str(e, &dp, &dp)))
            }
        };
        if let Some(b) = &guard {
            writeln!(s, "{pad}if ({v} < {b}) {{").unwrap();
        }
        let d = depth + usize::from(guard.is_some());
        gen_par_body(s, p, k, types, attr, reads, writes, block, "0u", d)?;
        if guard.is_some() {
            writeln!(s, "{pad}}}").unwrap();
        }
        return Ok(());
    }

    let bound = bound
        .as_const()
        .expect("non-grid-spanning par bounds are concrete");
    if attr.layout.is_identity() {
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
            types,
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
            types,
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

/// Reads whose value is only used inside one select branch, to be emitted
/// at that branch's entry instead of at the top of the par body. This keeps
/// [`SSAOpCode::Select`]'s short-circuit guarantee: the untaken side's
/// loads (and their possibly out-of-bounds address arithmetic) never
/// execute.
struct ReadSinks<'a> {
    reads: &'a [Access],
    /// The par block's operands (`[par index, one value per read...]`).
    operands: &'a [SSARes],
    attr: &'a ParAttr,
    vid: SSARes,
    slot: &'a str,
    /// Read indices sunk to each `(select node, is-else)` branch entry.
    sunk: HashMap<(SSANode, bool), Vec<usize>>,
}

impl ReadSinks<'_> {
    fn emit(
        &self,
        s: &mut String,
        p: &KernelProgram,
        nid: SSANode,
        is_else: bool,
        depth: usize,
    ) -> Result<(), CompileError> {
        let Some(rs) = self.sunk.get(&(nid, is_else)) else {
            return Ok(());
        };
        let pad = "    ".repeat(depth);
        for &ri in rs {
            let src = access_str(p, &self.reads[ri], self.attr, self.vid, self.slot)?;
            writeln!(
                s,
                "{pad}const {} {} = {src};",
                c_type(p.buffer(self.reads[ri].buf).elem),
                val(self.operands[1 + ri])
            )
            .unwrap();
        }
        Ok(())
    }
}

/// For each value, the first select crossing on the path from the par block
/// to its use sites: absent = unused, `Some(None)` = some use is reached
/// without entering a select branch (or uses disagree on the first
/// crossing), `Some(Some(c))` = every use sits below crossing `c`.
fn mark_use(
    first: &mut HashMap<SSARes, Option<(SSANode, bool)>>,
    v: SSARes,
    c: Option<(SSANode, bool)>,
) {
    match first.entry(v) {
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(c);
        }
        std::collections::hash_map::Entry::Occupied(mut e) => {
            if *e.get() != c {
                e.insert(None);
            }
        }
    }
}

fn walk_uses(
    k: &Kernel,
    body: &[SSANode],
    yields: &[SSARes],
    crossing: Option<(SSANode, bool)>,
    first: &mut HashMap<SSARes, Option<(SSANode, bool)>>,
) {
    for &nid in body {
        let op = k.op(nid);
        for &o in &op.operands {
            mark_use(first, o, crossing);
        }
        if let SSAOpCode::Select { else_block } = &op.opcode {
            let then = crossing.or(Some((nid, false)));
            walk_uses(k, &op.block.body, &op.block.yields, then, first);
            let other = crossing.or(Some((nid, true)));
            walk_uses(k, &else_block.body, &else_block.yields, other, first);
        } else {
            walk_uses(k, &op.block.body, &op.block.yields, crossing, first);
        }
    }
    for &y in yields {
        mark_use(first, y, crossing);
    }
}

/// Decides which reads sink into a select branch: every use of the read's
/// value must sit below one branch crossing, the read's index must not
/// depend on anything defined inside the par block (so it is emittable at
/// the branch entry), and no other read's index may consume the value (a
/// gather's index load must stay eager or the gather never becomes ready).
///
/// Also consulted by `check_accesses`: a sunk read only executes on the
/// taken side of its select, so the checker exempts it from bounds
/// validation. The two must agree — an eagerly emitted read must always be
/// checked.
pub(crate) fn compute_read_sinks(
    k: &Kernel,
    reads: &[Access],
    block: &SSABlock,
) -> HashMap<(SSANode, bool), Vec<usize>> {
    let mut defined_inside: HashSet<SSARes> = block.operands[1..].iter().copied().collect();
    fn collect_defs(k: &Kernel, body: &[SSANode], out: &mut HashSet<SSARes>) {
        for &nid in body {
            let op = k.op(nid);
            out.extend(op.results.iter().copied());
            out.extend(op.block.operands.iter().copied());
            collect_defs(k, &op.block.body, out);
            if let SSAOpCode::Select { else_block } = &op.opcode {
                collect_defs(k, &else_block.body, out);
            }
        }
    }
    collect_defs(k, &block.body, &mut defined_inside);

    let mut index_deps = BTreeSet::new();
    for r in reads {
        r.index_syms(&mut index_deps);
    }

    let mut first = HashMap::new();
    walk_uses(k, &block.body, &block.yields, None, &mut first);

    let mut sunk: HashMap<(SSANode, bool), Vec<usize>> = HashMap::new();
    for (ri, read) in reads.iter().enumerate() {
        let operand = block.operands[1 + ri];
        let Some(&Some(crossing)) = first.get(&operand) else {
            continue;
        };
        if index_deps.contains(&operand) {
            continue;
        }
        let mut syms = BTreeSet::new();
        read.index_syms(&mut syms);
        if syms.iter().any(|v| defined_inside.contains(v)) {
            continue;
        }
        sunk.entry(crossing).or_default().push(ri);
    }
    sunk
}

/// The body of one par point: bind the loaded block operands, run the SSA
/// ops, store the yields.
///
/// A [`IndexMap::Blackbox`] read's index references other loaded values or
/// op results, so reads are emitted as soon as every kernel value their
/// index uses is defined, interleaved with the ops. For kernels without
/// data-dependent reads this degenerates to the plain reads / ops / writes
/// order. Reads used only inside one select branch are sunk to that
/// branch's entry (see [`ReadSinks`]) so the untaken side never executes
/// them.
#[allow(clippy::too_many_arguments)]
fn gen_par_body(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    types: &ValTypes,
    attr: &ParAttr,
    reads: &[Access],
    writes: &[Access],
    block: &SSABlock,
    slot: &str,
    depth: usize,
) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    let vid = block.operands[0];

    // Values not yet defined at the current emission point: the loaded
    // block operands plus the results of the block's top-level ops.
    let mut undefined: HashSet<SSARes> = block.operands[1..].iter().copied().collect();
    for &nid in &block.body {
        undefined.extend(k.op(nid).results.iter().copied());
    }
    let sinks = ReadSinks {
        reads,
        operands: &block.operands,
        attr,
        vid,
        slot,
        sunk: compute_read_sinks(k, reads, block),
    };
    let sunk_set: HashSet<usize> = sinks.sunk.values().flatten().copied().collect();
    let mut pending: Vec<usize> = (0..reads.len())
        .filter(|ri| !sunk_set.contains(ri))
        .collect();

    // Emits every pending read whose index no longer references an
    // undefined value; repeats until a fixpoint (a gather's index buffer
    // load can itself unblock the gather).
    fn emit_ready(
        s: &mut String,
        p: &KernelProgram,
        pad: &str,
        reads: &[Access],
        block: &SSABlock,
        attr: &ParAttr,
        vid: SSARes,
        slot: &str,
        undefined: &mut HashSet<SSARes>,
        pending: &mut Vec<usize>,
    ) -> Result<(), CompileError> {
        loop {
            let mut progress = false;
            let mut i = 0;
            while i < pending.len() {
                let ri = pending[i];
                let mut syms = BTreeSet::new();
                reads[ri].index_syms(&mut syms);
                if syms.iter().any(|v| undefined.contains(v)) {
                    i += 1;
                    continue;
                }
                let operand = block.operands[1 + ri];
                let src = access_str(p, &reads[ri], attr, vid, slot)?;
                writeln!(
                    s,
                    "{pad}const {} {} = {src};",
                    c_type(p.buffer(reads[ri].buf).elem),
                    val(operand)
                )
                .unwrap();
                undefined.remove(&operand);
                pending.remove(i);
                progress = true;
            }
            if !progress {
                return Ok(());
            }
        }
    }

    emit_ready(
        s,
        p,
        &pad,
        reads,
        block,
        attr,
        vid,
        slot,
        &mut undefined,
        &mut pending,
    )?;
    for &nid in &block.body {
        gen_ops(s, p, k, types, std::slice::from_ref(&nid), depth, &sinks)?;
        for r in &k.op(nid).results {
            undefined.remove(r);
        }
        if !pending.is_empty() {
            emit_ready(
                s,
                p,
                &pad,
                reads,
                block,
                attr,
                vid,
                slot,
                &mut undefined,
                &mut pending,
            )?;
        }
    }
    if !pending.is_empty() {
        return Err(CompileError::Codegen(format!(
            "unresolvable data-dependent read cycle in kernel {}",
            k.name
        )));
    }
    for (i, access) in writes.iter().enumerate() {
        let dst = access_str(p, access, attr, vid, slot)?;
        writeln!(s, "{pad}{dst} = {};", val(block.yields[i])).unwrap();
    }
    Ok(())
}

/// Emits a [`SSAOpCode::ConvertLayout`]: `dst[i] = src[map(i)]` over `dst`'s
/// logical domain. Register-to-register conversions reduce to the map
/// `C = f_src^-1 ∘ map ∘ f_dst` from dst physical to src physical index and
/// become slot permutations or warp shuffles per [`classify_convert`];
/// register-to-shared stages the registers out through shared memory.
fn gen_convert(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
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
    let (dst_reg, src_reg) = (reg_name(dst), reg_name(src));
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

fn gen_ops(
    s: &mut String,
    p: &KernelProgram,
    k: &Kernel,
    types: &ValTypes,
    body: &[SSANode],
    depth: usize,
    sinks: &ReadSinks,
) -> Result<(), CompileError> {
    let pad = "    ".repeat(depth);
    for &nid in body {
        let op = k.op(nid);
        match &op.opcode {
            SSAOpCode::ConstU32(c) => {
                writeln!(s, "{pad}const uint32_t {} = {c}u;", val(op.results[0])).unwrap();
            }
            SSAOpCode::ConstSym(e) => {
                let dp = |v: VarId| device_param(p, v);
                writeln!(
                    s,
                    "{pad}const uint32_t {} = (uint32_t)({});",
                    val(op.results[0]),
                    sexpr_str(e, &dp, &dp)
                )
                .unwrap();
            }
            SSAOpCode::ConstField(c) => {
                // IR-level BabyBear constants are canonical `u32` in `[0, P)`;
                // emitted code operates in Montgomery form, so the constant
                // is folded at compile time via `to_monty`. `mont(0) = 0`
                // stays cheap.
                let m = to_monty(*c);
                writeln!(s, "{pad}const uint32_t {} = {m}u;", val(op.results[0])).unwrap();
            }
            SSAOpCode::ConstFpExt(c) => {
                // Each FpExt coefficient is a canonical BabyBear; convert
                // per-slot to Montgomery. `fpext_mul` operates on Montgomery
                // limbs (its BETA constant is likewise Montgomery-encoded).
                let m = [
                    to_monty(c[0]),
                    to_monty(c[1]),
                    to_monty(c[2]),
                    to_monty(c[3]),
                ];
                writeln!(
                    s,
                    "{pad}const FpExt {} = FpExt{{{{{}u, {}u, {}u, {}u}}}};",
                    val(op.results[0]),
                    m[0],
                    m[1],
                    m[2],
                    m[3]
                )
                .unwrap();
            }
            SSAOpCode::LiftFpExt => {
                writeln!(
                    s,
                    "{pad}const FpExt {} = FpExt{{{{{}, 0u, 0u, 0u}}}};",
                    val(op.results[0]),
                    val(op.operands[0])
                )
                .unwrap();
            }
            SSAOpCode::Bin(bop, ty) => {
                let a = val(op.operands[0]);
                let b = val(op.operands[1]);
                let res_ty = types[&op.results[0]];
                writeln!(
                    s,
                    "{pad}const {} {} = {};",
                    c_type(res_ty),
                    val(op.results[0]),
                    bin_str(*bop, *ty, &a, &b)
                )
                .unwrap();
            }
            SSAOpCode::Select { else_block } => {
                // Emit a real `if / else` so only the taken branch's
                // body runs — its loads and pointer arithmetic are
                // gated by the condition, matching the DSL's
                // short-circuit `if cond then A else B` semantics.
                let res_ty = types[&op.results[0]];
                let res = val(op.results[0]);
                let cond = val(op.operands[0]);
                writeln!(s, "{pad}{} {res}{{}};", c_type(res_ty)).unwrap();
                writeln!(s, "{pad}if ({cond}) {{").unwrap();
                sinks.emit(s, p, nid, false, depth + 1)?;
                gen_ops(s, p, k, types, &op.block.body, depth + 1, sinks)?;
                writeln!(s, "{pad}    {res} = {};", val(op.block.yields[0])).unwrap();
                writeln!(s, "{pad}}} else {{").unwrap();
                sinks.emit(s, p, nid, true, depth + 1)?;
                gen_ops(s, p, k, types, &else_block.body, depth + 1, sinks)?;
                writeln!(s, "{pad}    {res} = {};", val(else_block.yields[0])).unwrap();
                writeln!(s, "{pad}}}").unwrap();
            }
            SSAOpCode::Loop { bound } => {
                // The results double as the mutable loop-carried slots.
                for (i, res) in op.results.iter().enumerate() {
                    writeln!(
                        s,
                        "{pad}{} {} = {};",
                        c_type(types[res]),
                        val(*res),
                        val(op.operands[i])
                    )
                    .unwrap();
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
                        "{pad}    const {} {} = {};",
                        c_type(types[carried]),
                        val(*carried),
                        val(op.results[i])
                    )
                    .unwrap();
                }
                gen_ops(s, p, k, types, &op.block.body, depth + 1, sinks)?;
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
    match ty {
        ScalarType::BabyBear => match op {
            BinOp::Add => format!("bb_add({a}, {b})"),
            BinOp::Sub => format!("bb_sub({a}, {b})"),
            BinOp::Mul => format!("bb_mul({a}, {b})"),
            _ => unreachable!("op {op:?} is not defined on BabyBear"),
        },
        ScalarType::FpExt => match op {
            BinOp::Add => format!("fpext_add({a}, {b})"),
            BinOp::Sub => format!("fpext_sub({a}, {b})"),
            BinOp::Mul => format!("fpext_mul({a}, {b})"),
            _ => unreachable!("op {op:?} is not defined on FpExt"),
        },
        ScalarType::U32 | ScalarType::Bool => {
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
        return Ok(format!("{}[{slot}]", reg_name(access.buf)));
    }
    let logical = match &access.index {
        IndexMap::Linear(ll) => ll_apply_str(ll, &val(vid)),
        IndexMap::Affine { expr, bounds } => expr.emit(bounds, &mut CStrEmitter)?,
        IndexMap::SExpr(e) | IndexMap::Blackbox(e) => {
            sexpr_str(e, &|v| val(SSARes(v.0)), &|v| device_param(p, v))
        }
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
    let ct = c_type(p.buffer(buf).elem);
    match p.buffer(buf).kind {
        BufferKind::Input(k) => format!("({cq}{ct}*)p->inputs[{k}]"),
        BufferKind::Output(k) => format!("({cq}{ct}*)p->outputs[{k}]"),
        BufferKind::Scratch { offset } => {
            format!("({cq}{ct}*)((char*)p->scratch + {offset})")
        }
        BufferKind::Shared | BufferKind::Register => {
            unreachable!("kernel-local buffers are never kernel parameters")
        }
    }
}

fn gen_host(s: &mut String, p: &KernelProgram) {
    let n_in = p.input_bufs.len();
    let n_out = p.output_bufs.len();
    let n_par = p.params.len();
    let host_sym = |v: VarId| {
        let i = p
            .params
            .iter()
            .position(|(pv, _)| *pv == v)
            .expect("size expression references an undeclared module parameter");
        format!("p->params[{i}]")
    };
    // Buffer sizes are expressions over the runtime parameters (constants
    // fold to literals), so they're emitted as switches rather than static
    // arrays.
    let size_cases = |bufs: &[BufId]| -> String {
        bufs.iter()
            .enumerate()
            .map(|(i, &b)| {
                let d = p.buffer(b);
                format!(
                    "        case {i}: return (uint64_t)({}) * {}ull;\n",
                    sexpr_str(&d.len_expr(), &host_sym, &host_sym),
                    d.elem.size_bytes()
                )
            })
            .collect()
    };
    let in_cases = size_cases(&p.input_bufs);
    let out_cases = size_cases(&p.output_bufs);

    writeln!(
        s,
        r#"struct Prog {{
    void* inputs[{in_cap}];
    void* outputs[{out_cap}];
    void* scratch;
    int64_t params[{par_cap}];
}};

extern "C" Prog* make_module() {{
    Prog* p = new Prog;
    std::memset(p, 0, sizeof(Prog));
    return p;
}}
extern "C" void destroy_module(Prog* p) {{ delete p; }}
extern "C" void set_param(Prog* p, uint64_t i, int64_t v) {{ p->params[i] = v; }}
extern "C" uint64_t scratch_size(Prog*) {{ return {scratch}ull; }}
extern "C" uint64_t num_outputs(Prog*) {{ return {n_out}ull; }}
extern "C" uint64_t output_size(Prog* p, uint64_t i) {{
    (void)p;
    switch (i) {{
{out_cases}        default: return 0ull;
    }}
}}
extern "C" uint64_t num_inputs(Prog*) {{ return {n_in}ull; }}
extern "C" uint64_t input_size(Prog* p, uint64_t i) {{
    (void)p;
    switch (i) {{
{in_cases}        default: return 0ull;
    }}
}}
extern "C" void set_input(Prog* p, uint64_t i, void* ptr) {{ p->inputs[i] = ptr; }}
extern "C" void set_output(Prog* p, uint64_t i, void* ptr) {{ p->outputs[i] = ptr; }}
extern "C" void set_scratch_buf(Prog* p, void* ptr) {{ p->scratch = ptr; }}
"#,
        in_cap = n_in.max(1),
        out_cap = n_out.max(1),
        par_cap = n_par.max(1),
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
        let mut args = k
            .params
            .iter()
            .map(|&(buf, writable)| buf_arg(p, buf, writable))
            .collect::<Vec<_>>();
        for i in 0..n_par {
            args.push(format!("(uint32_t)p->params[{i}]"));
        }
        let args = args.join(", ");
        let grid = match &k.grid.bound {
            KBound::Const(c) => format!("{c}u"),
            KBound::Expr(e) => format!("(uint32_t)({})", sexpr_str(e, &host_sym, &host_sym)),
        };
        writeln!(
            s,
            "    {}<<<dim3({grid}), dim3({}u), 0, stream>>>({args});",
            k.name, k.block
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
