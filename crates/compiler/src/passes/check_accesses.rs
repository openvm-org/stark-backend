//! Optional exhaustive access checking, enabled by
//! [`crate::runtime::CompileOptions::check_accesses`].
//!
//! For one concrete instantiation of a module (every parameter bound), runs
//! the front half of the pipeline — full monomorphization, canonicalization
//! (which fires the concrete scatter bijectivity/inverse checks) and
//! `lower_to_kir` — then validates every par access whose index map is
//! analyzable ([`IndexMap::Linear`] / [`IndexMap::Affine`]):
//!
//! - reads and writes must stay inside the buffer's logical bounds;
//! - writes must additionally be injective across the parallel dimensions (the par's own index and
//!   the grid index) for any fixed assignment of the sequential loop variables: two parallel points
//!   storing to the same location is a data race, while a sequential overwrite is fine.
//!
//! [`IndexMap::SExpr`] and [`IndexMap::Blackbox`] accesses are skipped
//! silently — the kernel author is trusted for those. Reads that codegen
//! sinks into a select branch (see `codegen::compute_read_sinks`) are also
//! skipped: only the taken side of the select executes them, so the DSL's
//! short-circuit `if cond then A else B` may guard an otherwise
//! out-of-bounds index. Domains larger than [`EXHAUSTIVE_LIMIT`] fall back
//! to the (conservative) interval check alone. Nothing is compiled: no
//! codegen, no nvcc.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::{
    canonicalize, codegen::compute_read_sinks, lower_to_kir, monomorphize, rewrite_parallel_reduce,
    split_program, type_infer,
};
use crate::{
    ir::{Module, VarId},
    kernel_ir::{
        Access, AddressSpace, IndexMap, Kernel, KirProgram, LinearLayout, SSABlock, SSAOpCode,
    },
    quast::{Quast, EXHAUSTIVE_LIMIT},
    CompileError,
};

fn err(msg: String) -> CompileError {
    CompileError::AccessCheck(msg)
}

/// Checks one concrete instantiation: fully monomorphizes `module` against
/// `bindings` (keyed by parameter name), lowers it to KernelIR and runs
/// [`check_program_accesses`]. The concrete canonicalization along the way
/// performs the scatter bijectivity + inverse validation.
pub fn check_module_accesses(
    module: &Module,
    bindings: &BTreeMap<String, i64>,
) -> Result<(), CompileError> {
    let params = module.builder.params();
    assert_eq!(
        params.len(),
        bindings.len(),
        "module `{}` declares {} parameter(s) but got {} binding(s)",
        module.name,
        params.len(),
        bindings.len()
    );
    let env: BTreeMap<VarId, i64> = params
        .iter()
        .map(|(v, name)| {
            let &val = bindings.get(name).unwrap_or_else(|| {
                panic!(
                    "module `{}`: no binding for param `{name}` (got keys {:?})",
                    module.name,
                    bindings.keys().collect::<Vec<_>>(),
                )
            });
            (*v, val)
        })
        .collect();
    let module = monomorphize(module, &env)?;
    // The access check runs on a fully monomorphized module, so the env
    // for the rewrite is empty — every param already got substituted.
    let types = type_infer(&module)?;
    let module = rewrite_parallel_reduce(&module, &types, &BTreeMap::new())?.unwrap_or(module);
    let types = type_infer(&module)?;
    let program = canonicalize(module, types)?;
    // Multi-kernel programs are split into single-kernel modules before
    // lowering (`lower_to_kir` requires single-kernel input).
    let split = split_program(&program)?;
    for sk in &split.kernels {
        let types = type_infer(&sk.module)?;
        let sub = canonicalize((*sk.module).clone(), types)?;
        let prog = lower_to_kir(&sub)?;
        check_program_accesses(&prog)?;
    }
    Ok(())
}

/// Validates every par access of a fully concrete [`KirProgram`].
pub fn check_program_accesses(prog: &KirProgram) -> Result<(), CompileError> {
    for kernel in &prog.kernels {
        let grid_sym = VarId(kernel.grid_var().0);
        check_block(prog, kernel, &kernel.grid.block, grid_sym)?;
    }
    Ok(())
}

/// The par context an access is checked under.
struct ParCtx {
    par_sym: VarId,
    grid_sym: VarId,
    par_bound: usize,
    spans_grid: bool,
    /// `blockDim.x` of the enclosing kernel.
    block: usize,
}

fn check_block(
    prog: &KirProgram,
    kernel: &Kernel,
    block: &SSABlock,
    grid_sym: VarId,
) -> Result<(), CompileError> {
    for node in &block.body {
        let op = kernel.op(*node);
        match &op.opcode {
            SSAOpCode::Loop { .. } => check_block(prog, kernel, &op.block, grid_sym)?,
            SSAOpCode::Par {
                bound,
                spans_grid,
                reads,
                writes,
                ..
            } => {
                let Some(par_bound) = bound.as_const() else {
                    return Err(err(format!(
                        "kernel `{}`: par bound `{bound}` is not concrete; access \
                         checking requires a fully monomorphized module",
                        kernel.name
                    )));
                };
                let cx = ParCtx {
                    par_sym: VarId(op.block.operands[0].0),
                    grid_sym,
                    par_bound,
                    spans_grid: *spans_grid,
                    block: kernel.block,
                };
                // Reads codegen sinks into a select branch only execute on
                // the taken side; the branch condition is their guard.
                let guarded: HashSet<usize> = compute_read_sinks(kernel, reads, &op.block)
                    .into_values()
                    .flatten()
                    .collect();
                for (i, a) in reads.iter().enumerate() {
                    if guarded.contains(&i) {
                        continue;
                    }
                    check_access(prog, kernel, a, &cx, false)?;
                }
                for a in writes {
                    check_access(prog, kernel, a, &cx, true)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_access(
    prog: &KirProgram,
    kernel: &Kernel,
    access: &Access,
    cx: &ParCtx,
    is_write: bool,
) -> Result<(), CompileError> {
    let buf = prog.buffer(access.buf);
    let Some(len) = buf.len_expr().as_const() else {
        return Err(err(format!(
            "kernel `{}`: buffer `{}` keeps a symbolic shape; access checking \
             requires a fully monomorphized module",
            kernel.name, buf.name
        )));
    };
    let at = format!(
        "kernel `{}`: {} of buffer `{}`",
        kernel.name,
        if is_write { "write" } else { "read" },
        buf.name
    );
    // Shared and register buffers are per-block instances: writes from
    // different blocks land in different physical memories, so the grid
    // index is not a parallel dimension for them.
    let grid_parallel = buf.space == AddressSpace::Global;
    match &access.index {
        IndexMap::SExpr(_) | IndexMap::Blackbox(_) => Ok(()),
        IndexMap::Linear(layout) => check_linear(layout, cx.par_bound, len, is_write, &at),
        IndexMap::Affine { expr, bounds } => {
            check_affine(expr, bounds, len, is_write, cx, grid_parallel, &at)
        }
    }
}

/// `index = layout(par index)` over the domain `0..par_bound`.
fn check_linear(
    layout: &LinearLayout,
    par_bound: usize,
    len: i64,
    is_write: bool,
    at: &str,
) -> Result<(), CompileError> {
    if par_bound > EXHAUSTIVE_LIMIT {
        // Exact only when the domain covers the full `2^k` input space: a
        // non-invertible map then provably repeats an image.
        let k = layout.bases.len();
        if is_write && k < 64 && par_bound == 1usize << k && layout.inverse().is_none() {
            return Err(err(format!(
                "{at}: linear index map is not injective over its {par_bound}-point \
                 domain — parallel points write the same location"
            )));
        }
        return Ok(());
    }
    let mut seen: HashMap<u64, u64> = HashMap::new();
    for x in 0..par_bound as u64 {
        let idx = layout.apply(x);
        if idx >= len as u64 {
            return Err(err(format!(
                "{at}: index {idx} out of bounds [0, {len}) at par index {x}"
            )));
        }
        if is_write {
            if let Some(prev) = seen.insert(idx, x) {
                return Err(err(format!(
                    "{at}: par indices {prev} and {x} both write location {idx} — \
                     parallel data race"
                )));
            }
        }
    }
    Ok(())
}

fn check_affine(
    expr: &Quast,
    bounds: &BTreeMap<VarId, u64>,
    len: i64,
    is_write: bool,
    cx: &ParCtx,
    grid_parallel: bool,
    at: &str,
) -> Result<(), CompileError> {
    let mut syms = BTreeSet::new();
    expr.syms(&mut syms);
    if let Some(v) = syms.iter().find(|v| !bounds.contains_key(v)) {
        return Err(err(format!(
            "{at}: index expression `{expr}` uses unbounded symbol {v:?}"
        )));
    }
    if is_write && cx.par_bound > 1 && !syms.contains(&cx.par_sym) {
        return Err(err(format!(
            "{at}: index `{expr}` does not depend on the par index, so all \
             {} parallel points write the same location",
            cx.par_bound
        )));
    }

    // Conservative interval first: when it fits, every point fits.
    let fits = matches!(expr.range(bounds), Some((lo, hi)) if lo >= 0 && hi < len);
    // For a spans_grid par the grid index is dependent on the par index
    // (`grid = par / blockDim`), not a free dimension of its own.
    let derive_grid = cx.spans_grid && syms.contains(&cx.grid_sym) && syms.contains(&cx.par_sym);
    // `bounds` is the whole symbol environment of the kernel (a superset of
    // the expr's symbols); only enumerate the ones the expr actually uses.
    let vars: Vec<(VarId, u64)> = bounds
        .iter()
        .filter(|(v, _)| syms.contains(v) && !(derive_grid && **v == cx.grid_sym))
        .map(|(&v, &b)| (v, b))
        .collect();
    let total: u128 = vars.iter().map(|&(_, b)| b as u128).product();
    if total > EXHAUSTIVE_LIMIT as u128 {
        if fits {
            return Ok(());
        }
        let range = expr
            .range(bounds)
            .map_or("unknown".to_string(), |(lo, hi)| format!("[{lo}, {hi}]"));
        return Err(err(format!(
            "{at}: index `{expr}` has interval {range}, outside [0, {len}), and its \
             {total}-point domain is too large to enumerate"
        )));
    }
    if total == 0 || (fits && !is_write) {
        return Ok(());
    }

    let (par_is, seq_is): (Vec<usize>, Vec<usize>) = (0..vars.len())
        .partition(|&i| vars[i].0 == cx.par_sym || (grid_parallel && vars[i].0 == cx.grid_sym));
    let mut vals = vec![0i64; vars.len()];
    let mut env: BTreeMap<VarId, i64> = vars.iter().map(|&(v, _)| (v, 0)).collect();
    // (sequential assignment, index) -> first parallel assignment writing it.
    let mut seen: HashMap<(Vec<i64>, i64), Vec<i64>> = HashMap::new();
    loop {
        for (i, &(v, _)) in vars.iter().enumerate() {
            env.insert(v, vals[i]);
        }
        if derive_grid {
            let p = env[&cx.par_sym];
            env.insert(cx.grid_sym, p / cx.block as i64);
        }
        let idx = expr.eval(&env);
        if idx < 0 || idx >= len {
            let assign: Vec<String> = vars
                .iter()
                .zip(&vals)
                .map(|(&(v, _), &x)| format!("{v:?}={x}"))
                .collect();
            return Err(err(format!(
                "{at}: index {idx} out of bounds [0, {len}) at {}",
                assign.join(", ")
            )));
        }
        if is_write {
            let par_vals: Vec<i64> = par_is.iter().map(|&i| vals[i]).collect();
            let seq_vals: Vec<i64> = seq_is.iter().map(|&i| vals[i]).collect();
            if let Some(prev) = seen.insert((seq_vals, idx), par_vals.clone()) {
                if prev != par_vals {
                    return Err(err(format!(
                        "{at}: parallel points {prev:?} and {par_vals:?} both write \
                         location {idx} — parallel data race"
                    )));
                }
            }
        }
        let mut d = vars.len();
        loop {
            if d == 0 {
                return Ok(());
            }
            d -= 1;
            vals[d] += 1;
            if (vals[d] as u64) < vars[d].1 {
                break;
            }
            vals[d] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use smallvec::SmallVec;

    use super::*;
    use crate::{
        ir::{IRBuilder, ScalarType},
        kernel_ir::{
            AddressSpace, BufId, BufferDecl, BufferKind, KBound, SSABlock, SSAOp, SSAOpCode,
        },
    };

    /// One kernel, one par of `par_bound` points over a single global buffer
    /// of `len` elements. The grid index is `VarId(0)`, the par index
    /// `VarId(1)`; any other symbol acts as a sequential loop variable.
    fn one_par_prog(
        len: usize,
        par_bound: usize,
        reads: Vec<Access>,
        writes: Vec<Access>,
    ) -> KirProgram {
        par_prog(
            len,
            par_bound,
            false,
            128,
            AddressSpace::Global,
            reads,
            writes,
        )
    }

    fn par_prog(
        len: usize,
        par_bound: usize,
        spans_grid: bool,
        block_dim: usize,
        space: AddressSpace,
        reads: Vec<Access>,
        writes: Vec<Access>,
    ) -> KirProgram {
        let mut k = Kernel::new("k".into(), KBound::Const(1), block_dim);
        let par_idx = k.fresh_val();
        let mut block = SSABlock::default();
        block.operands.push(par_idx);
        // One loaded value per read; left unused, so no read is
        // select-guarded and all of them get checked.
        for _ in &reads {
            let v = k.fresh_val();
            block.operands.push(v);
        }
        let node = k.push_op(SSAOp {
            operands: SmallVec::new(),
            results: SmallVec::new(),
            opcode: SSAOpCode::Par {
                bound: KBound::Const(par_bound),
                spans_grid,
                attr: None,
                reads,
                writes,
            },
            block,
        });
        k.grid.block.body.push(node);
        let global = space == AddressSpace::Global;
        KirProgram {
            name: "p".into(),
            buffers: vec![BufferDecl {
                name: "buf".into(),
                elem: ScalarType::BabyBear,
                shape: vec![len.into()],
                kind: if global {
                    BufferKind::Output(0)
                } else {
                    BufferKind::Shared
                },
                space,
                layout: None,
            }],
            kernels: vec![k],
            input_bufs: vec![],
            output_bufs: if global { vec![BufId(0)] } else { vec![] },
            params: vec![],
        }
    }

    /// One par of 8 points over an 8-element global buffer with a single
    /// read, whose loaded value is consumed by the yields of an
    /// `if 1 then .. else ..` select as directed by `then_uses` /
    /// `else_uses` (a constant zero stands in when a branch doesn't use it).
    fn select_guard_prog(read: Access, then_uses: bool, else_uses: bool) -> KirProgram {
        let mut k = Kernel::new("k".into(), KBound::Const(1), 128);
        let par_idx = k.fresh_val();
        let r0 = k.fresh_val();
        let cond = k.fresh_val();
        let z = k.fresh_val();
        let res = k.fresh_val();
        let n_cond = k.push_op(SSAOp {
            operands: SmallVec::new(),
            results: SmallVec::from_slice(&[cond]),
            opcode: SSAOpCode::ConstU32(1),
            block: SSABlock::default(),
        });
        let n_z = k.push_op(SSAOp {
            operands: SmallVec::new(),
            results: SmallVec::from_slice(&[z]),
            opcode: SSAOpCode::ConstU32(0),
            block: SSABlock::default(),
        });
        let mut then_block = SSABlock::default();
        then_block.yields.push(if then_uses { r0 } else { z });
        let mut else_block = SSABlock::default();
        else_block.yields.push(if else_uses { r0 } else { z });
        let n_sel = k.push_op(SSAOp {
            operands: SmallVec::from_slice(&[cond]),
            results: SmallVec::from_slice(&[res]),
            opcode: SSAOpCode::Select { else_block },
            block: then_block,
        });
        let mut block = SSABlock::default();
        block.operands.push(par_idx);
        block.operands.push(r0);
        block.body.extend([n_cond, n_z, n_sel]);
        let node = k.push_op(SSAOp {
            operands: SmallVec::new(),
            results: SmallVec::new(),
            opcode: SSAOpCode::Par {
                bound: KBound::Const(8),
                spans_grid: false,
                attr: None,
                reads: vec![read],
                writes: vec![],
            },
            block,
        });
        k.grid.block.body.push(node);
        KirProgram {
            name: "p".into(),
            buffers: vec![BufferDecl {
                name: "buf".into(),
                elem: ScalarType::BabyBear,
                shape: vec![8.into()],
                kind: BufferKind::Output(0),
                space: AddressSpace::Global,
                layout: None,
            }],
            kernels: vec![k],
            input_bufs: vec![],
            output_bufs: vec![BufId(0)],
            params: vec![],
        }
    }

    fn affine(expr: Quast, bounds: &[(VarId, u64)]) -> Access {
        Access {
            buf: BufId(0),
            index: IndexMap::Affine {
                expr,
                bounds: bounds.iter().copied().collect(),
            },
        }
    }

    const PAR: VarId = VarId(1);
    const GRID: VarId = VarId(0);
    const LOOPV: VarId = VarId(9);

    #[test]
    fn affine_in_bounds_write_passes() {
        let w = affine(Quast::sym(PAR), &[(PAR, 8)]);
        check_program_accesses(&one_par_prog(8, 8, vec![], vec![w])).unwrap();
    }

    #[test]
    fn affine_oob_read_caught() {
        let r = affine(Quast::sym(PAR).add(&Quast::cst(4)), &[(PAR, 8)]);
        let e = check_program_accesses(&one_par_prog(8, 8, vec![r], vec![])).unwrap_err();
        assert!(e.to_string().contains("out of bounds"), "{e}");
    }

    /// Two par points mapping to the same location (`i / 2`) is a parallel
    /// data race for a write...
    #[test]
    fn affine_non_injective_write_caught() {
        let w = affine(Quast::sym(PAR).floordiv(2), &[(PAR, 8)]);
        let e = check_program_accesses(&one_par_prog(8, 8, vec![], vec![w])).unwrap_err();
        assert!(e.to_string().contains("data race"), "{e}");
    }

    /// ...but perfectly fine for a read.
    #[test]
    fn aliasing_affine_read_passes() {
        let r = affine(Quast::sym(PAR).floordiv(2), &[(PAR, 8)]);
        check_program_accesses(&one_par_prog(8, 8, vec![r], vec![])).unwrap();
    }

    /// Repeating the same location across a *sequential* loop variable is an
    /// overwrite, not a race: `i + 0*j` collides across `j` for fixed `i`.
    #[test]
    fn sequential_overwrite_passes() {
        let expr = Quast::sym(PAR).add(&Quast::sym(LOOPV).mul_c(0));
        let w = affine(expr, &[(PAR, 4), (LOOPV, 4)]);
        check_program_accesses(&one_par_prog(4, 4, vec![], vec![w])).unwrap();
    }

    /// The grid index is a parallel dimension: `i + 0*g` collides across
    /// blocks.
    #[test]
    fn grid_broadcast_write_caught() {
        let expr = Quast::sym(PAR).add(&Quast::sym(GRID).mul_c(0));
        let w = affine(expr, &[(PAR, 4), (GRID, 2)]);
        let e = check_program_accesses(&one_par_prog(4, 4, vec![], vec![w])).unwrap_err();
        assert!(e.to_string().contains("data race"), "{e}");
    }

    /// `lower_to_kir` stores the *whole* symbol environment in each access's
    /// bounds map; symbols the expr never uses (here the grid index) must not
    /// be treated as extra parallel dimensions.
    #[test]
    fn superset_bounds_ignored() {
        let w = affine(Quast::sym(PAR), &[(PAR, 8), (GRID, 2)]);
        check_program_accesses(&one_par_prog(8, 8, vec![], vec![w])).unwrap();
    }

    /// For a `spans_grid` par the grid index is derived (`grid = par /
    /// blockDim`), so `par % blockDim + grid * blockDim` is the identity —
    /// not a collision across independently-enumerated grid values.
    #[test]
    fn spans_grid_derived_grid_passes() {
        let expr = Quast::sym(PAR).rem_c(4).add(&Quast::sym(GRID).mul_c(4));
        let w = affine(expr, &[(PAR, 8), (GRID, 2)]);
        let prog = par_prog(8, 8, true, 4, AddressSpace::Global, vec![], vec![w]);
        check_program_accesses(&prog).unwrap();
    }

    /// Shared buffers are per-block instances, so a write index that ignores
    /// the grid index is not a cross-block race.
    #[test]
    fn shared_buffer_cross_block_write_passes() {
        let expr = Quast::sym(PAR).add(&Quast::sym(GRID).mul_c(0));
        let w = affine(expr, &[(PAR, 4), (GRID, 2)]);
        let prog = par_prog(4, 4, false, 128, AddressSpace::Shared, vec![], vec![w]);
        check_program_accesses(&prog).unwrap();
    }

    /// A read whose value is only consumed inside one select branch is sunk
    /// by codegen — the untaken side never executes the load — so its
    /// (possibly out-of-bounds) index is exempt from checking.
    #[test]
    fn select_guarded_read_skipped() {
        let r = affine(Quast::sym(PAR).add(&Quast::cst(4)), &[(PAR, 8)]);
        check_program_accesses(&select_guard_prog(r, true, false)).unwrap();
    }

    /// The same read consumed by *both* branches stays eagerly emitted and
    /// must pass the bounds check.
    #[test]
    fn read_used_in_both_branches_checked() {
        let r = affine(Quast::sym(PAR).add(&Quast::cst(4)), &[(PAR, 8)]);
        let e = check_program_accesses(&select_guard_prog(r, true, true)).unwrap_err();
        assert!(e.to_string().contains("out of bounds"), "{e}");
    }

    /// A multi-point par whose write index ignores the par index broadcasts
    /// every point onto the same location.
    #[test]
    fn write_ignoring_par_index_caught() {
        let w = affine(Quast::sym(LOOPV), &[(LOOPV, 4)]);
        let e = check_program_accesses(&one_par_prog(8, 4, vec![], vec![w])).unwrap_err();
        assert!(
            e.to_string().contains("does not depend on the par index"),
            "{e}"
        );
    }

    /// Author-trusted index kinds are skipped even when obviously wild.
    #[test]
    fn sexpr_and_blackbox_skipped() {
        let wild = crate::quast::SExpr::sym(PAR).add(&crate::quast::SExpr::cst(
            crate::quast::SymConst::Lit(1 << 40),
        ));
        let r = Access {
            buf: BufId(0),
            index: IndexMap::SExpr(wild.clone()),
        };
        let w = Access {
            buf: BufId(0),
            index: IndexMap::Blackbox(wild),
        };
        check_program_accesses(&one_par_prog(8, 8, vec![r], vec![w])).unwrap();
    }

    #[test]
    fn linear_oob_write_caught() {
        let w = Access {
            buf: BufId(0),
            index: IndexMap::Linear(LinearLayout::identity(4)),
        };
        let e = check_program_accesses(&one_par_prog(8, 16, vec![], vec![w])).unwrap_err();
        assert!(e.to_string().contains("out of bounds"), "{e}");
    }

    /// `bases = [1, 1]` maps `x=1` and `x=2` to the same image.
    #[test]
    fn linear_non_injective_write_caught() {
        let w = Access {
            buf: BufId(0),
            index: IndexMap::Linear(LinearLayout {
                bases: vec![1, 1],
                offset: 0,
            }),
        };
        let e = check_program_accesses(&one_par_prog(4, 4, vec![], vec![w])).unwrap_err();
        assert!(e.to_string().contains("data race"), "{e}");
    }

    /// Above [`EXHAUSTIVE_LIMIT`] the interval check alone decides: a
    /// fitting interval passes without enumeration...
    #[test]
    fn large_domain_fitting_interval_passes() {
        let w = affine(Quast::sym(PAR), &[(PAR, 1 << 20)]);
        check_program_accesses(&one_par_prog(1 << 20, 1 << 20, vec![], vec![w])).unwrap();
    }

    /// ...and a violating one errors even though the domain is too large to
    /// enumerate.
    #[test]
    fn large_domain_interval_violation_caught() {
        let w = affine(Quast::sym(PAR).add(&Quast::cst(1)), &[(PAR, 1 << 20)]);
        let e =
            check_program_accesses(&one_par_prog(1 << 20, 1 << 20, vec![], vec![w])).unwrap_err();
        assert!(e.to_string().contains("too large to enumerate"), "{e}");
    }

    /// Driver happy path: a valid symbolic module instantiated at 16.
    #[test]
    fn check_module_accesses_passes_valid_module() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| b.index(x, &[i]));
        let module = b.finish("valid", body);
        check_module_accesses(&module, &[("n".to_string(), 16)].into()).unwrap();
    }

    /// Driver catches an out-of-bounds load the static pipeline accepts:
    /// `x[i + 1]` over the full domain reads one element past the end.
    #[test]
    fn check_module_accesses_catches_oob_load() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| {
            let one = b.const_u32(1);
            let i1 = b.add(i, one);
            b.index(x, &[i1])
        });
        let module = b.finish("oob_shift", body);
        let e = check_module_accesses(&module, &[("n".to_string(), 8)].into()).unwrap_err();
        assert!(e.to_string().contains("out of bounds"), "{e}");
    }
}
