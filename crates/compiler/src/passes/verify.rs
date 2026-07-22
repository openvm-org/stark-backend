//! Structural verification of [`KernelProgram`]s.

use std::collections::{BTreeSet, HashSet};

use crate::{
    kernel_ir::{Kernel, KernelProgram, SSABlock, SSAOpCode, SSARes},
    CompileError,
};

fn verr(msg: String) -> CompileError {
    CompileError::Verify(msg)
}

/// Structural checks on a [`KernelProgram`], run after codegen:
///
/// 1. every kernel is in SSA form: each value is defined exactly once (as a block operand or an op
///    result), and every use — op operands, yields and access-index symbols — refers to a defined
///    value;
/// 2. every par block is primitive: no nested pars, no statement-style loops (loop-carried
///    `scf.for`s are allowed and checked recursively) and no syncs.
pub fn verify(p: &KernelProgram) -> Result<(), CompileError> {
    for kernel in &p.kernels {
        let mut defined = HashSet::new();
        collect_defs(kernel, &kernel.grid.block, &mut defined)?;
        check_uses(kernel, &kernel.grid.block, &defined)?;
        check_primitive(kernel, &kernel.grid.block, false)?;
    }
    Ok(())
}

fn collect_defs(
    k: &Kernel,
    block: &SSABlock,
    defined: &mut HashSet<SSARes>,
) -> Result<(), CompileError> {
    for &v in &block.operands {
        if !defined.insert(v) {
            return Err(verr(format!("{}: v{} defined more than once", k.name, v.0)));
        }
    }
    for &id in &block.body {
        let op = k.op(id);
        collect_defs(k, &op.block, defined)?;
        for &r in &op.results {
            if !defined.insert(r) {
                return Err(verr(format!("{}: v{} defined more than once", k.name, r.0)));
            }
        }
    }
    Ok(())
}

fn check_uses(k: &Kernel, block: &SSABlock, defined: &HashSet<SSARes>) -> Result<(), CompileError> {
    let check = |v: SSARes, what: &str| {
        if defined.contains(&v) {
            Ok(())
        } else {
            Err(verr(format!(
                "{}: {what} uses undefined value v{}",
                k.name, v.0
            )))
        }
    };
    for &id in &block.body {
        let op = k.op(id);
        for &u in &op.operands {
            check(u, "operand")?;
        }
        if let SSAOpCode::Par { reads, writes, .. } = &op.opcode {
            let mut syms = BTreeSet::new();
            for a in reads.iter().chain(writes) {
                a.index_syms(&mut syms);
            }
            for v in syms {
                check(v, "access index")?;
            }
        }
        check_uses(k, &op.block, defined)?;
    }
    for &y in &block.yields {
        check(y, "yield")?;
    }
    Ok(())
}

fn check_primitive(k: &Kernel, block: &SSABlock, in_par: bool) -> Result<(), CompileError> {
    for &id in &block.body {
        let op = k.op(id);
        match &op.opcode {
            SSAOpCode::Par { .. } => {
                if in_par {
                    return Err(verr(format!("{}: par nested inside a par", k.name)));
                }
                check_primitive(k, &op.block, true)?;
            }
            SSAOpCode::Loop { .. } => {
                if in_par && op.results.is_empty() {
                    return Err(verr(format!(
                        "{}: statement-style loop inside a par",
                        k.name
                    )));
                }
                check_primitive(k, &op.block, in_par)?;
            }
            SSAOpCode::Sync => {
                if in_par {
                    return Err(verr(format!("{}: sync inside a par", k.name)));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;
    use crate::{
        ir::{BinOp, ScalarType},
        kernel_ir::{Kernel, SSAOp},
    };

    fn prog_of(k: Kernel) -> KernelProgram {
        KernelProgram {
            name: k.name.clone(),
            buffers: vec![],
            kernels: vec![k],
            scratch_bytes: 0,
            input_bufs: vec![],
            output_bufs: vec![],
        }
    }

    fn empty_par(bound: usize) -> SSAOpCode {
        SSAOpCode::Par {
            bound,
            spans_grid: false,
            attr: None,
            reads: vec![],
            writes: vec![],
        }
    }

    #[test]
    fn verify_rejects_duplicate_definition() {
        let mut k = Kernel::new("bad".into(), 1, 32);
        let vid = k.fresh_val();
        let v = k.fresh_val();
        let c1 = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![v],
            opcode: SSAOpCode::ConstU32(0),
            block: SSABlock::default(),
        });
        let c2 = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![v],
            opcode: SSAOpCode::ConstU32(1),
            block: SSABlock::default(),
        });
        let par = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![],
            opcode: empty_par(1),
            block: SSABlock {
                operands: smallvec![vid],
                body: smallvec![c1, c2],
                yields: smallvec![],
            },
        });
        k.grid.block.body.push(par);
        let err = verify(&prog_of(k)).unwrap_err();
        assert!(matches!(err, CompileError::Verify(_)), "{err}");
    }

    #[test]
    fn verify_rejects_undefined_use() {
        let mut k = Kernel::new("bad".into(), 1, 32);
        let vid = k.fresh_val();
        let v = k.fresh_val();
        let ghost = SSARes(1000);
        let add = k.push_op(SSAOp {
            operands: smallvec![ghost, ghost],
            results: smallvec![v],
            opcode: SSAOpCode::Bin(BinOp::Add, ScalarType::U32),
            block: SSABlock::default(),
        });
        let par = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![],
            opcode: empty_par(1),
            block: SSABlock {
                operands: smallvec![vid],
                body: smallvec![add],
                yields: smallvec![],
            },
        });
        k.grid.block.body.push(par);
        let err = verify(&prog_of(k)).unwrap_err();
        assert!(matches!(err, CompileError::Verify(_)), "{err}");
    }

    #[test]
    fn verify_rejects_nested_par() {
        let mut k = Kernel::new("bad".into(), 1, 32);
        let outer_vid = k.fresh_val();
        let inner_vid = k.fresh_val();
        let inner = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![],
            opcode: empty_par(1),
            block: SSABlock {
                operands: smallvec![inner_vid],
                body: smallvec![],
                yields: smallvec![],
            },
        });
        let outer = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![],
            opcode: empty_par(1),
            block: SSABlock {
                operands: smallvec![outer_vid],
                body: smallvec![inner],
                yields: smallvec![],
            },
        });
        k.grid.block.body.push(outer);
        let err = verify(&prog_of(k)).unwrap_err();
        assert!(matches!(err, CompileError::Verify(_)), "{err}");
    }

    #[test]
    fn verify_rejects_statement_loop_in_par() {
        let mut k = Kernel::new("bad".into(), 1, 32);
        let vid = k.fresh_val();
        let iv = k.fresh_val();
        let l = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![],
            opcode: SSAOpCode::Loop { bound: 4 },
            block: SSABlock {
                operands: smallvec![iv],
                body: smallvec![],
                yields: smallvec![],
            },
        });
        let par = k.push_op(SSAOp {
            operands: smallvec![],
            results: smallvec![],
            opcode: empty_par(1),
            block: SSABlock {
                operands: smallvec![vid],
                body: smallvec![l],
                yields: smallvec![],
            },
        });
        k.grid.block.body.push(par);
        let err = verify(&prog_of(k)).unwrap_err();
        assert!(matches!(err, CompileError::Verify(_)), "{err}");
    }
}
