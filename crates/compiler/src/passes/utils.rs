//! Helpers shared between passes.

use std::collections::HashMap;

use crate::{
    ir::{BinOp, IRBuilder, Node, NodeId, VarId},
    passes::canonicalize::{CanonValue, TensorRef},
    quast::{Quast, SExpr, SymConst},
    CompileError,
};

pub(crate) fn ceil_log2(n: usize) -> usize {
    n.max(1).next_power_of_two().trailing_zeros() as usize
}

/// Extracts an HIR index expression as a quasi-affine [`Quast`].
///
/// `syms` maps in-scope binder variables (outer compute var, par index, loop
/// induction vars, ...) to the Quast the caller wants them to appear as;
/// `lets` resolves scalar let-bound variables to their defining nodes.
/// `Let`s encountered inside the expression itself are chased locally.
pub(crate) fn hir_to_quast(
    b: &IRBuilder,
    id: NodeId,
    syms: &dyn Fn(VarId) -> Option<Quast>,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
) -> Result<Quast, CompileError> {
    hir_to_quast_impl(b, id, syms, lets, &mut HashMap::new())
}

fn hir_to_quast_impl(
    b: &IRBuilder,
    id: NodeId,
    syms: &dyn Fn(VarId) -> Option<Quast>,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
    local: &mut HashMap<VarId, NodeId>,
) -> Result<Quast, CompileError> {
    match b.node(id).clone() {
        Node::ConstU32(c) => Ok(Quast::cst(c as i64)),
        Node::Var(v) => {
            if let Some(q) = syms(v) {
                Ok(q)
            } else if let Some(&n) = local.get(&v) {
                hir_to_quast_impl(b, n, syms, lets, local)
            } else if let Some(n) = lets(v) {
                hir_to_quast_impl(b, n, syms, lets, local)
            } else {
                Err(CompileError::Lower(format!(
                    "variable {v:?} is not usable in an index expression"
                )))
            }
        }
        Node::Bin(op, x, y) => match op {
            BinOp::Add => Ok(hir_to_quast_impl(b, x, syms, lets, local)?
                .add(&hir_to_quast_impl(b, y, syms, lets, local)?)),
            BinOp::Sub => Ok(hir_to_quast_impl(b, x, syms, lets, local)?
                .sub(&hir_to_quast_impl(b, y, syms, lets, local)?)),
            BinOp::Mul => {
                if let Some(c) = const_eval_impl(b, x, lets, local) {
                    Ok(hir_to_quast_impl(b, y, syms, lets, local)?.mul_c(c))
                } else if let Some(c) = const_eval_impl(b, y, lets, local) {
                    Ok(hir_to_quast_impl(b, x, syms, lets, local)?.mul_c(c))
                } else {
                    Err(CompileError::Lower(
                        "non-affine multiplication in index expression".into(),
                    ))
                }
            }
            BinOp::Div => {
                let c = const_eval_impl(b, y, lets, local)
                    .filter(|&c| c > 0)
                    .ok_or_else(|| {
                        CompileError::Lower(
                            "index division requires a positive constant divisor".into(),
                        )
                    })?;
                Ok(hir_to_quast_impl(b, x, syms, lets, local)?.floordiv(c))
            }
            BinOp::Rem => {
                let c = const_eval_impl(b, y, lets, local)
                    .filter(|&c| c > 0)
                    .ok_or_else(|| {
                        CompileError::Lower(
                            "index remainder requires a positive constant divisor".into(),
                        )
                    })?;
                Ok(hir_to_quast_impl(b, x, syms, lets, local)?.rem_c(c))
            }
            BinOp::Lt | BinOp::Le | BinOp::Eq => {
                Err(CompileError::Lower("comparison in index expression".into()))
            }
        },
        Node::Let { var, value, body } => {
            local.insert(var, value);
            hir_to_quast_impl(b, body, syms, lets, local)
        }
        other => Err(CompileError::Lower(format!(
            "non-quasi-affine index expression: {other:?}"
        ))),
    }
}

/// Extracts an HIR index expression as a symbolic [`SExpr`].
///
/// The fallback for indices that are not quasi-affine ([`hir_to_quast`])
/// but still only combine in-scope binders, module parameters and literals.
/// `syms` maps in-scope binder variables to the `SExpr` (an `Expr::Sym`)
/// the caller wants them to appear as; module parameters enter through
/// `Node::ConstSym` in `SymConst::Sym` position. Symbolic divisors are
/// trusted to be positive (the author knows their parameters).
pub(crate) fn hir_to_sexpr(
    b: &IRBuilder,
    id: NodeId,
    syms: &dyn Fn(VarId) -> Option<SExpr>,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
) -> Result<SExpr, CompileError> {
    hir_to_sexpr_impl(b, id, syms, lets, &mut HashMap::new())
}

/// A subexpression that folds to a (possibly symbolic) constant, for the
/// coefficient / divisor positions of [`hir_to_sexpr`].
fn sexpr_const_side(
    b: &IRBuilder,
    id: NodeId,
    syms: &dyn Fn(VarId) -> Option<SExpr>,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
    local: &mut HashMap<VarId, NodeId>,
) -> Option<SymConst> {
    match hir_to_sexpr_impl(b, id, syms, lets, local)
        .ok()?
        .fold_lits()
    {
        SExpr::Const(c) => Some(c),
        _ => None,
    }
}

fn hir_to_sexpr_impl(
    b: &IRBuilder,
    id: NodeId,
    syms: &dyn Fn(VarId) -> Option<SExpr>,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
    local: &mut HashMap<VarId, NodeId>,
) -> Result<SExpr, CompileError> {
    match b.node(id).clone() {
        Node::ConstU32(c) => Ok(SExpr::cst(SymConst::Lit(c as i64))),
        Node::ConstSym(e) => Ok(e),
        Node::Var(v) => {
            if let Some(e) = syms(v) {
                Ok(e)
            } else if let Some(&n) = local.get(&v) {
                hir_to_sexpr_impl(b, n, syms, lets, local)
            } else if let Some(n) = lets(v) {
                hir_to_sexpr_impl(b, n, syms, lets, local)
            } else {
                Err(CompileError::Lower(format!(
                    "variable {v:?} is not usable in an index expression"
                )))
            }
        }
        Node::Bin(op, x, y) => match op {
            BinOp::Add => Ok(hir_to_sexpr_impl(b, x, syms, lets, local)?
                .add(&hir_to_sexpr_impl(b, y, syms, lets, local)?)),
            BinOp::Sub => Ok(hir_to_sexpr_impl(b, x, syms, lets, local)?
                .sub(&hir_to_sexpr_impl(b, y, syms, lets, local)?)),
            BinOp::Mul => {
                if let Some(c) = sexpr_const_side(b, x, syms, lets, local) {
                    Ok(hir_to_sexpr_impl(b, y, syms, lets, local)?.mul_c(c))
                } else if let Some(c) = sexpr_const_side(b, y, syms, lets, local) {
                    Ok(hir_to_sexpr_impl(b, x, syms, lets, local)?.mul_c(c))
                } else {
                    Err(CompileError::Lower(
                        "non-affine multiplication in index expression".into(),
                    ))
                }
            }
            BinOp::Div => {
                let c = sexpr_const_side(b, y, syms, lets, local)
                    .filter(|c| c.as_lit().is_none_or(|l| l > 0))
                    .ok_or_else(|| {
                        CompileError::Lower(
                            "index division requires a positive constant divisor".into(),
                        )
                    })?;
                Ok(hir_to_sexpr_impl(b, x, syms, lets, local)?.floordiv(c))
            }
            BinOp::Rem => {
                let c = sexpr_const_side(b, y, syms, lets, local)
                    .filter(|c| c.as_lit().is_none_or(|l| l > 0))
                    .ok_or_else(|| {
                        CompileError::Lower(
                            "index remainder requires a positive constant divisor".into(),
                        )
                    })?;
                Ok(hir_to_sexpr_impl(b, x, syms, lets, local)?.rem_c(c))
            }
            BinOp::Lt | BinOp::Le | BinOp::Eq => {
                Err(CompileError::Lower("comparison in index expression".into()))
            }
        },
        Node::Let { var, value, body } => {
            local.insert(var, value);
            hir_to_sexpr_impl(b, body, syms, lets, local)
        }
        other => Err(CompileError::Lower(format!(
            "non-symbolic index expression: {other:?}"
        ))),
    }
}

/// Constant-folds an index subexpression, resolving variables through `lets`.
#[cfg_attr(not(test), allow(dead_code))] // exercised by pass unit tests
pub(crate) fn const_eval_index(
    b: &IRBuilder,
    id: NodeId,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
) -> Option<i64> {
    const_eval_impl(b, id, lets, &HashMap::new())
}

fn const_eval_impl(
    b: &IRBuilder,
    id: NodeId,
    lets: &dyn Fn(VarId) -> Option<NodeId>,
    local: &HashMap<VarId, NodeId>,
) -> Option<i64> {
    match b.node(id) {
        Node::ConstU32(c) => Some(*c as i64),
        Node::Var(v) => {
            let n = local.get(v).copied().or_else(|| lets(*v))?;
            const_eval_impl(b, n, lets, local)
        }
        Node::Bin(op, x, y) => {
            let a = const_eval_impl(b, *x, lets, local)?;
            let c = const_eval_impl(b, *y, lets, local)?;
            match op {
                BinOp::Add => a.checked_add(c),
                BinOp::Sub => a.checked_sub(c),
                BinOp::Mul => a.checked_mul(c),
                BinOp::Div => (c != 0).then(|| a.div_euclid(c)),
                BinOp::Rem => (c != 0).then(|| a.rem_euclid(c)),
                BinOp::Lt | BinOp::Le | BinOp::Eq => None,
            }
        }
        _ => None,
    }
}

/// Rebuilds `root` with every node in `map` replaced by its image.
///
/// Hash-consing makes syntactically identical subtrees a single [`NodeId`],
/// so one map entry replaces every occurrence. Untouched subtrees keep their
/// ids; binders are not renamed; replacement images are used verbatim (not
/// themselves rewritten). Callers must re-run type inference afterwards.
pub(crate) fn replace_nodes(
    b: &mut IRBuilder,
    root: NodeId,
    map: &HashMap<NodeId, NodeId>,
) -> NodeId {
    replace_nodes_impl(b, root, map, &mut HashMap::new())
}

fn replace_nodes_impl(
    b: &mut IRBuilder,
    id: NodeId,
    map: &HashMap<NodeId, NodeId>,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&r) = map.get(&id) {
        return r;
    }
    if let Some(&r) = memo.get(&id) {
        return r;
    }
    let new = match b.node(id).clone() {
        Node::Input(_)
        | Node::Var(_)
        | Node::ConstU32(_)
        | Node::ConstField(_)
        | Node::ConstFpExt(_)
        | Node::ConstSym(_) => id,
        Node::LiftFpExt(x) => {
            let x2 = replace_nodes_impl(b, x, map, memo);
            if x2 == x {
                id
            } else {
                b.intern(Node::LiftFpExt(x2))
            }
        }
        Node::Bin(op, x, y) => {
            let x2 = replace_nodes_impl(b, x, map, memo);
            let y2 = replace_nodes_impl(b, y, map, memo);
            if (x2, y2) == (x, y) {
                id
            } else {
                b.intern(Node::Bin(op, x2, y2))
            }
        }
        Node::Select {
            cond,
            then_val,
            else_val,
        } => {
            let c2 = replace_nodes_impl(b, cond, map, memo);
            let t2 = replace_nodes_impl(b, then_val, map, memo);
            let e2 = replace_nodes_impl(b, else_val, map, memo);
            if (c2, t2, e2) == (cond, then_val, else_val) {
                id
            } else {
                b.intern(Node::Select {
                    cond: c2,
                    then_val: t2,
                    else_val: e2,
                })
            }
        }
        Node::Index { tensor, indices } => {
            let t2 = replace_nodes_impl(b, tensor, map, memo);
            let ix2: Vec<NodeId> = indices
                .iter()
                .map(|&ix| replace_nodes_impl(b, ix, map, memo))
                .collect();
            if t2 == tensor && ix2 == indices {
                id
            } else {
                b.intern(Node::Index {
                    tensor: t2,
                    indices: ix2,
                })
            }
        }
        Node::Compute {
            bound,
            var,
            body,
            scatter,
            par,
            threads,
        } => {
            let body2 = replace_nodes_impl(b, body, map, memo);
            if body2 == body {
                id
            } else {
                b.intern(Node::Compute {
                    bound,
                    var,
                    body: body2,
                    scatter,
                    par,
                    threads,
                })
            }
        }
        Node::Reduce {
            op,
            bound,
            var,
            body,
        } => {
            let body2 = replace_nodes_impl(b, body, map, memo);
            if body2 == body {
                id
            } else {
                b.intern(Node::Reduce {
                    op,
                    bound,
                    var,
                    body: body2,
                })
            }
        }
        Node::Let { var, value, body } => {
            let v2 = replace_nodes_impl(b, value, map, memo);
            let b2 = replace_nodes_impl(b, body, map, memo);
            if (v2, b2) == (value, body) {
                id
            } else {
                b.intern(Node::Let {
                    var,
                    value: v2,
                    body: b2,
                })
            }
        }
        Node::Tuple(elems) => {
            let e2: Vec<NodeId> = elems
                .iter()
                .map(|&e| replace_nodes_impl(b, e, map, memo))
                .collect();
            if e2 == elems {
                id
            } else {
                b.intern(Node::Tuple(e2))
            }
        }
        Node::Proj(t, k) => {
            let t2 = replace_nodes_impl(b, t, map, memo);
            if t2 == t {
                id
            } else {
                b.intern(Node::Proj(t2, k))
            }
        }
        Node::Pack(elems) => {
            let e2: Vec<NodeId> = elems
                .iter()
                .map(|&e| replace_nodes_impl(b, e, map, memo))
                .collect();
            if e2 == elems {
                id
            } else {
                b.intern(Node::Pack(e2))
            }
        }
    };
    memo.insert(id, new);
    new
}

/// Resolves a tensor-typed HIR expression to a top-level tensor reference.
pub(crate) fn resolve_tensor_ref(
    b: &IRBuilder,
    env: &HashMap<VarId, CanonValue>,
    id: NodeId,
) -> Result<TensorRef, CompileError> {
    match b.node(id) {
        Node::Input(k) => Ok(TensorRef::Input(*k)),
        Node::Var(v) => match env.get(v) {
            Some(CanonValue::Tensors(refs)) if refs.len() == 1 => Ok(refs[0]),
            Some(CanonValue::Tensors(_)) => Err(CompileError::Lower(
                "tuple-valued variable used where a single tensor is expected".into(),
            )),
            Some(CanonValue::Scalar(_)) => Err(CompileError::Lower(
                "scalar variable indexed as a tensor".into(),
            )),
            None => Err(CompileError::Lower(format!(
                "indexed variable {v:?} is not bound to a top-level tensor; \
                 inner tensors are not supported yet"
            ))),
        },
        Node::Proj(t, k) => match b.node(*t) {
            Node::Var(v) => match env.get(v) {
                Some(CanonValue::Tensors(refs)) => refs.get(*k).copied().ok_or_else(|| {
                    CompileError::Lower(format!("projection index {k} out of bounds"))
                }),
                _ => Err(CompileError::Lower(
                    "projection from a non-tuple variable".into(),
                )),
            },
            Node::Tuple(elems) => {
                let e = *elems
                    .get(*k)
                    .ok_or_else(|| CompileError::Lower(format!("projection index {k} OOB")))?;
                resolve_tensor_ref(b, env, e)
            }
            _ => Err(CompileError::Lower(
                "projection from an unsupported expression".into(),
            )),
        },
        other => Err(CompileError::Lower(format!(
            "cannot index expression {other:?}; only module inputs and \
             top-level let results can be indexed"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BinOp;

    fn sym_of(target: VarId) -> impl Fn(VarId) -> Option<Quast> {
        move |v| (v == target).then(|| Quast::sym(target))
    }

    #[test]
    fn hir_to_quast_affine_forms() {
        let mut b = IRBuilder::new();
        let i = b.fresh_var();
        let iv = b.intern(Node::Var(i));
        let c2 = b.const_u32(2);
        let c5 = b.const_u32(5);
        let c4 = b.const_u32(4);
        let c3 = b.const_u32(3);
        let e = b.mul(iv, c2);
        let e = b.add(e, c5);
        let e = b.div(e, c4);
        let e = b.rem(e, c3);
        let q = hir_to_quast(&b, e, &sym_of(i), &|_| None).unwrap();
        let expected = Quast::sym(i)
            .mul_c(2)
            .add(&Quast::cst(5))
            .floordiv(4)
            .rem_c(3);
        assert_eq!(q, expected);
    }

    #[test]
    fn hir_to_quast_chases_lets() {
        let mut b = IRBuilder::new();
        let i = b.fresh_var();
        let iv = b.intern(Node::Var(i));
        let c4 = b.const_u32(4);
        let s_val = b.mul(iv, c4);
        // `let s = i * 4 in s + 1` inside the index expression itself.
        let c1 = b.const_u32(1);
        let expr = b.bind(s_val, |b, s| b.add(s, c1));
        let q = hir_to_quast(&b, expr, &sym_of(i), &|_| None).unwrap();
        assert_eq!(q, Quast::sym(i).mul_c(4).add(&Quast::cst(1)));

        // The same binding resolved through the external `lets` closure.
        let t = b.fresh_var();
        let tv = b.intern(Node::Var(t));
        let expr2 = b.add(tv, c1);
        let lets = |v: VarId| (v == t).then_some(s_val);
        let q2 = hir_to_quast(&b, expr2, &sym_of(i), &lets).unwrap();
        assert_eq!(q2, Quast::sym(i).mul_c(4).add(&Quast::cst(1)));
    }

    #[test]
    fn hir_to_quast_rejects_non_affine() {
        let mut b = IRBuilder::new();
        let i = b.fresh_var();
        let iv = b.intern(Node::Var(i));
        let e = b.mul(iv, iv);
        assert!(hir_to_quast(&b, e, &sym_of(i), &|_| None).is_err());
        let e = b.div(iv, iv);
        assert!(hir_to_quast(&b, e, &sym_of(i), &|_| None).is_err());
    }

    #[test]
    fn hir_to_sexpr_symbolic_forms() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let i = b.fresh_var();
        let iv = b.intern(Node::Var(i));
        let syms = |v: VarId| (v == i).then(|| SExpr::sym(i));

        // `#(n - 1) - i`: quasi-affine extraction fails, symbolic succeeds.
        let cs = b.const_sym(n - 1);
        let e = b.sub(cs, iv);
        assert!(hir_to_quast(&b, e, &sym_of(i), &|_| None).is_err());
        let se = hir_to_sexpr(&b, e, &syms, &|_| None).unwrap();
        let n_expr = SExpr::cst(SymConst::Sym(n.0));
        let expected = n_expr
            .sub(&SExpr::cst(SymConst::Lit(1)))
            .sub(&SExpr::sym(i));
        assert_eq!(se, expected);

        // A symbolic multiplication coefficient: `i * #n`.
        let cn = b.const_sym(n);
        let e2 = b.mul(iv, cn);
        let se2 = hir_to_sexpr(&b, e2, &syms, &|_| None).unwrap();
        assert_eq!(se2, SExpr::sym(i).mul_c(SymConst::Sym(n.0)));

        // A symbolic divisor is trusted; `i * i` still has no constant side.
        let e3 = b.div(iv, cn);
        assert_eq!(
            hir_to_sexpr(&b, e3, &syms, &|_| None).unwrap(),
            SExpr::sym(i).floordiv(SymConst::Sym(n.0))
        );
        let e4 = b.mul(iv, iv);
        assert!(hir_to_sexpr(&b, e4, &syms, &|_| None).is_err());
    }

    #[test]
    fn const_eval_index_folds() {
        let mut b = IRBuilder::new();
        let c2 = b.const_u32(2);
        let c3 = b.const_u32(3);
        let c4 = b.const_u32(4);
        let e = b.add(c2, c3);
        let e = b.mul(e, c4);
        assert_eq!(const_eval_index(&b, e, &|_| None), Some(20));
        let i = b.fresh_var();
        let iv = b.intern(Node::Var(i));
        assert_eq!(const_eval_index(&b, iv, &|_| None), None);
    }

    #[test]
    fn replace_nodes_replaces_all_occurrences() {
        let mut b = IRBuilder::new();
        let one = b.const_u32(1);
        let c7 = b.const_u32(7);
        // Hash-consing: both operands are the same NodeId.
        let sum = b.add(one, one);
        let outer = b.mul(sum, c7);

        let map = HashMap::from([(one, b.const_u32(2))]);
        let two = b.const_u32(2);
        let r = replace_nodes(&mut b, outer, &map);
        let Node::Bin(BinOp::Mul, s2, k) = *b.node(r) else {
            panic!("expected Mul, got {:?}", b.node(r));
        };
        assert_eq!(*b.node(s2), Node::Bin(BinOp::Add, two, two));
        assert_eq!(k, c7); // untouched subtree keeps its id

        // Empty map is the identity.
        assert_eq!(replace_nodes(&mut b, outer, &HashMap::new()), outer);
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use crate::{
        ir::Module,
        kernel_ir::{Kernel, KirProgram, SSAOpCode},
        passes::{canonicalize, lower_to_kir, type_infer},
    };

    /// Runs the HIR passes: type inference, canonicalization and lowering.
    pub(crate) fn lowered(module: Module) -> KirProgram {
        let types = type_infer(&module).unwrap();
        let program = canonicalize(module, types).unwrap();
        lower_to_kir(&program).unwrap()
    }

    pub(crate) fn stmt_kinds(kernel: &Kernel) -> Vec<&'static str> {
        kernel
            .grid
            .block
            .body
            .iter()
            .map(|&id| match kernel.op(id).opcode {
                SSAOpCode::Alloc { .. } => "alloc",
                SSAOpCode::Par { .. } => "par",
                SSAOpCode::Loop { .. } => "loop",
                SSAOpCode::Sync => "sync",
                SSAOpCode::ConvertLayout { .. } => "convert",
                _ => "scalar",
            })
            .collect()
    }
}
