//! Type inference and checking over HIR [`Module`]s.

use std::collections::HashMap;

use crate::{
    ir::{BinOp, IRBuilder, Module, Node, NodeId, ScalarType, Type, VarId},
    quast::Scatter,
    CompileError,
};

/// Bottom-up type assignment for every reachable node in a module.
///
/// Canonicalization rewrites (which preserve types) extend this map for the
/// nodes they create, so it stays total over all live nodes.
pub struct TypeMap {
    types: HashMap<NodeId, Type>,
}

impl TypeMap {
    pub fn get(&self, id: NodeId) -> &Type {
        &self.types[&id]
    }

    pub fn try_get(&self, id: NodeId) -> Option<&Type> {
        self.types.get(&id)
    }

    pub(crate) fn insert(&mut self, id: NodeId, ty: Type) {
        self.types.insert(id, ty);
    }
}

/// Infers the type of every reachable node of the module.
pub fn type_infer(module: &Module) -> Result<TypeMap, CompileError> {
    let mut cx = TypeCx {
        b: &module.builder,
        types: HashMap::new(),
        var_types: HashMap::new(),
    };
    cx.infer(module.body)?;
    Ok(TypeMap { types: cx.types })
}

/// Checks that the module is well-typed, discarding the typing info.
pub fn type_check(module: &Module) -> Result<(), CompileError> {
    type_infer(module).map(|_| ())
}

struct TypeCx<'a> {
    b: &'a IRBuilder,
    types: HashMap<NodeId, Type>,
    var_types: HashMap<VarId, Type>,
}

impl TypeCx<'_> {
    fn infer(&mut self, id: NodeId) -> Result<Type, CompileError> {
        if let Some(t) = self.types.get(&id) {
            return Ok(t.clone());
        }
        let ty = self.infer_uncached(id)?;
        self.types.insert(id, ty.clone());
        Ok(ty)
    }

    fn scalar_of(&mut self, id: NodeId, what: &str) -> Result<ScalarType, CompileError> {
        match self.infer(id)? {
            Type::Scalar(s) => Ok(s),
            other => Err(CompileError::Type(format!(
                "{what} must be a scalar, got {other:?}"
            ))),
        }
    }

    fn infer_uncached(&mut self, id: NodeId) -> Result<Type, CompileError> {
        let node = self.b.node(id).clone();
        match node {
            Node::Input(k) => {
                let decl = &self.b.inputs()[k];
                Ok(if decl.shape.is_empty() {
                    Type::Scalar(decl.elem)
                } else {
                    Type::Tensor(decl.elem, decl.shape.clone())
                })
            }
            Node::Var(v) => self
                .var_types
                .get(&v)
                .cloned()
                .ok_or_else(|| CompileError::Type(format!("unbound variable {v:?}"))),
            Node::ConstU32(_) => Ok(Type::Scalar(ScalarType::U32)),
            Node::ConstField(_) => Ok(Type::Scalar(ScalarType::BabyBear)),
            Node::Bin(op, a, b) => {
                let ta = self.scalar_of(a, "binary operand")?;
                let tb = self.scalar_of(b, "binary operand")?;
                if ta != tb {
                    return Err(CompileError::Type(format!(
                        "binary op {op:?} operand type mismatch: {ta:?} vs {tb:?}"
                    )));
                }
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul => {
                        if ta == ScalarType::Bool {
                            return Err(CompileError::Type(format!(
                                "arithmetic op {op:?} not supported on Bool"
                            )));
                        }
                        Ok(Type::Scalar(ta))
                    }
                    BinOp::Div | BinOp::Rem => {
                        if ta != ScalarType::U32 {
                            return Err(CompileError::Type(format!(
                                "{op:?} is only supported on U32, got {ta:?}"
                            )));
                        }
                        Ok(Type::Scalar(ScalarType::U32))
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Eq => {
                        if ta != ScalarType::U32 {
                            return Err(CompileError::Type(format!(
                                "comparison {op:?} is only supported on U32, got {ta:?}"
                            )));
                        }
                        Ok(Type::Scalar(ScalarType::Bool))
                    }
                }
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let tc = self.scalar_of(cond, "select condition")?;
                if tc != ScalarType::Bool {
                    return Err(CompileError::Type(format!(
                        "select condition must be Bool, got {tc:?}"
                    )));
                }
                let tt = self.scalar_of(then_val, "select branch")?;
                let te = self.scalar_of(else_val, "select branch")?;
                if tt != te {
                    return Err(CompileError::Type(format!(
                        "select branch type mismatch: {tt:?} vs {te:?}"
                    )));
                }
                Ok(Type::Scalar(tt))
            }
            Node::Index { tensor, indices } => {
                let tt = self.infer(tensor)?;
                let Type::Tensor(elem, shape) = tt else {
                    return Err(CompileError::Type(format!(
                        "indexing a non-tensor of type {tt:?}"
                    )));
                };
                if indices.len() != shape.len() {
                    return Err(CompileError::Type(format!(
                        "index arity {} does not match tensor rank {}",
                        indices.len(),
                        shape.len()
                    )));
                }
                for &ix in &indices {
                    let ti = self.scalar_of(ix, "index")?;
                    if ti != ScalarType::U32 {
                        return Err(CompileError::Type(format!(
                            "indices must be U32, got {ti:?}"
                        )));
                    }
                }
                Ok(Type::Scalar(elem))
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
            } => {
                self.var_types.insert(var, Type::Scalar(ScalarType::U32));
                let tb = self.infer(body)?;
                let ty = lift_compute_type(bound, &tb)?;
                match scatter {
                    None => Ok(ty),
                    Some(sc) => scatter_result_type(&sc, &ty),
                }
            }
            Node::Reduce {
                bound: _,
                var,
                body,
                ..
            } => {
                self.var_types.insert(var, Type::Scalar(ScalarType::U32));
                let tb = self.infer(body)?;
                match tb {
                    Type::Scalar(s) if s != ScalarType::Bool => Ok(Type::Scalar(s)),
                    other => Err(CompileError::Type(format!(
                        "reduce body must be an arithmetic scalar, got {other:?}"
                    ))),
                }
            }
            Node::Let { var, value, body } => {
                let tv = self.infer(value)?;
                self.var_types.insert(var, tv);
                self.infer(body)
            }
            Node::Tuple(elems) => {
                let ts = elems
                    .iter()
                    .map(|&e| self.infer(e))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Type::Tuple(ts))
            }
            Node::Proj(tuple, k) => {
                let tt = self.infer(tuple)?;
                let Type::Tuple(ts) = tt else {
                    return Err(CompileError::Type(format!(
                        "projection from non-tuple of type {tt:?}"
                    )));
                };
                ts.get(k).cloned().ok_or_else(|| {
                    CompileError::Type(format!(
                        "projection index {k} out of bounds for tuple of size {}",
                        ts.len()
                    ))
                })
            }
            Node::Pack(elems) => {
                if elems.is_empty() {
                    return Err(CompileError::Type("empty pack".into()));
                }
                let t0 = self.scalar_of(elems[0], "pack element")?;
                for &e in &elems[1..] {
                    let t = self.scalar_of(e, "pack element")?;
                    if t != t0 {
                        return Err(CompileError::Type(format!(
                            "pack element type mismatch: {t0:?} vs {t:?}"
                        )));
                    }
                }
                Ok(Type::Tensor(t0, vec![elems.len()]))
            }
        }
    }
}

/// Type of a compute with a `#[scatter(...)]` attribute, given its logical
/// (pre-scatter) type: the physical shape replaces the logical one.
fn scatter_result_type(sc: &Scatter, logical_ty: &Type) -> Result<Type, CompileError> {
    let Type::Tensor(elem, logical) = logical_ty else {
        return Err(CompileError::Type(format!(
            "scatter requires a tensor-valued compute, got {logical_ty:?}"
        )));
    };
    if sc.params.len() != logical.len() {
        return Err(CompileError::Type(format!(
            "scatter has {} parameters but the compute output has rank {}",
            sc.params.len(),
            logical.len()
        )));
    }
    let out = sc.out_shape_for(logical).map_err(|e| match e {
        CompileError::Quast(m) => CompileError::Type(m),
        other => other,
    })?;
    let n_log: usize = logical.iter().product();
    let n_out: usize = out.iter().product();
    if n_log != n_out {
        return Err(CompileError::Type(format!(
            "scatter output shape {out:?} has {n_out} elements but the logical \
             shape {logical:?} has {n_log}"
        )));
    }
    Ok(Type::Tensor(*elem, out))
}

/// Type of `compute [bound] |i| { body }` given the body type: prepends the
/// iteration bound to tensor shapes, distributing over tuples.
fn lift_compute_type(bound: usize, body_ty: &Type) -> Result<Type, CompileError> {
    match body_ty {
        Type::Scalar(s) => Ok(Type::Tensor(*s, vec![bound])),
        Type::Tensor(s, shape) => {
            let mut new_shape = Vec::with_capacity(shape.len() + 1);
            new_shape.push(bound);
            new_shape.extend_from_slice(shape);
            Ok(Type::Tensor(*s, new_shape))
        }
        Type::Tuple(ts) => {
            let lifted = ts
                .iter()
                .map(|t| match t {
                    Type::Tuple(_) => Err(CompileError::Type(
                        "nested tuples in compute results are not supported".into(),
                    )),
                    other => lift_compute_type(bound, other),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Type::Tuple(lifted))
        }
    }
}
