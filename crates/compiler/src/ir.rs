//! High-level functional IR.
//!
//! Programs are pure expression DAGs built through [`IRBuilder`]. The two core
//! primitives are `compute` (parallel map) and `reduce` (parallel associative
//! reduction), plus `let` bindings, `if`/select, tensor indexing, tuples, small
//! array literals (`pack`) and elementwise scalar ops.
//!
//! Nodes are hash-consed: structurally identical pure expressions share a
//! `NodeId`, which gives CSE for free downstream.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::{
    quast::{Quast, Scatter},
    CompileError,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub(crate) u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// BabyBear field element, canonical `u32` representation.
    BabyBear,
    U32,
    Bool,
}

pub type Shape = Vec<usize>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Scalar(ScalarType),
    Tensor(ScalarType, Shape),
    Tuple(Vec<Type>),
}

impl Type {
    pub fn scalar_type(&self) -> Option<ScalarType> {
        match self {
            Type::Scalar(s) | Type::Tensor(s, _) => Some(*s),
            Type::Tuple(_) => None,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Type::Scalar(_) => &[],
            Type::Tensor(_, shape) => shape,
            Type::Tuple(_) => panic!("tuple type has no shape"),
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// Field or u32 addition.
    Add,
    /// Field or u32 subtraction.
    Sub,
    /// Field or u32 multiplication.
    Mul,
    /// Integer division (u32 only).
    Div,
    /// Integer remainder (u32 only).
    Rem,
    /// u32 comparison, result is Bool.
    Lt,
    Le,
    Eq,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// Reference to module input `k`.
    Input(usize),
    /// Reference to a bound variable (loop index or let binding).
    Var(VarId),
    ConstU32(u32),
    /// BabyBear constant, canonical representation.
    ConstField(u32),
    Bin(BinOp, NodeId, NodeId),
    Select {
        cond: NodeId,
        then_val: NodeId,
        else_val: NodeId,
    },
    /// Full indexing of a tensor down to a scalar: `tensor[i, j, ...]`.
    Index {
        tensor: NodeId,
        indices: Vec<NodeId>,
    },
    /// Parallel map: `compute [bound] |var| { body }`. An optional
    /// [`Scatter`] stores results through a bijective quasi-affine map from
    /// logical to physical coordinates.
    Compute {
        bound: usize,
        var: VarId,
        body: NodeId,
        scatter: Option<Box<Scatter>>,
    },
    /// Parallel associative reduction: `reduce [bound] |var| { body }`.
    Reduce {
        op: ReduceOp,
        bound: usize,
        var: VarId,
        body: NodeId,
    },
    /// `let var = value in body`.
    Let {
        var: VarId,
        value: NodeId,
        body: NodeId,
    },
    Tuple(Vec<NodeId>),
    /// Tuple projection.
    Proj(NodeId, usize),
    /// Array literal from scalars of equal type; has type `T[k]`.
    Pack(Vec<NodeId>),
}

#[derive(Clone, Debug)]
pub struct InputDecl {
    pub name: String,
    pub elem: ScalarType,
    pub shape: Shape,
}

/// Arena of hash-consed IR nodes plus module-level input declarations.
#[derive(Default)]
pub struct IRBuilder {
    nodes: Vec<Node>,
    dedup: FxHashMap<Node, NodeId>,
    next_var: u32,
    inputs: Vec<InputDecl>,
    pending_lets: Vec<(VarId, NodeId)>,
}

impl IRBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Identity reborrow used by the `kernel!` expansion: method autoref
    /// makes the macro's context work for both owned `mut` builders and
    /// (non-`mut`) `&mut IRBuilder` bindings.
    pub fn as_builder_mut(&mut self) -> &mut Self {
        self
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn inputs(&self) -> &[InputDecl] {
        &self.inputs
    }

    pub(crate) fn intern(&mut self, node: Node) -> NodeId {
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.dedup.insert(node, id);
        id
    }

    pub(crate) fn fresh_var(&mut self) -> VarId {
        let v = VarId(self.next_var);
        self.next_var += 1;
        v
    }

    /// Declares a module input tensor (scalar input if `shape` is empty).
    pub fn input(
        &mut self,
        name: impl Into<String>,
        elem: ScalarType,
        shape: impl Into<Shape>,
    ) -> NodeId {
        let k = self.inputs.len();
        self.inputs.push(InputDecl {
            name: name.into(),
            elem,
            shape: shape.into(),
        });
        self.intern(Node::Input(k))
    }

    pub fn const_u32(&mut self, v: u32) -> NodeId {
        self.intern(Node::ConstU32(v))
    }

    /// BabyBear constant from its canonical `u32` representation.
    pub fn const_field(&mut self, v: u32) -> NodeId {
        self.intern(Node::ConstField(v))
    }

    pub fn bin(&mut self, op: BinOp, a: NodeId, b: NodeId) -> NodeId {
        self.intern(Node::Bin(op, a, b))
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Add, a, b)
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Sub, a, b)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Mul, a, b)
    }

    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Div, a, b)
    }

    pub fn rem(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Rem, a, b)
    }

    pub fn lt(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Lt, a, b)
    }

    pub fn le(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Le, a, b)
    }

    pub fn eq(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinOp::Eq, a, b)
    }

    /// `if cond then a else b` on scalars.
    pub fn select(&mut self, cond: NodeId, then_val: NodeId, else_val: NodeId) -> NodeId {
        self.intern(Node::Select {
            cond,
            then_val,
            else_val,
        })
    }

    /// Full tensor indexing: `tensor[indices...]`, yields a scalar.
    pub fn index(&mut self, tensor: NodeId, indices: &[NodeId]) -> NodeId {
        self.intern(Node::Index {
            tensor,
            indices: indices.to_vec(),
        })
    }

    /// `compute [bound] |i| { f(i) }`
    pub fn compute(&mut self, bound: usize, f: impl FnOnce(&mut Self, NodeId) -> NodeId) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Compute {
            bound,
            var,
            body,
            scatter: None,
        })
    }

    /// `#[scatter(...)] compute [bound] |i| { f(i) }`: results are stored
    /// through the bijective quasi-affine `scatter` map (see [`Scatter`]).
    pub fn compute_scatter(
        &mut self,
        bound: usize,
        scatter: Scatter,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Compute {
            bound,
            var,
            body,
            scatter: Some(Box::new(scatter)),
        })
    }

    /// Builds a [`Scatter`]: allocates `n_params` fresh symbols and passes
    /// them to `f` together with a constant constructor ([`Quast::cst`]) to
    /// build the physical coordinate expressions. `out_shape` is required
    /// when the map changes the number of dimensions.
    pub fn scatter_map(
        &mut self,
        n_params: usize,
        out_shape: Option<Vec<usize>>,
        f: impl FnOnce(&[Quast], fn(i64) -> Quast) -> Vec<Quast>,
    ) -> Scatter {
        let params: Vec<VarId> = (0..n_params).map(|_| self.fresh_var()).collect();
        let syms: Vec<Quast> = params.iter().map(|&p| Quast::sym(p)).collect();
        let exprs = f(&syms, Quast::cst);
        Scatter {
            params,
            exprs,
            out_shape,
            bounds: Default::default(),
        }
    }

    /// `reduce [bound] |i| { f(i) }` with the given associative operator.
    pub fn reduce(
        &mut self,
        op: ReduceOp,
        bound: usize,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Reduce {
            op,
            bound,
            var,
            body,
        })
    }

    pub fn reduce_add(
        &mut self,
        bound: usize,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        self.reduce(ReduceOp::Add, bound, f)
    }

    /// `let v = value in f(v)`
    pub fn bind(&mut self, value: NodeId, f: impl FnOnce(&mut Self, NodeId) -> NodeId) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Let { var, value, body })
    }

    /// Binds `value` to a fresh top-level variable and returns a reference to
    /// it. The `let` chain is materialized (in binding order) around the body
    /// passed to [`IRBuilder::finish`]. This enables building sequential
    /// pipelines with ordinary Rust loops.
    pub fn let_bound(&mut self, value: NodeId) -> NodeId {
        let var = self.fresh_var();
        self.pending_lets.push((var, value));
        self.intern(Node::Var(var))
    }

    pub fn tuple(&mut self, elems: &[NodeId]) -> NodeId {
        self.intern(Node::Tuple(elems.to_vec()))
    }

    pub fn proj(&mut self, tuple: NodeId, k: usize) -> NodeId {
        self.intern(Node::Proj(tuple, k))
    }

    /// Array literal `[e_0, ..., e_{k-1}]` from scalars of equal type.
    pub fn pack(&mut self, elems: &[NodeId]) -> NodeId {
        self.intern(Node::Pack(elems.to_vec()))
    }

    /// Finalizes the module: `body` is the expression whose value is the
    /// module output (a tensor or a tuple of tensors). Any [`let_bound`]
    /// bindings are folded as a `let` chain around `body`.
    ///
    /// [`let_bound`]: IRBuilder::let_bound
    pub fn finish(mut self, name: impl Into<String>, body: NodeId) -> Module {
        let mut body = body;
        for (var, value) in std::mem::take(&mut self.pending_lets).into_iter().rev() {
            body = self.intern(Node::Let { var, value, body });
        }
        Module {
            name: name.into(),
            builder: self,
            body,
        }
    }
}

/// A complete kernel module: declared inputs and the expression that
/// represents the entire sequence of computations.
pub struct Module {
    pub name: String,
    pub builder: IRBuilder,
    pub body: NodeId,
}

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

pub fn infer_types(module: &Module) -> Result<TypeMap, CompileError> {
    let mut cx = TypeCx {
        b: &module.builder,
        types: HashMap::new(),
        var_types: HashMap::new(),
    };
    cx.infer(module.body)?;
    Ok(TypeMap { types: cx.types })
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
