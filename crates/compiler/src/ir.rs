//! High-level functional IR.
//!
//! Programs are pure expression DAGs built through [`IRBuilder`]. The two core
//! primitives are `compute` (parallel map) and `reduce` (parallel associative
//! reduction), plus `let` bindings, `if`/select, tensor indexing, tuples, small
//! array literals (`pack`) and elementwise scalar ops.
//!
//! Nodes are hash-consed: structurally identical pure expressions share a
//! `NodeId`, which gives CSE for free downstream.

use rustc_hash::FxHashMap;

use crate::quast::{ParSpec, Quast, Scatter};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub(crate) u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub(crate) u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// BabyBear field element, canonical `u32` representation.
    BabyBear,
    /// Degree-4 binomial extension of BabyBear over `x^4 - 11`. Each
    /// element is four canonical BabyBear coefficients laid out as
    /// `(a0, a1, a2, a3)` for the polynomial `a0 + a1 x + a2 x^2 +
    /// a3 x^3`, packed to 16 bytes so a single element fits an `LDG.128`
    /// / `STG.128` instruction.
    FpExt,
    U32,
    Bool,
}

impl ScalarType {
    /// Size in bytes of one scalar element in a global or shared buffer.
    pub fn size_bytes(self) -> usize {
        match self {
            ScalarType::FpExt => 16,
            ScalarType::BabyBear | ScalarType::U32 | ScalarType::Bool => 4,
        }
    }
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
    /// FpExt constant `a0 + a1 x + a2 x^2 + a3 x^3` — each coefficient is
    /// a canonical BabyBear `u32`.
    ConstFpExt([u32; 4]),
    /// Lift a `BabyBear` value to `FpExt`: `x -> (x, 0, 0, 0)`.
    LiftFpExt(NodeId),
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
    /// logical to physical coordinates. An optional [`ParSpec`] assigns
    /// logical indices to physical (thread, seq) coordinates, and `threads`
    /// (`#[grid(threads = N)]`) overrides the kernel's block size.
    Compute {
        bound: usize,
        var: VarId,
        body: NodeId,
        scatter: Option<Box<Scatter>>,
        par: Option<Box<ParSpec>>,
        threads: Option<usize>,
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

    /// FpExt constant from four canonical BabyBear `u32` coefficients:
    /// `coeffs[0] + coeffs[1] x + coeffs[2] x^2 + coeffs[3] x^3`.
    pub fn const_fpext(&mut self, coeffs: [u32; 4]) -> NodeId {
        self.intern(Node::ConstFpExt(coeffs))
    }

    /// Lift a BabyBear-typed value to FpExt as `(x, 0, 0, 0)`.
    pub fn lift_fpext(&mut self, x: NodeId) -> NodeId {
        self.intern(Node::LiftFpExt(x))
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
        self.compute_with(bound, None, None, None, f)
    }

    /// `#[scatter(...)] compute [bound] |i| { f(i) }`: results are stored
    /// through the bijective quasi-affine `scatter` map (see [`Scatter`]).
    pub fn compute_scatter(
        &mut self,
        bound: usize,
        scatter: Scatter,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        self.compute_with(bound, Some(scatter), None, None, f)
    }

    /// `compute` with any combination of attributes: a [`Scatter`] store
    /// map, a [`ParSpec`] compute layout and a `#[grid(threads = N)]`
    /// block-size hint.
    pub fn compute_with(
        &mut self,
        bound: usize,
        scatter: Option<Scatter>,
        par: Option<ParSpec>,
        threads: Option<usize>,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Compute {
            bound,
            var,
            body,
            scatter: scatter.map(Box::new),
            par: par.map(Box::new),
            threads,
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

    /// Builds a [`ParSpec`]: allocates the thread and seq symbols (thread
    /// first, so it packs into the low bits of the physical index) and
    /// passes them to `f` together with a constant constructor
    /// ([`Quast::cst`]) to build the logical-index expression.
    pub fn par_map(
        &mut self,
        f: impl FnOnce(&Quast, &Quast, fn(i64) -> Quast) -> Quast,
    ) -> ParSpec {
        let thread = self.fresh_var();
        let seq = self.fresh_var();
        let expr = f(&Quast::sym(thread), &Quast::sym(seq), Quast::cst);
        ParSpec { thread, seq, expr }
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
