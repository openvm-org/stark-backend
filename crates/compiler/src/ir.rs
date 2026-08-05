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

use crate::quast::{ParSpec, Quast, SExpr, Scatter, SymConst};

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

/// A (possibly symbolic) size: a shape dimension or an iteration bound.
/// Loop-var `Sym` nodes never appear in sizes — only literals and module
/// parameters (`SymConst` positions).
pub type SizeExpr = SExpr;

pub type Shape = Vec<SizeExpr>;

/// Host-side handle to a module parameter declared with
/// [`IRBuilder::symbol`]. `Copy`, so it can be spliced (`#n`, `#(n + 2)`)
/// and reused freely; arithmetic on it yields a [`SymExpr`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sym(pub(crate) VarId);

/// A symbolic constant expression over [`Sym`] handles and integer
/// literals, built with ordinary Rust operators (`n + 2`, `n * 4`, `n / 2`,
/// `n - 1`). Multiplication is only defined by an integer or a plain `Sym`
/// ([`SExpr`] cannot represent a product of two compound expressions).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymExpr(pub(crate) SExpr);

impl From<Sym> for SymExpr {
    fn from(s: Sym) -> Self {
        SymExpr(SExpr::cst(SymConst::Sym(s.0)))
    }
}

impl From<Sym> for SizeExpr {
    fn from(s: Sym) -> Self {
        SExpr::cst(SymConst::Sym(s.0))
    }
}

impl From<SymExpr> for SizeExpr {
    fn from(e: SymExpr) -> Self {
        e.0
    }
}

macro_rules! impl_sym_ops {
    ($($lhs:ty),*) => {$(
        impl std::ops::Add<i64> for $lhs {
            type Output = SymExpr;
            fn add(self, c: i64) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.add(&SExpr::cst(SymConst::Lit(c))))
            }
        }

        impl std::ops::Sub<i64> for $lhs {
            type Output = SymExpr;
            fn sub(self, c: i64) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.sub(&SExpr::cst(SymConst::Lit(c))))
            }
        }

        impl std::ops::Mul<i64> for $lhs {
            type Output = SymExpr;
            fn mul(self, c: i64) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.mul_c(SymConst::Lit(c)))
            }
        }

        /// Floor division by a positive constant.
        impl std::ops::Div<i64> for $lhs {
            type Output = SymExpr;
            fn div(self, c: i64) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.floordiv(SymConst::Lit(c)))
            }
        }

        impl std::ops::Rem<i64> for $lhs {
            type Output = SymExpr;
            fn rem(self, c: i64) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.rem_c(SymConst::Lit(c)))
            }
        }

        impl std::ops::Add<Sym> for $lhs {
            type Output = SymExpr;
            fn add(self, s: Sym) -> SymExpr {
                let e: SymExpr = self.into();
                let r: SymExpr = s.into();
                SymExpr(e.0.add(&r.0))
            }
        }

        impl std::ops::Sub<Sym> for $lhs {
            type Output = SymExpr;
            fn sub(self, s: Sym) -> SymExpr {
                let e: SymExpr = self.into();
                let r: SymExpr = s.into();
                SymExpr(e.0.sub(&r.0))
            }
        }

        impl std::ops::Mul<Sym> for $lhs {
            type Output = SymExpr;
            fn mul(self, s: Sym) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.mul_c(SymConst::Sym(s.0)))
            }
        }

        impl std::ops::Add<SymExpr> for $lhs {
            type Output = SymExpr;
            fn add(self, r: SymExpr) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.add(&r.0))
            }
        }

        impl std::ops::Sub<SymExpr> for $lhs {
            type Output = SymExpr;
            fn sub(self, r: SymExpr) -> SymExpr {
                let e: SymExpr = self.into();
                SymExpr(e.0.sub(&r.0))
            }
        }
    )*};
}

impl_sym_ops!(Sym, SymExpr);

/// A value accepted by a `#x` / `#(expr)` splice in `kernel!`: a concrete
/// integer (interned as a `u32` constant) or a symbolic parameter
/// expression (interned as [`Node::ConstSym`]).
///
/// `usize` is the only integer impl so bare integer literals in splices
/// keep inferring (see the `From<usize> for SExpr` note in `quast.rs`).
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be spliced into a kernel with `#(..)`",
    note = "splices accept concrete `usize` values and symbolic `Sym`/`SymExpr` parameters \
            (declared with `IRBuilder::symbol`)"
)]
pub trait IntoDslConst {
    fn into_dsl_const(self, b: &mut IRBuilder) -> NodeId;
}

impl IntoDslConst for usize {
    fn into_dsl_const(self, b: &mut IRBuilder) -> NodeId {
        b.const_u32(self as u32)
    }
}

impl IntoDslConst for Sym {
    fn into_dsl_const(self, b: &mut IRBuilder) -> NodeId {
        b.const_sym(self)
    }
}

impl IntoDslConst for SymExpr {
    fn into_dsl_const(self, b: &mut IRBuilder) -> NodeId {
        b.const_sym(self)
    }
}

/// A `kernel!` position that must be a concrete integer: the
/// `#[grid(threads = N)]` block-size hint, `#[par]`/`#[scatter]` map
/// splices and scatter output bounds. Symbolic parameters are rejected
/// here because these values shape the compiled artifact itself.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not allowed here: this position must be a concrete integer",
    note = "`#[grid(threads = ..)]` hints, `#[par]`/`#[scatter]` map splices and scatter \
            bounds cannot be symbolic"
)]
pub trait IntoDslConcrete {
    fn into_dsl_concrete(self) -> usize;
}

impl IntoDslConcrete for usize {
    fn into_dsl_concrete(self) -> usize {
        self
    }
}

/// Conversion into a (possibly symbolic) shape, so concrete call sites keep
/// passing `vec![4]` / `[4, 8]` while symbolic ones pass `vec![n]` or
/// `vec![(n * 2).into()]`.
pub trait IntoShape {
    fn into_shape(self) -> Shape;
}

impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}

impl<T: Into<SizeExpr>, const N: usize> IntoShape for [T; N] {
    fn into_shape(self) -> Shape {
        self.into_iter().map(Into::into).collect()
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Shape {
        self.into_iter()
            .map(|d| SizeExpr::cst(SymConst::Lit(d as i64)))
            .collect()
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Shape {
        self.iter()
            .map(|&d| SizeExpr::cst(SymConst::Lit(d as i64)))
            .collect()
    }
}

impl IntoShape for Vec<Sym> {
    fn into_shape(self) -> Shape {
        self.into_iter().map(Into::into).collect()
    }
}

impl IntoShape for Vec<SymExpr> {
    fn into_shape(self) -> Shape {
        self.into_iter().map(Into::into).collect()
    }
}

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

    pub fn shape(&self) -> &[SizeExpr] {
        match self {
            Type::Scalar(_) => &[],
            Type::Tensor(_, shape) => shape,
            Type::Tuple(_) => panic!("tuple type has no shape"),
        }
    }

    /// Concrete shape; `None` if any dimension is symbolic.
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        self.shape()
            .iter()
            .map(|d| d.as_const().map(|c| c as usize))
            .collect()
    }

    /// Concrete element count; `None` if any dimension is symbolic.
    pub fn num_elements(&self) -> Option<usize> {
        self.shape()
            .iter()
            .try_fold(1usize, |acc, d| Some(acc * d.as_const()? as usize))
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
    /// Symbolic constant over module parameters (a `#n` / `#(n - 1)`
    /// splice used as a value). Typed `U32`; resolved to a literal at
    /// monomorphization.
    ConstSym(SExpr),
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
        bound: SizeExpr,
        var: VarId,
        body: NodeId,
        scatter: Option<Box<Scatter>>,
        par: Option<Box<ParSpec>>,
        threads: Option<usize>,
    },
    /// Parallel associative reduction: `reduce [bound] |var| { body }`.
    Reduce {
        op: ReduceOp,
        bound: SizeExpr,
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
#[derive(Clone, Default)]
pub struct IRBuilder {
    nodes: Vec<Node>,
    dedup: FxHashMap<Node, NodeId>,
    next_var: u32,
    inputs: Vec<InputDecl>,
    pending_lets: Vec<(VarId, NodeId)>,
    params: Vec<(VarId, String)>,
    shape_hint: Option<Vec<i64>>,
    block_hint: Option<usize>,
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

    /// One past the highest [`VarId`] allocated so far. Together with
    /// [`Self::raise_var_watermark`] this lets passes copy nodes across
    /// builders with `VarId`s preserved verbatim, without later
    /// [`Self::fresh_var`] calls in the destination colliding with them.
    pub(crate) fn var_watermark(&self) -> u32 {
        self.next_var
    }

    /// Raises the fresh-variable watermark to at least `n` (no-op if the
    /// builder is already past it).
    pub(crate) fn raise_var_watermark(&mut self, n: u32) {
        self.next_var = self.next_var.max(n);
    }

    /// Removes input declaration `pos`, shifting later declarations down.
    /// The caller must have already renumbered every `Node::Input` reference
    /// past `pos` that is reachable from the module body; stale `Node::Input`
    /// nodes may remain in the arena as long as they are unreachable.
    pub(crate) fn remove_input_decl(&mut self, pos: usize) {
        self.inputs.remove(pos);
    }

    /// Declares a module input tensor (scalar input if `shape` is empty).
    pub fn input(
        &mut self,
        name: impl Into<String>,
        elem: ScalarType,
        shape: impl IntoShape,
    ) -> NodeId {
        let k = self.inputs.len();
        self.inputs.push(InputDecl {
            name: name.into(),
            elem,
            shape: shape.into_shape(),
        });
        self.intern(Node::Input(k))
    }

    /// Declares a symbolic module parameter (`b.symbol("n")`): a named
    /// constant that resolves to a concrete value per kernel instantiation.
    /// The returned handle splices anywhere a constant is allowed.
    pub fn symbol(&mut self, name: impl Into<String>) -> Sym {
        let v = self.fresh_var();
        self.params.push((v, name.into()));
        Sym(v)
    }

    /// Declared symbolic parameters, in declaration order.
    pub fn params(&self) -> &[(VarId, String)] {
        &self.params
    }

    /// Re-declares a parameter copied from another builder with a caller
    /// -chosen `VarId` (preserved verbatim or pre-remapped). The caller is
    /// responsible for keeping the watermark past `v`.
    pub(crate) fn inherit_param(&mut self, v: VarId, name: String) {
        self.params.push((v, name));
    }

    /// Records a canonical concrete instantiation of the module's
    /// parameters (in declaration order): used for access checking and, for
    /// stand-alone kernels, as the monomorphization values when no
    /// graph-derived bindings exist. At most one hint per module; declare
    /// all symbols before adding it.
    ///
    /// Not part of the module-author API: hints are insertion-time data,
    /// supplied via [`crate::graph_ir::GraphBuilder::insert_kernel`]'s
    /// `shape_hint` argument (or the `*_with_hint` standalone compile entry
    /// points), which attach them here internally.
    pub(crate) fn add_shape_hint(&mut self, values: &[i64]) {
        assert!(
            self.shape_hint.is_none(),
            "at most one shape hint per module"
        );
        assert_eq!(
            values.len(),
            self.params.len(),
            "shape hint has {} values but the module declares {} parameters",
            values.len(),
            self.params.len()
        );
        self.shape_hint = Some(values.to_vec());
    }

    /// The shape hint, parallel to [`Self::params`], if one was added.
    pub fn shape_hint(&self) -> Option<&[i64]> {
        self.shape_hint.as_deref()
    }

    /// Appends a value for a param declared *after* the hint was recorded
    /// (fusion appends producer params to a merged module); keeps the hint
    /// parallel to [`Self::params`]. No-op without a hint.
    pub(crate) fn extend_shape_hint(&mut self, value: i64) {
        if let Some(h) = &mut self.shape_hint {
            h.push(value);
        }
    }

    /// Fixes the CUDA block size used by kernels whose outer bound stays
    /// symbolic after monomorphization (author `threads = ...` attributes
    /// still win). Set by the graph compiler from the block-size policy
    /// over a node's concrete bindings; part of [`crate::module_hash`], so
    /// modules that differ only in block size compile as distinct variants.
    pub fn set_block_hint(&mut self, block: usize) {
        assert!(
            (1..=1024).contains(&block),
            "block hint must be in 1..=1024, got {block}"
        );
        self.block_hint = Some(block);
    }

    /// The block-size hint, if one was set.
    pub fn block_hint(&self) -> Option<usize> {
        self.block_hint
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

    /// Symbolic `u32` constant over module parameters (`#n`, `#(n - 1)`).
    pub fn const_sym(&mut self, e: impl Into<SizeExpr>) -> NodeId {
        self.intern(Node::ConstSym(e.into()))
    }

    /// `#x` / `#(expr)` splice entry point used by `kernel!`: concrete
    /// values intern a [`Node::ConstU32`], symbolic ones a
    /// [`Node::ConstSym`].
    pub fn dsl_const(&mut self, v: impl IntoDslConst) -> NodeId {
        v.into_dsl_const(self)
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

    /// `x & c` for a compile-time-known low-bit mask `c = (1 << k) - 1`.
    /// Lowered to `x % (1 << k)` so it stays quasi-affine. `c == 0` folds to
    /// the constant `0`; the full-`u32` mask folds to `x`. Panics on any
    /// other constant (arbitrary masks have no quasi-affine equivalent).
    pub fn and(&mut self, x: NodeId, c: usize) -> NodeId {
        let c = c as u32;
        if c == 0 {
            return self.const_u32(0);
        }
        if c == u32::MAX {
            return x;
        }
        assert!(
            (c + 1).is_power_of_two(),
            "and: constant {c:#x} is not `(1 << k) - 1`; only low-bit masks lower to quasi-affine form"
        );
        let m = self.const_u32(c + 1);
        self.rem(x, m)
    }

    /// `x | c` under the precondition that `x & c == 0` (the caller
    /// guarantees the bits set in `c` are zero in `x`) — equivalent to `x + c`
    /// in that case, which is the emitted lowering. `c == 0` folds to `x`.
    pub fn or(&mut self, x: NodeId, c: usize) -> NodeId {
        let c = c as u32;
        if c == 0 {
            return x;
        }
        let k = self.const_u32(c);
        self.add(x, k)
    }

    /// `x ^ c` under the same disjoint-bit precondition as [`Self::or`] — with
    /// `x & c == 0`, `x ^ c == x + c`. `c == 0` folds to `x`.
    pub fn xor(&mut self, x: NodeId, c: usize) -> NodeId {
        let c = c as u32;
        if c == 0 {
            return x;
        }
        let k = self.const_u32(c);
        self.add(x, k)
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
    pub fn compute(
        &mut self,
        bound: impl Into<SizeExpr>,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        self.compute_with(bound, None, None, None, f)
    }

    /// `#[scatter(...)] compute [bound] |i| { f(i) }`: results are stored
    /// through the bijective quasi-affine `scatter` map (see [`Scatter`]).
    pub fn compute_scatter(
        &mut self,
        bound: impl Into<SizeExpr>,
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
        bound: impl Into<SizeExpr>,
        scatter: Option<Scatter>,
        par: Option<ParSpec>,
        threads: Option<usize>,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Compute {
            bound: bound.into(),
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
    ///
    /// `inv` receives one fresh symbol per physical dimension and must
    /// return the logical coordinates (one expression per logical
    /// dimension). Validation checks that it inverts `f` pointwise.
    pub fn scatter_map(
        &mut self,
        n_params: usize,
        out_shape: Option<Vec<usize>>,
        f: impl FnOnce(&[Quast], fn(i64) -> Quast) -> Vec<Quast>,
        inv: impl FnOnce(&[Quast], fn(i64) -> Quast) -> Vec<Quast>,
    ) -> Scatter {
        let params: Vec<VarId> = (0..n_params).map(|_| self.fresh_var()).collect();
        let syms: Vec<Quast> = params.iter().map(|&p| Quast::sym(p)).collect();
        let exprs = f(&syms, Quast::cst);
        let inv_params: Vec<VarId> = (0..exprs.len()).map(|_| self.fresh_var()).collect();
        let inv_syms: Vec<Quast> = inv_params.iter().map(|&p| Quast::sym(p)).collect();
        let inv_exprs = inv(&inv_syms, Quast::cst);
        Scatter {
            params,
            exprs,
            inv_params,
            inv_exprs,
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
        bound: impl Into<SizeExpr>,
        f: impl FnOnce(&mut Self, NodeId) -> NodeId,
    ) -> NodeId {
        let var = self.fresh_var();
        let var_node = self.intern(Node::Var(var));
        let body = f(self, var_node);
        self.intern(Node::Reduce {
            op,
            bound: bound.into(),
            var,
            body,
        })
    }

    pub fn reduce_add(
        &mut self,
        bound: impl Into<SizeExpr>,
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
#[derive(Clone)]
pub struct Module {
    pub name: String,
    pub builder: IRBuilder,
    pub body: NodeId,
}
