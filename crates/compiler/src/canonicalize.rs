//! Canonicalization: rewrites an arbitrary [`Module`] into canonical form —
//! a top-level ordered sequence of `let` bindings whose right-hand sides are
//! `compute` expressions, where each compute nests at most one more compute
//! (the thread level) and reduces only appear with scalar bodies.
//!
//! Rewrites performed:
//! - top-level `let` chains are flattened into an ordered kernel list;
//! - compute nests deeper than 2 are flattened by merging the innermost pair of computes into one
//!   over the product bound (indices recovered with div/mod, which is a no-op on the row-major
//!   memory layout);
//! - a top-level `reduce` is wrapped into `compute [1]`;
//! - scalar `let`s wrapping a compute body are peeled into an inline environment resolved during
//!   lowering;
//! - `let`s binding an inner `compute` (a per-block tile) directly under the outer compute are
//!   collected as [`InnerLet`]s; lowering materializes them in shared memory.
//!
//! All rewrites preserve types, and the [`TypeMap`] is extended for every
//! node they create.

use std::collections::{BTreeMap, HashMap};

use crate::{
    ir::{infer_types, IRBuilder, Module, Node, NodeId, ScalarType, Type, TypeMap, VarId},
    quast::{NodeEmitter, Quast, ScatterStore},
    CompileError,
};

/// Reference to a top-level tensor value: a module input or one output of a
/// top-level let (kernel).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorRef {
    Input(usize),
    Let { let_id: usize, out_idx: usize },
}

/// What a top-level `let` variable resolves to.
#[derive(Clone, Debug)]
pub enum CanonValue {
    /// One or more tensors (one per tuple member of the kernel result).
    Tensors(Vec<TensorRef>),
    /// A scalar expression, inlined at use sites.
    Scalar(NodeId),
}

/// One output of a canonical kernel.
#[derive(Clone, Debug)]
pub enum ResultExpr {
    /// Scalar expression per iteration point.
    Scalar(NodeId),
    /// Small array literal per iteration point (innermost dimension).
    Pack(Vec<NodeId>),
}

/// An inner `compute` bound by a `let` directly under the outer compute:
/// `let var = compute [bound] |iter_var| { ... }`. One tile is materialized
/// per outer iteration point (per block), so lowering places it in shared
/// memory.
#[derive(Clone, Debug)]
pub struct InnerLet {
    pub var: VarId,
    pub bound: usize,
    pub iter_var: VarId,
    pub result: ResultExpr,
    pub elem: ScalarType,
    /// Physical row-major shape of the tile (e.g. `[bound]`, or `[bound, k]`
    /// for a pack body), after applying any scatter.
    pub shape: Vec<usize>,
    /// Store map composed from the tile's `#[scatter(...)]` attribute:
    /// logical flat index -> physical flat index (`None` = identity).
    /// Readers index the physical layout.
    pub scatter: Option<ScatterStore>,
}

/// A canonical top-level compute: `compute [outer_bound] |i| { body }` where
/// the body optionally nests exactly one more compute (the thread level) and
/// otherwise consists of scalar expressions.
#[derive(Clone, Debug)]
pub struct CanonKernel {
    pub outer_bound: usize,
    pub outer_var: VarId,
    /// `Some((bound, var))` if the body is an inner `compute` over threads.
    pub inner: Option<(usize, VarId)>,
    /// Let-bound inner computes (shared-memory tiles), in binding order.
    pub inner_lets: Vec<InnerLet>,
    /// One entry per output buffer.
    pub results: Vec<ResultExpr>,
    /// Scalar lets peeled from body positions, resolved at use sites.
    pub inline_lets: HashMap<VarId, NodeId>,
    /// Logical result type as written by the user (before nest flattening),
    /// one member per entry of `results`.
    pub member_types: Vec<Type>,
    /// Store map composed from the compute's `#[scatter(...)]` attribute:
    /// logical flat index -> physical flat index (`None` = identity).
    pub scatter_store: Option<ScatterStore>,
    pub name: String,
}

/// A module in canonical form.
pub struct Program {
    pub module: Module,
    pub types: TypeMap,
    pub kernels: Vec<CanonKernel>,
    pub outputs: Vec<TensorRef>,
    /// Resolution of top-level let variables.
    pub env: HashMap<VarId, CanonValue>,
}

pub fn canonicalize(module: Module) -> Result<Program, CompileError> {
    let types = infer_types(&module)?;
    let Module {
        name,
        mut builder,
        body,
    } = module;

    let mut cx = CanonCx {
        b: &mut builder,
        types,
        kernels: Vec::new(),
        env: HashMap::new(),
    };
    let outputs = cx.walk_top(body)?;
    let CanonCx {
        types,
        kernels,
        env,
        ..
    } = cx;

    if outputs.is_empty() {
        return Err(CompileError::Canonicalize(
            "module has no tensor outputs".into(),
        ));
    }
    for (i, r) in outputs.iter().enumerate() {
        if matches!(r, TensorRef::Input(_)) {
            return Err(CompileError::Canonicalize(format!(
                "output {i} is a passthrough of an input, which is not supported"
            )));
        }
        if outputs[..i].contains(r) {
            return Err(CompileError::Canonicalize(format!(
                "output {i} duplicates an earlier output"
            )));
        }
    }

    Ok(Program {
        module: Module {
            name,
            builder,
            body,
        },
        types,
        kernels,
        outputs,
        env,
    })
}

struct CanonCx<'a> {
    b: &'a mut IRBuilder,
    types: TypeMap,
    kernels: Vec<CanonKernel>,
    env: HashMap<VarId, CanonValue>,
}

impl CanonCx<'_> {
    fn ty(&self, id: NodeId) -> Result<Type, CompileError> {
        self.types
            .try_get(id)
            .cloned()
            .ok_or_else(|| CompileError::Canonicalize(format!("missing type for node {id:?}")))
    }

    /// Walks the top-level let chain; returns the module outputs.
    fn walk_top(&mut self, id: NodeId) -> Result<Vec<TensorRef>, CompileError> {
        match self.b.node(id).clone() {
            Node::Let { var, value, body } => {
                let bound = self.canon_top_value(value)?;
                self.env.insert(var, bound);
                self.walk_top(body)
            }
            Node::Tuple(elems) => {
                let mut outs = Vec::new();
                for e in elems {
                    outs.extend(self.resolve_or_emit(e)?);
                }
                Ok(outs)
            }
            _ => self.resolve_or_emit(id),
        }
    }

    /// Canonicalizes the rhs of a top-level let.
    fn canon_top_value(&mut self, id: NodeId) -> Result<CanonValue, CompileError> {
        match self.b.node(id).clone() {
            Node::Compute { .. } | Node::Reduce { .. } => {
                let refs = self.emit_kernel(id)?;
                Ok(CanonValue::Tensors(refs))
            }
            Node::Var(v) => self.env.get(&v).cloned().ok_or_else(|| {
                CompileError::Canonicalize(format!("unbound top-level variable {v:?}"))
            }),
            Node::Input(k) => Ok(CanonValue::Tensors(vec![TensorRef::Input(k)])),
            Node::Tuple(elems) => {
                let mut refs = Vec::new();
                for e in elems {
                    refs.extend(self.resolve_or_emit(e)?);
                }
                Ok(CanonValue::Tensors(refs))
            }
            Node::Proj(_, _) => Ok(CanonValue::Tensors(vec![self.resolve_tensor_ref(id)?])),
            // Anything else is treated as a scalar bound at top level and
            // inlined at its use sites.
            _ => match self.ty(id)? {
                Type::Scalar(_) => Ok(CanonValue::Scalar(id)),
                other => Err(CompileError::Canonicalize(format!(
                    "unsupported top-level let value of type {other:?}"
                ))),
            },
        }
    }

    /// Resolves an expression to tensor refs, emitting a kernel if it is a
    /// fresh compute/reduce.
    fn resolve_or_emit(&mut self, id: NodeId) -> Result<Vec<TensorRef>, CompileError> {
        match self.b.node(id).clone() {
            Node::Compute { .. } | Node::Reduce { .. } => self.emit_kernel(id),
            Node::Tuple(elems) => {
                let mut outs = Vec::new();
                for e in elems {
                    outs.extend(self.resolve_or_emit(e)?);
                }
                Ok(outs)
            }
            Node::Var(v) => match self.env.get(&v).cloned() {
                Some(CanonValue::Tensors(refs)) => Ok(refs),
                Some(CanonValue::Scalar(_)) => Err(CompileError::Canonicalize(
                    "scalar values cannot be module outputs".into(),
                )),
                None => Err(CompileError::Canonicalize(format!(
                    "unbound top-level variable {v:?}"
                ))),
            },
            Node::Input(k) => Ok(vec![TensorRef::Input(k)]),
            Node::Proj(_, _) => Ok(vec![self.resolve_tensor_ref(id)?]),
            other => Err(CompileError::Canonicalize(format!(
                "unsupported top-level expression: {other:?}"
            ))),
        }
    }

    /// Resolves `Proj(Var(v), k)` / `Var(v)` / `Input(k)` to a single tensor.
    fn resolve_tensor_ref(&mut self, id: NodeId) -> Result<TensorRef, CompileError> {
        match self.b.node(id).clone() {
            Node::Input(k) => Ok(TensorRef::Input(k)),
            Node::Var(v) => match self.env.get(&v) {
                Some(CanonValue::Tensors(refs)) if refs.len() == 1 => Ok(refs[0]),
                Some(CanonValue::Tensors(_)) => Err(CompileError::Canonicalize(
                    "tuple-valued variable used where a single tensor is expected".into(),
                )),
                Some(CanonValue::Scalar(_)) => Err(CompileError::Canonicalize(
                    "scalar variable used where a tensor is expected".into(),
                )),
                None => Err(CompileError::Canonicalize(format!(
                    "unbound variable {v:?}"
                ))),
            },
            Node::Proj(t, k) => match self.b.node(t).clone() {
                Node::Var(v) => match self.env.get(&v) {
                    Some(CanonValue::Tensors(refs)) => refs.get(k).copied().ok_or_else(|| {
                        CompileError::Canonicalize(format!("projection index {k} out of bounds"))
                    }),
                    _ => Err(CompileError::Canonicalize(
                        "projection from a non-tuple variable".into(),
                    )),
                },
                Node::Tuple(elems) => {
                    let e = *elems.get(k).ok_or_else(|| {
                        CompileError::Canonicalize(format!("projection index {k} out of bounds"))
                    })?;
                    self.resolve_tensor_ref(e)
                }
                _ => Err(CompileError::Canonicalize(
                    "projection from an unsupported expression".into(),
                )),
            },
            other => Err(CompileError::Canonicalize(format!(
                "expected a tensor reference, got {other:?}"
            ))),
        }
    }

    /// Turns a compute/reduce into a canonical kernel; returns one tensor ref
    /// per kernel output.
    fn emit_kernel(&mut self, id: NodeId) -> Result<Vec<TensorRef>, CompileError> {
        // A top-level reduce is wrapped into `compute [1] |_| { reduce ... }`.
        let (compute_id, logical_ty) = match self.b.node(id).clone() {
            Node::Reduce { .. } => {
                let elem = match self.ty(id)? {
                    Type::Scalar(s) => s,
                    other => {
                        return Err(CompileError::Canonicalize(format!(
                            "top-level reduce must have scalar type, got {other:?}"
                        )))
                    }
                };
                let var = self.b.fresh_var();
                let var_node = self.b.intern(Node::Var(var));
                self.types.insert(var_node, Type::Scalar(ScalarType::U32));
                let wrapped = self.b.intern(Node::Compute {
                    bound: 1,
                    var,
                    body: id,
                    scatter: None,
                });
                let ty = Type::Tensor(elem, vec![1]);
                self.types.insert(wrapped, ty.clone());
                (wrapped, ty)
            }
            Node::Compute { .. } => (id, self.ty(id)?),
            _ => unreachable!("emit_kernel called on non-compute"),
        };

        let Node::Compute {
            bound: outer_bound,
            var: outer_var,
            body,
            scatter,
        } = self.b.node(compute_id).clone()
        else {
            unreachable!()
        };

        // Compose the scatter attribute into a flat store map while the
        // logical (pre-flattening) shape is still known.
        let mut scatter_store = match scatter {
            None => None,
            Some(sc) => {
                let mut logical = vec![outer_bound];
                logical.extend_from_slice(self.ty(body)?.shape());
                let mut sc = *sc;
                sc.bind_and_validate(&logical)?;
                Some(sc.store_map(self.b.fresh_var())?)
            }
        };

        // Peel lets wrapping the body: scalars are inlined at use sites,
        // let-bound inner computes become shared-memory tiles.
        let mut inline_lets = HashMap::new();
        let mut inner_lets = Vec::new();
        let body = self.peel_body_lets(body, &mut inline_lets, &mut inner_lets)?;

        // If the body is an inner compute, flatten any deeper nests into it.
        let (inner, result_root) = match self.b.node(body).clone() {
            Node::Compute {
                bound: m,
                var: j,
                body: inner_body,
                scatter: inner_scatter,
            } => {
                // An inner scatter permutes elements within one outer
                // iteration's block. Build its store map against the logical
                // (pre-flattening) shape, then lift it to the global flat
                // index and compose with the outer scatter, which addresses
                // the inner-scattered layout.
                let inner_store = match inner_scatter {
                    None => None,
                    Some(sc) => {
                        let mut logical = vec![m];
                        logical.extend_from_slice(self.ty(inner_body)?.shape());
                        let mut sc = *sc;
                        sc.bind_and_validate(&logical)?;
                        let block_len = logical.iter().product::<usize>();
                        Some((sc.store_map(self.b.fresh_var())?, block_len))
                    }
                };
                let (m, j, inner_body) = self.flatten_nests(m, j, inner_body)?;
                if let Some((inner, block_len)) = inner_store {
                    let f = self.b.fresh_var();
                    let fq = Quast::sym(f);
                    let within = inner
                        .expr
                        .substitute(&BTreeMap::from([(inner.flat, fq.rem_c(block_len as i64))]));
                    let lifted = fq
                        .floordiv(block_len as i64)
                        .mul_c(block_len as i64)
                        .add(&within);
                    let composed = match scatter_store.take() {
                        None => lifted,
                        Some(o) => o.expr.substitute(&BTreeMap::from([(o.flat, lifted)])),
                    };
                    let bounds = BTreeMap::from([(f, (outer_bound * block_len) as u64)]);
                    let expr = composed.simplify(&bounds)?;
                    scatter_store = Some(ScatterStore {
                        flat: f,
                        expr,
                        bounds,
                    });
                }
                let inner_body = self.peel_scalar_lets(inner_body, &mut inline_lets)?;
                (Some((m, j)), inner_body)
            }
            _ => (None, body),
        };

        // Split the result into one expression per output buffer.
        let results = match self.b.node(result_root).clone() {
            Node::Tuple(elems) => elems
                .iter()
                .map(|&e| self.classify_result(e))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![self.classify_result(result_root)?],
        };

        // Logical member types (pre-flattening).
        let member_types = match &logical_ty {
            Type::Tuple(ts) => ts.clone(),
            other => vec![other.clone()],
        };
        if member_types.len() != results.len() {
            return Err(CompileError::Canonicalize(format!(
                "kernel result arity mismatch: {} logical members vs {} results",
                member_types.len(),
                results.len()
            )));
        }

        let let_id = self.kernels.len();
        let num_outs = results.len();
        self.kernels.push(CanonKernel {
            outer_bound,
            outer_var,
            inner,
            inner_lets,
            results,
            inline_lets,
            member_types,
            scatter_store,
            name: format!("k{let_id}"),
        });
        Ok((0..num_outs)
            .map(|out_idx| TensorRef::Let { let_id, out_idx })
            .collect())
    }

    fn classify_result(&mut self, id: NodeId) -> Result<ResultExpr, CompileError> {
        match self.b.node(id).clone() {
            Node::Pack(elems) => Ok(ResultExpr::Pack(elems)),
            Node::Compute { .. } => Err(CompileError::Canonicalize(
                "compute nested inside a tuple result is not supported; \
                 bind it in its own top-level let"
                    .into(),
            )),
            _ => Ok(ResultExpr::Scalar(id)),
        }
    }

    /// Peels `let` wrappers off the body of an outer compute. Scalar values
    /// go to the inline environment; values that are themselves a `compute`
    /// become [`InnerLet`]s (per-block shared-memory tiles).
    fn peel_body_lets(
        &mut self,
        mut id: NodeId,
        inline_lets: &mut HashMap<VarId, NodeId>,
        inner_lets: &mut Vec<InnerLet>,
    ) -> Result<NodeId, CompileError> {
        while let Node::Let { var, value, body } = self.b.node(id).clone() {
            match self.b.node(value).clone() {
                Node::Compute {
                    bound,
                    var: iter_var,
                    body: tile_body,
                    scatter,
                } => {
                    // Compose the tile's scatter into a flat store map while
                    // the logical (pre-flattening) shape is still known.
                    let tile_scatter = match scatter {
                        None => None,
                        Some(sc) => {
                            let mut logical = vec![bound];
                            logical.extend_from_slice(self.ty(tile_body)?.shape());
                            let mut sc = *sc;
                            sc.bind_and_validate(&logical)?;
                            Some(sc.store_map(self.b.fresh_var())?)
                        }
                    };
                    let (elem, shape) = match self.ty(value)? {
                        Type::Tensor(e, s) => (e, s),
                        other => {
                            return Err(CompileError::Canonicalize(format!(
                                "let-bound inner compute must have tensor type, got {other:?}"
                            )))
                        }
                    };
                    let (bound, iter_var, tile_body) =
                        self.flatten_nests(bound, iter_var, tile_body)?;
                    let tile_body = self.peel_scalar_lets(tile_body, inline_lets)?;
                    let result = self.classify_result(tile_body)?;
                    inner_lets.push(InnerLet {
                        var,
                        bound,
                        iter_var,
                        result,
                        elem,
                        shape,
                        scatter: tile_scatter,
                    });
                    id = body;
                }
                _ => match self.ty(value)? {
                    Type::Scalar(_) => {
                        inline_lets.insert(var, value);
                        id = body;
                    }
                    other => {
                        return Err(CompileError::Canonicalize(format!(
                            "let-bound value of type {other:?} inside a compute body is not \
                             supported; only scalars and inner computes can be bound"
                        )))
                    }
                },
            }
        }
        Ok(id)
    }

    /// Peels `let v = <scalar> in ...` wrappers off an inner body into an
    /// inline environment. Tensor-typed lets are only supported directly
    /// under the outer compute (see [`CanonCx::peel_body_lets`]).
    fn peel_scalar_lets(
        &mut self,
        mut id: NodeId,
        inline_lets: &mut HashMap<VarId, NodeId>,
    ) -> Result<NodeId, CompileError> {
        while let Node::Let { var, value, body } = self.b.node(id).clone() {
            match self.ty(value)? {
                Type::Scalar(_) => {
                    inline_lets.insert(var, value);
                    id = body;
                }
                other => {
                    return Err(CompileError::Canonicalize(format!(
                        "let-bound value of type {other:?} is only supported directly \
                         under the outer compute"
                    )))
                }
            }
        }
        Ok(id)
    }

    /// Flattens `compute [m] |j| { compute [k] |l| { e } }` chains into a
    /// single `compute [m*k] |t| { e[j := t/k, l := t%k] }`, repeatedly,
    /// until the inner body is not a compute. Row-major layouts make this a
    /// metadata-only change.
    fn flatten_nests(
        &mut self,
        mut m: usize,
        mut j: VarId,
        mut body: NodeId,
    ) -> Result<(usize, VarId, NodeId), CompileError> {
        loop {
            let Node::Compute {
                bound: k,
                var: l,
                body: inner,
                scatter,
            } = self.b.node(body).clone()
            else {
                return Ok((m, j, body));
            };
            if scatter.is_some() {
                return Err(CompileError::Canonicalize(
                    "scatter is only supported on top-level computes".into(),
                ));
            }
            let t = self.b.fresh_var();
            let t_node = self.b.intern(Node::Var(t));
            self.types.insert(t_node, Type::Scalar(ScalarType::U32));
            // Index recovery as quasi-affine expressions of the flat index.
            let env = BTreeMap::from([(t, t_node)]);
            let bounds = BTreeMap::from([(t, (m * k) as u64)]);
            let flat = Quast::sym(t);
            let mut em = NodeEmitter {
                b: &mut *self.b,
                types: &mut self.types,
                env: &env,
            };
            let j_val = flat.floordiv(k as i64).emit(&bounds, &mut em)?;
            let l_val = flat.rem_c(k as i64).emit(&bounds, &mut em)?;
            let mut map = HashMap::new();
            map.insert(j, j_val);
            map.insert(l, l_val);
            body = subst(self.b, inner, &map, &mut self.types);
            m *= k;
            j = t;
        }
    }
}

/// Rebuilds `root` with variables substituted per `map`. Binders always use
/// globally fresh variables, so capture is impossible. Substitution preserves
/// types (U32 vars are replaced by U32 expressions), so each rebuilt node
/// inherits the type of the node it replaces.
pub(crate) fn subst(
    b: &mut IRBuilder,
    root: NodeId,
    map: &HashMap<VarId, NodeId>,
    types: &mut TypeMap,
) -> NodeId {
    let mut memo: HashMap<NodeId, NodeId> = HashMap::new();
    subst_rec(b, root, map, types, &mut memo)
}

fn subst_rec(
    b: &mut IRBuilder,
    id: NodeId,
    map: &HashMap<VarId, NodeId>,
    types: &mut TypeMap,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&r) = memo.get(&id) {
        return r;
    }
    let node = b.node(id).clone();
    let result = match node {
        Node::Var(v) => map.get(&v).copied().unwrap_or(id),
        Node::Input(_) | Node::ConstU32(_) | Node::ConstField(_) => id,
        Node::Bin(op, x, y) => {
            let x2 = subst_rec(b, x, map, types, memo);
            let y2 = subst_rec(b, y, map, types, memo);
            b.bin(op, x2, y2)
        }
        Node::Select {
            cond,
            then_val,
            else_val,
        } => {
            let c2 = subst_rec(b, cond, map, types, memo);
            let t2 = subst_rec(b, then_val, map, types, memo);
            let e2 = subst_rec(b, else_val, map, types, memo);
            b.select(c2, t2, e2)
        }
        Node::Index { tensor, indices } => {
            let t2 = subst_rec(b, tensor, map, types, memo);
            let ix2: Vec<_> = indices
                .iter()
                .map(|&ix| subst_rec(b, ix, map, types, memo))
                .collect();
            b.index(t2, &ix2)
        }
        Node::Compute {
            bound,
            var,
            body,
            scatter,
        } => {
            let body2 = subst_rec(b, body, map, types, memo);
            b.intern(Node::Compute {
                bound,
                var,
                body: body2,
                scatter,
            })
        }
        Node::Reduce {
            op,
            bound,
            var,
            body,
        } => {
            let body2 = subst_rec(b, body, map, types, memo);
            b.intern(Node::Reduce {
                op,
                bound,
                var,
                body: body2,
            })
        }
        Node::Let { var, value, body } => {
            let v2 = subst_rec(b, value, map, types, memo);
            let b2 = subst_rec(b, body, map, types, memo);
            b.intern(Node::Let {
                var,
                value: v2,
                body: b2,
            })
        }
        Node::Tuple(elems) => {
            let e2: Vec<_> = elems
                .iter()
                .map(|&e| subst_rec(b, e, map, types, memo))
                .collect();
            b.tuple(&e2)
        }
        Node::Proj(t, k) => {
            let t2 = subst_rec(b, t, map, types, memo);
            b.proj(t2, k)
        }
        Node::Pack(elems) => {
            let e2: Vec<_> = elems
                .iter()
                .map(|&e| subst_rec(b, e, map, types, memo))
                .collect();
            b.pack(&e2)
        }
    };
    if result != id {
        if let Some(t) = types.try_get(id).cloned() {
            types.insert(result, t);
        }
    }
    memo.insert(id, result);
    result
}
