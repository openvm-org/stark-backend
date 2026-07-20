//! Lowering from the canonical HIR [`Program`] to [`KernelProgram`].
//!
//! - Every top-level let output becomes a global buffer (input / output / scratch) with a row-major
//!   layout.
//! - Every canonical compute becomes one kernel with a compute layout: a flat grid-stride
//!   factorization, or grid = outer / threads = inner when the compute nests a thread-level
//!   compute.
//! - Reduces become sequential `for` loops over mutable accumulators.
//! - Scalar expression DAGs become SSA `Def` statements (hash-consing in the HIR provides CSE);
//!   values computed inside a loop are scoped to it.
//! - Scratch buffers are placed with a liveness-based first-fit allocator.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    canonicalize::{CanonKernel, CanonValue, Program, ResultExpr, TensorRef},
    ir::{BinOp, IRBuilder, Node, NodeId, ReduceOp, ScalarType, Type, VarId},
    kernel_ir::{
        AddressSpace, BufferDecl, BufferKind, KExpr, Kernel, KernelProgram, LaunchShape, Stmt,
        ValId,
    },
    CompileError,
};

const BLOCK_SIZE: usize = 256;
const SCRATCH_ALIGN: usize = 256;

pub fn lower(program: &Program) -> Result<KernelProgram, CompileError> {
    let b = &program.module.builder;

    // --- Buffer declarations -------------------------------------------
    let mut buffers: Vec<BufferDecl> = Vec::new();
    let mut buf_map: HashMap<TensorRef, usize> = HashMap::new();

    let mut input_bufs = Vec::new();
    for (k, decl) in b.inputs().iter().enumerate() {
        let id = buffers.len();
        buffers.push(BufferDecl {
            name: decl.name.clone(),
            elem: decl.elem,
            shape: decl.shape.clone(),
            kind: BufferKind::Input(k),
            space: AddressSpace::Global,
        });
        buf_map.insert(TensorRef::Input(k), id);
        input_bufs.push(id);
    }

    let output_pos: HashMap<TensorRef, usize> = program
        .outputs
        .iter()
        .enumerate()
        .map(|(pos, &r)| (r, pos))
        .collect();

    for (let_id, kernel) in program.kernels.iter().enumerate() {
        for (out_idx, member_ty) in kernel.member_types.iter().enumerate() {
            let (elem, shape) = match member_ty {
                Type::Tensor(s, shape) => (*s, shape.clone()),
                other => {
                    return Err(CompileError::Lower(format!(
                        "kernel {} output {out_idx} has non-tensor type {other:?}",
                        kernel.name
                    )))
                }
            };
            let tref = TensorRef::Let { let_id, out_idx };
            let kind = match output_pos.get(&tref) {
                Some(&pos) => BufferKind::Output(pos),
                None => BufferKind::Scratch { offset: 0 },
            };
            let id = buffers.len();
            buffers.push(BufferDecl {
                name: format!("{}_out{out_idx}", kernel.name),
                elem,
                shape,
                kind,
                space: AddressSpace::Global,
            });
            buf_map.insert(tref, id);
        }
    }

    let mut output_bufs = vec![usize::MAX; program.outputs.len()];
    for (&tref, &pos) in &output_pos {
        output_bufs[pos] = buf_map[&tref];
    }

    // --- Lower kernel bodies -------------------------------------------
    let mut kernels = Vec::new();
    let mut reads_per_kernel: Vec<BTreeSet<usize>> = Vec::new();
    let mod_name = sanitize(&program.module.name);
    for (let_id, ck) in program.kernels.iter().enumerate() {
        let mut cx = EmitCx {
            b,
            program,
            buffers: &buffers,
            buf_map: &buf_map,
            kernel: ck,
            stmts: vec![Vec::new()],
            memo: HashMap::new(),
            scopes: vec![Vec::new()],
            var_vals: HashMap::new(),
            next_val: 0,
            outer_val: None,
            inner_val: None,
            reads: BTreeSet::new(),
            writes: BTreeSet::new(),
        };
        cx.emit_body(let_id)?;

        let launch = match ck.inner {
            None => LaunchShape::Flat { n: ck.outer_bound },
            Some((m, _)) => LaunchShape::GridBlock {
                n: ck.outer_bound,
                m,
            },
        };
        let (grid, block) = match launch {
            LaunchShape::Flat { n } => {
                let block = n.min(BLOCK_SIZE);
                (n.div_ceil(block), block)
            }
            LaunchShape::GridBlock { n, m } => (n, m.min(BLOCK_SIZE)),
        };

        let mut params: Vec<(usize, bool)> = Vec::new();
        for &buf in &cx.reads {
            params.push((buf, cx.writes.contains(&buf)));
        }
        for &buf in &cx.writes {
            if !cx.reads.contains(&buf) {
                params.push((buf, true));
            }
        }
        params.sort();

        let body = cx.stmts.pop().expect("statement sink");
        assert!(cx.stmts.is_empty(), "unbalanced statement scopes");
        reads_per_kernel.push(cx.reads);
        kernels.push(Kernel {
            name: format!("{mod_name}_{}", ck.name),
            launch,
            grid,
            block,
            params,
            body,
        });
    }

    // --- Scratch planning (liveness + first-fit) ------------------------
    let scratch_bytes = plan_scratch(&mut buffers, &buf_map, program, &reads_per_kernel);

    Ok(KernelProgram {
        name: mod_name,
        buffers,
        kernels,
        scratch_bytes,
        input_bufs,
        output_bufs,
    })
}

fn sanitize(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() || s.chars().next().unwrap().is_ascii_digit() {
        s.insert(0, 'm');
    }
    s
}

// ---------------------------------------------------------------------------
// Kernel body emission
// ---------------------------------------------------------------------------

struct EmitCx<'a> {
    b: &'a IRBuilder,
    program: &'a Program,
    buffers: &'a [BufferDecl],
    buf_map: &'a HashMap<TensorRef, usize>,
    kernel: &'a CanonKernel,
    /// Stack of statement sinks; the top receives new statements.
    stmts: Vec<Vec<Stmt>>,
    memo: HashMap<NodeId, ValId>,
    /// Memo keys created per scope depth, evicted when the scope closes.
    scopes: Vec<Vec<NodeId>>,
    /// Loop induction variables and scalar-let bindings.
    var_vals: HashMap<VarId, ValId>,
    next_val: u32,
    outer_val: Option<ValId>,
    inner_val: Option<ValId>,
    reads: BTreeSet<usize>,
    writes: BTreeSet<usize>,
}

impl EmitCx<'_> {
    fn emit_body(&mut self, let_id: usize) -> Result<(), CompileError> {
        for (out_idx, result) in self.kernel.results.iter().cloned().enumerate() {
            let buf = self.buf_map[&TensorRef::Let { let_id, out_idx }];
            self.writes.insert(buf);
            let base = self.linear_iter_index()?;
            match result {
                ResultExpr::Scalar(node) => {
                    let value = self.emit(node)?;
                    self.push(Stmt::Store {
                        buf,
                        index: base,
                        value,
                    });
                }
                ResultExpr::Pack(elems) => {
                    let k = self.def(KExpr::ConstU32(elems.len() as u32));
                    let scaled = self.def_bin(BinOp::Mul, ScalarType::U32, base, k);
                    for (l, e) in elems.iter().enumerate() {
                        let off = self.def(KExpr::ConstU32(l as u32));
                        let index = self.def_bin(BinOp::Add, ScalarType::U32, scaled, off);
                        let value = self.emit(*e)?;
                        self.push(Stmt::Store { buf, index, value });
                    }
                }
            }
        }
        Ok(())
    }

    /// Linear index of the current iteration point in the kernel's row-major
    /// output space: `outer` for flat kernels, `outer * m + inner` otherwise.
    fn linear_iter_index(&mut self) -> Result<ValId, CompileError> {
        let outer = self.outer_idx();
        match self.kernel.inner {
            None => Ok(outer),
            Some((m, _)) => {
                let mc = self.def(KExpr::ConstU32(m as u32));
                let scaled = self.def_bin(BinOp::Mul, ScalarType::U32, outer, mc);
                let inner = self.inner_idx();
                Ok(self.def_bin(BinOp::Add, ScalarType::U32, scaled, inner))
            }
        }
    }

    fn fresh(&mut self) -> ValId {
        let v = ValId(self.next_val);
        self.next_val += 1;
        v
    }

    fn push(&mut self, stmt: Stmt) {
        self.stmts.last_mut().expect("statement sink").push(stmt);
    }

    fn def(&mut self, expr: KExpr) -> ValId {
        let dst = self.fresh();
        self.push(Stmt::Def { dst, expr });
        dst
    }

    fn def_bin(&mut self, op: BinOp, ty: ScalarType, lhs: ValId, rhs: ValId) -> ValId {
        self.def(KExpr::Bin { op, ty, lhs, rhs })
    }

    fn outer_idx(&mut self) -> ValId {
        if let Some(v) = self.outer_val {
            return v;
        }
        let v = self.def(KExpr::OuterIdx);
        self.outer_val = Some(v);
        v
    }

    fn inner_idx(&mut self) -> ValId {
        if let Some(v) = self.inner_val {
            return v;
        }
        let v = self.def(KExpr::InnerIdx);
        self.inner_val = Some(v);
        v
    }

    fn memoize(&mut self, id: NodeId, val: ValId) {
        self.memo.insert(id, val);
        self.scopes.last_mut().expect("scope").push(id);
    }

    fn scalar_ty(&self, id: NodeId) -> Result<ScalarType, CompileError> {
        match self.program.types.try_get(id) {
            Some(Type::Scalar(s)) => Ok(*s),
            Some(other) => Err(CompileError::Lower(format!(
                "expected scalar in kernel body, got {other:?}"
            ))),
            None => Err(CompileError::Lower(format!("missing type for node {id:?}"))),
        }
    }

    fn emit(&mut self, id: NodeId) -> Result<ValId, CompileError> {
        if let Some(&v) = self.memo.get(&id) {
            return Ok(v);
        }
        let node = self.b.node(id).clone();
        let val = match node {
            Node::Var(v) => {
                if v == self.kernel.outer_var {
                    self.outer_idx()
                } else if self.kernel.inner.is_some_and(|(_, iv)| iv == v) {
                    self.inner_idx()
                } else if let Some(&val) = self.var_vals.get(&v) {
                    // Loop vars / scalar lets: already a value, do not memoize
                    // the Var node itself (its binding is scope-dependent).
                    return Ok(val);
                } else if let Some(&value_node) = self.kernel.inline_lets.get(&v) {
                    self.emit(value_node)?
                } else if let Some(CanonValue::Scalar(n)) = self.program.env.get(&v) {
                    self.emit(*n)?
                } else {
                    return Err(CompileError::Lower(format!(
                        "variable {v:?} is not usable as a scalar in this kernel \
                         (unbound, or bound to a tensor)"
                    )));
                }
            }
            Node::ConstU32(c) => self.def(KExpr::ConstU32(c)),
            Node::ConstField(c) => self.def(KExpr::ConstField(c)),
            Node::Bin(op, x, y) => {
                let ty = self.scalar_ty(x)?;
                let lhs = self.emit(x)?;
                let rhs = self.emit(y)?;
                self.def_bin(op, ty, lhs, rhs)
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let c = self.emit(cond)?;
                let t = self.emit(then_val)?;
                let e = self.emit(else_val)?;
                self.def(KExpr::Select {
                    cond: c,
                    then_val: t,
                    else_val: e,
                })
            }
            Node::Index { tensor, indices } => {
                let tref = resolve_tensor_ref(self.b, &self.program.env, tensor)?;
                let buf = self.buf_map[&tref];
                self.reads.insert(buf);
                let shape = self.buffers[buf].shape.clone();
                if indices.len() != shape.len() {
                    return Err(CompileError::Lower(format!(
                        "index arity {} does not match buffer rank {} for {}",
                        indices.len(),
                        shape.len(),
                        self.buffers[buf].name
                    )));
                }
                let index = self.linearize(&indices, &shape)?;
                self.def(KExpr::Load { buf, index })
            }
            // A scalar module input (declared with empty shape).
            Node::Input(k) => {
                let buf = self.buf_map[&TensorRef::Input(k)];
                self.reads.insert(buf);
                let index = self.def(KExpr::ConstU32(0));
                self.def(KExpr::Load { buf, index })
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => self.emit_reduce(op, bound, var, body)?,
            Node::Let { var, value, body } => {
                let val = self.emit(value)?;
                self.var_vals.insert(var, val);
                self.emit(body)?
            }
            other @ (Node::Compute { .. } | Node::Tuple(_) | Node::Proj(_, _) | Node::Pack(_)) => {
                return Err(CompileError::Lower(format!(
                    "unsupported expression in scalar position: {other:?}; \
                     nested computes must be materialized in their own top-level let"
                )));
            }
        };
        self.memoize(id, val);
        Ok(val)
    }

    fn emit_reduce(
        &mut self,
        op: ReduceOp,
        bound: usize,
        var: VarId,
        body: NodeId,
    ) -> Result<ValId, CompileError> {
        let ty = self.scalar_ty(body)?;
        let init_const = match (op, ty) {
            (ReduceOp::Add, ScalarType::BabyBear) => KExpr::ConstField(0),
            (ReduceOp::Mul, ScalarType::BabyBear) => KExpr::ConstField(1),
            (ReduceOp::Add, ScalarType::U32) => KExpr::ConstU32(0),
            (ReduceOp::Mul, ScalarType::U32) => KExpr::ConstU32(1),
            (_, ScalarType::Bool) => {
                return Err(CompileError::Lower("cannot reduce over Bool".into()))
            }
        };
        let init = self.def(init_const);
        let acc = self.fresh();
        self.push(Stmt::DefAcc { dst: acc, init });

        let loop_var = self.fresh();
        self.var_vals.insert(var, loop_var);
        self.scopes.push(Vec::new());
        self.stmts.push(Vec::new());

        let value = self.emit(body)?;
        self.push(Stmt::AccUpdate { acc, op, ty, value });

        let body_stmts = self.stmts.pop().expect("loop sink");
        self.push(Stmt::Loop {
            var: loop_var,
            bound,
            body: body_stmts,
        });
        self.var_vals.remove(&var);
        for evicted in self.scopes.pop().expect("scope") {
            self.memo.remove(&evicted);
        }
        Ok(acc)
    }

    /// Row-major linearization of a multi-dimensional index.
    fn linearize(&mut self, indices: &[NodeId], shape: &[usize]) -> Result<ValId, CompileError> {
        if indices.is_empty() {
            return Ok(self.def(KExpr::ConstU32(0)));
        }
        let mut strides = vec![1usize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        let mut acc: Option<ValId> = None;
        for (&ix, &stride) in indices.iter().zip(&strides) {
            let v = self.emit(ix)?;
            let term = if stride == 1 {
                v
            } else {
                let s = self.def(KExpr::ConstU32(stride as u32));
                self.def_bin(BinOp::Mul, ScalarType::U32, v, s)
            };
            acc = Some(match acc {
                None => term,
                Some(a) => self.def_bin(BinOp::Add, ScalarType::U32, a, term),
            });
        }
        Ok(acc.expect("non-empty indices"))
    }
}

/// Resolves a tensor-typed HIR expression to a top-level tensor reference.
fn resolve_tensor_ref(
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

// ---------------------------------------------------------------------------
// Scratch planning
// ---------------------------------------------------------------------------

/// Assigns scratch offsets with an interval-liveness first-fit allocator and
/// returns the total scratch size in bytes. A scratch buffer is live from the
/// kernel that writes it through the last kernel that reads it.
fn plan_scratch(
    buffers: &mut [BufferDecl],
    buf_map: &HashMap<TensorRef, usize>,
    program: &Program,
    reads_per_kernel: &[BTreeSet<usize>],
) -> usize {
    // def kernel and last read per scratch buffer
    let mut intervals: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for (let_id, kernel) in program.kernels.iter().enumerate() {
        for out_idx in 0..kernel.results.len() {
            let buf = buf_map[&TensorRef::Let { let_id, out_idx }];
            if matches!(buffers[buf].kind, BufferKind::Scratch { .. }) {
                intervals.insert(buf, (let_id, let_id));
            }
        }
    }
    for (k_idx, reads) in reads_per_kernel.iter().enumerate() {
        for buf in reads {
            if let Some(iv) = intervals.get_mut(buf) {
                iv.1 = iv.1.max(k_idx);
            }
        }
    }

    let mut free: Vec<(usize, usize)> = Vec::new(); // (offset, size), sorted
    let mut active: Vec<(usize, usize, usize)> = Vec::new(); // (end, offset, size)
    let mut high = 0usize;

    let mut by_def: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&buf, &(def, _)) in &intervals {
        by_def.entry(def).or_default().push(buf);
    }

    for (&def, bufs) in &by_def {
        // Release allocations that died before this kernel.
        let mut still_active = Vec::new();
        for (end, offset, size) in active.drain(..) {
            if end < def {
                free_insert(&mut free, offset, size);
            } else {
                still_active.push((end, offset, size));
            }
        }
        active = still_active;

        for &buf in bufs {
            let size = buffers[buf].size_bytes().next_multiple_of(SCRATCH_ALIGN);
            let offset = match free.iter().position(|&(_, hole_size)| hole_size >= size) {
                Some(i) => {
                    let (off, hole_size) = free[i];
                    if hole_size == size {
                        free.remove(i);
                    } else {
                        free[i] = (off + size, hole_size - size);
                    }
                    off
                }
                None => {
                    let off = high;
                    high += size;
                    off
                }
            };
            buffers[buf].kind = BufferKind::Scratch { offset };
            active.push((intervals[&buf].1, offset, size));
        }
    }
    high
}

fn free_insert(free: &mut Vec<(usize, usize)>, offset: usize, size: usize) {
    let pos = free.partition_point(|&(o, _)| o < offset);
    free.insert(pos, (offset, size));
    // Coalesce with the next hole, then with the previous one.
    if pos + 1 < free.len() && free[pos].0 + free[pos].1 == free[pos + 1].0 {
        free[pos].1 += free[pos + 1].1;
        free.remove(pos + 1);
    }
    if pos > 0 && free[pos - 1].0 + free[pos - 1].1 == free[pos].0 {
        free[pos - 1].1 += free[pos].1;
        free.remove(pos);
    }
}
