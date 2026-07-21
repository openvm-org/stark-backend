//! Lowering from the canonical HIR [`Program`] to [`KernelProgram`].
//!
//! - Every top-level let output becomes a global buffer (input / output / scratch) with a row-major
//!   layout.
//! - Every canonical compute becomes one kernel. A flat compute is a single grid-spanning `Par`; a
//!   compute that nests a thread-level compute or binds inner tiles becomes a grid of `outer_bound`
//!   blocks with per-block `Par` statements.
//! - Let-bound inner computes become shared-memory buffers written by their own `Par`.
//! - A par is a primitive compute block: all loads are declared up front as read accesses with
//!   quasi-affine indices and enter its SSA block as operands; stores leave as yields. Index
//!   expressions are extracted as [`Quast`]s over the par / grid / loop induction variables, so
//!   data-dependent indexing is rejected here.
//! - A reduce whose body loads from memory is hoisted out of its par into a register accumulator
//!   ([`BufferKind::Register`]) initialized by one `Par` and updated by a `Par` inside a sequential
//!   [`Stmt::Loop`]. A load-free reduce stays inside the par as an [`ElemSSAOp::Loop`].
//! - Scratch buffers are placed with a liveness-based first-fit allocator.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use smallvec::{smallvec, SmallVec};

use crate::{
    canonicalize::{CanonKernel, CanonValue, Program, ResultExpr, TensorRef},
    ir::{BinOp, IRBuilder, Node, NodeId, ReduceOp, ScalarType, Type, VarId},
    kernel_ir::{
        Access, AddressSpace, BufId, BufferDecl, BufferKind, ElemSSAOp, IndexMap, Kernel,
        KernelProgram, SSABlock, SSANode, SSAOp, SSARes, Stmt, StmtNode,
    },
    quast::{self, Quast, ScatterStore},
    CompileError,
};

const BLOCK_SIZE: usize = 256;
const SCRATCH_ALIGN: usize = 256;

pub fn lower(program: &Program) -> Result<KernelProgram, CompileError> {
    let b = &program.module.builder;

    // --- Buffer declarations -------------------------------------------
    let mut buffers: Vec<BufferDecl> = Vec::new();
    let mut buf_map: HashMap<TensorRef, BufId> = HashMap::new();

    let mut input_bufs = Vec::new();
    for (k, decl) in b.inputs().iter().enumerate() {
        let id = BufId(buffers.len() as u32);
        buffers.push(BufferDecl {
            name: decl.name.clone(),
            elem: decl.elem,
            shape: decl.shape.clone(),
            kind: BufferKind::Input(k),
            space: AddressSpace::Global,
            layout: None,
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
            let id = BufId(buffers.len() as u32);
            buffers.push(BufferDecl {
                name: format!("{}_out{out_idx}", kernel.name),
                elem,
                shape,
                kind,
                space: AddressSpace::Global,
                layout: None,
            });
            buf_map.insert(tref, id);
        }
    }

    let mut output_bufs = vec![BufId(u32::MAX); program.outputs.len()];
    for (&tref, &pos) in &output_pos {
        output_bufs[pos] = buf_map[&tref];
    }

    // Block-local shared buffers: one per let-bound inner compute (tile).
    let mut shared_maps: Vec<HashMap<VarId, BufId>> = Vec::new();
    for kernel in &program.kernels {
        let mut map = HashMap::new();
        for (t_idx, il) in kernel.inner_lets.iter().enumerate() {
            let id = BufId(buffers.len() as u32);
            buffers.push(BufferDecl {
                name: format!("{}_tile{t_idx}", kernel.name),
                elem: il.elem,
                shape: il.shape.clone(),
                kind: BufferKind::Shared,
                space: AddressSpace::Shared,
                layout: None,
            });
            map.insert(il.var, id);
        }
        shared_maps.push(map);
    }

    // --- Lower kernel bodies -------------------------------------------
    let mut kernels = Vec::new();
    let mut reads_per_kernel: Vec<BTreeSet<BufId>> = Vec::new();
    let mod_name = sanitize(&program.module.name);
    for (let_id, ck) in program.kernels.iter().enumerate() {
        let flat = ck.inner.is_none() && ck.inner_lets.is_empty();
        let (grid_bound, block) = if flat {
            let n = ck.outer_bound;
            let block = n.min(BLOCK_SIZE);
            (n.div_ceil(block), block)
        } else {
            let max_par = ck
                .inner_lets
                .iter()
                .map(|il| il.bound)
                .chain(ck.inner.map(|(m, _)| m))
                .max()
                .unwrap_or(1);
            (ck.outer_bound, max_par.min(BLOCK_SIZE))
        };

        let k = Kernel::new(format!("{mod_name}_{}", ck.name), grid_bound, block);
        let mut sym_bounds = BTreeMap::new();
        sym_bounds.insert(VarId(k.grid.var.0), grid_bound as u64);
        let mut cx = LowerCx {
            b,
            program,
            buffers: &mut buffers,
            buf_map: &buf_map,
            shared_bufs: &shared_maps[let_id],
            ck,
            k,
            sinks: vec![Vec::new()],
            sym_bounds,
            hoisted: HashMap::new(),
            loads_memo: HashMap::new(),
            hoist_seen: HashSet::new(),
            loop_vars: HashMap::new(),
            reads: BTreeSet::new(),
            writes: BTreeSet::new(),
            flat,
        };
        cx.emit_kernel(let_id)?;

        let LowerCx {
            mut k,
            mut sinks,
            reads,
            writes,
            ..
        } = cx;
        k.grid.body = sinks.pop().expect("statement sink");
        assert!(sinks.is_empty(), "unbalanced statement scopes");

        let mut params: Vec<(BufId, bool)> = Vec::new();
        for &buf in &reads {
            params.push((buf, writes.contains(&buf)));
        }
        for &buf in &writes {
            if !reads.contains(&buf) {
                params.push((buf, true));
            }
        }
        params.sort();
        k.params = params;

        reads_per_kernel.push(reads);
        kernels.push(k);
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

fn bin_of(op: ReduceOp) -> BinOp {
    match op {
        ReduceOp::Add => BinOp::Add,
        ReduceOp::Mul => BinOp::Mul,
    }
}

fn init_const(op: ReduceOp, ty: ScalarType) -> Result<ElemSSAOp, CompileError> {
    match (op, ty) {
        (ReduceOp::Add, ScalarType::BabyBear) => Ok(ElemSSAOp::ConstField(0)),
        (ReduceOp::Mul, ScalarType::BabyBear) => Ok(ElemSSAOp::ConstField(1)),
        (ReduceOp::Add, ScalarType::U32) => Ok(ElemSSAOp::ConstU32(0)),
        (ReduceOp::Mul, ScalarType::U32) => Ok(ElemSSAOp::ConstU32(1)),
        (_, ScalarType::Bool) => Err(CompileError::Lower("cannot reduce over Bool".into())),
    }
}

/// Physical store index for the logical flat index `base * k + l`, remapped
/// through the scatter store map when present.
fn store_index(scatter: Option<&ScatterStore>, base: &Quast, k: usize, l: usize) -> Quast {
    let flat = if k == 1 {
        base.clone()
    } else {
        base.mul_c(k as i64).add(&Quast::cst(l as i64))
    };
    match scatter {
        None => flat,
        Some(st) => st.expr.substitute(&BTreeMap::from([(st.flat, flat)])),
    }
}

// ---------------------------------------------------------------------------
// Kernel body emission
// ---------------------------------------------------------------------------

struct LowerCx<'a> {
    b: &'a IRBuilder,
    program: &'a Program,
    buffers: &'a mut Vec<BufferDecl>,
    buf_map: &'a HashMap<TensorRef, BufId>,
    /// Shared buffer per let-bound inner compute of this kernel.
    shared_bufs: &'a HashMap<VarId, BufId>,
    ck: &'a CanonKernel,
    k: Kernel,
    /// Stack of statement sinks; the top receives new statements.
    sinks: Vec<Vec<StmtNode>>,
    /// Bounds of the kernel-level values usable as index symbols (grid var,
    /// par indices, `Stmt::Loop` induction variables), by the
    /// `VarId(i) <-> SSARes(i)` convention.
    sym_bounds: BTreeMap<VarId, u64>,
    /// Register accumulator per hoisted reduce node.
    hoisted: HashMap<NodeId, BufId>,
    loads_memo: HashMap<NodeId, bool>,
    /// Nodes already walked by [`Self::hoist_reduces`]. Without this the walk
    /// is exponential: every `Var` occurrence re-descends into its (shared)
    /// let-bound value.
    hoist_seen: HashSet<NodeId>,
    /// `Stmt::Loop` induction variables of enclosing hoisted reduces.
    loop_vars: HashMap<VarId, SSARes>,
    reads: BTreeSet<BufId>,
    writes: BTreeSet<BufId>,
    /// Flat kernel: every par spans the grid.
    flat: bool,
}

impl<'a> LowerCx<'a> {
    fn emit_kernel(&mut self, let_id: usize) -> Result<(), CompileError> {
        let ck = self.ck;
        let scatter = ck.scatter_store.as_ref();

        if self.flat {
            // Flat kernel: one grid-spanning par over all iteration points.
            let n = ck.outer_bound;
            for result in &ck.results {
                self.hoist_result_reduces(result, n, None)?;
            }
            return self.build_par(n, None, |bx| {
                let base = Quast::sym(VarId(bx.vid.0));
                bx.store_results(let_id, &base, scatter)
            });
        }

        // Grid kernel: outer = blockIdx.x, per-block pars.
        for il in &ck.inner_lets {
            let buf = self.shared_bufs[&il.var];
            self.push_stmt(Stmt::BufDecl { buf });
            self.hoist_result_reduces(&il.result, il.bound, Some(il.iter_var))?;
            self.build_par(il.bound, Some(il.iter_var), |bx| {
                let base = Quast::sym(VarId(bx.vid.0));
                bx.store_result(buf, &base, &il.result, il.scatter.as_ref())
            })?;
        }

        let g = self.k.grid.var;
        match ck.inner {
            Some((m, j)) => {
                for result in &ck.results {
                    self.hoist_result_reduces(result, m, Some(j))?;
                }
                self.build_par(m, Some(j), |bx| {
                    let base = Quast::sym(VarId(g.0))
                        .mul_c(m as i64)
                        .add(&Quast::sym(VarId(bx.vid.0)));
                    bx.store_results(let_id, &base, scatter)
                })
            }
            // Scalar results computed per block: a single-point par so only
            // one thread stores them.
            None => {
                for result in &ck.results {
                    self.hoist_result_reduces(result, 1, None)?;
                }
                self.build_par(1, None, |bx| {
                    let base = Quast::sym(VarId(g.0));
                    bx.store_results(let_id, &base, scatter)
                })
            }
        }
    }

    fn push_stmt(&mut self, stmt: Stmt) {
        let id = self.k.push_stmt(stmt);
        self.sinks.last_mut().expect("statement sink").push(id);
    }

    /// Emits `f` as the body of a new `Par` statement over `bound` logical
    /// indices, binding `var` (if given) to the par index.
    fn build_par<F>(&mut self, bound: usize, var: Option<VarId>, f: F) -> Result<(), CompileError>
    where
        F: FnOnce(&mut BlockCx<'_, 'a>) -> Result<(), CompileError>,
    {
        let vid = self.k.fresh_val();
        self.sym_bounds.insert(VarId(vid.0), bound as u64);
        let spans_grid = self.flat;
        let mut bx = BlockCx {
            cx: self,
            vid,
            par_var: var,
            operands: smallvec![vid],
            bodies: vec![SmallVec::new()],
            read_keys: Vec::new(),
            read_vals: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
            yields: SmallVec::new(),
            memo: HashMap::new(),
            scopes: vec![Vec::new()],
            var_vals: HashMap::new(),
            let_nodes: HashMap::new(),
        };
        if let Some(v) = var {
            bx.var_vals.insert(v, vid);
        }
        f(&mut bx)?;
        let BlockCx {
            operands,
            mut bodies,
            reads,
            writes,
            yields,
            ..
        } = bx;
        let body = bodies.pop().expect("par body");
        assert!(bodies.is_empty(), "unbalanced SSA blocks");
        self.push_stmt(Stmt::Par {
            bound,
            spans_grid,
            attr: None,
            reads,
            writes,
            block: SSABlock {
                operands,
                body,
                yields,
            },
        });
        self.sym_bounds.remove(&VarId(vid.0));
        Ok(())
    }

    fn hoist_result_reduces(
        &mut self,
        result: &ResultExpr,
        par_bound: usize,
        par_var: Option<VarId>,
    ) -> Result<(), CompileError> {
        match result {
            ResultExpr::Scalar(n) => self.hoist_reduces(*n, par_bound, par_var),
            ResultExpr::Pack(elems) => {
                for &e in elems {
                    self.hoist_reduces(e, par_bound, par_var)?;
                }
                Ok(())
            }
        }
    }

    /// Hoists every reduce under `id` whose body loads from memory into a
    /// register accumulator at the current statement level (loads cannot
    /// appear inside a par's SSA block).
    fn hoist_reduces(
        &mut self,
        id: NodeId,
        par_bound: usize,
        par_var: Option<VarId>,
    ) -> Result<(), CompileError> {
        if !self.hoist_seen.insert(id) {
            return Ok(());
        }
        match self.b.node(id).clone() {
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                if self.contains_loads(body) && !self.hoisted.contains_key(&id) {
                    self.emit_hoisted_reduce(id, op, bound, var, body, par_bound, par_var)?;
                }
                Ok(())
            }
            Node::Bin(_, x, y) => {
                self.hoist_reduces(x, par_bound, par_var)?;
                self.hoist_reduces(y, par_bound, par_var)
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                self.hoist_reduces(cond, par_bound, par_var)?;
                self.hoist_reduces(then_val, par_bound, par_var)?;
                self.hoist_reduces(else_val, par_bound, par_var)
            }
            Node::Let { value, body, .. } => {
                self.hoist_reduces(value, par_bound, par_var)?;
                self.hoist_reduces(body, par_bound, par_var)
            }
            Node::Var(v) => {
                if let Some(&n) = self.ck.inline_lets.get(&v) {
                    self.hoist_reduces(n, par_bound, par_var)
                } else if let Some(CanonValue::Scalar(n)) = self.program.env.get(&v) {
                    let n = *n;
                    self.hoist_reduces(n, par_bound, par_var)
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }

    /// Emits the hoisted pattern for a reduce with loads: a register
    /// accumulator over the par domain, an init par, and a sequential
    /// [`Stmt::Loop`] around an accumulate par (which re-binds `par_var` so
    /// the body sees the same logical index as the consumer par).
    #[allow(clippy::too_many_arguments)]
    fn emit_hoisted_reduce(
        &mut self,
        id: NodeId,
        op: ReduceOp,
        bound: usize,
        var: VarId,
        body: NodeId,
        par_bound: usize,
        par_var: Option<VarId>,
    ) -> Result<(), CompileError> {
        let ty = self.scalar_ty(body)?;
        let init = init_const(op, ty)?;

        let acc = BufId(self.buffers.len() as u32);
        self.buffers.push(BufferDecl {
            name: format!("{}_acc{}", self.ck.name, self.hoisted.len()),
            elem: ty,
            shape: vec![par_bound],
            kind: BufferKind::Register,
            space: AddressSpace::Register,
            layout: None,
        });
        self.push_stmt(Stmt::BufDecl { buf: acc });

        self.build_par(par_bound, None, |bx| {
            let v = bx.push_elem(init, smallvec![]);
            bx.write(acc, Quast::sym(VarId(bx.vid.0)), v);
            Ok(())
        })?;

        let r = self.k.fresh_val();
        self.loop_vars.insert(var, r);
        self.sym_bounds.insert(VarId(r.0), bound as u64);
        self.sinks.push(Vec::new());
        self.hoist_reduces(body, par_bound, par_var)?;
        self.build_par(par_bound, par_var, |bx| {
            let idx = Quast::sym(VarId(bx.vid.0));
            let cur = bx.read(acc, idx.clone());
            let value = bx.emit(body)?;
            let combined = bx.bin(bin_of(op), ty, cur, value);
            bx.write(acc, idx, combined);
            Ok(())
        })?;
        let body_stmts = self.sinks.pop().expect("loop sink");
        self.push_stmt(Stmt::Loop {
            var: r,
            bound,
            body: body_stmts,
        });

        self.loop_vars.remove(&var);
        self.sym_bounds.remove(&VarId(r.0));
        self.hoisted.insert(id, acc);
        Ok(())
    }

    /// Whether the expression loads from memory anywhere (tensor index or
    /// scalar input), resolving variables through lets.
    fn contains_loads(&mut self, id: NodeId) -> bool {
        if let Some(&v) = self.loads_memo.get(&id) {
            return v;
        }
        let v = match self.b.node(id).clone() {
            Node::Index { .. } | Node::Input(_) => true,
            Node::Var(v) => {
                if let Some(&n) = self.ck.inline_lets.get(&v) {
                    self.contains_loads(n)
                } else if let Some(CanonValue::Scalar(n)) = self.program.env.get(&v) {
                    let n = *n;
                    self.contains_loads(n)
                } else {
                    false
                }
            }
            Node::Bin(_, x, y) => self.contains_loads(x) || self.contains_loads(y),
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                self.contains_loads(cond)
                    || self.contains_loads(then_val)
                    || self.contains_loads(else_val)
            }
            Node::Let { value, body, .. } => {
                self.contains_loads(value) || self.contains_loads(body)
            }
            Node::Reduce { body, .. } => self.contains_loads(body),
            _ => false,
        };
        self.loads_memo.insert(id, v);
        v
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
}

/// Emission context for one par's SSA block.
struct BlockCx<'x, 'a> {
    cx: &'x mut LowerCx<'a>,
    /// The par's logical index (the block's first operand).
    vid: SSARes,
    /// The HIR variable bound to the par index, if any.
    par_var: Option<VarId>,
    operands: SmallVec<[SSARes; 2]>,
    /// Stack of op sinks: the par body, then nested SSA loop bodies.
    bodies: Vec<SmallVec<[SSANode; 8]>>,
    read_keys: Vec<(BufId, Quast)>,
    read_vals: Vec<SSARes>,
    reads: Vec<Access>,
    writes: Vec<Access>,
    yields: SmallVec<[SSARes; 1]>,
    memo: HashMap<NodeId, SSARes>,
    /// Memo keys created per scope depth, evicted when the scope closes.
    scopes: Vec<Vec<NodeId>>,
    /// SSA-loop induction variables and scalar-let bindings.
    var_vals: HashMap<VarId, SSARes>,
    /// Defining node of scalar lets, for index-expression extraction.
    let_nodes: HashMap<VarId, NodeId>,
}

impl BlockCx<'_, '_> {
    fn push_elem(&mut self, opcode: ElemSSAOp, operands: SmallVec<[SSARes; 2]>) -> SSARes {
        let res = self.cx.k.fresh_val();
        let node = self.cx.k.push_op(SSAOp {
            operands,
            results: smallvec![res],
            opcode,
            block: SSABlock::default(),
        });
        self.bodies.last_mut().expect("op sink").push(node);
        res
    }

    fn bin(&mut self, op: BinOp, ty: ScalarType, a: SSARes, b: SSARes) -> SSARes {
        self.push_elem(ElemSSAOp::Bin(op, ty), smallvec![a, b])
    }

    /// Declares a read access and returns the block operand holding the
    /// loaded value (deduplicated per `(buffer, index)`).
    fn read(&mut self, buf: BufId, expr: Quast) -> SSARes {
        if let Some(i) = self
            .read_keys
            .iter()
            .position(|(rb, rq)| *rb == buf && *rq == expr)
        {
            return self.read_vals[i];
        }
        let v = self.cx.k.fresh_val();
        self.read_keys.push((buf, expr.clone()));
        self.read_vals.push(v);
        self.operands.push(v);
        self.reads.push(Access {
            buf,
            index: IndexMap::Affine {
                expr,
                bounds: self.cx.sym_bounds.clone(),
            },
        });
        if self.cx.buffers[buf.0 as usize].space == AddressSpace::Global {
            self.cx.reads.insert(buf);
        }
        v
    }

    /// Declares a write access whose stored value is `val` (yielded by the
    /// block).
    fn write(&mut self, buf: BufId, expr: Quast, val: SSARes) {
        self.writes.push(Access {
            buf,
            index: IndexMap::Affine {
                expr,
                bounds: self.cx.sym_bounds.clone(),
            },
        });
        self.yields.push(val);
        if self.cx.buffers[buf.0 as usize].space == AddressSpace::Global {
            self.cx.writes.insert(buf);
        }
    }

    /// Stores every kernel result at logical flat index `base` of its output
    /// buffer.
    fn store_results(
        &mut self,
        let_id: usize,
        base: &Quast,
        scatter: Option<&ScatterStore>,
    ) -> Result<(), CompileError> {
        let results = &self.cx.ck.results;
        for (out_idx, result) in results.iter().enumerate() {
            let buf = self.cx.buf_map[&TensorRef::Let { let_id, out_idx }];
            self.store_result(buf, base, result, scatter)?;
        }
        Ok(())
    }

    /// Stores one result at logical flat index `base` (packs fan out to
    /// `base * k + l`), remapped through the scatter store map if present.
    fn store_result(
        &mut self,
        buf: BufId,
        base: &Quast,
        result: &ResultExpr,
        scatter: Option<&ScatterStore>,
    ) -> Result<(), CompileError> {
        match result {
            ResultExpr::Scalar(node) => {
                let index = store_index(scatter, base, 1, 0);
                let value = self.emit(*node)?;
                self.write(buf, index, value);
            }
            ResultExpr::Pack(elems) => {
                for (l, e) in elems.iter().enumerate() {
                    let index = store_index(scatter, base, elems.len(), l);
                    let value = self.emit(*e)?;
                    self.write(buf, index, value);
                }
            }
        }
        Ok(())
    }

    fn memoize(&mut self, id: NodeId, val: SSARes) {
        self.memo.insert(id, val);
        self.scopes.last_mut().expect("scope").push(id);
    }

    fn emit(&mut self, id: NodeId) -> Result<SSARes, CompileError> {
        if let Some(&v) = self.memo.get(&id) {
            return Ok(v);
        }
        let node = self.cx.b.node(id).clone();
        let val = match node {
            Node::Var(v) => {
                // Bindings are scope-dependent: do not memoize the Var node.
                if v == self.cx.ck.outer_var {
                    return Ok(if self.cx.flat {
                        self.vid
                    } else {
                        self.cx.k.grid.var
                    });
                } else if let Some(&val) = self.var_vals.get(&v) {
                    return Ok(val);
                } else if let Some(&r) = self.cx.loop_vars.get(&v) {
                    return Ok(r);
                } else if let Some(&value_node) = self.cx.ck.inline_lets.get(&v) {
                    self.emit(value_node)?
                } else if let Some(CanonValue::Scalar(n)) = self.cx.program.env.get(&v) {
                    let n = *n;
                    self.emit(n)?
                } else {
                    return Err(CompileError::Lower(format!(
                        "variable {v:?} is not usable as a scalar in this kernel \
                         (unbound, or bound to a tensor)"
                    )));
                }
            }
            Node::ConstU32(c) => self.push_elem(ElemSSAOp::ConstU32(c), smallvec![]),
            Node::ConstField(c) => self.push_elem(ElemSSAOp::ConstField(c), smallvec![]),
            Node::Bin(op, x, y) => {
                let ty = self.cx.scalar_ty(x)?;
                let lhs = self.emit(x)?;
                let rhs = self.emit(y)?;
                self.bin(op, ty, lhs, rhs)
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let c = self.emit(cond)?;
                let t = self.emit(then_val)?;
                let e = self.emit(else_val)?;
                self.push_elem(ElemSSAOp::Select, smallvec![c, t, e])
            }
            Node::Index { tensor, indices } => {
                let shared = match self.cx.b.node(tensor) {
                    Node::Var(v) => self.cx.shared_bufs.get(v).copied(),
                    _ => None,
                };
                let buf = match shared {
                    Some(buf) => buf,
                    None => {
                        let tref = resolve_tensor_ref(self.cx.b, &self.cx.program.env, tensor)?;
                        self.cx.buf_map[&tref]
                    }
                };
                let shape = self.cx.buffers[buf.0 as usize].shape.clone();
                if indices.len() != shape.len() {
                    return Err(CompileError::Lower(format!(
                        "index arity {} does not match buffer rank {} for {}",
                        indices.len(),
                        shape.len(),
                        self.cx.buffers[buf.0 as usize].name
                    )));
                }
                let exprs = indices
                    .iter()
                    .map(|&ix| self.extract_quast(ix))
                    .collect::<Result<Vec<_>, _>>()?;
                self.read(buf, quast::linearize(&exprs, &shape))
            }
            // A scalar module input (declared with empty shape).
            Node::Input(k) => {
                let buf = self.cx.buf_map[&TensorRef::Input(k)];
                self.read(buf, Quast::cst(0))
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                if let Some(&acc) = self.cx.hoisted.get(&id) {
                    self.read(acc, Quast::sym(VarId(self.vid.0)))
                } else {
                    // Hoisting guarantees the body is load-free here.
                    self.emit_ssaloop(op, bound, var, body)?
                }
            }
            Node::Let { var, value, body } => {
                let val = self.emit(value)?;
                self.var_vals.insert(var, val);
                self.let_nodes.insert(var, value);
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

    /// A load-free reduce as an MLIR-style `scf.for`: the accumulator is the
    /// loop-carried value.
    fn emit_ssaloop(
        &mut self,
        op: ReduceOp,
        bound: usize,
        var: VarId,
        body: NodeId,
    ) -> Result<SSARes, CompileError> {
        let ty = self.cx.scalar_ty(body)?;
        let init_op = init_const(op, ty)?;
        let init = self.push_elem(init_op, smallvec![]);

        let iv = self.cx.k.fresh_val();
        let carried = self.cx.k.fresh_val();
        self.var_vals.insert(var, iv);
        self.scopes.push(Vec::new());
        self.bodies.push(SmallVec::new());

        let value = self.emit(body)?;
        let combined = self.bin(bin_of(op), ty, carried, value);

        let loop_body = self.bodies.pop().expect("loop body");
        for evicted in self.scopes.pop().expect("scope") {
            self.memo.remove(&evicted);
        }
        self.var_vals.remove(&var);

        let result = self.cx.k.fresh_val();
        let node = self.cx.k.push_op(SSAOp {
            operands: smallvec![init],
            results: smallvec![result],
            opcode: ElemSSAOp::Loop { bound },
            block: SSABlock {
                operands: smallvec![iv, carried],
                body: loop_body,
                yields: smallvec![combined],
            },
        });
        self.bodies.last_mut().expect("op sink").push(node);
        Ok(result)
    }

    /// Extracts an index expression as a quasi-affine [`Quast`] over the
    /// kernel's index symbols (par index, grid var, loop induction vars).
    fn extract_quast(&mut self, id: NodeId) -> Result<Quast, CompileError> {
        match self.cx.b.node(id).clone() {
            Node::ConstU32(c) => Ok(Quast::cst(c as i64)),
            Node::Var(v) => {
                if v == self.cx.ck.outer_var {
                    Ok(if self.cx.flat {
                        Quast::sym(VarId(self.vid.0))
                    } else {
                        Quast::sym(VarId(self.cx.k.grid.var.0))
                    })
                } else if self.par_var == Some(v) {
                    Ok(Quast::sym(VarId(self.vid.0)))
                } else if let Some(&r) = self.cx.loop_vars.get(&v) {
                    Ok(Quast::sym(VarId(r.0)))
                } else if let Some(&n) = self.let_nodes.get(&v) {
                    self.extract_quast(n)
                } else if let Some(&n) = self.cx.ck.inline_lets.get(&v) {
                    self.extract_quast(n)
                } else if let Some(CanonValue::Scalar(n)) = self.cx.program.env.get(&v) {
                    let n = *n;
                    self.extract_quast(n)
                } else {
                    Err(CompileError::Lower(format!(
                        "variable {v:?} is not usable in an index expression"
                    )))
                }
            }
            Node::Bin(op, x, y) => match op {
                BinOp::Add => Ok(self.extract_quast(x)?.add(&self.extract_quast(y)?)),
                BinOp::Sub => Ok(self.extract_quast(x)?.sub(&self.extract_quast(y)?)),
                BinOp::Mul => {
                    if let Some(c) = self.const_eval(x) {
                        Ok(self.extract_quast(y)?.mul_c(c))
                    } else if let Some(c) = self.const_eval(y) {
                        Ok(self.extract_quast(x)?.mul_c(c))
                    } else {
                        Err(CompileError::Lower(
                            "non-affine multiplication in index expression".into(),
                        ))
                    }
                }
                BinOp::Div => {
                    let c = self.const_eval(y).filter(|&c| c > 0).ok_or_else(|| {
                        CompileError::Lower(
                            "index division requires a positive constant divisor".into(),
                        )
                    })?;
                    Ok(self.extract_quast(x)?.floordiv(c))
                }
                BinOp::Rem => {
                    let c = self.const_eval(y).filter(|&c| c > 0).ok_or_else(|| {
                        CompileError::Lower(
                            "index remainder requires a positive constant divisor".into(),
                        )
                    })?;
                    Ok(self.extract_quast(x)?.rem_c(c))
                }
                BinOp::Lt | BinOp::Le | BinOp::Eq => {
                    Err(CompileError::Lower("comparison in index expression".into()))
                }
            },
            Node::Let { var, value, body } => {
                self.let_nodes.insert(var, value);
                self.extract_quast(body)
            }
            other => Err(CompileError::Lower(format!(
                "non-quasi-affine index expression: {other:?}"
            ))),
        }
    }

    /// Constant-folds an index subexpression, resolving variables through
    /// lets.
    fn const_eval(&self, id: NodeId) -> Option<i64> {
        match self.cx.b.node(id) {
            Node::ConstU32(c) => Some(*c as i64),
            Node::Var(v) => {
                let n = self
                    .let_nodes
                    .get(v)
                    .copied()
                    .or_else(|| self.cx.ck.inline_lets.get(v).copied())
                    .or_else(|| match self.cx.program.env.get(v) {
                        Some(CanonValue::Scalar(n)) => Some(*n),
                        _ => None,
                    })?;
                self.const_eval(n)
            }
            Node::Bin(op, x, y) => {
                let a = self.const_eval(*x)?;
                let b = self.const_eval(*y)?;
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div => (b != 0).then(|| a.div_euclid(b)),
                    BinOp::Rem => (b != 0).then(|| a.rem_euclid(b)),
                    BinOp::Lt | BinOp::Le | BinOp::Eq => None,
                }
            }
            _ => None,
        }
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
    buf_map: &HashMap<TensorRef, BufId>,
    program: &Program,
    reads_per_kernel: &[BTreeSet<BufId>],
) -> usize {
    // def kernel and last read per scratch buffer
    let mut intervals: BTreeMap<BufId, (usize, usize)> = BTreeMap::new();
    for (let_id, kernel) in program.kernels.iter().enumerate() {
        for out_idx in 0..kernel.results.len() {
            let buf = buf_map[&TensorRef::Let { let_id, out_idx }];
            if matches!(buffers[buf.0 as usize].kind, BufferKind::Scratch { .. }) {
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

    let mut by_def: BTreeMap<usize, Vec<BufId>> = BTreeMap::new();
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
            let size = buffers[buf.0 as usize]
                .size_bytes()
                .next_multiple_of(SCRATCH_ALIGN);
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
            buffers[buf.0 as usize].kind = BufferKind::Scratch { offset };
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
