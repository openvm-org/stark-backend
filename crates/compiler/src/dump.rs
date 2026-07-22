//! Human-readable dumps of both IR levels.
//!
//! [`dump_hir`] prints a [`Module`] in expression form, mirroring the DSL:
//! nested `compute`/`reduce` blocks, `let ... in` chains and infix scalar
//! ops. Hash-consed nodes with multiple uses are not inlined at every use:
//! they are bound as `%id = expr` definitions in the innermost scope that
//! binds their free variables, which keeps deeply shared DAGs (poseidon2)
//! linear in the number of unique nodes.
//!
//! [`dump_kernel_ir`] prints a [`KernelProgram`] in an MLIR-ish concrete
//! syntax. SSA value names `v{n}` match the generated CUDA, so the two can
//! be cross-referenced; a par's declared reads and writes are shown as
//! `load`/`store` lines binding the block operands and yields; `Sync` ops
//! (placed by `passes::insert_sync`) are shown as `sync` lines.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt::Write as _,
    io,
    path::Path,
};

use crate::{
    ir::{BinOp, IRBuilder, Module, Node, NodeId, ReduceOp, ScalarType, VarId},
    kernel_ir::{
        Access, BufId, BufferDecl, BufferKind, IndexMap, Kernel, KernelProgram, SSABlock, SSANode,
        SSAOp, SSAOpCode, SSARes,
    },
    passes::SharedMemPlan,
    quast::{CStrEmitter, Quast, Scatter},
};

/// Writes `{name}.hir` and `{name}.kir` under `dir` (created if needed).
pub fn write_ir_dumps(dir: &Path, name: &str, hir: &str, kir: &str) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{name}.hir")), hir)?;
    std::fs::write(dir.join(format!("{name}.kir")), kir)
}

/// Writes the generated CUDA C++ as `{name}.cu` under `dir` (created if
/// needed).
pub fn write_cuda_dump(dir: &Path, name: &str, source: &str) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{name}.cu")), source)
}

// ---------------------------------------------------------------------------
// High-level IR
// ---------------------------------------------------------------------------

pub fn dump_hir(module: &Module) -> String {
    let b = &module.builder;
    let mut s = String::new();
    writeln!(s, "module {}", module.name).unwrap();
    for (k, decl) in b.inputs().iter().enumerate() {
        writeln!(
            s,
            "  input{k} \"{}\": {}",
            decl.name,
            tensor_ty(decl.elem, &decl.shape)
        )
        .unwrap();
    }
    s.push('\n');
    let mut p = HirPrinter::new(b, module.body);
    p.print_block(Some(None), module.body, 1);
    s.push_str(&p.out);
    s
}

fn tensor_ty(elem: ScalarType, shape: &[usize]) -> String {
    if shape.is_empty() {
        format!("{elem:?}")
    } else {
        format!("{elem:?}{shape:?}")
    }
}

/// Direct children of a node, in operand order.
fn children(node: &Node) -> Vec<NodeId> {
    match node {
        Node::Input(_) | Node::Var(_) | Node::ConstU32(_) | Node::ConstField(_) => vec![],
        Node::Bin(_, a, b) => vec![*a, *b],
        Node::Select {
            cond,
            then_val,
            else_val,
        } => vec![*cond, *then_val, *else_val],
        Node::Index { tensor, indices } => {
            let mut v = vec![*tensor];
            v.extend(indices.iter().copied());
            v
        }
        Node::Compute { body, .. } | Node::Reduce { body, .. } => vec![*body],
        Node::Let { value, body, .. } => vec![*value, *body],
        Node::Tuple(elems) | Node::Pack(elems) => elems.clone(),
        Node::Proj(tuple, _) => vec![*tuple],
    }
}

/// Ids reachable from `root`, ascending (children precede parents).
fn reachable(b: &IRBuilder, root: NodeId) -> Vec<NodeId> {
    let mut seen = BTreeSet::new();
    let mut work = vec![root];
    while let Some(id) = work.pop() {
        if seen.insert(id) {
            work.extend(children(b.node(id)));
        }
    }
    seen.into_iter().collect()
}

/// Records the nesting depth of every binder (compute/reduce/let variable)
/// on its unique root-to-binder path.
fn binder_depths(
    b: &IRBuilder,
    id: NodeId,
    d: usize,
    depth: &mut HashMap<VarId, usize>,
    visited: &mut HashSet<NodeId>,
) {
    if !visited.insert(id) {
        return;
    }
    match b.node(id) {
        Node::Compute { var, body, .. } | Node::Reduce { var, body, .. } => {
            depth.insert(*var, d + 1);
            binder_depths(b, *body, d + 1, depth, visited);
        }
        Node::Let { var, value, body } => {
            binder_depths(b, *value, d, depth, visited);
            depth.insert(*var, d + 1);
            binder_depths(b, *body, d + 1, depth, visited);
        }
        node => {
            for c in children(node) {
                binder_depths(b, c, d, depth, visited);
            }
        }
    }
}

/// Expression printer. Single-use nodes are inlined; multi-use nodes are
/// printed once as a `%id = expr` definition in the innermost scope that
/// binds their free variables (all free variables of a node lie on one
/// binder path, so the deepest one identifies that scope), which keeps
/// shared DAGs linear while every use site still sees the definition.
struct HirPrinter<'a> {
    b: &'a IRBuilder,
    named: HashSet<NodeId>,
    /// Definitions to print when entering a binder's scope (`None` = top
    /// level), in ascending id order so definitions precede their uses.
    defs_of: HashMap<Option<VarId>, Vec<NodeId>>,
    out: String,
}

impl<'a> HirPrinter<'a> {
    fn new(b: &'a IRBuilder, root: NodeId) -> Self {
        let ids = reachable(b, root);
        let mut uses: HashMap<NodeId, u32> = HashMap::new();
        for &id in &ids {
            for c in children(b.node(id)) {
                *uses.entry(c).or_default() += 1;
            }
        }
        // Trivial nodes are always inlined; Lets can't be shared (their
        // variables are fresh per construction) so naming them is pointless.
        let named: HashSet<NodeId> = ids
            .iter()
            .copied()
            .filter(|&id| {
                uses.get(&id).copied().unwrap_or(0) > 1
                    && !matches!(
                        b.node(id),
                        Node::Input(_)
                            | Node::Var(_)
                            | Node::ConstU32(_)
                            | Node::ConstField(_)
                            | Node::Let { .. }
                    )
            })
            .collect();

        // Free-variable sets, children first (ids ascend). Binders may
        // subtract their own variable from the union: it cannot occur free
        // in a Let's value because the variable is created after it.
        let mut free: HashMap<NodeId, BTreeSet<VarId>> = HashMap::new();
        for &id in &ids {
            let node = b.node(id);
            let mut f = BTreeSet::new();
            for c in children(node) {
                f.extend(free[&c].iter().copied());
            }
            match node {
                Node::Var(v) => {
                    f.insert(*v);
                }
                Node::Compute { var, .. } | Node::Reduce { var, .. } | Node::Let { var, .. } => {
                    f.remove(var);
                }
                _ => {}
            }
            free.insert(id, f);
        }

        let mut depth = HashMap::new();
        binder_depths(b, root, 0, &mut depth, &mut HashSet::new());

        let mut defs_of: HashMap<Option<VarId>, Vec<NodeId>> = HashMap::new();
        for &id in &ids {
            if named.contains(&id) {
                let anchor = free[&id].iter().copied().max_by_key(|v| depth[v]);
                defs_of.entry(anchor).or_default().push(id);
            }
        }
        HirPrinter {
            b,
            named,
            defs_of,
            out: String::new(),
        }
    }

    fn pad(&mut self, ind: usize) {
        for _ in 0..ind {
            self.out.push_str("  ");
        }
    }

    /// Prints a block body: pending definitions for `defs_key` (if given),
    /// then the `let ... in` chain and the final expression, one per line.
    fn print_block(&mut self, defs_key: Option<Option<VarId>>, body: NodeId, ind: usize) {
        if let Some(key) = defs_key {
            self.print_defs(&key, ind);
        }
        let mut cur = body;
        while let Node::Let { var, value, body } = self.b.node(cur) {
            let (var, value, body) = (*var, *value, *body);
            self.pad(ind);
            write!(self.out, "let v{} = ", var.0).unwrap();
            self.print_expr(value, 0, ind);
            self.out.push_str(" in\n");
            self.print_defs(&Some(var), ind);
            cur = body;
        }
        self.pad(ind);
        self.print_expr(cur, 0, ind);
        self.out.push('\n');
    }

    fn print_defs(&mut self, key: &Option<VarId>, ind: usize) {
        if let Some(list) = self.defs_of.remove(key) {
            for id in list {
                self.pad(ind);
                write!(self.out, "%{} = ", id.0).unwrap();
                self.print_raw(id, 0, ind);
                self.out.push('\n');
            }
        }
    }

    /// Prints `id` as an expression: named nodes as `%id`, everything else
    /// structurally. `prec` is the caller's precedence level; parenthesize
    /// if our own level is lower.
    fn print_expr(&mut self, id: NodeId, prec: u8, ind: usize) {
        if self.named.contains(&id) {
            write!(self.out, "%{}", id.0).unwrap();
        } else {
            self.print_raw(id, prec, ind);
        }
    }

    fn print_raw(&mut self, id: NodeId, prec: u8, ind: usize) {
        let b = self.b;
        match b.node(id) {
            Node::Input(k) => self.out.push_str(&b.inputs()[*k].name),
            Node::Var(v) => write!(self.out, "v{}", v.0).unwrap(),
            Node::ConstU32(c) => write!(self.out, "{c}").unwrap(),
            Node::ConstField(c) => write!(self.out, "{c}f").unwrap(),
            Node::Bin(op, a, c) => {
                let (op, a, c) = (*op, *a, *c);
                let lv = bin_level(op);
                if prec > lv {
                    self.out.push('(');
                }
                self.print_expr(a, lv, ind);
                write!(self.out, " {} ", bin_sym(op)).unwrap();
                // Same-op associative chains stay flat: a + b + c.
                let assoc = matches!(op, BinOp::Add | BinOp::Mul)
                    && matches!(b.node(c), Node::Bin(o, ..) if *o == op);
                self.print_expr(c, if assoc { lv } else { lv + 1 }, ind);
                if prec > lv {
                    self.out.push(')');
                }
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let (cond, t, e) = (*cond, *then_val, *else_val);
                if prec > 0 {
                    self.out.push('(');
                }
                self.out.push_str("if ");
                self.print_expr(cond, 1, ind);
                self.out.push_str(" then ");
                self.print_expr(t, 1, ind);
                self.out.push_str(" else ");
                self.print_expr(e, 1, ind);
                if prec > 0 {
                    self.out.push(')');
                }
            }
            Node::Index { tensor, indices } => {
                let (tensor, indices) = (*tensor, indices.clone());
                self.print_expr(tensor, 4, ind);
                self.out.push('[');
                for (i, ix) in indices.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    self.print_expr(*ix, 0, ind);
                }
                self.out.push(']');
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
            } => {
                let (bound, var, body) = (*bound, *var, *body);
                let sc = scatter.as_deref().map(scatter_str);
                write!(self.out, "compute[{bound}] |v{}|", var.0).unwrap();
                if let Some(sc) = sc {
                    write!(self.out, " {sc}").unwrap();
                }
                self.print_braced_body(var, body, ind);
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                let (bound, var, body) = (*bound, *var, *body);
                let name = match op {
                    ReduceOp::Add => "add",
                    ReduceOp::Mul => "mul",
                };
                write!(self.out, "reduce.{name}[{bound}] |v{}|", var.0).unwrap();
                self.print_braced_body(var, body, ind);
            }
            // A Let in expression position (not a block body): print it as
            // its own braced block.
            Node::Let { .. } => {
                self.out.push_str("{\n");
                self.print_block(None, id, ind + 1);
                self.pad(ind);
                self.out.push('}');
            }
            Node::Tuple(elems) => {
                let elems = elems.clone();
                self.out.push('(');
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    self.print_expr(*e, 0, ind);
                }
                self.out.push(')');
            }
            Node::Proj(tuple, k) => {
                let (tuple, k) = (*tuple, *k);
                self.print_expr(tuple, 4, ind);
                write!(self.out, ".{k}").unwrap();
            }
            Node::Pack(elems) => {
                let elems = elems.clone();
                self.out.push('[');
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        self.out.push_str(", ");
                    }
                    self.print_expr(*e, 0, ind);
                }
                self.out.push(']');
            }
        }
    }

    /// Prints ` { body }` after a binder head, on one line when the body is
    /// a short definition-free expression.
    fn print_braced_body(&mut self, var: VarId, body: NodeId, ind: usize) {
        let is_let = matches!(self.b.node(body), Node::Let { .. });
        if !is_let && !self.defs_of.contains_key(&Some(var)) {
            // Trial-render the body. The render is reused verbatim on the
            // multiline path: printing consumes `defs_of` entries, so it
            // must never be discarded.
            let saved = std::mem::take(&mut self.out);
            self.print_expr(body, 0, ind + 1);
            let inner = std::mem::replace(&mut self.out, saved);
            if !inner.contains('\n') && 2 * ind + inner.len() <= 78 {
                write!(self.out, " {{ {inner} }}").unwrap();
            } else {
                self.out.push_str(" {\n");
                self.pad(ind + 1);
                self.out.push_str(&inner);
                self.out.push('\n');
                self.pad(ind);
                self.out.push('}');
            }
        } else {
            self.out.push_str(" {\n");
            self.print_block(Some(Some(var)), body, ind + 1);
            self.pad(ind);
            self.out.push('}');
        }
    }
}

fn bin_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Rem => "rem",
        BinOp::Lt => "lt",
        BinOp::Le => "le",
        BinOp::Eq => "eq",
    }
}

fn bin_sym(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Eq => "==",
    }
}

/// Precedence level of a binary op (select is 0, atoms/postfix are 4).
fn bin_level(op: BinOp) -> u8 {
    match op {
        BinOp::Lt | BinOp::Le | BinOp::Eq => 1,
        BinOp::Add | BinOp::Sub => 2,
        BinOp::Mul | BinOp::Div | BinOp::Rem => 3,
    }
}

fn scatter_str(sc: &Scatter) -> String {
    let params = sc
        .params
        .iter()
        .map(|p| format!("v{}", p.0))
        .collect::<Vec<_>>()
        .join(", ");
    let exprs = sc
        .exprs
        .iter()
        .map(|e| quast_str(e, &sc.bounds))
        .collect::<Vec<_>>()
        .join(", ");
    let mut s = format!("scatter ({params}) -> [{exprs}]");
    if let Some(out) = &sc.out_shape {
        write!(s, " into {out:?}").unwrap();
    }
    s
}

// ---------------------------------------------------------------------------
// Kernel IR
// ---------------------------------------------------------------------------

pub fn dump_kernel_ir(p: &KernelProgram, shared: &SharedMemPlan) -> String {
    let mut s = String::new();
    writeln!(s, "program {} scratch={}B", p.name, p.scratch_bytes).unwrap();
    writeln!(s, "\nbuffers:").unwrap();
    for (i, decl) in p.buffers.iter().enumerate() {
        let shared_off = shared.offsets.get(&BufId(i as u32)).copied();
        writeln!(s, "  b{i} {}", buffer_str(decl, shared_off)).unwrap();
    }
    for (k, &shared_bytes) in p.kernels.iter().zip(&shared.per_kernel) {
        s.push('\n');
        dump_kernel(&mut s, k, shared_bytes);
    }
    s
}

fn buffer_str(decl: &BufferDecl, shared_off: Option<usize>) -> String {
    let kind = match (decl.kind, shared_off) {
        (BufferKind::Input(k), _) => format!("input#{k}"),
        (BufferKind::Output(k), _) => format!("output#{k}"),
        (BufferKind::Scratch { offset }, _) => format!("scratch@{offset}"),
        (BufferKind::Shared, Some(off)) => format!("shared@{off}"),
        (BufferKind::Shared, None) => "shared".into(),
        (BufferKind::Register, _) => "register".into(),
    };
    let mut s = format!(
        "\"{}\": {} {kind}",
        decl.name,
        tensor_ty(decl.elem, &decl.shape)
    );
    match &decl.layout {
        None => {}
        Some(l) if l.is_identity() => s.push_str(" layout=id"),
        Some(l) => write!(s, " layout={:?}", l.bases).unwrap(),
    }
    s
}

fn dump_kernel(s: &mut String, k: &Kernel, shared_bytes: usize) {
    let params = k
        .params
        .iter()
        .map(|&(buf, writable)| {
            let m = if writable { " mut" } else { "" };
            format!("b{}{m}", buf.0)
        })
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        s,
        "kernel {} grid[{}] block[{}] shared={shared_bytes}B params({params}) {{",
        k.name, k.grid.bound, k.block
    )
    .unwrap();
    writeln!(s, "^grid({}):", val(k.grid_var())).unwrap();
    dump_stmts(s, k, &k.grid.block.body, 1);
    writeln!(s, "}}").unwrap();
}

fn val(v: SSARes) -> String {
    format!("v{}", v.0)
}

fn vals(vs: &[SSARes]) -> String {
    vs.iter().map(|&v| val(v)).collect::<Vec<_>>().join(", ")
}

fn captures_str(op: &SSAOp, skip: usize) -> String {
    let caps = &op.operands[skip.min(op.operands.len())..];
    if caps.is_empty() {
        String::new()
    } else {
        format!(" captures({})", vals(caps))
    }
}

fn dump_stmts(s: &mut String, k: &Kernel, stmts: &[SSANode], depth: usize) {
    let pad = "  ".repeat(depth);
    for &sid in stmts {
        let op = k.op(sid);
        match &op.opcode {
            SSAOpCode::Alloc { buf } => {
                writeln!(s, "{pad}alloc b{}", buf.0).unwrap();
            }
            SSAOpCode::Sync => {
                writeln!(s, "{pad}sync").unwrap();
            }
            SSAOpCode::Loop { bound } => {
                writeln!(s, "{pad}loop[{bound}]{} {{", captures_str(op, 0)).unwrap();
                writeln!(s, "{pad}^{}:", val(op.block.operands[0])).unwrap();
                dump_stmts(s, k, &op.block.body, depth + 1);
                writeln!(s, "{pad}}}").unwrap();
            }
            SSAOpCode::Par {
                bound,
                spans_grid,
                attr,
                reads,
                writes,
            } => {
                let mut head = format!("{pad}par[{bound}]");
                if *spans_grid {
                    head.push_str(" spans_grid");
                }
                if !op.results.is_empty() {
                    write!(head, " -> ({})", vals(&op.results)).unwrap();
                }
                head.push_str(&captures_str(op, 0));
                if let Some(attr) = attr {
                    write!(head, " seq={}", attr.seq_size).unwrap();
                    if !attr.layout.is_identity() {
                        write!(head, " layout={:?}", attr.layout.bases).unwrap();
                    }
                }
                writeln!(s, "{head} {{").unwrap();
                dump_par_body(s, k, reads, writes, &op.block, depth);
                writeln!(s, "{pad}}}").unwrap();
            }
            other => unreachable!("scalar op {other:?} at statement level"),
        }
    }
}

fn dump_par_body(
    s: &mut String,
    k: &Kernel,
    reads: &[Access],
    writes: &[Access],
    block: &SSABlock,
    depth: usize,
) {
    let pad = "  ".repeat(depth);
    let inner = "  ".repeat(depth + 1);
    let vid = block.operands[0];
    writeln!(s, "{pad}^{}:", val(vid)).unwrap();
    for (i, access) in reads.iter().enumerate() {
        writeln!(
            s,
            "{inner}{} = load {}",
            val(block.operands[1 + i]),
            access_str(access, vid)
        )
        .unwrap();
    }
    dump_ops(s, k, &block.body, depth + 1);
    for (i, access) in writes.iter().enumerate() {
        writeln!(
            s,
            "{inner}store {} = {}",
            access_str(access, vid),
            val(block.yields[i])
        )
        .unwrap();
    }
}

fn dump_ops(s: &mut String, k: &Kernel, body: &[SSANode], depth: usize) {
    let pad = "  ".repeat(depth);
    for &nid in body {
        let op = k.op(nid);
        match &op.opcode {
            SSAOpCode::ConstU32(c) => {
                writeln!(s, "{pad}{} = const_u32 {c}", val(op.results[0])).unwrap();
            }
            SSAOpCode::ConstField(c) => {
                writeln!(s, "{pad}{} = const_field {c}", val(op.results[0])).unwrap();
            }
            SSAOpCode::Bin(bop, ty) => {
                let ty = match ty {
                    ScalarType::BabyBear => "bb",
                    ScalarType::U32 => "u32",
                    ScalarType::Bool => "bool",
                };
                writeln!(
                    s,
                    "{pad}{} = {}.{ty} {}, {}",
                    val(op.results[0]),
                    bin_name(*bop),
                    val(op.operands[0]),
                    val(op.operands[1])
                )
                .unwrap();
            }
            SSAOpCode::Select => {
                writeln!(
                    s,
                    "{pad}{} = select {}, {}, {}",
                    val(op.results[0]),
                    val(op.operands[0]),
                    val(op.operands[1]),
                    val(op.operands[2])
                )
                .unwrap();
            }
            SSAOpCode::Loop { bound } => {
                // Inline scf.for: operands are the carried inits then the
                // captures; block operands are [induction var, carried...].
                let n = op.results.len();
                let iter = op.block.operands[1..]
                    .iter()
                    .zip(&op.operands[..n])
                    .map(|(&c, &i)| format!("{} = {}", val(c), val(i)))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(
                    s,
                    "{pad}({}) = for[{bound}] ^{} iter({iter}){} {{",
                    vals(&op.results),
                    val(op.block.operands[0]),
                    captures_str(op, n)
                )
                .unwrap();
                dump_ops(s, k, &op.block.body, depth + 1);
                writeln!(s, "{pad}  yield({})", vals(&op.block.yields)).unwrap();
                writeln!(s, "{pad}}}").unwrap();
            }
            other @ (SSAOpCode::Par { .. } | SSAOpCode::Alloc { .. } | SSAOpCode::Sync) => {
                unreachable!("statement-level op {other:?} inside a par block")
            }
        }
    }
}

fn access_str(access: &Access, vid: SSARes) -> String {
    let index = match &access.index {
        IndexMap::Linear(ll) if ll.is_identity() => val(vid),
        IndexMap::Linear(ll) => format!("layout{:?}({})", ll.bases, val(vid)),
        IndexMap::Affine { expr, bounds } => quast_str(expr, bounds),
    };
    format!("b{}[{index}]", access.buf.0)
}

/// Prints a quast as C-style arithmetic (with `%` recovery); falls back to
/// the raw tree when emission fails.
fn quast_str(expr: &Quast, bounds: &BTreeMap<VarId, u64>) -> String {
    expr.emit(bounds, &mut CStrEmitter)
        .unwrap_or_else(|_| quast_raw(expr))
}

fn quast_raw(q: &Quast) -> String {
    match q {
        Quast::Sym(v) => format!("v{}", v.0),
        Quast::Const(c) => format!("{c}"),
        Quast::Add(a, b) => format!("({} + {})", quast_raw(a), quast_raw(b)),
        Quast::Mul(a, c) => format!("({} * {c})", quast_raw(a)),
        Quast::FloorDiv(a, c) => format!("({} / {c})", quast_raw(a)),
        Quast::Neg(a) => format!("(-{})", quast_raw(a)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{IRBuilder, ReduceOp, ScalarType},
        passes::{insert_sync, layout_infer, plan_shared_mem, utils::test_util},
    };

    /// Shared-memory tile: outer compute of a let-bound inner tile plus a
    /// reversed reader (same module as passes.rs's shared-tile test).
    fn tile_module() -> Module {
        let (blocks, t) = (4usize, 8usize);
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![blocks * t]);
        let body = b.compute(blocks, |b, i| {
            let tile = b.compute(t, |b, j| {
                let tc = b.const_u32(t as u32);
                let base = b.mul(i, tc);
                let ix = b.add(base, j);
                b.index(a, &[ix])
            });
            b.bind(tile, |b, tile| {
                b.compute(t, |b, j| {
                    let last = b.const_u32(t as u32 - 1);
                    let rev = b.sub(last, j);
                    b.index(tile, &[rev])
                })
            })
        });
        b.finish("tile", body)
    }

    /// Load-free reduce: stays inline as an scf.for inside the par.
    fn inline_reduce_module() -> Module {
        let mut b = IRBuilder::new();
        let body = b.compute(8, |b, i| b.reduce(ReduceOp::Add, 4, |b, j| b.mul(i, j)));
        b.finish("inline_reduce", body)
    }

    fn lowered(module: Module) -> KernelProgram {
        let mut kprog = test_util::lowered(module);
        layout_infer(&mut kprog);
        insert_sync(&mut kprog);
        kprog
    }

    #[test]
    fn hir_dump_expression_form() {
        let module = tile_module();
        let hir = dump_hir(&module);
        assert!(hir.starts_with("module tile\n"), "{hir}");
        assert!(hir.contains("input0 \"a\": BabyBear[32]"), "{hir}");
        assert!(hir.contains("compute[4] |v0| {"), "{hir}");
        assert!(
            hir.contains("let v2 = compute[8] |v1| { a[v0 * 8 + v1] } in"),
            "{hir}"
        );
        assert!(hir.contains("compute[8] |v3| { v2[7 - v3] }"), "{hir}");
    }

    /// Multi-use nodes are bound as `%id =` definitions in the innermost
    /// scope that binds their free variables and referenced by name.
    #[test]
    fn hir_dump_names_shared_subexpressions() {
        let mut b = IRBuilder::new();
        let body = b.compute(4, |b, i| {
            let sq = b.mul(i, i);
            let a2 = b.add(sq, sq);
            b.mul(a2, a2)
        });
        let module = b.finish("shared", body);
        let hir = dump_hir(&module);
        assert!(hir.contains("%1 = v0 * v0"), "{hir}");
        assert!(hir.contains("%2 = %1 + %1"), "{hir}");
        assert!(hir.contains("%2 * %2"), "{hir}");
    }

    #[test]
    fn hir_dump_shows_scatter() {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![12]);
        let sc = b.scatter_map(1, Some(vec![4, 3]), |p, _| {
            vec![p[0].rem_c(4), p[0].floordiv(4)]
        });
        let body = b.compute_scatter(12, sc, |b, i| b.index(a, &[i]));
        let module = b.finish("scatter_mod", body);
        let hir = dump_hir(&module);
        assert!(hir.contains("scatter (v0) ->"), "{hir}");
        assert!(hir.contains("into [4, 3]"), "{hir}");
        assert!(hir.contains("{ a[v1] }"), "{hir}");
    }

    /// Sharing must keep the dump linear in the number of unique nodes even
    /// for deeply shared DAGs (poseidon2 rounds reuse every state element).
    #[test]
    fn hir_dump_merkle_is_linear() {
        let constants = crate::kernels::Poseidon2Constants::p3_default();
        let module = crate::kernels::merkle_tree_module(8, &constants);
        let hir = dump_hir(&module);
        assert!(hir.len() < 2_000_000, "dump blew up: {} bytes", hir.len());
    }

    #[test]
    fn kir_dump_shows_tile_kernel() {
        let kprog = lowered(tile_module());
        let kir = dump_kernel_ir(&kprog, &plan_shared_mem(&kprog));
        assert!(kir.starts_with("program tile"), "{kir}");
        assert!(kir.contains("buffers:"), "{kir}");
        assert!(kir.contains("input#0"), "{kir}");
        assert!(kir.contains("output#0"), "{kir}");
        // One 8-element shared tile: 32 bytes at offset 0.
        assert!(kir.contains(" shared@0"), "{kir}");
        assert!(kir.contains("layout=id"), "{kir}");
        assert!(kir.contains("kernel "), "{kir}");
        assert!(kir.contains("grid[4] block[8] shared=32B"), "{kir}");
        assert!(kir.contains("^grid(v0):"), "{kir}");
        assert!(kir.contains("alloc b"), "{kir}");
        assert!(kir.contains("par[8]"), "{kir}");
        // The barrier between the tile producer and its consumer.
        assert!(kir.contains("\n  sync\n"), "{kir}");
        // Both pars capture the grid var and mirror their write as a result.
        assert!(kir.contains("captures(v0)"), "{kir}");
        assert!(kir.contains(" -> (v"), "{kir}");
        assert!(kir.contains(" = load b"), "{kir}");
        assert!(kir.contains("store b"), "{kir}");
    }

    #[test]
    fn kir_dump_shows_inline_for() {
        let kprog = lowered(inline_reduce_module());
        let kir = dump_kernel_ir(&kprog, &plan_shared_mem(&kprog));
        assert!(kir.contains("for[4]"), "{kir}");
        assert!(kir.contains("iter(v"), "{kir}");
        assert!(kir.contains("yield(v"), "{kir}");
        assert!(kir.contains("mul.u32"), "{kir}");
        assert!(kir.contains("add.u32"), "{kir}");
    }
}
