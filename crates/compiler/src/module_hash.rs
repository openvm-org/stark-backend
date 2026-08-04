//! Stable structural fingerprint of an [`ir::Module`].
//!
//! Rust's `Hash` trait gives values whose bytes depend on the concrete
//! `Hasher` in use and on `HashMap`'s randomized seed. That's fine for
//! in-memory dedup, but the on-disk kernel cache in [`crate::kernel_cache`]
//! needs a key that's identical across processes, runs, and target
//! architectures. This module walks a module's reachable nodes in a fixed
//! canonical order and feeds their bytes into SHA-3-256; the resulting
//! digest doubles as the cache directory name.
//!
//! The hash covers everything the compiler pipeline can observe on the
//! module: the module name, every input declaration (name, scalar type,
//! shape), and every reachable [`ir::Node`] reached from `body`, including
//! `Compute`'s optional `scatter` / `par` attributes and `threads` hint.
//! `NodeId` operands are rewritten into a canonical DAG order so two
//! structurally identical modules with different arena layouts produce the
//! same digest.

use sha3::{Digest, Sha3_256};

use crate::{
    ir::{BinOp, IRBuilder, Module, Node, NodeId, ReduceOp, ScalarType, VarId},
    quast::{ParSpec, Quast, SExpr, Scatter, SymConst},
};

/// Deterministic 32-byte structural fingerprint of `module`.
pub fn module_hash(module: &Module) -> [u8; 32] {
    let mut h = Hasher::new();
    h.str(&module.name);
    // Param declarations: count and VarIds (α-relevant identity), NOT names
    // or shape-hint bindings — those are metadata, not semantics.
    let params = module.builder.params();
    h.u64(params.len() as u64);
    for (v, _name) in params {
        h.u64(v.0 as u64);
    }
    // The block hint is semantic (it fixes the launch geometry of
    // symbolic-bound kernels), unlike the shape hint: two residuals that
    // differ only in block size are distinct compiled variants.
    match module.builder.block_hint() {
        None => h.tag(0),
        Some(block) => {
            h.tag(1);
            h.u64(block as u64);
        }
    }
    let inputs = module.builder.inputs();
    h.u64(inputs.len() as u64);
    for decl in inputs {
        h.str(&decl.name);
        h.scalar_ty(decl.elem);
        h.u64(decl.shape.len() as u64);
        for d in &decl.shape {
            h.sexpr(d);
        }
    }
    let mut ctx = WalkCtx {
        b: &module.builder,
        order: Vec::new(),
        seen: std::collections::HashMap::new(),
    };
    let root = ctx.number(module.body);
    h.u64(root as u64);
    h.u64(ctx.order.len() as u64);
    for (i, &node_id) in ctx.order.iter().enumerate() {
        h.u64(i as u64);
        h.node(&ctx, node_id);
    }
    h.finish()
}

/// Human-friendly hex form of [`module_hash`].
pub fn module_hash_hex(module: &Module) -> String {
    hex::encode(module_hash(module))
}

/// Rewrites the module's `NodeId`s and `VarId`s into a canonical, insertion-
/// order-free numbering: the root gets the largest index; children are
/// visited depth-first, left-to-right; each node is emitted the first time
/// its unique post-order index would be assigned.
struct WalkCtx<'a> {
    b: &'a IRBuilder,
    order: Vec<NodeId>,
    seen: std::collections::HashMap<NodeId, u32>,
}

impl WalkCtx<'_> {
    fn number(&mut self, id: NodeId) -> u32 {
        if let Some(&n) = self.seen.get(&id) {
            return n;
        }
        // Number children first (post-order) — but we must not recurse before
        // the entry is present, otherwise cycles through hash-consed shared
        // nodes would double-visit. Since the IR is a DAG (no cycles), a
        // straight post-order works.
        for child in children_of(self.b.node(id)) {
            self.number(child);
        }
        let n = self.order.len() as u32;
        self.order.push(id);
        self.seen.insert(id, n);
        n
    }

    fn n(&self, id: NodeId) -> u32 {
        self.seen[&id]
    }
}

pub(crate) fn children_of(node: &Node) -> Vec<NodeId> {
    match node {
        Node::Input(_)
        | Node::Var(_)
        | Node::ConstU32(_)
        | Node::ConstField(_)
        | Node::ConstFpExt(_)
        | Node::ConstSym(_) => vec![],
        Node::LiftFpExt(x) => vec![*x],
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
        Node::Proj(t, _) => vec![*t],
    }
}

struct Hasher(Sha3_256);

impl Hasher {
    fn new() -> Self {
        Hasher(Sha3_256::new())
    }

    fn tag(&mut self, t: u8) {
        self.0.update([t]);
    }

    fn u64(&mut self, v: u64) {
        self.0.update(v.to_le_bytes());
    }

    fn i64(&mut self, v: i64) {
        self.0.update(v.to_le_bytes());
    }

    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.update(s.as_bytes());
    }

    fn scalar_ty(&mut self, t: ScalarType) {
        let byte: u8 = match t {
            ScalarType::BabyBear => 0,
            ScalarType::FpExt => 1,
            ScalarType::U32 => 2,
            ScalarType::Bool => 3,
        };
        self.0.update([byte]);
    }

    fn bin_op(&mut self, op: BinOp) {
        let byte: u8 = match op {
            BinOp::Add => 0,
            BinOp::Sub => 1,
            BinOp::Mul => 2,
            BinOp::Div => 3,
            BinOp::Rem => 4,
            BinOp::Lt => 5,
            BinOp::Le => 6,
            BinOp::Eq => 7,
        };
        self.0.update([byte]);
    }

    fn reduce_op(&mut self, op: ReduceOp) {
        let byte: u8 = match op {
            ReduceOp::Add => 0,
            ReduceOp::Mul => 1,
        };
        self.0.update([byte]);
    }

    fn sym_const(&mut self, c: &SymConst) {
        match c {
            SymConst::Lit(l) => {
                self.tag(0);
                self.i64(*l);
            }
            SymConst::Sym(v) => {
                self.tag(1);
                self.u64(v.0 as u64);
            }
        }
    }

    fn sexpr(&mut self, e: &SExpr) {
        match e {
            SExpr::Sym(v) => {
                self.tag(0);
                self.u64(v.0 as u64);
            }
            SExpr::Const(c) => {
                self.tag(1);
                self.sym_const(c);
            }
            SExpr::Add(a, b) => {
                self.tag(2);
                self.sexpr(a);
                self.sexpr(b);
            }
            SExpr::Mul(a, c) => {
                self.tag(3);
                self.sexpr(a);
                self.sym_const(c);
            }
            SExpr::FloorDiv(a, c) => {
                self.tag(4);
                self.sexpr(a);
                self.sym_const(c);
            }
            SExpr::Neg(a) => {
                self.tag(5);
                self.sexpr(a);
            }
        }
    }

    fn quast(&mut self, q: &Quast) {
        match q {
            Quast::Sym(v) => {
                self.tag(0);
                self.u64(v.0 as u64);
            }
            Quast::Const(c) => {
                self.tag(1);
                self.i64(*c);
            }
            Quast::Add(a, b) => {
                self.tag(2);
                self.quast(a);
                self.quast(b);
            }
            Quast::Mul(a, c) => {
                self.tag(3);
                self.quast(a);
                self.i64(*c);
            }
            Quast::FloorDiv(a, c) => {
                self.tag(4);
                self.quast(a);
                self.i64(*c);
            }
            Quast::Neg(a) => {
                self.tag(5);
                self.quast(a);
            }
        }
    }

    fn scatter(&mut self, s: &Scatter) {
        self.u64(s.params.len() as u64);
        for p in &s.params {
            self.u64(p.0 as u64);
        }
        self.u64(s.exprs.len() as u64);
        for e in &s.exprs {
            self.quast(e);
        }
        self.u64(s.inv_params.len() as u64);
        for p in &s.inv_params {
            self.u64(p.0 as u64);
        }
        self.u64(s.inv_exprs.len() as u64);
        for e in &s.inv_exprs {
            self.quast(e);
        }
        match &s.out_shape {
            None => self.tag(0),
            Some(sh) => {
                self.tag(1);
                self.u64(sh.len() as u64);
                for &d in sh {
                    self.u64(d as u64);
                }
            }
        }
        self.u64(s.bounds.len() as u64);
        for (v, b) in &s.bounds {
            self.u64(v.0 as u64);
            self.u64(*b);
        }
    }

    fn par(&mut self, p: &ParSpec) {
        self.u64(p.thread.0 as u64);
        self.u64(p.seq.0 as u64);
        self.quast(&p.expr);
    }

    fn node(&mut self, ctx: &WalkCtx, id: NodeId) {
        let node = ctx.b.node(id);
        match node {
            Node::Input(k) => {
                self.tag(0);
                self.u64(*k as u64);
            }
            Node::Var(v) => {
                self.tag(1);
                self.u64(v.0 as u64);
            }
            Node::ConstU32(c) => {
                self.tag(2);
                self.u64(*c as u64);
            }
            Node::ConstField(c) => {
                self.tag(3);
                self.u64(*c as u64);
            }
            Node::ConstFpExt(c) => {
                self.tag(4);
                for &x in c {
                    self.u64(x as u64);
                }
            }
            Node::ConstSym(e) => {
                self.tag(15);
                self.sexpr(e);
            }
            Node::LiftFpExt(x) => {
                self.tag(5);
                self.u64(ctx.n(*x) as u64);
            }
            Node::Bin(op, a, b) => {
                self.tag(6);
                self.bin_op(*op);
                self.u64(ctx.n(*a) as u64);
                self.u64(ctx.n(*b) as u64);
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                self.tag(7);
                self.u64(ctx.n(*cond) as u64);
                self.u64(ctx.n(*then_val) as u64);
                self.u64(ctx.n(*else_val) as u64);
            }
            Node::Index { tensor, indices } => {
                self.tag(8);
                self.u64(ctx.n(*tensor) as u64);
                self.u64(indices.len() as u64);
                for i in indices {
                    self.u64(ctx.n(*i) as u64);
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
                self.tag(9);
                self.sexpr(bound);
                self.u64(var.0 as u64);
                self.u64(ctx.n(*body) as u64);
                match scatter {
                    None => self.tag(0),
                    Some(s) => {
                        self.tag(1);
                        self.scatter(s);
                    }
                }
                match par {
                    None => self.tag(0),
                    Some(p) => {
                        self.tag(1);
                        self.par(p);
                    }
                }
                match threads {
                    None => self.tag(0),
                    Some(t) => {
                        self.tag(1);
                        self.u64(*t as u64);
                    }
                }
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                self.tag(10);
                self.reduce_op(*op);
                self.sexpr(bound);
                self.u64(var.0 as u64);
                self.u64(ctx.n(*body) as u64);
            }
            Node::Let { var, value, body } => {
                self.tag(11);
                self.u64(var.0 as u64);
                self.u64(ctx.n(*value) as u64);
                self.u64(ctx.n(*body) as u64);
            }
            Node::Tuple(e) => {
                self.tag(12);
                self.u64(e.len() as u64);
                for id in e {
                    self.u64(ctx.n(*id) as u64);
                }
            }
            Node::Proj(t, k) => {
                self.tag(13);
                self.u64(ctx.n(*t) as u64);
                self.u64(*k as u64);
            }
            Node::Pack(e) => {
                self.tag(14);
                self.u64(e.len() as u64);
                for id in e {
                    self.u64(ctx.n(*id) as u64);
                }
            }
        }
    }

    fn finish(self) -> [u8; 32] {
        self.0.finalize().into()
    }
}

// Kill the unused-import warning until `VarId` gains a use inside the module.
const _: fn() = || {
    let _: VarId = VarId(0);
    let _: fn(&IRBuilder) = |_| {};
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRBuilder, ScalarType};

    fn scale_by_two() -> Module {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let body = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            let two = b.const_field(2);
            b.mul(ai, two)
        });
        b.finish("scale_by_two", body)
    }

    #[test]
    fn same_module_same_hash() {
        let a = scale_by_two();
        let b = scale_by_two();
        assert_eq!(module_hash(&a), module_hash(&b));
    }

    #[test]
    fn different_name_different_hash() {
        let a = scale_by_two();
        let mut b = scale_by_two();
        b.name = "scale_by_three".into();
        assert_ne!(module_hash(&a), module_hash(&b));
    }

    #[test]
    fn different_body_different_hash() {
        let a = scale_by_two();
        let three_module = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![4]);
            let body = b.compute(4, |b, i| {
                let ai = b.index(a, &[i]);
                let three = b.const_field(3);
                b.mul(ai, three)
            });
            b.finish("scale_by_two", body)
        };
        assert_ne!(module_hash(&a), module_hash(&three_module));
    }

    #[test]
    fn hex_form_is_64_lowercase_chars() {
        let m = scale_by_two();
        let hex = module_hash_hex(&m);
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
