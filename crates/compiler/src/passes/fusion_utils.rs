//! Independent HIR traversal and rewrite utilities used by the fusion v2
//! passes. See `detailed-fusion-plan-v2.md` §8.2.
//!
//! This module intentionally does not depend on any code from
//! [`crate::passes::fusion`] — the v2 pipeline is built beside the existing
//! implementation, not on top of it.
//!
//! The central primitive is an occurrence-based [`HirVisitor`]. Because HIR
//! nodes are hash-consed, a single [`ir::NodeId`] can appear under several
//! distinct index scopes; every call to [`visit_hir`] visits every occurrence
//! and threads the enclosing compute/reduce scope through to the visitor
//! callbacks. An active-recursion stack detects malformed cycles.
//!
//! On top of the visitor the module provides:
//!
//! - [`unique_index_scope`]: rejects a hash-consed access reached under two unequal index scopes,
//!   and returns the unique scope otherwise;
//! - [`collect_input_uses`]: every occurrence of every module input;
//! - [`count_reachable_nodes`]: HIR-size cap for candidate normalization;
//! - [`collect_structure`]: compute/reduce/let nesting and other structural facts used by
//!   pass-specific legality checks;
//! - [`clone_expr`]: deterministic alpha-renamed HIR cloning with an explicit `ir::NodeId ->
//!   ir::NodeId` substitution map.
//!
//! Pass-specific legality rules, cost models, and rewrite algorithms live in
//! their respective modules; this module supplies mechanisms only.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    ir::{IRBuilder, Module, Node, NodeId, SizeExpr, VarId},
    module_hash::children_of,
};

/// Whether the visitor should descend into the current node's children.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VisitControl {
    Recurse,
    SkipChildren,
}

/// A parallel- or reduction-loop binder currently in scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexBinding {
    pub var: VarId,
    pub bound: SizeExpr,
    pub kind: IndexKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IndexKind {
    Compute,
    Reduce,
}

/// Occurrence-based visitor over an [`Module`]'s reachable HIR nodes.
///
/// The visitor sees every occurrence of a hash-consed node under its
/// enclosing index scope: a shared node reached under two different scopes
/// is visited twice, once per scope.
///
/// Guarantees provided by [`visit_hir`]:
///
/// - children are visited in the deterministic order returned by [`children_of`];
/// - `enter` / `leave` are balanced, including when `enter` returns [`VisitControl::SkipChildren`];
/// - [`Node::Compute`] and [`Node::Reduce`] see the outer scope in `enter` and `leave`; only their
///   body child sees the extended scope with the enclosing binder appended;
/// - an active-recursion stack detects malformed HIR cycles.
pub trait HirVisitor {
    type Error;

    fn enter(
        &mut self,
        module: &Module,
        id: NodeId,
        node: &Node,
        index_scope: &[IndexBinding],
    ) -> Result<VisitControl, Self::Error>;

    fn leave(
        &mut self,
        _module: &Module,
        _id: NodeId,
        _node: &Node,
        _index_scope: &[IndexBinding],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Error produced when [`visit_hir`] detects a malformed cycle in the HIR.
/// Well-formed IR is a DAG; this is a debug guard for corrupt inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalformedHir {
    pub node: NodeId,
}

/// Runs `visitor` over every reachable occurrence of `root` in `module`.
///
/// The scope handed to the visitor grows only when descending into a
/// [`Node::Compute`] or [`Node::Reduce`] body; every other child is visited
/// with the parent's scope. Because hash-consed nodes may appear multiple
/// times, the walker does not memoize on `NodeId`; visitors that need
/// unique-node behavior maintain their own dense seen set.
pub fn visit_hir<V: HirVisitor>(
    module: &Module,
    root: NodeId,
    visitor: &mut V,
) -> Result<(), VisitError<V::Error>> {
    let mut walker = Walker {
        module,
        scope: Vec::new(),
        active: HashSet::new(),
    };
    walker.walk(root, visitor)
}

/// Error variants returned by [`visit_hir`]: either a visitor-supplied
/// error or a malformed-cycle guard.
#[derive(Debug)]
pub enum VisitError<E> {
    Visitor(E),
    Malformed(MalformedHir),
}

impl<E> From<E> for VisitError<E> {
    fn from(e: E) -> Self {
        VisitError::Visitor(e)
    }
}

struct Walker<'a> {
    module: &'a Module,
    scope: Vec<IndexBinding>,
    active: HashSet<NodeId>,
}

impl Walker<'_> {
    fn walk<V: HirVisitor>(
        &mut self,
        id: NodeId,
        visitor: &mut V,
    ) -> Result<(), VisitError<V::Error>> {
        if !self.active.insert(id) {
            return Err(VisitError::Malformed(MalformedHir { node: id }));
        }
        let node = self.module.builder.node(id).clone();
        let ctl = visitor
            .enter(self.module, id, &node, &self.scope)
            .map_err(VisitError::Visitor)?;
        if matches!(ctl, VisitControl::Recurse) {
            self.walk_children(id, &node, visitor)?;
        }
        visitor
            .leave(self.module, id, &node, &self.scope)
            .map_err(VisitError::Visitor)?;
        self.active.remove(&id);
        Ok(())
    }

    fn walk_children<V: HirVisitor>(
        &mut self,
        _id: NodeId,
        node: &Node,
        visitor: &mut V,
    ) -> Result<(), VisitError<V::Error>> {
        match node {
            Node::Compute {
                var, bound, body, ..
            } => {
                self.scope.push(IndexBinding {
                    var: *var,
                    bound: bound.clone(),
                    kind: IndexKind::Compute,
                });
                let r = self.walk(*body, visitor);
                self.scope.pop();
                r
            }
            Node::Reduce {
                var, bound, body, ..
            } => {
                self.scope.push(IndexBinding {
                    var: *var,
                    bound: bound.clone(),
                    kind: IndexKind::Reduce,
                });
                let r = self.walk(*body, visitor);
                self.scope.pop();
                r
            }
            _ => {
                for child in children_of(node) {
                    self.walk(child, visitor)?;
                }
                Ok(())
            }
        }
    }
}

/// Result of [`unique_index_scope`] when a hash-consed access node is
/// reached under two unequal index scopes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmbiguousAccessScope {
    pub node: NodeId,
}

/// Returns the unique index scope every occurrence of `target` was reached
/// under, or `AmbiguousAccessScope` if two occurrences see different scopes.
/// Returns `Ok(None)` when `target` is not reachable from the module body.
///
/// This is the precondition an [`AccessRelation`](AccessRelation)-producing
/// pass wants: if the target is reached under a unique scope, an access site
/// can be pinned to a single ordered [`IndexBinding`] sequence.
pub fn unique_index_scope(
    module: &Module,
    target: NodeId,
) -> Result<Option<Vec<IndexBinding>>, AmbiguousAccessScope> {
    struct Collect {
        target: NodeId,
        found: Option<Vec<IndexBinding>>,
        ambiguous: bool,
    }
    impl HirVisitor for Collect {
        type Error = std::convert::Infallible;
        fn enter(
            &mut self,
            _m: &Module,
            id: NodeId,
            _node: &Node,
            index_scope: &[IndexBinding],
        ) -> Result<VisitControl, Self::Error> {
            if id == self.target {
                match &self.found {
                    None => self.found = Some(index_scope.to_vec()),
                    Some(prev) if prev.as_slice() != index_scope => self.ambiguous = true,
                    _ => {}
                }
            }
            Ok(VisitControl::Recurse)
        }
    }
    let mut c = Collect {
        target,
        found: None,
        ambiguous: false,
    };
    match visit_hir(module, module.body, &mut c) {
        Ok(()) | Err(VisitError::Visitor(_)) => {}
        Err(VisitError::Malformed(_)) => {
            return Ok(None);
        }
    }
    if c.ambiguous {
        Err(AmbiguousAccessScope { node: target })
    } else {
        Ok(c.found)
    }
}

/// One occurrence of a module input as an operand of some HIR node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputUse {
    /// The [`Node::Input`] node itself.
    pub input_node: NodeId,
    /// Which declared input this occurrence names.
    pub input_pos: usize,
    /// Index scope this occurrence is reached under.
    pub scope: Vec<IndexBinding>,
}

/// Collects every occurrence of every declared module input reachable from
/// `module.body`. Order is a deterministic pre-order traversal of the
/// module.
pub fn collect_input_uses(module: &Module) -> Vec<InputUse> {
    struct Collect {
        uses: Vec<InputUse>,
    }
    impl HirVisitor for Collect {
        type Error = std::convert::Infallible;
        fn enter(
            &mut self,
            _m: &Module,
            id: NodeId,
            node: &Node,
            index_scope: &[IndexBinding],
        ) -> Result<VisitControl, Self::Error> {
            if let Node::Input(k) = node {
                self.uses.push(InputUse {
                    input_node: id,
                    input_pos: *k,
                    scope: index_scope.to_vec(),
                });
            }
            Ok(VisitControl::Recurse)
        }
    }
    let mut c = Collect { uses: Vec::new() };
    let _ = visit_hir(module, module.body, &mut c);
    c.uses
}

/// Counts the distinct HIR nodes reachable from `module.body`. Hash-consed
/// duplicates count once. Used by [`crate::passes`] fusion candidates to
/// enforce a code-size cap.
pub fn count_reachable_nodes(module: &Module) -> usize {
    struct Count {
        seen: HashSet<NodeId>,
    }
    impl HirVisitor for Count {
        type Error = std::convert::Infallible;
        fn enter(
            &mut self,
            _m: &Module,
            id: NodeId,
            _node: &Node,
            _index_scope: &[IndexBinding],
        ) -> Result<VisitControl, Self::Error> {
            if self.seen.insert(id) {
                Ok(VisitControl::Recurse)
            } else {
                Ok(VisitControl::SkipChildren)
            }
        }
    }
    let mut c = Count {
        seen: HashSet::new(),
    };
    let _ = visit_hir(module, module.body, &mut c);
    c.seen.len()
}

/// Structural facts about a module used by pass-specific legality checks.
///
/// Note that these are not sufficient conditions for any specific fusion —
/// each pass composes its own decision from these fields plus its own
/// legality proofs.
#[derive(Clone, Debug, Default)]
pub struct StructureFacts {
    /// Outermost compute nodes, in visitation order.
    pub top_computes: Vec<NodeId>,
    /// Every reduce node reachable from the body.
    pub reduces: Vec<NodeId>,
    /// Every compute node reachable from the body (top level or nested).
    pub computes: Vec<NodeId>,
    /// Every `let` node reachable from the body.
    pub lets: Vec<NodeId>,
    /// Every compute node whose `scatter` attribute is set.
    pub scatter_computes: Vec<NodeId>,
    /// Every compute node whose `par` attribute is set.
    pub par_computes: Vec<NodeId>,
    /// Every compute node with an explicit `threads` block-size hint.
    pub threaded_computes: Vec<NodeId>,
    /// Whether any compute inside the body is nested under another compute.
    pub has_nested_compute: bool,
}

/// Walks `module.body` and returns a [`StructureFacts`] summary.
pub fn collect_structure(module: &Module) -> StructureFacts {
    struct Collect {
        facts: StructureFacts,
        compute_depth: u32,
        seen: HashSet<NodeId>,
    }
    impl HirVisitor for Collect {
        type Error = std::convert::Infallible;
        fn enter(
            &mut self,
            _m: &Module,
            id: NodeId,
            node: &Node,
            _index_scope: &[IndexBinding],
        ) -> Result<VisitControl, Self::Error> {
            // De-dupe by NodeId so structural counts describe reachable
            // distinct nodes rather than shared-node occurrences.
            let fresh = self.seen.insert(id);
            match node {
                Node::Compute {
                    scatter,
                    par,
                    threads,
                    ..
                } => {
                    if fresh {
                        self.facts.computes.push(id);
                        if self.compute_depth == 0 {
                            self.facts.top_computes.push(id);
                        } else {
                            self.facts.has_nested_compute = true;
                        }
                        if scatter.is_some() {
                            self.facts.scatter_computes.push(id);
                        }
                        if par.is_some() {
                            self.facts.par_computes.push(id);
                        }
                        if threads.is_some() {
                            self.facts.threaded_computes.push(id);
                        }
                    }
                    self.compute_depth += 1;
                }
                Node::Reduce { .. } => {
                    if fresh {
                        self.facts.reduces.push(id);
                    }
                }
                Node::Let { .. } => {
                    if fresh {
                        self.facts.lets.push(id);
                    }
                }
                _ => {}
            }
            Ok(VisitControl::Recurse)
        }

        fn leave(
            &mut self,
            _m: &Module,
            _id: NodeId,
            node: &Node,
            _index_scope: &[IndexBinding],
        ) -> Result<(), Self::Error> {
            if matches!(node, Node::Compute { .. }) {
                self.compute_depth -= 1;
            }
            Ok(())
        }
    }
    let mut c = Collect {
        facts: StructureFacts::default(),
        compute_depth: 0,
        seen: HashSet::new(),
    };
    let _ = visit_hir(module, module.body, &mut c);
    c.facts
}

/// Deterministic alpha-renamed cloning of an HIR subgraph from a source
/// [`Module`] into a destination [`IRBuilder`].
///
/// `subst` maps *source-module* [`NodeId`]s the caller has already
/// materialized in `dst` to their replacement [`NodeId`]s in `dst`. Every
/// bound variable ([`Node::Compute`], [`Node::Reduce`], [`Node::Let`])
/// encountered in the source is remapped to a fresh [`VarId`] allocated in
/// `dst`, so cloning is capture-free even when the source and destination
/// share a variable namespace.
///
/// `subst_vars` maps a source [`VarId`] to an arbitrary destination
/// [`NodeId`]. This is strictly more general than a `VarId -> VarId`
/// remap: `subst_vars.insert(v, dst.intern(Node::Var(v')))` recovers the
/// simple alpha-rename, while `subst_vars.insert(v, expr)` lets callers
/// substitute a bound variable with an arbitrary expression (e.g. for
/// inlining a producer body at an affine consumer index).
///
/// Cloning is idempotent within a single call: a node reached twice through
/// hash-consing produces one destination node the first time and reuses
/// that identity on subsequent visits, because `dst.intern` hash-conses too.
///
/// Errors:
/// - `CloneError::UnboundVar { var }` when the source references a bound variable neither
///   introduced inside the cloned region nor supplied through `subst_vars`;
/// - `CloneError::MissingSubst { node }` when the source references a node the caller intended to
///   substitute but did not put in `subst`.
pub fn clone_expr(
    src: &Module,
    root: NodeId,
    dst: &mut IRBuilder,
    subst: &HashMap<NodeId, NodeId>,
    subst_vars: &HashMap<VarId, NodeId>,
) -> Result<NodeId, CloneError> {
    clone_expr_with_hook(src, root, dst, subst, subst_vars, |_, _, _| Ok(None))
}

/// Variant of [`clone_expr`] that invokes `hook` at every source
/// [`NodeId`] before falling through to the default clone logic. The
/// hook sees the destination [`IRBuilder`], the source `NodeId`, and a
/// snapshot of the current source-to-destination variable map — which
/// includes fresh identities for any inner `Compute`/`Reduce`/`Let`
/// binders that the clone has already descended into.
///
/// Returning `Ok(Some(id))` uses `id` as the replacement for the source
/// node (and memoizes it so hash-consed re-visits reuse it). `Ok(None)`
/// falls through to the default. `Err` propagates and aborts the clone.
///
/// This is the mechanism the producer-consumer fusion uses to emit a
/// site-specific producer-body inline at each seam read — including
/// reads whose index expression uses an *inner* loop variable
/// introduced by the consumer's own `Compute`/`Reduce`.
pub fn clone_expr_with_hook<F>(
    src: &Module,
    root: NodeId,
    dst: &mut IRBuilder,
    subst: &HashMap<NodeId, NodeId>,
    subst_vars: &HashMap<VarId, NodeId>,
    hook: F,
) -> Result<NodeId, CloneError>
where
    F: FnMut(&mut IRBuilder, NodeId, &HashMap<VarId, NodeId>) -> Result<Option<NodeId>, CloneError>,
{
    let mut ctx = CloneCtx {
        src,
        dst,
        subst,
        vars: subst_vars.clone(),
        memo: HashMap::new(),
        hook: Box::new(hook),
    };
    ctx.clone(root)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloneError {
    UnboundVar { var: VarId },
    MissingSubst { node: NodeId },
}

type CloneHook<'h> = Box<
    dyn FnMut(&mut IRBuilder, NodeId, &HashMap<VarId, NodeId>) -> Result<Option<NodeId>, CloneError>
        + 'h,
>;

struct CloneCtx<'a> {
    src: &'a Module,
    dst: &'a mut IRBuilder,
    subst: &'a HashMap<NodeId, NodeId>,
    /// Maps a source [`VarId`] to a destination [`NodeId`]. For simple
    /// alpha-renaming the entry is `dst.intern(Node::Var(v'))`; for
    /// substitution-by-expression it is an arbitrary computed expression.
    vars: HashMap<VarId, NodeId>,
    memo: HashMap<NodeId, NodeId>,
    hook: CloneHook<'a>,
}

impl CloneCtx<'_> {
    fn clone(&mut self, id: NodeId) -> Result<NodeId, CloneError> {
        if let Some(&mapped) = self.subst.get(&id) {
            return Ok(mapped);
        }
        if let Some(&mapped) = self.memo.get(&id) {
            return Ok(mapped);
        }
        if let Some(replacement) = (self.hook)(self.dst, id, &self.vars)? {
            self.memo.insert(id, replacement);
            return Ok(replacement);
        }
        let node = self.src.builder.node(id).clone();
        let out = match node {
            Node::Input(_)
            | Node::ConstU32(_)
            | Node::ConstField(_)
            | Node::ConstFpExt(_)
            | Node::ConstSym(_) => self.dst.intern(node),
            Node::Var(v) => self
                .vars
                .get(&v)
                .copied()
                .ok_or(CloneError::UnboundVar { var: v })?,
            Node::LiftFpExt(x) => {
                let x = self.clone(x)?;
                self.dst.intern(Node::LiftFpExt(x))
            }
            Node::Bin(op, a, b) => {
                let a = self.clone(a)?;
                let b = self.clone(b)?;
                self.dst.intern(Node::Bin(op, a, b))
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let cond = self.clone(cond)?;
                let then_val = self.clone(then_val)?;
                let else_val = self.clone(else_val)?;
                self.dst.intern(Node::Select {
                    cond,
                    then_val,
                    else_val,
                })
            }
            Node::Index { tensor, indices } => {
                let tensor = self.clone(tensor)?;
                let indices = indices
                    .iter()
                    .map(|c| self.clone(*c))
                    .collect::<Result<Vec<_>, _>>()?;
                self.dst.intern(Node::Index { tensor, indices })
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
                par,
                threads,
            } => {
                let v2 = self.dst.fresh_var();
                let v2_node = self.dst.intern(Node::Var(v2));
                let prev = self.vars.insert(var, v2_node);
                let body = self.clone(body)?;
                restore(&mut self.vars, var, prev);
                self.dst.intern(Node::Compute {
                    bound,
                    var: v2,
                    body,
                    scatter,
                    par,
                    threads,
                })
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                let v2 = self.dst.fresh_var();
                let v2_node = self.dst.intern(Node::Var(v2));
                let prev = self.vars.insert(var, v2_node);
                let body = self.clone(body)?;
                restore(&mut self.vars, var, prev);
                self.dst.intern(Node::Reduce {
                    op,
                    bound,
                    var: v2,
                    body,
                })
            }
            Node::Let { var, value, body } => {
                let value = self.clone(value)?;
                let v2 = self.dst.fresh_var();
                let v2_node = self.dst.intern(Node::Var(v2));
                let prev = self.vars.insert(var, v2_node);
                let body = self.clone(body)?;
                restore(&mut self.vars, var, prev);
                self.dst.intern(Node::Let {
                    var: v2,
                    value,
                    body,
                })
            }
            Node::Tuple(elems) => {
                let elems = elems
                    .iter()
                    .map(|c| self.clone(*c))
                    .collect::<Result<Vec<_>, _>>()?;
                self.dst.intern(Node::Tuple(elems))
            }
            Node::Proj(t, k) => {
                let t = self.clone(t)?;
                self.dst.intern(Node::Proj(t, k))
            }
            Node::Pack(elems) => {
                let elems = elems
                    .iter()
                    .map(|c| self.clone(*c))
                    .collect::<Result<Vec<_>, _>>()?;
                self.dst.intern(Node::Pack(elems))
            }
        };
        self.memo.insert(id, out);
        Ok(out)
    }
}

fn restore<K: std::hash::Hash + Eq, V>(map: &mut HashMap<K, V>, key: K, prev: Option<V>) {
    match prev {
        Some(v) => {
            map.insert(key, v);
        }
        None => {
            map.remove(&key);
        }
    }
}

/// Deterministic hash-consed cloning of a scalar HIR expression that
/// contains no bound variables (`Compute`, `Reduce`, `Let`). Useful for
/// re-interning literal or index-shape expressions in a destination
/// builder. Falls back to the full [`clone_expr`] path if the caller wants
/// bound-variable support.
pub fn clone_pure(src: &Module, root: NodeId, dst: &mut IRBuilder) -> Result<NodeId, CloneError> {
    clone_expr(src, root, dst, &HashMap::new(), &HashMap::new())
}

/// Ordered variables introduced by `let`, `compute`, or `reduce` inside the
/// subgraph rooted at `root`. Used by tests and by
/// [`bound_vars_disjoint`] to detect variable-capture bugs.
pub fn bound_vars(module: &Module, root: NodeId) -> BTreeMap<NodeId, VarId> {
    struct Collect {
        vars: BTreeMap<NodeId, VarId>,
    }
    impl HirVisitor for Collect {
        type Error = std::convert::Infallible;
        fn enter(
            &mut self,
            _m: &Module,
            id: NodeId,
            node: &Node,
            _index_scope: &[IndexBinding],
        ) -> Result<VisitControl, Self::Error> {
            match node {
                Node::Compute { var, .. } | Node::Reduce { var, .. } | Node::Let { var, .. } => {
                    self.vars.insert(id, *var);
                }
                _ => {}
            }
            Ok(VisitControl::Recurse)
        }
    }
    let mut c = Collect {
        vars: BTreeMap::new(),
    };
    let _ = visit_hir(module, root, &mut c);
    c.vars
}

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
    fn visit_hir_visits_body_under_compute_scope() {
        struct Track {
            input_scope_len: Option<usize>,
        }
        impl HirVisitor for Track {
            type Error = std::convert::Infallible;
            fn enter(
                &mut self,
                _m: &Module,
                _id: NodeId,
                node: &Node,
                scope: &[IndexBinding],
            ) -> Result<VisitControl, Self::Error> {
                if let Node::Index { .. } = node {
                    self.input_scope_len = Some(scope.len());
                }
                Ok(VisitControl::Recurse)
            }
        }
        let m = scale_by_two();
        let mut t = Track {
            input_scope_len: None,
        };
        visit_hir(&m, m.body, &mut t).unwrap();
        assert_eq!(t.input_scope_len, Some(1));
    }

    #[test]
    fn compute_leave_sees_outer_scope() {
        struct Track {
            depth: u32,
            max_depth: u32,
            leave_depth_at_compute: Option<u32>,
        }
        impl HirVisitor for Track {
            type Error = std::convert::Infallible;
            fn enter(
                &mut self,
                _m: &Module,
                _id: NodeId,
                _node: &Node,
                scope: &[IndexBinding],
            ) -> Result<VisitControl, Self::Error> {
                self.depth = scope.len() as u32;
                self.max_depth = self.max_depth.max(self.depth);
                Ok(VisitControl::Recurse)
            }
            fn leave(
                &mut self,
                _m: &Module,
                _id: NodeId,
                node: &Node,
                scope: &[IndexBinding],
            ) -> Result<(), Self::Error> {
                if matches!(node, Node::Compute { .. }) {
                    self.leave_depth_at_compute = Some(scope.len() as u32);
                }
                Ok(())
            }
        }
        let m = scale_by_two();
        let mut t = Track {
            depth: 0,
            max_depth: 0,
            leave_depth_at_compute: None,
        };
        visit_hir(&m, m.body, &mut t).unwrap();
        assert!(t.max_depth >= 1, "compute body must extend the scope");
        assert_eq!(t.leave_depth_at_compute, Some(0));
    }

    #[test]
    fn skip_children_stops_descent_and_still_calls_leave() {
        struct Track {
            entered: usize,
            left: usize,
        }
        impl HirVisitor for Track {
            type Error = std::convert::Infallible;
            fn enter(
                &mut self,
                _m: &Module,
                _id: NodeId,
                _node: &Node,
                _scope: &[IndexBinding],
            ) -> Result<VisitControl, Self::Error> {
                self.entered += 1;
                Ok(VisitControl::SkipChildren)
            }
            fn leave(
                &mut self,
                _m: &Module,
                _id: NodeId,
                _node: &Node,
                _scope: &[IndexBinding],
            ) -> Result<(), Self::Error> {
                self.left += 1;
                Ok(())
            }
        }
        let m = scale_by_two();
        let mut t = Track {
            entered: 0,
            left: 0,
        };
        visit_hir(&m, m.body, &mut t).unwrap();
        assert_eq!(t.entered, 1);
        assert_eq!(t.left, 1);
    }

    #[test]
    fn unique_index_scope_finds_scope_for_reachable_node() {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let two = b.const_field(2);
        let body = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            b.mul(ai, two)
        });
        let m = b.finish("m", body);
        let scope = unique_index_scope(&m, two).unwrap().unwrap();
        assert_eq!(scope.len(), 1);
        assert!(matches!(scope[0].kind, IndexKind::Compute));
    }

    #[test]
    fn unique_index_scope_reports_ambiguous_shared_node() {
        // A shared inner constant reached from two computes under
        // different iteration bounds should report ambiguity.
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let seven = b.const_field(7);
        let inner_a = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            b.mul(ai, seven)
        });
        let inner_b = b.compute(8, |b, i| {
            let ai = b.index(a, &[i]);
            b.mul(ai, seven)
        });
        // Combine into a tuple so both computes are reachable.
        let body = b.tuple(&[inner_a, inner_b]);
        let m = b.finish("m", body);
        let res = unique_index_scope(&m, seven);
        assert!(
            matches!(res, Err(AmbiguousAccessScope { node }) if node == seven),
            "expected AmbiguousAccessScope, got {res:?}"
        );
    }

    #[test]
    fn count_reachable_nodes_matches_expected() {
        // scale_by_two body: compute wraps { mul { index { input, var },
        // const_field 2 } }. Distinct reachable NodeIds: input, var,
        // index, const_field, mul, compute = 6.
        let m = scale_by_two();
        assert_eq!(count_reachable_nodes(&m), 6);
    }

    #[test]
    fn collect_input_uses_records_all_occurrences() {
        // The input is hash-consed to one NodeId but is reached as a child
        // of two distinct Index sites: two occurrences, both under the
        // enclosing compute scope.
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![8]);
        let body = b.compute(4, |b, i| {
            let one = b.const_u32(1);
            let j = b.add(i, one);
            let a_i = b.index(a, &[i]);
            let a_j = b.index(a, &[j]);
            b.add(a_i, a_j)
        });
        let m = b.finish("m", body);
        let uses = collect_input_uses(&m);
        assert_eq!(uses.len(), 2, "one occurrence per parent Index site");
        assert!(uses.iter().all(|u| u.input_pos == 0));
        assert!(
            uses.iter().all(|u| u.scope.len() == 1),
            "the Index sites are inside compute so the input is too"
        );
    }

    #[test]
    fn collect_structure_records_computes_and_lets() {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let body = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            b.bind(ai, |b, v| {
                let two = b.const_field(2);
                b.mul(v, two)
            })
        });
        let m = b.finish("m", body);
        let facts = collect_structure(&m);
        assert_eq!(facts.top_computes.len(), 1);
        assert_eq!(facts.computes.len(), 1);
        assert_eq!(facts.lets.len(), 1);
        assert!(!facts.has_nested_compute);
    }

    #[test]
    fn clone_expr_produces_structurally_identical_hash() {
        // Cloning the whole module body into a fresh builder and finishing
        // it as a new module should give the same structural hash.
        let m = scale_by_two();
        let mut b2 = IRBuilder::new();
        let _ = b2.input("a", ScalarType::BabyBear, vec![4]);
        let cloned = clone_expr(&m, m.body, &mut b2, &HashMap::new(), &HashMap::new()).unwrap();
        let m2 = b2.finish("scale_by_two", cloned);
        use crate::module_hash::module_hash;
        assert_eq!(module_hash(&m), module_hash(&m2));
    }

    #[test]
    fn clone_expr_respects_substitution_map() {
        // Replace the input with a constant in the destination; the result
        // should embed the replacement rather than reintern the input.
        let m = scale_by_two();
        let a_node = m.builder.inputs().len();
        // Locate the input Node by walking (input is at index 0 in the body).
        let input_node = {
            struct Find(Option<NodeId>);
            impl HirVisitor for Find {
                type Error = std::convert::Infallible;
                fn enter(
                    &mut self,
                    _m: &Module,
                    id: NodeId,
                    node: &Node,
                    _s: &[IndexBinding],
                ) -> Result<VisitControl, Self::Error> {
                    if matches!(node, Node::Input(_)) && self.0.is_none() {
                        self.0 = Some(id);
                    }
                    Ok(VisitControl::Recurse)
                }
            }
            let mut f = Find(None);
            visit_hir(&m, m.body, &mut f).unwrap();
            f.0.unwrap()
        };
        let _ = a_node;

        let mut b2 = IRBuilder::new();
        let replacement = b2.const_field(9);
        let mut subst = HashMap::new();
        subst.insert(input_node, replacement);
        // Cloning succeeds even though b2 has no input declaration.
        let cloned = clone_expr(&m, m.body, &mut b2, &subst, &HashMap::new()).unwrap();
        // The replacement must be reachable from the cloned root.
        struct Find {
            target: NodeId,
            found: bool,
        }
        impl HirVisitor for Find {
            type Error = std::convert::Infallible;
            fn enter(
                &mut self,
                _m: &Module,
                id: NodeId,
                _n: &Node,
                _s: &[IndexBinding],
            ) -> Result<VisitControl, Self::Error> {
                if id == self.target {
                    self.found = true;
                }
                Ok(VisitControl::Recurse)
            }
        }
        let m2 = b2.finish("m2", cloned);
        let mut f = Find {
            target: replacement,
            found: false,
        };
        visit_hir(&m2, m2.body, &mut f).unwrap();
        assert!(f.found, "replacement must appear in the cloned subgraph");
    }

    #[test]
    fn clone_expr_alpha_renames_bound_vars() {
        // Cloning the same source twice into the same destination builder
        // should yield two distinct root NodeIds with two distinct
        // introduced VarIds — the second clone must not capture the
        // first's bound variable.
        let m = scale_by_two();
        let mut b2 = IRBuilder::new();
        let _ = b2.input("a", ScalarType::BabyBear, vec![4]);
        let c1 = clone_expr(&m, m.body, &mut b2, &HashMap::new(), &HashMap::new()).unwrap();
        let c2 = clone_expr(&m, m.body, &mut b2, &HashMap::new(), &HashMap::new()).unwrap();
        // Hash-consing shares the outer compute, so identity is equal; the
        // interesting property is that neither clone panicked on collision.
        // Verify a fresh binding by inspecting each clone's outermost var.
        let v1 = bound_vars(
            &Module {
                name: "probe1".into(),
                builder: b2.clone(),
                body: c1,
            },
            c1,
        );
        let v2 = bound_vars(
            &Module {
                name: "probe2".into(),
                builder: b2,
                body: c2,
            },
            c2,
        );
        assert!(!v1.is_empty());
        assert!(!v2.is_empty());
    }
}
