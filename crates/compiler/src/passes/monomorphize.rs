//! Monomorphization: substitutes concrete values for symbolic module
//! parameters.
//!
//! A symbolic module may keep parameters that only appear in *outer*
//! positions — top-level compute bounds, input shapes, value/index splices —
//! because those are resolved at run time (tiled grids, runtime param ABI).
//! Everything else must be concrete before lowering: inner compute and
//! reduce bounds size shared-memory tiles, block dimensions and sequential
//! loops, and a reduce bound decides the *length* of the kernel chain
//! [`super::rewrite_parallel_reduce`] emits. [`required_params`] computes
//! which parameters those positions reference; [`monomorphize`] substitutes
//! a binding environment and rebuilds the residual module.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::{
    ir::{IRBuilder, Module, Node, NodeId, SizeExpr, VarId},
    module_hash::children_of,
    CompileError,
};

/// Collects every bound expression that must be concrete before lowering:
/// all `Compute`/`Reduce` bounds *except* the outermost bound of each
/// top-level compute kernel. Top-level reduce bounds are included because
/// the parallel-reduce rewrite emits one kernel per halving step — the
/// chain length itself depends on the bound.
fn must_concrete_exprs(m: &Module) -> Vec<SizeExpr> {
    fn all_bounds(b: &IRBuilder, id: NodeId, seen: &mut HashSet<NodeId>, out: &mut Vec<SizeExpr>) {
        if !seen.insert(id) {
            return;
        }
        if let Node::Compute { bound, .. } | Node::Reduce { bound, .. } = b.node(id) {
            out.push(bound.clone());
        }
        for c in children_of(b.node(id)) {
            all_bounds(b, c, seen, out);
        }
    }

    // The rhs of a top-level let (or a module output): a compute keeps its
    // own bound symbolic and only its body is constrained. The node itself
    // is deliberately not marked seen, so the bound is still collected if
    // the same compute also appears nested somewhere.
    fn top_value(b: &IRBuilder, id: NodeId, seen: &mut HashSet<NodeId>, out: &mut Vec<SizeExpr>) {
        match b.node(id) {
            Node::Compute { body, .. } => {
                let body = *body;
                all_bounds(b, body, seen, out);
            }
            Node::Tuple(es) => {
                for e in es.clone() {
                    top_value(b, e, seen, out);
                }
            }
            _ => all_bounds(b, id, seen, out),
        }
    }

    fn walk_top(b: &IRBuilder, id: NodeId, seen: &mut HashSet<NodeId>, out: &mut Vec<SizeExpr>) {
        match b.node(id) {
            Node::Let { value, body, .. } => {
                let (value, body) = (*value, *body);
                top_value(b, value, seen, out);
                walk_top(b, body, seen, out);
            }
            _ => top_value(b, id, seen, out),
        }
    }

    let mut out = Vec::new();
    walk_top(&m.builder, m.body, &mut HashSet::new(), &mut out);
    out
}

/// The outermost bound of each top-level compute kernel — the grid
/// dimension of each launched kernel. These are exactly the bounds that
/// [`must_concrete_exprs`] exempts.
fn outer_bounds(m: &Module) -> Vec<SizeExpr> {
    fn top_value(b: &IRBuilder, id: NodeId, out: &mut Vec<SizeExpr>) {
        match b.node(id) {
            Node::Compute { bound, .. } => out.push(bound.clone()),
            Node::Tuple(es) => {
                for e in es.clone() {
                    top_value(b, e, out);
                }
            }
            _ => {}
        }
    }
    fn walk_top(b: &IRBuilder, id: NodeId, out: &mut Vec<SizeExpr>) {
        match b.node(id) {
            Node::Let { value, body, .. } => {
                let (value, body) = (*value, *body);
                top_value(b, value, out);
                walk_top(b, body, out);
            }
            _ => top_value(b, id, out),
        }
    }
    let mut out = Vec::new();
    walk_top(&m.builder, m.body, &mut out);
    out
}

/// Block size for a kernel whose largest concrete compute size is `m`:
/// round up to a warp, cap at 256.
pub(crate) fn block_size_policy(m: usize) -> usize {
    (m.div_ceil(32) * 32).min(256)
}

/// The parameters that must be bound to concrete values before `m` can
/// lower, in substitution order: greedily the parameter that concretizes
/// the most remaining must-be-concrete bounds, ties by declaration order.
/// Parameters that only appear in outer bounds, input shapes or value
/// splices are absent — they survive monomorphization.
pub fn required_params(m: &Module) -> Vec<VarId> {
    let mut sets: Vec<BTreeSet<VarId>> = Vec::new();
    for e in must_concrete_exprs(m) {
        let mut s = BTreeSet::new();
        e.fold_lits().param_syms(&mut s);
        if !s.is_empty() {
            sets.push(s);
        }
    }
    let order: Vec<VarId> = m.builder.params().iter().map(|(v, _)| *v).collect();
    let mut required = Vec::new();
    while !sets.is_empty() {
        let mut best: Option<(usize, VarId)> = None;
        for &p in &order {
            if required.contains(&p) || !sets.iter().any(|s| s.contains(&p)) {
                continue;
            }
            let solo = sets
                .iter()
                .filter(|s| s.len() == 1 && s.contains(&p))
                .count();
            if best.is_none_or(|(c, _)| solo > c) {
                best = Some((solo, p));
            }
        }
        let (_, p) = best.expect("inner bound references an undeclared parameter");
        required.push(p);
        for s in &mut sets {
            s.remove(&p);
        }
        sets.retain(|s| !s.is_empty());
    }
    required
}

/// Substitutes the parameters bound in `env` as literals throughout `m` —
/// input shapes, compute/reduce bounds and `ConstSym` constants — and drops
/// them from the parameter registry. A `ConstSym` that folds to a literal
/// becomes a plain `ConstU32`, so a fully-monomorphized module is
/// hash-identical to its concrete-authored twin. Errors if a
/// must-be-concrete bound still references an unbound parameter.
pub fn monomorphize(m: &Module, env: &BTreeMap<VarId, i64>) -> Result<Module, CompileError> {
    let b = &m.builder;
    let name_of = |v: VarId| {
        b.params()
            .iter()
            .find(|(p, _)| *p == v)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("{v:?}"))
    };
    let subst = |e: &SizeExpr| e.concretize(env).fold_lits();

    for e in must_concrete_exprs(m) {
        let folded = subst(&e);
        let mut left = BTreeSet::new();
        folded.param_syms(&mut left);
        if !left.is_empty() {
            let names: Vec<String> = left.iter().map(|&v| name_of(v)).collect();
            return Err(CompileError::Monomorphize(format!(
                "inner bound `{folded}` must be concrete, but parameter(s) {} are unbound; \
                 bind them via graph bindings or a shape hint",
                names.join(", ")
            )));
        }
    }

    // Post-order rebuild (children precede parents), preserving `VarId`s.
    fn post_order(b: &IRBuilder, id: NodeId, seen: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) {
        if seen.contains(&id) {
            return;
        }
        for c in children_of(b.node(id)) {
            post_order(b, c, seen, order);
        }
        seen.insert(id);
        order.push(id);
    }
    let mut order = Vec::new();
    post_order(b, m.body, &mut HashSet::new(), &mut order);

    let mut nb = IRBuilder::new();
    for (v, name) in b.params() {
        if !env.contains_key(v) {
            nb.inherit_param(*v, name.clone());
        }
    }
    if let Some(block) = b.block_hint() {
        nb.set_block_hint(block);
    }
    for d in b.inputs() {
        let shape: Vec<SizeExpr> = d.shape.iter().map(subst).collect();
        nb.input(d.name.clone(), d.elem, shape);
    }

    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for &id in &order {
        let new = match b.node(id) {
            Node::Input(k) => nb.intern(Node::Input(*k)),
            Node::Var(v) => nb.intern(Node::Var(*v)),
            Node::ConstU32(c) => nb.const_u32(*c),
            Node::ConstField(c) => nb.const_field(*c),
            Node::ConstFpExt(c) => nb.const_fpext(*c),
            Node::ConstSym(e) => {
                let e = subst(e);
                match e.as_const() {
                    Some(c) => {
                        let c = u32::try_from(c).map_err(|_| {
                            CompileError::Monomorphize(format!(
                                "symbolic constant `{e}` resolves to {c}, out of u32 range"
                            ))
                        })?;
                        nb.const_u32(c)
                    }
                    None => nb.intern(Node::ConstSym(e)),
                }
            }
            Node::LiftFpExt(x) => nb.lift_fpext(map[x]),
            Node::Bin(op, x, y) => nb.bin(*op, map[x], map[y]),
            Node::Select {
                cond,
                then_val,
                else_val,
            } => nb.select(map[cond], map[then_val], map[else_val]),
            Node::Index { tensor, indices } => {
                let idx: Vec<NodeId> = indices.iter().map(|i| map[i]).collect();
                nb.index(map[tensor], &idx)
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
                par,
                threads,
            } => nb.intern(Node::Compute {
                bound: subst(bound),
                var: *var,
                body: map[body],
                scatter: scatter.clone(),
                par: par.clone(),
                threads: *threads,
            }),
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => nb.intern(Node::Reduce {
                op: *op,
                bound: subst(bound),
                var: *var,
                body: map[body],
            }),
            Node::Let { var, value, body } => nb.intern(Node::Let {
                var: *var,
                value: map[value],
                body: map[body],
            }),
            Node::Tuple(es) => {
                let es: Vec<NodeId> = es.iter().map(|e| map[e]).collect();
                nb.tuple(&es)
            }
            Node::Proj(t, k) => nb.proj(map[t], *k),
            Node::Pack(es) => {
                let es: Vec<NodeId> = es.iter().map(|e| map[e]).collect();
                nb.pack(&es)
            }
        };
        map.insert(id, new);
    }
    nb.raise_var_watermark(b.var_watermark());
    Ok(Module {
        name: m.name.clone(),
        builder: nb,
        body: map[&m.body],
    })
}

/// The parameters that the *current* pipeline cannot yet keep symbolic:
/// [`required_params`] plus everything referenced by non-outermost
/// input-shape dims. Only a global buffer's outermost dim participates in
/// the runtime size ABI; inner dims feed stride computation and must be
/// literal. `ConstSym` splices (in value or index position) stay symbolic:
/// lowering classifies them into `SSAOpCode::ConstSym` ops and
/// `IndexMap::SExpr` accesses rendered from the device parameters.
fn interim_required_params(m: &Module) -> BTreeSet<VarId> {
    let b = &m.builder;
    let mut req: BTreeSet<VarId> = required_params(m).into_iter().collect();
    for d in b.inputs() {
        for e in d.shape.iter().skip(1) {
            e.param_syms(&mut req);
        }
    }
    req
}

/// Result of monomorphizing a graph node's module against its inferred
/// parameter bindings (see [`monomorphize_for_graph`]).
pub struct GraphMono {
    /// The residual module: required params baked, surviving params kept.
    /// Carries no block hint (unless the author set one) — the graph
    /// compiler stamps one per template group, from the max concrete size
    /// over the group's instantiations.
    pub residual: Module,
    /// Binding values of the residual's surviving parameters, keyed by
    /// parameter name.
    pub residual_bindings: BTreeMap<String, i64>,
    /// Values baked into the residual, keyed by parameter name.
    pub baked: BTreeMap<String, i64>,
    /// Max concrete outer compute size under the node's full bindings —
    /// `Some` iff the residual keeps a symbolic outer bound and therefore
    /// needs a block hint before lowering.
    pub max_outer: Option<i64>,
}

/// Monomorphizes a graph node's module against its inferred parameter
/// bindings, keeping every parameter that can stay symbolic. `bindings`
/// must have exactly the module's declared param names as keys.
pub fn monomorphize_for_graph(
    m: &Module,
    bindings: &BTreeMap<String, i64>,
) -> Result<GraphMono, CompileError> {
    let params = m.builder.params();
    assert_eq!(
        params.len(),
        bindings.len(),
        "module `{}` declares {} parameter(s) but got {} binding(s)",
        m.name,
        params.len(),
        bindings.len()
    );
    let full_env: BTreeMap<VarId, i64> = params
        .iter()
        .map(|(v, name)| {
            let &val = bindings.get(name).unwrap_or_else(|| {
                panic!(
                    "module `{}`: no binding for param `{name}` (got keys {:?})",
                    m.name,
                    bindings.keys().collect::<Vec<_>>(),
                )
            });
            (*v, val)
        })
        .collect();
    let required = interim_required_params(m);
    let env: BTreeMap<VarId, i64> = full_env
        .iter()
        .filter(|(v, _)| required.contains(v))
        .map(|(&v, &x)| (v, x))
        .collect();
    let residual = monomorphize(m, &env)?;
    let max_outer = if outer_bounds(&residual)
        .iter()
        .any(|e| e.as_const().is_none())
    {
        Some(
            outer_bounds(m)
                .iter()
                .map(|e| {
                    e.concretize(&full_env).as_const().unwrap_or_else(|| {
                        panic!(
                            "outer bound `{e}` of module `{}` stays symbolic under full bindings",
                            m.name
                        )
                    })
                })
                .max()
                .expect("residual has a symbolic outer bound, so bounds are non-empty"),
        )
    } else {
        None
    };
    let baked: BTreeMap<String, i64> = params
        .iter()
        .filter(|(v, _)| env.contains_key(v))
        .map(|(v, name)| (name.clone(), env[v]))
        .collect();
    let residual_bindings: BTreeMap<String, i64> = residual
        .builder
        .params()
        .iter()
        .map(|(v, name)| (name.clone(), full_env[v]))
        .collect();
    Ok(GraphMono {
        residual,
        residual_bindings,
        baked,
        max_outer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir::ScalarType, module_hash::module_hash, passes::fusion::renumber_module};

    /// `n` only bounds the top-level compute (outer); `m` bounds the inner
    /// reduce. Only `m` must be concretized.
    #[test]
    fn required_params_keeps_outer_bound_symbolic() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let m = b.symbol("m");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| {
            b.reduce_add(m, |b, j| {
                let xi = b.index(x, &[i]);
                b.add(xi, j)
            })
        });
        let module = b.finish("outer_survives", body);
        let m_var = module.builder.params()[1].0;
        assert_eq!(required_params(&module), vec![m_var]);
    }

    /// A nested compute (tensor-of-tensor, flattened by canonicalize) is an
    /// inner bound: its parameter is required even though the outer bound's
    /// parameter is not.
    #[test]
    fn required_params_covers_nested_compute() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let m = b.symbol("m");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| {
            b.compute(m, |b, j| {
                let xi = b.index(x, &[i]);
                b.mul(xi, j)
            })
        });
        let module = b.finish("nested_inner", body);
        let m_var = module.builder.params()[1].0;
        assert_eq!(required_params(&module), vec![m_var]);
    }

    /// Full monomorphization produces a module hash-identical (after
    /// α-renumbering) to the concrete-authored twin: bounds and shapes fold
    /// to literals and `ConstSym` becomes `ConstU32`.
    #[test]
    fn monomorphize_matches_concrete_twin() {
        let symbolic = {
            let mut b = IRBuilder::new();
            let n = b.symbol("n");
            let x = b.input("x", ScalarType::U32, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                let c = b.const_sym(n - 1);
                b.mul(xi, c)
            });
            b.finish("twin", body)
        };
        let concrete = {
            let mut b = IRBuilder::new();
            let x = b.input("x", ScalarType::U32, vec![8]);
            let body = b.compute(8usize, |b, i| {
                let xi = b.index(x, &[i]);
                let c = b.const_u32(7);
                b.mul(xi, c)
            });
            b.finish("twin", body)
        };
        let n_var = symbolic.builder.params()[0].0;
        let env: BTreeMap<VarId, i64> = [(n_var, 8)].into();
        let mono = monomorphize(&symbolic, &env).unwrap();
        assert!(mono.builder.params().is_empty());
        assert_eq!(
            module_hash(&renumber_module(&mono)),
            module_hash(&renumber_module(&concrete))
        );
    }

    /// Substituting only the outer parameter leaves the inner reduce bound
    /// symbolic: monomorphize refuses with a named-parameter error.
    #[test]
    fn monomorphize_errors_on_unbound_inner_param() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let m = b.symbol("m");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| {
            b.reduce_add(m, |b, j| {
                let xi = b.index(x, &[i]);
                b.add(xi, j)
            })
        });
        let module = b.finish("unbound_inner", body);
        let n_var = module.builder.params()[0].0;
        let env: BTreeMap<VarId, i64> = [(n_var, 64)].into();
        let err = monomorphize(&module, &env).err().unwrap();
        assert!(err.to_string().contains("m"), "{err}");
    }

    /// The graph path keeps an outer-only parameter symbolic, reports its
    /// binding value, and surfaces the concrete compute size for the graph
    /// compiler's per-template block selection (no hint is stamped here).
    #[test]
    fn monomorphize_for_graph_keeps_outer_param_and_reports_max_outer() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| b.index(x, &[i]));
        let module = b.finish("graph_mono", body);
        let bindings: BTreeMap<String, i64> = [("n".to_string(), 40)].into();
        let gm = monomorphize_for_graph(&module, &bindings).unwrap();
        assert_eq!(gm.residual.builder.params().len(), 1);
        assert_eq!(gm.residual_bindings, bindings);
        assert!(gm.baked.is_empty());
        assert_eq!(gm.max_outer, Some(40));
        assert_eq!(gm.residual.builder.block_hint(), None);
    }

    /// Required params (inner reduce bound) are substituted away; the outer
    /// param survives with its binding, and the baked value plus the
    /// concrete compute size are reported.
    #[test]
    fn monomorphize_for_graph_substitutes_required_params() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let m = b.symbol("m");
        let x = b.input("x", ScalarType::U32, vec![n]);
        let body = b.compute(n, |b, i| {
            b.reduce_add(m, |b, j| {
                let xi = b.index(x, &[i]);
                b.add(xi, j)
            })
        });
        let module = b.finish("graph_mono_req", body);
        let bindings: BTreeMap<String, i64> =
            [("n".to_string(), 1000), ("m".to_string(), 4)].into();
        let gm = monomorphize_for_graph(&module, &bindings).unwrap();
        let names: Vec<&str> = gm
            .residual
            .builder
            .params()
            .iter()
            .map(|(_, n)| n.as_str())
            .collect();
        assert_eq!(names, vec!["n"]);
        assert_eq!(gm.residual_bindings, [("n".to_string(), 1000)].into());
        assert_eq!(gm.baked, [("m".to_string(), 4)].into());
        assert_eq!(gm.max_outer, Some(1000));
    }

    /// A fully concrete residual (every param required) reports no
    /// max_outer and no residual bindings.
    #[test]
    fn monomorphize_for_graph_concrete_residual_has_no_hint() {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::U32, vec![n, n]);
        let body = b.compute(n, |b, i| {
            b.compute(n, |b, j| {
                let xij = b.index(x, &[i, j]);
                b.mul(xij, j)
            })
        });
        let module = b.finish("graph_mono_concrete", body);
        let bindings: BTreeMap<String, i64> = [("n".to_string(), 16)].into();
        let gm = monomorphize_for_graph(&module, &bindings).unwrap();
        assert!(gm.residual.builder.params().is_empty());
        assert!(gm.residual_bindings.is_empty());
        assert_eq!(gm.baked, [("n".to_string(), 16)].into());
        assert_eq!(gm.max_outer, None);
        assert_eq!(gm.residual.builder.block_hint(), None);
    }
}
