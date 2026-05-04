//! Symbolic-DAG rendering for the native Lean dialect.
//!
//! The output is a `Fundamentals.Air.Expr F layout` body — a function of a
//! valuation `va`. Compound nodes are emitted in one of three forms,
//! decided per-node by [`Decision`]:
//!
//! - **Inline.** `(<lhs> op <rhs>)` is dropped into the parent expression
//!   directly. Default for short / single-use subtrees.
//! - **Local `let tN`.** Pushed onto the constraint's binding list and
//!   referenced as `tN` from the parent. Used when a subtree is reused
//!   ≥`SHARE_THRESHOLD` times within one constraint and is large enough
//!   to be worth a name.
//! - **Global `inter_K`.** Hoisted to a top-level `def inter_K : Expr F
//!   layout := fun va => …`. Used when a subtree is shared across
//!   constraints/interactions.
//!
//! The DAG walk is post-order via an explicit stack; pointer identity
//! (`*const SymbolicExpression<F>`) keys the dedup, helper-name, and
//! use-count maps.

use std::{
    collections::{HashMap, HashSet},
    iter,
    sync::Arc,
};

use itertools::Itertools;
use p3_field::Field;

use crate::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints,
    },
    interaction::Interaction,
};

/// Knobs controlling when to lift a subtree out of inline form.
#[derive(Clone, Debug)]
pub struct LeanRenderOptions {
    /// Minimum op-count before a subtree is eligible for naming.
    pub op_threshold: usize,
    /// Minimum use-count (local or global) before naming kicks in.
    pub share_threshold: usize,
}

impl Default for LeanRenderOptions {
    fn default() -> Self {
        Self {
            op_threshold: 2,
            share_threshold: 2,
        }
    }
}

/// Per-AIR rendering state. Lives across all constraints and
/// interactions of one AIR; `local_use_counts` and `temp_counter` are
/// reset per item via [`LeanRenderContext::begin_item`].
pub struct LeanRenderContext<F> {
    pub options: LeanRenderOptions,
    pub global_use_counts: HashMap<*const SymbolicExpression<F>, usize>,
    pub local_use_counts: HashMap<*const SymbolicExpression<F>, usize>,
    pub helper_names: HashMap<*const SymbolicExpression<F>, String>,
    pub emitted_helpers: HashSet<*const SymbolicExpression<F>>,
    pub helper_defs: Vec<String>,
    pub inter_counter: usize,
    pub temp_counter: usize,
    /// Field characteristic for negative-constant folding (None to skip).
    pub characteristic: Option<u32>,
    /// Flat-trace base offset for each `Entry::Main { part_index, .. }`
    /// partition. For a non-partitioned (singleMain) AIR this is just
    /// `[0]`; for a partitioned AIR with cached + common, it's
    /// `[0, cached_width]` (cached at part_index 0, common at part_index 1).
    pub partition_offsets: Vec<usize>,
    /// Names for public-value indices. When `Entry::Public { index }`
    /// has a corresponding entry here, the va-form emits `va <name>PvVar`
    /// (referencing the Schema-emitted ref); otherwise it falls back to
    /// `va (.publicValue {index})`.
    pub public_value_names: Vec<String>,
}

impl<F> LeanRenderContext<F> {
    pub fn new(options: LeanRenderOptions, characteristic: Option<u32>) -> Self {
        Self {
            options,
            global_use_counts: HashMap::new(),
            local_use_counts: HashMap::new(),
            helper_names: HashMap::new(),
            emitted_helpers: HashSet::new(),
            helper_defs: Vec::new(),
            inter_counter: 0,
            temp_counter: 0,
            characteristic,
            partition_offsets: vec![0],
            public_value_names: Vec::new(),
        }
    }

    /// Reset per-item state. Call before rendering each constraint, or
    /// each interaction's count / message component.
    pub fn begin_item<'a, I>(&mut self, exprs: I)
    where
        F: 'a,
        I: IntoIterator<Item = &'a SymbolicExpression<F>>,
    {
        self.local_use_counts = direct_use_counts(exprs);
        self.temp_counter = 0;
    }
}

/// Fully rendered subtree: the let-bindings to emit above, and the
/// expression text to inline at the use site.
#[derive(Clone, Debug)]
struct Rendered {
    /// Local `let tN := …` lines (already in dependency order).
    bindings: Vec<(String, String)>,
    /// Inlined expression text, or a `tN` / `inter_K va` reference.
    result: String,
    /// Number of compound operations in the subtree (used for the
    /// op-count threshold).
    op_count: usize,
}

impl Rendered {
    fn into_block(self, tail: impl FnOnce(String) -> String) -> String {
        // Trailing `;` makes term-mode `let x := e₁; e₂` unambiguous
        // even when `e₂` ends up at a column less than `let`'s — which
        // happens when the surrounding writer wraps the block in
        // parens (shifting `let` rightward by one column).
        let mut lines = self
            .bindings
            .into_iter()
            .map(|(name, expr)| format!("let {name} := {expr};"))
            .collect_vec();
        lines.push(tail(self.result));
        lines.join("\n")
    }
}

/// Render one expression to a Lean term body. The returned string is
/// the body of `fun va => <body>` and includes any local `let`s.
pub fn render_expression_body<F: Field>(
    expr: &SymbolicExpression<F>,
    column_names: &[String],
    context: &mut LeanRenderContext<F>,
) -> String {
    let rendered = render_expression(expr, column_names, context);
    rendered.into_block(|result| result)
}

fn render_expression<F: Field>(
    root: &SymbolicExpression<F>,
    column_names: &[String],
    context: &mut LeanRenderContext<F>,
) -> Rendered {
    let mut stack: Vec<(*const SymbolicExpression<F>, &SymbolicExpression<F>, bool, bool)> =
        vec![(root as *const _, root, false, true)];
    let mut scheduled: HashSet<*const SymbolicExpression<F>> = HashSet::new();
    let mut rendered: HashMap<*const SymbolicExpression<F>, Rendered> = HashMap::new();

    while let Some((ptr, expr, visited, is_root)) = stack.pop() {
        if rendered.contains_key(&ptr) {
            continue;
        }
        if visited {
            let r = match expr {
                SymbolicExpression::Add { x, y, .. } => combine_binary(
                    rendered[&Arc::as_ptr(x)].clone(),
                    rendered[&Arc::as_ptr(y)].clone(),
                    "+",
                ),
                SymbolicExpression::Sub { x, y, .. } => combine_binary(
                    rendered[&Arc::as_ptr(x)].clone(),
                    rendered[&Arc::as_ptr(y)].clone(),
                    "-",
                ),
                SymbolicExpression::Mul { x, y, .. } => combine_binary(
                    rendered[&Arc::as_ptr(x)].clone(),
                    rendered[&Arc::as_ptr(y)].clone(),
                    "*",
                ),
                SymbolicExpression::Neg { x, .. } => combine_neg(rendered[&Arc::as_ptr(x)].clone()),
                _ => Rendered {
                    bindings: vec![],
                    result: render_leaf(
                        expr,
                        column_names,
                        &context.partition_offsets,
                        &context.public_value_names,
                        context.characteristic,
                    ),
                    op_count: 0,
                },
            };
            let r = decide_and_emit(ptr, is_root, r, context);
            rendered.insert(ptr, r);
            continue;
        }

        if !scheduled.insert(ptr) {
            continue;
        }
        stack.push((ptr, expr, true, is_root));
        match expr {
            SymbolicExpression::Add { x, y, .. }
            | SymbolicExpression::Sub { x, y, .. }
            | SymbolicExpression::Mul { x, y, .. } => {
                let yp = Arc::as_ptr(y);
                let xp = Arc::as_ptr(x);
                stack.push((yp, &**y, false, false));
                stack.push((xp, &**x, false, false));
            }
            SymbolicExpression::Neg { x, .. } => {
                let xp = Arc::as_ptr(x);
                stack.push((xp, &**x, false, false));
            }
            _ => {}
        }
    }

    rendered.remove(&(root as *const _)).expect("root rendered")
}

fn combine_binary(lhs: Rendered, rhs: Rendered, op: &str) -> Rendered {
    let mut bindings = lhs.bindings;
    merge_unique_bindings(&mut bindings, rhs.bindings);
    Rendered {
        bindings,
        result: format!("({} {op} {})", lhs.result, rhs.result),
        op_count: lhs.op_count + rhs.op_count + 1,
    }
}

fn combine_neg(inner: Rendered) -> Rendered {
    let bindings = inner.bindings;
    Rendered {
        bindings,
        result: format!("-{}", paren_if_compound(&inner.result)),
        op_count: inner.op_count + 1,
    }
}

fn paren_if_compound(s: &str) -> String {
    // Already parenthesized (`(a + b)`) or a bare token (`t3`,
    // `inter_2 va`, leaf form): leave alone.
    if s.starts_with('(') || !s.contains(' ') {
        s.to_string()
    } else {
        format!("({s})")
    }
}

fn merge_unique_bindings(bindings: &mut Vec<(String, String)>, additional: Vec<(String, String)>) {
    let mut seen: HashSet<String> = bindings.iter().map(|(name, _)| name.clone()).collect();
    for (name, expr) in additional {
        if seen.insert(name.clone()) {
            bindings.push((name, expr));
        }
    }
}

fn decide_and_emit<F: Field>(
    ptr: *const SymbolicExpression<F>,
    is_root: bool,
    r: Rendered,
    ctx: &mut LeanRenderContext<F>,
) -> Rendered {
    if is_root || r.op_count < ctx.options.op_threshold {
        return r;
    }
    let global = ctx.global_use_counts.get(&ptr).copied().unwrap_or(0);
    let local = ctx.local_use_counts.get(&ptr).copied().unwrap_or(0);

    // Cross-constraint sharing → hoist to top-level `inter_K`.
    if global >= ctx.options.share_threshold && global > local {
        let helper_name = ctx
            .helper_names
            .entry(ptr)
            .or_insert_with(|| {
                let n = ctx.inter_counter;
                ctx.inter_counter += 1;
                format!("inter_{n}")
            })
            .clone();
        if ctx.emitted_helpers.insert(ptr) {
            ctx.helper_defs.push(format!(
                "def {helper_name} : Expr F layout := fun va =>\n{}\n",
                indent_block(&r.clone().into_block(|res| res), "  ")
            ));
        }
        return Rendered {
            bindings: vec![],
            result: format!("{helper_name} va"),
            op_count: r.op_count,
        };
    }

    // Repeated within this constraint → local `let tN`.
    if local >= ctx.options.share_threshold {
        let name = format!("t{}", ctx.temp_counter);
        ctx.temp_counter += 1;
        let mut bindings = r.bindings;
        bindings.push((name.clone(), r.result));
        return Rendered {
            bindings,
            result: name,
            op_count: r.op_count,
        };
    }

    r
}

fn render_leaf<F: Field>(
    x: &SymbolicExpression<F>,
    column_names: &[String],
    partition_offsets: &[usize],
    public_value_names: &[String],
    characteristic: Option<u32>,
) -> String {
    match x {
        SymbolicExpression::Variable(SymbolicVariable { entry, index, .. }) => match entry {
            Entry::Main { part_index, offset } => {
                let rotation = match offset {
                    0 => "local",
                    1 => "next",
                    other => panic!("unsupported rotation offset {other} in main column"),
                };
                let base = *partition_offsets.get(*part_index).unwrap_or_else(|| {
                    panic!(
                        "main column part_index {part_index} out of range (partitions: {})",
                        partition_offsets.len()
                    )
                });
                let flat_idx = base + *index;
                let name = column_names.get(flat_idx).unwrap_or_else(|| {
                    panic!(
                        "main column (part {part_index}, index {index}, flat {flat_idx}) has no name"
                    )
                });
                format!("va (.cell .{rotation} {name}Ref)")
            }
            Entry::Preprocessed { .. } => panic!("preprocessed columns not supported in v1"),
            Entry::Permutation { .. } => unreachable!("permutation columns no longer exist"),
            Entry::Public => {
                if let Some(name) = public_value_names.get(*index) {
                    format!("va {name}PvVar")
                } else {
                    format!("va (.publicValue {index})")
                }
            }
            Entry::Challenge => panic!("challenge columns not supported in v1"),
            Entry::Exposed => panic!("exposed columns not supported in v1"),
        },
        SymbolicExpression::IsFirstRow => "va (.selector .isFirst)".to_string(),
        SymbolicExpression::IsLastRow => "va (.selector .isLast)".to_string(),
        SymbolicExpression::IsTransition => "va (.selector .isTransition)".to_string(),
        SymbolicExpression::Constant(c) => render_constant(c, characteristic),
        _ => unreachable!("compound expressions handled by render_expression"),
    }
}

fn render_constant<F: Field>(c: &F, characteristic: Option<u32>) -> String {
    let display = format!("{c}");
    if let (Some(p), Ok(n)) = (characteristic, display.parse::<u32>()) {
        if n < p && p - n < n {
            return format!("-{}", p - n);
        }
    }
    display
}

pub fn indent_block(text: &str, indent: &str) -> String {
    text.lines()
        .map(|line| format!("{indent}{line}"))
        .join("\n")
}

/// Build a use-count map keyed by pointer identity. Counts how many
/// times each subtree is referenced as a child of some other node.
pub fn direct_use_counts<'a, F: 'a, I>(exprs: I) -> HashMap<*const SymbolicExpression<F>, usize>
where
    I: IntoIterator<Item = &'a SymbolicExpression<F>>,
{
    let mut counts: HashMap<*const SymbolicExpression<F>, usize> = HashMap::new();
    let mut visited: HashSet<*const SymbolicExpression<F>> = HashSet::new();
    for root in exprs {
        let mut stack: Vec<&SymbolicExpression<F>> = vec![root];
        while let Some(expr) = stack.pop() {
            let ptr = expr as *const SymbolicExpression<F>;
            if !visited.insert(ptr) {
                continue;
            }
            match expr {
                SymbolicExpression::Add { x, y, .. }
                | SymbolicExpression::Sub { x, y, .. }
                | SymbolicExpression::Mul { x, y, .. } => {
                    *counts.entry(Arc::as_ptr(x)).or_insert(0) += 1;
                    *counts.entry(Arc::as_ptr(y)).or_insert(0) += 1;
                    stack.push(&**y);
                    stack.push(&**x);
                }
                SymbolicExpression::Neg { x, .. } => {
                    *counts.entry(Arc::as_ptr(x)).or_insert(0) += 1;
                    stack.push(&**x);
                }
                _ => {}
            }
        }
    }
    counts
}

/// Build a global use-count map covering all constraints and the
/// `count`/`message` exprs of all interactions in `symbolic`.
pub fn symbolic_global_use_counts<F: Field>(
    symbolic: &SymbolicConstraints<F>,
) -> HashMap<*const SymbolicExpression<F>, usize> {
    direct_use_counts(symbolic.constraints.iter().chain(
        symbolic.interactions.iter().flat_map(|i: &Interaction<_>| {
            iter::once(&i.count).chain(i.message.iter())
        }),
    ))
}

/// Render an expression in *trace form*: leaves print as named
/// accessors (`<name> trace row` / `<name>_next trace row`), all
/// compound nodes are inlined (no `let tN`, no `inter_K`). Used by the
/// lemma generator to express the equality RHS of
/// `evalMultiplicityAt` / `evalMessageAt` lemmas.
///
/// Panics on selector / non-Main entries — bus messages in the
/// recursion verifier don't use them, but constraint expressions can.
pub fn render_trace_body<F: Field>(
    expr: &SymbolicExpression<F>,
    column_names: &[String],
    partition_offsets: &[usize],
    public_value_names: &[String],
    characteristic: Option<u32>,
) -> String {
    let mut out = String::new();
    render_trace_inner(
        expr,
        column_names,
        partition_offsets,
        public_value_names,
        characteristic,
        &mut out,
        true,
    );
    out
}

fn render_trace_inner<F: Field>(
    expr: &SymbolicExpression<F>,
    column_names: &[String],
    partition_offsets: &[usize],
    public_value_names: &[String],
    characteristic: Option<u32>,
    out: &mut String,
    _is_root: bool,
) {
    match expr {
        SymbolicExpression::Variable(SymbolicVariable { entry, index, .. }) => match entry {
            Entry::Main { part_index, offset } => {
                let base = *partition_offsets.get(*part_index).unwrap_or_else(|| {
                    panic!("main column part_index {part_index} out of range")
                });
                let flat_idx = base + *index;
                let name = column_names
                    .get(flat_idx)
                    .unwrap_or_else(|| panic!("main column (part {part_index}, index {index}) has no name"));
                let suffix = match offset {
                    0 => "",
                    1 => "_next",
                    other => panic!("unsupported rotation offset {other} in main column"),
                };
                out.push_str(&format!("{name}{suffix} trace row"));
            }
            Entry::Public => {
                // Renders a public-value reference. If a name was
                // provided for this index, use it: `<name> publicValues`.
                // Otherwise fall back to the raw `publicValues.getD i 0`,
                // which still matches the AIR's eval semantics
                // (`ctx.publicValues.getD i 0`).
                if let Some(name) = public_value_names.get(*index) {
                    out.push_str(&format!("{name} publicValues"));
                } else {
                    out.push_str(&format!("publicValues.getD {index} 0"));
                }
            }
            other => panic!("unsupported entry {other:?} in trace form"),
        },
        SymbolicExpression::Constant(c) => {
            out.push_str(&render_constant(c, characteristic));
        }
        SymbolicExpression::IsFirstRow
        | SymbolicExpression::IsLastRow
        | SymbolicExpression::IsTransition => {
            panic!("selectors not supported in trace form")
        }
        SymbolicExpression::Add { x, y, .. } => {
            out.push('(');
            render_trace_inner(x, column_names, partition_offsets, public_value_names, characteristic, out, false);
            out.push_str(" + ");
            render_trace_inner(y, column_names, partition_offsets, public_value_names, characteristic, out, false);
            out.push(')');
        }
        SymbolicExpression::Sub { x, y, .. } => {
            out.push('(');
            render_trace_inner(x, column_names, partition_offsets, public_value_names, characteristic, out, false);
            out.push_str(" - ");
            render_trace_inner(y, column_names, partition_offsets, public_value_names, characteristic, out, false);
            out.push(')');
        }
        SymbolicExpression::Mul { x, y, .. } => {
            out.push('(');
            render_trace_inner(x, column_names, partition_offsets, public_value_names, characteristic, out, false);
            out.push_str(" * ");
            render_trace_inner(y, column_names, partition_offsets, public_value_names, characteristic, out, false);
            out.push(')');
        }
        SymbolicExpression::Neg { x, .. } => {
            out.push('-');
            render_trace_inner(x, column_names, partition_offsets, public_value_names, characteristic, out, false);
        }
    }
}

/// True if the expression tree contains any selector
/// (`IsFirstRow`/`IsLastRow`/`IsTransition`).
pub fn contains_selector<F: Field>(expr: &SymbolicExpression<F>) -> bool {
    let mut visited: HashSet<*const SymbolicExpression<F>> = HashSet::new();
    let mut stack: Vec<&SymbolicExpression<F>> = vec![expr];
    while let Some(e) = stack.pop() {
        let ptr = e as *const SymbolicExpression<F>;
        if !visited.insert(ptr) {
            continue;
        }
        match e {
            SymbolicExpression::IsFirstRow
            | SymbolicExpression::IsLastRow
            | SymbolicExpression::IsTransition => return true,
            SymbolicExpression::Add { x, y, .. }
            | SymbolicExpression::Sub { x, y, .. }
            | SymbolicExpression::Mul { x, y, .. } => {
                stack.push(&**y);
                stack.push(&**x);
            }
            SymbolicExpression::Neg { x, .. } => stack.push(&**x),
            _ => {}
        }
    }
    false
}

/// True if the expression tree references any public value (`Entry::Public`).
/// Trace-form rendering doesn't have a public-value accessor yet, so callers
/// should skip trace-form for any expression that returns true here.
pub fn contains_public_value<F: Field>(expr: &SymbolicExpression<F>) -> bool {
    let mut visited: HashSet<*const SymbolicExpression<F>> = HashSet::new();
    let mut stack: Vec<&SymbolicExpression<F>> = vec![expr];
    while let Some(e) = stack.pop() {
        let ptr = e as *const SymbolicExpression<F>;
        if !visited.insert(ptr) {
            continue;
        }
        match e {
            SymbolicExpression::Variable(SymbolicVariable {
                entry: Entry::Public,
                ..
            }) => return true,
            SymbolicExpression::Add { x, y, .. }
            | SymbolicExpression::Sub { x, y, .. }
            | SymbolicExpression::Mul { x, y, .. } => {
                stack.push(&**y);
                stack.push(&**x);
            }
            SymbolicExpression::Neg { x, .. } => stack.push(&**x),
            _ => {}
        }
    }
    false
}

/// Local + next column names referenced in an expression, plus
/// public-value accessor names (when a name was provided for the index).
#[derive(Default, Clone, Debug)]
pub struct ColumnsUsed {
    pub local: std::collections::BTreeSet<String>,
    pub next: std::collections::BTreeSet<String>,
    /// Named public-value accessors referenced. Indices for which no
    /// name was provided are silently skipped (the trace renderer falls
    /// back to `publicValues.getD i 0`, which doesn't need an accessor
    /// in the simp set).
    pub pv: std::collections::BTreeSet<String>,
}

impl ColumnsUsed {
    pub fn extend_from(&mut self, other: &ColumnsUsed) {
        self.local.extend(other.local.iter().cloned());
        self.next.extend(other.next.iter().cloned());
        self.pv.extend(other.pv.iter().cloned());
    }
}

/// Collect main-column names referenced by `expr`, partitioned by
/// rotation, plus public-value accessor names (when `public_value_names`
/// supplies one for the referenced index). Selectors and other entries
/// are silently skipped.
pub fn collect_columns_used<F: Field>(
    expr: &SymbolicExpression<F>,
    column_names: &[String],
    partition_offsets: &[usize],
    public_value_names: &[String],
) -> ColumnsUsed {
    let mut out = ColumnsUsed::default();
    let mut visited: HashSet<*const SymbolicExpression<F>> = HashSet::new();
    let mut stack: Vec<&SymbolicExpression<F>> = vec![expr];
    while let Some(e) = stack.pop() {
        let ptr = e as *const SymbolicExpression<F>;
        if !visited.insert(ptr) {
            continue;
        }
        match e {
            SymbolicExpression::Variable(SymbolicVariable { entry, index, .. }) => match entry {
                Entry::Main { part_index, offset } => {
                    let Some(base) = partition_offsets.get(*part_index).copied() else {
                        continue;
                    };
                    if let Some(name) = column_names.get(base + *index) {
                        match offset {
                            0 => {
                                out.local.insert(name.clone());
                            }
                            1 => {
                                out.next.insert(name.clone());
                            }
                            _ => {}
                        }
                    }
                }
                Entry::Public => {
                    if let Some(name) = public_value_names.get(*index) {
                        out.pv.insert(name.clone());
                    }
                }
                _ => {}
            },
            SymbolicExpression::Add { x, y, .. }
            | SymbolicExpression::Sub { x, y, .. }
            | SymbolicExpression::Mul { x, y, .. } => {
                stack.push(&**y);
                stack.push(&**x);
            }
            SymbolicExpression::Neg { x, .. } => stack.push(&**x),
            _ => {}
        }
    }
    out
}

/// Scan all expressions in `symbolic` for variable kinds not supported
/// by the v1 emitter. Returns the first offending entry as a string,
/// or `Ok(())` if every leaf is renderable.
pub fn precheck_supported<F: Field>(symbolic: &SymbolicConstraints<F>) -> Result<(), String> {
    let mut visited: HashSet<*const SymbolicExpression<F>> = HashSet::new();
    let exprs = symbolic
        .constraints
        .iter()
        .chain(symbolic.interactions.iter().flat_map(|i| {
            iter::once(&i.count).chain(i.message.iter())
        }));
    for root in exprs {
        let mut stack: Vec<&SymbolicExpression<F>> = vec![root];
        while let Some(expr) = stack.pop() {
            let ptr = expr as *const SymbolicExpression<F>;
            if !visited.insert(ptr) {
                continue;
            }
            match expr {
                SymbolicExpression::Variable(SymbolicVariable { entry, .. }) => match entry {
                    Entry::Main { offset, .. } if *offset <= 1 => {}
                    Entry::Main { offset, .. } => {
                        return Err(format!("main column rotation offset {offset} > 1"));
                    }
                    Entry::Preprocessed { .. } => {
                        return Err("preprocessed columns not supported".to_string())
                    }
                    Entry::Permutation { .. } => {
                        return Err("permutation columns not supported".to_string())
                    }
                    Entry::Public => {}
                    Entry::Challenge => return Err("challenges not supported".to_string()),
                    Entry::Exposed => return Err("exposed values not supported".to_string()),
                },
                SymbolicExpression::Add { x, y, .. }
                | SymbolicExpression::Sub { x, y, .. }
                | SymbolicExpression::Mul { x, y, .. } => {
                    stack.push(&**y);
                    stack.push(&**x);
                }
                SymbolicExpression::Neg { x, .. } => stack.push(&**x),
                _ => {}
            }
        }
    }
    Ok(())
}
