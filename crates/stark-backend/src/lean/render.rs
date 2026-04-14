use std::{
    cmp::Ordering,
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

fn collect_variables<F>(
    expression: &SymbolicExpression<F>,
    cache: &mut HashMap<*const SymbolicExpression<F>, Vec<SymbolicVariable<F>>>,
    leaves: &mut HashSet<SymbolicVariable<F>>,
) where
    F: Clone + std::cmp::Eq + std::hash::Hash,
{
    let expr_ptr = expression as *const SymbolicExpression<F>;
    if let Some(cached_leaves) = cache.get(&expr_ptr) {
        leaves.extend(cached_leaves.iter().cloned());
        return;
    }

    let mut subtree_leaves = HashSet::new();
    match expression {
        SymbolicExpression::Variable(symbolic_variable) => {
            subtree_leaves.insert(symbolic_variable.clone());
        }
        SymbolicExpression::Add {
            x,
            y,
            degree_multiple: _,
        }
        | SymbolicExpression::Sub {
            x,
            y,
            degree_multiple: _,
        }
        | SymbolicExpression::Mul {
            x,
            y,
            degree_multiple: _,
        } => {
            collect_variables(x, cache, &mut subtree_leaves);
            collect_variables(y, cache, &mut subtree_leaves);
        }
        SymbolicExpression::Neg {
            x,
            degree_multiple: _,
        } => {
            collect_variables(x, cache, &mut subtree_leaves);
        }
        _ => {}
    }

    let cached_leaves = subtree_leaves.into_iter().collect_vec();
    leaves.extend(cached_leaves.iter().cloned());
    cache.insert(expr_ptr, cached_leaves);
}

fn get_entry_type_id(entry: &Entry) -> u8 {
    match entry {
        Entry::Preprocessed { offset: _ } => 0,
        Entry::Main {
            part_index: _,
            offset: _,
        } => 1,
        Entry::Permutation { offset: _ } => unreachable!("permutation columns no longer exist"),
        Entry::Public => 2,
        Entry::Challenge => 3,
        Entry::Exposed => 4,
    }
}

pub(super) fn placeholder_column_names<F>(constraints: &SymbolicConstraints<F>) -> String
where
    F: Clone + std::cmp::Eq + std::hash::Hash,
{
    let leaves = {
        let mut leaves = HashSet::new();
        let mut variable_cache = HashMap::new();

        constraints.constraints.iter().for_each(|expr| {
            collect_variables(expr, &mut variable_cache, &mut leaves);
        });

        constraints.interactions.iter().for_each(|interaction| {
            collect_variables(&interaction.count, &mut variable_cache, &mut leaves);
            interaction.message.iter().for_each(|expr| {
                collect_variables(expr, &mut variable_cache, &mut leaves);
            });
        });

        leaves
            .into_iter()
            .sorted_by(|lhs, rhs| {
                let type_order = get_entry_type_id(&lhs.entry).cmp(&get_entry_type_id(&rhs.entry));

                let index_order = lhs.index.cmp(&rhs.index);

                let (part_index_order, offset_order) = match (lhs.entry, rhs.entry) {
                    (
                        Entry::Preprocessed { offset: l_offset },
                        Entry::Preprocessed { offset: r_offset },
                    ) => (Ordering::Equal, l_offset.cmp(&r_offset)),
                    (
                        Entry::Main {
                            part_index: l_part_index,
                            offset: l_offset,
                        },
                        Entry::Main {
                            part_index: r_part_index,
                            offset: r_offset,
                        },
                    ) => (l_part_index.cmp(&r_part_index), l_offset.cmp(&r_offset)),
                    (Entry::Permutation { .. }, _) | (_, Entry::Permutation { .. }) => {
                        unreachable!("permutation columns no longer exist")
                    }
                    _ => (Ordering::Equal, Ordering::Equal),
                };

                type_order
                    .then(part_index_order)
                    .then(index_order)
                    .then(offset_order)
            })
            .collect_vec()
    };

    leaves
        .iter()
        .map(|leaf| {
            let column = leaf.index;
            match leaf.entry {
                Entry::Preprocessed { offset } => format!(
                    "--def Circuit._ (c: Circuit F ExtF) (row: N) := c.preprocessed (column := {column}) (row := row) (rotation := {offset})"
                ),
                Entry::Main { part_index, offset } => format!(
                    "--def Circuit._ (c: Circuit F ExtF) (row: N) := c.main (id := {part_index}) (column := {column}) (row := row) (rotation := {offset})"
                ),
                Entry::Permutation { offset: _ } =>
                    unreachable!("permutation columns no longer exist"),
                Entry::Public =>
                    format!("--def Circuit._ (c: Circuit F ExtF) := c.public (index := {column})"),
                Entry::Challenge => format!(
                    "--def Circuit._ (c: Circuit F ExtF) := c.challenge (index := {column})"
                ),
                Entry::Exposed =>
                    format!("--def Circuit._ (c: Circuit F ExtF) := c.exposed (index := {column})"),
            }
        })
        .join("\n")
}

pub(super) fn indent_block(text: &str, indent: &str) -> String {
    text.lines()
        .map(|line| format!("{indent}{line}"))
        .join("\n")
}

#[derive(Clone, Debug)]
pub(super) struct RenderedExpression {
    bindings: Vec<(String, String)>,
    result: String,
    op_count: usize,
}

impl RenderedExpression {
    fn into_block(self, tail: impl FnOnce(String) -> String) -> String {
        let mut lines = self
            .bindings
            .into_iter()
            .map(|(name, expr)| format!("let {name} := {expr}"))
            .collect_vec();
        lines.push(tail(self.result));
        lines.join("\n")
    }
}

#[derive(Debug, Default)]
pub(super) struct LeanRenderCounters {
    next_temp_idx: usize,
    next_intermediate_idx: usize,
}

#[derive(Debug, Default)]
pub(super) struct LeanRenderContext<F> {
    pub(super) counters: LeanRenderCounters,
    pub(super) helper_names: HashMap<*const SymbolicExpression<F>, String>,
    pub(super) emitted_helpers: HashSet<*const SymbolicExpression<F>>,
    pub(super) use_counts: HashMap<*const SymbolicExpression<F>, usize>,
}

pub(super) fn symbolic_constraint_to_lean_definitions<F: Field>(
    x: &SymbolicExpression<F>,
    constraint_idx: usize,
    scoping: &str,
    characteristic: Option<u32>,
    context: &mut LeanRenderContext<F>,
) -> (Vec<String>, String) {
    let (helper_defs, rendered) =
        render_symbolic_expression_with_intermediates(x, scoping, characteristic, context);

    let constraint_def = format!(
        "  @[simp]\n  def constraint_{constraint_idx} {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) (row: ℕ) :=\n{}\n",
        indent_block(&rendered.into_block(|result| format!("{result} = 0")), "    ")
    );
    (helper_defs, constraint_def)
}

#[cfg(test)]
pub(super) fn expression_direct_use_counts<F>(
    constraints: &[SymbolicExpression<F>],
) -> HashMap<*const SymbolicExpression<F>, usize> {
    expression_direct_use_counts_iter(constraints.iter())
}

pub(super) fn symbolic_constraints_use_counts<F: Field>(
    symbolic_constraints: &SymbolicConstraints<F>,
) -> HashMap<*const SymbolicExpression<F>, usize> {
    expression_direct_use_counts_iter(
        symbolic_constraints.constraints.iter().chain(
            symbolic_constraints
                .interactions
                .iter()
                .flat_map(|interaction| {
                    iter::once(&interaction.count).chain(interaction.message.iter())
                }),
        ),
    )
}

fn expression_direct_use_counts_iter<'a, F: 'a>(
    expressions: impl IntoIterator<Item = &'a SymbolicExpression<F>>,
) -> HashMap<*const SymbolicExpression<F>, usize> {
    let mut use_counts = HashMap::new();
    let mut visited = HashSet::new();

    for expression in expressions {
        let mut stack = vec![expression];
        while let Some(expr) = stack.pop() {
            let ptr = expr as *const SymbolicExpression<F>;
            if !visited.insert(ptr) {
                continue;
            }

            match expr {
                SymbolicExpression::Add { x, y, .. }
                | SymbolicExpression::Sub { x, y, .. }
                | SymbolicExpression::Mul { x, y, .. } => {
                    *use_counts.entry(Arc::as_ptr(x)).or_insert(0) += 1;
                    *use_counts.entry(Arc::as_ptr(y)).or_insert(0) += 1;
                    stack.push(&**y);
                    stack.push(&**x);
                }
                SymbolicExpression::Neg { x, .. } => {
                    *use_counts.entry(Arc::as_ptr(x)).or_insert(0) += 1;
                    stack.push(&**x);
                }
                _ => {}
            }
        }
    }

    use_counts
}

pub(super) fn symbolic_interaction_bus_to_string<F: Field>(
    interactions: &[Interaction<SymbolicExpression<F>>],
    scoping: &str,
    characteristic: Option<u32>,
    context: &mut LeanRenderContext<F>,
) -> (Vec<String>, String) {
    let mut helper_defs = vec![];
    let mut row_bindings = vec![];
    let mut row_items = vec![];

    for interaction in interactions {
        let (count_helper_defs, count_rendered) = render_symbolic_expression_with_intermediates(
            &interaction.count,
            scoping,
            characteristic,
            context,
        );
        helper_defs.extend(count_helper_defs);
        let RenderedExpression {
            bindings: count_bindings,
            result: count_result,
            ..
        } = count_rendered;
        merge_unique_bindings(&mut row_bindings, count_bindings);

        let mut message_items = vec![];
        for expr in &interaction.message {
            let (expr_helper_defs, expr_rendered) = render_symbolic_expression_with_intermediates(
                expr,
                scoping,
                characteristic,
                context,
            );
            helper_defs.extend(expr_helper_defs);
            let RenderedExpression {
                bindings, result, ..
            } = expr_rendered;
            merge_unique_bindings(&mut row_bindings, bindings);
            message_items.push(result);
        }

        row_items.push(format!("({count_result}, [{}])", message_items.join(", ")));
    }

    let row_body = RenderedExpression {
        bindings: row_bindings,
        result: format!("[{}]", row_items.join(", ")),
        op_count: 0,
    }
    .into_block(|result| result);

    (
        helper_defs,
        format!(
            "(List.range (Circuit.last_row c + 1)).flatMap (λ row =>\n{})",
            indent_block(&row_body, "  ")
        ),
    )
}

fn render_symbolic_expression_with_intermediates<F: Field>(
    root: &SymbolicExpression<F>,
    scoping: &str,
    characteristic: Option<u32>,
    context: &mut LeanRenderContext<F>,
) -> (Vec<String>, RenderedExpression) {
    let mut stack = vec![(root, false, true)];
    let mut scheduled = HashSet::new();
    let mut rendered: HashMap<*const SymbolicExpression<F>, RenderedExpression> = HashMap::new();
    let mut helper_defs = vec![];

    while let Some((expr, visited, is_root)) = stack.pop() {
        let ptr = expr as *const SymbolicExpression<F>;
        if rendered.contains_key(&ptr) {
            continue;
        }

        if visited {
            let rendered_expr = match expr {
                SymbolicExpression::Add { x, y, .. } => {
                    let lhs = rendered[&Arc::as_ptr(x)].clone();
                    let rhs = rendered[&Arc::as_ptr(y)].clone();
                    let current = combine_binary_expression(lhs, rhs, "+", &mut context.counters);
                    maybe_lift_intermediate(
                        ptr,
                        is_root,
                        current,
                        scoping,
                        &context.use_counts,
                        &mut context.helper_names,
                        &mut context.emitted_helpers,
                        &mut helper_defs,
                        &mut context.counters,
                    )
                }
                SymbolicExpression::Sub { x, y, .. } => {
                    let lhs = rendered[&Arc::as_ptr(x)].clone();
                    let rhs = rendered[&Arc::as_ptr(y)].clone();
                    let current = combine_binary_expression(lhs, rhs, "-", &mut context.counters);
                    maybe_lift_intermediate(
                        ptr,
                        is_root,
                        current,
                        scoping,
                        &context.use_counts,
                        &mut context.helper_names,
                        &mut context.emitted_helpers,
                        &mut helper_defs,
                        &mut context.counters,
                    )
                }
                SymbolicExpression::Neg { x, .. } => {
                    let inner = rendered[&Arc::as_ptr(x)].clone();
                    let current = combine_unary_expression(inner, "-", &mut context.counters);
                    maybe_lift_intermediate(
                        ptr,
                        is_root,
                        current,
                        scoping,
                        &context.use_counts,
                        &mut context.helper_names,
                        &mut context.emitted_helpers,
                        &mut helper_defs,
                        &mut context.counters,
                    )
                }
                SymbolicExpression::Mul { x, y, .. } => {
                    let lhs = rendered[&Arc::as_ptr(x)].clone();
                    let rhs = rendered[&Arc::as_ptr(y)].clone();
                    let current = combine_binary_expression(lhs, rhs, "*", &mut context.counters);
                    maybe_lift_intermediate(
                        ptr,
                        is_root,
                        current,
                        scoping,
                        &context.use_counts,
                        &mut context.helper_names,
                        &mut context.emitted_helpers,
                        &mut helper_defs,
                        &mut context.counters,
                    )
                }
                _ => RenderedExpression {
                    bindings: vec![],
                    result: symbolic_expression_leaf_to_string(expr, scoping, characteristic),
                    op_count: 0,
                },
            };
            rendered.insert(ptr, rendered_expr);
            continue;
        }

        if !scheduled.insert(ptr) {
            continue;
        }

        stack.push((expr, true, is_root));
        match expr {
            SymbolicExpression::Add { x, y, .. }
            | SymbolicExpression::Sub { x, y, .. }
            | SymbolicExpression::Mul { x, y, .. } => {
                stack.push((&**y, false, false));
                stack.push((&**x, false, false));
            }
            SymbolicExpression::Neg { x, .. } => {
                stack.push((&**x, false, false));
            }
            _ => {}
        }
    }

    let result = rendered
        .remove(&(root as *const SymbolicExpression<F>))
        .expect("symbolic expression root should be rendered");
    (helper_defs, result)
}

fn combine_binary_expression(
    lhs: RenderedExpression,
    rhs: RenderedExpression,
    op: &str,
    counters: &mut LeanRenderCounters,
) -> RenderedExpression {
    let mut bindings = lhs.bindings;
    merge_unique_bindings(&mut bindings, rhs.bindings);
    let name = format!("t{}", counters.next_temp_idx);
    counters.next_temp_idx += 1;
    bindings.push((
        name.clone(),
        format!("({} {op} {})", lhs.result, rhs.result),
    ));
    RenderedExpression {
        bindings,
        result: name,
        op_count: lhs.op_count + rhs.op_count + 1,
    }
}

fn combine_unary_expression(
    inner: RenderedExpression,
    op: &str,
    counters: &mut LeanRenderCounters,
) -> RenderedExpression {
    let mut bindings = inner.bindings;
    let name = format!("t{}", counters.next_temp_idx);
    counters.next_temp_idx += 1;
    bindings.push((name.clone(), format!("{op}({})", inner.result)));
    RenderedExpression {
        bindings,
        result: name,
        op_count: inner.op_count + 1,
    }
}

fn merge_unique_bindings(bindings: &mut Vec<(String, String)>, additional: Vec<(String, String)>) {
    let mut seen = bindings
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();
    for (name, expr) in additional {
        if seen.insert(name.clone()) {
            bindings.push((name, expr));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn maybe_lift_intermediate<F: Field>(
    expr_ptr: *const SymbolicExpression<F>,
    is_root: bool,
    rendered: RenderedExpression,
    scoping: &str,
    use_counts: &HashMap<*const SymbolicExpression<F>, usize>,
    helper_names: &mut HashMap<*const SymbolicExpression<F>, String>,
    emitted_helpers: &mut HashSet<*const SymbolicExpression<F>>,
    helper_defs: &mut Vec<String>,
    counters: &mut LeanRenderCounters,
) -> RenderedExpression {
    let use_count = use_counts.get(&expr_ptr).copied().unwrap_or(0);
    if is_root || rendered.op_count <= 1 || use_count <= 1 {
        return rendered;
    }

    let helper_name = helper_names.entry(expr_ptr).or_insert_with(|| {
        let name = format!("inter_{}", counters.next_intermediate_idx);
        counters.next_intermediate_idx += 1;
        name
    });

    if emitted_helpers.insert(expr_ptr) {
        helper_defs.push(format!(
            "  def {helper_name} {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) (row: ℕ) :=\n{}\n",
            indent_block(&rendered.clone().into_block(|result| result), "    ")
        ));
    }

    RenderedExpression {
        bindings: vec![],
        result: format!("{scoping}{helper_name} c row"),
        op_count: rendered.op_count,
    }
}

#[allow(clippy::useless_format)]
fn symbolic_expression_leaf_to_string<F: Field>(
    x: &SymbolicExpression<F>,
    scoping: &str,
    characteristic: Option<u32>,
) -> String {
    match x {
        SymbolicExpression::Variable(symbolic_variable) => format!(
            "{scoping}{}",
            match symbolic_variable.entry {
                Entry::Preprocessed { offset } => format!(
                    "(Circuit.preprocessed c (column := {}) (row := row) (rotation := {offset}))",
                    symbolic_variable.index
                ),
                Entry::Main { offset, part_index } => format!(
                    "(Circuit.main c (id := {part_index}) (column := {}) (row := row) (rotation := {offset}))",
                    symbolic_variable.index
                ),
                Entry::Permutation { offset: _ } =>
                    unreachable!("permutation columns no longer exist"),
                Entry::Public =>
                    format!("(Circuit.public c (index := {}))", symbolic_variable.index),
                Entry::Challenge =>
                    format!("(Circuit.challenge c (index := {}))", symbolic_variable.index),
                Entry::Exposed =>
                    format!("(Circuit.exposed c (index := {}))", symbolic_variable.index),
            },
        ),
        SymbolicExpression::IsFirstRow => format!("(Circuit.isFirstRow c row)"),
        SymbolicExpression::IsLastRow => format!("(Circuit.isLastRow c row)"),
        SymbolicExpression::IsTransition => format!("(Circuit.isTransitionRow c row)"),
        SymbolicExpression::Constant(x) => {
            let num = str::parse::<u32>(&format!("{x}"));
            match num {
                Ok(num) => match characteristic {
                    Some(characteristic) => {
                        if num >= characteristic {
                            format!("{x}")
                        } else if characteristic - num < num {
                            format!("-{}", characteristic - num)
                        } else {
                            format!("{x}")
                        }
                    }
                    None => format!("{x}"),
                },
                Err(_) => format!("{x}"),
            }
        }
        SymbolicExpression::Add { .. }
        | SymbolicExpression::Sub { .. }
        | SymbolicExpression::Neg { .. }
        | SymbolicExpression::Mul { .. } => {
            unreachable!("compound expressions are handled in render_symbolic_expression")
        }
    }
}
