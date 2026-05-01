//! Native-dialect Lean extraction for AIR constraints + interactions.
//!
//! Emits three files per AIR, designed to be consumed by the
//! `Fundamentals.Air` Lean library:
//!
//! - **Schema.lean** — width, layout, per-column `…Idx` / `…Ref` defs.
//! - **Constraints.lean** — symbolic constraints as `Expr F layout`,
//!   `constraintsList`, the `air : AIR F` value, named-column
//!   accessors, and a `RawConstraintsAt` extractor. Shared
//!   sub-expressions (`inter_K`) live here too — `Interactions.lean`
//!   can reference them via the open namespace.
//! - **Interactions.lean** — per-bus `…Interactions` lists referencing
//!   a hand-curated `BusIdx` enum, plus an `allInteractions` concat.
//!
//! Two-phase pipeline: [`render_air`] walks the DAG with one shared
//! [`LeanRenderContext`] across both files (so a subtree shared between
//! a constraint and an interaction picks a single `inter_K` name); the
//! `write_*` functions then dump pre-rendered text into files in the
//! right order.
//!
//! Cached partitions / preprocessed columns / public values are not
//! supported in v1; callers are expected to skip such AIRs.

use std::{
    collections::BTreeMap,
    io::{self, Write},
};

use itertools::Itertools;
use p3_field::Field;

use crate::{
    air_builders::symbolic::{SymbolicConstraints, SymbolicConstraintsDag},
    interaction::{BusIndex, Interaction},
};

mod columns;
mod render;

#[cfg(test)]
mod tests;

pub use columns::{flat_columns_of, flatten_lean_columns, LeanColumns, LeanEntry};
pub use openvm_codec_derive::LeanColumns;
pub use render::{
    collect_columns_used, contains_selector, direct_use_counts, indent_block, precheck_supported,
    render_expression_body, render_trace_body, symbolic_global_use_counts, ColumnsUsed,
    LeanRenderContext, LeanRenderOptions,
};

/// A bus referenced by interactions in an AIR.
#[derive(Clone, Debug)]
pub struct BusBinding {
    /// VK bus index as it appears in symbolic interactions.
    pub vk_index: BusIndex,
    /// Lean enum constructor name (without the leading `.`), e.g.
    /// `"transcriptBus"` or `"sumcheckClaimsBus"`.
    pub lean_name: String,
}

/// Sanitize an AIR name for use as a Lean identifier.
pub fn format_lean_air_name(air_name: &str) -> String {
    let mut formatted = String::with_capacity(air_name.len());
    let mut prev_was_underscore = false;
    for ch in air_name.chars() {
        let replacement = match ch {
            '<' => Some('_'),
            '>' => None,
            ',' | ' ' => Some('_'),
            _ => Some(ch),
        };
        if let Some(ch) = replacement {
            if ch == '_' {
                if prev_was_underscore {
                    continue;
                }
                prev_was_underscore = true;
            } else {
                prev_was_underscore = false;
            }
            formatted.push(ch);
        }
    }
    formatted.trim_end_matches('_').to_string()
}

/// Stem of an AIR name used as a file name (everything before the
/// first generic-arg `<`).
pub fn air_file_stem(air_name: &str) -> &str {
    air_name.split('<').next().unwrap_or(air_name)
}

/// Name of the per-AIR simp attribute used to tag hoisted `inter_K`
/// helpers. Registered (once) by `Schema.lean`; applied (`@[…]`) to
/// each helper def in `Constraints.lean`; named in `simp […]` lists in
/// `Interactions.lean`. Per-AIR so each `Schema.lean` can declare its
/// own without colliding with sibling AIRs in the same project.
pub fn air_inter_attr_name(air_name: &str) -> String {
    format!("{}_inter", air_file_stem(air_name))
}

/// Configuration for the per-AIR writers.
#[derive(Clone, Debug)]
pub struct LeanWriteOptions {
    /// Inline / let / inter-K thresholds.
    pub render: LeanRenderOptions,
    /// Field characteristic for negative-constant folding (e.g.
    /// BabyBear: 2_013_265_921). `None` to skip folding.
    pub characteristic: Option<u32>,
    /// Lean module path to import for the `Fundamentals.Air` library.
    pub fundamentals_import: &'static str,
    /// Lean module path of the shared bus-defs file (consumed by
    /// `Interactions.lean`).
    pub bus_defs_import: &'static str,
    /// Lean namespace under which `BusIdx` lives.
    pub bus_defs_namespace: &'static str,
    /// Lean namespace for this AIR (e.g.
    /// `Recursion.Stacking.UnivariateRoundAir`). Schema, Constraints,
    /// and Interactions all open this namespace.
    pub air_namespace: String,
    /// Schema module path used by `Constraints.lean`'s `import`.
    pub schema_import: String,
    /// Constraints module path used by `Interactions.lean`'s `import`.
    pub constraints_import: String,
    /// Flat-trace base offset for each `Entry::Main { part_index, .. }`
    /// partition. Defaults to `vec![0]` (singleMain). For a partitioned
    /// AIR with cached + common, set `vec![0, cached_width]`.
    pub partition_offsets: Vec<usize>,
}

/// Pre-rendered bodies for one AIR, ready for the file writers.
pub struct RenderedAir {
    pub constraint_bodies: Vec<String>,
    pub interactions: Vec<RenderedBusGroup>,
    pub helper_defs: Vec<String>,
}

/// All interactions on a single bus, post-render.
pub struct RenderedBusGroup {
    pub lean_name: String,
    pub entries: Vec<RenderedInteraction>,
}

/// One interaction post-render. Both `*_body` (va-form, used in the
/// list defs) and `*_trace` (named-accessor form, used in lemma RHSs)
/// are populated; if rendering trace form panics on selectors, the
/// trace fields are `None` and per-pick eval lemmas are skipped.
pub struct RenderedInteraction {
    pub multiplicity_body: String,
    pub message_bodies: Vec<String>,
    pub multiplicity_trace: Option<String>,
    pub message_traces: Option<Vec<String>>,
    /// Columns referenced in the multiplicity expression.
    pub multiplicity_cols: ColumnsUsed,
    /// Union of columns referenced across all message expressions.
    pub message_cols: ColumnsUsed,
}

/// Walk the symbolic DAG, render every constraint and interaction
/// expression once, and return the bodies + accumulated `inter_K` defs.
pub fn render_air<F: Field>(
    symbolic_dag: &SymbolicConstraintsDag<F>,
    column_names: &[String],
    bus_table: &[BusBinding],
    options: &LeanWriteOptions,
) -> io::Result<RenderedAir> {
    let symbolic: SymbolicConstraints<F> = symbolic_dag.into();
    if let Err(reason) = precheck_supported(&symbolic) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, reason));
    }
    let global = symbolic_global_use_counts(&symbolic);
    let mut ctx = LeanRenderContext::new(options.render.clone(), options.characteristic);
    ctx.global_use_counts = global;
    ctx.partition_offsets = options.partition_offsets.clone();

    let mut constraint_bodies = Vec::with_capacity(symbolic.constraints.len());
    for c in &symbolic.constraints {
        ctx.begin_item(std::iter::once(c));
        constraint_bodies.push(render_expression_body(c, column_names, &mut ctx));
    }

    let by_bus = group_interactions_by_bus(&symbolic, bus_table);
    let mut interactions = Vec::with_capacity(by_bus.len());
    for (lean_name, group) in by_bus {
        let mut entries = Vec::with_capacity(group.len());
        for interaction in group {
            ctx.begin_item(
                std::iter::once(&interaction.count).chain(interaction.message.iter()),
            );
            let multiplicity_body =
                render_expression_body(&interaction.count, column_names, &mut ctx);
            let message_bodies: Vec<String> = interaction
                .message
                .iter()
                .map(|m| render_expression_body(m, column_names, &mut ctx))
                .collect();

            // Trace-form rendering is best-effort. The trace renderer
            // panics on selectors / non-Main entries, so skip whenever
            // any expression in this interaction contains one.
            let trace_safe = !contains_selector(&interaction.count)
                && interaction.message.iter().all(|m| !contains_selector(m));
            let multiplicity_trace = if trace_safe {
                Some(render_trace_body(
                    &interaction.count,
                    column_names,
                    &options.partition_offsets,
                    options.characteristic,
                ))
            } else {
                None
            };
            let message_traces: Option<Vec<String>> = if trace_safe {
                Some(
                    interaction
                        .message
                        .iter()
                        .map(|m| {
                            render_trace_body(
                                m,
                                column_names,
                                &options.partition_offsets,
                                options.characteristic,
                            )
                        })
                        .collect(),
                )
            } else {
                None
            };

            let multiplicity_cols = collect_columns_used(
                &interaction.count,
                column_names,
                &options.partition_offsets,
            );
            let mut message_cols = ColumnsUsed::default();
            for m in &interaction.message {
                let cu = collect_columns_used(m, column_names, &options.partition_offsets);
                message_cols.extend_from(&cu);
            }

            entries.push(RenderedInteraction {
                multiplicity_body,
                message_bodies,
                multiplicity_trace,
                message_traces,
                multiplicity_cols,
                message_cols,
            });
        }
        interactions.push(RenderedBusGroup { lean_name, entries });
    }

    Ok(RenderedAir {
        constraint_bodies,
        interactions,
        helper_defs: std::mem::take(&mut ctx.helper_defs),
    })
}

/// Write Schema.lean. For singleMain (one partition) the layout is
/// `TraceLayout.singleMain W` and every column ref uses
/// `ColumnRef.commonMain`. For partitioned AIRs (cached + common) the
/// layout is `TraceLayout.ofWidths 0 [<cached_widths>] <common_width>`
/// and cached-partition columns use `ColumnRef.cachedMain ⟨group, _⟩
/// <localIdx>`.
pub fn write_schema<W: Write>(
    writer: &mut W,
    air_name: &str,
    column_names: &[String],
    options: &LeanWriteOptions,
) -> io::Result<()> {
    let total_width = column_names.len();
    let parts = partition_widths(&options.partition_offsets, total_width);
    let num_parts = parts.len();
    debug_assert!(num_parts >= 1);
    let cached_widths: Vec<usize> = parts[..num_parts - 1].iter().map(|p| p.1).collect();
    let common_width = parts[num_parts - 1].1;
    let is_partitioned = num_parts > 1;

    writeln!(writer, "import {}", options.fundamentals_import)?;
    writeln!(writer)?;
    writeln!(
        writer,
        "/-- Simp set for unfolding hoisted `inter_K` helpers of `{air_name}`. -/"
    )?;
    writeln!(writer, "register_simp_attr {}", air_inter_attr_name(air_name))?;
    writeln!(writer)?;
    writeln!(writer, "/-!")?;
    writeln!(writer, "# {air_name} (native)")?;
    writeln!(writer)?;
    writeln!(writer, "Generated schema layer for `{air_name}`.")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "This file owns the trace layout, raw column indices, and structured column"
    )?;
    writeln!(
        writer,
        "references. Constraints and interactions live in sibling generated modules."
    )?;
    writeln!(writer, "-/")?;
    writeln!(writer)?;
    writeln!(writer, "namespace {}", options.air_namespace)?;
    writeln!(writer)?;
    writeln!(writer, "open Fundamentals.Air")?;
    writeln!(writer)?;
    writeln!(writer, "/-- Width of `{air_name}`'s main trace. -/")?;
    writeln!(writer, "abbrev W : Nat := {total_width}")?;
    writeln!(writer)?;
    writeln!(writer, "/-- Structured trace layout for `{air_name}`. -/")?;
    if is_partitioned {
        let cached_list = cached_widths
            .iter()
            .map(|w| w.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            writer,
            "def layout : TraceLayout := TraceLayout.ofWidths 0 [{cached_list}] {common_width}"
        )?;
    } else {
        writeln!(writer, "def layout : TraceLayout := TraceLayout.singleMain W")?;
    }
    writeln!(writer)?;
    writeln!(writer, "/-! ## Column indices -/")?;
    writeln!(writer)?;
    for (flat_idx, name) in column_names.iter().enumerate() {
        let (group, base, width) = locate_partition(&parts, flat_idx);
        let local_idx = flat_idx - base;
        let is_common = group + 1 == num_parts;
        if is_common {
            if is_partitioned {
                writeln!(writer, "def {name}Idx : Fin {width} := ⟨{local_idx}, by decide⟩")?;
            } else {
                writeln!(writer, "def {name}Idx : Fin W := ⟨{local_idx}, by decide⟩")?;
            }
            writeln!(
                writer,
                "def {name}Ref : ColumnRef layout := ColumnRef.commonMain {name}Idx"
            )?;
        } else {
            writeln!(writer, "def {name}Idx : Fin {width} := ⟨{local_idx}, by decide⟩")?;
            writeln!(
                writer,
                "def {name}Ref : ColumnRef layout := ColumnRef.cachedMain ⟨{group}, by decide⟩ {name}Idx"
            )?;
        }
    }
    writeln!(writer)?;
    writeln!(writer, "end {}", options.air_namespace)?;
    Ok(())
}

/// Convert `partition_offsets` + total width into a list of
/// `(base, width)` pairs, one per partition.
fn partition_widths(partition_offsets: &[usize], total_width: usize) -> Vec<(usize, usize)> {
    if partition_offsets.is_empty() {
        return vec![(0, total_width)];
    }
    let mut out = Vec::with_capacity(partition_offsets.len());
    for i in 0..partition_offsets.len() {
        let base = partition_offsets[i];
        let end = partition_offsets
            .get(i + 1)
            .copied()
            .unwrap_or(total_width);
        out.push((base, end - base));
    }
    out
}

/// Find which partition a flat column index belongs to. Returns
/// `(group_index, partition_base, partition_width)`.
fn locate_partition(parts: &[(usize, usize)], flat_idx: usize) -> (usize, usize, usize) {
    for (i, (base, width)) in parts.iter().enumerate() {
        if flat_idx < *base + *width {
            return (i, *base, *width);
        }
    }
    panic!("flat index {flat_idx} out of range for partitions");
}

/// Write Constraints.lean from a pre-rendered AIR.
pub fn write_constraints<W: Write>(
    writer: &mut W,
    air_name: &str,
    column_names: &[String],
    rendered: &RenderedAir,
    options: &LeanWriteOptions,
) -> io::Result<()> {
    writeln!(writer, "import {}", options.fundamentals_import)?;
    writeln!(writer, "import {}", options.schema_import)?;
    writeln!(writer)?;
    writeln!(writer, "/-!")?;
    writeln!(writer, "# {air_name} (native)")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Generated constraints for `{air_name}`. {} constraint expression(s), {} columns.",
        rendered.constraint_bodies.len(),
        column_names.len(),
    )?;
    writeln!(writer, "-/")?;
    writeln!(writer)?;
    writeln!(writer, "set_option linter.unusedVariables false")?;
    writeln!(writer, "set_option linter.unusedSectionVars false")?;
    writeln!(writer, "set_option maxHeartbeats 800000")?;
    // Defeq-checking `[K]'h ∈ constraintsList` (against the literal-Nat
    // bound proof) walks 282-element lists in `constraintsList` for
    // some AIRs (Poseidon2). Default 512 isn't enough.
    writeln!(writer, "set_option maxRecDepth 4096")?;
    writeln!(writer)?;
    writeln!(writer, "namespace {}", options.air_namespace)?;
    writeln!(writer)?;
    writeln!(writer, "open Fundamentals.Air")?;
    writeln!(writer)?;
    writeln!(writer, "variable {{F : Type}} [Field F]")?;
    writeln!(writer)?;

    if !rendered.helper_defs.is_empty() {
        writeln!(writer, "/-! ## Shared sub-expressions -/")?;
        writeln!(writer)?;
        let attr = air_inter_attr_name(air_name);
        for def in &rendered.helper_defs {
            writeln!(writer, "@[{attr}]")?;
            writeln!(writer, "{def}")?;
        }
    }

    writeln!(writer, "/-! ## Constraint expressions -/")?;
    writeln!(writer)?;
    for (idx, body) in rendered.constraint_bodies.iter().enumerate() {
        writeln!(writer, "/-- `constraint_{idx}`. -/")?;
        writeln!(writer, "def expr_{idx} : Expr F layout := fun va =>")?;
        writeln!(writer, "{}", indent_block(body, "  "))?;
        writeln!(writer)?;
    }

    writeln!(writer, "/-- Full constraint list. -/")?;
    if rendered.constraint_bodies.is_empty() {
        writeln!(writer, "def constraintsList : List (Expr F layout) := []")?;
    } else {
        writeln!(writer, "def constraintsList : List (Expr F layout) :=")?;
        let names = (0..rendered.constraint_bodies.len())
            .map(|i| format!("expr_{i}"))
            .collect::<Vec<_>>();
        write_wrapped_list(writer, "  ", &names)?;
    }
    writeln!(writer)?;

    writeln!(writer, "/-! ## AIR value -/")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "/-- The AIR: structured layout plus all {} constraint(s). -/",
        rendered.constraint_bodies.len()
    )?;
    writeln!(
        writer,
        "def air : AIR F := {{ layout := layout, constraints := constraintsList }}"
    )?;
    writeln!(writer)?;

    writeln!(writer, "/-! ## Named-column accessors -/")?;
    writeln!(writer)?;
    for name in column_names {
        writeln!(
            writer,
            "abbrev {name} (t : (air (F := F)).Trace) (row : Fin t.height) : F :="
        )?;
        writeln!(writer, "  t.col {name}Ref row")?;
    }
    writeln!(writer)?;
    for name in column_names {
        writeln!(
            writer,
            "abbrev {name}_next (t : (air (F := F)).Trace) (row : Fin t.height) : F :="
        )?;
        writeln!(writer, "  t.colNext {name}Ref row")?;
    }
    writeln!(writer)?;

    writeln!(writer, "/-! ## Raw per-row constraint extraction -/")?;
    writeln!(writer)?;
    let n = rendered.constraint_bodies.len();
    writeln!(
        writer,
        "/-- All {n} raw constraint(s) of `air` at a row. Field `cK` is the raw fact"
    )?;
    writeln!(
        writer,
        "    that `expr_K` evaluates to zero at `⟨trace, row, []⟩`. -/"
    )?;
    writeln!(
        writer,
        "structure RawConstraintsAt (trace : (air (F := F)).Trace) (row : Fin trace.height) : Prop where"
    )?;
    for i in 0..n {
        writeln!(
            writer,
            "  c{i} : AIR.Expr.evalAt (A := air (F := F)) expr_{i} ⟨trace, row, []⟩ = 0"
        )?;
    }
    writeln!(writer)?;

    // One generic membership helper, used N times below. The bound
    // is stated as `K < <N>` (literal Nat) so call sites can discharge
    // it with closed `by decide`. `constraintsList.length` reduces to
    // `<N>` by kernel defeq (the def's body is a list literal), so
    // `[K]'h` accepts a `K < <N>` bound directly.
    if n > 0 {
        writeln!(
            writer,
            "private theorem mem_constraintsList (K : Nat) (h : K < {n}) :"
        )?;
        writeln!(
            writer,
            "    (constraintsList : List (Expr F layout))[K]'h ∈ constraintsList :="
        )?;
        writeln!(writer, "  List.getElem_mem _")?;
        writeln!(writer)?;
    }

    writeln!(writer, "namespace RawConstraintsAt")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "/-- Extract raw per-row constraint facts from `air.satisfiesRow`. -/"
    )?;
    writeln!(
        writer,
        "theorem of_satisfiesRow {{trace : (air (F := F)).Trace}} {{row : Fin trace.height}}"
    )?;
    writeln!(
        writer,
        "    (h : AIR.satisfiesRow (air (F := F)) trace row) : RawConstraintsAt trace row := by"
    )?;
    if n == 0 {
        writeln!(writer, "  exact ⟨⟩")?;
    } else {
        let placeholders = std::iter::repeat("?_").take(n).join(", ");
        writeln!(writer, "  refine ⟨{placeholders}⟩")?;
        for i in 0..n {
            writeln!(
                writer,
                "  · exact h expr_{i} (mem_constraintsList {i} (by decide))"
            )?;
        }
    }
    writeln!(writer)?;
    writeln!(writer, "end RawConstraintsAt")?;
    writeln!(writer)?;
    writeln!(writer, "end {}", options.air_namespace)?;
    Ok(())
}

/// Write Interactions.lean from a pre-rendered AIR. In addition to the
/// per-bus `…Interactions` lists and the `allInteractions` concat, this
/// emitter generates index-based named picks and a battery of mem /
/// cases / `_evalMultiplicityAt` / `_evalMessageAt` / `_allInteractions_mem`
/// / classification / `_of_allInteractions` lemmas — the same shape as
/// the hand-written `ws-fv` reference. Picks use index suffixes
/// (`<busName>_0`, `<busName>_1`, …); rename to semantic forms
/// (`Receive`, `Send`, etc.) by hand. Helpers (`inter_K`) live in
/// `Constraints.lean` and are in scope via the open namespace.
pub fn write_interactions<W: Write>(
    writer: &mut W,
    air_name: &str,
    rendered: &RenderedAir,
    options: &LeanWriteOptions,
) -> io::Result<()> {
    writeln!(writer, "import {}", options.constraints_import)?;
    writeln!(writer, "import {}", options.bus_defs_import)?;
    writeln!(writer)?;
    writeln!(writer, "/-!")?;
    writeln!(writer, "# {air_name} (native)")?;
    writeln!(writer)?;
    writeln!(writer, "Generated bus interactions for `{air_name}`.")?;
    writeln!(writer, "-/")?;
    writeln!(writer)?;
    writeln!(writer, "set_option linter.unusedVariables false")?;
    writeln!(writer, "set_option linter.unusedSectionVars false")?;
    writeln!(writer, "set_option maxHeartbeats 800000")?;
    writeln!(writer)?;
    writeln!(writer, "namespace {}", options.air_namespace)?;
    writeln!(writer)?;
    writeln!(writer, "open Fundamentals.Air")?;
    writeln!(writer)?;
    writeln!(writer, "variable {{F : Type}} [Field F] [DecidableEq F]")?;
    writeln!(writer)?;

    let bus_def_ty = format!(
        "Interaction (air (F := F)) ({}.busInventory F)",
        options.bus_defs_namespace
    );

    let bus_defs_ns = options.bus_defs_namespace;

    for group in &rendered.interactions {
        let lean_name = group.lean_name.as_str();
        writeln!(writer, "/-! ### {lean_name} -/")?;
        writeln!(writer)?;
        writeln!(writer, "def {lean_name}Interactions : List ({bus_def_ty}) :=")?;
        if group.entries.is_empty() {
            writeln!(writer, "  []")?;
            writeln!(writer)?;
            continue;
        }
        writeln!(writer, "  [")?;
        for (i, entry) in group.entries.iter().enumerate() {
            writeln!(writer, "    {{ bus := .{lean_name}")?;
            writeln!(writer, "      multExpr := fun va =>")?;
            // Multi-line bodies (e.g. with `let t0 := …` bindings)
            // need explicit parens to terminate the lambda cleanly
            // before the next struct field — otherwise Lean's parser
            // eats `msgExprs` as part of the lambda body.
            let mult_body = if entry.multiplicity_body.contains('\n') {
                format!("({})", entry.multiplicity_body)
            } else {
                entry.multiplicity_body.clone()
            };
            writeln!(writer, "{}", indent_block(&mult_body, "        "))?;
            writeln!(writer, "      msgExprs := #v[")?;
            for (j, body) in entry.message_bodies.iter().enumerate() {
                let suffix = if j + 1 == entry.message_bodies.len() {
                    ""
                } else {
                    ","
                };
                writeln!(writer, "        (fun va =>")?;
                writeln!(writer, "{}){suffix}", indent_block(body, "          "))?;
            }
            let trailing = if i + 1 == group.entries.len() { "" } else { "," };
            writeln!(writer, "      ] }}{trailing}")?;
        }
        writeln!(writer, "  ]")?;
        writeln!(writer)?;

        write_per_pick_lemmas(writer, air_name, group, &bus_def_ty)?;
    }

    if !rendered.interactions.is_empty() {
        writeln!(writer, "/-- All interactions for `{air_name}`. -/")?;
        writeln!(writer, "def allInteractions : List ({bus_def_ty}) :=")?;
        let names = rendered
            .interactions
            .iter()
            .map(|g| format!("{}Interactions", g.lean_name))
            .collect::<Vec<_>>();
        write_concat(writer, "  ", &names)?;
        writeln!(writer)?;

        write_classification_and_selector_lemmas(
            writer,
            &rendered.interactions,
            &bus_def_ty,
            bus_defs_ns,
        )?;
    }

    writeln!(writer, "end {}", options.air_namespace)?;
    Ok(())
}

fn write_wrapped_list<W: Write>(
    writer: &mut W,
    indent: &str,
    names: &[String],
) -> io::Result<()> {
    writeln!(writer, "{indent}[")?;
    let inner = format!("{indent}  ");
    let mut line = String::new();
    line.push_str(&inner);
    for (i, name) in names.iter().enumerate() {
        let token = if i + 1 == names.len() {
            name.clone()
        } else {
            format!("{name},")
        };
        if line.trim().len() + token.len() + 1 > 76 && line.trim() != inner.trim() {
            writeln!(writer, "{line}")?;
            line.clear();
            line.push_str(&inner);
        }
        if !line.ends_with(' ') && line.trim().len() > 0 {
            line.push(' ');
        }
        line.push_str(&token);
    }
    if !line.trim().is_empty() {
        writeln!(writer, "{line}")?;
    }
    writeln!(writer, "{indent}]")?;
    Ok(())
}

fn write_concat<W: Write>(writer: &mut W, indent: &str, names: &[String]) -> io::Result<()> {
    if names.is_empty() {
        writeln!(writer, "{indent}[]")?;
        return Ok(());
    }
    let joined = names.join(" ++ ");
    writeln!(writer, "{indent}{joined}")?;
    Ok(())
}

fn group_interactions_by_bus<'a, F>(
    symbolic: &'a SymbolicConstraints<F>,
    bus_table: &[BusBinding],
) -> Vec<(
    String,
    Vec<&'a Interaction<crate::air_builders::symbolic::symbolic_expression::SymbolicExpression<F>>>,
)>
where
    F: Field,
{
    let mut by_idx: BTreeMap<BusIndex, Vec<&Interaction<_>>> = BTreeMap::new();
    for interaction in &symbolic.interactions {
        by_idx
            .entry(interaction.bus_index)
            .or_default()
            .push(interaction);
    }
    // Preserve VK index order; map each to its Lean name. Unknown
    // indices fall back to `bus_<idx>` so emission never fails — the
    // operator can wire up the missing constructor in `BusIdx` later.
    let mut out: Vec<(String, Vec<&Interaction<_>>)> = Vec::new();
    for (idx, group) in by_idx {
        let lean_name = bus_table
            .iter()
            .find(|b| b.vk_index == idx)
            .map(|b| b.lean_name.clone())
            .unwrap_or_else(|| format!("bus_{idx}"));
        out.push((lean_name, group));
    }
    out
}

/// Emit, for one bus group: indexed picks, `_mem` lemmas, optional
/// `_cases` lemma, and per-pick `_evalMultiplicityAt`/`_evalMessageAt`
/// lemmas (when trace-form rendering succeeded for the entry).
fn write_per_pick_lemmas<W: Write>(
    writer: &mut W,
    air_name: &str,
    group: &RenderedBusGroup,
    bus_def_ty: &str,
) -> io::Result<()> {
    let bus = group.lean_name.as_str();
    let list_def = format!("{bus}Interactions");
    let n = group.entries.len();
    if n == 0 {
        return Ok(());
    }

    // Index-based named picks.
    for i in 0..n {
        writeln!(
            writer,
            "/-- Pick #{i} of `{list_def}`. Rename to a semantic suffix"
        )?;
        writeln!(writer, "    (e.g. `Receive`, `Send`) by hand if applicable. -/")?;
        writeln!(writer, "def {bus}_{i} : {bus_def_ty} :=")?;
        writeln!(
            writer,
            "  ({list_def}).get ⟨{i}, by simp [{list_def}]⟩"
        )?;
        writeln!(writer)?;
    }

    // Per-pick membership.
    for i in 0..n {
        writeln!(
            writer,
            "lemma {bus}_{i}_mem : {bus}_{i} (F := F) ∈ {list_def} (F := F) := by"
        )?;
        writeln!(writer, "  simp [{bus}_{i}, {list_def}]")?;
        writeln!(writer)?;
    }

    // Cases lemma for buses with 2+ entries.
    if n >= 2 {
        writeln!(writer, "lemma {list_def}_cases")?;
        writeln!(
            writer,
            "    {{I : {bus_def_ty}}} (hI : I ∈ {list_def} (F := F)) :"
        )?;
        let alts = (0..n)
            .map(|i| format!("I = {bus}_{i} (F := F)"))
            .collect::<Vec<_>>()
            .join(" ∨ ");
        writeln!(writer, "    {alts} := by")?;
        writeln!(writer, "  unfold {list_def} at hI")?;
        writeln!(
            writer,
            "  simp only [List.mem_cons, List.not_mem_nil, or_false] at hI"
        )?;
        let pattern = std::iter::repeat("rfl").take(n).collect::<Vec<_>>().join(" | ");
        writeln!(writer, "  rcases hI with {pattern}")?;
        for i in 0..n {
            // Place the goal in the i-th disjunct slot, then simp.
            // Build `left` ... `right; left` ... `right; right; ...; left/right`.
            let nav = build_disjunct_nav(i, n);
            writeln!(writer, "  · {nav}")?;
            writeln!(writer, "    simp [{bus}_{i}, {list_def}]")?;
        }
        writeln!(writer)?;
    }

    // Per-pick eval lemmas (when trace form is available).
    for (i, entry) in group.entries.iter().enumerate() {
        let (Some(mult_trace), Some(msg_traces)) =
            (&entry.multiplicity_trace, &entry.message_traces)
        else {
            continue;
        };

        // _evalMultiplicityAt
        let (acc_list, ref_list, ctx_list) =
            simp_args_for_columns(&entry.multiplicity_cols);
        let mult_uses_inter = body_uses_inter_helper(&entry.multiplicity_body);
        writeln!(writer, "lemma {bus}_{i}_evalMultiplicityAt")?;
        writeln!(
            writer,
            "    (trace : (air (F := F)).Trace) (row : Fin trace.height) :"
        )?;
        writeln!(
            writer,
            "    ({bus}_{i} (F := F)).evalMultiplicityAt ⟨trace, row, []⟩ ="
        )?;
        writeln!(writer, "      {mult_trace} := by")?;
        let mut simp_names: Vec<String> = vec![
            format!("{bus}_{i}"),
            list_def.clone(),
            "Interaction.evalMultiplicityAt".to_string(),
            "AIR.Expr.evalAt".to_string(),
        ];
        if mult_uses_inter {
            simp_names.push(air_inter_attr_name(air_name));
        }
        simp_names.extend(acc_list.iter().cloned());
        simp_names.extend(ctx_list.iter().cloned());
        simp_names.extend(ref_list.iter().cloned());
        writeln!(writer, "  simp [{}]", simp_names.join(", "))?;
        writeln!(writer)?;

        // _evalMessageAt
        let arity = msg_traces.len();
        if arity == 0 {
            // Empty message: just rfl after unfolding.
            writeln!(writer, "lemma {bus}_{i}_evalMessageAt")?;
            writeln!(
                writer,
                "    (trace : (air (F := F)).Trace) (row : Fin trace.height) :"
            )?;
            writeln!(
                writer,
                "    ({bus}_{i} (F := F)).evalMessageAt ⟨trace, row, []⟩ = #v[] := by"
            )?;
            writeln!(writer, "  simp [{bus}_{i}, {list_def}, Interaction.evalMessageAt]")?;
            writeln!(writer)?;
            continue;
        }

        let (msg_acc, msg_ref, msg_ctx) = simp_args_for_columns(&entry.message_cols);
        let msg_uses_inter = entry
            .message_bodies
            .iter()
            .any(|b| body_uses_inter_helper(b));
        writeln!(writer, "lemma {bus}_{i}_evalMessageAt")?;
        writeln!(
            writer,
            "    (trace : (air (F := F)).Trace) (row : Fin trace.height) :"
        )?;
        writeln!(
            writer,
            "    ({bus}_{i} (F := F)).evalMessageAt ⟨trace, row, []⟩ ="
        )?;
        writeln!(writer, "      #v[")?;
        for (j, body) in msg_traces.iter().enumerate() {
            let suffix = if j + 1 == arity { "" } else { "," };
            writeln!(writer, "        {body}{suffix}")?;
        }
        writeln!(writer, "      ] := by")?;
        writeln!(writer, "  apply Vector.ext")?;
        writeln!(writer, "  intro j hj")?;
        writeln!(writer, "  change j < {arity} at hj")?;
        let mut simp_names: Vec<String> = vec![
            format!("{bus}_{i}"),
            list_def.clone(),
            "Interaction.evalMessageAt".to_string(),
            "AIR.Expr.evalAt".to_string(),
        ];
        if msg_uses_inter {
            simp_names.push(air_inter_attr_name(air_name));
        }
        simp_names.extend(msg_acc.iter().cloned());
        simp_names.extend(msg_ctx.iter().cloned());
        simp_names.extend(msg_ref.iter().cloned());
        writeln!(
            writer,
            "  interval_cases j <;> simp [{}] <;> rfl",
            simp_names.join(", ")
        )?;
        writeln!(writer)?;
    }

    Ok(())
}

/// Build a sequence of `left`/`right` tactic invocations to place a
/// goal in the i-th disjunct of a balanced right-leaning N-way `∨`.
/// For a 2-element disjunct (i=0, n=2): `left`. (i=1, n=2): `right`.
/// For 3 elements: i=0 → `left`, i=1 → `right; left`, i=2 → `right; right`.
fn build_disjunct_nav(i: usize, n: usize) -> String {
    if n <= 1 {
        return String::new();
    }
    let mut steps: Vec<&str> = Vec::new();
    let mut k = i;
    let mut remaining = n;
    while remaining > 1 {
        if k == 0 {
            steps.push("left");
            break;
        } else {
            steps.push("right");
            k -= 1;
            remaining -= 1;
        }
    }
    steps.join("; ")
}

/// Whether a rendered expression body references any `inter_<N>`
/// helper. The eval-lemma emitter uses this to decide whether to add
/// the per-AIR `inter` simp attribute to a `simp` call. Word-boundary
/// aware: matches `inter_3`, not e.g. `winter_3`.
fn body_uses_inter_helper(body: &str) -> bool {
    let bytes = body.as_bytes();
    let mut search = bytes;
    while let Some(pos) = find_subslice(search, b"inter_") {
        let abs_start = (search.as_ptr() as usize) - (bytes.as_ptr() as usize) + pos;
        let after = abs_start + b"inter_".len();
        let prev_ok = abs_start == 0
            || !(bytes[abs_start - 1].is_ascii_alphanumeric() || bytes[abs_start - 1] == b'_');
        let has_digit = after < bytes.len() && bytes[after].is_ascii_digit();
        if prev_ok && has_digit {
            return true;
        }
        search = &search[pos + b"inter_".len()..];
    }
    false
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

/// Build the simp-arg fragments for an `evalMultiplicityAt` /
/// `evalMessageAt` lemma, given the columns referenced. Returns
/// `(accessors, refs, ctxs)` so the caller can place them in the
/// canonical order: pick & list defs, then `evalAt`/`Interaction.evalAt`,
/// then accessors, then ctxs (`localRow`/`nextRow`/`Trace.col[Next]`),
/// then refs.
fn simp_args_for_columns(cols: &ColumnsUsed) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut accessors: Vec<String> = Vec::new();
    let mut refs: Vec<String> = Vec::new();
    let mut ctxs: Vec<String> = Vec::new();

    for name in &cols.local {
        accessors.push(name.clone());
    }
    for name in &cols.next {
        accessors.push(format!("{name}_next"));
    }
    if !cols.local.is_empty() {
        ctxs.push("AIR.EvalCtx.localRow".to_string());
        ctxs.push("AIR.Trace.get".to_string());
        ctxs.push("AIR.Trace.col".to_string());
    }
    if !cols.next.is_empty() {
        if !ctxs.iter().any(|s| s == "AIR.Trace.get") {
            ctxs.push("AIR.Trace.get".to_string());
        }
        ctxs.push("AIR.EvalCtx.nextRow".to_string());
        ctxs.push("AIR.Trace.colNext".to_string());
    }
    let all_refs: std::collections::BTreeSet<&String> =
        cols.local.iter().chain(cols.next.iter()).collect();
    for name in all_refs {
        refs.push(format!("{name}Ref"));
    }
    (accessors, refs, ctxs)
}

/// Emit (after `allInteractions`) per-bus classification, per-pick
/// `_allInteractions_mem` lemmas, and the cross-bus
/// `_of_allInteractions` selector lemmas.
fn write_classification_and_selector_lemmas<W: Write>(
    writer: &mut W,
    groups: &[RenderedBusGroup],
    bus_def_ty: &str,
    _bus_defs_ns: &str,
) -> io::Result<()> {
    if groups.is_empty() {
        return Ok(());
    }

    // Per-pick allInteractions_mem lemmas.
    for group in groups {
        let bus = group.lean_name.as_str();
        for i in 0..group.entries.len() {
            writeln!(
                writer,
                "lemma {bus}_{i}_allInteractions_mem : {bus}_{i} (F := F) ∈ allInteractions (F := F) := by"
            )?;
            writeln!(writer, "  unfold allInteractions")?;
            writeln!(writer, "  simp [{bus}_{i}_mem]")?;
            writeln!(writer)?;
        }
    }

    // Per-bus classification: every entry's bus = .<busName>.
    for group in groups {
        let bus = group.lean_name.as_str();
        let list_def = format!("{bus}Interactions");
        let n = group.entries.len();
        if n == 0 {
            continue;
        }
        writeln!(writer, "private lemma {list_def}_bus")?;
        writeln!(
            writer,
            "    (I : {bus_def_ty}) (hI : I ∈ {list_def} (F := F)) :"
        )?;
        writeln!(writer, "    I.bus = .{bus} := by")?;
        writeln!(writer, "  unfold {list_def} at hI")?;
        writeln!(
            writer,
            "  simp only [List.mem_cons, List.not_mem_nil, or_false] at hI"
        )?;
        let pattern = std::iter::repeat("rfl").take(n).collect::<Vec<_>>().join(" | ");
        if n == 1 {
            writeln!(writer, "  rcases hI with {pattern}")?;
            writeln!(writer, "  rfl")?;
        } else {
            writeln!(writer, "  rcases hI with {pattern} <;> rfl")?;
        }
        writeln!(writer)?;
    }

    // Cross-bus selector lemma: any allInteractions member with bus=.<busName>
    // must be in <busName>Interactions.
    for (target_idx, target) in groups.iter().enumerate() {
        if target.entries.is_empty() {
            continue;
        }
        let target_bus = target.lean_name.as_str();
        let target_list = format!("{target_bus}Interactions");
        writeln!(writer, "lemma {target_list}_of_allInteractions")?;
        writeln!(
            writer,
            "    {{I : {bus_def_ty}}} (hI : I ∈ allInteractions (F := F))"
        )?;
        writeln!(writer, "    (hbus : I.bus = .{target_bus}) :")?;
        writeln!(writer, "    I ∈ {target_list} := by")?;
        writeln!(writer, "  unfold allInteractions at hI")?;

        if groups.len() == 1 {
            // Single bus group: `allInteractions` is a plain alias for
            // the only `<bus>Interactions`, no `++` to split.
            writeln!(writer, "  exact hI")?;
            writeln!(writer)?;
            continue;
        }

        writeln!(writer, "  simp only [List.mem_append] at hI")?;

        // Build the rcases pattern reflecting `(((g0 ∨ g1) ∨ g2) ∨ g3)`
        // shape produced by `++` left-association.
        let mut hyps: Vec<String> = Vec::new();
        for g in groups {
            hyps.push(format!("h_{}", g.lean_name));
        }
        let pat = build_left_assoc_or_pattern(&hyps);
        writeln!(writer, "  rcases hI with {pat}")?;
        for (idx, g) in groups.iter().enumerate() {
            let bus = g.lean_name.as_str();
            let list = format!("{bus}Interactions");
            let h = format!("h_{bus}");
            if idx == target_idx {
                writeln!(writer, "  · exact {h}")?;
            } else {
                writeln!(
                    writer,
                    "  · exfalso; have hb := {list}_bus _ {h}; rw [hbus] at hb; cases hb"
                )?;
            }
        }
        writeln!(writer)?;
    }

    Ok(())
}

/// Build an rcases pattern matching the left-associated `∨` shape
/// produced by repeated `List.mem_append` rewrites of an N-way `++`:
/// for groups [a, b, c, d] you get `(((ha | hb) | hc) | hd)`.
fn build_left_assoc_or_pattern(names: &[String]) -> String {
    if names.is_empty() {
        return String::new();
    }
    let mut acc = names[0].clone();
    for name in &names[1..] {
        acc = format!("({acc} | {name})");
    }
    acc
}
