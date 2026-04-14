//! Extraction of AIR constraints (with interactions) from symbolic DAGs to Lean4 code.
//! This extraction can be used for standalone Lean4 code generation from serialized circuit
//! verifying keys.
//
// Original extraction code was introduced in Nethermind fork <https://github.com/NethermindEth/openvm-stark-backend/releases/tag/v1.2.2-extraction>

use std::{
    collections::HashMap,
    io::{self, Write},
};

use itertools::Itertools;
use p3_field::Field;

use crate::{
    air_builders::symbolic::{SymbolicConstraints, SymbolicConstraintsDag},
    interaction::Interaction,
};

mod columns;
mod render;

#[cfg(test)]
mod tests;

pub use columns::{generate_lean_air_definition, LeanColumns, LeanEntry};
pub use openvm_codec_derive::LeanColumns;
use render::{
    indent_block, placeholder_column_names, symbolic_constraint_to_lean_definitions,
    symbolic_constraints_use_counts, symbolic_interaction_bus_to_string, LeanRenderContext,
};

fn format_lean_air_name(air_name: &str) -> String {
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

/// Code generation operates by printing the Lean4 code to stdout.
pub fn extract_constraints_to_lean<F: Field>(
    symbolic_constraints: &SymbolicConstraints<F>,
    air_name: &str,
) {
    let mut stdout = io::stdout().lock();
    extract_constraints_to_lean_writer(symbolic_constraints, air_name, &mut stdout)
        .expect("write Lean constraints to stdout");
}

pub fn extract_constraints_to_lean_writer<F: Field, W: Write>(
    symbolic_constraints: &SymbolicConstraints<F>,
    air_name: &str,
    writer: &mut W,
) -> io::Result<()> {
    let lean_air_name = format_lean_air_name(air_name);

    writeln!(writer, "import Mathlib.Algebra.Field.Basic")?;
    writeln!(writer)?;
    writeln!(writer, "import LeanZKCircuit.OpenVM.Circuit")?;
    writeln!(writer)?;
    writeln!(writer, "set_option linter.all false")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "register_simp_attr {lean_air_name}_air_simplification"
    )?;
    writeln!(
        writer,
        "register_simp_attr {lean_air_name}_constraint_and_interaction_simplification"
    )?;
    writeln!(writer)?;
    writeln!(writer, "namespace {lean_air_name}.extraction")?;
    writeln!(writer)?;
    writeln!(writer, "-----Constraints for {air_name}-----")?;
    writeln!(writer)?;
    writeln!(writer, "-----Used Columns-------------------")?;
    writeln!(writer, "{}", placeholder_column_names(symbolic_constraints))?;
    writeln!(writer)?;
    writeln!(writer, "-----Extracted constraints----------")?;
    let mut render_context = LeanRenderContext {
        use_counts: symbolic_constraints_use_counts(symbolic_constraints),
        ..Default::default()
    };
    let mut helper_defs = vec![];
    let mut constraint_defs = vec![];
    for (idx, constraint) in symbolic_constraints.constraints.iter().enumerate() {
        let (new_helper_defs, constraint_text) =
            symbolic_constraint_to_lean_definitions(constraint, idx, "", None, &mut render_context);
        helper_defs.extend(new_helper_defs);
        constraint_defs.push(constraint_text);
    }
    let mut interactions_by_bus: HashMap<u16, Vec<Interaction<_>>> = HashMap::new();
    for interaction in symbolic_constraints.interactions.iter() {
        interactions_by_bus
            .entry(interaction.bus_index)
            .or_default()
            .push(interaction.clone());
    }

    let mut interaction_branches = vec![];
    for (idx, (bus_idx, interactions)) in interactions_by_bus
        .iter()
        .sorted_by(|(a, _), (c, _)| a.cmp(c))
        .enumerate()
    {
        let (new_helper_defs, expr) =
            symbolic_interaction_bus_to_string(interactions, "", None, &mut render_context);
        helper_defs.extend(new_helper_defs);
        interaction_branches.push(format!(
            "      {}if index = {} then\n{}",
            if idx == 0 { "" } else { "else " },
            bus_idx,
            indent_block(&expr, "        "),
        ));
    }

    for helper_def in helper_defs {
        writeln!(writer, "{helper_def}")?;
    }
    for constraint_def in constraint_defs {
        writeln!(writer, "{constraint_def}")?;
    }

    writeln!(
        writer,
        "  def constrain_interactions {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) :="
    )?;
    writeln!(writer, "    Circuit.buses c = λ index =>")?;
    for branch in interaction_branches {
        writeln!(writer, "{branch}")?;
    }
    if interactions_by_bus.is_empty() {
        writeln!(writer, "    []")?;
    } else {
        writeln!(writer, "    else []")?;
    }
    writeln!(writer)?;
    writeln!(writer, "end {lean_air_name}.extraction")?;

    writeln!(writer, "------")?;
    Ok(())
}

pub fn extract_constraints_dag_to_lean_writer<F: Field, W: Write>(
    symbolic_constraints: &SymbolicConstraintsDag<F>,
    air_name: &str,
    writer: &mut W,
) -> io::Result<()> {
    let symbolic: SymbolicConstraints<_> = symbolic_constraints.into();
    extract_constraints_to_lean_writer(&symbolic, air_name, writer)
}
