//! Extraction of AIR constraints (with interactions) from symbolic DAGs to Lean4 code.
//! This extraction can be used for standalone Lean4 code generation from serialized circuit
//! verifying keys.
//
// Original extraction code was introduced in Nethermind fork <https://github.com/NethermindEth/openvm-stark-backend/releases/tag/v1.2.2-extraction>

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
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

#[derive(Clone, Debug)]
pub struct LeanExtractionOptions<'a> {
    pub attrs_import: Option<&'a str>,
    pub helper_attr: Option<&'a str>,
    pub register_inline_attrs: bool,
}

impl Default for LeanExtractionOptions<'_> {
    fn default() -> Self {
        Self {
            attrs_import: None,
            helper_attr: None,
            register_inline_attrs: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LeanConstraintsScaffoldOptions<'a> {
    pub air_import: &'a str,
    pub air_definition_name: Option<&'a str>,
    pub attrs_import: &'a str,
    pub extraction_import: &'a str,
    pub bus_defs_import: &'a str,
    pub bus_defs_namespace: &'a str,
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
    extract_constraints_to_lean_writer_with_options(
        symbolic_constraints,
        air_name,
        &LeanExtractionOptions::default(),
        writer,
    )
}

pub fn extract_constraints_to_lean_writer_with_options<F: Field, W: Write>(
    symbolic_constraints: &SymbolicConstraints<F>,
    air_name: &str,
    options: &LeanExtractionOptions<'_>,
    writer: &mut W,
) -> io::Result<()> {
    let lean_air_name = format_lean_air_name(air_name);

    if let Some(attrs_import) = options.attrs_import {
        writeln!(writer, "import {attrs_import}")?;
        writeln!(writer)?;
        writeln!(writer, "import LeanZKCircuit.OpenVM.Circuit")?;
        writeln!(writer, "import Mathlib.Algebra.Field.Basic")?;
    } else {
        writeln!(writer, "import Mathlib.Algebra.Field.Basic")?;
        writeln!(writer)?;
        writeln!(writer, "import LeanZKCircuit.OpenVM.Circuit")?;
    }
    writeln!(writer)?;
    writeln!(writer, "set_option linter.all false")?;
    if options.register_inline_attrs {
        writeln!(writer)?;
        writeln!(
            writer,
            "register_simp_attr {lean_air_name}_air_simplification"
        )?;
        writeln!(
            writer,
            "register_simp_attr {lean_air_name}_constraint_and_interaction_simplification"
        )?;
    }
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
        if let Some(helper_attr) = options.helper_attr {
            writeln!(writer, "  @[{helper_attr}]")?;
        }
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
    extract_constraints_dag_to_lean_writer_with_options(
        symbolic_constraints,
        air_name,
        &LeanExtractionOptions::default(),
        writer,
    )
}

pub fn extract_constraints_dag_to_lean_writer_with_options<F: Field, W: Write>(
    symbolic_constraints: &SymbolicConstraintsDag<F>,
    air_name: &str,
    options: &LeanExtractionOptions<'_>,
    writer: &mut W,
) -> io::Result<()> {
    let symbolic: SymbolicConstraints<_> = symbolic_constraints.into();
    extract_constraints_to_lean_writer_with_options(&symbolic, air_name, options, writer)
}

pub fn extraction_intermediate_attrs_to_lean_writer<'a, I, W>(
    air_names: I,
    writer: &mut W,
) -> io::Result<()>
where
    I: IntoIterator<Item = &'a str>,
    W: Write,
{
    let air_names = air_names
        .into_iter()
        .map(format_lean_air_name)
        .collect::<BTreeSet<_>>();

    writeln!(writer, "import Mathlib.Algebra.Field.Basic")?;
    writeln!(writer)?;
    for air_name in air_names {
        writeln!(
            writer,
            "register_simp_attr {air_name}_extraction_intermediates"
        )?;
    }
    Ok(())
}

pub fn bus_defs_to_lean_writer<'a, F: Field + 'a, I, W>(
    symbolic_constraints: I,
    namespace: &str,
    writer: &mut W,
) -> io::Result<()>
where
    I: IntoIterator<Item = &'a SymbolicConstraintsDag<F>>,
    W: Write,
{
    let bus_arities = collect_bus_arities(symbolic_constraints)?;

    writeln!(writer, "namespace {namespace}")?;
    writeln!(writer)?;

    for (idx, arity) in &bus_arities {
        writeln!(writer, "structure Bus{idx}Entry (F : Type) where")?;
        writeln!(writer, "  multiplicity : F")?;
        for field_idx in 0..*arity {
            writeln!(writer, "  v{field_idx} : F")?;
        }
        writeln!(writer, "deriving BEq, DecidableEq, Inhabited")?;
        writeln!(writer)?;
    }

    for (idx, arity) in &bus_arities {
        writeln!(writer, "@[simp]")?;
        writeln!(
            writer,
            "def Bus{idx}Entry.toRaw (entry : Bus{idx}Entry F) : F × List F :="
        )?;
        let fields = if *arity == 0 {
            String::from("[]")
        } else {
            let entries = (0..*arity)
                .map(|field_idx| format!("entry.v{field_idx}"))
                .join(", ");
            format!("[{entries}]")
        };
        writeln!(writer, "  (entry.multiplicity, {fields})")?;
        writeln!(writer)?;
    }

    writeln!(writer, "end {namespace}")?;
    Ok(())
}

pub fn constraints_scaffold_to_lean_writer<F: Field, W: Write>(
    symbolic_constraints: &SymbolicConstraintsDag<F>,
    air_name: &str,
    options: &LeanConstraintsScaffoldOptions<'_>,
    writer: &mut W,
) -> io::Result<()> {
    let symbolic: SymbolicConstraints<_> = symbolic_constraints.into();
    let lean_air_name = format_lean_air_name(air_name);
    let air_definition_name = options.air_definition_name.unwrap_or(&lean_air_name);
    let valid_air_name = format!("Valid_{air_definition_name}");
    let constraint_attr = format!("{lean_air_name}_constraint_and_interaction_simplification");
    let air_attr = format!("{lean_air_name}_air_simplification");
    let intermediates_attr = format!("{lean_air_name}_extraction_intermediates");
    let bus_stats = collect_bus_stats(std::iter::once(symbolic_constraints))?;

    writeln!(writer, "import {}", options.air_import)?;
    writeln!(writer, "import {}", options.bus_defs_import)?;
    writeln!(writer, "import {}", options.attrs_import)?;
    writeln!(writer, "import {}", options.extraction_import)?;
    writeln!(writer)?;
    writeln!(writer, "import Mathlib.Algebra.Field.Basic")?;
    writeln!(writer)?;
    writeln!(writer, "set_option linter.unusedVariables false")?;
    writeln!(writer, "set_option linter.unusedSimpArgs false")?;
    writeln!(writer)?;
    writeln!(writer, "namespace {lean_air_name}.constraints")?;
    writeln!(writer)?;
    writeln!(writer, "section constraint_simplification")?;
    writeln!(writer)?;
    writeln!(writer, "  variable {{F ExtF : Type}} [Field F] [Field ExtF]")?;
    writeln!(writer)?;
    writeln!(writer, "  section constraints")?;
    writeln!(writer)?;

    for idx in 0..symbolic.constraints.len() {
        writeln!(writer, "    @[{constraint_attr}]")?;
        writeln!(
            writer,
            "    def constraint_{idx} (air : {valid_air_name} F ExtF) (row : ℕ) : Prop :="
        )?;
        writeln!(writer, "      -- TODO: fill")?;
        writeln!(writer, "      True")?;
        writeln!(writer)?;
    }

    for idx in 0..symbolic.constraints.len() {
        writeln!(writer, "    @[{air_attr}]")?;
        writeln!(writer, "    lemma constraint_{idx}_of_extraction")?;
        writeln!(writer, "      (air : {valid_air_name} F ExtF) (row : ℕ)")?;
        writeln!(
            writer,
            "    : {lean_air_name}.extraction.constraint_{idx} air row ↔ constraint_{idx} air row := by"
        )?;
        writeln!(
            writer,
            "      simp only [{lean_air_name}.extraction.constraint_{idx}, constraint_{idx},"
        )?;
        writeln!(
            writer,
            "        {intermediates_attr}, openvm_encapsulation]"
        )?;
        writeln!(writer)?;
    }

    writeln!(writer, "  end constraints")?;
    writeln!(writer)?;
    writeln!(writer, "  section interactions")?;
    writeln!(writer)?;

    for (bus_idx, (arity, interaction_count)) in &bus_stats {
        writeln!(writer, "    @[{constraint_attr}]")?;
        writeln!(
            writer,
            "    def bus{bus_idx}_row (air : {valid_air_name} F ExtF) (row : ℕ) : List ({}.Bus{}Entry F) :=",
            options.bus_defs_namespace,
            bus_idx
        )?;
        for line in render_placeholder_bus_row_lines(air_name, *bus_idx, *arity, *interaction_count) {
            writeln!(writer, "{line}")?;
        }
        writeln!(writer)?;
    }

    for bus_idx in bus_stats.keys() {
        writeln!(writer, "    @[{air_attr}]")?;
        writeln!(writer, "    lemma constrain_bus{bus_idx}_interactions_of_extraction")?;
        writeln!(writer, "      (air : {valid_air_name} F ExtF)")?;
        writeln!(
            writer,
            "      (h : {lean_air_name}.extraction.constrain_interactions air)"
        )?;
        writeln!(
            writer,
            "    : air.buses {bus_idx} = (List.range (air.last_row + 1)).flatMap (fun row => (bus{bus_idx}_row air row).map {}.Bus{}Entry.toRaw) := by",
            options.bus_defs_namespace,
            bus_idx
        )?;
        writeln!(
            writer,
            "      unfold {lean_air_name}.extraction.constrain_interactions at h"
        )?;
        writeln!(writer, "      simp [openvm_encapsulation] at h")?;
        writeln!(
            writer,
            "      simp [h, {constraint_attr}, {intermediates_attr}, openvm_encapsulation]"
        )?;
        writeln!(writer)?;
    }

    writeln!(writer, "    @[{constraint_attr}]")?;
    writeln!(
        writer,
        "    def interactionBuses (air : {valid_air_name} F ExtF) (index : ℕ) : List (F × List F) :="
    )?;
    if bus_stats.is_empty() {
        writeln!(writer, "      []")?;
    } else {
        for (idx, bus_idx) in bus_stats.keys().enumerate() {
            let branch = if idx == 0 { "if" } else { "else if" };
            writeln!(writer, "      {branch} index = {bus_idx} then")?;
            writeln!(
                writer,
                "        (List.range (air.last_row + 1)).flatMap"
            )?;
            writeln!(
                writer,
                "          (fun row => (bus{bus_idx}_row air row).map {}.Bus{}Entry.toRaw)",
                options.bus_defs_namespace,
                bus_idx
            )?;
        }
        writeln!(writer, "      else")?;
        writeln!(writer, "        []")?;
    }
    writeln!(writer)?;

    writeln!(writer, "    @[{constraint_attr}]")?;
    writeln!(
        writer,
        "    def constrain_interactions (air : {valid_air_name} F ExtF) : Prop :="
    )?;
    writeln!(writer, "      air.buses = interactionBuses air")?;
    writeln!(writer)?;
    writeln!(writer, "  end interactions")?;
    writeln!(writer)?;
    writeln!(writer, "end constraint_simplification")?;
    writeln!(writer)?;
    writeln!(writer, "section allHold")?;
    writeln!(writer)?;
    writeln!(writer, "  variable {{F ExtF : Type}} [Field F] [Field ExtF]")?;
    writeln!(writer)?;
    writeln!(writer, "  def extracted_row_constraint_list")?;
    writeln!(writer, "    (air : {valid_air_name} F ExtF)")?;
    writeln!(writer, "    (row : ℕ)")?;
    writeln!(writer, "  : List Prop :=")?;
    if symbolic.constraints.is_empty() {
        writeln!(writer, "    []")?;
    } else {
        writeln!(writer, "    [")?;
        for idx in 0..symbolic.constraints.len() {
            let suffix = if idx + 1 == symbolic.constraints.len() {
                ""
            } else {
                ","
            };
            writeln!(
                writer,
                "      {lean_air_name}.extraction.constraint_{idx} air row{suffix}"
            )?;
        }
        writeln!(writer, "    ]")?;
    }
    writeln!(writer)?;
    writeln!(writer, "  @[simp]")?;
    writeln!(writer, "  def allHold")?;
    writeln!(writer, "    (air : {valid_air_name} F ExtF)")?;
    writeln!(writer, "    (row : ℕ)")?;
    writeln!(writer, "    (_ : row ≤ air.last_row)")?;
    writeln!(writer, "  : Prop :=")?;
    writeln!(writer, "    {lean_air_name}.extraction.constrain_interactions air ∧")?;
    writeln!(
        writer,
        "    List.Forall (·) (extracted_row_constraint_list air row)"
    )?;
    writeln!(writer)?;
    writeln!(writer, "  def row_constraint_list")?;
    writeln!(writer, "    (air : {valid_air_name} F ExtF)")?;
    writeln!(writer, "    (row : ℕ)")?;
    writeln!(writer, "  : List Prop :=")?;
    if symbolic.constraints.is_empty() {
        writeln!(writer, "    []")?;
    } else {
        writeln!(writer, "    [")?;
        for idx in 0..symbolic.constraints.len() {
            let suffix = if idx + 1 == symbolic.constraints.len() {
                ""
            } else {
                ","
            };
            writeln!(writer, "      constraint_{idx} air row{suffix}")?;
        }
        writeln!(writer, "    ]")?;
    }
    writeln!(writer)?;
    writeln!(writer, "  @[simp]")?;
    writeln!(writer, "  def allHold_simplified")?;
    writeln!(writer, "    (air : {valid_air_name} F ExtF)")?;
    writeln!(writer, "    (row : ℕ)")?;
    writeln!(writer, "    (_ : row ≤ air.last_row)")?;
    writeln!(writer, "  : Prop :=")?;
    writeln!(writer, "    constrain_interactions air ∧")?;
    writeln!(writer, "    List.Forall (·) (row_constraint_list air row)")?;
    writeln!(writer)?;
    writeln!(writer, "  lemma allHold_simplified_of_allHold")?;
    writeln!(writer, "    (air : {valid_air_name} F ExtF)")?;
    writeln!(writer, "    (row : ℕ)")?;
    writeln!(writer, "    (h_row : row ≤ air.last_row)")?;
    writeln!(
        writer,
        "  : allHold air row h_row ↔ allHold_simplified air row h_row := by"
    )?;
    writeln!(writer, "    unfold allHold allHold_simplified")?;
    writeln!(writer, "    apply Iff.and")?;
    writeln!(writer, "    · constructor")?;
    writeln!(writer, "      · intro h")?;
    writeln!(
        writer,
        "        unfold constrain_interactions {lean_air_name}.extraction.constrain_interactions interactionBuses at *"
    )?;
    writeln!(
        writer,
        "        simp [{constraint_attr}, {intermediates_attr}, openvm_encapsulation] at *"
    )?;
    writeln!(writer, "        exact h")?;
    writeln!(writer, "      · intro h")?;
    writeln!(
        writer,
        "        unfold constrain_interactions {lean_air_name}.extraction.constrain_interactions interactionBuses at *"
    )?;
    writeln!(
        writer,
        "        simp [{constraint_attr}, {intermediates_attr}, openvm_encapsulation] at *"
    )?;
    writeln!(writer, "        exact h")?;
    writeln!(
        writer,
        "    · simp only [extracted_row_constraint_list, row_constraint_list, {air_attr}]"
    )?;
    writeln!(writer)?;
    writeln!(writer, "end allHold")?;
    writeln!(writer)?;
    writeln!(writer, "end {lean_air_name}.constraints")?;
    Ok(())
}

fn collect_bus_arities<'a, F: Field + 'a, I>(
    symbolic_constraints: I,
) -> io::Result<BTreeMap<BusIndex, usize>>
where
    I: IntoIterator<Item = &'a SymbolicConstraintsDag<F>>,
{
    Ok(collect_bus_stats(symbolic_constraints)?
        .into_iter()
        .map(|(bus_idx, (arity, _))| (bus_idx, arity))
        .collect())
}

fn collect_bus_stats<'a, F: Field + 'a, I>(
    symbolic_constraints: I,
) -> io::Result<BTreeMap<BusIndex, (usize, usize)>>
where
    I: IntoIterator<Item = &'a SymbolicConstraintsDag<F>>,
{
    let mut bus_stats = BTreeMap::new();
    for symbolic_constraints in symbolic_constraints {
        let symbolic: SymbolicConstraints<_> = symbolic_constraints.into();
        for interaction in symbolic.interactions {
            let arity = interaction.message.len();
            match bus_stats.get_mut(&interaction.bus_index) {
                Some((prev_arity, count)) => {
                    if *prev_arity != arity {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "bus {} has inconsistent arity: {} vs {}",
                                interaction.bus_index, *prev_arity, arity
                            ),
                        ));
                    }
                    *count += 1;
                }
                None => {
                    bus_stats.insert(interaction.bus_index, (arity, 1));
                }
            }
        }
    }
    Ok(bus_stats)
}

fn render_placeholder_bus_row_lines(
    air_name: &str,
    bus_idx: BusIndex,
    arity: usize,
    interaction_count: usize,
) -> Vec<String> {
    if air_name == "PublicValuesAir" && bus_idx == 0 && arity == 4 && interaction_count == 1 {
        return vec![
            "      -- TODO: fill".to_string(),
            "      [".to_string(),
            "        {".to_string(),
            "          multiplicity := -air.is_valid row 0".to_string(),
            "          v0 := air.proof_idx row 0".to_string(),
            "          v1 := air.tidx row 0".to_string(),
            "          v2 := air.value row 0".to_string(),
            "          v3 := 0".to_string(),
            "        }".to_string(),
            "      ]".to_string(),
        ];
    }

    let mut lines = vec!["      -- TODO: fill".to_string(), "      [".to_string()];
    for entry_idx in 0..interaction_count {
        lines.push("        {".to_string());
        lines.push("          multiplicity := 0".to_string());
        for field_idx in 0..arity {
            lines.push(format!("          v{field_idx} := 0"));
        }
        if entry_idx + 1 == interaction_count {
            lines.push("        }".to_string());
        } else {
            lines.push("        },".to_string());
        }
    }
    lines.push("      ]".to_string());
    lines
}
