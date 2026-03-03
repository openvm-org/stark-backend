#![allow(clippy::useless_format)]
//! Extraction of AIR constraints (with interactions) from symbolic DAGs to Lean4 code.
//! This extraction can be used for standalone Lean4 code generation from serialized circuit
//! verifying keys.
//
// Original extraction code was introduced in Nethermind fork <https://github.com/NethermindEth/openvm-stark-backend/releases/tag/v1.2.2-extraction>

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
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

/// Code generation operates by printing the Lean4 code to stdout.
pub fn extract_constraints_to_lean<F: Field>(
    symbolic_constraints: &SymbolicConstraints<F>,
    air_name: &str,
) {
    println!("-----Constraints for {air_name}-----");
    println!("-----Used Columns-------------------");
    println!("{}", placeholder_column_names(symbolic_constraints));
    println!("-----Extracted constraints----------");
    for (idx, constraint) in symbolic_constraints.constraints.iter().enumerate() {
        let constraint_text = format!(
                    "  @[simp]\n  def constraint_{idx} {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) (row: ℕ) :=\n    {} = 0\n",
                    symbolic_expression_to_string(constraint, "", None)
                );
        if constraint_text.contains("Circuit.permutation") {
            let commented = constraint_text
                .split("\n")
                .map(|line| format!("-- {line}"))
                .join("\n");
            println!("{commented}");
        } else {
            println!("{constraint_text}");
        }
    }
    println!("  def constrain_interactions {{C : Type → Type → Type}} {{F ExtF : Type}} [Field F] [Field ExtF] [Circuit F ExtF C] (c : C F ExtF) :=");
    println!("    Circuit.buses c = λ index =>");

    let mut interactions_by_bus: HashMap<u16, Vec<Interaction<_>>> = HashMap::new();
    for interaction in symbolic_constraints.interactions.iter() {
        if let Some(list) = interactions_by_bus.get_mut(&interaction.bus_index) {
            list.push(interaction.clone());
        } else {
            interactions_by_bus.insert(interaction.bus_index, vec![interaction.clone()]);
        }
    }

    for (idx, (bus_idx, interactions)) in interactions_by_bus
        .iter()
        .sorted_by(|(a, _), (c, _)| (**a).cmp(*c))
        .enumerate()
    {
        let all_rows = "(List.range (Circuit.last_row c + 1))";
        let row_interactions = interactions
            .iter()
            .map(|interaction| {
                let multiplicity = symbolic_expression_to_string(&interaction.count, "", None);
                let data = format!(
                    "[{}]",
                    interaction
                        .message
                        .iter()
                        .map(|x| symbolic_expression_to_string(x, "", None))
                        .join(", ")
                );
                format!("({multiplicity}, {data})")
            })
            .join(", ");
        let expr = format!("{all_rows}.flatMap (λ row => [{row_interactions}])");

        println!(
            "      {}if index = {} then {expr}",
            if idx == 0 { "" } else { "else " },
            bus_idx,
        )
        // println!("\nInteraction {idx}:");
        // println!("    Bus {}", interaction.bus_index);
        // println!(
        //     "    Message: ({})",
        //     interaction
        //         .message
        //         .iter()
        //         .map(|x| symbolic_expression_to_string(x, "", false, None))
        //         .join(", ")
        // );
        // println!(
        //     "    Count: {}",
        //     symbolic_expression_to_string(&interaction.count, "", false, None)
        // );
        // println!(
        //     "    Count weight: {}",
        //     interaction.count_weight
        // );
    }
    if interactions_by_bus.is_empty() {
        println!("    []")
    } else {
        println!("    else []")
    }
    println!("-----Constraint simplification------");
    let simplification_proof = [
        "apply Iff.intro",
        ". intro h",
        "  simp [openvm_encapsulation, NAME_constraint_and_interaction_simplification] at h",
        "  simp only [NAME_constraint_and_interaction_simplification]",
        "  exact h",
        ". intro h",
        "  simp [openvm_encapsulation, NAME_constraint_and_interaction_simplification]",
        "  simp only [NAME_constraint_and_interaction_simplification] at h",
        "  exact h",
    ]
    .join("\n");
    for (idx, constraint) in symbolic_constraints.constraints.iter().enumerate() {
        let constraint_text = format!("{}", symbolic_expression_to_string(constraint, "", None));

        let simplified_constraint_text = [
            format!("@[NAME_constraint_and_interaction_simplification]"),
            format!("def constraint_{idx} (air : Valid_NAME F ExtF) (row : ℕ) : Prop :="),
            format!("  sorry"),
        ]
        .join("\n");

        let simplified_of_extracted = [
            format!("@[NAME_air_simplification]"),
            format!("lemma constraint_{idx}_of_extraction"),
            format!("    (air : Valid_NAME F ExtF) (row : ℕ)"),
            format!(": NAME.extraction.constraint_{idx} air row ↔ constraint_{idx} air row := by"),
            simplification_proof.clone(),
        ]
        .join("\n");

        let output_text = format!("{simplified_constraint_text}\n\n{simplified_of_extracted}");

        if constraint_text.contains("Circuit.permutation") {
            let commented = output_text
                .split("\n")
                .map(|line| format!("-- {line}"))
                .join("\n");
            println!("{commented}\n");
        } else {
            println!("{output_text}\n");
        }
    }
    println!("-----Interaction simplification-----");
    let mut full_expr = vec![];
    for (bus_idx, _interactions) in interactions_by_bus
        .iter()
        .sorted_by(|(a, _), (c, _)| (**a).cmp(*c))
    {
        let (bus_name, bus_idx_name) = if *bus_idx == 0 {
            (format!("execution"), format!("ExecutionBus"))
        } else if *bus_idx == 1 {
            (format!("memory"), format!("MemoryBus"))
        } else if *bus_idx == 4 {
            (format!("rangeChecker"), format!("RangeCheckerBus"))
        } else if *bus_idx == 8 {
            (format!("readInstruction"), format!("ReadInstructionBus"))
        } else if *bus_idx == 9 {
            (format!("bitwise"), format!("BitwiseBus"))
        } else if *bus_idx == 11 {
            (
                format!("rangeTupleChecker"),
                format!("RangeTupleCheckerBus"),
            )
        } else {
            (format!("bus_{bus_idx}"), format!("{bus_idx}"))
        };

        let row_expression = [
            format!("@[NAME_constraint_and_interaction_simplification]",),
            format!(
                "def {bus_name}Bus_row (air : Valid_NAME F ExtF) (row : ℕ) : List (F × List F) :=",
            ),
            format!("  sorry"),
        ]
        .join("\n");

        let constrain_lemma = [
                    format!("lemma constrain_{bus_name}_interactions"),
                    format!("  (air : Valid_NAME F ExtF)"),
                    format!("  (h : NAME.extraction.constrain_interactions air)"),
                    format!(":"),
                    format!("  air.buses {bus_idx_name} = (List.range (air.last_row + 1)).flatMap (λ row => {bus_name}Bus_row air row)"),
                    format!(":= by"),
                    format!("  unfold NAME.extraction.constrain_interactions at h"),
                    format!("  simp [openvm_encapsulation] at h"),
                    format!("  simp [h]; clear h"),
                    format!("  rfl"),
                ].join("\n");

        println!("{row_expression}\n\n{constrain_lemma}\n\n");

        full_expr.push(format!(
                    "if index = {bus_idx_name} then (List.range (air.last_row + 1)).flatMap ({bus_name}Bus_row air)"
                ))
    }

    full_expr.push(format!("[]"));

    let full_expr = full_expr.join("\nelse ");

    let constrain_interactions_lemma = [
        format!("def constrain_interactions (air : Valid_NAME F ExtF) : Prop :="),
        format!("air.buses = fun index ↦"),
        full_expr,
    ]
    .join("\n");

    let constrain_interactions_of_extraction_lemma = [
        "@[NAME_air_simplification]",
        "lemma constrain_interactions_of_extraction",
        "  (air : Valid_NAME F ExtF)",
        "  (h : NAME.extraction.constrain_interactions air)",
        ": constrain_interactions air := by",
        "  unfold NAME.extraction.constrain_interactions at h",
        "  simp [openvm_encapsulation] at h",
        "  exact h",
    ]
    .join("\n");

    println!("{constrain_interactions_lemma}\n\n{constrain_interactions_of_extraction_lemma}\n\n");

    println!("-----All hold definitions-----------");

    let num_constraints = symbolic_constraints.constraints.len();

    let num_comment_constraints = symbolic_constraints
        .constraints
        .iter()
        .filter(|constraint| {
            let constraint_text =
                format!("{}", symbolic_expression_to_string(constraint, "", None));
            constraint_text.contains("Circuit.permutation")
        })
        .count();

    let extracted_row_constraint_list = (0..num_constraints)
        .map(|idx| {
            let constraint = format!("    NAME.extraction.constraint_{idx} air row,");

            if idx >= num_constraints - num_comment_constraints {
                format!("-- {constraint}")
            } else {
                constraint
            }
        })
        .join("\n");

    let extract_row_constraint_list_def = [
        format!("@[simp]"),
        format!("def extracted_row_constraint_list"),
        format!("  [Field ExtF]"),
        format!("  (air : Valid_NAME FBB ExtF)"),
        format!("  (row : ℕ)"),
        format!(": List Prop :="),
        format!("  ["),
        extracted_row_constraint_list,
        format!("  ]"),
    ]
    .join("\n");

    let all_hold_def = [
        "@[simp]",
        "def allHold",
        "  [Field ExtF]",
        "  (air : Valid_NAME FBB ExtF)",
        "  (row : ℕ)",
        "  (_ : row ≤ air.last_row)",
        ": Prop :=",
        "  NAME.extraction.constrain_interactions air ∧",
        "  List.Forall (·) (extracted_row_constraint_list air row)",
    ]
    .join("\n");

    let row_constraint_list = (0..num_constraints)
        .map(|idx| {
            let constraint = format!("    constraint_{idx} air row,");

            if idx >= num_constraints - num_comment_constraints {
                format!("-- {constraint}")
            } else {
                constraint
            }
        })
        .join("\n");

    let row_constraint_list_def = [
        format!("@[simp]"),
        format!("def row_constraint_list"),
        format!("  [Field ExtF]"),
        format!("  (air : Valid_NAME FBB ExtF)"),
        format!("  (row : ℕ)"),
        format!(": List Prop :="),
        format!("  ["),
        row_constraint_list,
        format!("  ]"),
    ]
    .join("\n");

    let all_hold_simplified = [
        "@[simp]",
        "def allHold_simplified",
        "  [Field ExtF]",
        "  (air : Valid_NAME FBB ExtF)",
        "  (row : ℕ)",
        "  (_ : row ≤ air.last_row)",
        ": Prop :=",
        "  constrain_interactions air ∧",
        "  List.Forall (·) (row_constraint_list air row)",
    ]
    .join("\n");

    let all_hold_simplified_of_all_hold = [
        "lemma allHold_simplified_of_allHold",
        "  [Field ExtF]",
        "  (air : Valid_NAME FBB ExtF)",
        "  (row : ℕ)",
        "  (h_row : row ≤ air.last_row)",
        ": allHold air row h_row ↔ allHold_simplified air row h_row := by",
        "  unfold allHold allHold_simplified",
        "  apply Iff.and",
        "  . unfold NAME.extraction.constrain_interactions",
        "    simp [openvm_encapsulation]",
        "    rfl",
        "  . simp only [extracted_row_constraint_list,",
        "              row_constraint_list,",
        "              NAME_air_simplification]",
    ]
    .join("\n");

    let all_hold_section = [
        extract_row_constraint_list_def,
        all_hold_def,
        row_constraint_list_def,
        all_hold_simplified,
        all_hold_simplified_of_all_hold,
    ]
    .join("\n\n");

    println!("{all_hold_section}");

    println!("------");
}


fn collect_variables<F>(expression: &SymbolicExpression<F>, leaves: &mut HashSet<SymbolicVariable<F>>)
    where F: Clone + std::cmp::Eq + std::hash::Hash
{
    match expression {
        SymbolicExpression::Variable(symbolic_variable) => {
            leaves.insert(symbolic_variable.clone());
        },
        SymbolicExpression::Add { x, y, degree_multiple: _ } => {
            collect_variables(x, leaves);
            collect_variables(y, leaves);
        },
        SymbolicExpression::Sub { x, y, degree_multiple: _ } => {
            collect_variables(x, leaves);
            collect_variables(y, leaves);
        },
        SymbolicExpression::Neg { x, degree_multiple: _ } => {
            collect_variables(x, leaves);
        },
        SymbolicExpression::Mul { x, y, degree_multiple: _ } => {
            collect_variables(x, leaves);
            collect_variables(y, leaves);
        },
        _ => {}
    }
}


fn get_entry_type_id(entry: &Entry) -> u8 {
    match entry {
        Entry::Preprocessed { offset: _ } => 0,
        Entry::Main { part_index: _, offset: _ } => 1,
        Entry::Permutation { offset: _ } => 2,
        Entry::Public => 3,
        Entry::Challenge => 4,
        Entry::Exposed => 5,
    }
}

fn placeholder_column_names<F>(constraints: &SymbolicConstraints<F>) -> String
    where F: Clone + std::cmp::Eq + std::hash::Hash
{
    let leaves = {
        let mut leaves = HashSet::new();

        constraints.constraints.iter().for_each(|expr| {
            collect_variables(expr, &mut leaves);
        });

        constraints.interactions.iter().for_each(|interaction| {
            collect_variables(&interaction.count, &mut leaves);
            interaction.message.iter().for_each(|expr| {
                collect_variables(expr, &mut leaves);
            });
        });

        leaves.into_iter().sorted_by(|lhs, rhs| {
            let type_order = get_entry_type_id(&lhs.entry).cmp(&get_entry_type_id(&rhs.entry));

            let index_order = lhs.index.cmp(&rhs.index);

            let (part_index_order, offset_order) = match (lhs.entry, rhs.entry) {
                (Entry::Preprocessed { offset: l_offset }, Entry::Preprocessed { offset: r_offset }) => (Ordering::Equal, l_offset.cmp(&r_offset)),
                (Entry::Main { part_index: l_part_index, offset: l_offset }, Entry::Main { part_index: r_part_index, offset: r_offset }) => (l_part_index.cmp(&r_part_index), l_offset.cmp(&r_offset)),
                (Entry::Permutation { offset: l_offset }, Entry::Permutation { offset: r_offset }) => (Ordering::Equal, l_offset.cmp(&r_offset)),
                _ => (Ordering::Equal, Ordering::Equal),
            };

            type_order.then(part_index_order).then(index_order).then(offset_order)
        }).collect_vec()
    };

    leaves.iter().map(|leaf| {
        let column = leaf.index;
        match leaf.entry {
            Entry::Preprocessed { offset } =>
                format!("--def Circuit._ (c: Circuit F ExtF) (row: N) := c.preprocessed (column := {column}) (row := row) (rotation := {offset})"),
            Entry::Main { part_index, offset } =>
                format!("--def Circuit._ (c: Circuit F ExtF) (row: N) := c.main (id := {part_index}) (column := {column}) (row := row) (rotation := {offset})"),
            Entry::Permutation { offset } =>
                format!("--def Circuit._ (c: Circuit F ExtF) (row: N) := c.permutation (column := {column}) (row := row) (rotation := {offset})"),
            Entry::Public =>
                format!("--def Circuit._ (c: Circuit F ExtF) := c.public (index := {column})"),
            Entry::Challenge =>
                format!("--def Circuit._ (c: Circuit F ExtF) := c.challenge (index := {column})"),
            Entry::Exposed =>
                format!("--def Circuit._ (c: Circuit F ExtF) := c.exposed (index := {column})"),
        }
    }).join("\n")
}

fn symbolic_expression_to_string<F: Field>(x: &SymbolicExpression<F>, scoping: &str, characteristic: Option<u32>) -> String {
    let x = x.clone();
    symbolic_expression_to_string_impl(&x, scoping, characteristic)
}

fn symbolic_expression_to_string_impl<F: Field>(x: &SymbolicExpression<F>, scoping: &str, characteristic: Option<u32>) -> String {
    match x {
        SymbolicExpression::Variable(symbolic_variable) =>
            format!(
                "{scoping}{}",
                match symbolic_variable.entry {
                    Entry::Preprocessed{offset}=>format!("(Circuit.preprocessed c (column := {}) (row := row) (rotation := {offset}))",symbolic_variable.index),
                    Entry::Main{offset, part_index}=>format!("(Circuit.main c (id := {part_index}) (column := {}) (row := row) (rotation := {offset}))",symbolic_variable.index),
                    Entry::Permutation{offset}=>format!("(Circuit.permutation c (column := {}) (row := row) (rotation := {offset}))",symbolic_variable.index),
                    Entry::Public=>format!("(Circuit.public c (index := {}))",symbolic_variable.index),
                    Entry::Challenge=>format!("(Circuit.challenge c (index := {}))",symbolic_variable.index),
                    Entry::Exposed =>format!("(Circuitc.exposed c (index := {}))",symbolic_variable.index),
                },

            ),
        SymbolicExpression::IsFirstRow => format!("(Circuit.isFirstRow c row)"),
        SymbolicExpression::IsLastRow => format!("(Circuit.isLastRow c row)"),
        SymbolicExpression::IsTransition => format!("(Circuit.isTransitionRow c row)"),
        SymbolicExpression::Constant(x) => {
            let num = str::parse::<u32>(&format!("{x}"));
            match num {
                Ok(num) => {
                    match characteristic {
                        Some(characteristic) => {
                            if num >= characteristic {
                                format!("{x}")
                            } else if characteristic - num < num {
                                format!("-{}", characteristic - num)
                            } else {
                                format!("{x}")
                            }
                        },
                        None => format!("{x}"),
                    }
                },
                Err(_) => format!("{x}"),
            }
        },
        SymbolicExpression::Add { x, y, degree_multiple:_ } => {
            let lhs = symbolic_expression_to_string_impl(x, scoping, characteristic);
            let rhs = symbolic_expression_to_string_impl(y, scoping, characteristic);
            format!("({lhs} + {rhs})")
        },
        SymbolicExpression::Sub { x, y, degree_multiple:_ } => {
            let lhs = symbolic_expression_to_string_impl(x, scoping, characteristic);
            let rhs = symbolic_expression_to_string_impl(y, scoping, characteristic);
            format!("({lhs} - {rhs})")
        },
        SymbolicExpression::Neg { x, degree_multiple:_ } => {
            let leaf = symbolic_expression_to_string_impl(x, scoping, characteristic);
            format!("-({leaf})")
        },
        SymbolicExpression::Mul { x, y, degree_multiple:_ } => {
            let lhs = symbolic_expression_to_string_impl(x, scoping, characteristic);
            let rhs = symbolic_expression_to_string_impl(y, scoping, characteristic);
            format!("({lhs} * {rhs})")
        },
    }
}
