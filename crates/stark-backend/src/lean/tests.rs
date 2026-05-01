use std::sync::Arc;

use p3_baby_bear::BabyBear;

use super::{
    air_file_stem, format_lean_air_name, render_air, write_constraints, write_interactions,
    write_schema, BusBinding, LeanRenderOptions, LeanWriteOptions,
};
use crate::{
    air_builders::symbolic::{
        symbolic_expression::SymbolicExpression,
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraints, SymbolicConstraintsDag,
    },
    interaction::Interaction,
};

const BABY_BEAR_P: u32 = 2_013_265_921;

fn main_var(part_index: usize, index: usize, offset: usize) -> SymbolicExpression<BabyBear> {
    SymbolicExpression::Variable(SymbolicVariable::new(
        Entry::Main { part_index, offset },
        index,
    ))
}

fn options_for(namespace: &str) -> LeanWriteOptions {
    LeanWriteOptions {
        render: LeanRenderOptions::default(),
        characteristic: Some(BABY_BEAR_P),
        fundamentals_import: "Fundamentals.Air",
        bus_defs_import: "Recursion.BusDefs",
        bus_defs_namespace: "Recursion.BusDefs",
        air_namespace: namespace.to_string(),
        schema_import: format!("{namespace}.Generated.Schema"),
        constraints_import: format!("{namespace}.Generated.Constraints"),
    }
}

#[test]
fn schema_emits_layout_and_columns() {
    let mut out = Vec::new();
    let names = vec!["is_valid".to_string(), "tidx".to_string()];
    write_schema(
        &mut out,
        "TinyAir",
        &names,
        &options_for("Recursion.Test.TinyAir"),
    )
    .unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("abbrev W : Nat := 2"));
    assert!(s.contains("def layout : TraceLayout := TraceLayout.singleMain W"));
    assert!(s.contains("def is_validIdx : Fin W := ⟨0, by decide⟩"));
    assert!(s.contains("def tidxRef : ColumnRef layout :="));
    assert!(s.contains("namespace Recursion.Test.TinyAir"));
    assert!(s.contains("end Recursion.Test.TinyAir"));
}

#[test]
fn constraints_inline_short_subtrees() {
    // is_valid * (is_valid - 1)
    let is_valid = Arc::new(main_var(0, 0, 0));
    let one = Arc::new(SymbolicExpression::Constant(BabyBear::new(1)));
    let sub = Arc::new(SymbolicExpression::Sub {
        x: is_valid.clone(),
        y: one,
        degree_multiple: 1,
    });
    let mul = SymbolicExpression::Mul {
        x: is_valid,
        y: sub,
        degree_multiple: 2,
    };
    let dag = SymbolicConstraintsDag::from(SymbolicConstraints::<BabyBear> {
        constraints: vec![mul],
        interactions: vec![],
    });

    let names = vec!["is_valid".to_string()];
    let opts = options_for("Recursion.Test.TinyAir");
    let rendered = render_air(&dag, &names, &[], &opts).unwrap();
    let mut out = Vec::new();
    write_constraints(&mut out, "TinyAir", &names, &rendered, &opts).unwrap();
    let s = String::from_utf8(out).unwrap();

    // No `inter_K` or local `let tN` for short single-use subtrees.
    assert!(!s.contains("inter_0"));
    assert!(!s.contains("let t0 :="));
    // Inlined form.
    assert!(s.contains("def expr_0 : Expr F layout := fun va =>"));
    assert!(s.contains("(va (.cell .local is_validRef) * (va (.cell .local is_validRef) - 1))"));
    assert!(s.contains("def constraintsList"));
    assert!(s.contains("def air : AIR F"));
    assert!(s.contains("structure RawConstraintsAt"));
    assert!(s.contains("theorem of_satisfiesRow"));
}

#[test]
fn constraints_hoist_subtree_shared_across_constraints() {
    // shared = is_valid * (is_valid - 1) -- 2 ops, used in both constraint_0 and constraint_1.
    let is_valid = Arc::new(main_var(0, 0, 0));
    let one = Arc::new(SymbolicExpression::Constant(BabyBear::new(1)));
    let sub = Arc::new(SymbolicExpression::Sub {
        x: is_valid.clone(),
        y: one,
        degree_multiple: 1,
    });
    let shared = Arc::new(SymbolicExpression::Mul {
        x: is_valid,
        y: sub,
        degree_multiple: 2,
    });
    let is_first = Arc::new(SymbolicExpression::IsFirstRow);
    let is_last = Arc::new(SymbolicExpression::IsLastRow);
    let c0 = SymbolicExpression::Mul {
        x: is_first,
        y: shared.clone(),
        degree_multiple: 3,
    };
    let c1 = SymbolicExpression::Mul {
        x: is_last,
        y: shared,
        degree_multiple: 3,
    };
    let dag = SymbolicConstraintsDag::from(SymbolicConstraints::<BabyBear> {
        constraints: vec![c0, c1],
        interactions: vec![],
    });

    let names = vec!["is_valid".to_string()];
    let opts = options_for("Recursion.Test.TinyAir");
    let rendered = render_air(&dag, &names, &[], &opts).unwrap();
    let mut out = Vec::new();
    write_constraints(&mut out, "TinyAir", &names, &rendered, &opts).unwrap();
    let s = String::from_utf8(out).unwrap();

    assert!(s.contains("Shared sub-expressions"));
    assert!(s.contains("def inter_0 : Expr F layout := fun va =>"));
    // Both constraints reference the helper.
    assert_eq!(s.matches("inter_0 va").count(), 2);
}

#[test]
fn interactions_group_by_bus_name() {
    // Two interactions on bus 60 (stackingTranscriptBus), one on bus 0.
    let m = Arc::new(SymbolicExpression::IsFirstRow);
    let v = Arc::new(SymbolicExpression::IsLastRow);
    let mk = |bus: u16| Interaction {
        message: vec![(*v).clone()],
        count: (*m).clone(),
        bus_index: bus,
        count_weight: 1,
    };
    let dag = SymbolicConstraintsDag::from(SymbolicConstraints::<BabyBear> {
        constraints: vec![],
        interactions: vec![mk(0), mk(60), mk(60)],
    });

    let bus_table = vec![
        BusBinding {
            vk_index: 0,
            lean_name: "transcriptBus".to_string(),
        },
        BusBinding {
            vk_index: 60,
            lean_name: "stackingTranscriptBus".to_string(),
        },
    ];
    let opts = options_for("Recursion.Test.TinyAir");
    let rendered = render_air(&dag, &[], &bus_table, &opts).unwrap();
    let mut out = Vec::new();
    write_interactions(&mut out, "TinyAir", &rendered, &opts).unwrap();
    let s = String::from_utf8(out).unwrap();

    assert!(s.contains("def transcriptBusInteractions"));
    assert!(s.contains("def stackingTranscriptBusInteractions"));
    assert!(s.contains("bus := .transcriptBus"));
    assert!(s.contains("bus := .stackingTranscriptBus"));
    assert!(s.contains("def allInteractions"));
    assert!(s.contains("transcriptBusInteractions ++ stackingTranscriptBusInteractions"));
}

#[test]
fn formats_generic_air_names_for_lean_identifiers() {
    assert_eq!(
        format_lean_air_name("Sha2BlockHasherVmAir<Sha256Config, Sha512Config>"),
        "Sha2BlockHasherVmAir_Sha256Config_Sha512Config"
    );
    assert_eq!(air_file_stem("WhirRoundAir<u8, 1>"), "WhirRoundAir");
}
