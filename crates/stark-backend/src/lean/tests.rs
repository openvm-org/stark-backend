use p3_baby_bear::BabyBear;

use super::{
    extract_constraints_to_lean_writer, format_lean_air_name,
    render::{
        expression_direct_use_counts, symbolic_constraint_to_lean_definitions, LeanRenderContext,
    },
};
use crate::{
    air_builders::symbolic::{symbolic_expression::SymbolicExpression, SymbolicConstraints},
    interaction::Interaction,
};

#[test]
fn symbolic_constraint_avoids_inter_defs_for_low_use_subexpressions() {
    let delta =
        SymbolicExpression::from(BabyBear::new(2)) - SymbolicExpression::from(BabyBear::new(1));
    let inner = delta.clone() + (SymbolicExpression::from(BabyBear::new(3)) * delta);
    let expr = SymbolicExpression::IsFirstRow * inner;

    let mut context = LeanRenderContext {
        use_counts: expression_direct_use_counts(std::slice::from_ref(&expr)),
        ..Default::default()
    };
    let (helper_defs, rendered) =
        symbolic_constraint_to_lean_definitions(&expr, 5, "", None, &mut context);

    assert!(helper_defs.is_empty());
    assert!(!rendered.contains("def inter_0"));
    assert!(rendered.contains("let t0 := "));
    assert!(rendered.contains("def constraint_5"));
    assert!(rendered.contains("= 0"));
}

#[test]
fn symbolic_constraints_reuse_inter_defs_across_constraints() {
    let delta =
        SymbolicExpression::from(BabyBear::new(2)) - SymbolicExpression::from(BabyBear::new(1));
    let shared = delta.clone() + (SymbolicExpression::from(BabyBear::new(3)) * delta);
    let expr0 = SymbolicExpression::IsFirstRow * shared.clone();
    let expr1 = SymbolicExpression::IsLastRow * shared;

    let mut context = LeanRenderContext {
        use_counts: expression_direct_use_counts(&[expr0.clone(), expr1.clone()]),
        ..Default::default()
    };
    let (helper_defs0, rendered0) =
        symbolic_constraint_to_lean_definitions(&expr0, 0, "", None, &mut context);
    let (helper_defs1, rendered1) =
        symbolic_constraint_to_lean_definitions(&expr1, 1, "", None, &mut context);

    assert_eq!(
        helper_defs0
            .iter()
            .filter(|def| def.contains("def inter_0"))
            .count(),
        1
    );
    assert!(helper_defs1.is_empty());
    assert!(!rendered0.contains("def inter_0"));
    assert!(rendered1.contains("inter_0 c row"));
}

#[test]
fn symbolic_constraint_dedupes_reused_local_let_bindings() {
    let delta =
        SymbolicExpression::from(BabyBear::new(2)) - SymbolicExpression::from(BabyBear::new(1));
    let expr = delta.clone() + (SymbolicExpression::from(BabyBear::new(3)) * delta);

    let mut context = LeanRenderContext {
        use_counts: expression_direct_use_counts(std::slice::from_ref(&expr)),
        ..Default::default()
    };
    let (helper_defs, rendered) =
        symbolic_constraint_to_lean_definitions(&expr, 0, "", None, &mut context);

    assert!(helper_defs.is_empty());
    assert_eq!(rendered.matches("let t0 :=").count(), 1);
}

#[test]
fn constrain_interactions_uses_intermediates_and_is_well_formed() {
    let delta =
        SymbolicExpression::from(BabyBear::new(2)) - SymbolicExpression::from(BabyBear::new(1));
    let shared = delta.clone() + (SymbolicExpression::from(BabyBear::new(3)) * delta.clone());
    let count = SymbolicExpression::IsFirstRow * shared.clone();
    let field = SymbolicExpression::IsLastRow * shared;
    let symbolic_constraints = SymbolicConstraints {
        constraints: vec![],
        interactions: vec![Interaction {
            message: vec![field],
            count,
            bus_index: 7,
            count_weight: 1,
        }],
    };

    let mut out = Vec::new();
    extract_constraints_to_lean_writer(&symbolic_constraints, "TestAir", &mut out).unwrap();
    let rendered = String::from_utf8(out).unwrap();

    assert!(rendered.contains("def inter_0"));
    assert!(rendered.contains("def constrain_interactions"));
    assert!(rendered.contains("if index = 7 then\n"));
    assert!(rendered.contains("inter_0 c row"));
    assert!(!rendered.contains("let t0 :=\n        let t0 :="));
}

#[test]
fn extracted_file_includes_prologue_and_namespace() {
    let symbolic_constraints = SymbolicConstraints::<BabyBear> {
        constraints: vec![SymbolicExpression::IsFirstRow],
        interactions: vec![],
    };

    let mut out = Vec::new();
    extract_constraints_to_lean_writer(&symbolic_constraints, "Sha2BlockHasherVmAir", &mut out)
        .unwrap();
    let rendered = String::from_utf8(out).unwrap();

    assert!(rendered.contains("import Mathlib.Algebra.Field.Basic"));
    assert!(rendered.contains("import LeanZKCircuit.OpenVM.Circuit"));
    assert!(rendered.contains("set_option linter.all false"));
    assert!(rendered.contains("register_simp_attr Sha2BlockHasherVmAir_air_simplification"));
    assert!(rendered.contains(
        "register_simp_attr Sha2BlockHasherVmAir_constraint_and_interaction_simplification"
    ));
    assert!(rendered.contains("namespace Sha2BlockHasherVmAir.extraction"));
    assert!(rendered.contains("def constraint_0"));
    assert!(!rendered.contains("def allHold"));
    assert!(!rendered.contains("def extracted_row_constraint_list"));
    assert!(!rendered.contains("NAME"));
    assert!(rendered.contains("end Sha2BlockHasherVmAir.extraction"));
}

#[test]
fn formats_generic_air_names_for_lean_identifiers() {
    assert_eq!(
        format_lean_air_name("Sha2BlockHasherVmAir<Sha256Config, Sha512Config>"),
        "Sha2BlockHasherVmAir_Sha256Config_Sha512Config"
    );
    assert_eq!(
        format_lean_air_name("VerifierSubCircuit<4, CachedSymbolicExpressionColumns<u8>>"),
        "VerifierSubCircuit_4_CachedSymbolicExpressionColumns_u8"
    );
}
