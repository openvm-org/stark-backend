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
        partition_offsets: vec![0],
        public_value_names: Vec::new(),
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
    // Per-AIR simp attribute is registered here.
    assert!(s.contains("register_simp_attr TinyAir_inter"));
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
    // One generic `mem_constraintsList` helper used inline by
    // `of_satisfiesRow` for each sub-goal — no per-K helpers, no
    // `List.Mem.tail` chain.
    assert!(s.contains("private theorem mem_constraintsList"));
    assert!(s.contains("List.getElem_mem"));
    assert!(s.contains("· exact h expr_0 (mem_constraintsList 0 (by decide))"));
    assert!(!s.contains("simp [constraintsList]"));
    assert!(!s.contains("List.Mem.tail"));
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
    // Helper is tagged with the per-AIR simp attribute.
    assert!(s.contains("@[TinyAir_inter]\ndef inter_0"));
    // Both constraints reference the helper.
    assert_eq!(s.matches("inter_0 va").count(), 2);
}

#[test]
fn local_only_reuse_stays_local_let() {
    let a = Arc::new(main_var(0, 0, 0));
    let one = Arc::new(SymbolicExpression::Constant(BabyBear::new(1)));
    let a_minus_one = Arc::new(SymbolicExpression::Sub {
        x: a.clone(),
        y: one,
        degree_multiple: 1,
    });
    let shared = Arc::new(SymbolicExpression::Mul {
        x: a,
        y: a_minus_one,
        degree_multiple: 2,
    });
    let c0 = SymbolicExpression::Add {
        x: shared.clone(),
        y: shared,
        degree_multiple: 3,
    };
    let dag = SymbolicConstraintsDag::from(SymbolicConstraints::<BabyBear> {
        constraints: vec![c0],
        interactions: vec![],
    });

    let names = vec!["a".to_string()];
    let opts = options_for("Recursion.Test.TinyAir");
    let rendered = render_air(&dag, &names, &[], &opts).unwrap();

    assert!(rendered.helper_defs.is_empty());
    assert!(rendered.constraint_bodies[0].contains("let t0 :="));
    assert!(rendered.constraint_bodies[0].contains("(t0 + t0)"));
}

#[test]
fn helper_dependencies_are_emitted_before_parent_helpers() {
    // common is used by both parent helpers. In the first constraint it
    // appears twice locally, but it still needs to become an `inter_K`
    // before either parent helper is emitted, otherwise both parents
    // capture it as duplicate local lets and the later helper is unused.
    let a = Arc::new(main_var(0, 0, 0));
    let b = Arc::new(main_var(0, 1, 0));
    let c = Arc::new(main_var(0, 2, 0));
    let d = Arc::new(main_var(0, 3, 0));
    let one = Arc::new(SymbolicExpression::Constant(BabyBear::new(1)));
    let a_minus_one = Arc::new(SymbolicExpression::Sub {
        x: a.clone(),
        y: one,
        degree_multiple: 1,
    });
    let common = Arc::new(SymbolicExpression::Mul {
        x: a,
        y: a_minus_one,
        degree_multiple: 2,
    });
    let parent_0 = Arc::new(SymbolicExpression::Add {
        x: common.clone(),
        y: b,
        degree_multiple: 3,
    });
    let parent_1 = Arc::new(SymbolicExpression::Add {
        x: common,
        y: c,
        degree_multiple: 3,
    });
    let c0 = SymbolicExpression::Add {
        x: parent_0.clone(),
        y: parent_1.clone(),
        degree_multiple: 4,
    };
    let c1 = SymbolicExpression::Mul {
        x: parent_0,
        y: d.clone(),
        degree_multiple: 4,
    };
    let c2 = SymbolicExpression::Mul {
        x: parent_1,
        y: d,
        degree_multiple: 4,
    };
    let dag = SymbolicConstraintsDag::from(SymbolicConstraints::<BabyBear> {
        constraints: vec![c0, c1, c2],
        interactions: vec![],
    });

    let names = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];
    let opts = options_for("Recursion.Test.TinyAir");
    let rendered = render_air(&dag, &names, &[], &opts).unwrap();
    let helpers = rendered.helper_defs.join("\n");
    let parent_0_def =
        "def inter_1 : Expr F layout := fun va =>\n  (inter_0 va + va (.cell .local bRef))";
    let parent_1_def =
        "def inter_2 : Expr F layout := fun va =>\n  (inter_0 va + va (.cell .local cRef))";

    assert!(helpers.contains("def inter_0 : Expr F layout := fun va =>"));
    assert!(helpers.contains("(va (.cell .local aRef) * (va (.cell .local aRef) - 1))"));
    assert!(helpers.contains(parent_0_def));
    assert!(helpers.contains(parent_1_def));
    assert_eq!(helpers.matches("inter_0 va").count(), 2);
    assert!(
        !helpers.contains("let t0 :="),
        "shared helper dependency should not be captured as a local let:\n{helpers}"
    );
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
fn interactions_simp_uses_per_air_inter_attribute_when_helper_referenced() {
    // Helper-sharing across two constraints AND an interaction: a
    // shared compound subtree gets hoisted to `inter_0`. The
    // constraints exist only to drive global use-count above the
    // share threshold; the interaction's count is a `Neg` of the
    // same shared subtree, so its rendered multExpr is `-(inter_0
    // va)` and the per-pick `_evalMultiplicityAt` simp arg list
    // must include the per-AIR simp attribute (so transitive
    // `inter_K → inter_M` references can be unfolded by simp).
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
        y: shared.clone(),
        degree_multiple: 3,
    };
    // Trace-form rendering (used by per-pick `_evalMultiplicityAt`
    // lemmas) panics on selector entries, so message uses a plain
    // column reference, not `IsFirstRow`/`IsLastRow`.
    let interaction = Interaction {
        message: vec![main_var(0, 0, 0)],
        count: SymbolicExpression::Neg {
            x: shared,
            degree_multiple: 2,
        },
        bus_index: 0,
        count_weight: 1,
    };
    let dag = SymbolicConstraintsDag::from(SymbolicConstraints::<BabyBear> {
        constraints: vec![c0, c1],
        interactions: vec![interaction],
    });
    let bus_table = vec![BusBinding {
        vk_index: 0,
        lean_name: "transcriptBus".to_string(),
    }];
    let names = vec!["is_valid".to_string()];
    let opts = options_for("Recursion.Test.TinyAir");
    let rendered = render_air(&dag, &names, &bus_table, &opts).unwrap();

    // Constraints.lean tags the hoisted helper.
    let mut cout = Vec::new();
    write_constraints(&mut cout, "TinyAir", &names, &rendered, &opts).unwrap();
    let cs = String::from_utf8(cout).unwrap();
    assert!(
        cs.contains("@[TinyAir_inter]\ndef inter_0"),
        "expected `@[TinyAir_inter]` above hoisted helper, got:\n{cs}"
    );

    // Interactions.lean uses the attribute name (not the helper name)
    // in the per-pick eval-lemma simp arg list.
    let mut iout = Vec::new();
    write_interactions(&mut iout, "TinyAir", &rendered, &opts).unwrap();
    let is = String::from_utf8(iout).unwrap();
    assert!(is.contains("_evalMultiplicityAt"));
    assert!(
        is.contains("lemma transcriptBus_0_busEventAt"),
        "expected per-pick busEventAt lemma, got:\n{is}"
    );
    assert!(
        is.contains("(transcriptBus_0 (F := F)).toBusEventAt ⟨trace, row, []⟩ ="),
        "busEventAt lemma should normalize the typed BusEvent itself:\n{is}"
    );
    assert!(
        is.contains("msg := #v[\n          is_valid trace row\n        ]"),
        "busEventAt lemma should expose the normalized vector message:\n{is}"
    );
    let mult_simp_line = is
        .lines()
        .find(|l| {
            l.trim_start().starts_with("simp [transcriptBus_0,")
                && l.contains("Interaction.evalMultiplicityAt")
        })
        .expect("multiplicity simp line");
    assert!(
        mult_simp_line.contains("TinyAir_inter"),
        "multiplicity simp line missing per-AIR attribute: {mult_simp_line}"
    );
    // Per-helper enumeration is no longer emitted in simp lists.
    assert!(
        !mult_simp_line.contains(", inter_0,") && !mult_simp_line.contains(", inter_0]"),
        "multiplicity simp line should not enumerate inter_0: {mult_simp_line}"
    );
}

#[test]
fn formats_generic_air_names_for_lean_identifiers() {
    assert_eq!(
        format_lean_air_name("Sha2BlockHasherVmAir<Sha256Config, Sha512Config>"),
        "Sha2BlockHasherVmAir_Sha256Config_Sha512Config"
    );
    assert_eq!(air_file_stem("WhirRoundAir<u8, 1>"), "WhirRoundAir");
}
