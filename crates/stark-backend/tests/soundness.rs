//! ==========================================================================
//! Production configuration soundness tests
//! ==========================================================================

use openvm_stark_backend::{soundness::*, SystemParams};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, internal_params_with_100_bits_security as internal_params,
    leaf_params_with_100_bits_security as leaf_params, MAX_APP_LOG_STACKED_HEIGHT,
};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;

// ==========================================================================
// Circuit parameter upper bounds for soundness analysis
//
// These are conservative estimates based on actual production values.
// Stacking can only reduce width, so num_columns is an upper bound on stacked columns.
// ==========================================================================

/// Upper bound on n_logup derived from circuit parameters.
///
/// n_logup = ceil_log2(total_interactions) - l_skip, where:
/// - total_interactions ≤ num_airs × max_interactions_per_air × 2^max_log_height
/// - total_interactions < max_interaction_count (enforced by verifier as one of the
///   `LinearConstraint`s, keygen ensures this linear constraint is included)
///
/// So: n_logup ≤ min(ceil_log2(max_interaction_count) - l_skip, log2(num_airs ×
/// max_interactions) + max_log_height - l_skip)
fn n_logup_bound(
    l_skip: usize,
    num_airs: usize,
    max_interactions_per_air: usize,
    max_log_height: usize,
    max_interaction_count: usize,
) -> usize {
    let field_bound = (max_interaction_count as f64).log2().ceil() as usize - l_skip;
    let param_bound = (num_airs as f64).log2().ceil() as usize
        + (max_interactions_per_air as f64).log2().ceil() as usize
        + max_log_height
        - l_skip;
    field_bound.min(param_bound)
}

// App VM: large circuits with many AIRs
// Actual: max_constraints=4513 (keccak), num_airs=73, max_interactions=832
const APP_MAX_CONSTRAINTS: usize = 5000;
const APP_NUM_AIRS: usize = 100;
const APP_MAX_LOG_HEIGHT: usize = 24;
const APP_NUM_COLUMNS: usize = 30000;
const APP_MAX_INTERACTIONS_PER_AIR: usize = 1000;

// Recursion circuits: smaller, fixed structure
// Actual: num_airs=42
const RECURSION_MAX_CONSTRAINTS: usize = 1000;
const RECURSION_NUM_AIRS: usize = 50;
const RECURSION_NUM_COLUMNS: usize = 2000;
const RECURSION_MAX_INTERACTIONS_PER_AIR: usize = 100; // estimate, needs verification

const TARGET_SECURITY_BITS: usize = 100;

fn babybear_quartic_extension_bits() -> f64 {
    4.0 * (BabyBear::ORDER_U64 as f64).log2()
}

fn app_params() -> SystemParams {
    app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT)
}

fn check_soundness(
    name: &str,
    params: &SystemParams,
    max_constraints: usize,
    num_airs: usize,
    max_log_height: usize,
    num_columns: usize,
    n_logup: usize,
) -> SoundnessCalculator {
    let soundness = SoundnessCalculator::calculate(
        params,
        babybear_quartic_extension_bits(),
        max_constraints,
        num_airs,
        params.max_constraint_degree,
        max_log_height,
        num_columns,
        params.w_stack,
        n_logup,
    );

    println!("\n=== {} Soundness ===", name);
    println!(
        "Config: l_skip={}, n_stack={}, w_stack={}, log_blowup={}, k_whir={}, whir.rounds={:?}",
        params.l_skip,
        params.n_stack,
        params.w_stack,
        params.log_blowup,
        params.whir.k,
        params.whir.rounds
    );
    println!(
        "Context: max_constraints={}, num_airs={}, max_log_height={}, num_columns={}, n_logup={}",
        max_constraints, num_airs, max_log_height, num_columns, n_logup
    );
    println!();
    println!("LogUp (α/β + PoW):   {:.1} bits", soundness.logup_bits);
    println!(
        "GKR sumcheck:        {:.1} bits",
        soundness.gkr_sumcheck_bits
    );
    println!(
        "GKR batching (μ/λ):  {:.1} bits",
        soundness.gkr_batching_bits
    );
    println!(
        "ZeroCheck sumcheck:  {:.1} bits",
        soundness.zerocheck_sumcheck_bits
    );
    println!(
        "Constraint batching: {:.1} bits",
        soundness.constraint_batching_bits
    );
    println!(
        "Stacked reduction:   {:.1} bits",
        soundness.stacked_reduction_bits
    );
    println!("WHIR:                {:.1} bits", soundness.whir_bits);
    println!("TOTAL:               {:.1} bits", soundness.total_bits);

    println!("\nWHIR Error Source Breakdown:");
    let whir = &soundness.whir_details;
    println!("  Query error:          {:.1} bits", whir.query_bits);
    println!(
        "  Proximity gaps:       {:.1} bits",
        whir.proximity_gaps_bits
    );
    println!("  Sumcheck error:       {:.1} bits", whir.sumcheck_bits);
    println!("  Min ε_fold:           {:.1} bits", whir.fold_rbr_bits);
    println!("  OOD error:            {:.1} bits", whir.ood_rbr_bits);
    println!(
        "  γ batching error:     {:.1} bits",
        whir.gamma_batching_bits
    );
    println!("  Min ε_shift/ε_fin:    {:.1} bits", whir.shift_rbr_bits);
    println!("  μ batching error:     {:.1} bits", whir.mu_batching_bits);

    soundness
}

#[test]
fn test_app_vm_security() {
    let params = app_params();
    let n_logup = n_logup_bound(
        params.l_skip,
        APP_NUM_AIRS,
        APP_MAX_INTERACTIONS_PER_AIR,
        APP_MAX_LOG_HEIGHT,
        params.logup.max_interaction_count as usize,
    );
    let soundness = check_soundness(
        "App VM",
        &params,
        APP_MAX_CONSTRAINTS,
        APP_NUM_AIRS,
        APP_MAX_LOG_HEIGHT,
        APP_NUM_COLUMNS,
        n_logup,
    );
    assert!(
        soundness.total_bits >= TARGET_SECURITY_BITS as f64,
        "App VM: got {:.1} bits",
        soundness.total_bits
    );
}

#[test]
fn test_leaf_aggregation_security() {
    let params = leaf_params();
    let max_log_height = 20;
    let n_logup = n_logup_bound(
        params.l_skip,
        RECURSION_NUM_AIRS,
        RECURSION_MAX_INTERACTIONS_PER_AIR,
        max_log_height,
        params.logup.max_interaction_count as usize,
    );
    let soundness = check_soundness(
        "Leaf Aggregation",
        &params,
        RECURSION_MAX_CONSTRAINTS,
        RECURSION_NUM_AIRS,
        max_log_height,
        RECURSION_NUM_COLUMNS,
        n_logup,
    );
    assert!(
        soundness.total_bits >= TARGET_SECURITY_BITS as f64,
        "Leaf: got {:.1} bits",
        soundness.total_bits
    );
}

#[test]
fn test_internal_aggregation_security() {
    let params = internal_params();
    let max_log_height = 19;
    let n_logup = n_logup_bound(
        params.l_skip,
        RECURSION_NUM_AIRS,
        RECURSION_MAX_INTERACTIONS_PER_AIR,
        max_log_height,
        params.logup.max_interaction_count as usize,
    );
    let soundness = check_soundness(
        "Internal Aggregation",
        &params,
        RECURSION_MAX_CONSTRAINTS,
        RECURSION_NUM_AIRS,
        max_log_height,
        RECURSION_NUM_COLUMNS,
        n_logup,
    );
    assert!(
        soundness.total_bits >= TARGET_SECURITY_BITS as f64,
        "Internal: got {:.1} bits",
        soundness.total_bits
    );
}

#[test]
fn test_all_production_configs() {
    println!("\n========== ALL PRODUCTION CONFIGS ==========");

    let app = app_params();
    let leaf = leaf_params();
    let internal = internal_params();

    // (name, params, max_constraints, num_airs, max_log_height, num_columns,
    // max_interactions_per_air)
    let configs: [(&str, &SystemParams, usize, usize, usize, usize, usize); _] = [
        (
            "App VM",
            &app,
            APP_MAX_CONSTRAINTS,
            APP_NUM_AIRS,
            APP_MAX_LOG_HEIGHT,
            APP_NUM_COLUMNS,
            APP_MAX_INTERACTIONS_PER_AIR,
        ),
        (
            "Leaf",
            &leaf,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            20,
            RECURSION_NUM_COLUMNS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
        ),
        (
            "Internal",
            &internal,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            19,
            RECURSION_NUM_COLUMNS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
        ),
    ];

    for (name, params, max_constraints, num_airs, max_log_height, num_columns, max_interactions) in
        configs
    {
        let n_logup = n_logup_bound(
            params.l_skip,
            num_airs,
            max_interactions,
            max_log_height,
            params.logup.max_interaction_count as usize,
        );
        let soundness = check_soundness(
            name,
            params,
            max_constraints,
            num_airs,
            max_log_height,
            num_columns,
            n_logup,
        );
        assert!(
            soundness.total_bits >= TARGET_SECURITY_BITS as f64,
            "{}: got {:.1} bits",
            name,
            soundness.total_bits
        );
    }

    println!("\n========== ALL CONFIGS PASSED ==========");
}
