//! ==========================================================================
//! Production configuration soundness tests
//! ==========================================================================

use openvm_stark_backend::{soundness::*, SystemParams};
use openvm_stark_sdk::config::{
    app_params_with_128_bits_field_security,
    baby_bear_poseidon2::{BabyBearPoseidon2Config, DuplexSponge},
    challenge_field_bits, hook_params_with_128_bits_field_security as hook_params,
    internal_params_with_128_bits_field_security as internal_params,
    leaf_params_with_128_bits_field_security as leaf_params,
    root_params_with_128_bits_field_security as root_params, MAX_APP_LOG_STACKED_HEIGHT,
};

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
const ROOT_NUM_COLUMNS: usize = 2400;
const RECURSION_MAX_INTERACTIONS_PER_AIR: usize = 100; // estimate, needs verification

const TARGET_FIELD_PROTOCOL_SECURITY_BITS: f64 = 128.0;
const MIN_POSEIDON2_END_TO_END_BITS: f64 = 123.0;
const MAX_POSEIDON2_END_TO_END_BITS: f64 = 124.0;

fn app_params() -> SystemParams {
    app_params_with_128_bits_field_security(MAX_APP_LOG_STACKED_HEIGHT)
}

fn check_soundness(
    name: &str,
    params: &SystemParams,
    max_constraints: usize,
    num_airs: usize,
    max_log_height: usize,
    num_columns: usize,
    n_logup: usize,
) -> ConfigSoundnessAssessment {
    let soundness =
        SoundnessCalculator::calculate_for_config::<BabyBearPoseidon2Config, DuplexSponge>(
            params,
            max_constraints,
            num_airs,
            params.max_constraint_degree,
            max_log_height,
            num_columns,
            params.w_stack,
            n_logup,
        );
    let field_protocol = &soundness.field_protocol;
    let profile = &soundness.config_security_profile;

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
    println!("Challenge field:      {:.1} bits", challenge_field_bits());
    println!();
    println!("LogUp (α/β + PoW):   {:.1} bits", field_protocol.logup_bits);
    println!(
        "GKR sumcheck:        {:.1} bits",
        field_protocol.gkr_sumcheck_bits
    );
    println!(
        "GKR batching (μ/λ):  {:.1} bits",
        field_protocol.gkr_batching_bits
    );
    println!(
        "ZeroCheck sumcheck:  {:.1} bits",
        field_protocol.zerocheck_sumcheck_bits
    );
    println!(
        "Fused boundary/batching: {:.1} bits",
        field_protocol.constraint_batching_bits
    );
    println!(
        "Stacked reduction:   {:.1} bits",
        field_protocol.stacked_reduction_bits
    );
    println!("WHIR:                {:.1} bits", field_protocol.whir_bits);
    println!(
        "FIELD/PROTOCOL SECURITY: {:.1} bits",
        field_protocol.field_protocol_security_bits
    );
    println!(
        "CONFIG COMMITMENT: collision={:.1}, preimage={:.1} bits",
        profile.commitment.collision_bits, profile.commitment.preimage_bits
    );
    println!(
        "CONFIG TRANSCRIPT: collision={:.1}, preimage={:.1} bits",
        profile.transcript.collision_bits, profile.transcript.preimage_bits
    );
    println!("CONFIG SAMPLING:     {:?}", profile.sampling);
    println!(
        "CONFIG HASH-SIDE SECURITY: {:.1} bits",
        soundness.config_hash_security_bits()
    );
    println!(
        "END-TO-END SECURITY: {:.1} bits",
        soundness.end_to_end_security_bits
    );

    println!("\nWHIR Error Source Breakdown:");
    let whir = &field_protocol.whir_details;
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

fn assert_poseidon2_security(name: &str, soundness: &ConfigSoundnessAssessment) {
    let field_protocol_bits = soundness.field_protocol.field_protocol_security_bits;
    let config_hash_bits = soundness.config_hash_security_bits();
    let end_to_end_bits = soundness.end_to_end_security_bits;

    assert!(
        field_protocol_bits >= TARGET_FIELD_PROTOCOL_SECURITY_BITS,
        "{name}: field/protocol estimate is only {field_protocol_bits:.1} bits"
    );
    assert!(
        (MIN_POSEIDON2_END_TO_END_BITS..MAX_POSEIDON2_END_TO_END_BITS).contains(&config_hash_bits),
        "{name}: unexpected Poseidon2 hash-side estimate {config_hash_bits:.1} bits"
    );
    assert!(
        end_to_end_bits < TARGET_FIELD_PROTOCOL_SECURITY_BITS,
        "{name}: Poseidon2 must remain below 128 bits end-to-end, got {end_to_end_bits:.1}"
    );
    assert!(
        (end_to_end_bits - field_protocol_bits.min(config_hash_bits)).abs() < f64::EPSILON,
        "{name}: end-to-end estimate is not min(field/protocol, config hash-side)"
    );
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
    assert_poseidon2_security("App VM", &soundness);
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
    assert_poseidon2_security("Leaf", &soundness);
}

#[test]
fn test_internal_aggregation_security() {
    let params = internal_params();
    let max_log_height = 21;
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
    assert_poseidon2_security("Internal", &soundness);
}

#[test]
fn test_root_aggregation_security() {
    let params = root_params();
    let max_log_height = 20;
    let n_logup = n_logup_bound(
        params.l_skip,
        RECURSION_NUM_AIRS,
        RECURSION_MAX_INTERACTIONS_PER_AIR,
        max_log_height,
        params.logup.max_interaction_count as usize,
    );
    let soundness = check_soundness(
        "Root Aggregation",
        &params,
        RECURSION_MAX_CONSTRAINTS,
        RECURSION_NUM_AIRS,
        max_log_height,
        ROOT_NUM_COLUMNS,
        n_logup,
    );
    assert_poseidon2_security("Root", &soundness);
}

#[test]
fn test_hook_security() {
    let params = hook_params();
    let max_log_height = 20;
    let n_logup = n_logup_bound(
        params.l_skip,
        RECURSION_NUM_AIRS,
        RECURSION_MAX_INTERACTIONS_PER_AIR,
        max_log_height,
        params.logup.max_interaction_count as usize,
    );
    let soundness = check_soundness(
        "Hook",
        &params,
        RECURSION_MAX_CONSTRAINTS,
        RECURSION_NUM_AIRS,
        max_log_height,
        RECURSION_NUM_COLUMNS,
        n_logup,
    );
    assert_poseidon2_security("Hook", &soundness);
}

#[test]
fn test_all_production_configs() {
    println!("\n========== ALL PRODUCTION CONFIGS ==========");

    let app = app_params();
    let leaf = leaf_params();
    let internal = internal_params();
    let root = root_params();
    let hook = hook_params();

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
            21,
            RECURSION_NUM_COLUMNS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
        ),
        (
            "Root",
            &root,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            20,
            ROOT_NUM_COLUMNS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
        ),
        (
            "Hook",
            &hook,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            20,
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
        assert_poseidon2_security(name, &soundness);
    }

    println!("\n========== ALL CONFIGS PASSED ==========");
}
