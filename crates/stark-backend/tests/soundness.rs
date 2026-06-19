//! ==========================================================================
//! Production configuration soundness tests
//! ==========================================================================

use openvm_stark_backend::{
    soundness::*, SystemParams, WhirConfig, WhirParams, WhirProximityStrategy,
};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, base_field_order, challenge_field_bits,
    hook_params_with_100_bits_security as hook_params,
    internal_params_with_100_bits_security as internal_params,
    leaf_params_with_100_bits_security as leaf_params, log_up_params_for_whir,
    params_with_100_bits_security, root_params_with_100_bits_security as root_params,
    APP_MAX_CONSTRAINT_DEGREE, MAX_APP_LOG_STACKED_HEIGHT, RECURSION_MAX_CONSTRAINT_DEGREE,
    SECURITY_BITS_TARGET, WHIR_MAX_LOG_FINAL_POLY_LEN,
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
const RECURSION_MAX_INTERACTIONS_PER_AIR: usize = 100; // estimate, needs verification

const TARGET_SECURITY_BITS: usize = 100;

fn app_params() -> SystemParams {
    app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT)
}

fn limiting_soundness_component(soundness: &SoundnessCalculator) -> &'static str {
    [
        ("logup", soundness.logup_bits),
        ("gkr_sumcheck", soundness.gkr_sumcheck_bits),
        ("gkr_batching", soundness.gkr_batching_bits),
        ("zerocheck_sumcheck", soundness.zerocheck_sumcheck_bits),
        ("constraint_batching", soundness.constraint_batching_bits),
        ("stacked_reduction", soundness.stacked_reduction_bits),
        ("whir", soundness.whir_bits),
    ]
    .into_iter()
    .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
    .map(|(component, _)| component)
    .expect("soundness has at least one component")
}

fn interesting_configs() -> Vec<SystemParams> {
    vec![
        // Production anchors outside the internal-params sweep shape.
        app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT),
        leaf_params(),
        root_params(),
        hook_params(),
        // Low blowup, maximum app stacked height, and minimal skip.
        params_with_100_bits_security(
            1,
            1,
            23,
            2048,
            8,
            20,
            WhirProximityStrategy::UniqueDecoding,
            APP_MAX_CONSTRAINT_DEGREE,
        ),
        // Large skip factor, where ZeroCheck and stacked reduction degree terms are stressed.
        params_with_100_bits_security(
            2,
            18,
            2,
            256,
            12,
            20,
            WhirProximityStrategy::UniqueDecoding,
            RECURSION_MAX_CONSTRAINT_DEGREE,
        ),
        // Transition from unique decoding into list decoding after the first WHIR round.
        params_with_100_bits_security(
            3,
            4,
            15,
            512,
            20,
            22,
            WhirProximityStrategy::SplitUniqueList {
                m: 2,
                list_start_round: 1,
            },
            RECURSION_MAX_CONSTRAINT_DEGREE,
        ),
        // Small list-decoding multiplicity with a high blowup and very small stacked width.
        params_with_100_bits_security(
            4,
            3,
            16,
            32,
            22,
            22,
            WhirProximityStrategy::ListDecoding { m: 1 },
            RECURSION_MAX_CONSTRAINT_DEGREE,
        ),
    ]
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
        base_field_order(),
        challenge_field_bits(),
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
        "Fused boundary/batching: {:.1} bits",
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
fn generate_root_params() {
    let max_log_height = 20;
    let log_blowups = (2..=5);
    let k_whirs = vec![3, 4];
    let l_skips = (1..=8);
    let mut good_params = vec![];
    for k_whir in k_whirs.clone() {
        for log_blowup in log_blowups.clone() {
            for l_skip in l_skips.clone() {
                println!("k_whir {k_whir}, log_blowup {log_blowup}, l_skip {l_skip}");
                let n_stack = max_log_height - l_skip;
                let w_stack = 18;
                let folding_pow_bits = 20;
                let mu_pow_bits = 20;
                let proximity = WhirProximityStrategy::ListDecoding { m: 1 };
                let log_final_poly_len = WHIR_MAX_LOG_FINAL_POLY_LEN;
                let security_bits = SECURITY_BITS_TARGET;
                let max_constraint_degree = RECURSION_MAX_CONSTRAINT_DEGREE;

                let log_stacked_height = l_skip + n_stack;
                const WHIR_QUERY_PHASE_POW_BITS: usize = 20;
                let logup =
                    log_up_params_for_whir(proximity, l_skip + n_stack, log_blowup, w_stack);

                let params = SystemParams {
                    l_skip,
                    n_stack,
                    w_stack,
                    log_blowup,
                    whir: WhirConfig::new(
                        log_blowup,
                        log_stacked_height,
                        WhirParams {
                            k: k_whir,
                            log_final_poly_len,
                            query_phase_pow_bits: WHIR_QUERY_PHASE_POW_BITS,
                            proximity,
                            folding_pow_bits,
                            mu_pow_bits,
                        },
                        security_bits,
                    ),
                    logup,
                    max_constraint_degree,
                };

                let n_logup = n_logup_bound(
                    params.l_skip,
                    RECURSION_NUM_AIRS,
                    RECURSION_MAX_INTERACTIONS_PER_AIR,
                    max_log_height,
                    params.logup.max_interaction_count as usize,
                );

                let soundness = SoundnessCalculator::calculate(
                    &params,
                    base_field_order(),
                    challenge_field_bits(),
                    RECURSION_MAX_CONSTRAINTS,
                    RECURSION_NUM_AIRS,
                    params.max_constraint_degree,
                    max_log_height,
                    RECURSION_NUM_COLUMNS,
                    params.w_stack,
                    n_logup,
                );
                if soundness.total_bits >= 100.0 {
                    let limiting_component = limiting_soundness_component(&soundness);
                    println!(
                        "k_whir {k_whir}, log_blogup {log_blowup}, l_skip {l_skip}, n_stack {n_stack}, limiting_component: {limiting_component}, soundness: {}",
                        soundness.total_bits
                    );
                    good_params.push(params);
                }
            }
        }
    }

    let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("root_params.json");
    let file = std::fs::File::create(&output_path).expect("failed to create good_params.json");
    serde_json::to_writer_pretty(file, &good_params).expect("failed to write good_params.json");
    println!(
        "wrote {} good params to {}",
        good_params.len(),
        output_path.display()
    );
}

#[test]
fn generate_internal_params() {
    let max_log_height = 19;
    let log_blowups = (1..=5);
    let k_whirs = vec![3, 4];
    let l_skips = (1..=8);
    let mut good_params = vec![];
    for k_whir in k_whirs.clone() {
        for log_blowup in log_blowups.clone() {
            for l_skip in l_skips.clone() {
                let n_stack = max_log_height - l_skip;
                let w_stack = 512;
                let folding_pow_bits = 18;
                let mu_pow_bits = 20;
                let proximity = WhirProximityStrategy::ListDecoding { m: 2 };
                let log_final_poly_len = WHIR_MAX_LOG_FINAL_POLY_LEN;
                let security_bits = SECURITY_BITS_TARGET;
                let max_constraint_degree = RECURSION_MAX_CONSTRAINT_DEGREE;

                let log_stacked_height = l_skip + n_stack;
                const WHIR_QUERY_PHASE_POW_BITS: usize = 20;
                let logup =
                    log_up_params_for_whir(proximity, l_skip + n_stack, log_blowup, w_stack);

                let params = SystemParams {
                    l_skip,
                    n_stack,
                    w_stack,
                    log_blowup,
                    whir: WhirConfig::new(
                        log_blowup,
                        log_stacked_height,
                        WhirParams {
                            k: k_whir,
                            log_final_poly_len,
                            query_phase_pow_bits: WHIR_QUERY_PHASE_POW_BITS,
                            proximity,
                            folding_pow_bits,
                            mu_pow_bits,
                        },
                        security_bits,
                    ),
                    logup,
                    max_constraint_degree,
                };

                let n_logup = n_logup_bound(
                    params.l_skip,
                    RECURSION_NUM_AIRS,
                    RECURSION_MAX_INTERACTIONS_PER_AIR,
                    max_log_height,
                    params.logup.max_interaction_count as usize,
                );

                let soundness = SoundnessCalculator::calculate(
                    &params,
                    base_field_order(),
                    challenge_field_bits(),
                    RECURSION_MAX_CONSTRAINTS,
                    RECURSION_NUM_AIRS,
                    params.max_constraint_degree,
                    max_log_height,
                    RECURSION_NUM_COLUMNS,
                    params.w_stack,
                    n_logup,
                );
                if soundness.total_bits >= 100.0 {
                    let limiting_component = limiting_soundness_component(&soundness);
                    println!(
                        "k_whir {k_whir}, log_blogup {log_blowup}, l_skip {l_skip}, n_stack {n_stack}, limiting_component: {limiting_component}, soundness: {}",
                        soundness.total_bits
                    );
                    good_params.push(params);
                }
            }
        }
    }

    let output_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("internal_params.json");
    let file = std::fs::File::create(&output_path).expect("failed to create good_params.json");
    serde_json::to_writer_pretty(file, &good_params).expect("failed to write good_params.json");
    println!(
        "wrote {} good params to {}",
        good_params.len(),
        output_path.display()
    );
}

#[test]
fn test_interesting_configs_security_and_dump() {
    let configs = interesting_configs();
    assert!(!configs.is_empty(), "expected interesting configs");

    for (index, params) in configs.iter().enumerate() {
        let max_log_height = params.log_stacked_height();
        let (max_constraints, num_airs, num_columns, max_interactions_per_air) =
            if params.max_constraint_degree == APP_MAX_CONSTRAINT_DEGREE {
                (
                    APP_MAX_CONSTRAINTS,
                    APP_NUM_AIRS,
                    APP_NUM_COLUMNS,
                    APP_MAX_INTERACTIONS_PER_AIR,
                )
            } else {
                (
                    RECURSION_MAX_CONSTRAINTS,
                    RECURSION_NUM_AIRS,
                    RECURSION_NUM_COLUMNS,
                    RECURSION_MAX_INTERACTIONS_PER_AIR,
                )
            };
        let n_logup = n_logup_bound(
            params.l_skip,
            num_airs,
            max_interactions_per_air,
            max_log_height,
            params.logup.max_interaction_count as usize,
        );
        let soundness = SoundnessCalculator::calculate(
            params,
            base_field_order(),
            challenge_field_bits(),
            max_constraints,
            num_airs,
            params.max_constraint_degree,
            max_log_height,
            num_columns,
            params.w_stack,
            n_logup,
        );
        let limiting_component = limiting_soundness_component(&soundness);
        println!(
            "interesting_config #{index}: l_skip={}, n_stack={}, w_stack={}, log_blowup={}, k_whir={}, limiting_component={limiting_component}, soundness={}",
            params.l_skip,
            params.n_stack,
            params.w_stack,
            params.log_blowup,
            params.whir.k,
            soundness.total_bits
        );
        assert!(
            soundness.total_bits >= TARGET_SECURITY_BITS as f64,
            "interesting config #{index}: limiting_component={limiting_component}, got {:.1} bits",
            soundness.total_bits
        );
    }

    let output_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("interesting_configs.json");
    let file =
        std::fs::File::create(&output_path).expect("failed to create interesting_configs.json");
    serde_json::to_writer_pretty(file, &configs).expect("failed to write interesting_configs.json");
    println!(
        "wrote {} interesting configs to {}",
        configs.len(),
        output_path.display()
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
fn test_root_aggregation_security() {
    let params = root_params();
    let max_log_height = 21;
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
        RECURSION_NUM_COLUMNS,
        n_logup,
    );
    assert!(
        soundness.total_bits >= TARGET_SECURITY_BITS as f64,
        "Root: got {:.1} bits",
        soundness.total_bits
    );
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
    assert!(
        soundness.total_bits >= TARGET_SECURITY_BITS as f64,
        "Hook: got {:.1} bits",
        soundness.total_bits
    );
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
            19,
            RECURSION_NUM_COLUMNS,
            RECURSION_MAX_INTERACTIONS_PER_AIR,
        ),
        (
            "Root",
            &root,
            RECURSION_MAX_CONSTRAINTS,
            RECURSION_NUM_AIRS,
            20,
            RECURSION_NUM_COLUMNS,
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
        assert!(
            soundness.total_bits >= TARGET_SECURITY_BITS as f64,
            "{}: got {:.1} bits",
            name,
            soundness.total_bits
        );
    }

    println!("\n========== ALL CONFIGS PASSED ==========");
}
