use openvm_stark_backend::{
    interaction::LogUpSecurityParameters, soundness, soundness::SoundnessCalculator, SystemParams,
    WhirProximityStrategy,
};

use crate::config::{
    baby_bear_poseidon2::BabyBearPoseidon2Config,
    log_up_params::log_up_security_params_baby_bear_100_bits,
};

/// STARK config where the base field is BabyBear, extension field is BabyBear^4, and the hasher is
/// `Poseidon2<Bn254>`.
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub mod baby_bear_bn254_poseidon2;
/// STARK config where the base field is BabyBear, extension field is BabyBear^4, and the hasher is
/// `Poseidon2<BabyBear>`.
pub mod baby_bear_poseidon2;
/// BN254 Poseidon2 permutations used by the BabyBear + BN254 configuration.
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub mod bn254_poseidon2;
pub mod log_up_params;

// ==========================================================================
// Production configurations
// ==========================================================================
// These configurations target 100-bits of proven round-by-round (RBR) security with BabyBear as the
// base field and BabyBear^4 as the extension field.

pub const WHIR_MAX_LOG_FINAL_POLY_LEN: usize = 10;
pub const SECURITY_BITS_TARGET: usize = 100;

pub const DEFAULT_APP_L_SKIP: usize = 4;
pub const DEFAULT_APP_LOG_BLOWUP: usize = 1;
pub const DEFAULT_LEAF_LOG_BLOWUP: usize = 2;
pub const DEFAULT_INTERNAL_LOG_BLOWUP: usize = 3;
pub const DEFAULT_ROOT_LOG_BLOWUP: usize = 4;
pub const DEFAULT_HOOK_LOG_BLOWUP: usize = 2;

pub const APP_MAX_CONSTRAINT_DEGREE: usize = 3;
pub const RECURSION_MAX_CONSTRAINT_DEGREE: usize = 4;

pub const MAX_APP_LOG_STACKED_HEIGHT: usize = 24;

/// Order `p` of the base field for the production [`BabyBearPoseidon2Config`].
pub fn base_field_order() -> f64 {
    soundness::base_field_order::<BabyBearPoseidon2Config>()
}

/// Number of bits in the challenge field for the production [`BabyBearPoseidon2Config`].
pub fn challenge_field_bits() -> f64 {
    soundness::challenge_field_bits::<BabyBearPoseidon2Config>()
}

/// LogUp grinding sufficient for 100-bit security, accounting for the PCS list-size union bound of
/// the initial WHIR proximity regime (`log2_pcs_list_size` is 0 for unique decoding).
pub fn log_up_params_for_whir(
    proximity: WhirProximityStrategy,
    log_stacked_height: usize,
    log_blowup: usize,
    w_stack: usize,
) -> LogUpSecurityParameters {
    let log2_pcs_list_size = SoundnessCalculator::whir_proximity_gap_security(
        proximity.initial_round(),
        challenge_field_bits(),
        log_stacked_height,
        log_blowup,
        w_stack,
    )
    .log2_list_size;
    log_up_security_params_baby_bear_100_bits(log2_pcs_list_size)
}

/// Builds production `SystemParams` for 100-bit security, calibrating LogUp grinding to the WHIR
/// proximity regime's PCS list size. The LogUp params are derived up front (from the same inputs
/// `SystemParams::new` uses to build the WHIR config) so they can be passed straight in.
#[allow(clippy::too_many_arguments)]
pub fn params_with_100_bits_security(
    log_blowup: usize,
    l_skip: usize,
    n_stack: usize,
    w_stack: usize,
    folding_pow_bits: usize,
    mu_pow_bits: usize,
    proximity: WhirProximityStrategy,
    max_constraint_degree: usize,
) -> SystemParams {
    let logup = log_up_params_for_whir(proximity, l_skip + n_stack, log_blowup, w_stack);
    SystemParams::new(
        log_blowup,
        l_skip,
        n_stack,
        w_stack,
        WHIR_MAX_LOG_FINAL_POLY_LEN,
        folding_pow_bits,
        mu_pow_bits,
        proximity,
        SECURITY_BITS_TARGET,
        logup,
        max_constraint_degree,
    )
}

/// Returns `SystemParams` targeting 100 bits of proven RBR security for App VM circuits.
///
/// # Assumptions for 100-bit security
/// - **Max trace height**: `log_stacked_height` ≤ [`MAX_APP_LOG_STACKED_HEIGHT`] (24)
/// - **Max constraints per AIR**: ≤ 5,000
/// - **Num AIRs**: ≤ 100
/// - **Max interactions per AIR**: ≤ 1,000
/// - **Num trace columns** (unstacked, total across all AIRs): ≤ 30,000
/// - **`w_stack`** = 2,048, bounding total stacked cells to `w_stack × 2^(n_stack + l_skip)`
//
// See `test_all_production_configs` in `crates/stark-backend/tests/soundness.rs` for the
// full soundness analysis.
pub fn app_params_with_100_bits_security(log_stacked_height: usize) -> SystemParams {
    assert!(
        log_stacked_height <= MAX_APP_LOG_STACKED_HEIGHT,
        "log_stacked_height must be <= {MAX_APP_LOG_STACKED_HEIGHT}",
    );
    params_with_100_bits_security(
        DEFAULT_APP_LOG_BLOWUP,
        DEFAULT_APP_L_SKIP,
        log_stacked_height.saturating_sub(DEFAULT_APP_L_SKIP), // n_stack
        2048,                                                  // w_stack
        5,                                                     // folding pow
        15,                                                    // mu pow
        WhirProximityStrategy::UniqueDecoding,
        APP_MAX_CONSTRAINT_DEGREE,
    )
}

/// Returns `SystemParams` targeting 100 bits of proven RBR security for leaf aggregation circuits.
///
/// # Assumptions for 100-bit security
/// - **Max trace height**: ≤ 2^21
/// - **Max constraints per AIR**: ≤ 1,000
/// - **Num AIRs**: ≤ 50
/// - **Max interactions per AIR**: ≤ 100 (maximum number of interactions in a single AIR for a
///   single row)
/// - **Num trace columns** (unstacked, total across all AIRs): ≤ 2,000
/// - **`w_stack`** = 2,048, bounding total stacked cells to `w_stack × 2^(n_stack + l_skip)`
///
/// Config: `l_skip=4, n_stack=17, log_blowup=2`.
//
// See `test_all_production_configs` in `crates/stark-backend/tests/soundness.rs` for the
// full soundness analysis.
pub fn leaf_params_with_100_bits_security() -> SystemParams {
    params_with_100_bits_security(
        DEFAULT_LEAF_LOG_BLOWUP,
        4,    // l_skip
        17,   // n_stack
        2048, // w_stack
        4,    // folding pow
        13,   // mu pow
        WhirProximityStrategy::UniqueDecoding,
        RECURSION_MAX_CONSTRAINT_DEGREE,
    )
}

/// Returns `SystemParams` targeting 100 bits of proven RBR security for internal aggregation
/// circuits.
///
/// # Assumptions for 100-bit security
/// - **Max trace height**: ≤ 2^19
/// - **Max constraints per AIR**: ≤ 1,000
/// - **Num AIRs**: ≤ 50
/// - **Max interactions per AIR**: ≤ 100
/// - **Num trace columns** (unstacked, total across all AIRs): ≤ 2,000
/// - **`w_stack`** = 512, bounding total stacked cells to `w_stack × 2^(n_stack + l_skip)`
///
/// Config: `l_skip=2, n_stack=17, log_blowup=3`.
//
// See `test_all_production_configs` in `crates/stark-backend/tests/soundness.rs` for the
// full soundness analysis.
pub fn internal_params_with_100_bits_security() -> SystemParams {
    params_with_100_bits_security(
        DEFAULT_INTERNAL_LOG_BLOWUP,
        2,   // l_skip
        17,  // n_stack
        512, // w_stack
        18,  // folding pow
        20,  // mu pow
        WhirProximityStrategy::ListDecoding { m: 2 },
        RECURSION_MAX_CONSTRAINT_DEGREE,
    )
}

/// Returns `SystemParams` targeting 100 bits of proven RBR security for root circuits.
///
/// # Assumptions for 100-bit security
/// - **Max trace height**: ≤ 2^20
/// - **Max constraints per AIR**: ≤ 1,000
/// - **Num AIRs**: ≤ 50
/// - **Max interactions per AIR**: ≤ 100
/// - **Num trace columns** (unstacked, total across all AIRs): ≤ 2,000
/// - **`w_stack`** = 18, bounding total stacked cells to `w_stack × 2^(n_stack + l_skip)`
///
/// Config: `l_skip=2, n_stack=18, log_blowup=4`.
//
// See `test_all_production_configs` in `crates/stark-backend/tests/soundness.rs` for the
// full soundness analysis.
pub fn root_params_with_100_bits_security() -> SystemParams {
    params_with_100_bits_security(
        DEFAULT_ROOT_LOG_BLOWUP,
        2,  // l_skip
        18, // n_stack
        18, // w_stack
        20, // folding pow
        20, // mu pow
        WhirProximityStrategy::ListDecoding { m: 1 },
        RECURSION_MAX_CONSTRAINT_DEGREE,
    )
}

/// Returns `SystemParams` targeting 100 bits of proven RBR security for deferral hook circuits.
///
/// # Assumptions for 100-bit security
/// - **Max trace height**: ≤ 2^20
/// - **Max constraints per AIR**: ≤ 1,000
/// - **Num AIRs**: ≤ 50
/// - **Max interactions per AIR**: ≤ 100
/// - **Num trace columns** (unstacked, total across all AIRs): ≤ 2,000
/// - **`w_stack`** = 80, bounding total stacked cells to `w_stack × 2^(n_stack + l_skip)`
///
/// Config: `l_skip=2, n_stack=18, log_blowup=2`.
//
// See `test_all_production_configs` in `crates/stark-backend/tests/soundness.rs` for the
// full soundness analysis.
pub fn hook_params_with_100_bits_security() -> SystemParams {
    params_with_100_bits_security(
        DEFAULT_HOOK_LOG_BLOWUP,
        2,  // l_skip
        18, // n_stack
        80, // w_stack
        12, // folding pow
        11, // mu pow
        WhirProximityStrategy::ListDecoding { m: 1 },
        RECURSION_MAX_CONSTRAINT_DEGREE,
    )
}
