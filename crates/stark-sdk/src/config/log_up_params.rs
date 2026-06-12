use openvm_stark_backend::{
    interaction::LogUpSecurityParameters, p3_field::PrimeField32, soundness::SoundnessCalculator,
};
use p3_baby_bear::BabyBear;

use crate::config::challenge_field_bits;

const TARGET_LOGUP_SECURITY_BITS: f64 = 100.0;
const MIN_BABY_BEAR_LOGUP_POW_BITS: usize = 18;

/// Returns BabyBear LogUp parameters with at least 100 bits after the PCS list-size union bound.
///
/// `log2_pcs_list_size` is `log2(L_PCS)` for the initial WHIR proximity regime. It is zero for
/// unique decoding.
pub fn log_up_security_params_baby_bear_100_bits(
    log2_pcs_list_size: f64,
) -> LogUpSecurityParameters {
    assert!(
        log2_pcs_list_size.is_finite() && log2_pcs_list_size >= 0.0,
        "log2_pcs_list_size must be finite and nonnegative"
    );

    let challenge_field_bits = challenge_field_bits();
    let max_interaction_count = BabyBear::ORDER_U32;
    let log_max_message_length = 7;

    // Pre-grinding LogUp security via the backend (the single source of truth). Security is linear
    // in `pow_bits` with unit slope, so the grinding needed to reach the target is exactly the
    // remaining gap. Floor at `MIN_BABY_BEAR_LOGUP_POW_BITS` to keep the historical baseline margin
    // for the unique-decoding configs. The authoritative check that the resulting params clear the
    // target is `test_all_production_configs` in the backend's soundness tests.
    let security_without_pow = SoundnessCalculator::logup_soundness(
        max_interaction_count,
        log_max_message_length,
        challenge_field_bits,
        log2_pcs_list_size,
    );
    let pow_bits = ((TARGET_LOGUP_SECURITY_BITS - security_without_pow)
        .ceil()
        .max(0.0) as usize)
        .max(MIN_BABY_BEAR_LOGUP_POW_BITS);

    LogUpSecurityParameters {
        max_interaction_count,
        log_max_message_length,
        pow_bits,
    }
}
