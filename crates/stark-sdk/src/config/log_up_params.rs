use openvm_stark_backend::{
    interaction::LogUpSecurityParameters,
    p3_field::{PrimeField32, PrimeField64},
};
use p3_baby_bear::BabyBear;

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

    let mut params = LogUpSecurityParameters {
        max_interaction_count: BabyBear::ORDER_U32,
        log_max_message_length: 7,
        pow_bits: 0,
    };

    // Security is linear in `pow_bits`, so the grinding needed to reach the target after the
    // list-size penalty is exact. Floor at `MIN_BABY_BEAR_LOGUP_POW_BITS` to keep the historical
    // baseline margin for the unique-decoding configs.
    let security_without_pow = baby_bear_logup_security_bits(&params, log2_pcs_list_size);
    let required_pow = (TARGET_LOGUP_SECURITY_BITS - security_without_pow)
        .ceil()
        .max(0.0) as usize;
    params.pow_bits = required_pow.max(MIN_BABY_BEAR_LOGUP_POW_BITS);

    assert!(
        baby_bear_logup_security_bits(&params, log2_pcs_list_size) >= TARGET_LOGUP_SECURITY_BITS
    );
    params
}

/// LogUp security bits with grinding, after the PCS list-size union bound. Matches
/// `SoundnessCalculator::calculate_logup_soundness` in the backend soundness module.
fn baby_bear_logup_security_bits(params: &LogUpSecurityParameters, log2_pcs_list_size: f64) -> f64 {
    let challenge_field_bits = 4.0 * (BabyBear::ORDER_U64 as f64).log2();
    challenge_field_bits
        - (2.0 * params.max_interaction_count as f64).log2()
        - params.log_max_message_length as f64
        - log2_pcs_list_size
        + params.pow_bits as f64
}
