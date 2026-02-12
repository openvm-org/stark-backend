use p3_baby_bear::BabyBear;
use stark_backend_v2::{
    interaction::LogUpSecurityParameters,
    p3_field::{extension::BinomialExtensionField, PrimeField32},
};

pub fn log_up_security_params_baby_bear_100_bits() -> LogUpSecurityParameters {
    let params = LogUpSecurityParameters {
        max_interaction_count: BabyBear::ORDER_U32,
        log_max_message_length: 7,
        pow_bits: 18,
    };
    assert!(params.bits_of_security::<BinomialExtensionField<BabyBear, 4>>() >= 100);
    params
}
