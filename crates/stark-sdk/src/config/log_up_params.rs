use openvm_stark_backend::{
    interaction::LogUpSecurityParameters,
    p3_field::{extension::BinomialExtensionField, PrimeField32},
};
use p3_baby_bear::BabyBear;

pub fn log_up_security_params_baby_bear_100_bits() -> LogUpSecurityParameters {
    let params = LogUpSecurityParameters {
        max_interaction_count: BabyBear::ORDER_U32,
        log_max_message_length: 7,
        log_up_pow_bits: 15,
    };
    assert!(params.conjectured_bits_of_security::<BinomialExtensionField<BabyBear, 4>>() >= 100);
    params
}
