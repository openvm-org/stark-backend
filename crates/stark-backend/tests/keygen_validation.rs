use std::sync::Arc;

use openvm_stark_backend::{
    keygen::{
        types::{KeygenError, MultiStarkProvingKey},
        MultiStarkKeygenBuilder,
    },
    test_utils::{
        default_test_params_small,
        dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    },
    AirRef, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues};

type SC = BabyBearPoseidon2Config;

#[derive(Clone, Copy)]
struct NoopAir {
    width: usize,
}

impl<F> BaseAir<F> for NoopAir {
    fn width(&self) -> usize {
        self.width
    }
}

impl<F> BaseAirWithPublicValues<F> for NoopAir {}
impl<F> PartitionedBaseAir<F> for NoopAir {}

impl<AB: AirBuilder> Air<AB> for NoopAir {
    fn eval(&self, _builder: &mut AB) {}
}

fn generate_pk_for_air(air: AirRef<SC>) -> Result<MultiStarkProvingKey<SC>, KeygenError> {
    let config = BabyBearPoseidon2Config::default_from_params(default_test_params_small());
    let mut keygen_builder = MultiStarkKeygenBuilder::new(config);
    keygen_builder.add_air(air);
    keygen_builder.generate_pk()
}

fn expect_keygen_err(air: AirRef<SC>) -> KeygenError {
    match generate_pk_for_air(air) {
        Ok(_) => panic!("keygen unexpectedly succeeded"),
        Err(err) => err,
    }
}

#[test]
fn keygen_rejects_zero_width_air() {
    let err = expect_keygen_err(Arc::new(NoopAir { width: 0 }));
    assert!(matches!(err, KeygenError::AirWidthZero { .. }));
}

#[test]
fn keygen_rejects_air_without_constraints_or_interactions() {
    let err = expect_keygen_err(Arc::new(NoopAir { width: 1 }));
    assert!(matches!(
        err,
        KeygenError::AirNoConstraintsOrInteractions { .. }
    ));
}

#[test]
fn keygen_rejects_interaction_with_empty_message() {
    let err = expect_keygen_err(Arc::new(DummyInteractionAir::new(0, true, 0)));
    assert!(matches!(
        err,
        KeygenError::InteractionMessageEmpty {
            interaction_index: 0,
            ..
        }
    ));
}
