//! Compute evaluations of quotient polynomial for keccakf AIR using symbolic DAG interpreter

use std::{sync::Arc, time::Instant};

use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_challenger::FieldChallenger,
    p3_commit::{Pcs, PolynomialSpace},
    p3_matrix::Matrix,
    p3_util::log2_strict_usize,
    prover::{cpu::quotient::QuotientCommitter, types::RapView},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        setup_tracing, FriParameters,
    },
    engine::StarkFriEngine,
    openvm_stark_backend::engine::StarkEngine,
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::Rng;

const NUM_PERMUTATIONS: usize = 1 << 10;
const LOG_BLOWUP: usize = 1;

// Newtype to implement extended traits
struct TestAir(KeccakAir);

impl<F> BaseAir<F> for TestAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.0)
    }
}
impl<F> BaseAirWithPublicValues<F> for TestAir {}
impl<F> PartitionedBaseAir<F> for TestAir {}

impl<AB: AirBuilder> Air<AB> for TestAir {
    fn eval(&self, builder: &mut AB) {
        self.0.eval(builder);
    }
}

fn main() {
    type SC = BabyBearPoseidon2Config;
    type Challenge = <SC as StarkGenericConfig>::Challenge;
    type Challenger = <SC as StarkGenericConfig>::Challenger;
    setup_tracing();
    let mut rng = create_seeded_rng();
    let air = TestAir(KeccakAir {});

    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(LOG_BLOWUP),
    );
    let mut keygen_builder = engine.keygen_builder();
    let _air_id = keygen_builder.add_air(Arc::new(air));
    let pk = keygen_builder.generate_pk();

    let inputs = (0..NUM_PERMUTATIONS)
        .map(|_| rng.random())
        .collect::<Vec<_>>();
    let trace = p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0);
    let trace_height = trace.height();
    let pcs = engine.config().pcs();
    let trace_domain: Domain<SC> =
        Pcs::<Challenge, Challenger>::natural_domain_for_degree(pcs, trace_height);
    let log_trace_height = log2_strict_usize(trace.height());
    let (_, data) = Pcs::<Challenge, Challenger>::commit(pcs, vec![(trace_domain, trace)]);

    let timer = Instant::now();
    let mut challenger = engine.new_challenger();
    let alpha: <BabyBearPoseidon2Config as StarkGenericConfig>::Challenge =
        challenger.sample_algebra_element();
    let qc: QuotientCommitter<'_, SC> = QuotientCommitter::new(pcs, alpha, LOG_BLOWUP);
    let quotient_degree = 1 << LOG_BLOWUP;
    let constraints_dag = &pk.per_air[0].vk.symbolic_constraints.constraints;
    let quotient_domain = trace_domain.create_disjoint_domain(trace_height * quotient_degree);
    let lde_on_quot_domain =
        Pcs::<Challenge, Challenger>::get_evaluations_on_domain(pcs, &data, 0, quotient_domain);
    let extended_view = RapView {
        log_trace_height: log_trace_height as u8,
        preprocessed: None,
        partitioned_main: vec![lde_on_quot_domain],
        public_values: vec![],
        per_phase: vec![],
    };
    let _quotient_values = qc.quotient_values(
        &[constraints_dag],
        vec![extended_view],
        &[quotient_degree as u8],
    );
    println!(
        "compute quotient values with DAG interpreter took: {:?}",
        timer.elapsed()
    );
}
