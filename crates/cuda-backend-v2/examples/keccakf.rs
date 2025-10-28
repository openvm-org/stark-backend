//! Prove keccakf-air over BabyBear using poseidon2 for FRI hash.

use std::sync::Arc;

use cuda_backend_v2::{GpuBackendV2, GpuDeviceV2};
use eyre::eyre;
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::Field,
    prover::Prover,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::default_engine,
    openvm_stark_backend::engine::StarkEngine,
};
use p3_baby_bear::BabyBear;
use p3_keccak_air::KeccakAir;
use rand::{Rng, SeedableRng, rngs::StdRng};
use stark_backend_v2::{
    keygen::types::{MultiStarkProvingKeyV2, SystemParams},
    poseidon2::sponge::DuplexSponge,
    prover::{
        AirProvingContextV2, ColMajorMatrix, CoordinatorV2, DeviceDataTransporterV2,
        ProvingContextV2,
    },
    verifier::verify,
};
use tracing::info_span;

const NUM_PERMUTATIONS: usize = 1 << 10;

// Newtype to implement extended traits
struct TestAir(KeccakAir);

impl<F: Field> BaseAir<F> for TestAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.0)
    }
}
impl<F: Field> BaseAirWithPublicValues<F> for TestAir {}
impl<F: Field> PartitionedBaseAir<F> for TestAir {}

impl<AB: AirBuilder> Air<AB> for TestAir {
    fn eval(&self, builder: &mut AB) {
        self.0.eval(builder);
    }
}

fn main() -> eyre::Result<()> {
    let l_skip = 4;
    let n_stack = 17;
    let k_whir = 4;
    let log_final_poly_len = (l_skip + n_stack) % k_whir;
    let params = SystemParams {
        l_skip,
        n_stack,
        log_blowup: 1,
        k_whir,
        num_whir_queries: 100,
        log_final_poly_len,
        logup_pow_bits: 16,
        whir_pow_bits: 16,
    };

    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<()> {
        let mut rng = StdRng::seed_from_u64(42);
        let air = TestAir(KeccakAir {});

        let engine = default_engine();
        let mut keygen_builder = engine.keygen_builder();
        let air_id = keygen_builder.add_air(Arc::new(air));
        let pk_v1 = keygen_builder.generate_pk();
        let pk = MultiStarkProvingKeyV2::from_v1(params, pk_v1);

        let inputs = (0..NUM_PERMUTATIONS)
            .map(|_| rng.random())
            .collect::<Vec<_>>();
        let trace = info_span!("generate_trace")
            .in_scope(|| p3_keccak_air::generate_trace_rows::<BabyBear>(inputs, 0));
        let device = GpuDeviceV2::new(params);
        let d_trace = device.transport_matrix_to_device(&ColMajorMatrix::from_row_major(&trace));

        let air_ctx = AirProvingContextV2::simple_no_pis(d_trace);
        let d_pk = device.transport_pk_to_device(&pk);
        let mut prover = CoordinatorV2::new(GpuBackendV2, device, DuplexSponge::default());
        let proof = prover.prove(&d_pk, ProvingContextV2::new(vec![(air_id, air_ctx)]));

        verify(&pk.get_vk(), &proof, &mut DuplexSponge::default())
            .map_err(|e| eyre!("Proof failed to verify: {e}"))
    })
}
