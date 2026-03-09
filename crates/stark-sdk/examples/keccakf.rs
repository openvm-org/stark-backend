//! Prove keccakf-air over BabyBear using poseidon2

use std::sync::Arc;

use cfg_if::cfg_if;
use eyre::eyre;
use openvm_stark_sdk::{
    config::app_params_with_100_bits_security,
    openvm_stark_backend::{
        p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues},
        p3_field::Field,
        prover::{AirProvingContext, DeviceDataTransporter, ProvingContext},
        PartitionedBaseAir, StarkEngine,
    },
};
use p3_keccak_air::KeccakAir;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tracing::info_span;

const NUM_PERMUTATIONS: usize = 1 << 10;

// Newtype to implement extended traits
struct TestAir(KeccakAir);

impl<F> BaseAir<F> for TestAir {
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
    let params = app_params_with_100_bits_security(21);
    let mut rng = StdRng::seed_from_u64(42);
    let air = TestAir(KeccakAir {});

    let inputs = (0..NUM_PERMUTATIONS)
        .map(|_| rng.random())
        .collect::<Vec<_>>();
    let trace = info_span!("generate_trace").in_scope(|| {
        p3_keccak_air::generate_trace_rows::<openvm_stark_sdk::p3_baby_bear::BabyBear>(inputs, 0)
    });

    cfg_if! {
        if #[cfg(feature = "cpu-backend")] {
            cfg_if! {
                if #[cfg(feature = "baby-bear-bn254-poseidon2")] {
                    use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine as Engine;
                    println!("Using BabyBearBn254Poseidon2CpuEngine");
                } else {
                    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine as Engine;
                    println!("Using BabyBearPoseidon2CpuEngine");
                }
            }

            let engine: Engine = StarkEngine::new(params);
            let (pk, vk) = engine.keygen(&[Arc::new(air)]);
            let trace_ctx = AirProvingContext::simple_no_pis(trace);
            let d_pk = engine.device().transport_pk_to_device(&pk);
            let proof = engine
                .prove(&d_pk, ProvingContext::new(vec![(0, trace_ctx)]))
                .map_err(|e| eyre!("Proving failed: {e:?}"))?;
            engine
                .verify(&vk, &proof)
                .map_err(|e| eyre!("Proof failed to verify: {e}"))?;
        } else {
            use openvm_stark_sdk::openvm_stark_backend::prover::ColMajorMatrix;

            cfg_if! {
                if #[cfg(feature = "baby-bear-bn254-poseidon2")] {
                    use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2RefEngine as Engine;
                    println!("Using BabyBearBn254Poseidon2RefEngine");
                } else {
                    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2RefEngine as Engine;
                    println!("Using BabyBearPoseidon2RefEngine");
                }
            }

            let engine: Engine = StarkEngine::new(params);
            let (pk, vk) = engine.keygen(&[Arc::new(air)]);
            let trace_ctx = AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace));
            let d_pk = engine.device().transport_pk_to_device(&pk);
            let proof = engine
                .prove(&d_pk, ProvingContext::new(vec![(0, trace_ctx)]))
                .map_err(|e| eyre!("Proving failed: {e:?}"))?;
            engine
                .verify(&vk, &proof)
                .map_err(|e| eyre!("Proof failed to verify: {e}"))?;
        }
    }

    Ok(())
}
