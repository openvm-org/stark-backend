use std::sync::Arc;

use itertools::{izip, Itertools};
use openvm_stark_backend::{
    keygen::MultiStarkKeygenBuilder,
    p3_matrix::dense::RowMajorMatrix,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        hal::DeviceDataTransporter,
        types::{AirProvingContext, ProvingContext},
        MultiTraceStarkProver, Prover,
    },
    verifier::{MultiTraceStarkVerifier, VerificationError},
    AirRef,
};
use p3_baby_bear::BabyBear;

use crate::config::{self, baby_bear_poseidon2::BabyBearPoseidon2Config};

pub mod dummy_interaction_air;

type Val = BabyBear;

pub fn verify_interactions(
    traces: Vec<RowMajorMatrix<Val>>,
    airs: Vec<AirRef<BabyBearPoseidon2Config>>,
    pis: Vec<Vec<Val>>,
) -> Result<(), VerificationError> {
    let perm = config::baby_bear_poseidon2::random_perm();
    let config = config::baby_bear_poseidon2::default_config(&perm);

    let mut keygen_builder = MultiStarkKeygenBuilder::new(&config);
    let air_ids = airs
        .into_iter()
        .map(|air| keygen_builder.add_air(air))
        .collect_vec();
    let pk = keygen_builder.generate_pk();
    let vk = pk.get_vk();

    let backend = CpuBackend::default();
    let pk = backend.transport_pk_to_device(&pk, air_ids.clone());
    let per_air: Vec<_> = izip!(air_ids, traces, pis)
        .map(|(air_id, trace, pvs)| {
            (
                air_id,
                AirProvingContext {
                    cached_mains: vec![],
                    common_main: Some(Arc::new(trace)),
                    public_values: pvs,
                },
            )
        })
        .collect();

    let challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
    let mut prover = MultiTraceStarkProver::new(backend, CpuDevice::new(&config), challenger);
    let proof = prover.prove(pk, ProvingContext::new(per_air));

    // Verify the proof:
    // Start from clean challenger
    let mut challenger = config::baby_bear_poseidon2::Challenger::new(perm.clone());
    let verifier = MultiTraceStarkVerifier::new(prover.device.config());
    verifier.verify(&mut challenger, &vk, &proof.into())
}
