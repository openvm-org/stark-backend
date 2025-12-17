use std::{cmp::Reverse, iter::zip};

use itertools::Itertools;
use openvm_stark_backend::{
    config::StarkGenericConfig,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        types::AirProvingContext,
    },
    verifier::VerificationError,
    AirRef,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::engine::{StarkFriEngine, VerificationDataWithFriParams};

/// `stark-backend::prover::types::ProofInput` without specifying AIR IDs.
pub struct ProofInputForTest<SC: StarkGenericConfig> {
    pub airs: Vec<AirRef<SC>>,
    pub per_air: Vec<AirProvingContext<CpuBackend<SC>>>,
}

impl<SC: StarkGenericConfig> ProofInputForTest<SC> {
    pub fn run_test(
        self,
        engine: &impl StarkFriEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    ) -> Result<VerificationDataWithFriParams<SC>, VerificationError> {
        assert_eq!(self.airs.len(), self.per_air.len());
        engine.run_test(self.airs, self.per_air)
    }
    /// Sort AIRs by their trace height in descending order. This should not be used outside
    /// static-verifier because a dynamic verifier should support any AIR order.
    /// This is related to an implementation detail of FieldMerkleTreeMMCS which is used in most
    /// configs. Reference: <https://github.com/Plonky3/Plonky3/blob/27b3127dab047e07145c38143379edec2960b3e1/merkle-tree/src/merkle_tree.rs#L53>
    pub fn sort_chips(&mut self) {
        let airs = std::mem::take(&mut self.airs);
        let air_proving_ctxs = std::mem::take(&mut self.per_air);
        let (airs, air_proof_inputs): (Vec<_>, Vec<_>) = zip(airs, air_proving_ctxs)
            .sorted_by_key(|(_, air_proving_ctx)| {
                Reverse(
                    air_proving_ctx
                        .common_main
                        .as_ref()
                        .map(|trace| trace.height())
                        .unwrap_or(0),
                )
            })
            .unzip();
        self.airs = airs;
        self.per_air = air_proof_inputs;
    }
}

/// Deterministic seeded RNG, for testing use
pub fn create_seeded_rng() -> StdRng {
    let seed = [42; 32];
    StdRng::from_seed(seed)
}

pub fn create_seeded_rng_with_seed(seed: u64) -> StdRng {
    let seed_be = seed.to_be_bytes();
    let mut seed = [0u8; 32];
    seed[24..32].copy_from_slice(&seed_be);
    StdRng::from_seed(seed)
}

// Returns row major matrix
pub fn generate_random_matrix<F: PrimeCharacteristicRing>(
    mut rng: impl Rng,
    height: usize,
    width: usize,
) -> Vec<Vec<F>> {
    (0..height)
        .map(|_| (0..width).map(|_| F::from_u32(rng.random())).collect_vec())
        .collect_vec()
}

pub fn to_field_vec<F: PrimeCharacteristicRing>(v: Vec<u32>) -> Vec<F> {
    v.into_iter().map(F::from_u32).collect()
}

/// A macro to create a `Vec<Arc<dyn AnyRap<_>>>` from a list of AIRs because Rust cannot infer the
/// type correctly when using `vec!`.
#[macro_export]
macro_rules! any_rap_arc_vec {
    [$($e:expr),*] => {
        {
            let chips: Vec<std::sync::Arc<dyn openvm_stark_backend::rap::AnyRap<_>>> = vec![$(std::sync::Arc::new($e)),*];
            chips
        }
    };
}

#[macro_export]
macro_rules! assert_sc_compatible_with_serde {
    ($sc:ty) => {
        static_assertions::assert_impl_all!(openvm_stark_backend::keygen::types::MultiStarkProvingKey<$sc>: serde::Serialize, serde::de::DeserializeOwned);
        static_assertions::assert_impl_all!(openvm_stark_backend::keygen::types::MultiStarkVerifyingKey<$sc>: serde::Serialize, serde::de::DeserializeOwned);
        static_assertions::assert_impl_all!(openvm_stark_backend::proof::Proof<$sc>: serde::Serialize, serde::de::DeserializeOwned);
    };
}
