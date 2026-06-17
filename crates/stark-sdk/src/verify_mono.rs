//! Concrete monomorphization entry point for Charon extraction (rust-to-lean,
//! Phase E). The SWIRL noninteractive verifier `verifier::verify` is generic over
//! `SC: StarkProtocolConfig` and `TS: FiatShamirTranscript<SC>`, whose associated
//! types reach GAT-bearing p3 traits. This shim pins them to the concrete
//! `BabyBearPoseidon2Config` + `CpuTranscript` so Charon's `--monomorphize` can
//! collapse the generics to concrete types.
//!
//! It is purely an extraction shim: it forwards, unchanged, to the production
//! verifier. Used as the `--start-from` root.

use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    proof::Proof,
    verifier::{verify, VerifierError},
    StarkProtocolConfig,
};

use crate::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CpuTranscript};

/// Concrete instantiation of the SWIRL noninteractive verifier at
/// `BabyBearPoseidon2Config` with a `CpuTranscript`.
pub fn verify_babybear_poseidon2(
    config: &BabyBearPoseidon2Config,
    mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    proof: &Proof<BabyBearPoseidon2Config>,
    transcript: &mut CpuTranscript,
) -> Result<(), VerifierError<<BabyBearPoseidon2Config as StarkProtocolConfig>::EF>> {
    verify(config, mvk, proof, transcript)
}
