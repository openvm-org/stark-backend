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
    poly_common::Squarable,
    proof::Proof,
    verifier::{verify, VerifierError},
    StarkProtocolConfig,
};

use crate::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CpuTranscript};

/// Extraction shim (rust-to-lean, Phase E): force `ExpPowers2::next` to be
/// monomorphized at the verifier's concrete element types. `ExpPowers2` (a custom
/// `Iterator` yielding `x, x², x⁴, …`) is only ever consumed inside the *opaque* core
/// `Take`/`Zip::next`, so under `--monomorphize` its own `next` is never reached at a
/// concrete type and Charon emits nothing for it. Calling it concretely here lets the
/// translator model the lazy iterator faithfully (see rust-to-lean
/// docs/ITERATOR_MODEL.md). An additional `--start-from` root; never executed.
pub fn force_exp_powers_next(
    f: <BabyBearPoseidon2Config as StarkProtocolConfig>::F,
    ef: <BabyBearPoseidon2Config as StarkProtocolConfig>::EF,
) {
    let mut f_powers = f.exp_powers_of_2();
    let _ = f_powers.next();
    let mut ef_powers = ef.exp_powers_of_2();
    let _ = ef_powers.next();
}

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
