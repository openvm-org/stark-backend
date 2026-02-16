use openvm_stark_backend::{
    proof::{BatchConstraintProof, GkrProof},
    prover::{DeviceMultiStarkProvingKey, ProvingContext},
};

use crate::{
    prelude::{EF, SC},
    sponge::DuplexSpongeMetal,
    MetalBackend,
};

pub mod batch_mle;
pub mod batch_mle_monomial;
pub mod errors;
pub mod fold_ple;
pub mod fractional;
pub mod gkr_input;
pub mod mle_round;
pub mod round0;
pub mod rules;

pub fn prove_zerocheck_and_logup_metal(
    transcript: &mut DuplexSpongeMetal,
    mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
    ctx: &ProvingContext<MetalBackend>,
    save_memory: bool,
    monomial_num_y_threshold: u32,
    sm_count: u32,
) -> (GkrProof<SC>, BatchConstraintProof<SC>, Vec<EF>) {
    todo!("zerocheck and logup proof not yet implemented for Metal backend")
}
