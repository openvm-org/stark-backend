use openvm_cuda_backend::prelude::F;
use p3_field::{FieldAlgebra, PrimeField32};
use rand::{Rng, SeedableRng, rngs::StdRng};
use stark_backend_v2::{
    poseidon2::sponge::DuplexSponge, verifier::sumcheck::verify_sumcheck_multilinear,
};

use crate::sumcheck::sumcheck_multilinear_gpu;

#[test]
fn test_plain_multilinear_sumcheck() -> Result<(), String> {
    let n = 15;
    let mut rng = StdRng::from_seed([228; 32]);

    let num_pts = 1 << n;
    assert!((F::ORDER_U32 - 1) % num_pts == 0);

    let evals = (0..num_pts)
        .map(|_| F::from_canonical_u32(rng.random_range(0..F::ORDER_U32)))
        .collect::<Vec<_>>();
    let mut prover_sponge_gpu = DuplexSponge::default();
    let mut verifier_sponge_gpu = DuplexSponge::default();

    let (proof_gpu, _) = sumcheck_multilinear_gpu(&mut prover_sponge_gpu, &evals);

    verify_sumcheck_multilinear::<F, _>(&mut verifier_sponge_gpu, &proof_gpu)
}
