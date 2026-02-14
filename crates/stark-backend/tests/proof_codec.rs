use std::io;

use itertools::Itertools;
use openvm_stark_backend::{
    codec::{Decode, Encode},
    proof::Proof,
    test_utils::{
        test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
        PreprocessedFibFixture, TestFixture,
    },
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::*;

type ConcreteSC = BabyBearPoseidon2Config;

fn test_proof_encode_decode<Fx: TestFixture<ConcreteSC>>(
    fx: Fx,
    params: SystemParams,
) -> io::Result<()> {
    let engine = BabyBearPoseidon2CpuEngine::new(params);
    let pk = fx.keygen(&engine).0;
    let proof = fx.prove_from_transcript(&engine, &pk, &mut default_duplex_sponge_recorder());

    let mut proof_bytes = Vec::new();
    proof.encode(&mut proof_bytes).unwrap();

    let decoded_proof = Proof::<ConcreteSC>::decode(&mut &proof_bytes[..]).unwrap();
    assert_eq!(proof, decoded_proof);
    Ok(())
}

#[test]
fn test_fib_proof_encode_decode() -> io::Result<()> {
    let log_trace_height = 5;
    let fx = FibFixture::new(0, 1, 1 << log_trace_height);
    let params = SystemParams::new_for_testing(log_trace_height);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_interactions_proof_encode_decode() -> io::Result<()> {
    let fx = InteractionsFixture11;
    let params = test_system_params_small(2, 5, 3);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_cached_proof_encode_decode() -> io::Result<()> {
    let params = test_system_params_small(2, 5, 3);
    let config = BabyBearPoseidon2Config::default_from_params(params.clone());
    let fx = CachedFixture11::new(config);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_preprocessed_proof_encode_decode() -> io::Result<()> {
    let log_trace_height = 5;
    let params = SystemParams::new_for_testing(log_trace_height);
    let sels = (0..(1 << log_trace_height))
        .map(|i| i % 2 == 0)
        .collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    test_proof_encode_decode(fx, params)
}
