//! Serialization round-trip tests for `Proof::encode` / `Proof::decode`.
//!
//! Each test generates a proof with `BabyBearPoseidon2CpuEngine`, encodes it,
//! decodes it, and asserts equality. The codec is backend-independent (the same
//! proof bytes are produced regardless of which backend generated the proof), so
//! running these on a GPU backend would exercise the same codec path. Not in the
//! shared backend test suite for this reason.

use itertools::Itertools;
use openvm_stark_backend::{
    codec::{Decode, Encode},
    proof::Proof,
    test_utils::{
        test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
        PreprocessedAndCachedFixture, PreprocessedFibFixture, TestFixture,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::*;

type ConcreteSC = BabyBearPoseidon2Config;

fn test_proof_encode_decode<Fx: TestFixture<ConcreteSC>>(
    fx: Fx,
    params: SystemParams,
) -> eyre::Result<()> {
    let engine = BabyBearPoseidon2CpuEngine::new(params);
    let pk = fx.keygen(&engine).0;
    let proof = fx.prove_from_transcript(&engine, &pk, &mut default_duplex_sponge_recorder());

    let mut proof_bytes = Vec::new();
    proof.encode(&mut proof_bytes)?;

    let decoded_proof = Proof::<ConcreteSC>::decode(&mut &proof_bytes[..])?;
    assert_eq!(proof, decoded_proof);
    Ok(())
}

#[test]
fn test_fib_proof_encode_decode() -> eyre::Result<()> {
    let log_trace_height = 5;
    let fx = FibFixture::new(0, 1, 1 << log_trace_height);
    let params = SystemParams::new_for_testing(log_trace_height);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_interactions_proof_encode_decode() -> eyre::Result<()> {
    let fx = InteractionsFixture11;
    let params = test_system_params_small(2, 5, 3);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_cached_proof_encode_decode() -> eyre::Result<()> {
    let params = test_system_params_small(2, 5, 3);
    let config = BabyBearPoseidon2Config::default_from_params(params.clone());
    let fx = CachedFixture11::new(config);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_preprocessed_proof_encode_decode() -> eyre::Result<()> {
    let log_trace_height = 5;
    let params = SystemParams::new_for_testing(log_trace_height);
    let sels = (0..(1 << log_trace_height))
        .map(|i| i % 2 == 0)
        .collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_preprocessed_and_multi_cached_proof_encode_decode() -> eyre::Result<()> {
    let log_trace_height = 5;
    let params = SystemParams::new_for_testing(log_trace_height);
    let sels = (0..(1 << log_trace_height))
        .map(|i| i % 2 == 0)
        .collect_vec();
    for num_cached_parts in [1, 2, 3] {
        let config = BabyBearPoseidon2Config::default_from_params(params.clone());
        let fx = PreprocessedAndCachedFixture::new(sels.clone(), config, num_cached_parts);
        test_proof_encode_decode(fx, params.clone())?;
    }
    Ok(())
}
