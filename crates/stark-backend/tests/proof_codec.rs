//! Round-trip and malformed-input tests for vanilla and VK-aware proof decoding.
//! The wire format is backend-independent, so CPU-generated proofs cover both paths.

use std::io::{Cursor, ErrorKind};

use itertools::Itertools;
use openvm_stark_backend::{
    codec::{Decode, Encode},
    keygen::types::MultiStarkVerifyingKey0,
    proof::{Proof, WhirProof},
    test_utils::{
        test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
        MixtureFixture, MixtureFixtureEnum, PreprocessedAndCachedFixture, PreprocessedFibFixture,
        TestFixture,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::*;

type ConcreteSC = BabyBearPoseidon2Config;

fn test_proof_encode_decode<Fx: TestFixture<ConcreteSC>>(
    fx: Fx,
    params: SystemParams,
) -> eyre::Result<()> {
    let engine = BabyBearPoseidon2RefEngine::new(params);
    let pk = fx.keygen(&engine).0;
    let proof = fx.prove_from_transcript(&engine, &pk, &mut default_duplex_sponge_recorder());

    let mut proof_bytes = Vec::new();
    proof.encode(&mut proof_bytes)?;

    let decoded_proof = Proof::<ConcreteSC>::decode(&mut &proof_bytes[..])?;
    assert_eq!(proof, decoded_proof);

    let vk = pk.get_vk();
    let decoded_proof = Proof::<ConcreteSC>::decode_from_bytes_using_vk(&vk.inner, &proof_bytes)?;
    openvm_stark_backend::verifier::proof_shape::verify_proof_shape(&vk.inner, &decoded_proof)?;
    assert_eq!(proof, decoded_proof);

    // Streaming decoding leaves the following message untouched.
    proof_bytes.push(0);
    let mut reader = proof_bytes.as_slice();
    assert_eq!(
        proof,
        Proof::<ConcreteSC>::decode_using_vk(&vk.inner, &mut reader)?
    );
    assert_eq!(reader, &[0]);

    Ok(())
}

// ==================== Vanilla Tests ====================

#[test]
fn test_fib_proof_encode_decode() -> eyre::Result<()> {
    for log_trace_height in [3, 5] {
        test_proof_encode_decode(
            FibFixture::new(0, 1, 1 << log_trace_height),
            SystemParams::new_for_testing(5),
        )?;
    }
    Ok(())
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

// ==================== Vk-Aware Tests ====================

type Vk = MultiStarkVerifyingKey0<ConcreteSC>;

// Builds synthetic proof bytes and records each length field's name, byte offset, and expected
// VK-derived bound. Tests mutate these prefixes and truncate the payload to check that oversized
// lengths are rejected before payload reads.
#[derive(Default)]
struct Encoding {
    bytes: Vec<u8>,
    lengths: Vec<(&'static str, usize, usize)>,
}

impl Encoding {
    fn zeros(&mut self, bytes: usize) {
        self.bytes.resize(self.bytes.len() + bytes, 0);
    }

    fn length(&mut self, name: &'static str, value: usize, bound: usize) {
        self.lengths.push((name, self.bytes.len(), bound));
        self.bytes.extend_from_slice(&(value as u32).to_le_bytes());
    }

    fn offset(&self, name: &str) -> usize {
        self.lengths
            .iter()
            .find(|(field, _, _)| *field == name)
            .unwrap()
            .1
    }

    fn prefix(&self, name: &str, value: usize) -> Vec<u8> {
        let offset = self.offset(name);
        let mut bytes = self.bytes[..offset].to_vec();
        bytes.extend_from_slice(&(value as u32).to_le_bytes());
        bytes
    }
}

fn encoding(vk: &Vk, proof: &Proof<ConcreteSC>) -> Encoding {
    let mut e = Encoding::default();
    let p = &vk.params;
    let air = &vk.per_air[0];
    assert_eq!(vk.per_air.len(), 1);
    assert_eq!(air.num_interactions(), 0);
    e.bytes
        .extend_from_slice(&proof.encode_to_vec().unwrap()[..4]);
    e.zeros(32);
    e.length("trace_vdata length", 1, 1);
    e.bytes.push(1);
    e.zeros(4); // log_height
    let parts = air.num_parts();
    let degree = vk.max_constraint_degree() + 1;
    let skipped = (1 << p.l_skip) - 1;
    let columns = proof.batch_constraint_proof.column_openings[0]
        .iter()
        .map(Vec::len)
        .max()
        .unwrap();
    let rounds = p.num_whir_rounds();
    let sumchecks = p.num_whir_sumcheck_rounds();
    let depth = proof.whir_proof.initial_round_merkle_proofs[0][0].len();
    let witnesses = (rounds - 1) * 48 + (sumchecks + rounds) * 4;
    // (field, encoded length, maximum length, bytes following the prefix)
    for (name, value, bound, bytes) in [
        ("cached_commitments length", 2, 2, 64),
        ("public_values length", 1, 1, 0),
        (
            "public_values inner length",
            0,
            air.params.num_public_values,
            20,
        ),
        ("claims_per_layer length", 0, 0, 0),
        ("numerator_term_per_air length", 1, 1, 32),
        (
            "batch univariate_round_coeffs length",
            0,
            degree * skipped + 1,
            0,
        ),
        ("batch sumcheck_round_polys length", 1, p.n_stack, 0),
        ("batch sumcheck round width", 0, degree, 0),
        ("column_openings parts length", 1, parts, 0),
        ("column_openings columns length", 0, columns, 0),
        (
            "stacking univariate_round_coeffs length",
            0,
            2 * skipped + 1,
            0,
        ),
        ("stacking sumcheck_round_polys length", 0, p.n_stack, 0),
        ("stacking_openings length", 1, parts, 0),
        ("stacking_openings width", 0, p.w_stack, 4),
        (
            "whir_sumcheck_polys length",
            sumchecks,
            sumchecks,
            sumchecks * 32,
        ),
        ("codeword_commits length", rounds - 1, rounds - 1, witnesses),
        ("WHIR num_commits", 1, parts, 0),
        (
            "WHIR initial query count",
            1,
            p.whir.rounds[0].num_queries,
            0,
        ),
        ("WHIR initial Merkle depth", 0, depth, 0),
        ("WHIR initial opened row width", 0, p.w_stack, 0),
    ] {
        e.length(name, value, bound);
        e.zeros(bytes);
    }
    for round in &p.whir.rounds[1..] {
        e.length("WHIR non-initial query count", 1, round.num_queries);
        e.zeros((1 << p.k_whir()) * 16);
    }
    e.length("WHIR first non-initial Merkle depth", depth - 1, depth - 1);
    for i in 0..rounds - 1 {
        e.zeros((depth - 1 - i) * 32);
    }
    e.length("final_poly length", 0, 1 << p.log_final_poly_len());
    e
}

fn fixture() -> (Vk, Proof<ConcreteSC>) {
    let mut params = test_system_params_small(2, 5, 3);
    params.max_constraint_degree = 4;
    let engine: BabyBearPoseidon2RefEngine = BabyBearPoseidon2RefEngine::new(params.clone());
    let config = BabyBearPoseidon2Config::default_from_params(params);
    let (vk, proof) =
        PreprocessedAndCachedFixture::new((0..32).map(|i| i % 2 == 0).collect(), config, 2)
            .keygen_and_prove(&engine);
    (vk.inner, proof)
}

fn decode(vk: &Vk, proof: &Proof<ConcreteSC>) -> Proof<ConcreteSC> {
    Proof::decode_from_bytes_using_vk(vk, &proof.encode_to_vec().unwrap()).unwrap()
}

fn reject_prefix(vk: &Vk, bytes: &[u8], field: &str) {
    let mut reader = Cursor::new(bytes);
    let err = Proof::<ConcreteSC>::decode_using_vk(vk, &mut reader).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData, "{field}: {err}");
    assert!(err.to_string().contains(field), "{field}: {err}");
    assert_eq!(reader.position(), bytes.len() as u64, "{field}");
}

fn check_length(vk: &Vk, prefix: &[u8], name: &str, bound: usize) {
    for value in [bound, bound + 1, u32::MAX as usize] {
        let mut bytes = prefix.to_vec();
        bytes.extend_from_slice(&(value as u32).to_le_bytes());
        if value == bound {
            let err = Proof::<ConcreteSC>::decode_from_bytes_using_vk(vk, &bytes).unwrap_err();
            assert_eq!(err.kind(), ErrorKind::UnexpectedEof, "{name}: {err}");
        } else {
            reject_prefix(vk, &bytes, name);
        }
    }
}

// Construct only the WHIR section, including shapes the encoder cannot represent.
fn whir_bytes(k: usize, queries: &[usize], depth: usize) -> Vec<u8> {
    let rounds = queries.len() + 1;
    let mut bytes = vec![0; 4];
    bytes.extend_from_slice(&((k * rounds) as u32).to_le_bytes());
    bytes.resize(bytes.len() + k * rounds * 32, 0);
    bytes.extend_from_slice(&(queries.len() as u32).to_le_bytes());
    bytes.resize(
        bytes.len() + queries.len() * 48 + (k * rounds + rounds) * 4,
        0,
    );
    bytes.extend_from_slice(&1u32.to_le_bytes()); // one commitment
    bytes.extend_from_slice(&0u32.to_le_bytes()); // no initial queries
    for &count in queries {
        bytes.extend_from_slice(&(count as u32).to_le_bytes());
        bytes.resize(bytes.len() + count * (1 << k) * 16, 0);
    }
    bytes.extend_from_slice(&(depth as u32).to_le_bytes());
    for (i, &count) in queries.iter().enumerate() {
        bytes.resize(bytes.len() + count * depth.saturating_sub(i) * 32, 0);
    }
    bytes.extend_from_slice(&0u32.to_le_bytes()); // final polynomial
    bytes
}

#[test]
fn test_vk_gkr_claims_length_bound() -> eyre::Result<()> {
    let engine: BabyBearPoseidon2RefEngine =
        BabyBearPoseidon2RefEngine::new(test_system_params_small(2, 5, 3));
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let interactions: usize = vk
        .inner
        .per_air
        .iter()
        .map(|air| air.num_interactions())
        .sum();
    assert!(interactions > 0);

    // Each interaction contributes at most one term per row at the maximum trace height.
    // Bound the claim-layer count by the bit length of that maximum term count.
    let max_terms = interactions << vk.inner.params.log_stacked_height();
    let max_claims = max_terms.ilog2() as usize + 1;

    let bytes = proof.encode_to_vec()?;
    let gkr_start = bytes.len()
        - proof.whir_proof.encode_to_vec()?.len()
        - proof.stacking_proof.encode_to_vec()?.len()
        - proof.batch_constraint_proof.encode_to_vec()?.len()
        - proof.gkr_proof.encode_to_vec()?.len();
    let claims_prefix = gkr_start + 4 + 16; // Base-field witness, then extension-field claim.
    check_length(
        &vk.inner,
        &bytes[..claims_prefix],
        "claims_per_layer length",
        max_claims,
    );
    Ok(())
}

#[test]
fn test_vk_length_boundaries() {
    let (vk, proof) = fixture();
    let e = encoding(&vk, &proof);
    let decoded = Proof::<ConcreteSC>::decode_from_bytes_using_vk(&vk, &e.bytes).unwrap();
    assert_eq!(
        decoded,
        Proof::<ConcreteSC>::decode_from_bytes(&e.bytes).unwrap()
    );
    for &(name, offset, bound) in &e.lengths {
        check_length(&vk, &e.bytes[..offset], name, bound);
    }
}

#[test]
fn test_mixed_heights_proof_encode_decode() -> eyre::Result<()> {
    let params = SystemParams::new_for_testing(6);
    let fx = MixtureFixture::<ConcreteSC>::new(vec![
        MixtureFixtureEnum::FibFixture(FibFixture::new(0, 1, 8)),
        MixtureFixtureEnum::FibFixture(FibFixture::new(0, 1, 64)),
        MixtureFixtureEnum::InteractionsFixture11(InteractionsFixture11),
        MixtureFixtureEnum::FibFixture(FibFixture::new(0, 1, 32)),
    ]);
    test_proof_encode_decode(fx, params)
}

#[test]
fn test_vk_metadata_and_air_indexing() {
    let (mut vk, mut proof) = fixture();
    let template = vk.per_air[0].clone();
    let data = proof.trace_vdata[0].clone().unwrap();
    vk.per_air.clear();
    proof.trace_vdata.clear();
    proof.public_values.clear();
    // Different per-AIR limits, absent AIRs between present ones, and a byte boundary.
    for i in 0..9 {
        let mut air = template.clone();
        air.params.width.cached_mains.resize(i, 1);
        air.params.num_public_values = i;
        air.is_required = true;
        let mut data = data.clone();
        data.cached_commitments
            .resize(i, data.cached_commitments[0]);
        data.log_height = u32::MAX as usize;
        vk.per_air.push(air);
        proof.trace_vdata.push((i % 2 == 0).then_some(data));
        proof.public_values.push(vec![F::default(); i]);
    }
    for constraint in &mut vk.trace_height_constraints {
        constraint.coefficients.resize(9, 0);
        constraint.threshold = 0;
    }
    // Required AIRs, preprocessed heights, and trace constraints are verifier concerns.
    assert_eq!(proof, decode(&vk, &proof));
    for i in [0, 2, 8] {
        let mut bad = proof.clone();
        bad.trace_vdata[i]
            .as_mut()
            .unwrap()
            .cached_commitments
            .push(data.cached_commitments[0]);
        assert!(Proof::<ConcreteSC>::decode_from_bytes_using_vk(
            &vk,
            &bad.encode_to_vec().unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("cached_commitments"));
    }
    for i in [0, 1, 8] {
        let mut bad = proof.clone();
        bad.public_values[i].push(F::default());
        assert!(Proof::<ConcreteSC>::decode_from_bytes_using_vk(
            &vk,
            &bad.encode_to_vec().unwrap()
        )
        .unwrap_err()
        .to_string()
        .contains("public_values inner"));
    }
    let mut bytes = proof.encode_to_vec().unwrap();
    bytes[41] |= 0x80;
    assert!(Proof::<ConcreteSC>::decode_from_bytes_using_vk(&vk, &bytes)
        .unwrap_err()
        .to_string()
        .contains("padding"));

    // Conservative bounds include absent AIRs; exact shapes are left to verification.
    let batch = &mut proof.batch_constraint_proof;
    batch.numerator_term_per_air.push(EF::default());
    batch.denominator_term_per_air.push(EF::default());
    batch.column_openings.push(vec![]);
    proof.stacking_proof.stacking_openings.push(vec![]);
    let whir = &mut proof.whir_proof;
    let queries = whir.initial_round_opened_rows[0].len();
    whir.initial_round_opened_rows
        .push(vec![vec![vec![]; 1 << vk.params.k_whir()]; queries]);
    whir.initial_round_merkle_proofs
        .push(whir.initial_round_merkle_proofs[0].clone());
    proof.trace_vdata.fill(None);
    assert_eq!(proof, decode(&vk, &proof));
    assert!(openvm_stark_backend::verifier::proof_shape::verify_proof_shape(&vk, &proof).is_err());
}

#[test]
fn test_vanilla_codec_checks_are_shared() {
    let (vk, mut proof) = fixture();
    proof.public_values.push(vec![]);
    let bytes = proof.encode_to_vec().unwrap();
    assert_eq!(
        proof,
        Proof::<ConcreteSC>::decode_from_bytes(&bytes).unwrap()
    );
    assert!(Proof::<ConcreteSC>::decode_from_bytes_using_vk(&vk, &bytes).is_err());

    let e = encoding(&vk, &proof);
    let whir_start = e.offset("whir_sumcheck_polys length") - 4;
    let reject_both = |bytes: &[u8], kind, message: &str| {
        for result in [
            Proof::<ConcreteSC>::decode_from_bytes(bytes),
            Proof::<ConcreteSC>::decode_from_bytes_using_vk(&vk, bytes),
        ] {
            let err = result.unwrap_err();
            assert_eq!(err.kind(), kind, "{err}");
            assert!(err.to_string().contains(message), "{err}");
        }
    };
    for (offset, value, message, kind) in [
        (0, 0, "CODEC_VERSION", ErrorKind::Other),
        (4, u32::MAX, "F::ORDER_U32", ErrorKind::Other),
        (whir_start, u32::MAX, "F::ORDER_U32", ErrorKind::Other),
        (whir_start + 8, u32::MAX, "F::ORDER_U32", ErrorKind::Other),
        (
            e.offset("WHIR num_commits"),
            0,
            "num_commits",
            ErrorKind::InvalidData,
        ),
    ] {
        let mut bad = e.bytes.clone();
        bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        reject_both(&bad, kind, message);
    }
    let mut bad = e.bytes.clone();
    bad[40] |= 0x80;
    reject_both(&bad, ErrorKind::Other, "padding");
    let mut bad = e.bytes.clone();
    bad.push(0);
    reject_both(&bad, ErrorKind::Other, "trailing bytes");
    for end in 0..e.bytes.len() {
        reject_both(&e.bytes[..end], ErrorKind::UnexpectedEof, "");
    }
    let mut bad = e.bytes.clone();
    let count = vk.params.num_whir_sumcheck_rounds();
    bad[whir_start + 4..whir_start + 8].copy_from_slice(&((count - 1) as u32).to_le_bytes());
    bad.drain(whir_start + 8..whir_start + 40);
    reject_both(&bad, ErrorKind::InvalidData, "must be a multiple");
}

#[test]
fn test_vk_empty_vector_attacks_and_total_widths() {
    let (vk, proof) = fixture();
    let e = encoding(&vk, &proof);
    // These checks must fire at the count, without visiting even empty children.
    for (field, suffix) in [
        ("batch sumcheck_round_polys length", vec![0u32]),
        ("WHIR num_commits", vec![0]),
        ("WHIR initial query count", vec![0, 0]),
    ] {
        let mut bytes = e.prefix(field, u32::MAX as usize);
        let end = bytes.len();
        for word in suffix {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let mut reader = Cursor::new(bytes);
        let err = Proof::<ConcreteSC>::decode_using_vk(&vk, &mut reader).unwrap_err();
        assert!(err.to_string().contains(field), "{err}");
        assert_eq!(reader.position(), end as u64);
    }

    for (count, width, payload_bytes) in [
        (
            "stacking_openings length",
            "stacking_openings width",
            vk.params.w_stack * 16,
        ),
        ("WHIR num_commits", "WHIR initial opened row width", 0),
    ] {
        let mut bytes = e.prefix(width, vk.params.w_stack);
        let offset = e.offset(count);
        bytes[offset..offset + 4].copy_from_slice(&2u32.to_le_bytes());
        bytes.resize(bytes.len() + payload_bytes, 0);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        reject_prefix(&vk, &bytes, width);
    }
    // Zero queries omit widths and permit the unused zero-depth sentinel.
    let mut bytes = e.bytes[..e.offset("whir_sumcheck_polys length") - 4].to_vec();
    bytes.extend(whir_bytes(
        vk.params.k_whir(),
        &vec![0; vk.params.num_whir_rounds() - 1],
        0,
    ));
    Proof::<ConcreteSC>::decode_from_bytes_using_vk(&vk, &bytes).unwrap();

    // Reducing the encoded round count must not increase the derived folding factor.
    reject_prefix(
        &vk,
        &e.prefix("codeword_commits length", 0),
        "WHIR folding factor",
    );
}

#[test]
fn test_whir_arithmetic_edges() {
    WhirProof::<ConcreteSC>::decode_from_bytes(&whir_bytes(usize::BITS as usize - 1, &[], 0))
        .unwrap();
    let err = WhirProof::<ConcreteSC>::decode_from_bytes(&whir_bytes(usize::BITS as usize, &[], 0))
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("dimension overflow"));
    for queries in [&[0, 0][..], &[1][..], &[1, 0][..]] {
        WhirProof::<ConcreteSC>::decode_from_bytes(&whir_bytes(0, queries, 0)).unwrap();
    }
    let err = WhirProof::<ConcreteSC>::decode_from_bytes(&whir_bytes(0, &[0, 1], 0)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("depth underflow"));
}
