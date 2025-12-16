use std::io::{Error, Read, Result, Write};

use serde::{Deserialize, Serialize};

use crate::{
    codec::{decode_into_vec, encode_iter, Decode, Encode},
    Digest, EF, F,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    /// The commitment to the data in common_main.
    pub common_main_commit: Digest,

    /// For each AIR in vkey order, the corresponding trace shape, or None if
    /// the trace is empty. In a valid proof, if `vk.per_air[i].is_required`,
    /// then `trace_vdata[i]` must be `Some(_)`.
    pub trace_vdata: Vec<Option<TraceVData>>,

    /// For each AIR in vkey order, the public values. Public values should be empty if the AIR has
    /// an empty trace.
    pub public_values: Vec<Vec<F>>,

    pub gkr_proof: GkrProof,
    pub batch_constraint_proof: BatchConstraintProof,
    pub stacking_proof: StackingProof,
    pub whir_proof: WhirProof,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize, Encode, Decode)]
pub struct TraceVData {
    /// The base 2 logarithm of the trace height. This should be a nonnegative integer and is
    /// allowed to be `< l_skip`.
    ///
    /// If the corresponding AIR has a preprocessed trace, this must match the
    /// value in the vkey.
    pub log_height: usize,
    /// The cached commitments used.
    ///
    /// The length must match the value in the vkey.
    pub cached_commitments: Vec<Digest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GkrProof {
    // TODO[jpw]: I'm not sure this is concepturally the place to put it, but recursion gkr module
    // samples alpha,beta
    pub logup_pow_witness: F,
    /// The denominator of the root layer.
    ///
    /// Note that the numerator claim is always zero, so we don't include it in
    /// the proof. Despite that the numerator is zero, the representation of the
    /// denominator is important for the verification procedure and thus must be
    /// provided.
    pub q0_claim: EF,
    /// The claims for p_j(xi, 0), p_j(xi, 1), q_j(xi, 0), and q_j(xi, 0) for each layer j > 0.
    pub claims_per_layer: Vec<GkrLayerClaims>,
    /// The sumcheck polynomials for each layer, for each sumcheck round, given by their
    /// evaluations on {1, 2, 3}.
    pub sumcheck_polys: Vec<Vec<[EF; 3]>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct GkrLayerClaims {
    pub p_xi_0: EF,
    pub p_xi_1: EF,
    pub q_xi_0: EF,
    pub q_xi_1: EF,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchConstraintProof {
    /// The terms \textnormal{sum}_{\hat{p}, T, I} as defined in Protocol 3.4.6, per present AIR
    /// **in sorted AIR order**.
    pub numerator_term_per_air: Vec<EF>,
    /// The terms \textnormal{sum}_{\hat{q}, T, I} as defined in Protocol 3.4.6, per present AIR
    /// **in sorted AIR order**.
    pub denominator_term_per_air: Vec<EF>,

    /// Polynomial for initial round, given by `(vk.d + 1) * (2^{l_skip} - 1) + 1` coefficients.
    pub univariate_round_coeffs: Vec<EF>,
    /// For rounds `1, ..., n_max`; evaluations on `{1, ..., vk.d + 1}`.
    pub sumcheck_round_polys: Vec<Vec<EF>>,

    /// Per AIR **in sorted AIR order**, per AIR part, per column index in that part, opening of
    /// the prismalinear column polynomial and its rotational convolution.
    /// The trace parts are ordered: [CommonMain (part
    /// 0), Preprocessed (if any), Cached(0), Cached(1), ...]
    pub column_openings: Vec<Vec<Vec<(EF, EF)>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackingProof {
    /// Polynomial for round 0, given by `2 * (2^{l_skip} - 1) + 1` coefficients.
    pub univariate_round_coeffs: Vec<EF>,
    /// Rounds 1, ..., n_stack; evaluations at {1, 2}.
    pub sumcheck_round_polys: Vec<[EF; 2]>,
    /// Per commit, per column.
    pub stacking_openings: Vec<Vec<EF>>,
}

pub type MerkleProof = Vec<Digest>;

/// WHIR polynomial opening proof for multiple polynomials of the same height, committed to in
/// multiple commitments.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhirProof {
    /// Per sumcheck round; evaluations on {1, 2}. This list is "flattened" with respect to the
    /// WHIR rounds.
    pub whir_sumcheck_polys: Vec<[EF; 2]>,
    /// The codeword commits after each fold, except the final round.
    pub codeword_commits: Vec<Digest>,
    /// The out-of-domain values "y0" per round, except the final round.
    pub ood_values: Vec<EF>,
    /// For each WHIR round, the PoW witness.
    pub whir_pow_witnesses: Vec<F>,
    /// For the initial round: per committed matrix, per in-domain query.
    // num_commits x num_queries x (1 << k) x stacking_width[i]
    pub initial_round_opened_rows: Vec<Vec<Vec<Vec<F>>>>,
    pub initial_round_merkle_proofs: Vec<Vec<MerkleProof>>,
    /// Per non-initial round, per in-domain-query.
    pub codeword_opened_values: Vec<Vec<Vec<EF>>>,
    pub codeword_merkle_proofs: Vec<Vec<MerkleProof>>,
    /// Coefficients of the polynomial after the final round.
    pub final_poly: Vec<EF>,
}

// ==================== Encode implementations ====================

/// Codec version should change only when proof system or proof format changes.
/// It does correspond to the main openvm version (which may change more frequently).
pub(crate) const CODEC_VERSION: u32 = 2;

// TODO: custom encode/decode for Proof that takes in a vk
impl Encode for Proof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // We explicitly implement Encode for Proof to add CODEC_VERSION
        CODEC_VERSION.encode(writer)?;
        self.common_main_commit.encode(writer)?;

        // We encode trace_vdata by encoding the number of AIRs, encoding a bitmap of
        // which AIRs are present, and then encoding each present TraceVData.
        let num_airs: usize = self.trace_vdata.len();
        num_airs.encode(writer)?;
        for chunk in self.trace_vdata.chunks(8) {
            let mut ret = 0u8;
            for (i, vdata) in chunk.iter().enumerate() {
                ret |= (vdata.is_some() as u8) << (i as u8);
            }
            ret.encode(writer)?;
        }
        for vdata in self.trace_vdata.iter().flatten() {
            vdata.encode(writer)?;
        }

        self.public_values.encode(writer)?;
        self.gkr_proof.encode(writer)?;
        self.batch_constraint_proof.encode(writer)?;
        self.stacking_proof.encode(writer)?;
        self.whir_proof.encode(writer)
    }
}

impl Encode for GkrProof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.logup_pow_witness.encode(writer)?;
        self.q0_claim.encode(writer)?;
        self.claims_per_layer.encode(writer)?;
        // We should know the length of sumcheck_polys and each nested vector based
        // on the length of claims_per_layer.
        encode_iter(self.sumcheck_polys.iter().flatten(), writer)?;
        Ok(())
    }
}

impl Encode for BatchConstraintProof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Length of numerator_term_per_air is number of present AIRs
        self.numerator_term_per_air.encode(writer)?;
        encode_iter(self.denominator_term_per_air.iter(), writer)?;

        self.univariate_round_coeffs.encode(writer)?;

        // Each nested vector should be the same length
        let n_max = self.sumcheck_round_polys.len();
        n_max.encode(writer)?;
        if n_max > 0 {
            self.sumcheck_round_polys[0].len().encode(writer)?;
            for round_polys in &self.sumcheck_round_polys {
                encode_iter(round_polys.iter(), writer)?;
            }
        }

        // There is one outer vector per present AIR
        for part_col_openings in &self.column_openings {
            part_col_openings.encode(writer)?;
        }
        Ok(())
    }
}

impl Encode for StackingProof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.univariate_round_coeffs.encode(writer)?;
        self.sumcheck_round_polys.encode(writer)?;
        self.stacking_openings.encode(writer)
    }
}

impl Encode for WhirProof {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.whir_sumcheck_polys.encode(writer)?;

        // Each length can be derived from num_whir_rounds
        self.codeword_commits.encode(writer)?;
        encode_iter(self.ood_values.iter(), writer)?;
        encode_iter(self.whir_pow_witnesses.iter(), writer)?;
        let num_whir_rounds = self.codeword_commits.len() + 1;

        let num_commits = self.initial_round_opened_rows.len();
        assert!(num_commits > 0);
        num_commits.encode(writer)?;
        let num_whir_queries = self.initial_round_opened_rows[0].len();
        num_whir_queries.encode(writer)?;
        if num_whir_queries > 0 {
            let k_whir_exp = self.initial_round_opened_rows[0][0].len();
            k_whir_exp.encode(writer)?;
            let merkle_depth = self.initial_round_merkle_proofs[0][0].len();
            merkle_depth.encode(writer)?;
            encode_iter(
                self.initial_round_opened_rows.iter().flatten().flatten(),
                writer,
            )?;
            encode_iter(
                self.initial_round_merkle_proofs.iter().flatten().flatten(),
                writer,
            )?;
        } else {
            let k_whir_exp = 0usize;
            k_whir_exp.encode(writer)?;
        }

        // Length of outer vector is num_whir_rounds, then num_whir_queries, then
        // k_whir_exp.
        encode_iter(
            self.codeword_opened_values.iter().flatten().flatten(),
            writer,
        )?;

        // Length of outer vector is num_whir_rounds, then num_whir_queries. Each
        // inner vector length is one less than the one that precedes it.
        let mut first_merkle_depth = 0;
        if num_whir_rounds > 1 && num_whir_queries > 0 {
            first_merkle_depth = self.codeword_merkle_proofs[0][0].len();
        }
        first_merkle_depth.encode(writer)?;
        encode_iter(
            self.codeword_merkle_proofs.iter().flatten().flatten(),
            writer,
        )?;

        self.final_poly.encode(writer)
    }
}

// ==================== Decode implementations ====================

impl Decode for Proof {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        // We explicitly implement Decode for Proof to check CODEC_VERSION
        let codec_version = u32::decode(reader)?;
        if codec_version != CODEC_VERSION {
            return Err(Error::other(format!(
                "CODEC_VERSION mismatch, expected: {}, actual: {}",
                CODEC_VERSION, codec_version
            )));
        }
        let common_main_commit = Digest::decode(reader)?;

        let num_airs = usize::decode(reader)?;
        let bitmap_len = num_airs.div_ceil(8);
        let mut bitmap: Vec<u8> = Vec::with_capacity(bitmap_len);
        for _ in 0..bitmap_len {
            bitmap.push(u8::decode(reader)?);
        }
        let mut trace_vdata = Vec::with_capacity(num_airs);
        for byte in bitmap {
            for i in 0u8..8 {
                if trace_vdata.len() >= num_airs {
                    break;
                }
                if byte & (1u8 << i) != 0 {
                    trace_vdata.push(Some(TraceVData::decode(reader)?));
                } else {
                    trace_vdata.push(None);
                }
            }
        }

        Ok(Self {
            common_main_commit,
            trace_vdata,
            public_values: Vec::<Vec<F>>::decode(reader)?,
            gkr_proof: GkrProof::decode(reader)?,
            batch_constraint_proof: BatchConstraintProof::decode(reader)?,
            stacking_proof: StackingProof::decode(reader)?,
            whir_proof: WhirProof::decode(reader)?,
        })
    }
}

impl Decode for GkrProof {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let logup_pow_witness = F::decode(reader)?;
        let q0_claim = EF::decode(reader)?;
        let claims_per_layer = Vec::<GkrLayerClaims>::decode(reader)?;

        let num_sumcheck_polys = claims_per_layer.len().saturating_sub(1);
        let mut sumcheck_polys = Vec::with_capacity(num_sumcheck_polys);
        for round_idx_minus_one in 0..num_sumcheck_polys {
            sumcheck_polys.push(decode_into_vec(reader, round_idx_minus_one + 1)?);
        }

        Ok(Self {
            logup_pow_witness,
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
        })
    }
}

impl Decode for BatchConstraintProof {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let numerator_term_per_air = Vec::<EF>::decode(reader)?;
        let num_present_airs = numerator_term_per_air.len();
        let denominator_term_per_air = decode_into_vec(reader, num_present_airs)?;

        let univariate_round_coeffs = Vec::<EF>::decode(reader)?;

        let n_max = usize::decode(reader)?;
        let mut sumcheck_round_polys = Vec::with_capacity(n_max);
        if n_max > 0 {
            let max_degree_plus_one = usize::decode(reader)?;
            for _ in 0..n_max {
                sumcheck_round_polys.push(decode_into_vec(reader, max_degree_plus_one)?);
            }
        }

        let mut column_openings = Vec::with_capacity(num_present_airs);
        for _ in 0..num_present_airs {
            column_openings.push(Vec::<Vec<(EF, EF)>>::decode(reader)?);
        }

        Ok(Self {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        })
    }
}

impl Decode for StackingProof {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            univariate_round_coeffs: Vec::<EF>::decode(reader)?,
            sumcheck_round_polys: Vec::<[EF; 2]>::decode(reader)?,
            stacking_openings: Vec::<Vec<EF>>::decode(reader)?,
        })
    }
}

impl Decode for WhirProof {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let whir_sumcheck_polys = Vec::<[EF; 2]>::decode(reader)?;
        let codeword_commits = Vec::<Digest>::decode(reader)?;
        let num_whir_rounds = codeword_commits.len() + 1;
        let ood_values = decode_into_vec(reader, num_whir_rounds - 1)?;
        let whir_pow_witnesses = decode_into_vec(reader, num_whir_rounds)?;

        let num_commits = usize::decode(reader)?;
        assert!(num_commits > 0);
        let num_whir_queries = usize::decode(reader)?;
        let k_whir_exp = usize::decode(reader)?;
        let mut merkle_depth = 0;
        if num_whir_queries > 0 {
            merkle_depth = usize::decode(reader)?;
        }

        let mut initial_round_opened_rows = Vec::with_capacity(num_commits);
        for _ in 0..num_commits {
            let mut opened_rows = Vec::with_capacity(num_whir_queries);
            for _ in 0..num_whir_queries {
                opened_rows.push(decode_into_vec(reader, k_whir_exp)?);
            }
            initial_round_opened_rows.push(opened_rows);
        }

        let mut initial_round_merkle_proofs = Vec::with_capacity(num_commits);
        for _ in 0..num_commits {
            let mut merkle_proofs = Vec::with_capacity(num_whir_queries);
            for _ in 0..num_whir_queries {
                merkle_proofs.push(decode_into_vec(reader, merkle_depth)?);
            }
            initial_round_merkle_proofs.push(merkle_proofs);
        }

        let mut codeword_opened_values = Vec::with_capacity(num_whir_rounds - 1);
        for _ in 0..num_whir_rounds - 1 {
            let mut opened_values = Vec::with_capacity(num_whir_queries);
            for _ in 0..num_whir_queries {
                opened_values.push(decode_into_vec(reader, k_whir_exp)?);
            }
            codeword_opened_values.push(opened_values);
        }

        merkle_depth = usize::decode(reader)?;
        let mut codeword_merkle_proofs = Vec::with_capacity(num_whir_rounds - 1);
        for _ in 0..num_whir_rounds - 1 {
            let mut merkle_proof: Vec<_> = Vec::with_capacity(num_whir_queries);
            for _ in 0..num_whir_queries {
                merkle_proof.push(decode_into_vec(reader, merkle_depth)?);
            }
            codeword_merkle_proofs.push(merkle_proof);
            merkle_depth -= 1;
        }

        let final_poly = Vec::<EF>::decode(reader)?;

        Ok(Self {
            whir_sumcheck_polys,
            codeword_commits,
            ood_values,
            whir_pow_witnesses,
            initial_round_opened_rows,
            initial_round_merkle_proofs,
            codeword_opened_values,
            codeword_merkle_proofs,
            final_poly,
        })
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::{
        poseidon2::sponge::DuplexSpongeRecorder,
        test_utils::{
            test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
            PreprocessedFibFixture, TestFixture,
        },
        BabyBearPoseidon2CpuEngineV2, SystemParams,
    };

    fn test_proof_encode_decode<Fx: TestFixture>(fx: Fx, params: SystemParams) -> Result<()> {
        let engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let pk = fx.keygen(&engine).0;
        let proof = fx.prove_from_transcript(&engine, &pk, &mut DuplexSpongeRecorder::default());

        let mut proof_bytes = Vec::new();
        proof.encode(&mut proof_bytes).unwrap();

        let decoded_proof = Proof::decode(&mut &proof_bytes[..]).unwrap();
        assert_eq!(proof, decoded_proof);
        Ok(())
    }

    #[test]
    fn test_fib_proof_encode_decode() -> Result<()> {
        let log_trace_height = 5;
        let fx = FibFixture::new(0, 1, 1 << log_trace_height);
        let params = SystemParams::new_for_testing(log_trace_height);
        test_proof_encode_decode(fx, params)
    }

    #[test]
    fn test_interactions_proof_encode_decode() -> Result<()> {
        let fx = InteractionsFixture11;
        let params = test_system_params_small(2, 5, 3);
        test_proof_encode_decode(fx, params)
    }

    #[test]
    fn test_cached_proof_encode_decode() -> Result<()> {
        let params = test_system_params_small(2, 5, 3);
        let fx = CachedFixture11::new(params.clone());
        test_proof_encode_decode(fx, params)
    }

    #[test]
    fn test_preprocessed_proof_encode_decode() -> Result<()> {
        let log_trace_height = 5;
        let params = SystemParams::new_for_testing(log_trace_height);
        let sels = (0..(1 << log_trace_height))
            .map(|i| i % 2 == 0)
            .collect_vec();
        let fx = PreprocessedFibFixture::new(0, 1, sels);
        test_proof_encode_decode(fx, params)
    }
}
