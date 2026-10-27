use std::io::{Error, Read, Result, Write};

use derivative::Derivative;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use crate::{
    codec::{DecodableConfig, Decode, EncodableConfig, Encode},
    decode, StarkProtocolConfig,
};

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct Proof<SC: StarkProtocolConfig> {
    /// The commitment to the data in common_main.
    pub common_main_commit: SC::Digest,

    /// For each AIR in vkey order, the corresponding trace shape, or None if
    /// the trace is empty. In a valid proof, if `vk.per_air[i].is_required`,
    /// then `trace_vdata[i]` must be `Some(_)`.
    pub trace_vdata: Vec<Option<TraceVData<SC>>>,

    /// For each AIR in vkey order, the public values. Public values should be empty if the AIR has
    /// an empty trace.
    pub public_values: Vec<Vec<SC::F>>,

    pub gkr_proof: GkrProof<SC>,
    pub batch_constraint_proof: BatchConstraintProof<SC>,
    pub stacking_proof: StackingProof<SC>,
    pub whir_proof: WhirProof<SC>,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = ""),
    Default(bound = "")
)]
#[serde(bound = "")]
pub struct TraceVData<SC: StarkProtocolConfig> {
    /// The base 2 logarithm of the trace height. This should be a nonnegative integer and is
    /// allowed to be `< l_skip`.
    ///
    /// If the corresponding AIR has a preprocessed trace, this must match the
    /// value in the vkey.
    pub log_height: usize,
    /// The cached commitments used.
    ///
    /// The length must match the value in the vkey.
    pub cached_commitments: Vec<SC::Digest>,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct GkrProof<SC: StarkProtocolConfig> {
    /// The grinding proof-of-work witness before sampling LogUp alpha,beta
    pub logup_pow_witness: SC::F,
    /// The denominator of the root layer.
    ///
    /// Note that the numerator claim is always zero, so we don't include it in
    /// the proof. Despite that the numerator is zero, the representation of the
    /// denominator is important for the verification procedure and thus must be
    /// provided.
    pub q0_claim: SC::EF,
    /// The claims for p_j(xi, 0), p_j(xi, 1), q_j(xi, 0), and q_j(xi, 0) for each layer j > 0.
    pub claims_per_layer: Vec<GkrLayerClaims<SC>>,
    /// The sumcheck polynomials for each layer, for each sumcheck round, given by their
    /// evaluations on {1, 2, 3}.
    pub sumcheck_polys: Vec<Vec<[SC::EF; 3]>>,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct GkrLayerClaims<SC: StarkProtocolConfig> {
    pub p_xi_0: SC::EF,
    pub p_xi_1: SC::EF,
    pub q_xi_0: SC::EF,
    pub q_xi_1: SC::EF,
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct BatchConstraintProof<SC: StarkProtocolConfig> {
    /// The terms \textnormal{sum}_{\hat{p}, T, I} as defined in Protocol 3.4.6, per present AIR
    /// **in sorted AIR order**.
    pub numerator_term_per_air: Vec<SC::EF>,
    /// The terms \textnormal{sum}_{\hat{q}, T, I} as defined in Protocol 3.4.6, per present AIR
    /// **in sorted AIR order**.
    pub denominator_term_per_air: Vec<SC::EF>,

    /// Polynomial for initial round, given by `(vk.d + 1) * (2^{l_skip} - 1) + 1` coefficients.
    pub univariate_round_coeffs: Vec<SC::EF>,
    /// For rounds `1, ..., n_max`; evaluations on `{1, ..., vk.d + 1}`.
    pub sumcheck_round_polys: Vec<Vec<SC::EF>>,

    /// Per AIR **in sorted AIR order**, per AIR part, per column index in that part, openings for
    /// the prismalinear column polynomial and (optionally) its rotational convolution. All column
    /// openings are stored in a flat way, so only column openings or them interleaved with
    /// rotations.
    /// For example, if the rotated claims are included for a trace part, then the corresponding
    /// list of openings will look like [col_1, rot_1, col_2, rot_2, ...], and should be treated
    /// as "the i-th column's plain and rotated claims are (col_i, rot_i)".
    /// Otherwise, it will look like [col_1, col_2, col_3, ...], and should be treated as "the
    /// i-th column's plain and rotated claims are (col_i, 0)".
    /// The trace parts are ordered: [CommonMain (part 0), Preprocessed (if any), Cached(0),
    /// Cached(1), ...]
    pub column_openings: Vec<Vec<Vec<SC::EF>>>,
}

pub fn column_openings_by_rot<'a, EF: PrimeCharacteristicRing + Copy + 'a>(
    openings: &'a [EF],
    need_rot: bool,
) -> Box<dyn Iterator<Item = (EF, EF)> + 'a> {
    if need_rot {
        Box::new(openings.chunks_exact(2).map(|chunk| (chunk[0], chunk[1])))
    } else {
        Box::new(openings.iter().map(|&claim| (claim, EF::ZERO)))
    }
}

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct StackingProof<SC: StarkProtocolConfig> {
    /// Polynomial for round 0, given by `2 * (2^{l_skip} - 1) + 1` coefficients.
    pub univariate_round_coeffs: Vec<SC::EF>,
    /// Rounds 1, ..., n_stack; evaluations at {1, 2}.
    pub sumcheck_round_polys: Vec<[SC::EF; 2]>,
    /// Per commit, per column.
    pub stacking_openings: Vec<Vec<SC::EF>>,
}

pub type MerkleProof<Digest> = Vec<Digest>;

/// WHIR polynomial opening proof for multiple polynomials of the same height, committed to in
/// multiple commitments.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
#[serde(bound = "")]
pub struct WhirProof<SC: StarkProtocolConfig> {
    /// Proof-of-work witness for μ batching challenge.
    pub mu_pow_witness: SC::F,
    /// Per sumcheck round; evaluations on {1, 2}. This list is "flattened" with respect to the
    /// WHIR rounds.
    pub whir_sumcheck_polys: Vec<[SC::EF; 2]>,
    /// The codeword commits after each fold, except the final round.
    pub codeword_commits: Vec<SC::Digest>,
    /// The out-of-domain values "y0" per round, except the final round.
    pub ood_values: Vec<SC::EF>,
    /// For each sumcheck round, the folding PoW witness. Length is `num_whir_sumcheck_rounds =
    /// num_whir_rounds * k_whir`.
    pub folding_pow_witnesses: Vec<SC::F>,
    /// For each WHIR round, the query phase PoW witness. Length is `num_whir_rounds`.
    pub query_phase_pow_witnesses: Vec<SC::F>,
    /// For the initial round: per committed matrix, per in-domain query.
    // num_commits x num_queries x (1 << k) x stacking_width[i]
    pub initial_round_opened_rows: Vec<Vec<Vec<Vec<SC::F>>>>,
    pub initial_round_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>>,
    /// Per non-initial round, per in-domain-query.
    pub codeword_opened_values: Vec<Vec<Vec<SC::EF>>>,
    pub codeword_merkle_proofs: Vec<Vec<MerkleProof<SC::Digest>>>,
    /// Coefficients of the polynomial after the final round.
    pub final_poly: Vec<SC::EF>,
}

// ==================== Encode implementations ====================

impl<SC: EncodableConfig> Encode for TraceVData<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        self.log_height.encode(writer)?;
        SC::encode_digest_slice(&self.cached_commitments, writer)
    }
}

impl<SC: EncodableConfig> Encode for GkrLayerClaims<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_extension_field(&self.p_xi_0, writer)?;
        SC::encode_extension_field(&self.p_xi_1, writer)?;
        SC::encode_extension_field(&self.q_xi_0, writer)?;
        SC::encode_extension_field(&self.q_xi_1, writer)?;
        Ok(())
    }
}

/// Codec version should change only when proof system or proof format changes.
/// The codec version **must** change if the verifying key serialization/deserialization format
/// changes. It does _not_ correspond to the main openvm version (which may change more frequently).
pub(crate) const CODEC_VERSION: u32 = 3;

impl<SC: EncodableConfig> Encode for Proof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // We explicitly implement Encode for Proof to add CODEC_VERSION
        CODEC_VERSION.encode(writer)?;
        SC::encode_digest(&self.common_main_commit, writer)?;

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

        // public_values: Vec<Vec<SC::F>>
        self.public_values.len().encode(writer)?;
        for pv in &self.public_values {
            SC::encode_base_field_slice(pv, writer)?;
        }
        self.gkr_proof.encode(writer)?;
        self.batch_constraint_proof.encode(writer)?;
        self.stacking_proof.encode(writer)?;
        self.whir_proof.encode(writer)
    }
}

impl<SC: EncodableConfig> Encode for GkrProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_base_field(&self.logup_pow_witness, writer)?;
        SC::encode_extension_field(&self.q0_claim, writer)?;
        self.claims_per_layer.encode(writer)?;
        // We should know the length of sumcheck_polys and each nested vector based
        // on the length of claims_per_layer.
        for round in &self.sumcheck_polys {
            for arr in round {
                SC::encode_extension_field_iter(arr.iter(), writer)?;
            }
        }
        Ok(())
    }
}

impl<SC: EncodableConfig> Encode for BatchConstraintProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Length of numerator_term_per_air is number of present AIRs
        SC::encode_extension_field_slice(&self.numerator_term_per_air, writer)?;
        SC::encode_extension_field_iter(self.denominator_term_per_air.iter(), writer)?;

        SC::encode_extension_field_slice(&self.univariate_round_coeffs, writer)?;

        // Each nested vector should be the same length
        let n_max = self.sumcheck_round_polys.len();
        n_max.encode(writer)?;
        if n_max > 0 {
            self.sumcheck_round_polys[0].len().encode(writer)?;
            for round_polys in &self.sumcheck_round_polys {
                SC::encode_extension_field_iter(round_polys.iter(), writer)?;
            }
        }

        // There is one outer vector per present AIR
        // column_openings: Vec<Vec<Vec<SC::EF>>>
        for part_col_openings in &self.column_openings {
            // part_col_openings: Vec<Vec<SC::EF>>
            part_col_openings.len().encode(writer)?;
            for col_opening in part_col_openings {
                SC::encode_extension_field_slice(col_opening, writer)?;
            }
        }
        Ok(())
    }
}

impl<SC: EncodableConfig> Encode for StackingProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_extension_field_slice(&self.univariate_round_coeffs, writer)?;
        // sumcheck_round_polys: Vec<[SC::EF; 2]>
        self.sumcheck_round_polys.len().encode(writer)?;
        for arr in &self.sumcheck_round_polys {
            SC::encode_extension_field_iter(arr.iter(), writer)?;
        }
        // stacking_openings: Vec<Vec<SC::EF>>
        self.stacking_openings.len().encode(writer)?;
        for opening in &self.stacking_openings {
            SC::encode_extension_field_slice(opening, writer)?;
        }
        Ok(())
    }
}

impl<SC: EncodableConfig> Encode for WhirProof<SC> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        SC::encode_base_field(&self.mu_pow_witness, writer)?;
        // whir_sumcheck_polys: Vec<[SC::EF; 2]>
        self.whir_sumcheck_polys.len().encode(writer)?;
        for arr in &self.whir_sumcheck_polys {
            SC::encode_extension_field_iter(arr.iter(), writer)?;
        }
        let num_whir_sumcheck_rounds = self.whir_sumcheck_polys.len();

        // Each length can be derived from num_whir_rounds
        SC::encode_digest_slice(&self.codeword_commits, writer)?;
        SC::encode_extension_field_iter(self.ood_values.iter(), writer)?;
        let num_whir_rounds = self.codeword_commits.len() + 1;
        if !num_whir_sumcheck_rounds.is_multiple_of(num_whir_rounds) {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "num_whir_sumcheck_rounds must be a multiple of num_whir_rounds",
            ));
        }
        assert_eq!(num_whir_rounds, self.query_phase_pow_witnesses.len());
        SC::encode_base_field_iter(self.folding_pow_witnesses.iter(), writer)?;
        SC::encode_base_field_iter(self.query_phase_pow_witnesses.iter(), writer)?;

        let num_commits = self.initial_round_opened_rows.len();
        assert!(num_commits > 0);
        num_commits.encode(writer)?;
        let initial_num_whir_queries = self.initial_round_opened_rows[0].len();
        initial_num_whir_queries.encode(writer)?;

        if initial_num_whir_queries > 0 {
            let merkle_depth = self.initial_round_merkle_proofs[0][0].len();
            merkle_depth.encode(writer)?;

            // We avoid per-row Vec length prefixes by encoding each commit's stacked width,
            // which we can use to determine the shapes of the remaining WHIR proof fields.
            let widths: Vec<usize> = self
                .initial_round_opened_rows
                .iter()
                .map(|commit_rows| {
                    // If there are any queries/rows, infer width from the first row.
                    commit_rows
                        .first()
                        .and_then(|q| q.first())
                        .map(|row| row.len())
                        .unwrap_or(0)
                })
                .collect();

            // Encode widths (length is implicit via num_commits).
            for w in &widths {
                w.encode(writer)?;
            }

            // Encode all opened row values (no per-row length prefixes).
            for (commit_rows, &width) in self.initial_round_opened_rows.iter().zip(&widths) {
                debug_assert_eq!(commit_rows.len(), initial_num_whir_queries);
                for query_rows in commit_rows {
                    for row in query_rows {
                        debug_assert_eq!(row.len(), width);
                        SC::encode_base_field_iter(row.iter(), writer)?;
                    }
                }
            }

            for merkle_proofs in &self.initial_round_merkle_proofs {
                for proof in merkle_proofs {
                    SC::encode_digest_iter(proof.iter(), writer)?;
                }
            }
        }

        // Length of outer vector is num_whir_rounds
        for non_init_round in &self.codeword_opened_values {
            let num_queries = non_init_round.len();
            num_queries.encode(writer)?;
            // Length of nested vector is num_whir_queries, then k_whir_exp.
            for query_vals in non_init_round {
                SC::encode_extension_field_iter(query_vals.iter(), writer)?;
            }
        }

        // Length of outer vector is num_whir_rounds, then num_whir_queries. Each
        // inner vector length is one less than the one that precedes it.
        let mut first_merkle_depth = 0;
        if num_whir_rounds > 1 && initial_num_whir_queries > 0 {
            first_merkle_depth = self.codeword_merkle_proofs[0][0].len();
        }
        first_merkle_depth.encode(writer)?;
        for round_proofs in &self.codeword_merkle_proofs {
            for proof in round_proofs {
                SC::encode_digest_iter(proof.iter(), writer)?;
            }
        }

        SC::encode_extension_field_slice(&self.final_poly, writer)
    }
}

// ==================== Decode implementations ====================

impl<SC: DecodableConfig> Decode for TraceVData<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        decode::decode_trace_vdata(reader, None)
    }
}

impl<SC: DecodableConfig> Decode for GkrLayerClaims<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        Ok(Self {
            p_xi_0: SC::decode_extension_field(reader)?,
            p_xi_1: SC::decode_extension_field(reader)?,
            q_xi_0: SC::decode_extension_field(reader)?,
            q_xi_1: SC::decode_extension_field(reader)?,
        })
    }
}

impl<SC: DecodableConfig> Decode for Proof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        let preamble = decode::decode_preamble(reader, None)?;

        Ok(Self {
            common_main_commit: preamble.common_main_commit,
            trace_vdata: preamble.trace_vdata,
            public_values: preamble.public_values,
            gkr_proof: GkrProof::decode(reader)?,
            batch_constraint_proof: BatchConstraintProof::decode(reader)?,
            stacking_proof: StackingProof::decode(reader)?,
            whir_proof: WhirProof::decode(reader)?,
        })
    }
}

impl<SC: DecodableConfig> Decode for GkrProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        decode::decode_gkr_proof(reader, None)
    }
}

impl<SC: DecodableConfig> Decode for BatchConstraintProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        decode::decode_batch_constraint_proof(reader, None)
    }
}

impl<SC: DecodableConfig> Decode for StackingProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        decode::decode_stacking_proof(reader, None)
    }
}

impl<SC: DecodableConfig> Decode for WhirProof<SC> {
    fn decode<R: Read>(reader: &mut R) -> Result<Self> {
        decode::decode_whir_proof(reader, None)
    }
}
