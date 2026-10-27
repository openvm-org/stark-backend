use std::io::{Error, ErrorKind, Read, Result};

use crate::{
    codec::{decode_into_vec, vec_with_capped_capacity, DecodableConfig, Decode},
    keygen::types::MultiStarkVerifyingKey0,
    proof::{
        BatchConstraintProof, GkrLayerClaims, GkrProof, Proof, StackingProof, TraceVData,
        WhirProof, CODEC_VERSION,
    },
    StarkProtocolConfig,
};

// ==================== Proof Shape Limits ====================

pub(crate) struct BatchLimits {
    num_airs: usize,
    univariate: usize,
    num_rounds: usize,
    round_width: usize,
    num_parts: usize,
    part_width: usize,
}

pub(crate) struct StackingLimits {
    univariate: usize,
    num_rounds: usize,
    num_commits: usize,
    max_total_width: usize,
}

pub(crate) struct WhirLimits {
    num_rounds: usize,
    num_sumcheck_rounds: usize,
    num_commits: usize,
    max_total_width: usize,
    queries: Vec<usize>,
    k_whir: usize,
    initial_depth: usize,
    final_poly: usize,
}

struct ProofLimits {
    gkr_rounds: usize,
    batch: BatchLimits,
    stacking: StackingLimits,
    whir: WhirLimits,
}

impl ProofLimits {
    fn from_vk<SC: StarkProtocolConfig>(mvk: &MultiStarkVerifyingKey0<SC>) -> Self {
        let params = &mvk.params;
        let per_air = &mvk.per_air;
        let skipped = (1 << params.l_skip) - 1;
        let round_width = mvk.max_constraint_degree() + 1;

        let num_commits = 1 + per_air.iter().map(|vk| vk.num_parts() - 1).sum::<usize>();
        let num_parts = per_air.iter().map(|vk| vk.num_parts()).max().unwrap_or(0);

        let part_width = per_air
            .iter()
            .flat_map(|vk| {
                let widths = &vk.params.width;
                std::iter::once(&widths.common_main)
                    .chain(&widths.preprocessed)
                    .chain(&widths.cached_mains)
                    .map(|width| width * if vk.params.need_rot { 2 } else { 1 })
            })
            .max()
            .unwrap_or(0);

        let interactions = per_air
            .iter()
            .map(|vk| vk.num_interactions())
            .sum::<usize>();
        let gkr_rounds = if interactions == 0 {
            0
        } else {
            params.log_stacked_height() + (usize::BITS - interactions.leading_zeros()) as usize
        };

        Self {
            gkr_rounds,
            batch: BatchLimits {
                num_airs: per_air.len(),
                univariate: round_width * skipped + 1,
                num_rounds: params.n_stack,
                round_width,
                num_parts,
                part_width,
            },
            stacking: StackingLimits {
                univariate: 2 * skipped + 1,
                num_rounds: params.n_stack,
                num_commits,
                max_total_width: params.w_stack,
            },
            whir: WhirLimits {
                num_rounds: params.num_whir_rounds(),
                num_sumcheck_rounds: params.num_whir_sumcheck_rounds(),
                num_commits,
                max_total_width: params.w_stack,
                queries: params
                    .whir
                    .rounds
                    .iter()
                    .map(|round| round.num_queries)
                    .collect(),
                k_whir: params.k_whir(),
                initial_depth: params.log_stacked_height() + params.log_blowup - params.k_whir(),
                final_poly: 1 << params.log_final_poly_len(),
            },
        }
    }
}

// ==================== Decode Implementation ====================

pub(crate) struct ProofPreamble<SC: StarkProtocolConfig> {
    pub common_main_commit: SC::Digest,
    pub trace_vdata: Vec<Option<TraceVData<SC>>>,
    pub public_values: Vec<Vec<SC::F>>,
}

pub(crate) fn decode_trace_vdata<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    cached_len: Option<usize>,
) -> Result<TraceVData<SC>> {
    let log_height = usize::decode(reader)?;
    let len = decode_len(reader, cached_len, "cached_commitments length")?;
    Ok(TraceVData {
        log_height,
        cached_commitments: SC::decode_digest_n(reader, len)?,
    })
}

pub(crate) fn decode_preamble<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    mvk: Option<&MultiStarkVerifyingKey0<SC>>,
) -> Result<ProofPreamble<SC>> {
    let codec_version = u32::decode(reader)?;
    if codec_version != CODEC_VERSION {
        return Err(Error::other(format!(
            "CODEC_VERSION mismatch, expected: {CODEC_VERSION}, actual: {codec_version}"
        )));
    }
    let common_main_commit = SC::decode_digest(reader)?;

    let expected_num_airs = mvk.map(|vk| vk.per_air.len());
    let num_airs = decode_len(reader, expected_num_airs, "trace_vdata length")?;
    let bitmap = decode_into_vec::<u8, _>(reader, num_airs.div_ceil(8))?;
    if !num_airs.is_multiple_of(8) && bitmap.last().unwrap() >> (num_airs % 8) != 0 {
        return Err(Error::other("trace_vdata bitmap padding bits set"));
    }
    let present = bitmap
        .iter()
        .flat_map(|byte| (0..8).map(move |bit| byte & (1 << bit) != 0))
        .take(num_airs);

    let mut trace_vdata = vec_with_capped_capacity(num_airs);
    for (i, present) in present.enumerate() {
        trace_vdata.push(
            present
                .then(|| decode_trace_vdata(reader, mvk.map(|vk| vk.per_air[i].num_cached_mains())))
                .transpose()?,
        );
    }

    let num_pvs = decode_len(reader, expected_num_airs, "public_values length")?;
    let mut public_values = vec_with_capped_capacity(num_pvs);
    let mut limit_pvs = mvk
        .into_iter()
        .flat_map(|vk| &vk.per_air)
        .map(|vk| vk.params.num_public_values);
    for _ in 0..num_pvs {
        let limit = limit_pvs.next();
        let len = decode_len(reader, limit, "public_values inner length")?;
        public_values.push(SC::decode_base_field_n(reader, len)?);
    }

    Ok(ProofPreamble {
        common_main_commit,
        trace_vdata,
        public_values,
    })
}

pub(crate) fn decode_gkr_proof<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    num_rounds: Option<usize>,
) -> Result<GkrProof<SC>> {
    let logup_pow_witness = SC::decode_base_field(reader)?;
    let q0_claim = SC::decode_extension_field(reader)?;
    let len = decode_len(reader, num_rounds, "claims_per_layer length")?;
    let claims_per_layer = decode_into_vec::<GkrLayerClaims<SC>, _>(reader, len)?;

    let mut sumcheck_polys = vec_with_capped_capacity(len.saturating_sub(1));
    for n in 1..len {
        sumcheck_polys.push(decode_extension_arrays::<SC, _, 3>(reader, n)?);
    }

    Ok(GkrProof {
        logup_pow_witness,
        q0_claim,
        claims_per_layer,
        sumcheck_polys,
    })
}

pub(crate) fn decode_batch_constraint_proof<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    limit: Option<&BatchLimits>,
) -> Result<BatchConstraintProof<SC>> {
    let numerator_term_per_air = decode_extension_vec::<SC, _>(
        reader,
        limit.map(|l| l.num_airs),
        "numerator_term_per_air length",
    )?;
    let num_present = numerator_term_per_air.len();
    let denominator_term_per_air = SC::decode_extension_field_n(reader, num_present)?;

    let univariate_round_coeffs = decode_extension_vec::<SC, _>(
        reader,
        limit.map(|l| l.univariate),
        "batch univariate_round_coeffs length",
    )?;

    let n_max = decode_len(
        reader,
        limit.map(|l| l.num_rounds),
        "batch sumcheck_round_polys length",
    )?;
    let mut sumcheck_round_polys = vec_with_capped_capacity(n_max);
    if n_max > 0 {
        let width = decode_len(
            reader,
            limit.map(|l| l.round_width),
            "batch sumcheck round width",
        )?;
        for _ in 0..n_max {
            sumcheck_round_polys.push(SC::decode_extension_field_n(reader, width)?);
        }
    }

    let mut column_openings = vec_with_capped_capacity(num_present);
    for _ in 0..num_present {
        let num_parts = decode_len(
            reader,
            limit.map(|l| l.num_parts),
            "column_openings parts length",
        )?;
        let mut parts = vec_with_capped_capacity(num_parts);
        for _ in 0..num_parts {
            parts.push(decode_extension_vec::<SC, _>(
                reader,
                limit.map(|l| l.part_width),
                "column_openings columns length",
            )?);
        }
        column_openings.push(parts);
    }

    Ok(BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
    })
}

pub(crate) fn decode_stacking_proof<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    limit: Option<&StackingLimits>,
) -> Result<StackingProof<SC>> {
    let univariate_round_coeffs = decode_extension_vec::<SC, _>(
        reader,
        limit.map(|l| l.univariate),
        "stacking univariate_round_coeffs length",
    )?;

    let num_rounds = decode_len(
        reader,
        limit.map(|l| l.num_rounds),
        "stacking sumcheck_round_polys length",
    )?;
    let sumcheck_round_polys = decode_extension_arrays::<SC, _, 2>(reader, num_rounds)?;

    let num_openings = decode_len(
        reader,
        limit.map(|l| l.num_commits),
        "stacking_openings length",
    )?;
    let mut stacking_openings = vec_with_capped_capacity(num_openings);
    let mut remaining = limit.map(|l| l.max_total_width);
    for _ in 0..num_openings {
        let width = decode_width(reader, &mut remaining, "stacking_openings width")?;
        stacking_openings.push(SC::decode_extension_field_n(reader, width)?);
    }

    Ok(StackingProof {
        univariate_round_coeffs,
        sumcheck_round_polys,
        stacking_openings,
    })
}

pub(crate) fn decode_whir_proof<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    limit: Option<&WhirLimits>,
) -> Result<WhirProof<SC>> {
    let mu_pow_witness = SC::decode_base_field(reader)?;
    let num_sumcheck_rounds = decode_len(
        reader,
        limit.map(|l| l.num_sumcheck_rounds),
        "whir_sumcheck_polys length",
    )?;
    let whir_sumcheck_polys = decode_extension_arrays::<SC, _, 2>(reader, num_sumcheck_rounds)?;

    let num_codeword_commits = decode_len(
        reader,
        limit.map(|l| l.num_rounds - 1),
        "codeword_commits length",
    )?;
    let codeword_commits = SC::decode_digest_n(reader, num_codeword_commits)?;
    let num_rounds = num_codeword_commits
        .checked_add(1)
        .ok_or_else(|| invalid("WHIR round count overflow"))?;
    if !num_sumcheck_rounds.is_multiple_of(num_rounds) {
        return Err(invalid(
            "num_whir_sumcheck_rounds must be a multiple of num_whir_rounds",
        ));
    }

    let k_whir = num_sumcheck_rounds / num_rounds;
    check_len(k_whir, limit.map(|l| l.k_whir), "WHIR folding factor")?;
    let ood_values = SC::decode_extension_field_n(reader, num_rounds - 1)?;
    let folding_pow_witnesses = SC::decode_base_field_n(reader, num_sumcheck_rounds)?;
    let query_phase_pow_witnesses = SC::decode_base_field_n(reader, num_rounds)?;

    let num_commits = decode_len(reader, limit.map(|l| l.num_commits), "WHIR num_commits")?;
    if num_commits == 0 {
        return Err(invalid("num_commits must be nonzero"));
    }

    let initial_queries = decode_len(
        reader,
        limit.map(|l| l.queries[0]),
        "WHIR initial query count",
    )?;
    let k_exp = u32::try_from(k_whir)
        .ok()
        .and_then(|k| 1usize.checked_shl(k))
        .ok_or_else(|| invalid("proof dimension overflow"))?;
    let mut initial_depth = 0;
    let mut widths = vec_with_capped_capacity(num_commits);
    if initial_queries > 0 {
        initial_depth = decode_len(
            reader,
            limit.map(|l| l.initial_depth),
            "WHIR initial Merkle depth",
        )?;
        let mut remaining = limit.map(|l| l.max_total_width);
        for _ in 0..num_commits {
            widths.push(decode_width(
                reader,
                &mut remaining,
                "WHIR initial opened row width",
            )?);
        }
    }

    let decoded_widths = widths.len();
    let mut initial_round_opened_rows = vec_with_capped_capacity(num_commits);
    for width in widths
        .into_iter()
        .chain(std::iter::repeat_n(0, num_commits - decoded_widths))
    {
        let mut queries = vec_with_capped_capacity(initial_queries);
        for _ in 0..initial_queries {
            let mut rows = vec_with_capped_capacity(k_exp);
            for _ in 0..k_exp {
                rows.push(SC::decode_base_field_n(reader, width)?);
            }
            queries.push(rows);
        }
        initial_round_opened_rows.push(queries);
    }

    let mut initial_round_merkle_proofs = vec_with_capped_capacity(num_commits);
    for _ in 0..num_commits {
        let mut queries = vec_with_capped_capacity(initial_queries);
        for _ in 0..initial_queries {
            queries.push(SC::decode_digest_n(reader, initial_depth)?);
        }
        initial_round_merkle_proofs.push(queries);
    }

    let mut codeword_opened_values = vec_with_capped_capacity(num_rounds - 1);
    for round in 1..num_rounds {
        let num_queries = decode_len(
            reader,
            limit.map(|l| l.queries[round]),
            "WHIR non-initial query count",
        )?;
        let mut queries = vec_with_capped_capacity(num_queries);
        for _ in 0..num_queries {
            queries.push(SC::decode_extension_field_n(reader, k_exp)?);
        }
        codeword_opened_values.push(queries);
    }

    let limit_depth = limit.map(|l| l.initial_depth.saturating_sub(1));
    let first_depth = decode_len(reader, limit_depth, "WHIR first non-initial Merkle depth")?;
    let mut codeword_merkle_proofs = vec_with_capped_capacity(num_rounds - 1);
    for (idx, opened) in codeword_opened_values.iter().enumerate() {
        let mut queries = vec_with_capped_capacity(opened.len());
        // Empty query lists may use the zero-depth sentinel.
        if !opened.is_empty() {
            let depth = first_depth
                .checked_sub(idx)
                .ok_or_else(|| invalid("WHIR Merkle depth underflow"))?;
            let limit_depth = limit.map(|l| l.initial_depth - idx - 1);
            check_len(depth, limit_depth, "WHIR non-initial Merkle depth")?;
            for _ in opened {
                queries.push(SC::decode_digest_n(reader, depth)?);
            }
        }
        codeword_merkle_proofs.push(queries);
    }
    let final_poly =
        decode_extension_vec::<SC, _>(reader, limit.map(|l| l.final_poly), "final_poly length")?;

    Ok(WhirProof {
        mu_pow_witness,
        whir_sumcheck_polys,
        codeword_commits,
        ood_values,
        folding_pow_witnesses,
        query_phase_pow_witnesses,
        initial_round_opened_rows,
        initial_round_merkle_proofs,
        codeword_opened_values,
        codeword_merkle_proofs,
        final_poly,
    })
}

impl<SC: DecodableConfig> Proof<SC> {
    /// Decode with conservative size limits from a trusted, valid verifying key. Note that
    /// exact proof shapes and cryptographic validity are checked by the verifier.
    pub fn decode_using_vk<R: Read>(
        mvk: &MultiStarkVerifyingKey0<SC>,
        reader: &mut R,
    ) -> Result<Self> {
        let preamble = decode_preamble(reader, Some(mvk))?;
        let limits = ProofLimits::from_vk(mvk);

        Ok(Self {
            common_main_commit: preamble.common_main_commit,
            trace_vdata: preamble.trace_vdata,
            public_values: preamble.public_values,
            gkr_proof: decode_gkr_proof(reader, Some(limits.gkr_rounds))?,
            batch_constraint_proof: decode_batch_constraint_proof(reader, Some(&limits.batch))?,
            stacking_proof: decode_stacking_proof(reader, Some(&limits.stacking))?,
            whir_proof: decode_whir_proof(reader, Some(&limits.whir))?,
        })
    }

    /// Decode a complete standard encoding using the VK, also rejecting trailing bytes.
    pub fn decode_from_bytes_using_vk(
        mvk: &MultiStarkVerifyingKey0<SC>,
        bytes: &[u8],
    ) -> Result<Self> {
        let mut reader = bytes;
        let proof = Self::decode_using_vk(mvk, &mut reader)?;
        if !reader.is_empty() {
            return Err(Error::other("trailing bytes after decoded value"));
        }

        Ok(proof)
    }
}

// ==================== Decode Helpers ====================

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

fn check_len(actual: usize, limit: Option<usize>, field: &str) -> Result<()> {
    if let Some(limit) = limit {
        if actual > limit {
            return Err(invalid(format!(
                "{field}: {actual} exceeds maximum {limit}"
            )));
        }
    }
    Ok(())
}

fn decode_len<R: Read>(reader: &mut R, limit: Option<usize>, field: &str) -> Result<usize> {
    let actual = usize::decode(reader)?;
    check_len(actual, limit, field)?;
    Ok(actual)
}

fn decode_width<R: Read>(
    reader: &mut R,
    remaining: &mut Option<usize>,
    field: &str,
) -> Result<usize> {
    let width = usize::decode(reader)?;
    if let Some(remaining) = remaining {
        *remaining = remaining
            .checked_sub(width)
            .ok_or_else(|| invalid(format!("{field}: total width exceeds w_stack")))?;
    }
    Ok(width)
}

fn decode_extension_vec<SC: DecodableConfig, R: Read>(
    reader: &mut R,
    limit: Option<usize>,
    field: &str,
) -> Result<Vec<SC::EF>> {
    let len = decode_len(reader, limit, field)?;
    SC::decode_extension_field_n(reader, len)
}

fn decode_extension_arrays<SC: DecodableConfig, R: Read, const N: usize>(
    reader: &mut R,
    len: usize,
) -> Result<Vec<[SC::EF; N]>> {
    let mut result = vec_with_capped_capacity(len);
    for _ in 0..len {
        let mut values = [SC::EF::default(); N];
        for value in &mut values {
            *value = SC::decode_extension_field(reader)?;
        }
        result.push(values);
    }
    Ok(result)
}
