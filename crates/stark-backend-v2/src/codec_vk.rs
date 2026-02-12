//! VK-shaped encoding/decoding for [`crate::proof::Proof`].
//!
//! This module provides an alternative codec that omits many length prefixes by
//! deriving shapes from the verifying key and the trace-vdata bitmap.

use std::{
    cmp::max,
    io::{Error, ErrorKind, Read, Result, Write},
};

use crate::{
    calculate_n_logup,
    codec::{decode_into_vec, encode_iter, Decode, Encode},
    keygen::types::{MultiStarkVerifyingKey0V2, StarkVerifyingKeyV2},
    prover::stacked_pcs::StackedLayout,
    Digest, EF, F,
};

use crate::proof::{
    BatchConstraintProof, GkrLayerClaims, GkrProof, Proof, StackingProof, TraceVData, WhirProof,
};

/// VK-based codec version.
///
/// This codec is different from [`Encode`] / [`Decode`] for [`Proof`]: it omits
/// length prefixes for many collections whose shapes can be derived from the
/// verifying key and the trace vdata bitmap.
pub(crate) const CODEC_VERSION_VK: u32 = 0;

fn per_trace_sorted<'a>(
    mvk: &'a MultiStarkVerifyingKey0V2,
    trace_vdata: &'a [Option<TraceVData>],
) -> Result<
    Vec<(
        usize,
        &'a crate::keygen::types::StarkVerifyingKeyV2<F, Digest>,
        &'a TraceVData,
    )>,
> {
    use std::cmp::Reverse;

    if trace_vdata.len() != mvk.per_air.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "trace_vdata length does not match mvk.per_air length",
        ));
    }

    let mut per_trace = mvk
        .per_air
        .iter()
        .zip(trace_vdata)
        .enumerate()
        .filter_map(|(air_idx, (vk, vdata))| vdata.as_ref().map(|vdata| (air_idx, vk, vdata)))
        .collect::<Vec<_>>();

    per_trace.sort_by_key(|(air_idx, _, vdata)| (Reverse(vdata.log_height), *air_idx));
    Ok(per_trace)
}

fn compute_layouts(
    mvk: &MultiStarkVerifyingKey0V2,
    per_trace: &[(usize, &StarkVerifyingKeyV2<F, Digest>, &TraceVData)],
) -> Vec<StackedLayout> {
    let l_skip = mvk.params.l_skip;
    let log_stacked_height = mvk.params.n_stack + l_skip;

    let common_main_layout = StackedLayout::new(
        l_skip,
        log_stacked_height,
        per_trace
            .iter()
            .map(|(_, vk, vdata)| (vk.params.width.common_main, vdata.log_height))
            .collect(),
    );

    let mut other_layouts = vec![];
    for (_, vk, vdata) in per_trace {
        if let Some(preprocessed_width) = vk.params.width.preprocessed {
            other_layouts.push(StackedLayout::new(
                l_skip,
                log_stacked_height,
                vec![(preprocessed_width, vdata.log_height)],
            ));
        }
        for &width in &vk.params.width.cached_mains {
            other_layouts.push(StackedLayout::new(
                l_skip,
                log_stacked_height,
                vec![(width, vdata.log_height)],
            ));
        }
    }

    [common_main_layout]
        .into_iter()
        .chain(other_layouts)
        .collect()
}

fn compute_num_gkr_rounds(
    mvk: &MultiStarkVerifyingKey0V2,
    per_trace: &[(usize, &StarkVerifyingKeyV2<F, Digest>, &TraceVData)],
) -> usize {
    let l_skip = mvk.params.l_skip;
    let total_interactions = per_trace.iter().fold(0u64, |acc, (_, vk, vdata)| {
        acc + ((vk.num_interactions() as u64) << max(vdata.log_height, l_skip))
    });
    if total_interactions == 0 {
        0
    } else {
        l_skip + calculate_n_logup(l_skip, total_interactions)
    }
}

fn decode_exact_vec<T: Decode, R: Read>(reader: &mut R, len: usize) -> Result<Vec<T>> {
    decode_into_vec(reader, len)
}

fn encode_exact_iter<'a, T: Encode + 'a, W: Write>(
    iter: impl Iterator<Item = &'a T>,
    writer: &mut W,
) -> Result<()> {
    encode_iter(iter, writer)
}

impl Proof {
    /// Encode this proof using `mvk` to derive shapes (omitting many lengths).
    pub fn encode_to_bytes_using_vk(&self, mvk: &MultiStarkVerifyingKey0V2) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.encode_using_vk(mvk, &mut out)?;
        Ok(out)
    }

    /// Decode a proof using `mvk` to derive shapes (omitting many lengths).
    pub fn decode_from_bytes_using_vk(
        mvk: &MultiStarkVerifyingKey0V2,
        bytes: &[u8],
    ) -> Result<Self> {
        let mut reader = bytes;
        let proof = Self::decode_using_vk(mvk, &mut reader)?;
        if !reader.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "trailing bytes after Proof::decode_using_vk",
            ));
        }
        Ok(proof)
    }

    pub fn encode_using_vk<W: Write>(
        &self,
        mvk: &MultiStarkVerifyingKey0V2,
        writer: &mut W,
    ) -> Result<()> {
        // Header
        CODEC_VERSION_VK.encode(writer)?;
        self.common_main_commit.encode(writer)?;

        // TRACE VDATA BITMAP (fixed size from mvk)
        let num_airs = mvk.per_air.len();
        if self.trace_vdata.len() != num_airs || self.public_values.len() != num_airs {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Proof AIR vector lengths do not match mvk.per_air length",
            ));
        }
        for chunk in self.trace_vdata.chunks(8) {
            let mut ret = 0u8;
            for (i, vdata) in chunk.iter().enumerate() {
                ret |= (vdata.is_some() as u8) << (i as u8);
            }
            ret.encode(writer)?;
        }

        // TRACE VDATA (no Vec lengths: cached commitments count is from vk)
        for (air_idx, (vk, vdata_opt)) in mvk.per_air.iter().zip(&self.trace_vdata).enumerate() {
            let Some(vdata) = vdata_opt.as_ref() else {
                continue;
            };
            vdata.log_height.encode(writer)?;
            if vdata.cached_commitments.len() != vk.num_cached_mains() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "AIR {air_idx} cached_commitments len {} != expected {}",
                        vdata.cached_commitments.len(),
                        vk.num_cached_mains()
                    ),
                ));
            }
            encode_exact_iter(vdata.cached_commitments.iter(), writer)?;
        }

        // PUBLIC VALUES (outer/inner lengths derived from mvk + bitmap)
        for (air_idx, (vk, (vdata_opt, pvs))) in mvk
            .per_air
            .iter()
            .zip(self.trace_vdata.iter().zip(&self.public_values))
            .enumerate()
        {
            if vdata_opt.is_none() {
                if !pvs.is_empty() {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("AIR {air_idx} has no vdata but non-empty public values"),
                    ));
                }
                continue;
            }
            if pvs.len() != vk.params.num_public_values {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "AIR {air_idx} public values len {} != expected {}",
                        pvs.len(),
                        vk.params.num_public_values
                    ),
                ));
            }
            encode_exact_iter(pvs.iter(), writer)?;
        }

        // Compute per-trace ordering / layouts (also sanity checks)
        let per_trace = per_trace_sorted(mvk, &self.trace_vdata)?;
        if per_trace.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "cannot encode vk-based proof with zero present AIRs",
            ));
        }
        let layouts = compute_layouts(mvk, &per_trace);
        let num_present_airs = per_trace.len();

        // GKR PROOF (omit Vec lengths; shape derived from mvk + vdata)
        let num_gkr_rounds = compute_num_gkr_rounds(mvk, &per_trace);
        self.gkr_proof.logup_pow_witness.encode(writer)?;
        self.gkr_proof.q0_claim.encode(writer)?;
        if self.gkr_proof.claims_per_layer.len() != num_gkr_rounds {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "gkr_proof.claims_per_layer has wrong length",
            ));
        }
        encode_exact_iter(self.gkr_proof.claims_per_layer.iter(), writer)?;

        if self.gkr_proof.sumcheck_polys.len() != num_gkr_rounds.saturating_sub(1) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "gkr_proof.sumcheck_polys has wrong length",
            ));
        }
        for (round_idx_minus_one, poly) in self.gkr_proof.sumcheck_polys.iter().enumerate() {
            let expected = round_idx_minus_one + 1;
            if poly.len() != expected {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "gkr_proof.sumcheck_polys round has wrong length",
                ));
            }
            encode_exact_iter(poly.iter(), writer)?;
        }

        // BATCH CONSTRAINTS PROOF (omit Vec lengths; shape derived from mvk + vdata)
        let batch = &self.batch_constraint_proof;
        if batch.numerator_term_per_air.len() != num_present_airs
            || batch.denominator_term_per_air.len() != num_present_airs
            || batch.column_openings.len() != num_present_airs
        {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "batch_constraint_proof AIR-sorted vectors have wrong length",
            ));
        }
        encode_exact_iter(batch.numerator_term_per_air.iter(), writer)?;
        encode_exact_iter(batch.denominator_term_per_air.iter(), writer)?;

        let l_skip = mvk.params.l_skip;
        let univariate_expected = (mvk.max_constraint_degree() + 1) * ((1 << l_skip) - 1) + 1;
        if batch.univariate_round_coeffs.len() != univariate_expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "batch_constraint_proof.univariate_round_coeffs has wrong length",
            ));
        }
        encode_exact_iter(batch.univariate_round_coeffs.iter(), writer)?;

        let n_max = per_trace[0].2.log_height.saturating_sub(l_skip);
        if batch.sumcheck_round_polys.len() != n_max {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "batch_constraint_proof.sumcheck_round_polys has wrong length",
            ));
        }
        let round_poly_len = mvk.max_constraint_degree() + 1;
        for round in &batch.sumcheck_round_polys {
            if round.len() != round_poly_len {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "batch_constraint_proof.sumcheck_round_polys round has wrong length",
                ));
            }
            encode_exact_iter(round.iter(), writer)?;
        }

        for ((air_idx, vk, _), part_openings) in per_trace.iter().zip(&batch.column_openings) {
            if part_openings.len() != vk.num_parts() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("AIR {air_idx} column_openings has wrong number of parts"),
                ));
            }

            let openings_per_col = if mvk.per_air[*air_idx].params.need_rot {
                2
            } else {
                1
            };

            // Part 0: common main
            if part_openings[0].len() != vk.params.width.common_main * openings_per_col {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("AIR {air_idx} common main openings wrong width"),
                ));
            }
            encode_exact_iter(part_openings[0].iter(), writer)?;

            let mut part_idx = 1;
            // Optional preprocessed
            if let Some(pre_w) = vk.params.width.preprocessed {
                if part_openings[part_idx].len() != pre_w * openings_per_col {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("AIR {air_idx} preprocessed openings wrong width"),
                    ));
                }
                encode_exact_iter(part_openings[part_idx].iter(), writer)?;
                part_idx += 1;
            }
            // Cached mains
            for (cached_idx, &cached_w) in vk.params.width.cached_mains.iter().enumerate() {
                if part_openings[part_idx].len() != cached_w * openings_per_col {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("AIR {air_idx} cached {cached_idx} openings wrong width"),
                    ));
                }
                encode_exact_iter(part_openings[part_idx].iter(), writer)?;
                part_idx += 1;
            }
        }

        // STACKING PROOF (omit Vec lengths; shape derived from mvk + vdata)
        let stacking = &self.stacking_proof;
        let stacking_univariate_expected = 2 * ((1 << l_skip) - 1) + 1;
        if stacking.univariate_round_coeffs.len() != stacking_univariate_expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "stacking_proof.univariate_round_coeffs has wrong length",
            ));
        }
        encode_exact_iter(stacking.univariate_round_coeffs.iter(), writer)?;

        if stacking.sumcheck_round_polys.len() != mvk.params.n_stack {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "stacking_proof.sumcheck_round_polys has wrong length",
            ));
        }
        encode_exact_iter(stacking.sumcheck_round_polys.iter(), writer)?;

        if stacking.stacking_openings.len() != layouts.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "stacking_proof.stacking_openings has wrong number of commits",
            ));
        }
        for (commit_idx, (openings, layout)) in
            stacking.stacking_openings.iter().zip(&layouts).enumerate()
        {
            let expected = layout.width();
            if openings.len() != expected {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "stacking_proof.stacking_openings[{commit_idx}] len {} != expected {expected}",
                        openings.len()
                    ),
                ));
            }
            encode_exact_iter(openings.iter(), writer)?;
        }

        // WHIR PROOF (omit Vec lengths; shape derived from params + stacking widths)
        let whir = &self.whir_proof;
        let num_whir_rounds = mvk.params.num_whir_rounds();
        let num_whir_sumcheck_rounds = mvk.params.num_whir_sumcheck_rounds();
        let k_whir = mvk.params.k_whir();
        let k_whir_exp = 1usize << k_whir;
        let log_stacked_height = mvk.params.log_stacked_height();

        if whir.whir_sumcheck_polys.len() != num_whir_sumcheck_rounds
            || whir.codeword_commits.len() != num_whir_rounds - 1
            || whir.ood_values.len() != num_whir_rounds - 1
            || whir.folding_pow_witnesses.len() != num_whir_sumcheck_rounds
            || whir.query_phase_pow_witnesses.len() != num_whir_rounds
            || whir.initial_round_opened_rows.len() != layouts.len()
            || whir.initial_round_merkle_proofs.len() != layouts.len()
            || whir.codeword_opened_values.len() != num_whir_rounds - 1
            || whir.codeword_merkle_proofs.len() != num_whir_rounds - 1
            || whir.final_poly.len() != (1usize << mvk.params.log_final_poly_len())
        {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "whir_proof has wrong top-level lengths",
            ));
        }

        encode_exact_iter(whir.whir_sumcheck_polys.iter(), writer)?;
        encode_exact_iter(whir.codeword_commits.iter(), writer)?;
        encode_exact_iter(whir.ood_values.iter(), writer)?;
        encode_exact_iter(whir.folding_pow_witnesses.iter(), writer)?;
        encode_exact_iter(whir.query_phase_pow_witnesses.iter(), writer)?;

        let initial_num_queries = mvk.params.whir.rounds[0].num_queries;
        let merkle_depth0 = (log_stacked_height + mvk.params.log_blowup).saturating_sub(k_whir);

        for (commit_idx, rows_per_commit) in whir.initial_round_opened_rows.iter().enumerate() {
            let width = stacking.stacking_openings[commit_idx].len();
            if rows_per_commit.len() != initial_num_queries {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "whir_proof.initial_round_opened_rows wrong query count",
                ));
            }
            for rows_per_query in rows_per_commit {
                if rows_per_query.len() != k_whir_exp {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "whir_proof.initial_round_opened_rows wrong rows-per-query",
                    ));
                }
                for row in rows_per_query {
                    if row.len() != width {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            "whir_proof.initial_round_opened_rows wrong row width",
                        ));
                    }
                    encode_exact_iter(row.iter(), writer)?;
                }
            }
        }

        for proofs_per_commit in whir.initial_round_merkle_proofs.iter() {
            if proofs_per_commit.len() != initial_num_queries {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "whir_proof.initial_round_merkle_proofs wrong query count",
                ));
            }
            for proof in proofs_per_commit {
                if proof.len() != merkle_depth0 {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "whir_proof.initial_round_merkle_proofs wrong depth",
                    ));
                }
                encode_exact_iter(proof.iter(), writer)?;
            }
        }

        for round in 1..num_whir_rounds {
            let round_minus_one = round - 1;
            let num_queries = mvk.params.whir.rounds[round].num_queries;
            let opened_values_per_query = &whir.codeword_opened_values[round_minus_one];
            let merkle_proofs = &whir.codeword_merkle_proofs[round_minus_one];
            let merkle_depth =
                (log_stacked_height + mvk.params.log_blowup).saturating_sub(k_whir + round);

            if opened_values_per_query.len() != num_queries || merkle_proofs.len() != num_queries {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "whir_proof non-initial round wrong query count",
                ));
            }
            for opened_values in opened_values_per_query {
                if opened_values.len() != k_whir_exp {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "whir_proof non-initial round opened values wrong length",
                    ));
                }
                encode_exact_iter(opened_values.iter(), writer)?;
            }
            for proof in merkle_proofs {
                if proof.len() != merkle_depth {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "whir_proof non-initial round merkle proof wrong depth",
                    ));
                }
                encode_exact_iter(proof.iter(), writer)?;
            }
        }

        encode_exact_iter(whir.final_poly.iter(), writer)?;
        Ok(())
    }

    pub fn decode_using_vk<R: Read>(
        mvk: &MultiStarkVerifyingKey0V2,
        reader: &mut R,
    ) -> Result<Self> {
        let codec_version = u32::decode(reader)?;
        if codec_version != CODEC_VERSION_VK {
            return Err(Error::other(format!(
                "CODEC_VERSION_VK mismatch, expected: {}, actual: {}",
                CODEC_VERSION_VK, codec_version
            )));
        }

        let common_main_commit = Digest::decode(reader)?;

        // TRACE VDATA BITMAP
        let num_airs = mvk.per_air.len();
        let bitmap_len = num_airs.div_ceil(8);
        let mut bitmap = Vec::with_capacity(bitmap_len);
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
                    // TraceVData without cached_commitments length prefix
                    let log_height = usize::decode(reader)?;
                    let cached_len = mvk.per_air[trace_vdata.len()].num_cached_mains();
                    let cached_commitments = decode_exact_vec::<Digest, _>(reader, cached_len)?;
                    trace_vdata.push(Some(TraceVData {
                        log_height,
                        cached_commitments,
                    }));
                } else {
                    trace_vdata.push(None);
                }
            }
        }

        // PUBLIC VALUES (no lengths)
        let mut public_values = Vec::with_capacity(num_airs);
        for (vk, vdata_opt) in mvk.per_air.iter().zip(&trace_vdata) {
            if vdata_opt.is_none() {
                public_values.push(vec![]);
                continue;
            }
            let pvs = decode_exact_vec::<F, _>(reader, vk.params.num_public_values)?;
            public_values.push(pvs);
        }

        // Per-trace ordering / layouts
        let per_trace = per_trace_sorted(mvk, &trace_vdata)?;
        if per_trace.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "cannot decode vk-based proof with zero present AIRs",
            ));
        }
        let layouts = compute_layouts(mvk, &per_trace);
        let num_present_airs = per_trace.len();

        // GKR PROOF (no lengths)
        let num_gkr_rounds = compute_num_gkr_rounds(mvk, &per_trace);
        let logup_pow_witness = F::decode(reader)?;
        let q0_claim = EF::decode(reader)?;
        let claims_per_layer = decode_exact_vec::<GkrLayerClaims, _>(reader, num_gkr_rounds)?;
        let mut sumcheck_polys = Vec::with_capacity(num_gkr_rounds.saturating_sub(1));
        for round_idx_minus_one in 0..num_gkr_rounds.saturating_sub(1) {
            let poly = decode_exact_vec::<[EF; 3], _>(reader, round_idx_minus_one + 1)?;
            sumcheck_polys.push(poly);
        }
        let gkr_proof = GkrProof {
            logup_pow_witness,
            q0_claim,
            claims_per_layer,
            sumcheck_polys,
        };

        // BATCH CONSTRAINTS PROOF (no lengths)
        let numerator_term_per_air = decode_exact_vec::<EF, _>(reader, num_present_airs)?;
        let denominator_term_per_air = decode_exact_vec::<EF, _>(reader, num_present_airs)?;

        let l_skip = mvk.params.l_skip;
        let univariate_len = (mvk.max_constraint_degree() + 1) * ((1 << l_skip) - 1) + 1;
        let univariate_round_coeffs = decode_exact_vec::<EF, _>(reader, univariate_len)?;

        let n_max = per_trace[0].2.log_height.saturating_sub(l_skip);
        let round_poly_len = mvk.max_constraint_degree() + 1;
        let mut sumcheck_round_polys = Vec::with_capacity(n_max);
        for _ in 0..n_max {
            sumcheck_round_polys.push(decode_exact_vec::<EF, _>(reader, round_poly_len)?);
        }

        let mut column_openings = Vec::with_capacity(num_present_airs);
        for (air_idx, vk, _) in &per_trace {
            let mut per_part = Vec::with_capacity(vk.num_parts());
            let openings_per_col = if mvk.per_air[*air_idx].params.need_rot {
                2
            } else {
                1
            };

            // Common main
            let main_len = vk.params.width.common_main * openings_per_col;
            let main = decode_exact_vec::<EF, _>(reader, main_len)?;
            per_part.push(main);

            // Preprocessed (optional)
            if let Some(pre_w) = vk.params.width.preprocessed {
                let pre_len = pre_w * openings_per_col;
                let pre = decode_exact_vec::<EF, _>(reader, pre_len)?;
                per_part.push(pre);
            }

            // Cached
            for &cached_w in &vk.params.width.cached_mains {
                let cached_len = cached_w * openings_per_col;
                let cached = decode_exact_vec::<EF, _>(reader, cached_len)?;
                per_part.push(cached);
            }

            if per_part.len() != vk.num_parts() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("AIR {air_idx} decoded wrong number of parts"),
                ));
            }
            column_openings.push(per_part);
        }

        let batch_constraint_proof = BatchConstraintProof {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        };

        // STACKING PROOF (no lengths)
        let stacking_univariate_len = 2 * ((1 << l_skip) - 1) + 1;
        let stacking_univariate_round_coeffs =
            decode_exact_vec::<EF, _>(reader, stacking_univariate_len)?;
        let sumcheck_round_polys = decode_exact_vec::<[EF; 2], _>(reader, mvk.params.n_stack)?;

        let mut stacking_openings = Vec::with_capacity(layouts.len());
        for layout in &layouts {
            stacking_openings.push(decode_exact_vec::<EF, _>(reader, layout.width())?);
        }

        let stacking_proof = StackingProof {
            univariate_round_coeffs: stacking_univariate_round_coeffs,
            sumcheck_round_polys,
            stacking_openings,
        };

        // WHIR PROOF (no lengths)
        let num_whir_rounds = mvk.params.num_whir_rounds();
        let num_whir_sumcheck_rounds = mvk.params.num_whir_sumcheck_rounds();
        let k_whir = mvk.params.k_whir();
        let k_whir_exp = 1usize << k_whir;
        let log_stacked_height = mvk.params.log_stacked_height();

        let whir_sumcheck_polys = decode_exact_vec::<[EF; 2], _>(reader, num_whir_sumcheck_rounds)?;
        let codeword_commits =
            decode_exact_vec::<Digest, _>(reader, num_whir_rounds.saturating_sub(1))?;
        let ood_values = decode_exact_vec::<EF, _>(reader, num_whir_rounds.saturating_sub(1))?;
        let folding_pow_witnesses = decode_exact_vec::<F, _>(reader, num_whir_sumcheck_rounds)?;
        let query_phase_pow_witnesses = decode_exact_vec::<F, _>(reader, num_whir_rounds)?;

        let initial_num_queries = mvk.params.whir.rounds[0].num_queries;
        let merkle_depth0 = (log_stacked_height + mvk.params.log_blowup).saturating_sub(k_whir);

        let mut initial_round_opened_rows = Vec::with_capacity(layouts.len());
        for width in stacking_proof.stacking_openings.iter().map(|v| v.len()) {
            let mut per_query = Vec::with_capacity(initial_num_queries);
            for _ in 0..initial_num_queries {
                let mut rows = Vec::with_capacity(k_whir_exp);
                for _ in 0..k_whir_exp {
                    rows.push(decode_exact_vec::<F, _>(reader, width)?);
                }
                per_query.push(rows);
            }
            initial_round_opened_rows.push(per_query);
        }

        let mut initial_round_merkle_proofs = Vec::with_capacity(layouts.len());
        for _ in 0..layouts.len() {
            let mut per_query = Vec::with_capacity(initial_num_queries);
            for _ in 0..initial_num_queries {
                per_query.push(decode_exact_vec::<Digest, _>(reader, merkle_depth0)?);
            }
            initial_round_merkle_proofs.push(per_query);
        }

        let mut codeword_opened_values = Vec::with_capacity(num_whir_rounds.saturating_sub(1));
        let mut codeword_merkle_proofs = Vec::with_capacity(num_whir_rounds.saturating_sub(1));
        for round in 1..num_whir_rounds {
            let num_queries = mvk.params.whir.rounds[round].num_queries;
            let merkle_depth =
                (log_stacked_height + mvk.params.log_blowup).saturating_sub(k_whir + round);

            let mut opened = Vec::with_capacity(num_queries);
            for _ in 0..num_queries {
                opened.push(decode_exact_vec::<EF, _>(reader, k_whir_exp)?);
            }
            codeword_opened_values.push(opened);

            let mut proofs = Vec::with_capacity(num_queries);
            for _ in 0..num_queries {
                proofs.push(decode_exact_vec::<Digest, _>(reader, merkle_depth)?);
            }
            codeword_merkle_proofs.push(proofs);
        }

        let final_poly =
            decode_exact_vec::<EF, _>(reader, 1usize << mvk.params.log_final_poly_len())?;

        let whir_proof = WhirProof {
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
        };

        Ok(Self {
            common_main_commit,
            trace_vdata,
            public_values,
            gkr_proof,
            batch_constraint_proof,
            stacking_proof,
            whir_proof,
        })
    }
}
