use crate::{config::SystemParams, keygen::types::MultiStarkVerifyingKey0V2, Digest, EF, F};

/// Length prefixes are encoded as u32, not usize
const LEN_SIZE: usize = size_of::<u32>();
const F_SIZE: usize = size_of::<F>();
const EF_SIZE: usize = size_of::<EF>();
pub const DIGEST_SIZE: usize = size_of::<Digest>();

/// Computes the expected size of each proof component given the MVK and auxiliary parameters.
///
/// This is an approximate calculation that assumes all AIRs are present.
///
/// # Arguments
/// * `mvk` - The multi-stark verifying key
/// * `stacked_widths` - The width of each stacked commitment (common_main first, then preprocessed/cached)
/// * `n_logup` - The number of logup rounds (derived from total interactions)
/// * `n_max` - The maximum log height minus l_skip (i.e., max_log_height - l_skip)
///
/// # Returns
/// The total expected proof size in bytes.
pub fn expected_proof_size(
    mvk: &MultiStarkVerifyingKey0V2,
    stacked_widths: &[usize],
    n_logup: usize,
    n_max: usize,
) -> usize {
    let mut total_size = 0usize;
    let l_skip = mvk.params.l_skip;
    let num_airs = mvk.per_air.len();

    // ==================== CODEC_VERSION ====================
    let codec_version_size = size_of::<u32>();
    tracing::info!("codec_version: {} bytes", codec_version_size);
    total_size += codec_version_size;

    // ==================== COMMON MAIN COMMIT ====================
    let common_main_commit_size = DIGEST_SIZE;
    tracing::info!("common_main_commit: {} bytes", common_main_commit_size);
    total_size += common_main_commit_size;

    // ==================== TRACE VDATA ====================
    // Encoded as: num_airs (usize) + bitmap (ceil(num_airs/8) bytes) + per-present TraceVData
    let trace_vdata_header_size = LEN_SIZE + num_airs.div_ceil(8);

    let mut trace_vdata_content_size = 0usize;
    for vk in &mvk.per_air {
        // TraceVData = log_height (usize) + cached_commitments (Vec<Digest>)
        // Vec<Digest> = len (usize) + len * DIGEST_SIZE
        let num_cached = vk.num_cached_mains();
        let vdata_size = LEN_SIZE + LEN_SIZE + num_cached * DIGEST_SIZE;
        trace_vdata_content_size += vdata_size;
    }
    let trace_vdata_size = trace_vdata_header_size + trace_vdata_content_size;
    tracing::info!("trace_vdata: {} bytes", trace_vdata_size);
    total_size += trace_vdata_size;

    // ==================== PUBLIC VALUES ====================
    // For each AIR: Vec<F> = len (usize) + len * F_SIZE
    let mut public_values_size = 0usize;
    for vk in &mvk.per_air {
        public_values_size += LEN_SIZE + vk.params.num_public_values * F_SIZE;
    }
    tracing::info!("public_values: {} bytes", public_values_size);
    total_size += public_values_size;

    // ==================== GKR PROOF ====================
    let gkr_proof_size = compute_gkr_proof_size(mvk, n_logup, l_skip);
    tracing::info!("gkr_proof: {} bytes", gkr_proof_size);
    total_size += gkr_proof_size;

    // ==================== BATCH CONSTRAINT PROOF ====================
    let batch_proof_size = compute_batch_constraint_proof_size(mvk, n_max, l_skip);
    tracing::info!("batch_constraint_proof: {} bytes", batch_proof_size);
    total_size += batch_proof_size;

    // ==================== STACKING PROOF ====================
    let stacking_proof_size = compute_stacking_proof_size(mvk, stacked_widths, l_skip);
    tracing::info!("stacking_proof: {} bytes", stacking_proof_size);
    total_size += stacking_proof_size;

    // ==================== WHIR PROOF ====================
    let whir_proof_size = compute_whir_proof_size(&mvk.params, stacked_widths);
    tracing::info!("whir_proof: {} bytes", whir_proof_size);
    total_size += whir_proof_size;

    tracing::info!("total_proof_size: {} bytes", total_size);
    total_size
}

fn compute_gkr_proof_size(
    _mvk: &MultiStarkVerifyingKey0V2,
    n_logup: usize,
    l_skip: usize,
) -> usize {
    // num_gkr_rounds = l_skip + n_logup (if there are interactions)
    // If n_logup == 0, assume no interactions
    let num_gkr_rounds = if n_logup == 0 { 0 } else { l_skip + n_logup };

    let mut size = 0usize;

    // logup_pow_witness: F
    size += F_SIZE;
    // q0_claim: EF
    size += EF_SIZE;

    // claims_per_layer: Vec<GkrLayerClaims> = len (usize) + num_gkr_rounds * GkrLayerClaims
    // GkrLayerClaims = 4 * EF (p_xi_0, p_xi_1, q_xi_0, q_xi_1)
    let gkr_layer_claims_size = 4 * EF_SIZE;
    size += LEN_SIZE + num_gkr_rounds * gkr_layer_claims_size;

    // sumcheck_polys: flattened [EF; 3] elements
    // Total: sum over i in 0..(num_gkr_rounds-1) of (i+1) * 3 * EF
    // = 3 * EF * (1 + 2 + ... + (num_gkr_rounds-1))
    // = 3 * EF * (num_gkr_rounds-1) * num_gkr_rounds / 2
    let num_sumcheck_polys = num_gkr_rounds.saturating_sub(1);
    let sumcheck_evals_count = num_sumcheck_polys * (num_sumcheck_polys + 1) / 2;
    size += sumcheck_evals_count * 3 * EF_SIZE;

    tracing::info!(
        "  gkr: num_gkr_rounds={}, claims_per_layer={} bytes, sumcheck_polys={} bytes",
        num_gkr_rounds,
        LEN_SIZE + num_gkr_rounds * gkr_layer_claims_size,
        sumcheck_evals_count * 3 * EF_SIZE
    );

    size
}

fn compute_batch_constraint_proof_size(
    mvk: &MultiStarkVerifyingKey0V2,
    n_max: usize,
    l_skip: usize,
) -> usize {
    let num_airs_present = mvk.per_air.len();

    if num_airs_present == 0 {
        return 0;
    }

    let s_0_deg = (mvk.max_constraint_degree() + 1) * ((1 << l_skip) - 1);
    let max_degree_plus_one = mvk.max_constraint_degree() + 1;

    let mut size = 0usize;

    // numerator_term_per_air: Vec<EF> = len (usize) + num_airs_present * EF
    size += LEN_SIZE + num_airs_present * EF_SIZE;
    // denominator_term_per_air: num_airs_present * EF (no length prefix, known from numerator)
    size += num_airs_present * EF_SIZE;

    // univariate_round_coeffs: Vec<EF> = len (usize) + (s_0_deg + 1) * EF
    size += LEN_SIZE + (s_0_deg + 1) * EF_SIZE;

    // sumcheck_round_polys: n_max (usize) + max_degree_plus_one (usize if n_max > 0) + n_max * max_degree_plus_one * EF
    size += LEN_SIZE; // n_max
    if n_max > 0 {
        size += LEN_SIZE; // max_degree_plus_one
        size += n_max * max_degree_plus_one * EF_SIZE;
    }

    // column_openings: per AIR, Vec<Vec<(EF, EF)>> (pairs)
    for vk in &mvk.per_air {
        size += LEN_SIZE; // Vec<Vec<_>> outer length

        // Part 0: common_main
        let common_main_width = vk.params.width.common_main;
        size += LEN_SIZE + common_main_width * 2 * EF_SIZE;

        // Part 1: preprocessed (if present)
        if let Some(preprocessed_width) = vk.params.width.preprocessed {
            size += LEN_SIZE + preprocessed_width * 2 * EF_SIZE;
        }

        // Remaining parts: cached_mains
        for &cached_width in &vk.params.width.cached_mains {
            size += LEN_SIZE + cached_width * 2 * EF_SIZE;
        }
    }

    tracing::info!(
        "  batch: n_max={}, s_0_deg={}, max_degree={}, num_airs_present={}",
        n_max,
        s_0_deg,
        mvk.max_constraint_degree(),
        num_airs_present
    );

    size
}

fn compute_stacking_proof_size(
    mvk: &MultiStarkVerifyingKey0V2,
    stacked_widths: &[usize],
    l_skip: usize,
) -> usize {
    let n_stack = mvk.params.n_stack;
    let s_0_deg = 2 * ((1 << l_skip) - 1);

    let mut size = 0usize;

    // univariate_round_coeffs: Vec<EF> = len (usize) + (s_0_deg + 1) * EF
    size += LEN_SIZE + (s_0_deg + 1) * EF_SIZE;

    // sumcheck_round_polys: Vec<[EF; 2]> = len (usize) + n_stack * 2 * EF
    size += LEN_SIZE + n_stack * 2 * EF_SIZE;

    // stacking_openings: Vec<Vec<EF>> = len (usize) + per layout
    size += LEN_SIZE; // outer vec length
    for &width in stacked_widths {
        // Vec<EF> = len (usize) + width * EF
        size += LEN_SIZE + width * EF_SIZE;
    }

    tracing::info!(
        "  stacking: n_stack={}, s_0_deg={}, num_layouts={}, stacked_widths={:?}",
        n_stack,
        s_0_deg,
        stacked_widths.len(),
        stacked_widths
    );

    size
}

/// Computes the WHIR proof size in bytes.
///
/// ## Parameters
/// - \( R \) = `num_whir_rounds`
/// - \( k \) = `k_whir` (folding factor)
/// - \( N \) = `num_commits` = `stacked_widths.len()`
/// - \( W = \sum_j w_j \) = total stacked width
/// - \( H \) = `log_stacked_height`
/// - \( \rho \) = `log_blowup`
/// - \( q_r \) = number of queries for round \( r \)
///
/// ## Constants
/// - \( L = 4 \) bytes (length prefix, encoded as u32)
/// - \( F = 4 \) bytes (base field element)
/// - \( E = 16 \) bytes (extension field element)
/// - \( D = 32 \) bytes (digest)
///
/// ## Formula
/// ```text
/// WHIR_size = whir_sumcheck_polys + codeword_commits + ood_values + pow_witnesses
///           + metadata + initial_opened_rows + initial_merkle_proofs
///           + codeword_opened_values + codeword_merkle_proofs + final_poly
/// ```
///
/// Where:
/// - `whir_sumcheck_polys` = \( L + 2 R k \cdot E \)
/// - `codeword_commits` = \( L + (R-1) \cdot D \)
/// - `ood_values` = \( (R-1) \cdot E \)
/// - `pow_witnesses` = \( R \cdot F \)
/// - `metadata` = \( 4L \) (num_commits, num_queries, k_exp, merkle_depth)
/// - `initial_opened_rows` = \( q_0 \cdot 2^k \cdot (N \cdot L + W \cdot F) \)
/// - `initial_merkle_proofs` = \( N \cdot q_0 \cdot (H + \rho - k) \cdot D \)
/// - `codeword_opened_values` = \( (R-1) \cdot L + 2^k \cdot E \cdot \sum_{r=1}^{R-1} q_r \)
/// - `codeword_merkle_proofs` = \( L + D \cdot \sum_{r=1}^{R-1} q_r \cdot (H + \rho - k - r) \)
/// - `final_poly` = \( L + 2^{H - Rk} \cdot E \)
///
/// ## Closed Form (assuming constant queries \( q \) per round)
///
/// Using \( M = H + \rho - k \) (initial merkle depth):
///
/// - `codeword_opened_values` = \( (R-1)(L + q \cdot 2^k \cdot E) \)
/// - `codeword_merkle_proofs` = \( L + q(R-1)(M - \frac{R}{2}) \cdot D \)
///
/// **Total WHIR size:**
/// ```text
/// WHIR = 8L + 2RkE + (R-1)(D + E) + RF + 2^{H-Rk}·E
///      + q·2^k·[ NL + WF + (R-1)E ]
///      + qD·[ NM + (R-1)(M - R/2) ]
/// ```
///
/// Grouping by scaling behavior:
/// - **Fixed overhead:** \( 8L + RF + 2^{H-Rk} \cdot E \)
/// - **Scales with R:** \( 2Rk \cdot E + (R-1)(D + E + L) \)
/// - **Scales with q·2^k:** \( N \cdot L + W \cdot F + (R-1) \cdot E \)
/// - **Scales with q (merkle):** \( D \cdot [NM + (R-1)(M - R/2)] \)
fn compute_whir_proof_size(params: &SystemParams, stacked_widths: &[usize]) -> usize {
    let log_stacked_height = params.log_stacked_height();
    let num_whir_rounds = params.num_whir_rounds();
    let k_whir = params.k_whir();
    let k_whir_exp = 1 << k_whir;
    let num_commits = stacked_widths.len();
    let log_blowup = params.log_blowup;

    let mut size = 0usize;

    // whir_sumcheck_polys: Vec<[EF; 2]> = len (usize) + num_sumcheck_rounds * 2 * EF
    let num_whir_sumcheck_rounds = params.num_whir_sumcheck_rounds();
    size += LEN_SIZE + num_whir_sumcheck_rounds * 2 * EF_SIZE;
    tracing::info!(
        "  whir: whir_sumcheck_polys={} bytes (num_sumcheck_rounds={})",
        LEN_SIZE + num_whir_sumcheck_rounds * 2 * EF_SIZE,
        num_whir_sumcheck_rounds
    );

    // codeword_commits: Vec<Digest> = len (usize) + (num_whir_rounds - 1) * Digest
    let codeword_commits_size = LEN_SIZE + (num_whir_rounds - 1) * DIGEST_SIZE;
    size += codeword_commits_size;
    tracing::info!("  whir: codeword_commits={} bytes", codeword_commits_size);

    // ood_values: (num_whir_rounds - 1) * EF (no length prefix, derived from codeword_commits)
    let ood_values_size = (num_whir_rounds - 1) * EF_SIZE;
    size += ood_values_size;
    tracing::info!("  whir: ood_values={} bytes", ood_values_size);

    // whir_pow_witnesses: num_whir_rounds * F (no length prefix)
    let pow_witnesses_size = num_whir_rounds * F_SIZE;
    size += pow_witnesses_size;
    tracing::info!("  whir: whir_pow_witnesses={} bytes", pow_witnesses_size);

    // Initial round metadata
    // num_commits (usize) + initial_num_whir_queries (usize) + k_whir_exp (usize)
    size += 3 * LEN_SIZE;

    let initial_whir_round_num_queries = params.whir.rounds[0].num_queries;
    let initial_merkle_depth = (log_stacked_height + log_blowup).saturating_sub(k_whir);

    // merkle_depth (usize) if initial_num_whir_queries > 0
    if initial_whir_round_num_queries > 0 {
        size += LEN_SIZE;
    }

    // initial_round_opened_rows: num_commits x num_queries x k_whir_exp x width[i]
    // Each row is a Vec<F> encoded with length prefix + elements
    let mut initial_opened_rows_size = 0usize;
    for &width in stacked_widths {
        // Each query has k_whir_exp rows, each row has length prefix + `width` F elements
        let num_rows = initial_whir_round_num_queries * k_whir_exp;
        let row_size = LEN_SIZE + width * F_SIZE;
        initial_opened_rows_size += num_rows * row_size;
    }
    size += initial_opened_rows_size;
    tracing::info!(
        "  whir: initial_round_opened_rows={} bytes (num_commits={}, num_queries={}, k_whir_exp={})",
        initial_opened_rows_size,
        num_commits,
        initial_whir_round_num_queries,
        k_whir_exp
    );

    // initial_round_merkle_proofs: num_commits x num_queries x merkle_depth x Digest
    let initial_merkle_proofs_size =
        num_commits * initial_whir_round_num_queries * initial_merkle_depth * DIGEST_SIZE;
    size += initial_merkle_proofs_size;
    tracing::info!(
        "  whir: initial_round_merkle_proofs={} bytes (depth={})",
        initial_merkle_proofs_size,
        initial_merkle_depth
    );
    tracing::info!(
        "  whir round 0: num_queries={}, merkle_depth={}",
        initial_whir_round_num_queries,
        initial_merkle_depth
    );

    // codeword_opened_values: (num_whir_rounds - 1) rounds
    // Each round: num_queries (usize) + num_queries * k_whir_exp * EF
    let mut codeword_opened_size = 0usize;
    for round in 1..num_whir_rounds {
        let num_queries = params.whir.rounds[round].num_queries;
        codeword_opened_size += LEN_SIZE; // num_queries
        codeword_opened_size += num_queries * k_whir_exp * EF_SIZE;
    }
    size += codeword_opened_size;
    tracing::info!(
        "  whir: codeword_opened_values={} bytes",
        codeword_opened_size
    );

    // codeword_merkle_proofs: first_merkle_depth (usize) + flattened proofs
    size += LEN_SIZE; // first_merkle_depth

    let mut codeword_merkle_size = 0usize;
    for round in 1..num_whir_rounds {
        let num_queries = params.whir.rounds[round].num_queries;
        let merkle_depth = log_stacked_height + log_blowup - k_whir - round;
        codeword_merkle_size += num_queries * merkle_depth * DIGEST_SIZE;
        tracing::info!(
            "  whir round {}: num_queries={}, merkle_depth={}",
            round,
            num_queries,
            merkle_depth
        );
    }
    size += codeword_merkle_size;
    tracing::info!(
        "  whir: codeword_merkle_proofs={} bytes",
        codeword_merkle_size
    );

    // final_poly: Vec<EF> = len (usize) + (1 << log_final_poly_len) * EF
    let final_poly_len = 1 << params.log_final_poly_len();
    let final_poly_size = LEN_SIZE + final_poly_len * EF_SIZE;
    size += final_poly_size;
    tracing::info!(
        "  whir: final_poly={} bytes (len={})",
        final_poly_size,
        final_poly_len
    );

    tracing::info!(
        "  whir: num_whir_rounds={}, k_whir={}, log_stacked_height={}, log_blowup={}",
        num_whir_rounds,
        k_whir,
        log_stacked_height,
        log_blowup
    );

    size
}

#[test]
fn find_optimal_whir_proof_size() {
    let l_skip = 2;
    let mut best_size = usize::MAX;
    let mut best_params: Option<SystemParams> = None;
    let num_cells = 1usize << 24;

    for n_stack in 18..=24 {
        for k_whir in 1..=5 {
            for log_blowup in 2..=5 {
                for num_rounds in 2..=8 {
                    if k_whir * num_rounds > n_stack + l_skip {
                        continue;
                    }
                    let log_final_poly_len = n_stack + l_skip - (k_whir * num_rounds);
                    let whir_params = crate::WhirParams {
                        k: k_whir,
                        log_final_poly_len,
                        query_phase_pow_bits: 20,
                    };
                    let whir_config =
                        crate::WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, 100);
                    let params = SystemParams {
                        l_skip,
                        n_stack,
                        log_blowup,
                        whir: whir_config,
                        logup: openvm_stark_sdk::config::log_up_params::log_up_security_params_baby_bear_100_bits(),
                        max_constraint_degree: 4,
                    };
                    let main_width = num_cells.div_ceil(1 << (n_stack + l_skip));
                    let size = compute_whir_proof_size(&params, &[main_width, 1]);

                    if size < best_size {
                        best_size = size;
                        best_params = Some(params);
                    }
                }
            }
        }
    }

    println!("best params: {:?}", best_params.unwrap());
    println!("best size: {best_size}");
}
