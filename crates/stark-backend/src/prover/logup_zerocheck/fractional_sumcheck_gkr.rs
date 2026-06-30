use std::{cmp::max, ops::Add};

use p3_field::{Field, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::{
    proof::GkrLayerClaims,
    prover::{
        error::LogupZerocheckError,
        poly::evals_eq_hypercube,
        sumcheck::{fold_mle_evals, sumcheck_round_poly_evals},
        ColMajorMatrix,
    },
    FiatShamirTranscript, StarkProtocolConfig,
};

/// Proof for fractional sumcheck protocol
pub struct FracSumcheckProof<SC: StarkProtocolConfig> {
    /// The fractional sum p_0 / q_0
    pub fractional_sum: (SC::EF, SC::EF),
    /// The claims for p_j(0, rho), p_j(1, rho), q_j(0, rho), and q_j(1, rho) for each layer j > 0.
    pub claims_per_layer: Vec<GkrLayerClaims<SC>>,
    /// Sumcheck polynomials for each layer, for each sumcheck round, given by their evaluations on
    /// {1, 2, 3}.
    pub sumcheck_polys: Vec<Vec<[SC::EF; 3]>>,
}

#[derive(Clone, Copy, Debug, Default, derive_new::new)]
#[repr(C)]
pub struct Frac<EF> {
    // PERF[jpw]: in the initial round, we can keep `p` in base field
    pub p: EF,
    pub q: EF,
}

const FRACTIONAL_GKR_SP_DEG: usize = 2;
const FRACTIONAL_GKR_DEFAULT_BLOCK_SIZE: usize = 256;
const FRACTIONAL_GKR_MIN_BLOCK_SIZE: usize = 64;
const FRACTIONAL_GKR_WARP_SIZE: usize = 32;

/// Fractional-GKR sizing model used by CUDA batching and memory metering.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FractionalGkrMemoryModel {
    base_field_bytes: usize,
    extension_degree: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WorkBufferMemoryCandidates {
    fold_eval: usize,
    precompute_m: usize,
    default_precompute_aux: usize,
    max_tuned_precompute_aux: usize,
    common_aux: usize,
}

/// Fractional-GKR work-buffer strategy selected by the CUDA backend.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FractionalGkrWorkBufferStrategy {
    /// Non-default CUDA tuning can select either work-buffer shape, so reserve and meter both.
    #[default]
    Conservative,
    /// Precompute-M is disabled, so CUDA allocates the fold-eval work buffer.
    FoldEval,
    /// Precompute-M is enabled with default tuning.
    PrecomputeM,
}

impl FractionalGkrMemoryModel {
    const INPUT_EF_ELEMENTS_PER_INTERACTION: usize = 2;
    pub const WINDOW_SIZE: usize = 3;
    pub const WINDOW_DEFAULT_MIN_N: usize = 22;
    pub const PRECOMPUTE_M_MIN_TAIL_TILE: usize = 256;
    pub const PRECOMPUTE_M_DEFAULT_TAIL_TILE: usize = 4096;
    pub const PRECOMPUTE_M_DEFAULT_MIN_BLOCKS: usize = 64;
    pub const PRECOMPUTE_M_DEFAULT_TARGET_BLOCKS: usize = 1024;
    pub const ROUND_COMPUTE_FALLBACK_BLOCKS: usize = 228;

    #[inline]
    pub const fn new(base_field_bytes: usize, extension_degree: usize) -> Self {
        Self {
            base_field_bytes,
            extension_degree,
        }
    }

    #[inline]
    pub(crate) fn logical_len(interaction_cells: usize) -> usize {
        if interaction_cells == 0 {
            return 0;
        }

        let log_len = usize::BITS - interaction_cells.leading_zeros();
        1usize.checked_shl(log_len).unwrap_or(usize::MAX)
    }

    #[inline]
    pub(crate) fn input_memory_bytes(&self, interaction_cells: usize) -> usize {
        interaction_cells
            .saturating_mul(Self::INPUT_EF_ELEMENTS_PER_INTERACTION)
            .saturating_mul(self.extension_degree)
            .saturating_mul(self.base_field_bytes)
    }

    #[inline]
    pub fn peak_work_buffer_memory_bytes(&self, logical_len: usize) -> usize {
        self.peak_work_buffer_memory_bytes_with_strategy(
            logical_len,
            FractionalGkrWorkBufferStrategy::Conservative,
        )
    }

    #[inline]
    pub fn peak_work_buffer_memory_bytes_with_strategy(
        &self,
        logical_len: usize,
        strategy: FractionalGkrWorkBufferStrategy,
    ) -> usize {
        self.peak_work_buffer_memory_bytes_with_allocator(logical_len, strategy, |bytes| bytes)
    }

    #[inline]
    pub(crate) fn peak_work_buffer_memory_bytes_with_allocator(
        &self,
        logical_len: usize,
        strategy: FractionalGkrWorkBufferStrategy,
        mut allocation_bytes: impl FnMut(usize) -> usize,
    ) -> usize {
        let candidates = self.work_buffer_memory_candidates(logical_len);
        match strategy {
            FractionalGkrWorkBufferStrategy::Conservative => {
                allocation_bytes(candidates.common_aux)
                    .saturating_add(allocation_bytes(max(
                        candidates.fold_eval,
                        candidates.precompute_m,
                    )))
                    .saturating_add(allocation_bytes(candidates.max_tuned_precompute_aux))
            }
            FractionalGkrWorkBufferStrategy::FoldEval => allocation_bytes(candidates.common_aux)
                .saturating_add(allocation_bytes(candidates.fold_eval)),
            FractionalGkrWorkBufferStrategy::PrecomputeM => allocation_bytes(candidates.common_aux)
                .saturating_add(allocation_bytes(candidates.precompute_m))
                .saturating_add(allocation_bytes(candidates.default_precompute_aux)),
        }
    }

    #[inline]
    pub(crate) fn peak_memory_bytes_with_allocator(
        &self,
        interaction_cells: usize,
        logical_len: usize,
        strategy: FractionalGkrWorkBufferStrategy,
        mut allocation_bytes: impl FnMut(usize) -> usize,
    ) -> usize {
        allocation_bytes(self.input_memory_bytes(interaction_cells)).saturating_add(
            self.peak_work_buffer_memory_bytes_with_allocator(
                logical_len,
                strategy,
                allocation_bytes,
            ),
        )
    }

    #[inline]
    fn work_buffer_memory_candidates(&self, logical_len: usize) -> WorkBufferMemoryCandidates {
        if logical_len <= 2 {
            return WorkBufferMemoryCandidates::default();
        }

        // The input Frac layer is counted separately by `input_memory_bytes`.
        let frac_size = Self::INPUT_EF_ELEMENTS_PER_INTERACTION
            .saturating_mul(self.extension_degree)
            .saturating_mul(self.base_field_bytes);
        let fold_eval = Self::fold_eval_work_buffer_elements(logical_len).saturating_mul(frac_size);
        let precompute_m =
            Self::precompute_m_work_buffer_elements(logical_len).saturating_mul(frac_size);
        let default_precompute_aux = self.precompute_m_auxiliary_memory_bytes(logical_len);
        let max_tuned_precompute_aux =
            self.max_tuned_precompute_m_auxiliary_memory_bytes(logical_len);
        let common_aux = self.common_auxiliary_memory_bytes(logical_len);
        WorkBufferMemoryCandidates {
            fold_eval,
            precompute_m,
            default_precompute_aux,
            max_tuned_precompute_aux,
            common_aux,
        }
    }

    #[inline]
    pub fn fold_eval_work_buffer_elements(logical_len: usize) -> usize {
        if logical_len <= 4 {
            return 0;
        }

        logical_len / 4
    }

    #[inline]
    pub fn precompute_m_work_buffer_elements(logical_len: usize) -> usize {
        if logical_len <= 4 {
            return 0;
        }

        (logical_len >> (1 + Self::WINDOW_SIZE)).max(1usize << Self::WINDOW_DEFAULT_MIN_N)
    }

    #[inline]
    fn precompute_m_auxiliary_memory_bytes(&self, logical_len: usize) -> usize {
        let ext_size = self.extension_degree.saturating_mul(self.base_field_bytes);
        if !Self::has_default_precompute_m_window(logical_len) {
            return 0;
        }

        // m_buffer has 4^w EF entries; prefix/suffix buffers each have 2^w EF entries.
        // m_partial_buffer is retained across rounds and uses the same default tail tiling as CUDA.
        let precompute_ef_elements =
            (1usize << (2 * Self::WINDOW_SIZE)) + (1usize << (Self::WINDOW_SIZE + 1));
        precompute_ef_elements
            .saturating_add(Self::precompute_m_partial_buffer_elements(logical_len))
            .saturating_mul(ext_size)
    }

    #[inline]
    fn common_auxiliary_memory_bytes(&self, logical_len: usize) -> usize {
        let ext_size = self.extension_degree.saturating_mul(self.base_field_bytes);
        // SqrtEqLayers and the tiny d_sum_evals/copy_scratch buffers are live with both strategies.
        Self::sqrt_eq_layers_elements(logical_len)
            .saturating_add(4)
            .saturating_mul(ext_size)
    }

    #[inline]
    fn precompute_m_partial_buffer_elements(logical_len: usize) -> usize {
        let Some(total_rounds) = Self::log2_len(logical_len) else {
            return 0;
        };

        let mut max_partial_len = 0usize;
        for round in 1..total_rounds {
            let stop = round.div_ceil(2);
            if stop <= 1 {
                continue;
            }

            let rem_n = round - 1;
            let rounds_left = stop - 1;
            if rem_n < Self::WINDOW_DEFAULT_MIN_N || rounds_left < Self::WINDOW_SIZE {
                continue;
            }

            let tail_n = rem_n - Self::WINDOW_SIZE;
            let tail_points = 1usize << tail_n;
            let target_blocks =
                Self::PRECOMPUTE_M_DEFAULT_TARGET_BLOCKS.max(Self::PRECOMPUTE_M_DEFAULT_MIN_BLOCKS);
            let tail_tile = tail_points.div_ceil(target_blocks).max(1).clamp(
                Self::PRECOMPUTE_M_MIN_TAIL_TILE,
                Self::PRECOMPUTE_M_DEFAULT_TAIL_TILE,
            );
            let num_blocks = tail_points.div_ceil(tail_tile);
            if num_blocks < Self::PRECOMPUTE_M_DEFAULT_MIN_BLOCKS {
                continue;
            }

            let m_len = 1usize << (2 * Self::WINDOW_SIZE);
            max_partial_len = max(max_partial_len, num_blocks.saturating_mul(m_len));
        }

        max_partial_len
    }

    #[inline]
    fn max_tuned_precompute_m_auxiliary_memory_bytes(&self, logical_len: usize) -> usize {
        let ext_size = self.extension_degree.saturating_mul(self.base_field_bytes);
        let max_partial_len = Self::max_tuned_precompute_m_partial_buffer_elements(logical_len);
        if max_partial_len == 0 {
            return 0;
        }

        let precompute_ef_elements =
            (1usize << (2 * Self::WINDOW_SIZE)) + (1usize << (Self::WINDOW_SIZE + 1));
        precompute_ef_elements
            .saturating_add(max_partial_len)
            .saturating_mul(ext_size)
    }

    #[inline]
    fn max_tuned_precompute_m_partial_buffer_elements(logical_len: usize) -> usize {
        let Some(total_rounds) = Self::log2_len(logical_len) else {
            return 0;
        };

        let mut max_partial_len = 0usize;
        for round in 1..total_rounds {
            let stop = round.div_ceil(2);
            if stop <= 1 {
                continue;
            }

            let rem_n = round - 1;
            let rounds_left = stop - 1;
            if rem_n < Self::WINDOW_SIZE || rounds_left < Self::WINDOW_SIZE {
                continue;
            }

            let tail_n = rem_n - Self::WINDOW_SIZE;
            let tail_points = 1usize << tail_n;
            let num_blocks = tail_points.div_ceil(Self::PRECOMPUTE_M_MIN_TAIL_TILE);
            let m_len = 1usize << (2 * Self::WINDOW_SIZE);
            max_partial_len = max(max_partial_len, num_blocks.saturating_mul(m_len));
        }

        max_partial_len
    }

    #[inline]
    fn has_default_precompute_m_window(logical_len: usize) -> bool {
        Self::precompute_m_partial_buffer_elements(logical_len) != 0
    }

    #[inline]
    fn sqrt_eq_layers_elements(logical_len: usize) -> usize {
        let Some(total_rounds) = Self::log2_len(logical_len) else {
            return 0;
        };

        // In outer round `r`, CUDA builds layers for `xi_prev[1..]`, length `r - 1`.
        // EqEvalLayers keeps every layer 1, 2, ..., 2^n; the shared length-1 seed is counted once.
        let n = total_rounds - 2;
        let low_n = n / 2;
        let high_n = n - low_n;
        ((1usize << (low_n + 1)) - 1).saturating_add((1usize << (high_n + 1)) - 2)
    }

    #[inline]
    fn log2_len(logical_len: usize) -> Option<usize> {
        (logical_len > 2 && logical_len.is_power_of_two())
            .then_some(logical_len.trailing_zeros() as usize)
    }

    #[inline]
    pub fn round_temp_buffer_elements(num_x: usize, min_blocks: usize) -> usize {
        // Mirrors CUDA round-compute launch sizing for `_frac_compute_round_temp_buffer_size`.
        let elements = num_x >> 1;
        let min_blocks = min_blocks.max(1);
        let mut block_size = FRACTIONAL_GKR_DEFAULT_BLOCK_SIZE;
        let mut blocks_needed = elements.div_ceil(block_size);
        if blocks_needed < min_blocks && elements >= FRACTIONAL_GKR_MIN_BLOCK_SIZE {
            block_size = elements.div_ceil(min_blocks);
            block_size = block_size
                .div_ceil(FRACTIONAL_GKR_WARP_SIZE)
                .saturating_mul(FRACTIONAL_GKR_WARP_SIZE)
                .max(FRACTIONAL_GKR_MIN_BLOCK_SIZE);
            blocks_needed = elements.div_ceil(block_size);
        }

        blocks_needed.saturating_mul(FRACTIONAL_GKR_SP_DEG)
    }
}

impl<EF: Field> Add<Frac<EF>> for Frac<EF> {
    type Output = Frac<EF>;

    fn add(self, other: Frac<EF>) -> Self::Output {
        Frac {
            p: self.p * other.q + self.q * other.p,
            q: self.q * other.q,
        }
    }
}

/// Runs the fractional sumcheck protocol using GKR layered circuit.
///
/// # Arguments
/// * `transcript` - The Fiat-Shamir transcript
/// * `evals` - list of `(p, q)` pairs of fractions in projective coordinates representing
///   evaluations on the hypercube
/// * `assert_zero` - Whether to assert that the final sum is zero. If `true`, then the transcript
///   will not observe the numerator of the final sum.
///
/// # Returns
/// The fractional sumcheck proof and the final random evaluation vector.
#[instrument(level = "info", skip_all)]
pub fn fractional_sumcheck<SC: StarkProtocolConfig, TS: FiatShamirTranscript<SC>>(
    transcript: &mut TS,
    evals: &[Frac<SC::EF>],
    assert_zero: bool,
) -> Result<(FracSumcheckProof<SC>, Vec<SC::EF>), LogupZerocheckError> {
    if evals.is_empty() {
        return Ok((
            FracSumcheckProof {
                fractional_sum: (SC::EF::ZERO, SC::EF::ONE),
                claims_per_layer: vec![],
                sumcheck_polys: vec![],
            },
            vec![],
        ));
    }
    // total_rounds = l_skip + n_logup
    let total_rounds = log2_strict_usize(evals.len());
    // sumcheck polys for layers j=2,...,total_rounds
    let mut sumcheck_polys = Vec::with_capacity(total_rounds);

    // segment tree: layer i=0,...,total_rounds starts at 2^i (index 0 unused)
    let mut tree_evals: Vec<Frac<SC::EF>> = vec![Frac::default(); 2 << total_rounds];
    tree_evals[(1 << total_rounds)..].copy_from_slice(evals);

    for node_idx in (1..(1 << total_rounds)).rev() {
        tree_evals[node_idx] = tree_evals[2 * node_idx] + tree_evals[2 * node_idx + 1];
    }
    let frac_sum = tree_evals[1];
    if assert_zero {
        if frac_sum.p != SC::EF::ZERO {
            return Err(LogupZerocheckError::NonZeroRootSum);
        }
    } else {
        transcript.observe_ext(frac_sum.p);
    }
    transcript.observe_ext(frac_sum.q);

    // Index i is for layer i+1
    let mut claims_per_layer: Vec<GkrLayerClaims<SC>> = Vec::with_capacity(total_rounds);

    // Process each GKR round
    // `j = round + 1` goes from `1, ..., total_rounds`
    //
    // Round `j = 1` is special since "sumcheck" is directly checked by verifier
    claims_per_layer.push(GkrLayerClaims::<SC> {
        p_xi_0: tree_evals[2].p,
        q_xi_0: tree_evals[2].q,
        p_xi_1: tree_evals[3].p,
        q_xi_1: tree_evals[3].q,
    });
    transcript.observe_ext(claims_per_layer[0].p_xi_0);
    transcript.observe_ext(claims_per_layer[0].q_xi_0);
    transcript.observe_ext(claims_per_layer[0].p_xi_1);
    transcript.observe_ext(claims_per_layer[0].q_xi_1);
    let mu_1 = transcript.sample_ext();
    debug!(gkr_round = 0, mu = %mu_1);
    // ξ^{(j-1)}
    let mut xi_prev = vec![mu_1];

    // GKR rounds
    for round in 1..total_rounds {
        // Number of hypercube points
        let eval_size = 1 << round;
        // We apply batch sumcheck to the polynomials
        // \eq(ξ^{(j-1)}, Y) (\hat p_j(0, Y) \hat q_j(1, Y) + \hat p_j(1, Y) \hat q_j(0, Y))
        // \eq(ξ^{(j-1)}, Y) (\hat q_j(0, Y) \hat q_j(1, Y))
        // Note: these are polynomials of degree 3 in each Y_i coordinate.

        // Sample λ_j for batching
        let lambda = transcript.sample_ext();
        debug!(gkr_round = round, %lambda);

        // Columns are p_j0, q_j0, p_j1, q_j1
        // PERF: use a view instead of re-allocating memory
        let mut pq_j_evals = SC::EF::zero_vec(4 * eval_size);
        let segment = &tree_evals[2 * eval_size..4 * eval_size];
        for x in 0..eval_size {
            pq_j_evals[x] = segment[2 * x].p;
            pq_j_evals[eval_size + x] = segment[2 * x].q;
            pq_j_evals[2 * eval_size + x] = segment[2 * x + 1].p;
            pq_j_evals[3 * eval_size + x] = segment[2 * x + 1].q;
        }
        let mut pq_j_evals = ColMajorMatrix::new(pq_j_evals, 4);
        let mut eq_xis = ColMajorMatrix::new(evals_eq_hypercube(&xi_prev), 1);

        // Batch sumcheck where the round polynomials are evaluated at {1, 2, 3}
        let (round_polys_eval, rho) = {
            let n = round;
            let mut round_polys_eval = Vec::with_capacity(n);
            let mut r_vec = Vec::with_capacity(n);

            // Sumcheck rounds: apply fraction addition in projective coordinates to MLEs
            for sumcheck_round in 0..n {
                // Evaluate the univariate polynomial at {1, 2, 3}
                // :projective fraction addition is degree 2, and then another +1 for eq
                let [s_evals] = sumcheck_round_poly_evals(
                    n - sumcheck_round,
                    3,
                    &[eq_xis.as_view(), pq_j_evals.as_view()],
                    |_x, _y, row| {
                        let eq_xi = row[0][0];
                        let &[p_j0, q_j0, p_j1, q_j1] = row[1].as_slice() else {
                            unreachable!("pq_j_evals always has 4 columns")
                        };
                        let p_prev = p_j0 * q_j1 + p_j1 * q_j0;
                        let q_prev = q_j0 * q_j1;
                        // batch using lambda
                        [eq_xi * (p_prev + lambda * q_prev)]
                    },
                );
                let s_evals: [SC::EF; 3] = s_evals.try_into().unwrap();
                for &eval in &s_evals {
                    transcript.observe_ext(eval);
                }
                round_polys_eval.push(s_evals);

                let r_round = transcript.sample_ext();
                pq_j_evals = fold_mle_evals(pq_j_evals, r_round);
                eq_xis = fold_mle_evals(eq_xis, r_round);
                r_vec.push(r_round);
                debug!(gkr_round = round, %sumcheck_round, %r_round);
            }
            (round_polys_eval, r_vec)
        };
        claims_per_layer.push(GkrLayerClaims::<SC> {
            p_xi_0: pq_j_evals.column(0)[0],
            q_xi_0: pq_j_evals.column(1)[0],
            p_xi_1: pq_j_evals.column(2)[0],
            q_xi_1: pq_j_evals.column(3)[0],
        });
        transcript.observe_ext(claims_per_layer[round].p_xi_0);
        transcript.observe_ext(claims_per_layer[round].q_xi_0);
        transcript.observe_ext(claims_per_layer[round].p_xi_1);
        transcript.observe_ext(claims_per_layer[round].q_xi_1);

        // Sample μ_j for reduction to single evaluation point
        let mu = transcript.sample_ext();
        debug!(gkr_round = round, %mu);

        // Update ξ^{(j)} = (μ_j, ρ^{(j-1)})
        xi_prev = [vec![mu], rho].concat();

        sumcheck_polys.push(round_polys_eval);
    }

    Ok((
        FracSumcheckProof {
            fractional_sum: (frac_sum.p, frac_sum.q),
            claims_per_layer,
            sumcheck_polys,
        },
        xi_prev,
    ))
}
