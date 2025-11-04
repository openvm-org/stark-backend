use serde::{Deserialize, Serialize};

use crate::{Digest, EF, F};

#[derive(Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct TraceVData {
    /// The base-2 logarithm of the height of the trace.
    ///
    /// If the corresponding AIR has a preprocessed trace, this must match the
    /// value in the vkey.
    pub hypercube_dim: usize,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// For the initial round: per commited matrix, per in-domain query.
    pub initial_round_opened_rows: Vec<Vec<Vec<Vec<F>>>>,
    pub initial_round_merkle_proofs: Vec<Vec<MerkleProof>>,
    /// Per non-initial round, per in-domain-query.
    pub codeword_opened_values: Vec<Vec<Vec<EF>>>,
    pub codeword_merkle_proofs: Vec<Vec<MerkleProof>>,
    /// Coefficients of the polynomial after the final round.
    pub final_poly: Vec<EF>,
}
