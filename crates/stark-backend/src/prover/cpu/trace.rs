use std::sync::Arc;

use derivative::Derivative;
use itertools::{izip, Itertools};
use p3_commit::Pcs;
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    Matrix,
};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use super::hal::ProverBackend;
use crate::{
    commit::CommittedSingleMatrixView,
    config::{Com, Domain, PcsProverData, StarkGenericConfig, Val},
    keygen::view::MultiStarkProvingKeyView,
    prover::quotient::{helper::QuotientVkDataHelper, ProverQuotientData, QuotientCommitter},
    utils::metrics_span,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_quotient_traces<'a, SC: StarkGenericConfig>(
    pcs: &SC::Pcs,
    mpk: &MultiStarkProvingKeyView<SC>,
    alpha: SC::Challenge,
    challenges: &[Vec<SC::Challenge>],
    public_values_per_air: &[Vec<Val<SC>>],
    domain_per_air: Vec<Domain<SC>>,
    cached_mains_pdata_per_air: &'a [Vec<ProverTraceData<SC>>],
    common_main_prover_data: &'a ProverTraceData<SC>,
    perm_prover_data: &'a Option<ProverTraceData<SC>>,
    exposed_values_after_challenge: Vec<Vec<Vec<SC::Challenge>>>,
) -> ProverQuotientData<SC> {
    // let trace_views = create_trace_view_per_air(
    //     domain_per_air,
    //     cached_mains_pdata_per_air,
    //     mpk,
    //     exposed_values_after_challenge,
    //     common_main_prover_data,
    //     perm_prover_data,
    // );
    let quotient_committer = QuotientCommitter::new(pcs, challenges, alpha);
    let qvks = mpk
        .per_air
        .iter()
        .map(|pk| pk.get_quotient_vk_data())
        .collect_vec();
    let quotient_values = metrics_span("quotient_poly_compute_time_ms", || {
        quotient_committer.quotient_values(&qvks, &trace_views, public_values_per_air)
    });
    // Commit to quotient polynomials. One shared commit for all quotient polynomials
    metrics_span("quotient_poly_commit_time_ms", || {
        quotient_committer.commit(quotient_values)
    })
}

/// Prover that commits to a batch of trace matrices, possibly of different heights.
pub struct TraceCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
}

impl<SC: StarkGenericConfig> Clone for TraceCommitter<'_, SC> {
    fn clone(&self) -> Self {
        Self { pcs: self.pcs }
    }
}

impl<'pcs, SC: StarkGenericConfig> TraceCommitter<'pcs, SC> {
    pub fn new(pcs: &'pcs SC::Pcs) -> Self {
        Self { pcs }
    }

    /// Uses the PCS to commit to a sequence of trace matrices.
    /// The commitment will depend on the order of the matrices.
    /// The matrices may be of different heights.
    pub fn commit(&self, traces: Vec<RowMajorMatrix<Val<SC>>>) -> ProverTraceData<SC> {
        info_span!("commit to trace data").in_scope(|| {
            let traces_with_domains: Vec<_> = traces
                .into_iter()
                .map(|matrix| {
                    let height = matrix.height();
                    // Recomputing the domain is lightweight
                    let domain = self.pcs.natural_domain_for_degree(height);
                    (domain, matrix)
                })
                .collect();
            let (commit, data) = self.pcs.commit(traces_with_domains);
            ProverTraceData {
                commit,
                data: Arc::new(data),
            }
        })
    }
}

/// Prover data for multi-matrix trace commitments.
/// The data is for the traces committed into a single commitment.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "Com<SC>: Clone"))]
#[serde(bound(
    serialize = "Com<SC>: Serialize, PcsProverData<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>, PcsProverData<SC>: Deserialize<'de>"
))]
pub struct ProverTraceData<SC: StarkGenericConfig> {
    /// Commitment to the trace matrices.
    pub commit: Com<SC>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    /// The data is stored as a thread-safe smart [Arc] pointer because [PcsProverData] does
    /// not implement clone and should not be cloned. The prover only needs a reference to
    /// this data, so we use a smart pointer to elide lifetime concerns.
    pub data: Arc<PcsProverData<SC>>,
}

/// The PCS commits to multiple matrices at once, so this struct stores
/// references to get PCS data relevant to a single matrix (e.g., LDE matrix, openings).
#[derive(Derivative, derive_new::new)]
#[derivative(Clone(bound = ""))]
pub struct CommittedSingleMatrixView<PB: ProverBackend> {
    /// Prover data, includes LDE matrix of trace and Merkle tree.
    /// The prover data can commit to multiple trace matrices, so
    /// `matrix_index` is needed to identify this trace.
    pub data: PB::PcsDataRef,
    /// The index of the trace matrix in the prover data.
    pub matrix_index: usize,
}
