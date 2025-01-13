use std::{marker::PhantomData, ops::Deref, sync::Arc};

use derivative::Derivative;
use itertools::{zip_eq, Itertools};
use opener::OpeningProver;
use p3_challenger::FieldChallenger;
use p3_commit::Pcs;
use p3_field::FieldExtensionAlgebra;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;

use super::{
    hal::{self, MatrixView, ProverBackend},
    types::{PairView, ProverViewAfterRapPhases, StarkProvingKeyView},
};
use crate::{
    air_builders::symbolic::SymbolicConstraints,
    config::{
        Com, PcsProof, PcsProverData, RapPhaseSeqPartialProof, RapPhaseSeqProvingKey,
        StarkGenericConfig, Val,
    },
    interaction::RapPhaseSeq,
    keygen::view::MultiStarkVerifyingKeyView,
    proof::OpeningProof,
    prover::{hal::TraceCommitter, types::RapSinglePhaseView},
    utils::metrics_span,
};

/// Polynomial opening proofs
pub mod opener;
/// Computation of DEEP quotient polynomial and commitment
pub mod quotient;

/// Proves multiple chips with interactions together.
/// This prover implementation is specialized for Interactive AIRs.
pub struct MultiTraceStarkProver<'c, SC: StarkGenericConfig> {
    pub config: &'c SC,
}

/// CPU backend using Plonky3 traits.
#[derive(Default, Clone, Copy)]
pub struct Cpu<SC> {
    phantom: PhantomData<SC>,
}

#[derive(Clone, derive_new::new)]
pub struct CpuDevice<SC> {
    config: SC,
}

impl<SC: StarkGenericConfig> ProverBackend for Cpu<SC> {
    const CHALLENGE_EXT_DEGREE: u8 = <SC::Challenge as FieldExtensionAlgebra<Val<SC>>>::D as u8;

    type Val = Val<SC>;
    type Challenge = SC::Challenge;
    type OpeningProof = OpeningProof<PcsProof<SC>, SC::Challenge>;
    type Commitment = Com<SC>;
    type Challenger = SC::Challenger;
    // Note[jpw]: Use Arc to get around lifetime issues
    type MatrixView = Arc<RowMajorMatrix<Val<SC>>>;
    type PcsDataView = PcsDataView<SC>;
}

#[derive(Derivative)]
#[derivative(Clone(bound = ""))]
pub struct PcsDataView<SC: StarkGenericConfig> {
    // Note[jpw]: Use Arc to get around lifetime issues
    /// The preimage of a single commitment.
    pub data: Arc<PcsProverData<SC>>,
    /// A mixed matrix commitment scheme commits to multiple trace matrices within a single commitment.
    /// This is the ordered list of log2 heights of all committed trace matrices.
    pub log_trace_heights: Vec<u8>,
}

impl<T: Send + Sync + Clone> MatrixView for Arc<RowMajorMatrix<T>> {
    fn height(&self) -> usize {
        self.deref().height()
    }
    fn width(&self) -> usize {
        self.deref().width()
    }
}

impl<SC> CpuDevice<SC> {
    pub fn config(&self) -> &SC {
        &self.config
    }
}

impl<SC: StarkGenericConfig> CpuDevice<SC> {
    pub fn pcs(&self) -> &SC::Pcs {
        self.config.pcs()
    }
}

impl<SC: StarkGenericConfig> hal::TraceCommitter<Cpu<SC>> for CpuDevice<SC> {
    fn commit(&self, traces: &[Arc<RowMajorMatrix<Val<SC>>>]) -> (Com<SC>, PcsDataView<SC>) {
        let pcs = self.pcs();
        let (log_trace_heights, traces_with_domains): (Vec<_>, Vec<_>) = traces
            .iter()
            .map(|matrix| {
                let height = matrix.height();
                let log_height: u8 = log2_strict_usize(height).try_into().unwrap();
                // Recomputing the domain is lightweight
                let domain = pcs.natural_domain_for_degree(height);
                (log_height, (domain, matrix.as_ref().clone()))
            })
            .unzip();
        let (commit, data) = pcs.commit(traces_with_domains);
        (
            commit,
            PcsDataView {
                data: Arc::new(data),
                log_trace_heights,
            },
        )
    }
}

impl<SC: StarkGenericConfig> hal::RapPartialProver<Cpu<SC>> for CpuDevice<SC> {
    // `None` when there is no sumcheck
    type RapPartialProof = Option<RapPhaseSeqPartialProof<SC>>;
    type RapPartialProvingKeyView<'a>
        = &'a RapPhaseSeqProvingKey<SC>
    where
        Self: 'a;

    fn partially_prove<'a>(
        &self,
        challenger: &mut SC::Challenger,
        pk_views: &[StarkProvingKeyView<'a, Cpu<SC>, &'a RapPhaseSeqProvingKey<SC>>],
        trace_views: Vec<PairView<Arc<RowMajorMatrix<Val<SC>>>, Val<SC>>>,
    ) -> (Self::RapPartialProof, ProverViewAfterRapPhases<Cpu<SC>>) {
        assert_eq!(pk_views.len(), trace_views.len());
        let (constraints_per_air, rap_pk_per_air): (Vec<_>, Vec<_>) = pk_views
            .iter()
            .map(|pk| {
                (
                    // TODO[jpw]: remove this after RapPhaseSeq trait is modified
                    SymbolicConstraints::from(&pk.vk.symbolic_constraints),
                    pk.rap_partial_pk,
                )
            })
            .unzip();

        let (rap_phase_seq_proof, rap_phase_seq_data) = self
            .config()
            .rap_phase_seq()
            .partially_prove(
                challenger,
                &rap_pk_per_air,
                &constraints_per_air.iter().map(|c| c).collect_vec(),
                &trace_views,
            )
            .map_or((None, None), |(p, d)| (Some(p), Some(d)));

        let mvk_view = MultiStarkVerifyingKeyView::new(pk_views.iter().map(|pk| pk.vk).collect());

        let num_airs = pk_views.len();
        let mut perm_matrix_idx = 0usize;
        let rap_views_per_phase;
        let perm_trace_per_air = if let Some(phase_data) = rap_phase_seq_data {
            assert_eq!(mvk_view.num_phases(), 1);
            assert_eq!(
                mvk_view.num_challenges_in_phase(0),
                phase_data.challenges.len()
            );
            let perm_views = zip_eq(
                &phase_data.after_challenge_trace_per_air,
                phase_data.exposed_values_per_air,
            )
            .map(|(perm_trace, exposed_values)| {
                let mut matrix_idx = None;
                if perm_trace.is_some() {
                    matrix_idx = Some(perm_matrix_idx);
                    perm_matrix_idx += 1;
                }
                RapSinglePhaseView {
                    inner: matrix_idx,
                    challenges: phase_data.challenges.clone(),
                    exposed_values: exposed_values.unwrap_or_default(),
                }
            })
            .collect_vec();
            rap_views_per_phase = vec![perm_views]; // 1 challenge phase
            phase_data.after_challenge_trace_per_air
        } else {
            assert_eq!(mvk_view.num_phases(), 0);
            rap_views_per_phase = vec![];
            vec![None; num_airs]
        };

        // Commit to permutation traces: this means only 1 challenge round right now
        // One shared commit for all permutation traces
        let committed_pcs_data_per_phase: Vec<(Com<SC>, PcsDataView<SC>)> =
            metrics_span("perm_trace_commit_time_ms", || {
                let flattened_traces: Vec<_> = perm_trace_per_air
                    .into_iter()
                    .flat_map(|perm_trace| {
                        perm_trace.map(|trace| Arc::new(trace.flatten_to_base()))
                    })
                    .collect();
                // Only commit if there are permutation traces
                if !flattened_traces.is_empty() {
                    let (commit, data) = self.commit(&flattened_traces);
                    Some((commit, data))
                } else {
                    None
                }
            })
            .into_iter()
            .collect();
        let prover_view = ProverViewAfterRapPhases {
            committed_pcs_data_per_phase,
            rap_views_per_phase,
        };
        (rap_phase_seq_proof, prover_view)
    }
}

impl<SC: StarkGenericConfig> hal::OpeningProver<Cpu<SC>> for CpuDevice<SC> {
    fn open(
        &self,
        challenger: &mut SC::Challenger,
        // For each preprocessed trace commitment, the prover data and
        // the log height of the matrix, in order
        preprocessed: Vec<PcsDataView<SC>>,
        // For each main trace commitment, the prover data and
        // the log height of each matrix, in order
        // Note: this is all one challenge phase.
        main: Vec<PcsDataView<SC>>,
        // `after_phase[i]` has shared commitment prover data for all matrices in phase `i + 1`.
        after_phase: Vec<PcsDataView<SC>>,
        // Quotient poly commitment prover data
        quotient_data: PcsDataView<SC>,
        // Quotient degree for each RAP committed in quotient_data, in order
        quotient_degrees: &[u8],
    ) -> OpeningProof<PcsProof<SC>, SC::Challenge> {
        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let pcs = self.pcs();
        let domain = |log_height| pcs.natural_domain_for_degree(1 << log_height);
        let opener = OpeningProver::new(pcs, zeta);
        let preprocessed = preprocessed
            .iter()
            .map(|v| {
                assert_eq!(v.log_trace_heights.len(), 1);
                (&v.data, domain(v.log_trace_heights[0]))
            })
            .collect();
        let main = main
            .iter()
            .map(|v| {
                let domains = v.log_trace_heights.iter().copied().map(domain).collect();
                (&v.data, domains)
            })
            .collect();
        let after_phase = after_phase
            .iter()
            .map(|v| {
                let domains = v.log_trace_heights.iter().copied().map(domain).collect();
                (&v.data, domains)
            })
            .collect();
        opener.open(
            challenger,
            preprocessed,
            main,
            after_phase,
            &quotient_data.data,
            quotient_degrees,
        )
    }
}
