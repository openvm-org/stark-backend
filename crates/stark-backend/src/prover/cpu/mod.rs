use std::{marker::PhantomData, ops::Deref, sync::Arc};

use derivative::Derivative;
use itertools::{zip_eq, Itertools};
use opener::OpeningProver;
use p3_challenger::FieldChallenger;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::FieldExtensionAlgebra;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use quotient::QuotientCommitter;

use super::{
    hal::{self, DeviceDataTransporterpter, MatrixView, ProverBackend, ProverDevice},
    types::{
        CommittedTraceView, MultiStarkProvingKeyView, PairView, ProverViewAfterRapPhases, RapView,
        StarkProvingKeyView,
    },
};
use crate::{
    air_builders::symbolic::{SymbolicConstraints, SymbolicExpressionDag},
    config::{
        Com, PcsProof, PcsProverData, RapPhaseSeqPartialProof, RapPhaseSeqProvingKey,
        StarkGenericConfig, Val,
    },
    interaction::RapPhaseSeq,
    keygen::{types::MultiStarkProvingKey, view::MultiStarkVerifyingKeyView},
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
#[derive(Derivative)]
#[derivative(Clone(bound = ""), Copy(bound = ""), Default(bound = ""))]
pub struct CpuBackend<SC> {
    phantom: PhantomData<SC>,
}

#[derive(Derivative, derive_new::new)]
#[derivative(Clone(bound = ""), Copy(bound = ""))]
pub struct CpuDevice<'a, SC> {
    config: &'a SC,
}

impl<SC: StarkGenericConfig> ProverBackend for CpuBackend<SC> {
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

impl<SC> CpuDevice<'_, SC> {
    pub fn config(&self) -> &SC {
        self.config
    }
}

impl<SC: StarkGenericConfig> CpuDevice<'_, SC> {
    pub fn pcs(&self) -> &SC::Pcs {
        self.config.pcs()
    }
}

impl<SC: StarkGenericConfig> ProverDevice<CpuBackend<SC>> for CpuDevice<'_, SC> {}

impl<SC: StarkGenericConfig> hal::TraceCommitter<CpuBackend<SC>> for CpuDevice<'_, SC> {
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

impl<SC: StarkGenericConfig> hal::RapPartialProver<CpuBackend<SC>> for CpuDevice<'_, SC> {
    // `None` when there is no sumcheck
    type RapPartialProof = Option<RapPhaseSeqPartialProof<SC>>;
    type RapPartialProvingKeyView<'a>
        = &'a RapPhaseSeqProvingKey<SC>
    where
        Self: 'a;

    fn partially_prove<'a>(
        &self,
        challenger: &mut SC::Challenger,
        pk_views: &[StarkProvingKeyView<'a, CpuBackend<SC>, &'a RapPhaseSeqProvingKey<SC>>],
        trace_views: Vec<PairView<Arc<RowMajorMatrix<Val<SC>>>, Val<SC>>>,
    ) -> (
        Self::RapPartialProof,
        ProverViewAfterRapPhases<CpuBackend<SC>>,
    ) {
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
                &constraints_per_air.iter().collect_vec(),
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

type RapLdeView<SC> =
    RapView<Arc<RowMajorMatrix<Val<SC>>>, Val<SC>, <SC as StarkGenericConfig>::Challenge>;

impl<SC: StarkGenericConfig> hal::QuotientCommitter<CpuBackend<SC>> for CpuDevice<'_, SC> {
    fn get_extended_matrix(
        &self,
        view: &PcsDataView<SC>,
        matrix_idx: usize,
        quotient_degree: u8,
    ) -> Option<Arc<RowMajorMatrix<Val<SC>>>> {
        let pcs = self.pcs();
        let log_trace_height = *view.log_trace_heights.get(matrix_idx)?;
        let trace_domain = pcs.natural_domain_for_degree(1usize << log_trace_height);
        let quotient_domain =
            trace_domain.create_disjoint_domain(trace_domain.size() * quotient_degree as usize);
        // NOTE[jpw]: (perf) this clones under the hood!
        let lde_matrix = self
            .pcs()
            .get_evaluations_on_domain(&view.data, matrix_idx, quotient_domain)
            .to_row_major_matrix();
        Some(Arc::new(lde_matrix))
    }

    fn eval_and_commit_quotient(
        &self,
        challenger: &mut SC::Challenger,
        constraints: &[&SymbolicExpressionDag<Val<SC>>],
        extended_views: Vec<RapLdeView<SC>>,
        quotient_degrees: &[u8],
    ) -> (Com<SC>, PcsDataView<SC>) {
        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");

        let qc = QuotientCommitter::new(self.pcs(), alpha);
        let quotient_values = metrics_span("quotient_poly_compute_time_ms", || {
            qc.quotient_values(constraints, extended_views, quotient_degrees)
        });

        // Commit to quotient polynomials. One shared commit for all quotient polynomials
        metrics_span("quotient_poly_commit_time_ms", || {
            qc.commit(quotient_values)
        })
    }
}

impl<SC: StarkGenericConfig> hal::OpeningProver<CpuBackend<SC>> for CpuDevice<'_, SC> {
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
        let domain = |log_height| pcs.natural_domain_for_degree(1usize << log_height);
        let opener = OpeningProver::<SC>::new(pcs, zeta);
        let preprocessed = preprocessed
            .iter()
            .map(|v| {
                assert_eq!(v.log_trace_heights.len(), 1);
                (v.data.deref(), domain(v.log_trace_heights[0]))
            })
            .collect();
        let main = main
            .iter()
            .map(|v| {
                let domains = v.log_trace_heights.iter().copied().map(domain).collect();
                (v.data.deref(), domains)
            })
            .collect();
        let after_phase: Vec<_> = after_phase
            .iter()
            .map(|v| {
                let domains = v.log_trace_heights.iter().copied().map(domain).collect();
                (v.data.deref(), domains)
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

impl<SC> DeviceDataTransporterpter<SC, CpuBackend<SC>> for CpuBackend<SC>
where
    SC: StarkGenericConfig,
{
    fn transport_pk_to_device<'a>(
        &self,
        mpk: &'a MultiStarkProvingKey<SC>,
        air_ids: Vec<usize>,
    ) -> MultiStarkProvingKeyView<'a, CpuBackend<SC>, &'a RapPhaseSeqProvingKey<SC>>
    where
        SC: 'a,
    {
        assert!(
            air_ids.len() <= mpk.per_air.len(),
            "filtering more AIRs than available"
        );
        let per_air = air_ids
            .iter()
            .map(|&air_idx| {
                let pk = &mpk.per_air[air_idx];
                let preprocessed_data = pk.preprocessed_data.clone().map(|pd| {
                    let pcs_data_view = PcsDataView {
                        data: pd.data,
                        log_trace_heights: vec![log2_strict_usize(pd.trace.height()) as u8],
                    };
                    CommittedTraceView {
                        trace: pd.trace,
                        data: pcs_data_view,
                        matrix_idx: 0,
                    }
                });
                StarkProvingKeyView {
                    air_name: &pk.air_name,
                    vk: &pk.vk,
                    preprocessed_data,
                    rap_partial_pk: &pk.rap_phase_seq_pk,
                }
            })
            .collect();
        MultiStarkProvingKeyView::new(air_ids, per_air)
    }
    fn transport_matrix_to_device(
        &self,
        matrix: &Arc<RowMajorMatrix<Val<SC>>>,
    ) -> Arc<RowMajorMatrix<Val<SC>>> {
        matrix.clone()
    }

    fn transport_pcs_data_to_device(&self, data: &PcsDataView<SC>) -> PcsDataView<SC> {
        data.clone()
    }
}
