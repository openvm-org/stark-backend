use std::{iter::zip, marker::PhantomData, ops::Deref, sync::Arc};

use derivative::Derivative;
use itertools::{izip, zip_eq, Itertools};
use opener::OpeningProver;
use p3_challenger::FieldChallenger;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::FieldExtensionAlgebra;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use quotient::QuotientCommitter;

use super::{
    hal::{self, DeviceDataTransporter, MatrixDimensions, ProverBackend, ProverDevice},
    types::{
        DeviceMultiStarkProvingKey, DeviceStarkProvingKey, PairView, ProverDataAfterRapPhases,
        RapView, SingleCommitPreimage,
    },
};
use crate::{
    air_builders::symbolic::SymbolicConstraints,
    config::{
        Com, PcsProof, PcsProverData, RapPartialProvingKey, RapPhaseSeqPartialProof,
        StarkGenericConfig, Val,
    },
    interaction::RapPhaseSeq,
    keygen::types::MultiStarkProvingKey,
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
    type RapPartialProof = Option<RapPhaseSeqPartialProof<SC>>;
    type Commitment = Com<SC>;
    type Challenger = SC::Challenger;
    type Matrix = Arc<RowMajorMatrix<Val<SC>>>;
    type PcsData = PcsData<SC>;
    type RapPartialProvingKey = RapPartialProvingKey<SC>;
}

#[derive(Derivative)]
#[derivative(Clone(bound = ""))]
pub struct PcsData<SC: StarkGenericConfig> {
    /// The preimage of a single commitment.
    pub data: Arc<PcsProverData<SC>>,
    /// A mixed matrix commitment scheme commits to multiple trace matrices within a single commitment.
    /// This is the ordered list of log2 heights of all committed trace matrices.
    pub log_trace_heights: Vec<u8>,
}

impl<T: Send + Sync + Clone> MatrixDimensions for Arc<RowMajorMatrix<T>> {
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

impl<SC: StarkGenericConfig> TraceCommitter<CpuBackend<SC>> for CpuDevice<'_, SC> {
    fn commit(&self, traces: &[Arc<RowMajorMatrix<Val<SC>>>]) -> (Com<SC>, PcsData<SC>) {
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
            PcsData {
                data: Arc::new(data),
                log_trace_heights,
            },
        )
    }
}

impl<SC: StarkGenericConfig> hal::RapPartialProver<CpuBackend<SC>> for CpuDevice<'_, SC> {
    fn partially_prove<'a>(
        &self,
        challenger: &mut SC::Challenger,
        mpk: &DeviceMultiStarkProvingKey<'a, CpuBackend<SC>>,
        trace_views: Vec<PairView<&'a Arc<RowMajorMatrix<Val<SC>>>, Val<SC>>>,
    ) -> (
        Option<RapPhaseSeqPartialProof<SC>>,
        ProverDataAfterRapPhases<CpuBackend<SC>>,
    ) {
        let num_airs = mpk.per_air.len();
        assert_eq!(num_airs, trace_views.len());

        let (constraints_per_air, rap_pk_per_air): (Vec<_>, Vec<_>) = mpk
            .per_air
            .iter()
            .map(|pk| {
                (
                    SymbolicConstraints::from(&pk.vk.symbolic_constraints),
                    &pk.rap_partial_pk,
                )
            })
            .unzip();

        let trace_views = trace_views
            .iter()
            .map(|v| PairView {
                log_trace_height: v.log_trace_height,
                preprocessed: v.preprocessed.as_ref().map(|p| p.as_ref()),
                partitioned_main: v.partitioned_main.iter().map(|m| m.as_ref()).collect(),
                public_values: v.public_values.clone(),
            })
            .collect_vec();
        let (rap_phase_seq_proof, rap_phase_seq_data) = self
            .config()
            .rap_phase_seq()
            .partially_prove(
                challenger,
                &constraints_per_air.iter().collect_vec(),
                &rap_pk_per_air,
                &trace_views,
            )
            .map_or((None, None), |(p, d)| (Some(p), Some(d)));

        let mvk_view = mpk.vk_view();

        let mut perm_matrix_idx = 0usize;
        let rap_views_per_phase;
        let perm_trace_per_air = if let Some(phase_data) = rap_phase_seq_data {
            assert_eq!(mvk_view.num_phases(), 1);
            assert_eq!(phase_data.challenges_per_phase.len(), 1);
            // mvk_view.num_challenges_in_phase are the number of challenge variables used in the AIR constraints.
            // there may be additional randomness used to generate the trace data that is constrained externally.
            assert!(
                mvk_view.num_challenges_in_phase(0) <= phase_data.challenges_per_phase[0].len()
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
                    challenges: phase_data.challenges_per_phase[0].clone(),
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
        let committed_pcs_data_per_phase: Vec<(Com<SC>, PcsData<SC>)> =
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
        let prover_view = ProverDataAfterRapPhases {
            committed_pcs_data_per_phase,
            rap_views_per_phase,
        };
        (rap_phase_seq_proof, prover_view)
    }
}

impl<SC: StarkGenericConfig> hal::QuotientCommitter<CpuBackend<SC>> for CpuDevice<'_, SC> {
    fn eval_and_commit_quotient(
        &self,
        challenger: &mut SC::Challenger,
        pk_views: &[DeviceStarkProvingKey<CpuBackend<SC>>],
        public_values: &[Vec<Val<SC>>],
        cached_views_per_air: &[Vec<
            SingleCommitPreimage<&Arc<RowMajorMatrix<Val<SC>>>, &PcsData<SC>>,
        >],
        common_main_pcs_data: &PcsData<SC>,
        prover_data_after: &ProverDataAfterRapPhases<CpuBackend<SC>>,
    ) -> (Com<SC>, PcsData<SC>) {
        let pcs = self.pcs();
        // Generate `alpha` challenge
        let alpha: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("alpha: {alpha:?}");
        // Prepare extended views:
        let mut common_main_idx = 0;
        let extended_views = izip!(pk_views, cached_views_per_air, public_values)
            .enumerate()
            .map(|(i, (pk, cached_views, pvs))| {
                let quotient_degree = pk.vk.quotient_degree;
                let log_trace_height = if pk.vk.has_common_main() {
                    common_main_pcs_data.log_trace_heights[common_main_idx]
                } else {
                    log2_strict_usize(cached_views[0].trace.height()) as u8
                };
                let trace_domain = pcs.natural_domain_for_degree(1usize << log_trace_height);
                let quotient_domain = trace_domain
                    .create_disjoint_domain(trace_domain.size() * quotient_degree as usize);
                // **IMPORTANT**: the return type of `get_evaluations_on_domain` is a matrix view. DO NOT call to_row_major_matrix as this will allocate new memory
                let preprocessed = pk.preprocessed_data.as_ref().map(|cv| {
                    pcs.get_evaluations_on_domain(
                        &cv.data.data,
                        cv.matrix_idx as usize,
                        quotient_domain,
                    )
                });
                let mut partitioned_main: Vec<_> = cached_views
                    .iter()
                    .map(|cv| {
                        pcs.get_evaluations_on_domain(
                            &cv.data.data,
                            cv.matrix_idx as usize,
                            quotient_domain,
                        )
                    })
                    .collect();
                if pk.vk.has_common_main() {
                    partitioned_main.push(pcs.get_evaluations_on_domain(
                        &common_main_pcs_data.data,
                        common_main_idx,
                        quotient_domain,
                    ));
                    common_main_idx += 1;
                }
                let pair = PairView {
                    log_trace_height,
                    preprocessed,
                    partitioned_main,
                    public_values: pvs.to_vec(),
                };
                let mut per_phase = zip(
                    &prover_data_after.committed_pcs_data_per_phase,
                    &prover_data_after.rap_views_per_phase,
                )
                .map(|((_, pcs_data), rap_views)| -> Option<_> {
                    let rap_view = rap_views.get(i)?;
                    let matrix_idx = rap_view.inner?;
                    let extended_matrix =
                        pcs.get_evaluations_on_domain(&pcs_data.data, matrix_idx, quotient_domain);
                    Some(RapSinglePhaseView {
                        inner: Some(extended_matrix),
                        challenges: rap_view.challenges.clone(),
                        exposed_values: rap_view.exposed_values.clone(),
                    })
                })
                .collect_vec();
                while let Some(last) = per_phase.last() {
                    if last.is_none() {
                        per_phase.pop();
                    } else {
                        break;
                    }
                }
                let per_phase = per_phase
                    .into_iter()
                    .map(|v| v.unwrap_or_default())
                    .collect();

                RapView { pair, per_phase }
            })
            .collect_vec();

        let (constraints, quotient_degrees): (Vec<_>, Vec<_>) = pk_views
            .iter()
            .map(|pk| {
                (
                    &pk.vk.symbolic_constraints.constraints,
                    pk.vk.quotient_degree,
                )
            })
            .unzip();
        let qc = QuotientCommitter::new(self.pcs(), alpha);
        let quotient_values = metrics_span("quotient_poly_compute_time_ms", || {
            qc.quotient_values(&constraints, extended_views, &quotient_degrees)
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
        preprocessed: Vec<&PcsData<SC>>,
        // For each main trace commitment, the prover data and
        // the log height of each matrix, in order
        // Note: this is all one challenge phase.
        main: Vec<&PcsData<SC>>,
        // `after_phase[i]` has shared commitment prover data for all matrices in phase `i + 1`.
        after_phase: Vec<PcsData<SC>>,
        // Quotient poly commitment prover data
        quotient_data: PcsData<SC>,
        // Quotient degree for each RAP committed in quotient_data, in order
        quotient_degrees: &[u8],
    ) -> OpeningProof<PcsProof<SC>, SC::Challenge> {
        // Draw `zeta` challenge
        let zeta: SC::Challenge = challenger.sample_ext_element();
        tracing::debug!("zeta: {zeta:?}");

        let pcs = self.pcs();
        let domain = |log_height| pcs.natural_domain_for_degree(1usize << log_height);

        let after_phase_domains = after_phase.first().map(|v| {
            v.log_trace_heights
                .iter()
                .copied()
                .map(domain)
                .collect_vec()
        });
        let extra_after_challenge_points = self
            .config
            .rap_phase_seq()
            .extra_opening_points(zeta, &after_phase_domains.unwrap_or_default());

        let opener = OpeningProver::<SC>::new(pcs, zeta, extra_after_challenge_points);
        let preprocessed = preprocessed
            .iter()
            .map(|v| {
                assert_eq!(v.log_trace_heights.len(), 1);
                (v.data.as_ref(), domain(v.log_trace_heights[0]))
            })
            .collect();
        let main = main
            .iter()
            .map(|v| {
                let domains = v.log_trace_heights.iter().copied().map(domain).collect();
                (v.data.as_ref(), domains)
            })
            .collect();
        let after_phase: Vec<_> = after_phase
            .iter()
            .map(|v| {
                let domains = v.log_trace_heights.iter().copied().map(domain).collect();
                (v.data.as_ref(), domains)
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

impl<SC> DeviceDataTransporter<SC, CpuBackend<SC>> for CpuBackend<SC>
where
    SC: StarkGenericConfig,
{
    fn transport_pk_to_device<'a>(
        &self,
        mpk: &'a MultiStarkProvingKey<SC>,
        air_ids: Vec<usize>,
    ) -> DeviceMultiStarkProvingKey<'a, CpuBackend<SC>>
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
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|pd| {
                    let pcs_data_view = PcsData {
                        data: pd.data.clone(),
                        log_trace_heights: vec![log2_strict_usize(pd.trace.height()) as u8],
                    };
                    SingleCommitPreimage {
                        trace: pd.trace.clone(),
                        data: pcs_data_view,
                        matrix_idx: 0,
                    }
                });
                DeviceStarkProvingKey {
                    air_name: &pk.air_name,
                    vk: &pk.vk,
                    preprocessed_data,
                    rap_partial_pk: pk.rap_partial_pk.clone(),
                }
            })
            .collect();
        DeviceMultiStarkProvingKey::new(
            air_ids,
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.vk_pre_hash.clone(),
        )
    }
    fn transport_matrix_to_device(
        &self,
        matrix: &Arc<RowMajorMatrix<Val<SC>>>,
    ) -> Arc<RowMajorMatrix<Val<SC>>> {
        matrix.clone()
    }

    fn transport_pcs_data_to_device(&self, data: &PcsData<SC>) -> PcsData<SC> {
        data.clone()
    }
}
