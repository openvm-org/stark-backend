use std::{iter, marker::PhantomData};

use itertools::{izip, multiunzip, Itertools};
use p3_challenger::CanObserve;
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;
use tracing::instrument;

use super::{
    hal::{ProverBackend, ProverDevice, QuotientCommitter},
    types::{
        HalProof, MultiStarkProvingKeyView, ProverViewAfterRapPhases, ProvingContext,
        RapSinglePhaseView, RapView,
    },
    Prover,
};
use crate::{
    config::{Com, StarkGenericConfig, Val},
    keygen::view::MultiStarkVerifyingKeyView,
    proof::{AirProofData, Commitments},
    prover::{
        hal::MatrixView,
        metrics::trace_metrics,
        types::{CommittedTraceView, PairView},
    },
    utils::metrics_span,
};

/// Host-to-device coordinator for full prover implementation.
///
/// The generics are:
/// - `SC`: Stark configuration for proving key (from host)
/// - `PB`: Prover backend types
/// - `PD`: Prover device methods
// TODO[jpw]: the SC generic is awkward and should be revisited; only being used for challenger
pub struct Coordinator<SC: StarkGenericConfig, PB, PD> {
    pub backend: PB,
    pub device: PD,
    challenger: SC::Challenger,
    phantom: PhantomData<(SC, PB)>,
}

impl<SC: StarkGenericConfig, PB, PD> Coordinator<SC, PB, PD> {
    pub fn new(backend: PB, device: PD, challenger: SC::Challenger) -> Self {
        Self {
            backend,
            device,
            challenger,
            phantom: PhantomData,
        }
    }
}

impl<SC, PB, PD> Prover for Coordinator<SC, PB, PD>
where
    SC: StarkGenericConfig,
    PB: ProverBackend<
        Val = Val<SC>,
        Challenge = SC::Challenge,
        Commitment = Com<SC>,
        Challenger = SC::Challenger,
    >,
    PD: ProverDevice<PB>,
{
    type Proof = HalProof<PB, PD::RapPartialProof>;
    type ProvingKeyView<'a>
        = MultiStarkProvingKeyView<'a, PB, PD::RapPartialProvingKeyView<'a>>
    where
        Self: 'a;

    type ProvingContext = ProvingContext<PB>;

    /// Specialized prove for InteractiveAirs.
    /// Handles trace generation of the permutation traces.
    /// Assumes the main traces have been generated and committed already.
    ///
    /// The [MultiStarkProvingKeyView] should already be filtered to only include the relevant AIR's proving keys.
    #[instrument(name = "Coordinator::prove", level = "info", skip_all)]
    fn prove<'a>(
        &'a mut self,
        mpk: Self::ProvingKeyView<'a>,
        ctx: Self::ProvingContext,
    ) -> Self::Proof {
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        assert!(mpk.validate(&ctx), "Invalid proof input");

        let (air_ids, air_ctxs): (Vec<_>, Vec<_>) = ctx.into_iter().unzip();
        let num_air = air_ids.len();
        #[allow(clippy::type_complexity)]
        let (cached_commits_per_air, cached_views_per_air, common_main_per_air, pvs_per_air): (
            Vec<Vec<PB::Commitment>>,
            Vec<Vec<CommittedTraceView<PB>>>,
            Vec<Option<PB::MatrixView>>,
            Vec<Vec<PB::Val>>,
        ) = multiunzip(air_ctxs.into_iter().map(|ctx| {
            let (cached_commits, cached_views): (Vec<_>, Vec<_>) =
                ctx.cached_mains.into_iter().unzip();
            (
                cached_commits,
                cached_views,
                ctx.common_main,
                ctx.public_values,
            )
        }));

        // ==================== All trace commitments that do not require challenges ====================
        // Commit all common main traces in a commitment. Traces inside are ordered by AIR id.
        let (common_main_trace_views, (common_main_commit, common_main_pcs_data_view)) =
            metrics_span("main_trace_commit_time_ms", || {
                let trace_views = common_main_per_air.into_iter().flatten().collect_vec();
                let prover_data = self.device.commit(&trace_views);
                (trace_views, prover_data)
            });

        // Commitments order:
        // - for each air:
        //   - for each cached main trace
        //     - 1 commitment
        // - 1 commitment of all common main traces
        let main_trace_commitments: Vec<PB::Commitment> = cached_commits_per_air
            .iter()
            .flatten()
            .chain(iter::once(&common_main_commit))
            .cloned()
            .collect();

        // All commitments that don't require challenges have been made, so we collect them into trace views:
        let mut common_main_idx = 0;
        let mut log_trace_height_per_air: Vec<u8> = Vec::with_capacity(num_air);
        let mut pair_trace_view_per_air = Vec::with_capacity(num_air);
        for (pk, cached_views, pvs) in izip!(&mpk.per_air, &cached_views_per_air, &pvs_per_air) {
            let mut main_trace_views = cached_views
                .iter()
                .map(|view| view.trace.clone())
                .collect_vec();
            if pk.vk.has_common_main() {
                main_trace_views.push(common_main_trace_views[common_main_idx].clone());
                common_main_idx += 1;
            }
            let trace_height = main_trace_views.first().expect("no main trace").height();
            let log_trace_height: u8 = log2_strict_usize(trace_height).try_into().unwrap();
            let pair_trace_view = PairView {
                log_trace_height,
                preprocessed: pk.preprocessed_data.as_ref().map(|d| d.trace.clone()),
                partitioned_main: main_trace_views,
                public_values: pvs.to_vec(),
            };
            log_trace_height_per_air.push(log_trace_height);
            pair_trace_view_per_air.push(pair_trace_view);
        }
        tracing::info!("{}", trace_metrics(&mpk.per_air, &log_trace_height_per_air));
        #[cfg(feature = "bench-metrics")]
        trace_metrics(&mpk.per_air, &log_trace_height_per_air).emit();

        // ============ Challenger observations before additional RAP phases =============
        // Observe public values:
        for pvs in &pvs_per_air {
            self.challenger.observe_slice(pvs);
        }

        // Observes preprocessed and main commitments:
        let mvk = mpk.vk_view();
        let preprocessed_commits = mvk.flattened_preprocessed_commits();
        self.challenger.observe_slice(&preprocessed_commits);
        self.challenger.observe_slice(&main_trace_commitments);
        // Observe trace domain size per air:
        self.challenger.observe_slice(
            &log_trace_height_per_air
                .iter()
                .copied()
                .map(Val::<SC>::from_canonical_u8)
                .collect_vec(),
        );

        // ==================== Partially prove all RAP phases that require challenges ====================
        let (rap_partial_proof, prover_view) = self.device.partially_prove(
            &mut self.challenger,
            &mpk.per_air,
            pair_trace_view_per_air,
        );

        let (commitments_after, pcs_data_view_after): (Vec<_>, Vec<_>) = prover_view
            .committed_pcs_data_per_phase
            .iter()
            .cloned()
            .unzip();
        // Challenger observes additional commitments if any exist:
        for commit in &commitments_after {
            self.challenger.observe(commit.clone());
        }

        // ==================== Quotient polynomial computation and commitment, if any ====================
        // Note[jpw]: Currently we always call this step, we could add a flag to skip it for protocols that
        // do not require quotient poly.
        let extended_rap_views = create_trace_view_per_air(
            &self.device,
            &mpk,
            &log_trace_height_per_air,
            &cached_views_per_air,
            &common_main_pcs_data_view,
            &pvs_per_air,
            &prover_view,
        );
        let (constraints, quotient_degrees): (Vec<_>, Vec<_>) = mpk
            .vk_view()
            .per_air
            .iter()
            .map(|vk| (&vk.symbolic_constraints.constraints, vk.quotient_degree))
            .unzip();
        let (quotient_commit, quotient_data) = self.device.eval_and_commit_quotient(
            &mut self.challenger,
            &constraints,
            extended_rap_views,
            &quotient_degrees,
        );
        // Observe quotient commitment
        self.challenger.observe(quotient_commit.clone());

        // ==================== Polynomial Opening Proofs ====================
        let opening = metrics_span("pcs_opening_time_ms", || {
            let preprocessed = mpk
                .per_air
                .iter()
                .flat_map(|pk| pk.preprocessed_data.as_ref().map(|d| d.data.clone()))
                .collect_vec();
            let main = cached_views_per_air
                .into_iter()
                .flatten()
                .map(|cv| cv.data)
                .chain(iter::once(common_main_pcs_data_view))
                .collect_vec();
            self.device.open(
                &mut self.challenger,
                preprocessed,
                main,
                pcs_data_view_after,
                quotient_data,
                &quotient_degrees,
            )
        });

        // ==================== Collect data into proof ====================
        // Collect the commitments
        let commitments = Commitments {
            main_trace: main_trace_commitments,
            after_challenge: commitments_after,
            quotient: quotient_commit,
        };
        // transpose per_phase, per_air -> per_air, per_phase
        let exposed_values_per_air = (0..num_air)
            .map(|i| {
                let mut values = prover_view
                    .rap_views_per_phase
                    .iter()
                    .map(|per_air| {
                        per_air
                            .get(i)
                            .and_then(|v| v.inner.map(|_| v.exposed_values.clone()))
                    })
                    .collect_vec();
                // Prune Nones
                while let Some(last) = values.last() {
                    if last.is_none() {
                        values.pop();
                    } else {
                        break;
                    }
                }
                values
                    .into_iter()
                    .map(|v| v.unwrap_or_default())
                    .collect_vec()
            })
            .collect_vec();
        let proof = HalProof {
            commitments,
            opening,
            per_air: izip!(
                mpk.air_ids,
                log_trace_height_per_air,
                exposed_values_per_air,
                pvs_per_air
            )
            .map(
                |(air_id, log_height, exposed_values, public_values)| AirProofData {
                    air_id,
                    degree: 1 << log_height,
                    public_values,
                    exposed_values_after_challenge: exposed_values,
                },
            )
            .collect(),
            rap_partial_proof,
        };

        #[cfg(feature = "bench-metrics")]
        ::metrics::gauge!("stark_prove_excluding_trace_time_ms")
            .set(start.elapsed().as_millis() as f64);

        proof
    }
}

impl<'a, PB: ProverBackend, R> MultiStarkProvingKeyView<'a, PB, R> {
    pub(crate) fn validate(&self, ctx: &ProvingContext<PB>) -> bool {
        if ctx.per_air.len() != self.air_ids.len() {
            return false;
        }
        if !ctx
            .per_air
            .iter()
            .zip(&self.air_ids)
            .all(|((id1, _), id2)| id1 == id2)
        {
            return false;
        }
        if !ctx.per_air.iter().tuple_windows().all(|(a, b)| a.0 < b.0) {
            return false;
        }
        true
    }

    pub(crate) fn vk_view(&self) -> MultiStarkVerifyingKeyView<'a, PB::Val, PB::Commitment> {
        MultiStarkVerifyingKeyView::new(self.per_air.iter().map(|pk| pk.vk).collect())
    }
}

/// Takes in views of pcs data and returns extended views of all matrices evaluated on quotient domains
/// for quotient poly calculation.
fn create_trace_view_per_air<PB: ProverBackend, R>(
    device: &impl QuotientCommitter<PB>,
    mpk: &MultiStarkProvingKeyView<PB, R>,
    log_trace_height_per_air: &[u8],
    cached_views_per_air: &[Vec<CommittedTraceView<PB>>],
    common_main_pcs_data: &PB::PcsDataView,
    pvs_per_air: &[Vec<PB::Val>],
    view_after: &ProverViewAfterRapPhases<PB>,
) -> Vec<RapView<PB::MatrixView, PB::Val, PB::Challenge>> {
    let mut common_main_idx = 0;
    izip!(
        &mpk.per_air,
        log_trace_height_per_air,
        cached_views_per_air,
        pvs_per_air
    )
    .enumerate()
    .map(|(i, (pk, &log_trace_height, cached_views, pvs))| {
        let quotient_degree = pk.vk.quotient_degree;
        // The AIR will be treated as the full RAP with virtual columns after this
        let preprocessed = pk.preprocessed_data.as_ref().map(|cv| {
            device
                .get_extended_matrix(&cv.data, cv.matrix_idx as usize, quotient_degree)
                .unwrap()
        });
        let mut partitioned_main: Vec<_> = cached_views
            .iter()
            .map(|cv| {
                device
                    .get_extended_matrix(&cv.data, cv.matrix_idx as usize, quotient_degree)
                    .unwrap()
            })
            .collect();
        if pk.vk.has_common_main() {
            partitioned_main.push(
                device
                    .get_extended_matrix(common_main_pcs_data, common_main_idx, quotient_degree)
                    .unwrap_or_else(|| {
                        panic!("common main commitment could not get matrix_idx={common_main_idx}")
                    }),
            );
            common_main_idx += 1;
        }
        let pair = PairView {
            log_trace_height,
            preprocessed,
            partitioned_main,
            public_values: pvs.to_vec(),
        };
        let mut per_phase = view_after
            .committed_pcs_data_per_phase
            .iter()
            .zip_eq(&view_after.rap_views_per_phase)
            .map(|((_, pcs_data), rap_views)| -> Option<RapSinglePhaseView<PB::MatrixView, PB::Challenge>> {
                let rap_view = rap_views.get(i)?;
                let matrix_idx = rap_view.inner?;
                let extended_matrix = device.get_extended_matrix(pcs_data, matrix_idx, quotient_degree);
                let extended_matrix = extended_matrix.unwrap_or_else(|| {
                    panic!("could not get matrix_idx={matrix_idx} for rap {i}")
                });
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
        let per_phase = per_phase.into_iter().map(|v| v.unwrap_or_default()).collect();

        RapView { pair, per_phase }
    })
    .collect()
}
