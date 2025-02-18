use std::{iter, marker::PhantomData};

use itertools::{izip, Itertools};
use p3_challenger::CanObserve;
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;
use tracing::instrument;

use super::{
    hal::{ProverBackend, ProverDevice},
    types::{DeviceMultiStarkProvingKey, HalProof, ProvingContext},
    Prover,
};
use crate::{
    config::{Com, StarkGenericConfig, Val},
    keygen::view::MultiStarkVerifyingKeyView,
    proof::{AirProofData, Commitments},
    prover::{
        hal::MatrixDimensions,
        metrics::trace_metrics,
        types::{PairView, SingleCommitPreimage},
    },
    utils::metrics_span,
};

/// Host-to-device coordinator for full prover implementation.
///
/// The generics are:
/// - `SC`: Stark configuration for proving key (from host)
/// - `PB`: Prover backend types
/// - `PD`: Prover device methods
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
    type Proof = HalProof<PB>;
    type ProvingKeyView<'a>
        = &'a DeviceMultiStarkProvingKey<'a, PB>
    where
        Self: 'a;

    type ProvingContext<'a>
        = ProvingContext<'a, PB>
    where
        Self: 'a;

    /// Specialized prove for InteractiveAirs.
    /// Handles trace generation of the permutation traces.
    /// Assumes the main traces have been generated and committed already.
    ///
    /// The [DeviceMultiStarkProvingKey] should already be filtered to only include the relevant AIR's proving keys.
    #[instrument(name = "Coordinator::prove", level = "info", skip_all)]
    fn prove<'a>(
        &'a mut self,
        mpk: Self::ProvingKeyView<'a>,
        ctx: Self::ProvingContext<'a>,
    ) -> Self::Proof {
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        assert!(mpk.validate(&ctx), "Invalid proof input");

        let num_air = ctx.per_air.len();
        let (cached_commits_per_air, cached_views_per_air, common_main_per_air, pvs_per_air): (
            Vec<Vec<PB::Commitment>>,
            Vec<Vec<SingleCommitPreimage<&'a PB::Matrix, &'a PB::PcsData>>>,
            Vec<Option<PB::Matrix>>,
            Vec<Vec<PB::Val>>,
        ) = ctx
            .into_iter()
            .map(|(_, ctx)| {
                let (cached_commits, cached_views): (Vec<_>, Vec<_>) =
                    ctx.cached_mains.into_iter().unzip();
                (
                    cached_commits,
                    cached_views,
                    ctx.common_main,
                    ctx.public_values,
                )
            })
            .multiunzip();

        // ==================== All trace commitments that do not require challenges ====================
        // Commit all common main traces in a commitment. Traces inside are ordered by AIR id.
        let (common_main_traces, (common_main_commit, common_main_pcs_data)) =
            metrics_span("main_trace_commit_time_ms", || {
                let traces = common_main_per_air.into_iter().flatten().collect_vec();
                let prover_data = self.device.commit(&traces);
                (traces, prover_data)
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
            let mut main_trace_views: Vec<&PB::Matrix> =
                cached_views.iter().map(|view| view.trace).collect_vec();
            if pk.vk.has_common_main() {
                main_trace_views.push(&common_main_traces[common_main_idx]);
                common_main_idx += 1;
            }
            let trace_height = main_trace_views.first().expect("no main trace").height();
            let log_trace_height: u8 = log2_strict_usize(trace_height).try_into().unwrap();
            let pair_trace_view = PairView {
                log_trace_height,
                preprocessed: pk.preprocessed_data.as_ref().map(|d| &d.trace),
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
        let (rap_partial_proof, prover_data_after) = self.device.partially_prove(
            &mut self.challenger,
            &mpk.per_air,
            pair_trace_view_per_air,
        );
        // Challenger observes additional commitments if any exist:
        for (commit, _) in &prover_data_after.committed_pcs_data_per_phase {
            self.challenger.observe(commit.clone());
        }

        // Collect exposed_values_per_air for the proof:
        // - transpose per_phase, per_air -> per_air, per_phase
        let exposed_values_per_air: Vec<Vec<_>> = (0..num_air)
            .map(|i| {
                let mut values: Vec<_> = prover_data_after
                    .rap_views_per_phase
                    .iter()
                    .filter_map(|per_air| {
                        per_air
                            .get(i)?
                            .inner
                            .map(|_| per_air[i].exposed_values.clone())
                    })
                    .collect();

                while values.last().map_or(false, |v| v.is_empty()) {
                    values.pop();
                }

                values
            })
            .collect();

        // ==================== Quotient polynomial computation and commitment, if any ====================
        // Note[jpw]: Currently we always call this step, we could add a flag to skip it for protocols that
        // do not require quotient poly.
        let (quotient_commit, quotient_data) = self.device.eval_and_commit_quotient(
            &mut self.challenger,
            &mpk.per_air,
            &pvs_per_air,
            &cached_views_per_air,
            &common_main_pcs_data,
            &prover_data_after,
        );
        // Observe quotient commitment
        self.challenger.observe(quotient_commit.clone());

        let (commitments_after, pcs_data_after): (Vec<_>, Vec<_>) = prover_data_after
            .committed_pcs_data_per_phase
            .into_iter()
            .unzip();
        // ==================== Polynomial Opening Proofs ====================
        let opening = metrics_span("pcs_opening_time_ms", || {
            let quotient_degrees = mpk
                .per_air
                .iter()
                .map(|pk| pk.vk.quotient_degree)
                .collect_vec();
            let preprocessed = mpk
                .per_air
                .iter()
                .flat_map(|pk| pk.preprocessed_data.as_ref().map(|d| &d.data))
                .collect_vec();
            let main = cached_views_per_air
                .into_iter()
                .flatten()
                .map(|cv| cv.data)
                .chain(iter::once(&common_main_pcs_data))
                .collect_vec();
            self.device.open(
                &mut self.challenger,
                preprocessed,
                main,
                pcs_data_after,
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
        let proof = HalProof {
            commitments,
            opening,
            per_air: izip!(
                &mpk.air_ids,
                log_trace_height_per_air,
                exposed_values_per_air,
                pvs_per_air
            )
            .map(
                |(&air_id, log_height, exposed_values, public_values)| AirProofData {
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

impl<'a, PB: ProverBackend> DeviceMultiStarkProvingKey<'a, PB> {
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
