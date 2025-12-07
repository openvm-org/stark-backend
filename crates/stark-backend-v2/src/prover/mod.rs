use itertools::{izip, Itertools};
use openvm_stark_backend::prover::{MatrixDimensions, Prover};
use p3_field::FieldAlgebra;
use p3_util::log2_strict_usize;
use tracing::{info, info_span, instrument};

#[cfg(feature = "metrics")]
use crate::prover::metrics::trace_metrics;
use crate::{
    poseidon2::sponge::FiatShamirTranscript,
    proof::{BatchConstraintProof, GkrProof, Proof, StackingProof, TraceVData, WhirProof},
    Digest, EF, F,
};

mod cpu_backend;
mod hal;
mod logup_zerocheck;
mod matrix;
pub mod metrics;
pub mod poly;
pub mod stacked_pcs;
pub mod stacked_reduction;
pub mod sumcheck;
mod types;
pub mod whir;

pub use cpu_backend::*;
pub use hal::*;
pub use logup_zerocheck::*;
pub use matrix::*;
pub use types::*;

#[derive(derive_new::new)]
pub struct CoordinatorV2<PB: ProverBackendV2, PD, TS> {
    pub backend: PB,
    pub device: PD,
    pub(crate) transcript: TS,
}

impl<PB, PD, TS> Prover for CoordinatorV2<PB, PD, TS>
where
    // TODO[jpw]: make generic in F, EF, Commitment
    PB: ProverBackendV2<Val = F, Challenge = EF, Commitment = Digest>,
    PD: ProverDeviceV2<PB, TS>,
    PD::Artifacts: Into<PD::OpeningPoints>,
    PD::PartialProof: Into<(GkrProof, BatchConstraintProof)>,
    PD::OpeningProof: Into<(StackingProof, WhirProof)>,
    TS: FiatShamirTranscript,
{
    type Proof = Proof;
    type ProvingKeyView<'a>
        = &'a DeviceMultiStarkProvingKeyV2<PB>
    where
        Self: 'a;

    type ProvingContext<'a>
        = ProvingContextV2<PB>
    where
        Self: 'a;

    /// Specialized prove for InteractiveAirs.
    /// Handles trace generation of the permutation traces.
    /// Assumes the main traces have been generated and committed already.
    ///
    /// The [DeviceMultiStarkProvingKey] should already be filtered to only include the relevant
    /// AIR's proving keys.
    #[instrument(
        name = "stark_prove_excluding_trace",
        level = "info",
        skip_all,
        fields(phase = "prover")
    )]
    fn prove<'a>(
        &'a mut self,
        mpk: &'a DeviceMultiStarkProvingKeyV2<PB>,
        unsorted_ctx: ProvingContextV2<PB>,
    ) -> Self::Proof {
        assert_eq!(self.device.config(), mpk.params);
        let transcript = &mut self.transcript;
        transcript.observe_commit(mpk.vk_pre_hash);

        let ctx = unsorted_ctx.into_sorted();
        // `ctx` should NOT be permuted anymore: the ordering by `trace_idx` is now fixed.

        let num_airs_present = ctx.per_trace.len();
        info!(num_airs_present);

        let _main_commit_span = info_span!("prover.main_trace_commit", phase = "prover").entered();
        let (common_main_commit, common_main_pcs_data) = {
            let traces = ctx
                .common_main_traces()
                .map(|(_, trace)| trace)
                .collect_vec();
            self.device.commit(&traces)
        };

        let mut trace_vdata: Vec<Option<TraceVData>> = vec![None; mpk.per_air.len()];
        let mut public_values: Vec<Vec<F>> = vec![Vec::new(); mpk.per_air.len()];

        // Hypercube dimension per trace (present AIR)
        for (air_id, air_ctx) in &ctx.per_trace {
            let trace_height = air_ctx.common_main.height();
            let log_height = log2_strict_usize(trace_height);

            trace_vdata[*air_id] = Some(TraceVData {
                log_height,
                cached_commitments: air_ctx
                    .cached_mains
                    .iter()
                    .map(|cd| cd.commitment)
                    .collect(),
            });
            public_values[*air_id] = air_ctx.public_values.clone();
        }
        #[cfg(feature = "metrics")]
        trace_metrics(mpk, &trace_vdata);

        // Only observe commits for present AIRs.
        // Commitments order:
        // - 1 commitment of all common main traces
        // - for each air:
        //   - preprocessed commit if present
        //   - for each cached main trace
        //     - 1 commitment
        transcript.observe_commit(common_main_commit);
        drop(_main_commit_span);

        for (trace_vdata, pvs, pk) in izip!(&trace_vdata, &public_values, &mpk.per_air) {
            if !pk.vk.is_required {
                transcript.observe(F::from_bool(trace_vdata.is_some()));
            }
            if let Some(trace_vdata) = trace_vdata {
                if let Some(cd) = &pk.preprocessed_data {
                    transcript.observe_commit(cd.commitment);
                } else {
                    transcript.observe(F::from_canonical_usize(trace_vdata.log_height));
                }
                for commit in &trace_vdata.cached_commitments {
                    transcript.observe_commit(*commit);
                }
            }
            for pv in pvs {
                transcript.observe(*pv);
            }
        }

        let (constraints_proof, r) =
            self.device
                .prove_rap_constraints(transcript, mpk, &ctx, &common_main_pcs_data);

        let opening_proof =
            self.device
                .prove_openings(transcript, mpk, ctx, common_main_pcs_data, r.into());

        let (gkr_proof, batch_constraint_proof) = constraints_proof.into();
        let (stacking_proof, whir_proof) = opening_proof.into();

        Proof {
            public_values,
            trace_vdata,
            common_main_commit,
            gkr_proof,
            batch_constraint_proof,
            stacking_proof,
            whir_proof,
        }
    }
}
