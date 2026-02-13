//! CPU [ProverBackend] trait implementation.

use std::marker::PhantomData;

use getset::Getters;
use itertools::Itertools;
use p3_field::{ExtensionField, TwoAdicField};

use crate::{
    keygen::types::MultiStarkProvingKey,
    poly_common::Squarable,
    proof::{BatchConstraintProof, GkrProof, StackingProof, WhirProof},
    prover::{
        prove_zerocheck_and_logup,
        stacked_pcs::{stacked_commit, StackedPcsData},
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        whir::WhirProver,
        ColMajorMatrix, CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        DeviceStarkProvingKey, MultiRapProver, OpeningProver, ProverBackend, ProverDevice,
        ProvingContext, TraceCommitter,
    },
    FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};

#[derive(Clone, Copy)]
pub struct CpuBackend<SC: StarkProtocolConfig>(PhantomData<SC>);

impl<SC: StarkProtocolConfig> CpuBackend<SC> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<SC: StarkProtocolConfig> Default for CpuBackend<SC> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Getters, derive_new::new)]
pub struct CpuDevice<SC> {
    #[getset(get = "pub")]
    config: SC,
}

impl<SC: StarkProtocolConfig> CpuDevice<SC> {
    pub fn params(&self) -> &SystemParams {
        self.config.params()
    }
}

impl<SC: StarkProtocolConfig> ProverBackend for CpuBackend<SC> {
    const CHALLENGE_EXT_DEGREE: u8 = SC::D_EF as u8;

    type Val = SC::F;
    type Challenge = SC::EF;
    type Commitment = SC::Digest;
    type Matrix = ColMajorMatrix<SC::F>;
    type OtherAirData = ();
    type PcsData = StackedPcsData<SC::F, SC::Digest>;
}

impl<SC, TS> ProverDevice<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
}

impl<SC: StarkProtocolConfig> TraceCommitter<CpuBackend<SC>> for CpuDevice<SC>
where
    SC::F: Ord,
{
    fn commit(
        &self,
        traces: &[&ColMajorMatrix<SC::F>],
    ) -> (SC::Digest, StackedPcsData<SC::F, SC::Digest>) {
        stacked_commit(
            self.config().hasher(),
            self.params().l_skip,
            self.params().n_stack,
            self.params().log_blowup,
            self.params().k_whir(),
            traces,
        )
    }
}

impl<SC, TS> MultiRapProver<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
    TS: FiatShamirTranscript<SC>,
{
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    /// The random opening point `r` where the batch constraint sumcheck reduces to evaluation
    /// claims of trace matrices `T, T_{rot}` at `r_{n_T}`.
    type Artifacts = Vec<SC::EF>;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: &ProvingContext<CpuBackend<SC>>,
        _common_main_pcs_data: &StackedPcsData<SC::F, SC::Digest>,
    ) -> ((GkrProof<SC>, BatchConstraintProof<SC>), Vec<SC::EF>) {
        let (gkr_proof, batch_constraint_proof, r) =
            prove_zerocheck_and_logup::<SC, _>(transcript, mpk, ctx);
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl<SC, TS> OpeningProver<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    type OpeningProof = (StackingProof<SC>, WhirProof<SC>);
    /// The shared vector `r` where each trace matrix `T, T_{rot}` is opened at `r_{n_T}`.
    type OpeningPoints = Vec<SC::EF>;

    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: ProvingContext<CpuBackend<SC>>,
        common_main_pcs_data: StackedPcsData<SC::F, SC::Digest>,
        r: Vec<SC::EF>,
    ) -> (StackingProof<SC>, WhirProof<SC>) {
        let params = self.params();

        let need_rot_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
            .collect_vec();

        // Currently alternates between preprocessed and cached pcs data
        let pre_cached_pcs_data_per_commit: Vec<_> = ctx
            .per_trace
            .iter()
            .flat_map(|(air_idx, trace_ctx)| {
                mpk.per_air[*air_idx]
                    .preprocessed_data
                    .iter()
                    .chain(&trace_ctx.cached_mains)
                    .map(|cd| cd.data.clone())
            })
            .collect();

        let mut stacked_per_commit = vec![&common_main_pcs_data];
        for data in &pre_cached_pcs_data_per_commit {
            stacked_per_commit.push(data);
        }
        let mut need_rot_per_commit = vec![need_rot_per_trace];
        for (air_idx, trace_ctx) in &ctx.per_trace {
            let need_rot = mpk.per_air[*air_idx].vk.params.need_rot;
            if mpk.per_air[*air_idx].preprocessed_data.is_some() {
                need_rot_per_commit.push(vec![need_rot]);
            }
            for _ in &trace_ctx.cached_mains {
                need_rot_per_commit.push(vec![need_rot]);
            }
        }
        let (stacking_proof, u_prisma) =
            prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
                self,
                transcript,
                params.n_stack,
                stacked_per_commit,
                need_rot_per_commit,
                &r,
            );

        let (&u0, u_rest) = u_prisma.split_first().unwrap();
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        let whir_proof = self.prove_whir(
            transcript,
            common_main_pcs_data,
            pre_cached_pcs_data_per_commit,
            &u_cube,
        );
        (stacking_proof, whir_proof)
    }
}

impl<SC: StarkProtocolConfig> DeviceDataTransporter<SC, CpuBackend<SC>> for CpuDevice<SC> {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<CpuBackend<SC>> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    let trace = d.mat_view(0).to_matrix();
                    CommittedTraceData {
                        commitment: d.commit(),
                        trace,
                        data: d.clone(),
                    }
                });
                DeviceStarkProvingKey {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                    other_data: (),
                }
            })
            .collect();
        DeviceMultiStarkProvingKey::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params.clone(),
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<SC::F>) -> ColMajorMatrix<SC::F> {
        matrix.clone()
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<SC::F, SC::Digest>,
    ) -> StackedPcsData<SC::F, SC::Digest> {
        pcs_data.clone()
    }

    fn transport_matrix_from_device_to_host(
        &self,
        matrix: &ColMajorMatrix<SC::F>,
    ) -> ColMajorMatrix<SC::F> {
        matrix.clone()
    }
}
