//! CPU [ProverBackend] trait implementation.

use std::marker::PhantomData;

use getset::Getters;
use itertools::Itertools;

use crate::{
    keygen::types::MultiStarkProvingKeyV2,
    poly_common::Squarable,
    poseidon2::sponge::Poseidon2Hasher,
    proof::{BatchConstraintProof, GkrProof, StackingProof, WhirProof},
    prover::{
        prove_zerocheck_and_logup,
        stacked_pcs::{stacked_commit, StackedPcsData},
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        whir::WhirProver,
        ColMajorMatrix, CommittedTraceDataV2, DeviceDataTransporterV2,
        DeviceMultiStarkProvingKeyV2, DeviceStarkProvingKeyV2, MultiRapProver, OpeningProverV2,
        ProverBackendV2, ProverDeviceV2, ProvingContextV2, TraceCommitterV2,
    },
    baby_bear_poseidon2::{BabyBearPoseidon2ConfigV2, Digest, EF, F},
    FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};

#[derive(Clone, Copy)]
pub struct CpuBackendV2<SC: StarkProtocolConfig>(PhantomData<SC>);

impl<SC: StarkProtocolConfig> CpuBackendV2<SC> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<SC: StarkProtocolConfig> Default for CpuBackendV2<SC> {
    fn default() -> Self {
        Self::new()
    }
}

type SCV2 = BabyBearPoseidon2ConfigV2;

#[derive(Clone, Getters, derive_new::new)]
pub struct CpuDeviceV2 {
    #[getset(get = "pub")]
    config: SystemParams,
}

impl<SC: StarkProtocolConfig> ProverBackendV2 for CpuBackendV2<SC> {
    const CHALLENGE_EXT_DEGREE: u8 = SC::D_EF as u8;

    type Val = SC::F;
    type Challenge = SC::EF;
    type Commitment = SC::Digest;
    type Matrix = ColMajorMatrix<SC::F>;
    type OtherAirData = ();
    type PcsData = StackedPcsData<SC::F, SC::Digest>;
}

impl<TS: FiatShamirTranscript<SCV2>> ProverDeviceV2<CpuBackendV2<SCV2>, TS> for CpuDeviceV2 {
    fn config(&self) -> &SystemParams {
        &self.config
    }
}

impl TraceCommitterV2<CpuBackendV2<SCV2>> for CpuDeviceV2 {
    fn commit(&self, traces: &[&ColMajorMatrix<F>]) -> (Digest, StackedPcsData<F, Digest>) {
        stacked_commit::<Poseidon2Hasher>(
            self.config.l_skip,
            self.config.n_stack,
            self.config.log_blowup,
            self.config.k_whir(),
            traces,
        )
    }
}

impl<TS: FiatShamirTranscript<SCV2>> MultiRapProver<CpuBackendV2<SCV2>, TS> for CpuDeviceV2 {
    type PartialProof = (GkrProof<SCV2>, BatchConstraintProof<SCV2>);
    /// The random opening point `r` where the batch constraint sumcheck reduces to evaluation
    /// claims of trace matrices `T, T_{rot}` at `r_{n_T}`.
    type Artifacts = Vec<EF>;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<CpuBackendV2<SCV2>>,
        ctx: &ProvingContextV2<CpuBackendV2<SCV2>>,
        _common_main_pcs_data: &StackedPcsData<F, Digest>,
    ) -> ((GkrProof<SCV2>, BatchConstraintProof<SCV2>), Vec<EF>) {
        let (gkr_proof, batch_constraint_proof, r) =
            prove_zerocheck_and_logup::<SCV2, _>(transcript, mpk, ctx);
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl<TS: FiatShamirTranscript<SCV2>> OpeningProverV2<CpuBackendV2<SCV2>, TS> for CpuDeviceV2 {
    type OpeningProof = (StackingProof<SCV2>, WhirProof<SCV2>);
    /// The shared vector `r` where each trace matrix `T, T_{rot}` is opened at `r_{n_T}`.
    type OpeningPoints = Vec<EF>;

    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<CpuBackendV2<SCV2>>,
        ctx: ProvingContextV2<CpuBackendV2<SCV2>>,
        common_main_pcs_data: StackedPcsData<F, Digest>,
        r: Vec<EF>,
    ) -> (StackingProof<SCV2>, WhirProof<SCV2>) {
        let params = &self.config;

        let need_rot_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
            .collect_vec();

        // Currently alternates between preprocessed and cached pcs data
        let pre_cached_pcs_data_per_commit: Vec<_> = ctx
            .per_trace
            .iter()
            .flat_map(|(air_idx, air_ctx)| {
                mpk.per_air[*air_idx]
                    .preprocessed_data
                    .iter()
                    .chain(&air_ctx.cached_mains)
                    .map(|cd| cd.data.clone())
            })
            .collect();

        let mut stacked_per_commit = vec![&common_main_pcs_data];
        for data in &pre_cached_pcs_data_per_commit {
            stacked_per_commit.push(data);
        }
        let mut need_rot_per_commit = vec![need_rot_per_trace];
        for (air_idx, air_ctx) in &ctx.per_trace {
            let need_rot = mpk.per_air[*air_idx].vk.params.need_rot;
            if mpk.per_air[*air_idx].preprocessed_data.is_some() {
                need_rot_per_commit.push(vec![need_rot]);
            }
            for _ in &air_ctx.cached_mains {
                need_rot_per_commit.push(vec![need_rot]);
            }
        }
        let (stacking_proof, u_prisma) =
            prove_stacked_opening_reduction::<SCV2, _, _, _, StackedReductionCpu<SCV2>>(
                self,
                transcript,
                self.config.n_stack,
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

impl DeviceDataTransporterV2<CpuBackendV2<SCV2>> for CpuDeviceV2 {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKeyV2<SCV2>,
    ) -> DeviceMultiStarkProvingKeyV2<CpuBackendV2<SCV2>> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    let trace = d.mat_view(0).to_matrix();
                    CommittedTraceDataV2 {
                        commitment: d.commit(),
                        trace,
                        data: d.clone(),
                    }
                });
                DeviceStarkProvingKeyV2 {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                    other_data: (),
                }
            })
            .collect();
        DeviceMultiStarkProvingKeyV2::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params.clone(),
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<F>) -> ColMajorMatrix<F> {
        matrix.clone()
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<F, Digest>,
    ) -> StackedPcsData<F, Digest> {
        pcs_data.clone()
    }

    fn transport_matrix_from_device_to_host(
        &self,
        matrix: &ColMajorMatrix<F>,
    ) -> ColMajorMatrix<F> {
        matrix.clone()
    }
}
