use itertools::Itertools;
use openvm_stark_backend::{
    poly_common::Squarable,
    proof::*,
    prover::{
        stacked_pcs::{stacked_commit, StackedPcsData},
        stacked_reduction::{prove_stacked_opening_reduction, StackedReductionCpu},
        whir::prove_whir_opening,
        ColMajorMatrix, DeviceMultiStarkProvingKey, MatrixDimensions, MultiRapProver,
        OpeningProver, ProverBackend, ProverDevice, ProvingContext, TraceCommitter,
    },
    StarkProtocolConfig,
};
use openvm_metal_common::copy::MemCopyD2H;
use tracing::instrument;

use crate::{
    base::MetalMatrix,
    prelude::{Digest, D_EF, EF, F, SC},
    sponge::DuplexSpongeMetal,
    AirDataMetal, MetalDevice,
};

#[derive(Clone, Copy)]
pub struct MetalBackend;

/// PCS data type for the Metal backend, wrapping the CPU `StackedPcsData`.
pub struct StackedPcsDataMetal {
    pub inner: StackedPcsData<F, Digest>,
}

// SAFETY: StackedPcsData fields are Send+Sync
unsafe impl Send for StackedPcsDataMetal {}
unsafe impl Sync for StackedPcsDataMetal {}

impl Clone for StackedPcsDataMetal {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl ProverBackend for MetalBackend {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;

    type Val = F;
    type Challenge = EF;
    type Commitment = Digest;
    type Matrix = MetalMatrix<F>;
    type PcsData = StackedPcsDataMetal;
    type OtherAirData = AirDataMetal;
}

impl ProverDevice<MetalBackend, DuplexSpongeMetal> for MetalDevice {}

/// Convert a MetalMatrix to a ColMajorMatrix by reading from unified memory.
fn metal_to_col_major(m: &MetalMatrix<F>) -> ColMajorMatrix<F> {
    ColMajorMatrix::new(m.to_host(), m.width())
}

impl TraceCommitter<MetalBackend> for MetalDevice {
    #[instrument(name = "metal.trace_commit", skip_all, fields(phase = "prover"))]
    fn commit(&self, traces: &[&MetalMatrix<F>]) -> (Digest, StackedPcsDataMetal) {
        let config = self.config();
        let sc = SC::default_from_params(config.clone());
        let hasher = sc.hasher();
        // Convert MetalMatrix refs to ColMajorMatrix
        let col_matrices: Vec<ColMajorMatrix<F>> =
            traces.iter().map(|m| metal_to_col_major(m)).collect();
        let col_refs: Vec<&ColMajorMatrix<F>> = col_matrices.iter().collect();
        let (commit, pcs_data) = stacked_commit(
            hasher,
            config.l_skip,
            config.n_stack,
            config.log_blowup,
            config.k_whir(),
            &col_refs,
        );
        (commit, StackedPcsDataMetal { inner: pcs_data })
    }
}

impl MultiRapProver<MetalBackend, DuplexSpongeMetal> for MetalDevice {
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    type Artifacts = Vec<EF>;

    #[instrument(name = "metal.rap_constraints", skip_all, fields(phase = "prover"))]
    fn prove_rap_constraints(
        &self,
        transcript: &mut DuplexSpongeMetal,
        mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: &ProvingContext<MetalBackend>,
        _common_main_pcs_data: &StackedPcsDataMetal,
    ) -> ((GkrProof<SC>, BatchConstraintProof<SC>), Vec<EF>) {
        // Convert Metal types to CPU types and delegate
        let cpu_mpk = crate::convert::mpk_to_cpu(mpk);
        let cpu_ctx = crate::convert::ctx_to_cpu(ctx);

        use openvm_stark_backend::prover::prove_zerocheck_and_logup;
        let (gkr_proof, batch_constraint_proof, r) =
            prove_zerocheck_and_logup::<SC, _>(transcript, &cpu_mpk, &cpu_ctx);
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl OpeningProver<MetalBackend, DuplexSpongeMetal> for MetalDevice {
    type OpeningProof = (StackingProof<SC>, WhirProof<SC>);
    type OpeningPoints = Vec<EF>;

    #[instrument(name = "metal.openings", skip_all, fields(phase = "prover"))]
    fn prove_openings(
        &self,
        transcript: &mut DuplexSpongeMetal,
        mpk: &DeviceMultiStarkProvingKey<MetalBackend>,
        ctx: ProvingContext<MetalBackend>,
        common_main_pcs_data: StackedPcsDataMetal,
        r: Vec<EF>,
    ) -> (StackingProof<SC>, WhirProof<SC>) {
        let config = self.config();
        let sc = SC::default_from_params(config.clone());
        let hasher = sc.hasher();

        let need_rot_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
            .collect_vec();

        // Collect preprocessed/cached PCS data
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

        let cpu_common_main = &common_main_pcs_data.inner;
        let cpu_pre_cached: Vec<_> = pre_cached_pcs_data_per_commit
            .iter()
            .map(|d| &d.inner)
            .collect();

        let mut stacked_per_commit: Vec<&StackedPcsData<F, Digest>> = vec![cpu_common_main];
        for data in &cpu_pre_cached {
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

        // Use CPU stacked reduction
        let cpu_device = crate::convert::make_cpu_device(config);
        let (stacking_proof, u_prisma) =
            prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpu<SC>>(
                &cpu_device,
                transcript,
                config.n_stack,
                stacked_per_commit,
                need_rot_per_commit,
                &r,
            );

        let (&u0, u_rest) = u_prisma.split_first().unwrap();
        let u_cube = u0
            .exp_powers_of_2()
            .take(config.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        // Collect committed mats for WHIR
        let all_pcs_data: Vec<&StackedPcsData<F, Digest>> = std::iter::once(cpu_common_main)
            .chain(cpu_pre_cached.into_iter())
            .collect();
        let committed_mats: Vec<_> = all_pcs_data
            .iter()
            .map(|d| (&d.matrix, &d.tree))
            .collect();

        let whir_proof = prove_whir_opening::<SC, _>(
            transcript,
            hasher,
            config.l_skip,
            config.log_blowup,
            &config.whir,
            &committed_mats,
            &u_cube,
        );

        (stacking_proof, whir_proof)
    }
}
