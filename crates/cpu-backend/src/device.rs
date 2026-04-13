//! [CpuDevice] implementation: TraceCommitter, DeviceDataTransporter, MultiRapProver,
//! OpeningProver.

use getset::Getters;
use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    poly_common::Squarable,
    proof::{BatchConstraintProof, GkrProof, StackingProof, WhirProof},
    prover::{
        poly::Mle,
        stacked_pcs::{stacked_matrix, StackedPcsData},
        stacked_reduction::prove_stacked_opening_reduction,
        ColMajorMatrix, CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        DeviceStarkProvingKey, MatrixDimensions, MultiRapProver, OpeningProver, ProverDevice,
        ProvingContext, StridedColMajorMatrixView, TraceCommitter,
    },
    FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::{
    backend::CpuBackend,
    error::CpuProverError,
    merkle::{rs_encode_and_merkle_cpu, CpuMerkleTree},
    pcs_data::CpuStackedPcsData,
    stacked_reduction::StackedReductionCpuNew,
    two_adic::DftTwiddles,
};

/// Row-major CPU prover device.
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

impl<SC, TS> ProverDevice<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::F: Ord,
    SC::EF: TwoAdicField + ExtensionField<SC::F> + Ord,
    TS: FiatShamirTranscript<SC>,
{
    type Error = CpuProverError;
    type DeviceCtx = ();

    fn device_ctx(&self) -> &() {
        &()
    }
}

impl<SC: StarkProtocolConfig> TraceCommitter<CpuBackend<SC>> for CpuDevice<SC>
where
    SC::F: Ord,
{
    type Error = CpuProverError;

    #[instrument(level = "info", name = "trace_commit_cpu", skip_all)]
    fn commit(
        &self,
        traces: &[&RowMajorMatrix<SC::F>],
    ) -> Result<(SC::Digest, CpuStackedPcsData<SC::F, SC::Digest>), Self::Error> {
        // Convert row-major to col-major for stacking commitment
        let col_major_traces: Vec<ColMajorMatrix<SC::F>> = traces
            .iter()
            .map(|rm| ColMajorMatrix::from_row_major(rm))
            .collect();
        let col_major_refs: Vec<&ColMajorMatrix<SC::F>> = col_major_traces.iter().collect();

        let params = self.params();
        let (q_trace, layout) = stacked_matrix(params.l_skip, params.n_stack, &col_major_refs)?;
        let tree = rs_encode_and_merkle_cpu(
            self.config().hasher(),
            params.l_skip,
            params.log_blowup,
            &q_trace,
            1 << params.k_whir(),
        );
        let root = tree.root()?;
        let data = CpuStackedPcsData::new(layout, q_trace, tree);
        Ok((root, data))
    }
}

impl<SC, TS> MultiRapProver<CpuBackend<SC>, TS> for CpuDevice<SC>
where
    SC: StarkProtocolConfig,
    SC::EF: TwoAdicField + ExtensionField<SC::F>,
    TS: FiatShamirTranscript<SC>,
{
    type PartialProof = (GkrProof<SC>, BatchConstraintProof<SC>);
    type Artifacts = Vec<SC::EF>;

    type Error = CpuProverError;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: &ProvingContext<CpuBackend<SC>>,
        _common_main_pcs_data: &CpuStackedPcsData<SC::F, SC::Digest>,
    ) -> Result<((GkrProof<SC>, BatchConstraintProof<SC>), Vec<SC::EF>), Self::Error> {
        let (gkr_proof, batch_constraint_proof, r) =
            crate::logup_zerocheck::prove_zerocheck_and_logup::<SC, _>(transcript, mpk, ctx)?;
        Ok(((gkr_proof, batch_constraint_proof), r))
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
    type OpeningPoints = Vec<SC::EF>;

    type Error = CpuProverError;

    fn prove_openings(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKey<CpuBackend<SC>>,
        ctx: ProvingContext<CpuBackend<SC>>,
        common_main_pcs_data: CpuStackedPcsData<SC::F, SC::Digest>,
        r: Vec<SC::EF>,
    ) -> Result<(StackingProof<SC>, WhirProof<SC>), Self::Error> {
        let params = self.params();

        let need_rot_per_trace = ctx
            .per_trace
            .iter()
            .map(|(air_idx, _)| mpk.per_air[*air_idx].vk.params.need_rot)
            .collect_vec();

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
            prove_stacked_opening_reduction::<SC, _, _, _, StackedReductionCpuNew<SC>>(
                self,
                transcript,
                params.n_stack,
                stacked_per_commit,
                need_rot_per_commit,
                &r,
            );

        let (&u0, u_rest) = u_prisma
            .split_first()
            .ok_or(openvm_stark_backend::prover::error::WhirProverError::UPrismaEmpty)?;
        let u_cube = u0
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(u_rest.iter().copied())
            .collect_vec();

        // Convert to col-major for WHIR
        let committed_mats = std::iter::once(&common_main_pcs_data)
            .chain(pre_cached_pcs_data_per_commit.iter().map(|d| d.as_ref()))
            .map(|d| (&d.matrix, &d.tree))
            .collect_vec();

        let whir_proof = crate::whir::prove_whir_opening_cpu::<SC, _>(
            transcript,
            self.config().hasher(),
            params.l_skip,
            params.log_blowup,
            &params.whir,
            &committed_mats,
            &u_cube,
        )?;
        Ok((stacking_proof, whir_proof))
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
                    let view: StridedColMajorMatrixView<'_, SC::F> = d.mat_view(0).into();
                    let trace = view.to_row_major_matrix();
                    CommittedTraceData {
                        commitment: d.commit().unwrap(),
                        trace,
                        data: std::sync::Arc::new(stacked_pcs_data_to_cpu::<SC>(d)),
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

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<SC::F>) -> RowMajorMatrix<SC::F> {
        let view: StridedColMajorMatrixView<'_, SC::F> = matrix.as_view().into();
        view.to_row_major_matrix()
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<SC::F, SC::Digest>,
    ) -> CpuStackedPcsData<SC::F, SC::Digest> {
        stacked_pcs_data_to_cpu::<SC>(pcs_data)
    }

    fn transport_matrix_from_device_to_host(
        &self,
        matrix: &RowMajorMatrix<SC::F>,
    ) -> ColMajorMatrix<SC::F> {
        ColMajorMatrix::from_row_major(matrix)
    }
}

/// Convert stark-backend's `StackedPcsData` (ColMajor backing) to cpu-backend's
/// `CpuStackedPcsData` (RowMajor backing). The eval matrix is cloned as-is (ColMajor),
/// while the Merkle tree's backing matrix is transposed to RowMajor.
fn stacked_pcs_data_to_cpu<SC: StarkProtocolConfig>(
    pcs_data: &StackedPcsData<SC::F, SC::Digest>,
) -> CpuStackedPcsData<SC::F, SC::Digest> {
    let cm_backing = pcs_data.tree.backing_matrix();
    let height = cm_backing.height();
    let width = cm_backing.width();
    // Transpose ColMajor backing → RowMajor backing
    let mut rm_values = SC::F::zero_vec(height * width);
    rm_values
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..width {
                row[j] = cm_backing.values[j * height + i];
            }
        });
    let rm_backing = RowMajorMatrix::new(rm_values, width);
    let cpu_tree = unsafe {
        CpuMerkleTree::from_raw_parts(
            rm_backing,
            pcs_data.tree.digest_layers().clone(),
            pcs_data.tree.rows_per_query(),
        )
    };
    CpuStackedPcsData::new(pcs_data.layout.clone(), pcs_data.matrix.clone(), cpu_tree)
}

/// In-place PLE evaluation-to-coefficient conversion.
/// Uses inline DIF iDFT with reusable twiddle factors to eliminate the per-chunk allocation
/// overhead of the shared `eval_to_coeff_rs_message`.
pub(crate) fn eval_to_coeff_cpu<F: TwoAdicField>(evals: &[F], twiddles: &DftTwiddles<F>) -> Vec<F> {
    let chunk_len = twiddles.size();
    let mut buf = evals.to_vec();

    // Phase 1: In-place DIF iDFT on each chunk.
    for chunk in buf.chunks_exact_mut(chunk_len) {
        twiddles.idft_inplace(chunk);
    }

    // Phase 2: Convert MLE coefficients to evaluations in-place.
    buf.par_chunks_exact_mut(chunk_len)
        .for_each(Mle::coeffs_to_evals_inplace);

    buf
}
