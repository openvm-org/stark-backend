use std::{fmt::Debug, sync::Arc};

use itertools::Itertools;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    error::MemCopyError,
};
use openvm_stark_backend::prover::hal::MatrixDimensions;
use stark_backend_v2::{
    SystemParams,
    keygen::types::MultiStarkProvingKeyV2,
    poly_common::Squarable,
    poseidon2::sponge::FiatShamirTranscript,
    proof::*,
    prover::{
        ColMajorMatrix, CommittedTraceDataV2, CpuBackendV2, DeviceDataTransporterV2,
        DeviceMultiStarkProvingKeyV2, DeviceStarkProvingKeyV2, MatrixView, MultiRapProver,
        OpeningProverV2, ProverBackendV2, ProverDeviceV2, ProvingContextV2, TraceCommitterV2,
        prove_zerocheck_and_logup,
        stacked_pcs::{MerkleTree, StackedPcsData},
        stacked_reduction::prove_stacked_opening_reduction,
        whir::WhirProver,
    },
};

use crate::{
    D_EF, Digest, EF, F, GpuDeviceV2,
    logup_zerocheck::LogupZerocheckGpu,
    merkle_tree::MerkleTreeGpu,
    stacked_pcs::{StackedPcsDataGpu, stacked_commit},
    stacked_reduction::StackedReductionGpu,
};

#[derive(Clone, Copy)]
pub struct GpuBackendV2;

impl ProverBackendV2 for GpuBackendV2 {
    const CHALLENGE_EXT_DEGREE: u8 = D_EF as u8;

    type Val = F;
    type Challenge = EF;
    type Commitment = Digest;
    type Matrix = DeviceMatrix<F>;
    type PcsData = StackedPcsDataGpu<F, Digest>;
}

impl<TS: FiatShamirTranscript> ProverDeviceV2<GpuBackendV2, TS> for GpuDeviceV2 {
    fn config(&self) -> SystemParams {
        self.config
    }
}

impl TraceCommitterV2<GpuBackendV2> for GpuDeviceV2 {
    fn commit(&self, traces: &[&DeviceMatrix<F>]) -> (Digest, StackedPcsDataGpu<F, Digest>) {
        stacked_commit(
            self.config.l_skip,
            self.config.n_stack,
            self.config.log_blowup,
            self.config.k_whir,
            traces,
        )
        .unwrap()
    }
}

impl<TS: FiatShamirTranscript> MultiRapProver<GpuBackendV2, TS> for GpuDeviceV2 {
    type PartialProof = (GkrProof, BatchConstraintProof);
    /// The random opening point `r` where the batch constraint sumcheck reduces to evaluation
    /// claims of trace matrices `T, T_{rot}` at `r_{n_T}`.
    type Artifacts = Vec<EF>;

    fn prove_rap_constraints(
        &self,
        transcript: &mut TS,
        mpk: &DeviceMultiStarkProvingKeyV2<GpuBackendV2>,
        ctx: ProvingContextV2<GpuBackendV2>,
    ) -> ((GkrProof, BatchConstraintProof), Vec<EF>) {
        let (gkr_proof, batch_constraint_proof, r) =
            prove_zerocheck_and_logup::<_, _, TS, LogupZerocheckGpu>(self, transcript, mpk, ctx);
        ((gkr_proof, batch_constraint_proof), r)
    }
}

impl<TS: FiatShamirTranscript> OpeningProverV2<GpuBackendV2, TS> for GpuDeviceV2 {
    type OpeningProof = (StackingProof, WhirProof);
    /// The shared vector `r` where each trace matrix `T, T_{rot}` is opened at `r_{n_T}`.
    type OpeningPoints = Vec<EF>;

    fn prove_openings(
        &self,
        transcript: &mut TS,
        common_main_pcs_data: StackedPcsDataGpu<F, Digest>,
        pre_cached_pcs_data_per_commit: Vec<Arc<StackedPcsDataGpu<F, Digest>>>,
        r: Vec<EF>,
    ) -> (StackingProof, WhirProof) {
        let params = self.config;
        let mut stacked_per_commit = vec![&common_main_pcs_data];
        for data in &pre_cached_pcs_data_per_commit {
            stacked_per_commit.push(data);
        }
        let (stacking_proof, u_prisma) =
            prove_stacked_opening_reduction::<_, _, _, StackedReductionGpu>(
                self,
                transcript,
                self.config.n_stack,
                stacked_per_commit,
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

impl DeviceDataTransporterV2<GpuBackendV2> for GpuDeviceV2 {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKeyV2,
    ) -> DeviceMultiStarkProvingKeyV2<GpuBackendV2> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    let d_data = self.transport_pcs_data_to_device(d.as_ref());
                    // Sanity check. Not a strong assert because we transport the merkle tree
                    // instead of recomputing it above.
                    assert_eq!(d_data.tree.root(), d.commit());
                    CommittedTraceDataV2 {
                        commitment: d.commit(),
                        data: Arc::new(d_data),
                        height: d.mat_view(0).height(),
                    }
                });

                DeviceStarkProvingKeyV2 {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                }
            })
            .collect();

        DeviceMultiStarkProvingKeyV2::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.max_constraint_degree,
            mpk.params,
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<F>) -> DeviceMatrix<F> {
        transport_matrix_h2d_col_major(matrix).unwrap()
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<F, Digest>,
    ) -> StackedPcsDataGpu<F, Digest> {
        let StackedPcsData {
            layout,
            matrix,
            tree,
        } = pcs_data;
        let d_matrix = self.transport_matrix_to_device(matrix);
        let d_tree = transport_merkle_tree_to_device(tree).unwrap();

        StackedPcsDataGpu {
            layout: layout.clone(),
            matrix: d_matrix,
            tree: d_tree,
        }
    }

    fn transport_matrix_from_device_to_host(&self, matrix: &DeviceMatrix<F>) -> ColMajorMatrix<F> {
        transport_matrix_d2h_col_major(matrix).unwrap()
    }

    fn transport_pcs_data_from_device_to_host(
        &self,
        pcs_data: &StackedPcsDataGpu<F, Digest>,
    ) -> StackedPcsData<F, Digest> {
        transport_stacked_pcs_data_to_host(pcs_data)
    }
}

pub fn transport_matrix_h2d_col_major<T>(
    matrix: &ColMajorMatrix<T>,
) -> Result<DeviceMatrix<T>, MemCopyError> {
    // matrix is already col-major, so this is just H2D buffer transfer
    let buffer = matrix.values.to_device()?;
    Ok(DeviceMatrix::new(
        Arc::new(buffer),
        matrix.height(),
        matrix.width(),
    ))
}

pub fn transport_matrix_d2h_col_major<T>(
    matrix: &DeviceMatrix<T>,
) -> Result<ColMajorMatrix<T>, MemCopyError> {
    let values_host = matrix.buffer().to_host()?;
    Ok(ColMajorMatrix::new(values_host, matrix.width()))
}

/// Transports backing matrix and tree digest layers from host to device.
pub fn transport_merkle_tree_to_device<F, Digest>(
    tree: &MerkleTree<F, Digest>,
) -> Result<MerkleTreeGpu<F, Digest>, MemCopyError> {
    let backing_matrix = transport_matrix_h2d_col_major(tree.backing_matrix())?;
    let digest_layers = tree
        .digest_layers()
        .iter()
        .map(|layer| layer.to_device())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MerkleTreeGpu {
        backing_matrix,
        digest_layers,
        rows_per_query: tree.rows_per_query(),
    })
}

pub fn transport_merkle_tree_to_host(tree: &MerkleTreeGpu<F, Digest>) -> MerkleTree<F, Digest> {
    let backing_matrix = transport_matrix_d2h_col_major(&tree.backing_matrix).unwrap();
    MerkleTree::<F, Digest>::new(backing_matrix, tree.rows_per_query)
}

pub fn transport_stacked_pcs_data_to_host(
    pcs_data: &StackedPcsDataGpu<F, Digest>,
) -> StackedPcsData<F, Digest> {
    let layout = pcs_data.layout.clone();
    let matrix = transport_matrix_d2h_col_major(&pcs_data.matrix).unwrap();
    let tree = transport_merkle_tree_to_host(&pcs_data.tree);

    StackedPcsData::new(layout, matrix, tree)
}

pub fn transport_committed_trace_data_to_host(
    data: &CommittedTraceDataV2<GpuBackendV2>,
) -> CommittedTraceDataV2<CpuBackendV2> {
    let commitment = data.commitment;
    let height = data.height;
    let pcs_data = transport_stacked_pcs_data_to_host(&data.data);
    CommittedTraceDataV2 {
        commitment,
        data: Arc::new(pcs_data),
        height,
    }
}

pub fn assert_eq_host_and_device_matrix_col_maj<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: &ColMajorMatrix<T>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = gpu.to_host().unwrap();
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu[c * cpu.height() + r],
                *cpu.get(r, c).unwrap(),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

mod v1_shims {
    use openvm_cuda_backend::{base::DeviceMatrix, prover_backend::GpuBackend};
    use stark_backend_v2::{SystemParams, v1_shims::V1Compat};

    use crate::{F, GpuBackendV2, stacked_pcs::stacked_commit};

    impl V1Compat for GpuBackendV2 {
        type V1 = GpuBackend;

        fn convert_trace(matrix: DeviceMatrix<F>) -> DeviceMatrix<F> {
            matrix
        }

        fn convert_pcs_data(
            params: SystemParams,
            matrix: DeviceMatrix<F>,
        ) -> (Self::Commitment, Self::PcsData) {
            stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir,
                &[&matrix],
            )
            .unwrap()
        }
    }
}
