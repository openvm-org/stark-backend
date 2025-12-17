use std::sync::Arc;

use openvm_stark_backend::{
    prover::{
        cpu::CpuBackend,
        types::{AirProvingContext, ProvingContext},
        ProverBackend,
    },
    Chip,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    prover::{
        stacked_pcs::stacked_commit, AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2,
        CpuBackendV2, ProverBackendV2, ProvingContextV2,
    },
    ChipV2, SystemParams, F,
};

type SC = BabyBearPoseidon2Config;

pub trait V1Compat: ProverBackendV2 + Sized {
    type V1: ProverBackend<Val = <Self as ProverBackendV2>::Val>;

    fn dummy_matrix() -> Self::Matrix;
    fn convert_trace(matrix: <Self::V1 as ProverBackend>::Matrix) -> Self::Matrix;

    fn convert_committed_trace(
        params: &SystemParams,
        matrix: <Self::V1 as ProverBackend>::Matrix,
    ) -> CommittedTraceDataV2<Self>;
}

impl<PB: V1Compat> ProvingContextV2<PB> {
    pub fn from_v1(params: &SystemParams, ctx: ProvingContext<PB::V1>) -> Self {
        let per_trace = ctx
            .per_air
            .into_iter()
            .map(|(air_idx, air_ctx)| (air_idx, AirProvingContextV2::from_v1(params, air_ctx)))
            .filter(|(_, air_ctx)| air_ctx.height() > 0)
            .collect();
        Self::new(per_trace)
    }

    pub fn from_v1_no_cached(ctx: ProvingContext<PB::V1>) -> Self {
        let per_trace = ctx
            .per_air
            .into_iter()
            .map(|(air_idx, air_ctx)| (air_idx, AirProvingContextV2::from_v1_no_cached(air_ctx)))
            .collect();
        Self::new(per_trace)
    }
}

impl<PB: V1Compat> AirProvingContextV2<PB> {
    pub fn from_v1(params: &SystemParams, ctx: AirProvingContext<PB::V1>) -> Self {
        let common_main = ctx
            .common_main
            .map(<PB as V1Compat>::convert_trace)
            .unwrap_or_else(|| <PB as V1Compat>::dummy_matrix());
        let cached_mains = ctx
            .cached_mains
            .into_iter()
            .map(|d| <PB as V1Compat>::convert_committed_trace(params, d.trace))
            .collect();
        Self {
            cached_mains,
            common_main,
            public_values: ctx.public_values,
        }
    }

    pub fn from_v1_no_cached(ctx: AirProvingContext<PB::V1>) -> Self {
        assert!(ctx.cached_mains.is_empty());
        let common_main = ctx
            .common_main
            .map(<PB as V1Compat>::convert_trace)
            .unwrap_or_else(|| <PB as V1Compat>::dummy_matrix());
        Self {
            cached_mains: vec![],
            common_main,
            public_values: ctx.public_values,
        }
    }
}

impl<C, R, PB: V1Compat> ChipV2<R, PB> for C
where
    C: Chip<R, PB::V1>,
{
    fn generate_proving_ctx(&self, records: R) -> AirProvingContextV2<PB> {
        let v1_ctx = self.generate_proving_ctx(records);
        AirProvingContextV2::from_v1_no_cached(v1_ctx)
    }
}

impl V1Compat for CpuBackendV2 {
    type V1 = CpuBackend<SC>;

    fn dummy_matrix() -> Self::Matrix {
        ColMajorMatrix::dummy()
    }

    fn convert_trace(matrix: Arc<RowMajorMatrix<F>>) -> Self::Matrix {
        ColMajorMatrix::from_row_major(&matrix)
    }

    fn convert_committed_trace(
        params: &SystemParams,
        matrix: Arc<RowMajorMatrix<F>>,
    ) -> CommittedTraceDataV2<CpuBackendV2> {
        let trace = ColMajorMatrix::from_row_major(&matrix);
        let (commitment, data) = stacked_commit(
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir(),
            &[&trace],
        );

        CommittedTraceDataV2 {
            commitment,
            trace,
            data: Arc::new(data),
        }
    }
}
