use std::sync::Arc;

use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, StarkProvingKey},
    prover::{
        cpu::CpuBackend,
        types::{AirProvingContext, ProvingContext},
        MatrixDimensions, ProverBackend,
    },
    Chip,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use crate::{
    keygen::types::{
        find_unused_vars, MultiStarkProvingKeyV2, StarkProvingKeyV2, StarkVerifyingKeyV2,
        StarkVerifyingParamsV2, VerifierSinglePreprocessedData,
    },
    prover::{
        stacked_pcs::stacked_commit, AirProvingContextV2, ColMajorMatrix, CommittedTraceDataV2,
        CpuBackendV2, ProverBackendV2, ProvingContextV2,
    },
    ChipV2, SystemParams, F,
};

type SC = BabyBearPoseidon2Config;

impl MultiStarkProvingKeyV2 {
    pub fn from_v1(params: SystemParams, pk: MultiStarkProvingKey<SC>) -> Self {
        let per_air = pk
            .per_air
            .into_iter()
            .map(|pk| StarkProvingKeyV2::from_v1(params, pk))
            .collect();
        let trace_height_constraints = pk.trace_height_constraints;
        let max_constraint_degree = pk.max_constraint_degree;

        Self {
            per_air,
            trace_height_constraints,
            max_constraint_degree,
            params,
            vk_pre_hash: pk.vk_pre_hash.into(),
        }
    }
}

impl StarkProvingKeyV2 {
    pub fn from_v1(params: SystemParams, pk: StarkProvingKey<SC>) -> Self {
        // If preprocessed trace exists, re-commit using stacked PCS
        let mut preprocessed_vdata = None;
        let preprocessed_data = pk.preprocessed_data.map(|d| {
            let trace = ColMajorMatrix::from_row_major(&d.trace);
            let (commit, data) = stacked_commit(
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.k_whir,
                &[&trace],
            );
            preprocessed_vdata = Some(VerifierSinglePreprocessedData {
                commit,
                hypercube_dim: log2_strict_usize(trace.height()) as isize - params.l_skip as isize,
                stacking_width: data.matrix.width(),
            });
            Arc::new(data)
        });
        let vparams = StarkVerifyingParamsV2 {
            width: pk.vk.params.width,
            num_public_values: pk.vk.params.num_public_values,
        };
        let symbolic_constraints = pk.vk.symbolic_constraints;
        let unused_variables = find_unused_vars(&symbolic_constraints, &vparams.width);
        let vk = StarkVerifyingKeyV2 {
            preprocessed_data: preprocessed_vdata,
            params: vparams,
            symbolic_constraints,
            max_constraint_degree: pk.vk.max_constraint_degree,
            is_required: false, // no AIRs are required in v1
            unused_variables,
        };
        Self {
            air_name: pk.air_name,
            vk,
            preprocessed_data,
        }
    }
}

pub trait V1Compat: ProverBackendV2 + Sized {
    type V1: ProverBackend<Val = <Self as ProverBackendV2>::Val>;

    fn dummy_matrix() -> Self::Matrix;
    fn convert_trace(matrix: <Self::V1 as ProverBackend>::Matrix) -> Self::Matrix;

    fn convert_committed_trace(
        params: SystemParams,
        matrix: <Self::V1 as ProverBackend>::Matrix,
    ) -> CommittedTraceDataV2<Self>;
}

impl<PB: V1Compat> ProvingContextV2<PB> {
    pub fn from_v1(params: SystemParams, ctx: ProvingContext<PB::V1>) -> Self {
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
    pub fn from_v1(params: SystemParams, ctx: AirProvingContext<PB::V1>) -> Self {
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
        params: SystemParams,
        matrix: Arc<RowMajorMatrix<F>>,
    ) -> CommittedTraceDataV2<CpuBackendV2> {
        let trace = ColMajorMatrix::from_row_major(&matrix);
        let (commitment, data) = stacked_commit(
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.k_whir,
            &[&trace],
        );

        CommittedTraceDataV2 {
            commitment,
            trace,
            data: Arc::new(data),
        }
    }
}
