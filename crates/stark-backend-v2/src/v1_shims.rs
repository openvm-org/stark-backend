use std::sync::Arc;

use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, StarkProvingKey},
    prover::{
        MatrixDimensions,
        cpu::CpuBackend,
        types::{AirProvingContext, ProvingContext},
    },
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_util::log2_strict_usize;

use crate::{
    keygen::types::{
        MultiStarkProvingKeyV2, StarkProvingKeyV2, StarkVerifyingKeyV2, StarkVerifyingParamsV2,
        SystemParams, VerifierSinglePreprocessedData,
    },
    prover::{
        AirProvingContextV2, ColMajorMatrix, CpuBackendV2, ProvingContextV2,
        stacked_pcs::stacked_commit,
    },
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
        let vk = StarkVerifyingKeyV2 {
            preprocessed_data: preprocessed_vdata,
            params: vparams,
            symbolic_constraints,
            max_constraint_degree: pk.vk.quotient_degree + 1,
            is_required: false, // no AIRs are required in v1
        };
        Self {
            air_name: pk.air_name,
            vk,
            preprocessed_data,
        }
    }
}

impl ProvingContextV2<CpuBackendV2> {
    pub fn from_v1(params: SystemParams, ctx: ProvingContext<CpuBackend<SC>>) -> Self {
        let per_air = ctx
            .per_air
            .into_iter()
            .map(|(air_id, air_ctx)| (air_id, AirProvingContextV2::from_v1(params, air_ctx)))
            .collect();
        Self { per_trace: per_air }
    }
}

impl AirProvingContextV2<CpuBackendV2> {
    pub fn from_v1(params: SystemParams, ctx: AirProvingContext<CpuBackend<SC>>) -> Self {
        let common_main =
            ColMajorMatrix::from_row_major(&ctx.common_main.expect("must have common main"));
        let cached_mains = ctx
            .cached_mains
            .iter()
            .map(|d| {
                let trace = ColMajorMatrix::from_row_major(&d.trace);
                let (commit, data) = stacked_commit(
                    params.l_skip,
                    params.n_stack,
                    params.log_blowup,
                    params.k_whir,
                    &[&trace],
                );
                (commit, Arc::new(data))
            })
            .collect();
        Self {
            cached_mains,
            common_main,
            public_values: ctx.public_values,
        }
    }
}
