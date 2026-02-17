use std::sync::Arc;

use openvm_metal_common::copy::MemCopyH2D;
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    prover::{
        stacked_pcs::StackedPcsData, ColMajorMatrix, CommittedTraceData, DeviceDataTransporter,
        DeviceMultiStarkProvingKey, DeviceStarkProvingKey, MatrixDimensions,
    },
};

use crate::{
    base::MetalMatrix,
    prelude::{Digest, F, SC},
    AirDataMetal, MetalBackend, MetalDevice, StackedPcsDataMetal,
};

impl DeviceDataTransporter<SC, MetalBackend> for MetalDevice {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<MetalBackend> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|d| {
                    // Transport the preprocessed trace to a MetalMatrix
                    let cpu_mat = d.mat_view(0).to_matrix();
                    let metal_trace = transport_col_major_to_metal(&cpu_mat);
                    // Wrap the CPU PCS data
                    let metal_pcs = StackedPcsDataMetal {
                        inner: d.as_ref().clone(),
                    };
                    CommittedTraceData {
                        commitment: d.commit(),
                        trace: metal_trace,
                        data: Arc::new(metal_pcs),
                    }
                });
                DeviceStarkProvingKey {
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                    other_data: AirDataMetal,
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

    fn transport_matrix_to_device(&self, matrix: &ColMajorMatrix<F>) -> MetalMatrix<F> {
        transport_col_major_to_metal(matrix)
    }

    fn transport_pcs_data_to_device(
        &self,
        pcs_data: &StackedPcsData<F, Digest>,
    ) -> StackedPcsDataMetal {
        StackedPcsDataMetal {
            inner: pcs_data.clone(),
        }
    }

    fn transport_matrix_from_device_to_host(&self, matrix: &MetalMatrix<F>) -> ColMajorMatrix<F> {
        let values_host = matrix.buffer().to_vec();
        ColMajorMatrix::new(values_host, matrix.width())
    }
}

/// Transport a CPU ColMajorMatrix to a MetalMatrix.
fn transport_col_major_to_metal(matrix: &ColMajorMatrix<F>) -> MetalMatrix<F> {
    let buffer = matrix.values.to_device();
    MetalMatrix::new(Arc::new(buffer), matrix.height(), matrix.width())
}
