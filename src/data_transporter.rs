use std::sync::Arc;

use cuda_kernels::matrix::matrix_transpose;
use cuda_utils::copy::MemCopyH2D;
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey,
    prover::{
        cpu::PcsData,
        hal::{DeviceDataTransporter, MatrixDimensions, TraceCommitter},
        types::{DeviceMultiStarkProvingKey, DeviceStarkProvingKey, SingleCommitPreimage},
    },
};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    base::DeviceMatrix,
    gpu_device::GpuDevice,
    prelude::{F, SC},
    prover_backend::{GpuBackend, GpuPcsData},
};

impl DeviceDataTransporter<SC, GpuBackend> for GpuDevice {
    fn transport_pk_to_device<'a>(
        &self,
        mpk: &'a MultiStarkProvingKey<SC>,
        air_ids: Vec<usize>,
    ) -> DeviceMultiStarkProvingKey<'a, GpuBackend>
    where
        SC: 'a,
    {
        assert!(
            air_ids.len() <= mpk.per_air.len(),
            "filtering more AIRs than available"
        );

        let per_air = air_ids
            .iter()
            .map(|&air_idx| {
                let pk = &mpk.per_air[air_idx];
                let preprocessed_data = pk.preprocessed_data.as_ref().map(|pd| {
                    let trace = self.transport_matrix_to_device(&pd.trace);
                    let (_, data) = self.commit(&[trace.clone()]);
                    SingleCommitPreimage {
                        trace,
                        data,
                        matrix_idx: 0,
                    }
                });

                DeviceStarkProvingKey {
                    air_name: &pk.air_name,
                    vk: &pk.vk,
                    preprocessed_data,
                    rap_partial_pk: pk.rap_partial_pk.clone(),
                }
            })
            .collect();

        DeviceMultiStarkProvingKey::new(
            air_ids,
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &Arc<RowMajorMatrix<F>>) -> DeviceMatrix<F> {
        // Convert RowMajorMatrix to flat vector and transfer to GPU
        let data = matrix.values.as_slice();
        let input_buffer = data.to_device().unwrap();
        let output = DeviceMatrix::<F>::with_capacity(matrix.height(), matrix.width());
        unsafe {
            matrix_transpose::<F>(
                output.buffer(),
                &input_buffer,
                matrix.width(),
                matrix.height(),
            )
            .unwrap();
        }
        assert_eq!(output.strong_count(), 1);

        output
    }

    fn transport_pcs_data_to_device(&self, _pcs_data: &PcsData<SC>) -> GpuPcsData {
        unimplemented!()
    }
}
