use std::{fmt::Debug, sync::Arc};

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use openvm_stark_backend::{
    config::{Com, PcsProverData, Val},
    keygen::types::MultiStarkProvingKey,
    prover::{
        hal::{DeviceDataTransporter, MatrixDimensions, TraceCommitter},
        types::{
            CommittedTraceData, DeviceMultiStarkProvingKey, DeviceStarkProvingKey,
            SingleCommitPreimage,
        },
    },
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    base::DeviceMatrix,
    cuda::kernels::matrix::matrix_transpose,
    gpu_device::GpuDevice,
    prelude::{F, SC},
    prover_backend::GpuBackend,
};

impl DeviceDataTransporter<SC, GpuBackend> for GpuDevice {
    fn transport_pk_to_device(
        &self,
        mpk: &MultiStarkProvingKey<SC>,
    ) -> DeviceMultiStarkProvingKey<GpuBackend> {
        let per_air = mpk
            .per_air
            .iter()
            .map(|pk| {
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
                    air_name: pk.air_name.clone(),
                    vk: pk.vk.clone(),
                    preprocessed_data,
                    rap_partial_pk: pk.rap_partial_pk.clone(),
                }
            })
            .collect();

        DeviceMultiStarkProvingKey::new(
            per_air,
            mpk.trace_height_constraints.clone(),
            mpk.vk_pre_hash,
        )
    }

    fn transport_matrix_to_device(&self, matrix: &Arc<RowMajorMatrix<F>>) -> DeviceMatrix<F> {
        transport_matrix_to_device(matrix.clone())
    }

    /// We ignore the host prover data because it's faster to just re-commit on GPU instead of doing
    /// H2D transfer.
    fn transport_committed_trace_to_device(
        &self,
        commitment: Com<SC>,
        trace: &Arc<RowMajorMatrix<Val<SC>>>,
        _: &Arc<PcsProverData<SC>>,
    ) -> CommittedTraceData<GpuBackend> {
        let trace = self.transport_matrix_to_device(trace);
        let (d_commitment, data) = self.commit(&[trace.clone()]);
        assert_eq!(
            d_commitment, commitment,
            "GPU commitment does not match host"
        );
        CommittedTraceData {
            commitment,
            trace,
            data,
        }
    }

    fn transport_matrix_from_device_to_host(
        &self,
        matrix: &DeviceMatrix<F>,
    ) -> Arc<RowMajorMatrix<F>> {
        let matrix_host = transport_device_matrix_to_host(matrix);
        Arc::new(matrix_host)
    }
}

pub fn transport_matrix_to_device(matrix: Arc<RowMajorMatrix<F>>) -> DeviceMatrix<F> {
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

pub fn transport_device_matrix_to_host<T: Clone + Send + Sync>(
    matrix: &DeviceMatrix<T>,
) -> RowMajorMatrix<T> {
    let matrix_buffer = DeviceBuffer::<T>::with_capacity(matrix.height() * matrix.width());
    unsafe {
        matrix_transpose::<T>(
            &matrix_buffer,
            matrix.buffer(),
            matrix.height(),
            matrix.width(),
        )
        .unwrap();
    }
    RowMajorMatrix::<T>::new(matrix_buffer.to_host().unwrap(), matrix.width())
}

pub fn assert_eq_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    a: &DeviceMatrix<T>,
    b: &DeviceMatrix<T>,
) {
    assert_eq!(a.height(), b.height());
    assert_eq!(a.width(), b.width());
    assert_eq!(a.buffer().len(), b.buffer().len());
    let a_host = a.to_host().unwrap();
    let b_host = b.to_host().unwrap();
    for r in 0..a.height() {
        for c in 0..a.width() {
            assert_eq!(
                a_host[c * a.height() + r],
                b_host[c * b.height() + r],
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

pub fn assert_eq_host_and_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: Arc<RowMajorMatrix<T>>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = gpu.to_host().unwrap();
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu[c * cpu.height() + r],
                cpu.get(r, c),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}
