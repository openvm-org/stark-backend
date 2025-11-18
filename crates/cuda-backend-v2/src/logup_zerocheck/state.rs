use openvm_cuda_backend::base::DeviceMatrix;

use crate::{EF, F};

#[derive(Default)]
pub struct Round0Buffers {
    pub selectors_base: Vec<DeviceMatrix<F>>,
    pub eq_xi: Vec<DeviceMatrix<EF>>,
}
