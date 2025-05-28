use std::marker::PhantomData;

use openvm_stark_backend::prover::hal::MatrixDimensions;

use crate::{base::DeviceMatrix, gpu_device::GpuDevice, prelude::F};

pub(crate) mod ops;
use ops::*;

/// The top-level LDE abstraction, composed of general matrix access (dimensions),
/// trace access, and LDE behavior (which varies by mode).
pub trait GpuLde: MatrixDimensions + LdeCommon {
    /// Constructs a new LDE wrapper using the given trace matrix, blowup factor, and shift.
    ///
    /// - `added_bits` determines the blowup factor for LDE computation.
    /// - `shift` determines the coset used for LDE.
    fn new(device: &GpuDevice, matrix: DeviceMatrix<F>, added_bits: usize, shift: F) -> Self
    where
        Self: Sized;

    /// Returns the LDE matrix on specific domain
    ///
    /// - Cached: returns precomputed LDE supposed that domain size < height.
    /// - OnDemand: always computes a fresh copy.
    fn take_lde(&self, domain_size: usize) -> DeviceMatrix<F>;

    /// Returns the LDE rows for the given indices.
    fn get_lde_rows(&self, row_indices: &[usize]) -> DeviceMatrix<F>;

    /// Converts the trace to coefficient form (Cached does nothing)
    fn to_coefficient_form(&mut self);
}

pub trait LdeCommon {
    /// Returns the number of rows in the original trace matrix.
    fn trace_height(&self) -> usize;

    fn shift(&self) -> F;
}

#[derive(Clone, Debug)]
enum GpuLdeDataType {
    Trace {
        trace: DeviceMatrix<F>,
        coef_form: bool,
    },
    Lde(DeviceMatrix<F>),
}

pub trait GpuLdeMode {}

/// Fully precomputes and stores LDE matrix at construction.
pub struct Cached;
/// Computes LDE matrix on each request without storing it.
pub struct OnDemand;

impl GpuLdeMode for Cached {}
impl GpuLdeMode for OnDemand {}

pub type GpuLdeCached = GpuLdeImpl<Cached>;
pub type GpuLdeOnDemand = GpuLdeImpl<OnDemand>;

pub struct GpuLdeImpl<M: GpuLdeMode> {
    device_id: u32,
    data: GpuLdeDataType,
    added_bits: usize,
    shift: F,
    _mode: PhantomData<M>,
}

impl<M: GpuLdeMode> MatrixDimensions for GpuLdeImpl<M> {
    fn height(&self) -> usize {
        match &self.data {
            GpuLdeDataType::Lde(lde) => lde.height(),
            GpuLdeDataType::Trace { trace, .. } => trace.height() << self.added_bits,
        }
    }

    fn width(&self) -> usize {
        match &self.data {
            GpuLdeDataType::Lde(lde) => lde.width(),
            GpuLdeDataType::Trace { trace, .. } => trace.width(),
        }
    }
}

impl<M: GpuLdeMode> LdeCommon for GpuLdeImpl<M> {
    fn shift(&self) -> F {
        self.shift
    }

    fn trace_height(&self) -> usize {
        match &self.data {
            GpuLdeDataType::Trace { trace, .. } => trace.height(),
            GpuLdeDataType::Lde(lde) => lde.height() >> self.added_bits,
        }
    }
}

impl GpuLde for GpuLdeCached {
    fn new(device: &GpuDevice, matrix: DeviceMatrix<F>, added_bits: usize, shift: F) -> Self {
        if added_bits == 0 {
            return Self {
                device_id: device.id,
                data: GpuLdeDataType::Lde(matrix),
                added_bits,
                shift,
                _mode: PhantomData,
            };
        }
        let trace_height = matrix.height();
        let lde_height = trace_height << added_bits;
        let lde = compute_lde_matrix::<true>(matrix, device.id, lde_height, shift);
        Self {
            device_id: device.id,
            data: GpuLdeDataType::Lde(lde),
            added_bits,
            shift,
            _mode: PhantomData,
        }
    }

    fn take_lde(&self, domain_size: usize) -> DeviceMatrix<F> {
        match &self.data {
            GpuLdeDataType::Lde(lde) => {
                assert!(lde.height() >= domain_size);
                lde.clone()
            }
            _ => panic!("Cached LDE mode: LDE should have been precomputed"),
        }
    }

    fn get_lde_rows(&self, row_indices: &[usize]) -> DeviceMatrix<F> {
        assert!(!row_indices.is_empty());
        match &self.data {
            GpuLdeDataType::Lde(lde) => get_rows_from_matrix(lde, row_indices),
            _ => panic!("Cached LDE mode: LDE should have been precomputed"),
        }
    }

    fn to_coefficient_form(&mut self) {}
}

impl GpuLde for GpuLdeOnDemand {
    fn new(device: &GpuDevice, matrix: DeviceMatrix<F>, added_bits: usize, shift: F) -> Self {
        let data = if added_bits == 0 {
            GpuLdeDataType::Lde(matrix)
        } else {
            GpuLdeDataType::Trace {
                trace: matrix,
                coef_form: false,
            }
        };
        Self {
            device_id: device.id,
            data,
            added_bits,
            shift,
            _mode: PhantomData,
        }
    }

    fn take_lde(&self, domain_size: usize) -> DeviceMatrix<F> {
        match &self.data {
            GpuLdeDataType::Lde(lde) => {
                assert!(lde.height() >= domain_size);
                lde.clone()
            }

            GpuLdeDataType::Trace { trace, coef_form } => {
                if *coef_form {
                    compute_lde_matrix::<false>(
                        trace.clone(),
                        self.device_id,
                        self.height(),
                        self.shift,
                    )
                } else {
                    compute_lde_matrix::<true>(
                        trace.clone(),
                        self.device_id,
                        self.height(),
                        self.shift,
                    )
                }
            }
        }
    }

    fn get_lde_rows(&self, row_indices: &[usize]) -> DeviceMatrix<F> {
        assert!(!row_indices.is_empty());
        match &self.data {
            GpuLdeDataType::Lde(lde) => get_rows_from_matrix(lde, row_indices),
            GpuLdeDataType::Trace { trace, coef_form } => {
                if *coef_form {
                    polynomial_evaluate(trace, self.shift, self.height(), row_indices)
                } else {
                    let trace = inplace_ifft(trace.clone(), self.device_id);
                    polynomial_evaluate(&trace, self.shift, self.height(), row_indices)
                }
            }
        }
    }

    // TODO: rename
    fn to_coefficient_form(&mut self) {
        if let GpuLdeDataType::Trace { trace, coef_form } = &self.data {
            if !coef_form {
                let trace = inplace_ifft(trace.clone(), self.device_id);
                self.data = GpuLdeDataType::Trace {
                    trace,
                    coef_form: true,
                };
            }
        }
    }
}

#[cfg(feature = "lde-on-demand")]
pub type GpuLdeDefault = GpuLdeOnDemand;

#[cfg(not(feature = "lde-on-demand"))]
pub type GpuLdeDefault = GpuLdeCached;
