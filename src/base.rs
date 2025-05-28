use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use cuda_utils::{copy::MemCopyD2H, d_buffer::DeviceBuffer, error::MemCopyError};
use openvm_stark_backend::prover::hal::MatrixDimensions;

pub struct DeviceMatrix<T> {
    buffer: Arc<DeviceBuffer<T>>,
    height: usize,
    width: usize,
}

unsafe impl<T> Send for DeviceMatrix<T> {}
unsafe impl<T> Sync for DeviceMatrix<T> {}

impl<T> Clone for DeviceMatrix<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            height: self.height,
            width: self.width,
        }
    }
}

impl<T> Drop for DeviceMatrix<T> {
    fn drop(&mut self) {
        tracing::debug!(
            "Dropping DeviceMatrix of size {} with Arc strong count={}",
            self.buffer.len(),
            self.strong_count()
        );
    }
}

impl<T> DeviceMatrix<T> {
    pub fn new(buffer: Arc<DeviceBuffer<T>>, height: usize, width: usize) -> Self {
        assert_ne!(
            height * width,
            0,
            "Zero dimensions h {} w {} are wrong",
            height,
            width
        );
        assert_eq!(
            buffer.len(),
            height * width,
            "Buffer size must match dimensions"
        );
        Self {
            buffer,
            height,
            width,
        }
    }

    pub fn with_capacity(height: usize, width: usize) -> Self {
        Self {
            buffer: Arc::new(DeviceBuffer::with_capacity(height * width)),
            height,
            width,
        }
    }

    pub fn dummy() -> Self {
        Self {
            buffer: Arc::new(DeviceBuffer::new()),
            height: 0,
            width: 0,
        }
    }

    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.buffer)
    }
}

impl<T> MatrixDimensions for DeviceMatrix<T> {
    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }
}

impl<T> MemCopyD2H<T> for DeviceMatrix<T> {
    fn to_host(&self) -> Result<Vec<T>, MemCopyError> {
        self.buffer.to_host()
    }
}

impl<T: Debug> Debug for DeviceMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceMatrix (height = {}, width = {}): {:?}",
            self.height(),
            self.width(),
            self.buffer()
        )
    }
}

/// The following trait and types are borrowed from [halo2](https:://github.com/zcash/halo2).
/// The basis over which a polynomial is described.
#[allow(dead_code)]
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

pub struct DevicePoly<T, B> {
    #[allow(dead_code)]
    pub is_bit_reversed: bool,
    pub coeff: DeviceBuffer<T>,
    _marker: PhantomData<B>,
}

impl<T, B> DevicePoly<T, B> {
    pub fn new(is_bit_reversed: bool, coeff: DeviceBuffer<T>) -> Self {
        Self {
            is_bit_reversed,
            coeff,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coeff.len()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_matrix() {
        let buffer = Arc::new(DeviceBuffer::<i32>::with_capacity(12));
        let matrix = DeviceMatrix::<i32>::new(buffer, 3, 4);
        assert_eq!(matrix.height(), 3);
        assert_eq!(matrix.width(), 4);
    }
}
