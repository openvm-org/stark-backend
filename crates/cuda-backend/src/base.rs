use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use openvm_cuda_common::{
    copy::MemCopyD2H,
    d_buffer::DeviceBuffer,
    error::{MemCopyError, MemoryError},
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::prover::MatrixDimensions;

/// Why a [`DeviceMatrix::adopt_release_stream_after_sync`] handoff was refused.
///
/// Deliberately its own type rather than a new `MemoryError` variant: callers
/// outside this crate match `MemoryError` exhaustively, and this failure is
/// specific to matrix ownership rather than to the allocator.
#[derive(Debug, thiserror::Error)]
pub enum DeviceMatrixStreamHandoffError {
    /// The buffer has other owners, so no single stream can order its release.
    #[error(
        "cannot hand off release ordering for an aliased device matrix          (Arc::strong_count = {strong_count}, expected 1)"
    )]
    Aliased { strong_count: usize },

    /// The allocator rejected the handoff, e.g. the pointer is not live.
    #[error(transparent)]
    Memory(#[from] MemoryError),
}

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

    pub fn with_capacity_on(height: usize, width: usize, device_ctx: &GpuDeviceCtx) -> Self {
        let buffer = DeviceBuffer::with_capacity_on(height * width, device_ctx);
        Self::new(Arc::new(buffer), height, width)
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

    /// Hands release ordering for this matrix's buffer to `target`'s stream.
    ///
    /// See [`DeviceBuffer::adopt_release_stream_after_sync`]. A matrix produced
    /// during trace generation on one stream and then proved on another must
    /// move its release onto the proving stream, or a sibling stream can reuse
    /// the allocation while the prove is still reading it.
    ///
    /// Fails rather than rebinding when the buffer is aliased. Release ordering
    /// names exactly one stream, so an aliased buffer would leave every other
    /// owner reading memory that some other stream is now free to reuse — the
    /// bug this operation exists to prevent. The error carries the observed
    /// `Arc::strong_count` so the caller can trace the alias.
    ///
    /// # Safety
    /// The caller must uphold [`DeviceBuffer::adopt_release_stream_after_sync`]'s
    /// contract: all work on the current release stream is complete, and this
    /// matrix is the sole owner of the buffer for the rest of its life.
    pub unsafe fn adopt_release_stream_after_sync(
        &mut self,
        target: &GpuDeviceCtx,
    ) -> Result<(), DeviceMatrixStreamHandoffError> {
        let strong_count = Arc::strong_count(&self.buffer);
        let buffer = Arc::get_mut(&mut self.buffer)
            .ok_or(DeviceMatrixStreamHandoffError::Aliased { strong_count })?;
        buffer.adopt_release_stream_after_sync(target)?;
        Ok(())
    }

    pub fn as_view(&self) -> DeviceMatrixView<'_, T> {
        // SAFETY: buffer is borrowed for lifetime 'a of the view
        unsafe { DeviceMatrixView::from_raw_parts(self.buffer.as_ptr(), self.height, self.width) }
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
    fn to_host_on(&self, device_ctx: &GpuDeviceCtx) -> Result<Vec<T>, MemCopyError> {
        self.buffer.to_host_on(device_ctx)
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

/// View of a device matrix. Dropping does not free memory.
#[derive(Clone, Copy)]
pub struct DeviceMatrixView<'a, T> {
    ptr: *const T,
    height: usize,
    width: usize,
    _ptr_lifetime: PhantomData<&'a T>,
}

unsafe impl<T> Send for DeviceMatrixView<'_, T> {}
unsafe impl<T> Sync for DeviceMatrixView<'_, T> {}

impl<T> DeviceMatrixView<'_, T> {
    /// # Safety
    /// - The pointer must be valid for the lifetime of the view.
    /// - The pointer must have memory allocated for the following `height * width` elements of `T`.
    pub unsafe fn from_raw_parts(ptr: *const T, height: usize, width: usize) -> Self {
        Self {
            ptr,
            height,
            width,
            _ptr_lifetime: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T> MatrixDimensions for DeviceMatrixView<'_, T> {
    #[inline]
    fn height(&self) -> usize {
        self.height
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }
}

/// The following trait and types are borrowed from [halo2](https:://github.com/zcash/halo2).
/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

pub struct DevicePoly<T, B> {
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
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_matrix() {
        let device_ctx = GpuDeviceCtx::for_current_device().unwrap();
        let buffer = Arc::new(DeviceBuffer::<i32>::with_capacity_on(12, &device_ctx));
        let matrix = DeviceMatrix::<i32>::new(buffer, 3, 4);
        assert_eq!(matrix.height(), 3);
        assert_eq!(matrix.width(), 4);
    }

    #[test]
    fn release_stream_handoff_accepts_a_solely_owned_matrix() {
        let producer = GpuDeviceCtx::for_current_device().unwrap();
        let consumer = GpuDeviceCtx::for_current_device().unwrap();
        let buffer = Arc::new(DeviceBuffer::<i32>::with_capacity_on(12, &producer));
        let mut matrix = DeviceMatrix::<i32>::new(buffer, 3, 4);
        assert_eq!(matrix.strong_count(), 1);

        unsafe { matrix.adopt_release_stream_after_sync(&consumer) }
            .expect("a solely owned matrix may hand off its release stream");
    }

    #[test]
    fn release_stream_handoff_refuses_an_aliased_matrix() {
        let producer = GpuDeviceCtx::for_current_device().unwrap();
        let consumer = GpuDeviceCtx::for_current_device().unwrap();
        let buffer = Arc::new(DeviceBuffer::<i32>::with_capacity_on(12, &producer));
        let mut matrix = DeviceMatrix::<i32>::new(buffer, 3, 4);
        // The alias is what makes a single release stream wrong: this clone can
        // go on reading on some third stream after the handoff.
        let alias = matrix.clone();

        let error = unsafe { matrix.adopt_release_stream_after_sync(&consumer) }
            .expect_err("an aliased matrix must not silently rebind");
        assert!(matches!(
            error,
            DeviceMatrixStreamHandoffError::Aliased { strong_count: 2 }
        ));
        drop(alias);
    }
}
