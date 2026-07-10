use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use openvm_cuda_common::{
    copy::{cuda_memcpy_on, MemCopyD2H},
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    stream::GpuDeviceCtx,
};
use openvm_stark_backend::prover::MatrixDimensions;

pub struct DeviceMatrix<T> {
    buffer: Arc<DeviceBuffer<T>>,
    /// Element offset of this matrix's first element within `buffer`. Non-zero only for views into
    /// a larger backing buffer (e.g. a trace column-view into the shared eval-form stacked arena).
    /// Pointer math must go through [`DeviceMatrix::as_ptr`] / [`DeviceMatrix::as_mut_ptr`], which
    /// honor this offset; `buffer()` still returns the whole backing buffer.
    offset: usize,
    height: usize,
    width: usize,
}

unsafe impl<T> Send for DeviceMatrix<T> {}
unsafe impl<T> Sync for DeviceMatrix<T> {}

impl<T> Clone for DeviceMatrix<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            offset: self.offset,
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
            offset: 0,
            height,
            width,
        }
    }

    /// Constructs a matrix that is a view into `buffer` starting at element `offset`.
    ///
    /// The view borrows a shared reference to `buffer` (via `Arc`) but does not own the whole
    /// allocation exclusively: multiple views may share one backing buffer. Dropping the view only
    /// frees the backing buffer once the last `Arc` reference is gone.
    ///
    /// # Panics
    /// Panics on zero dimensions or if `[offset, offset + height * width)` is out of bounds.
    pub fn view_of(
        buffer: Arc<DeviceBuffer<T>>,
        offset: usize,
        height: usize,
        width: usize,
    ) -> Self {
        assert_ne!(
            height * width,
            0,
            "Zero dimensions h {height} w {width} are wrong"
        );
        assert!(
            offset
                .checked_add(height * width)
                .is_some_and(|end| end <= buffer.len()),
            "View [offset {offset}, +{}) exceeds buffer length {}",
            height * width,
            buffer.len()
        );
        Self {
            buffer,
            offset,
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
            offset: 0,
            height: 0,
            width: 0,
        }
    }

    /// The full backing buffer, which may be larger than `height * width` for a view. Use
    /// [`DeviceMatrix::as_ptr`] / [`DeviceMatrix::as_mut_ptr`] for pointer math into this matrix's
    /// own data, and this only for capacity/whole-buffer operations on non-views.
    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    /// Element offset of this matrix within its backing [`DeviceMatrix::buffer`]. Zero unless this
    /// is a view.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Const device pointer to this matrix's first element, honoring the view offset.
    pub fn as_ptr(&self) -> *const T {
        // SAFETY: `offset` is within `buffer` bounds by construction (`new`/`view_of`).
        unsafe { self.buffer.as_ptr().add(self.offset) }
    }

    /// Mutable device pointer to this matrix's first element, honoring the view offset.
    pub fn as_mut_ptr(&self) -> *mut T {
        // SAFETY: `offset` is within `buffer` bounds by construction (`new`/`view_of`).
        unsafe { self.buffer.as_mut_ptr().add(self.offset) }
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.buffer)
    }

    pub fn as_view(&self) -> DeviceMatrixView<'_, T> {
        // SAFETY: buffer is borrowed for lifetime 'a of the view; `as_ptr` honors the offset.
        unsafe { DeviceMatrixView::from_raw_parts(self.as_ptr(), self.height, self.width) }
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
        // Only copy this matrix's own `height * width` elements starting at `offset`; for a view
        // the backing buffer is larger than the matrix.
        let len = self.height * self.width;
        if self.offset == 0 && self.buffer.len() == len {
            return self.buffer.to_host_on(device_ctx);
        }
        let mut host_vec = Vec::<T>::with_capacity(len);
        // SAFETY: `[offset, offset + len)` is in bounds (view invariant); `host_vec` has `len`
        // capacity. `to_host_sync` synchronizes before the host reads the data.
        unsafe {
            cuda_memcpy_on::<true, false>(
                host_vec.as_mut_ptr() as *mut std::ffi::c_void,
                self.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of::<T>() * len,
                device_ctx,
            )?;
            device_ctx.stream.to_host_sync()?;
            host_vec.set_len(len);
        }
        Ok(host_vec)
    }

    fn to_pinned_on(
        &self,
        dst: &mut openvm_cuda_common::pinned::PinnedBuffer<T>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<usize, MemCopyError>
    where
        T: Copy,
    {
        let len = self.height * self.width;
        if self.offset == 0 && self.buffer.len() == len {
            return self.buffer.to_pinned_on(dst, device_ctx);
        }
        dst.ensure_capacity(len)?;
        // SAFETY: `[offset, offset + len)` is in bounds (view invariant); `dst` has `len` capacity.
        unsafe {
            cuda_memcpy_on::<true, false>(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                self.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of::<T>() * len,
                device_ctx,
            )?;
        }
        device_ctx.stream.to_host_sync()?;
        Ok(len)
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
}
