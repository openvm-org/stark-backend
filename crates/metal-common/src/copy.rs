use std::ptr;

use crate::{
    command,
    d_buffer::MetalBuffer,
    device::get_context,
    error::MemCopyError,
};

/// Host-to-Device copy trait.
///
/// On Metal with `StorageModeShared`, this is a direct `memcpy` into the
/// buffer's unified memory — no DMA transfer is needed.
pub trait MemCopyH2D<T> {
    fn copy_to(&self, dst: &MetalBuffer<T>) -> Result<(), MemCopyError>;
    fn to_device(&self) -> MetalBuffer<T>;
}

impl<T> MemCopyH2D<T> for [T] {
    fn copy_to(&self, dst: &MetalBuffer<T>) -> Result<(), MemCopyError> {
        if self.len() > dst.len() {
            return Err(MemCopyError::SizeMismatch {
                src_len: self.len(),
                dst_len: dst.len(),
            });
        }
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), dst.as_mut_ptr(), self.len());
        }
        Ok(())
    }

    fn to_device(&self) -> MetalBuffer<T> {
        MetalBuffer::from_slice(self)
    }
}

/// Device-to-Host copy trait.
///
/// On Metal with `StorageModeShared`, this is a direct read from unified memory.
/// **Ensure GPU work has completed before calling** (e.g., via `cmd_buffer.wait_until_completed()`).
pub trait MemCopyD2H<T> {
    fn to_host(&self) -> Vec<T>;
}

impl<T> MemCopyD2H<T> for MetalBuffer<T> {
    fn to_host(&self) -> Vec<T> {
        self.to_vec()
    }
}

/// Device-to-Device copy trait.
///
/// Uses a Metal blit command encoder for GPU-side copies.
pub trait MemCopyD2D<T> {
    fn device_copy(&self) -> MetalBuffer<T>;
    fn device_copy_to(&self, dst: &MetalBuffer<T>) -> Result<(), MemCopyError>;
}

impl<T> MemCopyD2D<T> for MetalBuffer<T> {
    fn device_copy(&self) -> MetalBuffer<T> {
        let dst = MetalBuffer::<T>::with_capacity(self.len());
        self.device_copy_to(&dst)
            .expect("device_copy failed");
        dst
    }

    fn device_copy_to(&self, dst: &MetalBuffer<T>) -> Result<(), MemCopyError> {
        if self.len() > dst.len() {
            return Err(MemCopyError::SizeMismatch {
                src_len: self.len(),
                dst_len: dst.len(),
            });
        }
        let size_bytes = self.size_bytes();
        let ctx = get_context();
        command::blit_operation(&ctx.queue, |blit| {
            blit.copy_from_buffer(
                self.gpu_buffer(),
                0,
                dst.gpu_buffer(),
                0,
                size_bytes as u64,
            );
        })
        .expect("blit copy failed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2d_and_d2h() {
        let data = vec![1u32, 2, 3, 4, 5];
        let buf = data.to_device();
        let result = buf.to_host();
        assert_eq!(data, result);
    }

    #[test]
    fn test_copy_to_existing() {
        let data = vec![10u32, 20, 30];
        let buf = MetalBuffer::<u32>::with_capacity(5);
        data.copy_to(&buf).unwrap();
        let result = buf.to_vec();
        assert_eq!(&result[..3], &[10, 20, 30]);
    }

    #[test]
    fn test_copy_to_size_mismatch() {
        let data = vec![1u32; 10];
        let buf = MetalBuffer::<u32>::with_capacity(5);
        let err = data.copy_to(&buf);
        assert!(err.is_err());
    }

    #[test]
    fn test_d2d_copy() {
        let data = vec![42u64, 43, 44, 45];
        let src = MetalBuffer::from_slice(&data);
        let dst = src.device_copy();
        assert_eq!(dst.to_vec(), data);
    }
}
