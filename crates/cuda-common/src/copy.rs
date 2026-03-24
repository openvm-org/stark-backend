use std::{ffi::c_void, sync::Mutex};

use lazy_static::lazy_static;

use crate::{
    d_buffer::DeviceBuffer,
    error::{check, MemCopyError},
    stream::{cudaStreamPerThread, cudaStream_t, CudaEvent},
};

lazy_static! {
    static ref COPY_EVENT: Mutex<CudaEvent> = Mutex::new(CudaEvent::new().unwrap());
}

#[repr(i32)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}

#[link(name = "cudart")]
extern "C" {
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> i32;
}

/// FFI binding for the `cudaMemcpyAsync` function on the default cuda stream.
///
/// # Safety
/// Must follow the rules of the `cudaMemcpyAsync` function from the CUDA runtime API.
pub unsafe fn cuda_memcpy<const SRC_DEVICE: bool, const DST_DEVICE: bool>(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
) -> Result<(), MemCopyError> {
    check(unsafe {
        cudaMemcpyAsync(
            dst,
            src,
            size_bytes,
            std::mem::transmute::<i32, cudaMemcpyKind>(
                if DST_DEVICE { 1 } else { 0 } + if SRC_DEVICE { 2 } else { 0 },
            ),
            cudaStreamPerThread,
        )
    })
    .map_err(MemCopyError::from)
}

// Host -> Device
pub trait MemCopyH2D<T> {
    fn copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError>;
    fn to_device(&self) -> Result<DeviceBuffer<T>, MemCopyError>;
}

impl<T> MemCopyH2D<T> for [T] {
    fn copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError> {
        if self.len() > dst.len() {
            return Err(MemCopyError::SizeMismatch {
                operation: "copy_to_device",
                host_len: self.len(),
                device_len: dst.len(),
            });
        }
        let size_bytes = std::mem::size_of_val(self);
        check(unsafe {
            cudaMemcpyAsync(
                dst.as_mut_raw_ptr(),
                self.as_ptr() as *const c_void,
                size_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                cudaStreamPerThread,
            )
        })
        .map_err(MemCopyError::from)
    }

    fn to_device(&self) -> Result<DeviceBuffer<T>, MemCopyError> {
        let mut dst = DeviceBuffer::with_capacity(self.len());
        self.copy_to(&mut dst)?;
        Ok(dst)
    }
}

// Device -> Host
pub trait MemCopyD2H<T> {
    fn to_host(&self) -> Result<Vec<T>, MemCopyError>;
}

impl<T> MemCopyD2H<T> for DeviceBuffer<T> {
    fn to_host(&self) -> Result<Vec<T>, MemCopyError> {
        let mut host_vec = Vec::with_capacity(self.len());
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            cudaMemcpyAsync(
                host_vec.as_mut_ptr() as *mut c_void,
                self.as_raw_ptr(),
                size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                cudaStreamPerThread,
            )
        })?;
        unsafe {
            COPY_EVENT
                .lock()
                .unwrap()
                .record_and_wait(cudaStreamPerThread)?;

            host_vec.set_len(self.len());
        }

        Ok(host_vec)
    }
}

pub trait MemCopyD2D<T> {
    fn device_copy(&self) -> Result<DeviceBuffer<T>, MemCopyError>;
    fn device_copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError>;
}

impl<T> MemCopyD2D<T> for DeviceBuffer<T> {
    fn device_copy(&self) -> Result<DeviceBuffer<T>, MemCopyError> {
        let mut dst = DeviceBuffer::<T>::with_capacity(self.len());
        self.device_copy_to(&mut dst)?;
        Ok(dst)
    }

    fn device_copy_to(&self, dst: &mut DeviceBuffer<T>) -> Result<(), MemCopyError> {
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            cudaMemcpyAsync(
                dst.as_mut_raw_ptr(),
                self.as_raw_ptr(),
                size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                cudaStreamPerThread,
            )
        })
        .map_err(MemCopyError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d_buffer::DeviceBuffer;

    #[test]
    fn test_mem_copy() {
        // Our source data on the host
        let h = vec![1, 2, 3, 4, 5];

        // 1) Copy to a newly allocated device buffer
        let d1 = h.to_device().unwrap();

        // 2) Create another device buffer of the same size
        let mut d2 = DeviceBuffer::<i32>::with_capacity(h.len());

        // 3) Copy into that existing buffer
        h.copy_to(&mut d2).unwrap();

        // 4) Copy both buffers back to host
        let h1 = d1.to_host().unwrap();
        let h2 = d2.to_host().unwrap();

        assert_eq!(h, h1, "First device buffer mismatch");
        assert_eq!(h, h2, "Second device buffer mismatch");
    }
}
