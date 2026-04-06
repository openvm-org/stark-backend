use std::ffi::c_void;

use crate::{
    d_buffer::DeviceBuffer,
    error::{check, MemCopyError},
    stream::{cudaStream_t, DeviceContext},
};

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

/// FFI binding for `cudaMemcpyAsync` on an explicit stream.
///
/// # Safety
/// Must follow the rules of the `cudaMemcpyAsync` function from the CUDA runtime API.
pub unsafe fn cuda_memcpy_on<const SRC_DEVICE: bool, const DST_DEVICE: bool>(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    ctx: &DeviceContext,
) -> Result<(), MemCopyError> {
    check(unsafe {
        cudaMemcpyAsync(
            dst,
            src,
            size_bytes,
            std::mem::transmute::<i32, cudaMemcpyKind>(
                if DST_DEVICE { 1 } else { 0 } + if SRC_DEVICE { 2 } else { 0 },
            ),
            ctx.stream.as_raw(),
        )
    })
    .map_err(MemCopyError::from)
}

// ---- Host -> Device ----

pub trait MemCopyH2D<T> {
    fn copy_to_on(
        &self,
        dst: &mut DeviceBuffer<T>,
        ctx: &DeviceContext,
    ) -> Result<(), MemCopyError>;
    fn to_device_on(&self, ctx: &DeviceContext) -> Result<DeviceBuffer<T>, MemCopyError>;
}

impl<T> MemCopyH2D<T> for [T] {
    fn copy_to_on(
        &self,
        dst: &mut DeviceBuffer<T>,
        ctx: &DeviceContext,
    ) -> Result<(), MemCopyError> {
        if self.len() > dst.len() {
            return Err(MemCopyError::SizeMismatch {
                operation: "copy_to_device_on",
                src_len: self.len(),
                dst_len: dst.len(),
            });
        }
        let size_bytes = std::mem::size_of_val(self);
        check(unsafe {
            cudaMemcpyAsync(
                dst.as_mut_raw_ptr(),
                self.as_ptr() as *const c_void,
                size_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                ctx.stream.as_raw(),
            )
        })
        .map_err(MemCopyError::from)
    }

    fn to_device_on(&self, ctx: &DeviceContext) -> Result<DeviceBuffer<T>, MemCopyError> {
        let mut dst = DeviceBuffer::with_capacity_on(self.len(), ctx);
        self.copy_to_on(&mut dst, ctx)?;
        Ok(dst)
    }
}

// ---- Device -> Host ----

pub trait MemCopyD2H<T> {
    fn to_host_on(&self, ctx: &DeviceContext) -> Result<Vec<T>, MemCopyError>;
}

impl<T> MemCopyD2H<T> for DeviceBuffer<T> {
    fn to_host_on(&self, ctx: &DeviceContext) -> Result<Vec<T>, MemCopyError> {
        let mut host_vec = Vec::with_capacity(self.len());
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            cudaMemcpyAsync(
                host_vec.as_mut_ptr() as *mut c_void,
                self.as_raw_ptr(),
                size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ctx.stream.as_raw(),
            )
        })?;
        ctx.stream.to_host_sync()?;
        unsafe {
            host_vec.set_len(self.len());
        }

        Ok(host_vec)
    }
}

// ---- Device -> Device ----

pub trait MemCopyD2D<T> {
    fn device_copy_on(&self, ctx: &DeviceContext) -> Result<DeviceBuffer<T>, MemCopyError>;
    fn device_copy_to_on(
        &self,
        dst: &mut DeviceBuffer<T>,
        ctx: &DeviceContext,
    ) -> Result<(), MemCopyError>;
}

impl<T> MemCopyD2D<T> for DeviceBuffer<T> {
    fn device_copy_on(&self, ctx: &DeviceContext) -> Result<DeviceBuffer<T>, MemCopyError> {
        let mut dst = DeviceBuffer::<T>::with_capacity_on(self.len(), ctx);
        self.device_copy_to_on(&mut dst, ctx)?;
        Ok(dst)
    }

    fn device_copy_to_on(
        &self,
        dst: &mut DeviceBuffer<T>,
        ctx: &DeviceContext,
    ) -> Result<(), MemCopyError> {
        if self.len() > dst.len() {
            return Err(MemCopyError::SizeMismatch {
                operation: "device_copy_to_on",
                src_len: self.len(),
                dst_len: dst.len(),
            });
        }
        let size_bytes = std::mem::size_of::<T>() * self.len();

        check(unsafe {
            cudaMemcpyAsync(
                dst.as_mut_raw_ptr(),
                self.as_raw_ptr(),
                size_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream.as_raw(),
            )
        })
        .map_err(MemCopyError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d_buffer::DeviceBuffer;

    fn test_ctx() -> DeviceContext {
        DeviceContext::for_current_device().unwrap()
    }

    #[test]
    fn test_mem_copy() {
        let ctx = test_ctx();
        let h = vec![1, 2, 3, 4, 5];

        let d1 = h.to_device_on(&ctx).unwrap();
        let mut d2 = DeviceBuffer::<i32>::with_capacity_on(h.len(), &ctx);
        h.copy_to_on(&mut d2, &ctx).unwrap();

        let h1 = d1.to_host_on(&ctx).unwrap();
        let h2 = d2.to_host_on(&ctx).unwrap();

        assert_eq!(h, h1, "First device buffer mismatch");
        assert_eq!(h, h2, "Second device buffer mismatch");
    }
}
