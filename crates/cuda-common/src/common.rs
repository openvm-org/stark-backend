use std::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::error::{check, CudaError};

#[link(name = "cudart")]
extern "C" {
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceReset() -> i32;
}

static DEVICE_RESET_EPOCH: AtomicU64 = AtomicU64::new(0);

pub fn get_device() -> Result<i32, CudaError> {
    let mut device = 0;
    unsafe {
        check(cudaGetDevice(&mut device))?;
    }
    assert!(device >= 0);
    Ok(device)
}

pub fn device_reset_epoch() -> u64 {
    DEVICE_RESET_EPOCH.load(Ordering::Acquire)
}

pub fn set_device_by_id(device: i32) -> Result<(), CudaError> {
    assert!(device >= 0);
    unsafe {
        check(cudaSetDevice(device))?;
        // Force primary-context initialization on the selected device.
        check(cudaFree(std::ptr::null_mut()))?;
    }
    Ok(())
}

pub fn set_device() -> Result<i32, CudaError> {
    let device = get_device()?;
    set_device_by_id(device)?;
    Ok(device)
}

pub fn reset_device() -> Result<(), CudaError> {
    check(unsafe { cudaDeviceReset() })?;
    DEVICE_RESET_EPOCH.fetch_add(1, Ordering::AcqRel);
    Ok(())
}
