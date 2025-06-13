use std::ffi::c_void;

use crate::cuda::error::{check, CudaError};

#[allow(non_camel_case_types)]
type cudaMemPool_t = *mut c_void;
#[allow(non_upper_case_globals)]
pub const cudaMemPoolAttrReleaseThreshold: u32 = 4;
#[allow(non_upper_case_globals)]
pub const cudaMemPoolAttrReuseFollowEventDependencies: u32 = 1;

#[link(name = "cudart")]
extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaSetDevice(id: i32) -> i32;
    fn cudaDeviceReset() -> i32;
    fn cudaDeviceGetDefaultMemPool(pool: *mut cudaMemPool_t, device: i32) -> i32;
    fn cudaMemPoolSetAttribute(pool: cudaMemPool_t, attr: u32, value: *const c_void) -> i32;
    fn cudaDeviceSetMemPool(device: i32, pool: cudaMemPool_t) -> i32;
}

pub fn set_device(id: i32) -> Result<(), CudaError> {
    std::env::set_var("CUDA_DEVICE_DEFAULT_STREAM", "per-thread");
    unsafe {
        check(cudaSetDevice(id))?;

        let mut device = 0;
        check(cudaGetDevice(&mut device))?;

        let mut pool: cudaMemPool_t = std::ptr::null_mut();
        check(cudaDeviceGetDefaultMemPool(&mut pool, device))?;

        let reuse: i32 = 1;
        check(cudaMemPoolSetAttribute(
            pool,
            cudaMemPoolAttrReuseFollowEventDependencies,
            &reuse as *const i32 as *const c_void,
        ))?;

        // 2. Set release threshold to 512 MB
        let threshold: usize = 512 * 1024 * 1024;
        check(cudaMemPoolSetAttribute(
            pool,
            cudaMemPoolAttrReleaseThreshold,
            &threshold as *const usize as *const c_void,
        ))?;

        // 3. Optional but safe: assign pool back to device
        check(cudaDeviceSetMemPool(device, pool))?;

        assert_eq!(device, id);
    }
    Ok(())
}

pub fn reset_device() -> Result<(), CudaError> {
    check(unsafe { cudaDeviceReset() })
}
