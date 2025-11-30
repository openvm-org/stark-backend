use crate::error::CudaError;

#[allow(non_camel_case_types)]
pub(super) type CUdeviceptr = u64;
#[allow(non_camel_case_types)]
pub(super) type CUmemGenericAllocationHandle = u64;

extern "C" {
    fn _vpmm_check_support(device_ordinal: i32) -> i32;
    fn _vpmm_min_granularity(device_ordinal: i32, out: *mut usize) -> i32;
    fn _vpmm_reserve(size: usize, align: usize, out_va: *mut CUdeviceptr) -> i32;
    fn _vpmm_release_va(base: CUdeviceptr, size: usize) -> i32;
    fn _vpmm_create_physical(
        device_ordinal: i32,
        bytes: usize,
        out_h: *mut CUmemGenericAllocationHandle,
    ) -> i32;
    fn _vpmm_map(va: CUdeviceptr, bytes: usize, h: CUmemGenericAllocationHandle) -> i32;
    fn _vpmm_set_access(va: CUdeviceptr, bytes: usize, device_ordinal: i32) -> i32;
    fn _vpmm_unmap(va: CUdeviceptr, bytes: usize) -> i32;
    fn _vpmm_release(h: CUmemGenericAllocationHandle) -> i32;
}

pub(super) unsafe fn vpmm_check_support(device_ordinal: i32) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_check_support(device_ordinal))
}

pub(super) unsafe fn vpmm_min_granularity(device_ordinal: i32) -> Result<usize, CudaError> {
    let mut granularity: usize = 0;
    CudaError::from_result(_vpmm_min_granularity(device_ordinal, &mut granularity))?;
    Ok(granularity)
}

pub(super) unsafe fn vpmm_reserve(size: usize, align: usize) -> Result<CUdeviceptr, CudaError> {
    let mut va_base: CUdeviceptr = 0;
    CudaError::from_result(_vpmm_reserve(size, align, &mut va_base))?;
    Ok(va_base)
}

pub(super) unsafe fn vpmm_release_va(base: CUdeviceptr, size: usize) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_release_va(base, size))
}

pub(super) unsafe fn vpmm_create_physical(
    device_ordinal: i32,
    bytes: usize,
) -> Result<CUmemGenericAllocationHandle, CudaError> {
    let mut h: CUmemGenericAllocationHandle = 0;
    CudaError::from_result(_vpmm_create_physical(device_ordinal, bytes, &mut h))?;
    Ok(h)
}

pub(super) unsafe fn vpmm_map(
    va: CUdeviceptr,
    bytes: usize,
    h: CUmemGenericAllocationHandle,
) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_map(va, bytes, h))
}

pub(super) unsafe fn vpmm_set_access(
    va: CUdeviceptr,
    bytes: usize,
    device_ordinal: i32,
) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_set_access(va, bytes, device_ordinal))
}

pub(super) unsafe fn vpmm_unmap(va: CUdeviceptr, bytes: usize) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_unmap(va, bytes))
}

pub(super) unsafe fn vpmm_release(h: CUmemGenericAllocationHandle) -> Result<(), CudaError> {
    CudaError::from_result(_vpmm_release(h))
}
