use crate::error::CudaError;

#[allow(non_camel_case_types)]
pub(super) type CUdeviceptr = u64;
#[allow(non_camel_case_types)]
pub(super) type CUmemGenericAllocationHandle = u64;

extern "C" {
    fn ax_vmm_check_support(device_ordinal: i32) -> i32;
    fn ax_vmm_min_granularity(device_ordinal: i32, out: *mut usize) -> i32;
    fn ax_vmm_reserve(size: usize, align: usize, out_va: *mut CUdeviceptr) -> i32;
    fn ax_vmm_release_va(base: CUdeviceptr, size: usize) -> i32;
    fn ax_vmm_create_physical(
        device_ordinal: i32,
        bytes: usize,
        out_h: *mut CUmemGenericAllocationHandle,
    ) -> i32;
    fn ax_vmm_map_set_access(
        va: CUdeviceptr,
        bytes: usize,
        h: CUmemGenericAllocationHandle,
        device_ordinal: i32,
    ) -> i32;
    fn ax_vmm_unmap_release(va: CUdeviceptr, bytes: usize, h: CUmemGenericAllocationHandle) -> i32;
    fn ax_vmm_error_string(code: i32, out: *mut *const i8) -> i32;
}

pub(super) unsafe fn vmm_check_support(device_ordinal: i32) -> Result<(), CudaError> {
    CudaError::from_result(ax_vmm_check_support(device_ordinal))
}

pub(super) unsafe fn vmm_min_granularity(device_ordinal: i32) -> Result<usize, CudaError> {
    let mut granularity: usize = 0;
    CudaError::from_result(ax_vmm_min_granularity(device_ordinal, &mut granularity))?;
    Ok(granularity)
}

pub(super) unsafe fn vmm_reserve(size: usize, align: usize) -> Result<CUdeviceptr, CudaError> {
    let mut va_base: CUdeviceptr = 0;
    CudaError::from_result(ax_vmm_reserve(size, align, &mut va_base))?;
    Ok(va_base)
}

pub(super) unsafe fn vmm_release_va(base: CUdeviceptr, size: usize) -> Result<(), CudaError> {
    CudaError::from_result(ax_vmm_release_va(base, size))
}

pub(super) unsafe fn vmm_create_physical(
    device_ordinal: i32,
    bytes: usize,
) -> Result<CUmemGenericAllocationHandle, CudaError> {
    let mut h: CUmemGenericAllocationHandle = 0;
    CudaError::from_result(ax_vmm_create_physical(device_ordinal, bytes, &mut h))?;
    Ok(h)
}

pub(super) unsafe fn vmm_map_set_access(
    va: CUdeviceptr,
    bytes: usize,
    h: CUmemGenericAllocationHandle,
    device_ordinal: i32,
) -> Result<(), CudaError> {
    CudaError::from_result(ax_vmm_map_set_access(va, bytes, h, device_ordinal))
}

pub(super) unsafe fn vmm_unmap_release(
    va: CUdeviceptr,
    bytes: usize,
    h: CUmemGenericAllocationHandle,
) -> Result<(), CudaError> {
    CudaError::from_result(ax_vmm_unmap_release(va, bytes, h))
}
