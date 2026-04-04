use openvm_cuda_common::{error::CudaError, stream::cudaStream_t};

extern "C" {
    fn _cuda_get_sm_count(device_id: u32, out_sm_count: *mut u32, stream: cudaStream_t) -> i32;
}

pub fn get_sm_count(device_id: u32, stream: cudaStream_t) -> Result<u32, CudaError> {
    let mut sm_count = 0u32;
    unsafe { CudaError::from_result(_cuda_get_sm_count(device_id, &mut sm_count, stream))? };
    Ok(sm_count)
}
