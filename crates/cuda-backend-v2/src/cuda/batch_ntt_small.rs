use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    error::{CudaError, check},
};

use crate::F;

extern "C" {
    pub fn _batch_ntt_small(
        d_buffer: *mut F,
        l_skip: usize,
        cnt_blocks: usize,
        is_intt: bool,
    ) -> i32;
}

pub unsafe fn batch_ntt_small(
    buffer: &mut DeviceBuffer<F>,
    l_skip: usize,
    cnt_blocks: usize,
    is_intt: bool,
) -> Result<(), CudaError> {
    check(_batch_ntt_small(
        buffer.as_mut_ptr(),
        l_skip,
        cnt_blocks,
        is_intt,
    ))
}
