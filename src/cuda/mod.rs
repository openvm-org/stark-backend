#![allow(clippy::missing_safety_doc)]

pub mod common;
pub mod copy;
pub mod d_buffer;
pub mod error;
pub(crate) mod kernels;
pub mod memory_manager;
pub mod stream;

pub mod poseidon2_constants {
    use crate::cuda::error::CudaError;

    extern "C" {
        fn _init_poseidon2_constants(
            initial_round_constants: *const std::ffi::c_void,
            terminal_round_constants: *const std::ffi::c_void,
            internal_round_constants: *const std::ffi::c_void,
        ) -> i32;
    }

    pub unsafe fn init_poseidon2_constants<T>(
        initial_round_constants: *const [T; 16],
        terminal_round_constants: *const [T; 16],
        internal_round_constants: *const T,
    ) -> Result<(), CudaError> {
        CudaError::from_result(_init_poseidon2_constants(
            initial_round_constants as *const _,
            terminal_round_constants as *const _,
            internal_round_constants as *const _,
        ))
    }
}
