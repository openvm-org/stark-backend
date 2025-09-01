use std::{
    ffi::CStr,
    os::raw::{c_char, c_int},
};

use thiserror::Error;

#[link(name = "cudart")]
extern "C" {
    fn cudaGetErrorString(error: c_int) -> *const c_char;
    fn cudaGetErrorName(error: c_int) -> *const c_char;
}

/// Safely convert a C string pointer returned by CUDA into a Rust `String`.
fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "Unknown CUDA error (null pointer)".to_string();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

/// Returns the symbolic error name (e.g. "cudaErrorMemoryAllocation")
pub fn get_cuda_error_name(error_code: i32) -> String {
    let name_ptr = unsafe { cudaGetErrorName(error_code) };
    cstr_to_string(name_ptr)
}

/// Returns a descriptive error string (e.g. "out of memory")
pub fn get_cuda_error_string(error_code: i32) -> String {
    let str_ptr = unsafe { cudaGetErrorString(error_code) };
    cstr_to_string(str_ptr)
}

/// A CUDA error with code, name, and message
#[derive(Error, Debug)]
#[error("{message} ({name})")]
pub struct CudaError {
    pub code: i32,
    pub name: String,
    pub message: String,
}

impl CudaError {
    /// Construct from a raw CUDA error code (non-zero).
    pub fn new(code: i32) -> Self {
        CudaError {
            code,
            name: get_cuda_error_name(code),
            message: get_cuda_error_string(code),
        }
    }

    /// Returns `Ok(())` if `code == 0` (cudaSuccess), or `Err(CudaError)` if non-zero.
    pub fn from_result(code: i32) -> Result<(), Self> {
        if code == 0 {
            Ok(())
        } else {
            Err(Self::new(code))
        }
    }
}

#[inline]
pub fn check(code: i32) -> Result<(), CudaError> {
    CudaError::from_result(code)
}

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[error("Attempted to free null pointer")]
    NullPointer,

    #[error("Attempted to free untracked pointer")]
    UntrackedPointer,

    #[error("Failed to acquire memory manager lock")]
    LockError,

    #[error("Invalid memory size: {size}")]
    InvalidMemorySize { size: usize },

    #[error(
        "Out of memory in pool (size requested: {requested} bytes, available: {available} bytes)"
    )]
    OutOfMemory { requested: usize, available: usize },

    #[error("Memory manager not initialized")]
    NotInitialized,

    #[error("Invalid pointer: pointer not found in allocation table")]
    InvalidPointer,

    #[error("CUDA driver error: {code}")]
    CudaDriverError { code: i32 },
}

#[derive(Error, Debug)]
pub enum MemCopyError {
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error("Size mismatch in {operation}: host len={host_len}, device len={device_len}")]
    SizeMismatch {
        operation: &'static str,
        host_len: usize,
        device_len: usize,
    },
}

#[derive(Error, Debug)]
pub enum KernelError {
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[error("Unsupported type size {size}")]
    UnsupportedTypeSize { size: usize },
}
