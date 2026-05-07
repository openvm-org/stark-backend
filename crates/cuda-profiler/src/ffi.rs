//! FFI to the CUDA host glue (`shadow_ctx.cu`) and to the cudart symbols we
//! need for pinned-mapped allocation.
//!
//! The `shadow_*` symbols are defined in this crate's own static library
//! (`openvm_cuda_profiler_glue`); the `cuda*` symbols come from `libcudart`
//! which the cuda-builder already links.

use std::ffi::c_void;

/// Mirrors `CtaProbeCtx` in `cta_probe.cuh`. Layout must match exactly.
/// 24 bytes: 8 (pointer) + 4 (mask) + 4 (pad) + 8 (head pointer).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CtaProbeCtx {
    pub ring: *mut CtaRecord,
    pub mask: u32,
    pub _pad: u32,
    pub head: *mut u64,
}

// SAFETY: these pointers are only meaningful from the GPU; we don't ever
// dereference them on the host. Sending the struct between threads is fine.
unsafe impl Send for CtaProbeCtx {}
unsafe impl Sync for CtaProbeCtx {}

impl CtaProbeCtx {
    pub const fn null() -> Self {
        Self {
            ring: std::ptr::null_mut(),
            mask: 0,
            _pad: 0,
            head: std::ptr::null_mut(),
        }
    }
}

/// Mirrors `CtaRecord` in `cta_probe.cuh`. 32 bytes (4×u32 + 2×u64 with the
/// natural 8-byte alignment for u64 fields).
///
/// `seq_tag` is the producer's completion marker — the low 32 bits of
/// `(slot + 1)`. The host drain reads it last and only consumes a record
/// whose `seq_tag` matches the expected `((slot + 1) & 0xFFFFFFFF)` for the
/// slot it is reading; this rejects stale records left over from a previous
/// ring wrap and torn slots that haven't yet been flushed by the producer's
/// `__threadfence_system()`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct CtaRecord {
    pub kernel_id: u32,
    pub smid: u32,
    pub block_linear: u32,
    pub seq_tag: u32,
    pub t_start: u64,
    pub t_end: u64,
}

const _: () = {
    // Compile-time layout asserts. These must agree with the static_asserts
    // in `cta_probe.cuh` — drift here is silent corruption of the wire format.
    assert!(std::mem::size_of::<CtaProbeCtx>() == 24);
    assert!(std::mem::align_of::<CtaProbeCtx>() == 8);
    assert!(std::mem::size_of::<CtaRecord>() == 32);
    assert!(std::mem::align_of::<CtaRecord>() == 8);
};

#[link(name = "openvm_cuda_profiler_glue")]
extern "C" {
    pub fn shadow_cta_ctx() -> CtaProbeCtx;
    pub fn shadow_set_cta_ctx(ctx: CtaProbeCtx);
    pub fn shadow_clear_cta_ctx();
}

// ---- cudart bindings we need for pinned-mapped allocation ---------------

/// `cudaHostAllocMapped`. We don't use the other flags (Portable would be for
/// multi-device sharing; WriteCombined is for host-write/device-read).
pub const CUDA_HOST_ALLOC_MAPPED: u32 = 0x02;

#[link(name = "cudart")]
extern "C" {
    pub fn cudaHostAlloc(p_host: *mut *mut c_void, size: usize, flags: u32) -> i32;
    pub fn cudaFreeHost(host_ptr: *mut c_void) -> i32;
    pub fn cudaHostGetDevicePointer(
        p_device: *mut *mut c_void,
        p_host: *mut c_void,
        flags: u32,
    ) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    /// `cudaGetDeviceProperties(prop, device)` — fills a `cudaDeviceProp`
    /// struct whose first field is `char name[256]`. We only ever read that
    /// prefix, so we pass a generously-sized zero-initialized buffer.
    pub fn cudaGetDeviceProperties(prop: *mut std::os::raw::c_void, device: i32) -> i32;
}

/// Best-effort GPU name lookup. Returns `None` on FFI failure.
pub fn device_name(device: i32) -> Option<String> {
    // `cudaDeviceProp` is ~3 KB on recent CUDA runtimes; over-allocate to be
    // safe across versions. We only read the first 256 bytes (the `name`
    // field, which is the struct's leading member).
    let mut buf = vec![0u8; 8192];
    let rc =
        unsafe { cudaGetDeviceProperties(buf.as_mut_ptr() as *mut std::os::raw::c_void, device) };
    if rc != 0 {
        return None;
    }
    let nul = buf.iter().take(256).position(|&b| b == 0).unwrap_or(256);
    std::str::from_utf8(&buf[..nul]).ok().map(|s| s.to_string())
}
