//! H2D / D2H PCIe bandwidth benchmark.
//!
//! Copies a 2 GiB buffer between pinned host memory and device memory using
//! `cudaMemcpyAsync` on a non-blocking stream, and reports bandwidth measured
//! with CUDA events (GPU-side timing).

use std::{ffi::c_void, ptr};

use openvm_cuda_common::{
    copy::{cudaMemcpyKind, cuda_memcpy_on},
    d_buffer::DeviceBuffer,
    stream::{cudaEvent_t, cudaStream_t, GpuDeviceCtx},
};

#[link(name = "cudart")]
extern "C" {
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> i32;
    fn cudaEventCreate(event: *mut cudaEvent_t) -> i32;
    fn cudaEventDestroy(event: cudaEvent_t) -> i32;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> i32;
    fn cudaEventSynchronize(event: cudaEvent_t) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> i32;
}

const CUDA_HOST_ALLOC_DEFAULT: u32 = 0;

struct PinnedHostBuffer {
    ptr: *mut u8,
}

impl PinnedHostBuffer {
    fn new(len: usize) -> Self {
        let mut ptr: *mut c_void = ptr::null_mut();
        let rc = unsafe { cudaHostAlloc(&mut ptr, len, CUDA_HOST_ALLOC_DEFAULT) };
        assert_eq!(rc, 0, "cudaHostAlloc failed: {rc}");
        Self {
            ptr: ptr as *mut u8,
        }
    }

    fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }

    fn as_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let rc = unsafe { cudaFreeHost(self.ptr as *mut c_void) };
            debug_assert_eq!(rc, 0, "cudaFreeHost failed: {rc}");
        }
    }
}

struct Event(cudaEvent_t);

impl Event {
    fn new() -> Self {
        let mut e: cudaEvent_t = ptr::null_mut();
        let rc = unsafe { cudaEventCreate(&mut e) };
        assert_eq!(rc, 0, "cudaEventCreate failed: {rc}");
        Self(e)
    }

    fn record(&self, stream: cudaStream_t) {
        let rc = unsafe { cudaEventRecord(self.0, stream) };
        assert_eq!(rc, 0, "cudaEventRecord failed: {rc}");
    }

    fn synchronize(&self) {
        let rc = unsafe { cudaEventSynchronize(self.0) };
        assert_eq!(rc, 0, "cudaEventSynchronize failed: {rc}");
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        let rc = unsafe { cudaEventDestroy(self.0) };
        debug_assert_eq!(rc, 0, "cudaEventDestroy failed: {rc}");
    }
}

fn elapsed_ms(start: &Event, stop: &Event) -> f32 {
    let mut ms: f32 = 0.0;
    let rc = unsafe { cudaEventElapsedTime(&mut ms, start.0, stop.0) };
    assert_eq!(rc, 0, "cudaEventElapsedTime failed: {rc}");
    ms
}

/// Times `iters` async memcpys of `size_bytes` and returns (avg_ms, gb_per_s, gib_per_s).
fn bench_copy(
    kind: cudaMemcpyKind,
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    stream: cudaStream_t,
    warmup: usize,
    iters: usize,
) -> (f32, f64, f64) {
    // Warmup.
    for _ in 0..warmup {
        let rc = unsafe { cudaMemcpyAsync(dst, src, size_bytes, kind, stream) };
        assert_eq!(rc, 0, "cudaMemcpyAsync (warmup) failed: {rc}");
    }
    let start = Event::new();
    let stop = Event::new();
    start.record(stream);
    for _ in 0..iters {
        let rc = unsafe { cudaMemcpyAsync(dst, src, size_bytes, kind, stream) };
        assert_eq!(rc, 0, "cudaMemcpyAsync failed: {rc}");
    }
    stop.record(stream);
    stop.synchronize();

    let total_ms = elapsed_ms(&start, &stop);
    let avg_ms = total_ms / iters as f32;
    let seconds_per_copy = (avg_ms as f64) / 1e3;
    let gb_per_s = (size_bytes as f64) / seconds_per_copy / 1e9;
    let gib_per_s = (size_bytes as f64) / seconds_per_copy / (1u64 << 30) as f64;
    (avg_ms, gb_per_s, gib_per_s)
}

fn main() {
    openvm_cuda_common::common::set_device().expect("set CUDA device");
    let ctx = GpuDeviceCtx::for_current_device().expect("device context");
    let stream = ctx.stream.as_raw();

    const SIZE_BYTES: usize = 2 * (1usize << 30); // 2 GiB
    const WARMUP: usize = 3;
    const ITERS: usize = 10;

    println!("=== H2D / D2H Bandwidth Benchmark ===");
    println!("Buffer size: {} bytes (2 GiB)", SIZE_BYTES);
    println!("Warmup iters: {WARMUP}, timed iters: {ITERS}");
    println!();

    // Pinned host buffer + device buffer, both 2 GiB.
    let host = PinnedHostBuffer::new(SIZE_BYTES);
    let device = DeviceBuffer::<u8>::with_capacity_on(SIZE_BYTES, &ctx);

    // Prime the device buffer once so the first D2H doesn't read uninitialized memory.
    unsafe {
        cuda_memcpy_on::<false, true>(device.as_mut_raw_ptr(), host.as_ptr(), SIZE_BYTES, &ctx)
            .expect("prime device buffer");
    }
    ctx.stream.synchronize().expect("sync");

    let (h2d_ms, h2d_gb, h2d_gib) = bench_copy(
        cudaMemcpyKind::cudaMemcpyHostToDevice,
        device.as_mut_raw_ptr(),
        host.as_ptr(),
        SIZE_BYTES,
        stream,
        WARMUP,
        ITERS,
    );
    let (d2h_ms, d2h_gb, d2h_gib) = bench_copy(
        cudaMemcpyKind::cudaMemcpyDeviceToHost,
        host.as_mut_ptr(),
        device.as_raw_ptr(),
        SIZE_BYTES,
        stream,
        WARMUP,
        ITERS,
    );

    println!(
        "| {:>10} | {:>12} | {:>14} | {:>14} |",
        "Direction", "Avg (ms)", "GB/s (10^9)", "GiB/s (2^30)"
    );
    println!("|{:-<12}|{:-<14}|{:-<16}|{:-<16}|", "", "", "", "");
    println!(
        "| {:>10} | {:>12.3} | {:>14.2} | {:>14.2} |",
        "H2D", h2d_ms, h2d_gb, h2d_gib
    );
    println!(
        "| {:>10} | {:>12.3} | {:>14.2} | {:>14.2} |",
        "D2H", d2h_ms, d2h_gb, d2h_gib
    );
}
