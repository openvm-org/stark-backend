use std::{
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::Instant,
};

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

// Statistics tracking
pub struct FunctionStats {
    call_count: AtomicUsize,
    total_nanos: AtomicU64,
}

impl FunctionStats {
    pub const fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            total_nanos: AtomicU64::new(0),
        }
    }

    pub fn record_call(&self, duration_nanos: u64) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.total_nanos
            .fetch_add(duration_nanos, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> (usize, u64) {
        (
            self.call_count.load(Ordering::Relaxed),
            self.total_nanos.load(Ordering::Relaxed),
        )
    }
}

static STATS_CHECK_SUPPORT: FunctionStats = FunctionStats::new();
static STATS_MIN_GRANULARITY: FunctionStats = FunctionStats::new();
static STATS_RESERVE: FunctionStats = FunctionStats::new();
static STATS_RELEASE_VA: FunctionStats = FunctionStats::new();
static STATS_CREATE_PHYSICAL: FunctionStats = FunctionStats::new();
static STATS_MAP: FunctionStats = FunctionStats::new();
static STATS_SET_ACCESS: FunctionStats = FunctionStats::new();
static STATS_UNMAP: FunctionStats = FunctionStats::new();
static STATS_RELEASE: FunctionStats = FunctionStats::new();
pub static DEFRAG_CALLS: FunctionStats = FunctionStats::new();

/// Print statistics for all VPMM function calls
pub fn print_vpmm_stats() {
    let functions = [
        ("vpmm_check_support", &STATS_CHECK_SUPPORT),
        ("vpmm_min_granularity", &STATS_MIN_GRANULARITY),
        ("vpmm_reserve", &STATS_RESERVE),
        ("vpmm_release_va", &STATS_RELEASE_VA),
        ("vpmm_create_physical", &STATS_CREATE_PHYSICAL),
        ("vpmm_map", &STATS_MAP),
        ("vpmm_set_access", &STATS_SET_ACCESS),
        ("vpmm_unmap", &STATS_UNMAP),
        ("vpmm_release", &STATS_RELEASE),
        ("vpmm_defrag", &DEFRAG_CALLS),
    ];

    println!("=== VPMM Function Statistics ===");
    for (name, stats) in functions.iter() {
        let (count, nanos) = stats.get_stats();
        if count > 0 {
            let millis = nanos as f64 / 1_000_000.0;
            let avg_micros = (nanos / count as u64) as f64 / 1_000.0;
            println!(
                "{:25} calls: {:6}  total: {:8.2}ms  avg: {:6.2}µs",
                name, count, millis, avg_micros
            );
        }
    }
}

pub(super) unsafe fn vpmm_check_support(device_ordinal: i32) -> Result<bool, CudaError> {
    let start = Instant::now();
    let status = _vpmm_check_support(device_ordinal);
    STATS_CHECK_SUPPORT.record_call(start.elapsed().as_nanos() as u64);

    if status == 0 {
        Ok(true)
    } else {
        Err(CudaError::new(status))
    }
}

pub(super) unsafe fn vpmm_min_granularity(device_ordinal: i32) -> Result<usize, CudaError> {
    let start = Instant::now();
    let mut granularity: usize = 0;
    let result = CudaError::from_result(_vpmm_min_granularity(device_ordinal, &mut granularity));
    STATS_MIN_GRANULARITY.record_call(start.elapsed().as_nanos() as u64);

    result?;
    Ok(granularity)
}

pub(super) unsafe fn vpmm_reserve(size: usize, align: usize) -> Result<CUdeviceptr, CudaError> {
    let start = Instant::now();
    let mut va_base: CUdeviceptr = 0;
    let result = CudaError::from_result(_vpmm_reserve(size, align, &mut va_base));
    STATS_RESERVE.record_call(start.elapsed().as_nanos() as u64);

    result?;
    Ok(va_base)
}

pub(super) unsafe fn vpmm_release_va(base: CUdeviceptr, size: usize) -> Result<(), CudaError> {
    let start = Instant::now();
    let result = CudaError::from_result(_vpmm_release_va(base, size));
    STATS_RELEASE_VA.record_call(start.elapsed().as_nanos() as u64);
    result
}

pub(super) unsafe fn vpmm_create_physical(
    device_ordinal: i32,
    bytes: usize,
) -> Result<CUmemGenericAllocationHandle, CudaError> {
    let start = Instant::now();
    let mut h: CUmemGenericAllocationHandle = 0;
    let result = CudaError::from_result(_vpmm_create_physical(device_ordinal, bytes, &mut h));
    STATS_CREATE_PHYSICAL.record_call(start.elapsed().as_nanos() as u64);

    result?;
    Ok(h)
}

pub(super) unsafe fn vpmm_map(
    va: CUdeviceptr,
    bytes: usize,
    h: CUmemGenericAllocationHandle,
) -> Result<(), CudaError> {
    let start = Instant::now();
    let result = CudaError::from_result(_vpmm_map(va, bytes, h));
    STATS_MAP.record_call(start.elapsed().as_nanos() as u64);
    result
}

pub(super) unsafe fn vpmm_set_access(
    va: CUdeviceptr,
    bytes: usize,
    device_ordinal: i32,
) -> Result<(), CudaError> {
    let start = Instant::now();
    let result = CudaError::from_result(_vpmm_set_access(va, bytes, device_ordinal));
    STATS_SET_ACCESS.record_call(start.elapsed().as_nanos() as u64);
    result
}

pub(super) unsafe fn vpmm_unmap(va: CUdeviceptr, bytes: usize) -> Result<(), CudaError> {
    let start = Instant::now();
    let result = CudaError::from_result(_vpmm_unmap(va, bytes));
    STATS_UNMAP.record_call(start.elapsed().as_nanos() as u64);
    result
}

pub(super) unsafe fn vpmm_release(h: CUmemGenericAllocationHandle) -> Result<(), CudaError> {
    let start = Instant::now();
    let result = CudaError::from_result(_vpmm_release(h));
    STATS_RELEASE.record_call(start.elapsed().as_nanos() as u64);
    result
}
