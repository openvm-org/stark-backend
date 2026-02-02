//! Extension Field Benchmark
//!
//! Benchmarks for field arithmetic operations on GPU.
//! Measures compute throughput by doing many ops per element to amortize memory access.
//!
//! Throughput is reported in Gops/s (Giga-operations per second = billion ops/sec).

use std::ffi::c_void;
use std::time::Instant;

use crate::d_buffer::DeviceBuffer;
use crate::copy::MemCopyH2D;
use crate::stream::current_stream_sync;

// ============================================================================
// FFI Bindings
// ============================================================================

#[link(name = "ext_field_bench")]
extern "C" {
    fn launch_bench_init_fp(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn launch_bench_add_fp(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn launch_bench_mul_fp(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn launch_bench_inv_fp(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    fn launch_bench_init_fpext(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn launch_bench_add_fpext(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn launch_bench_mul_fpext(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn launch_bench_inv_fpext(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;
}

/// Check CUDA return code, panic on error
fn cuda_check(code: i32) {
    assert!(code == 0, "CUDA error: {}", code);
}

/// Sync and check
fn sync() {
    current_stream_sync().expect("sync failed");
}

// ============================================================================
// Configuration & Results
// ============================================================================

pub struct BenchConfig {
    pub num_elements: usize,
    pub warmup_iters: usize,
    pub bench_iters: usize,
    pub ops_per_element: i32,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            num_elements: 1 << 25,
            warmup_iters: 3,
            bench_iters: 10,
            ops_per_element: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpResult {
    pub avg_time_ms: f64,
    pub throughput_gops: f64,
}

pub struct FieldBenchResult {
    pub field_name: String,
    pub u32s_per_element: usize,
    pub init: OpResult,
    pub add: OpResult,
    pub mul: OpResult,
    pub inv: OpResult,
}

impl FieldBenchResult {
    pub fn print(&self, baseline: Option<&FieldBenchResult>) {
        println!("--- {} ({} u32s/elem) ---", self.field_name, self.u32s_per_element);
        
        let print_op = |name: &str, op: &OpResult, base_op: Option<&OpResult>| {
            let delta = match base_op {
                Some(base) if base.throughput_gops > 0.0 => {
                    let ratio = base.throughput_gops / op.throughput_gops;
                    if ratio >= 1.0 {
                        format!(" (x{:.1} slower)", ratio)
                    } else {
                        format!(" (x{:.1} faster)", 1.0 / ratio)
                    }
                }
                _ => String::new(),
            };
            println!("  {:4}: {:>8.3} ms, {:>8.2} Gops/s{}", name, op.avg_time_ms, op.throughput_gops, delta);
        };
        
        print_op("init", &self.init, baseline.map(|b| &b.init));
        print_op("add", &self.add, baseline.map(|b| &b.add));
        print_op("mul", &self.mul, baseline.map(|b| &b.mul));
        print_op("inv", &self.inv, baseline.map(|b| &b.inv));
    }
}

// ============================================================================
// Benchmark Implementation
// ============================================================================

fn random_u32s(count: usize, seed: u64) -> Vec<u32> {
    let mut rng = seed;
    (0..count).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 32) as u32
    }).collect()
}

fn measure<F: FnMut()>(config: &BenchConfig, total_ops: u64, mut f: F) -> OpResult {
    // Warmup
    for _ in 0..config.warmup_iters {
        f();
        sync();
    }
    
    // Timed
    let start = Instant::now();
    for _ in 0..config.bench_iters {
        f();
        sync();
    }
    let elapsed = start.elapsed();
    
    let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / config.bench_iters as f64;
    let throughput_gops = (total_ops * config.bench_iters as u64) as f64 / elapsed.as_secs_f64() / 1e9;
    
    OpResult { avg_time_ms, throughput_gops }
}

pub fn bench_fp(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    let raw_a = random_u32s(n, 12345);
    let raw_b = random_u32s(n, 67890);
    let d_raw_a = raw_a.to_device().unwrap();
    let d_raw_b = raw_b.to_device().unwrap();
    let d_a = DeviceBuffer::<u32>::with_capacity(n);
    let d_b = DeviceBuffer::<u32>::with_capacity(n);
    let d_out = DeviceBuffer::<u32>::with_capacity(n);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { launch_bench_init_fp(d_a.as_mut_raw_ptr(), d_raw_a.as_ptr(), n) });
    });
    
    // Initialize both buffers
    cuda_check(unsafe { launch_bench_init_fp(d_a.as_mut_raw_ptr(), d_raw_a.as_ptr(), n) });
    cuda_check(unsafe { launch_bench_init_fp(d_b.as_mut_raw_ptr(), d_raw_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { launch_bench_add_fp(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { launch_bench_mul_fp(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { launch_bench_inv_fp(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp".into(), u32s_per_element: 1, init, add, mul, inv }
}

pub fn bench_fpext(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    let raw_a = random_u32s(n * 4, 12345);
    let raw_b = random_u32s(n * 4, 67890);
    let d_raw_a = raw_a.to_device().unwrap();
    let d_raw_b = raw_b.to_device().unwrap();
    let d_a = DeviceBuffer::<u32>::with_capacity(n * 4);
    let d_b = DeviceBuffer::<u32>::with_capacity(n * 4);
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 4);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { launch_bench_init_fpext(d_a.as_mut_raw_ptr(), d_raw_a.as_ptr(), n) });
    });
    
    cuda_check(unsafe { launch_bench_init_fpext(d_a.as_mut_raw_ptr(), d_raw_a.as_ptr(), n) });
    cuda_check(unsafe { launch_bench_init_fpext(d_b.as_mut_raw_ptr(), d_raw_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { launch_bench_add_fpext(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { launch_bench_mul_fpext(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { launch_bench_inv_fpext(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "FpExt".into(), u32s_per_element: 4, init, add, mul, inv }
}

pub fn run_all_benchmarks(config: &BenchConfig) {
    println!("=== Extension Field Benchmark ===");
    println!("Elements: {}", config.num_elements);
    println!("Ops per element: {} (to measure compute, not memory bandwidth)", config.ops_per_element);
    println!("Warmup: {}, Bench iters: {}", config.warmup_iters, config.bench_iters);
    println!();
    println!("Throughput in Gops/s = billion operations per second");
    println!("Delta shows comparison to Fp baseline (slower/faster)");
    println!();
    
    let fp = bench_fp(config);
    fp.print(None);
    println!();
    
    let fpext = bench_fpext(config);
    fpext.print(Some(&fp));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_field_benchmark() {
        run_all_benchmarks(&BenchConfig::default());
    }
}
