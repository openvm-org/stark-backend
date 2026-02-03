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

// Benchmark kernels
#[link(name = "ext_field_bench")]
extern "C" {
    fn init_fp(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fp(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fp(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fp(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // BabyBear quartic extension (simple implementation)
    fn init_fp4(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fp4(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fp4(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fp4(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // BabyBear quartic extension (optimized bb31_4_t)
    fn init_fpext(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fpext(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fpext(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fpext(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    fn init_fp5(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fp5(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fp5(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fp5(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    fn init_fp6(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fp6(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fp6(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fp6(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    fn init_fp2x3(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fp2x3(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fp2x3(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fp2x3(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    fn init_fp3x2(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_fp3x2(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_fp3x2(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_fp3x2(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // KoalaBear base field
    fn init_kb(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_kb(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_kb(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_kb(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // KoalaBear quintic extension (x^5 + x + 4)
    fn init_kb5(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_kb5(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_kb5(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_kb5(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // KoalaBear sextic extension (x^6 + x^3 + 1)
    fn init_kb6(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_kb6(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_kb6(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_kb6(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // KoalaBear 2×3 tower (u²=3, v³=1+u)
    fn init_kb2x3(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_kb2x3(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_kb2x3(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_kb2x3(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // KoalaBear 3×2 tower (w³=-w-4, z²=3)
    fn init_kb3x2(out: *mut c_void, raw_data: *const u32, n: usize) -> i32;
    fn add_kb3x2(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_kb3x2(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_kb3x2(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // Goldilocks base field (64-bit prime: 2^64 - 2^32 + 1)
    fn init_gl(out: *mut c_void, raw_data: *const u64, n: usize) -> i32;
    fn add_gl(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_gl(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_gl(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // Goldilocks cubic extension (X³ - X - 1)
    fn init_gl3(out: *mut c_void, raw_data: *const u64, n: usize) -> i32;
    fn add_gl3(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn mul_gl3(out: *mut c_void, a: *const c_void, b: *const c_void, n: usize, reps: i32) -> i32;
    fn inv_gl3(out: *mut c_void, a: *const c_void, n: usize, reps: i32) -> i32;

    // Verification kernels
    fn verify_inv_fp4(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp4(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_fp5(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp5(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_fp6(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp6(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_fp2x3(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp2x3(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_fp3x2(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp3x2(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_kb(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_kb5(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb5(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_kb6(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb6(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_kb2x3(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb2x3(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;

    fn verify_inv_kb3x2(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb3x2(failures: *mut u32, a: *const c_void, b: *const c_void, c: *const c_void, n: usize) -> i32;
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
            num_elements: 1 << 22, // 4M elements
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

#[derive(Clone)]
pub struct FieldBenchResult {
    pub field_name: String,
    pub bits: usize,           // bits of provable security
    pub u32s_per_element: usize,
    pub init: OpResult,
    pub add: OpResult,
    pub mul: OpResult,
    pub inv: OpResult,
}

/// Print a table of benchmark results
fn print_benchmark_tables(results: &[FieldBenchResult], baseline: &FieldBenchResult) {
    // Find max field name length for alignment
    let max_name_len = results.iter().map(|r| r.field_name.len()).max().unwrap_or(10);
    
    // Table 1: Time (ms)
    println!("### Time (ms)");
    println!();
    print!("| {:width$} | bits |", "Field", width = max_name_len);
    println!("    init |     add |     mul |     inv |");
    print!("|{:-<width$}--|-----:|", "", width = max_name_len);
    println!("--------:|--------:|--------:|--------:|");
    for r in results {
        print!("| {:width$} | {:>4} |", r.field_name, r.bits, width = max_name_len);
        println!(" {:>7.3} | {:>7.3} | {:>7.3} | {:>7.3} |", 
            r.init.avg_time_ms, r.add.avg_time_ms, r.mul.avg_time_ms, r.inv.avg_time_ms);
    }
    println!();
    
    // Table 2: Throughput (Gops/s)
    println!("### Throughput (Gops/s)");
    println!();
    print!("| {:width$} | bits |", "Field", width = max_name_len);
    println!("    init |     add |     mul |     inv |");
    print!("|{:-<width$}--|-----:|", "", width = max_name_len);
    println!("--------:|--------:|--------:|--------:|");
    for r in results {
        print!("| {:width$} | {:>4} |", r.field_name, r.bits, width = max_name_len);
        println!(" {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} |", 
            r.init.throughput_gops, r.add.throughput_gops, r.mul.throughput_gops, r.inv.throughput_gops);
    }
    println!();
    
    // Table 3: Relative to baseline (xN)
    println!("### Relative to {} (xN)", baseline.field_name);
    println!();
    print!("| {:width$} | bits |", "Field", width = max_name_len);
    println!("    init |     add |     mul |     inv |");
    print!("|{:-<width$}--|-----:|", "", width = max_name_len);
    println!("--------:|--------:|--------:|--------:|");
    for r in results {
        let init_ratio = baseline.init.throughput_gops / r.init.throughput_gops;
        let add_ratio = baseline.add.throughput_gops / r.add.throughput_gops;
        let mul_ratio = baseline.mul.throughput_gops / r.mul.throughput_gops;
        let inv_ratio = baseline.inv.throughput_gops / r.inv.throughput_gops;
        
        print!("| {:width$} | {:>4} |", r.field_name, r.bits, width = max_name_len);
        println!(" {:>7.1} | {:>7.1} | {:>7.1} | {:>7.1} |", 
            init_ratio, add_ratio, mul_ratio, inv_ratio);
    }
    println!();
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

fn random_u64s(count: usize, seed: u64) -> Vec<u64> {
    let mut rng = seed;
    (0..count).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        rng
    }).collect()
}

fn measure<F: FnMut()>(config: &BenchConfig, total_ops: u64, mut f: F) -> OpResult {
    // Warmup
    for _ in 0..config.warmup_iters {
        f();
    }
    sync();
    
    // Timed
    let start = Instant::now();
    for _ in 0..config.bench_iters {
        f();
    }
    sync();
    let elapsed = start.elapsed();
    
    let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / config.bench_iters as f64;
    let throughput_gops = (total_ops * config.bench_iters as u64) as f64 / elapsed.as_secs_f64() / 1e9;
    
    OpResult { avg_time_ms, throughput_gops }
}

pub fn bench_fp(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Only 3 device buffers: a, b, out (init works in-place)
    let d_a = random_u32s(n, 12345).to_device().unwrap();
    let d_b = random_u32s(n, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n);
    
    // Benchmark init (in-place: raw u32 -> Fp) - also initializes d_a
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fp(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fp(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fp(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fp(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fp(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp".into(), bits: 31, u32s_per_element: 1, init, add, mul, inv }
}

pub fn bench_fp4(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    let d_a = random_u32s(n * 4, 11111).to_device().unwrap();
    let d_b = random_u32s(n * 4, 22222).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 4);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fp4(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fp4(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fp4(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fp4(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fp4(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp4".into(), bits: 124, u32s_per_element: 4, init, add, mul, inv }
}

pub fn bench_fpext(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Only 3 device buffers: a, b, out (init works in-place)
    let d_a = random_u32s(n * 4, 12345).to_device().unwrap();
    let d_b = random_u32s(n * 4, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 4);
    
    // Benchmark init (in-place: raw u32s -> FpExt) - also initializes d_a
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fpext(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fpext(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fpext(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fpext(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fpext(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "FpExt".into(), bits: 124, u32s_per_element: 4, init, add, mul, inv }
}

pub fn bench_fp5(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Only 3 device buffers: a, b, out (init works in-place)
    let d_a = random_u32s(n * 5, 12345).to_device().unwrap();
    let d_b = random_u32s(n * 5, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 5);
    
    // Benchmark init (in-place: raw u32s -> Fp5) - also initializes d_a
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fp5(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fp5(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fp5(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fp5(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fp5(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp5".into(), bits: 155, u32s_per_element: 5, init, add, mul, inv }
}

pub fn bench_fp6(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Only 3 device buffers: a, b, out (init works in-place)
    let d_a = random_u32s(n * 6, 12345).to_device().unwrap();
    let d_b = random_u32s(n * 6, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 6);
    
    // Benchmark init (in-place: raw u32s -> Fp6) - also initializes d_a
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fp6(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fp6(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fp6(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fp6(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fp6(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp6".into(), bits: 186, u32s_per_element: 6, init, add, mul, inv }
}

pub fn bench_fp2x3(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    let d_a = random_u32s(n * 6, 12345).to_device().unwrap();
    let d_b = random_u32s(n * 6, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 6);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fp2x3(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fp2x3(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fp2x3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fp2x3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fp2x3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp2x3".into(), bits: 186, u32s_per_element: 6, init, add, mul, inv }
}

pub fn bench_fp3x2(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    let d_a = random_u32s(n * 6, 12345).to_device().unwrap();
    let d_b = random_u32s(n * 6, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 6);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_fp3x2(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_fp3x2(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_fp3x2(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_fp3x2(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_fp3x2(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Fp3x2".into(), bits: 186, u32s_per_element: 6, init, add, mul, inv }
}

pub fn bench_kb(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Only 3 device buffers: a, b, out (init works in-place)
    let d_a = random_u32s(n, 12345).to_device().unwrap();
    let d_b = random_u32s(n, 67890).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n);
    
    // Benchmark init (in-place: raw u32 -> Kb) - also initializes d_a
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_kb(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_kb(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_kb(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_kb(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_kb(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Kb".into(), bits: 31, u32s_per_element: 1, init, add, mul, inv }
}

pub fn bench_kb5(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Kb5 has 5 u32s per element
    let d_a = random_u32s(n * 5, 11111).to_device().unwrap();
    let d_b = random_u32s(n * 5, 22222).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 5);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_kb5(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_kb5(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_kb5(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_kb5(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_kb5(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Kb5".into(), bits: 155, u32s_per_element: 5, init, add, mul, inv }
}

pub fn bench_kb6(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Kb6 has 6 u32s per element
    let d_a = random_u32s(n * 6, 33333).to_device().unwrap();
    let d_b = random_u32s(n * 6, 44444).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 6);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_kb6(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_kb6(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_kb6(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_kb6(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_kb6(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Kb6".into(), bits: 186, u32s_per_element: 6, init, add, mul, inv }
}

pub fn bench_kb2x3(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Kb2x3 has 6 u32s per element
    let d_a = random_u32s(n * 6, 55555).to_device().unwrap();
    let d_b = random_u32s(n * 6, 66666).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 6);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_kb2x3(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_kb2x3(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_kb2x3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_kb2x3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_kb2x3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Kb2x3".into(), bits: 186, u32s_per_element: 6, init, add, mul, inv }
}

pub fn bench_kb3x2(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Kb3x2 has 6 u32s per element
    let d_a = random_u32s(n * 6, 77777).to_device().unwrap();
    let d_b = random_u32s(n * 6, 88888).to_device().unwrap();
    let d_out = DeviceBuffer::<u32>::with_capacity(n * 6);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_kb3x2(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_kb3x2(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_kb3x2(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_kb3x2(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_kb3x2(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    FieldBenchResult { field_name: "Kb3x2".into(), bits: 186, u32s_per_element: 6, init, add, mul, inv }
}

pub fn bench_gl(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Goldilocks uses u64 elements (8 bytes)
    let h_a = random_u64s(n, 99999);
    let h_b = random_u64s(n, 88888);
    
    let d_a = h_a.to_device().unwrap();
    let d_b = h_b.to_device().unwrap();
    let d_out = DeviceBuffer::<u64>::with_capacity(n);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_gl(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_gl(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_gl(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_gl(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_gl(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    // u32s_per_element = 2 for Goldilocks (64-bit)
    FieldBenchResult { field_name: "Gl".into(), bits: 64, u32s_per_element: 2, init, add, mul, inv }
}

pub fn bench_gl3(config: &BenchConfig) -> FieldBenchResult {
    let n = config.num_elements;
    let reps = config.ops_per_element;
    
    // Gl3 uses 3 x u64 elements (24 bytes)
    let h_a = random_u64s(n * 3, 77777);
    let h_b = random_u64s(n * 3, 66666);
    
    let d_a = h_a.to_device().unwrap();
    let d_b = h_b.to_device().unwrap();
    let d_out = DeviceBuffer::<u64>::with_capacity(n * 3);
    
    let init = measure(config, n as u64, || {
        cuda_check(unsafe { init_gl3(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    });
    cuda_check(unsafe { init_gl3(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    sync();
    
    let ops = n as u64 * reps as u64;
    
    let add = measure(config, ops, || {
        cuda_check(unsafe { add_gl3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let mul = measure(config, ops, || {
        cuda_check(unsafe { mul_gl3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), d_b.as_raw_ptr(), n, reps) });
    });
    
    let inv = measure(config, ops, || {
        cuda_check(unsafe { inv_gl3(d_out.as_mut_raw_ptr(), d_a.as_raw_ptr(), n, reps) });
    });
    
    // u32s_per_element = 6 for Gl3 (3 x 64-bit = 24 bytes)
    FieldBenchResult { field_name: "Gl3".into(), bits: 192, u32s_per_element: 6, init, add, mul, inv }
}

pub fn run_all_benchmarks(config: &BenchConfig) {
    println!("=== Extension Field Benchmark ===");
    println!();
    println!("Elements: {}", config.num_elements);
    println!("Ops per element: {}", config.ops_per_element);
    println!("Warmup: {}, Bench iters: {}", config.warmup_iters, config.bench_iters);
    println!();
    
    // Collect all BabyBear results
    let fp = bench_fp(config);
    let fpext = bench_fpext(config);
    let fp5 = bench_fp5(config);
    let fp6 = bench_fp6(config);
    let fp2x3 = bench_fp2x3(config);
    let fp3x2 = bench_fp3x2(config);
    
    // Collect all KoalaBear results
    let kb = bench_kb(config);
    let kb5 = bench_kb5(config);
    let kb6 = bench_kb6(config);
    let kb2x3 = bench_kb2x3(config);
    let kb3x2 = bench_kb3x2(config);
    
    // Collect Goldilocks results
    let gl = bench_gl(config);
    let gl3 = bench_gl3(config);
    
    // Combine all results and sort by bits (then by name for consistent ordering)
    let mut all_results = vec![
        fp.clone(), fpext, fp5, fp6, fp2x3, fp3x2,
        kb, kb5, kb6, kb2x3, kb3x2,
        gl, gl3,
    ];
    all_results.sort_by(|a, b| {
        a.bits.cmp(&b.bits).then_with(|| a.field_name.cmp(&b.field_name))
    });
    print_benchmark_tables(&all_results, &fp);
}

// ============================================================================
// Verification
// ============================================================================

/// Generic verification function for extension fields
fn verify_field(
    field_name: &str,
    n: usize,
    elems_per_field: usize,
    init_fn: unsafe extern "C" fn(*mut c_void, *const u32, usize) -> i32,
    inv_fn: unsafe extern "C" fn(*mut u32, *const c_void, usize) -> i32,
    distrib_fn: unsafe extern "C" fn(*mut u32, *const c_void, *const c_void, *const c_void, usize) -> i32,
) -> bool {
    use crate::copy::MemCopyD2H;
    
    println!("  Testing {}...", field_name);
    
    // Create test data
    let d_a = random_u32s(n * elems_per_field, 11111).to_device().unwrap();
    let d_b = random_u32s(n * elems_per_field, 22222).to_device().unwrap();
    let d_c = random_u32s(n * elems_per_field, 33333).to_device().unwrap();
    
    // Initialize
    cuda_check(unsafe { init_fn(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    cuda_check(unsafe { init_fn(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    cuda_check(unsafe { init_fn(d_c.as_mut_raw_ptr(), d_c.as_ptr(), n) });
    sync();
    
    // Test 1: a * inv(a) = 1
    let d_failures = vec![0u32].to_device().unwrap();
    cuda_check(unsafe { inv_fn(d_failures.as_mut_raw_ptr() as *mut u32, d_a.as_raw_ptr(), n) });
    sync();
    let inv_failures = d_failures.to_host().unwrap();
    
    // Test 2: Distributivity
    let d_failures2 = vec![0u32].to_device().unwrap();
    cuda_check(unsafe { distrib_fn(
        d_failures2.as_mut_raw_ptr() as *mut u32,
        d_a.as_raw_ptr(), d_b.as_raw_ptr(), d_c.as_raw_ptr(), n
    ) });
    sync();
    let distrib_failures = d_failures2.to_host().unwrap();
    
    let passed = inv_failures[0] == 0 && distrib_failures[0] == 0;
    println!("    Inversion: {} failures, Distributivity: {} failures -> {}",
             inv_failures[0], distrib_failures[0], if passed { "PASSED" } else { "FAILED" });
    
    passed
}

/// Verify all extension field implementations
pub fn verify_all_fields(num_elements: usize) -> bool {
    println!("=== Extension Field Verification ===");
    println!("Testing {} elements per field...", num_elements);
    println!();
    
    let mut all_passed = true;
    
    // Use existing benchmark init functions for initialization
    all_passed &= verify_field("Fp4 (simple)", num_elements, 4,
        init_fp4, verify_inv_fp4, verify_distrib_fp4);
    
    all_passed &= verify_field("Fp5", num_elements, 5,
        init_fp5, verify_inv_fp5, verify_distrib_fp5);
    
    all_passed &= verify_field("Fp6", num_elements, 6,
        init_fp6, verify_inv_fp6, verify_distrib_fp6);
    
    all_passed &= verify_field("Fp2x3 (2×3 tower)", num_elements, 6,
        init_fp2x3, verify_inv_fp2x3, verify_distrib_fp2x3);
    
    all_passed &= verify_field("Fp3x2 (3×2 tower)", num_elements, 6,
        init_fp3x2, verify_inv_fp3x2, verify_distrib_fp3x2);
    
    println!();
    println!("=== KoalaBear Fields ===");
    println!();
    
    all_passed &= verify_field("Kb (KoalaBear base)", num_elements, 1,
        init_kb, verify_inv_kb, verify_distrib_kb);
    
    all_passed &= verify_field("Kb5 (KoalaBear quintic)", num_elements, 5,
        init_kb5, verify_inv_kb5, verify_distrib_kb5);
    
    all_passed &= verify_field("Kb6 (KoalaBear sextic)", num_elements, 6,
        init_kb6, verify_inv_kb6, verify_distrib_kb6);
    
    all_passed &= verify_field("Kb2x3 (KoalaBear 2×3 tower)", num_elements, 6,
        init_kb2x3, verify_inv_kb2x3, verify_distrib_kb2x3);
    
    all_passed &= verify_field("Kb3x2 (KoalaBear 3×2 tower)", num_elements, 6,
        init_kb3x2, verify_inv_kb3x2, verify_distrib_kb3x2);
    
    println!();
    println!("Overall: {}", if all_passed { "ALL PASSED" } else { "SOME FAILED" });
    
    all_passed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_field_verification() {
        assert!(verify_all_fields(1024), "Extension field verification failed!");
    }
    
    #[test]
    fn test_extension_field_benchmark() {
        run_all_benchmarks(&BenchConfig::default());
    }
}
