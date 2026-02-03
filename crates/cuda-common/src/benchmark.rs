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

    // Verification kernels
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
    
    FieldBenchResult { field_name: "Fp".into(), u32s_per_element: 1, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "FpExt".into(), u32s_per_element: 4, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "Fp5".into(), u32s_per_element: 5, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "Fp6".into(), u32s_per_element: 6, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "Fp2x3".into(), u32s_per_element: 6, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "Fp3x2".into(), u32s_per_element: 6, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "Kb".into(), u32s_per_element: 1, init, add, mul, inv }
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
    
    FieldBenchResult { field_name: "Kb5".into(), u32s_per_element: 5, init, add, mul, inv }
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
    println!();
    
    let fp5 = bench_fp5(config);
    fp5.print(Some(&fp));
    println!();
    
    let fp6 = bench_fp6(config);
    fp6.print(Some(&fp));
    println!();
    
    let fp2x3 = bench_fp2x3(config);
    fp2x3.print(Some(&fp));
    println!();
    
    let fp3x2 = bench_fp3x2(config);
    fp3x2.print(Some(&fp));
    println!();
    
    println!("=== KoalaBear Fields ===");
    println!();
    
    let kb = bench_kb(config);
    kb.print(Some(&fp));
    println!();
    
    let kb5 = bench_kb5(config);
    kb5.print(Some(&fp));
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
