//! High-level convenience wrapper around [`KernelModule`] that owns the GPU
//! stream and one device buffer per input / output, so tests and benchmarks
//! can just push host `Vec<u32>` data through the pipeline.

use std::time::Instant;

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};

use crate::{
    compile_and_load,
    ir::{Module, ScalarType},
    runtime::{CompileOptions, KernelModule},
    CompileError,
};

/// BabyBear prime.
const BB_P: u64 = 2_013_265_921;

/// Convert canonical BabyBear `u32` `x` in `[0, P)` to Montgomery form
/// (`x * R mod P`, `R = 2^32`). Used to encode BabyBear inputs at the DSL
/// boundary; the emitted CUDA operates entirely in Montgomery form.
pub fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % BB_P) as u32
}

/// Convert Montgomery-form BabyBear `u32` back to canonical `[0, P)`. This
/// is the tightest possible one-limb Montgomery reduction — the same
/// `mul_by_1` idiom used by sppark's `mont32_t::operator uint32_t()`.
pub fn from_monty(x: u32) -> u32 {
    const P: u64 = BB_P;
    // t = x + ((x * M0) mod 2^32) * P; canonical value is t / R.
    const M0: u64 = 0x77ff_ffff;
    let red = (x as u64).wrapping_mul(M0) & 0xffff_ffff;
    let t = x as u64 + red * P;
    let mut out = (t >> 32) as u32;
    if out as u64 >= P {
        out = (out as u64 - P) as u32;
    }
    out
}

/// Per-iteration timing summary of a benchmarked kernel, in milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct KernelBenchTime {
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
}

/// Owns a JIT-compiled [`KernelModule`], its [`GpuDeviceCtx`], and one
/// pre-allocated device buffer per input and output. Buffers are allocated
/// once at construction and reused by [`Self::set_inputs`] /
/// [`Self::read_outputs`].
pub struct ModuleRunner {
    module: KernelModule,
    name: String,
    ctx: GpuDeviceCtx,
    in_bufs: Vec<DeviceBuffer<u32>>,
    out_bufs: Vec<DeviceBuffer<u32>>,
}

impl ModuleRunner {
    /// JIT-compiles `module`, allocates and binds device buffers for every
    /// input and output, and reserves the module's scratch. The runner is
    /// ready to `set_inputs` + `run` after this call.
    ///
    /// Modules with runtime parameters defer buffer allocation until every
    /// parameter is bound via [`Self::set_symbol`] (sizes are functions of
    /// the parameters).
    pub fn new(module: &Module, options: &CompileOptions) -> Result<Self, CompileError> {
        let ctx = GpuDeviceCtx::for_current_device()
            .map_err(|e| CompileError::Runtime(format!("GpuDeviceCtx: {e:?}")))?;
        let km = compile_and_load(module, options)?;

        let mut runner = Self {
            module: km,
            name: module.name.clone(),
            ctx,
            in_bufs: Vec::new(),
            out_bufs: Vec::new(),
        };
        if runner.module.params().is_empty() {
            runner.bind_buffers()?;
        }
        Ok(runner)
    }

    /// (Re)allocates one device buffer per input / output at the module's
    /// current sizes, binds them and the scratch through the C ABI.
    fn bind_buffers(&mut self) -> Result<(), CompileError> {
        let km = &mut self.module;
        self.in_bufs = (0..km.num_inputs())
            .map(|i| DeviceBuffer::with_capacity_on(km.input_size(i) / 4, &self.ctx))
            .collect();
        for (i, buf) in self.in_bufs.iter().enumerate() {
            km.set_input(i, buf)?;
        }
        self.out_bufs = (0..km.num_outputs())
            .map(|i| DeviceBuffer::with_capacity_on(km.output_size(i) / 4, &self.ctx))
            .collect();
        for (i, buf) in self.out_bufs.iter().enumerate() {
            km.set_output(i, buf)?;
        }
        km.ensure_scratch(&self.ctx);
        Ok(())
    }

    /// Binds the module's runtime parameter `name` to `value`. Once every
    /// parameter is bound, the input / output device buffers are
    /// (re)allocated at the implied sizes.
    pub fn set_symbol(&mut self, name: &str, value: i64) -> Result<(), CompileError> {
        let i = self
            .module
            .params()
            .iter()
            .position(|p| p == name)
            .ok_or_else(|| {
                CompileError::Runtime(format!(
                    "module `{}` has no runtime parameter `{name}`",
                    self.name
                ))
            })?;
        self.module.set_param(i, value);
        if self.module.params_bound() {
            self.bind_buffers()?;
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_inputs(&self) -> usize {
        self.in_bufs.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.out_bufs.len()
    }

    /// Copies host input tensors into the pre-allocated device buffers on
    /// the runner's stream. Every inner buffer's length (in `u32`s) must
    /// match the module's declared input size.
    ///
    /// Callers pass **canonical** BabyBear (and FpExt) `u32`s in `[0, P)`.
    /// The runner Montgomery-encodes them on the fly to match the emitted
    /// kernel's on-device representation; `U32` and `Bool` inputs pass
    /// through unchanged.
    pub fn set_inputs(&mut self, inputs: &[Vec<u32>]) {
        assert_eq!(
            inputs.len(),
            self.in_bufs.len(),
            "expected {} inputs, got {}",
            self.in_bufs.len(),
            inputs.len()
        );
        for (i, data) in inputs.iter().enumerate() {
            assert_eq!(
                data.len(),
                self.in_bufs[i].len(),
                "input {i} length {} does not match device buffer length {}",
                data.len(),
                self.in_bufs[i].len()
            );
            let ty = self.module.input_type(i);
            // BabyBear (and FpExt whose four coefficients are BabyBears)
            // uses Montgomery on device — convert per-limb before H2D.
            if matches!(ty, ScalarType::BabyBear | ScalarType::FpExt) {
                let mont: Vec<u32> = data.iter().map(|&v| to_monty(v)).collect();
                mont.as_slice()
                    .copy_to_on(&mut self.in_bufs[i], &self.ctx)
                    .expect("H2D copy");
            } else {
                data.as_slice()
                    .copy_to_on(&mut self.in_bufs[i], &self.ctx)
                    .expect("H2D copy");
            }
        }
    }

    /// Launches the kernel sequence asynchronously on the runner's stream.
    pub fn run(&mut self) {
        self.module.run(&self.ctx.stream).expect("kernel run");
    }

    /// Blocks until every kernel previously launched via [`Self::run`] on the
    /// runner's stream has finished. Useful when callers do their own timing
    /// or profiling loop and don't want to go through [`Self::read_outputs`].
    pub fn sync(&self) {
        self.ctx.stream.synchronize().expect("stream sync");
    }

    /// Copies every output buffer back to host, synchronizing the stream in
    /// the process (via `MemCopyD2H::to_host_on`).
    ///
    /// BabyBear (and FpExt) outputs are Montgomery-encoded on device;
    /// this method decodes them back to canonical `u32` in `[0, P)` before
    /// returning, so callers can compare directly against p3 references.
    pub fn read_outputs(&self) -> Vec<Vec<u32>> {
        self.out_bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let raw: Vec<u32> = b.to_host_on(&self.ctx).expect("D2H copy");
                let ty = self.module.output_type(i);
                if matches!(ty, ScalarType::BabyBear | ScalarType::FpExt) {
                    raw.into_iter().map(from_monty).collect()
                } else {
                    raw
                }
            })
            .collect()
    }

    /// Runs the kernel `n_warmup + n_iter` times and returns the median,
    /// q25 and q75 per-launch times in milliseconds. Each timed iteration
    /// syncs the stream so its measurement isolates a single launch.
    pub fn bench(&mut self, n_warmup: usize, n_iter: usize) -> KernelBenchTime {
        assert!(n_iter > 0, "bench: n_iter must be > 0");
        for _ in 0..n_warmup {
            self.module.run(&self.ctx.stream).expect("warmup run");
        }
        self.ctx.stream.synchronize().expect("warmup sync");

        let mut samples: Vec<f64> = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let t0 = Instant::now();
            self.module.run(&self.ctx.stream).expect("bench run");
            self.ctx.stream.synchronize().expect("bench sync");
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pick = |q: f64| {
            let idx = ((n_iter as f64 * q) as usize).min(n_iter - 1);
            samples[idx]
        };
        KernelBenchTime {
            median: pick(0.5),
            q25: pick(0.25),
            q75: pick(0.75),
        }
    }
}

/// If the `BENCH_KERNEL` environment variable is set, benchmarks `runner`
/// and prints its median / q25 / q75 launch time. No-op otherwise.
///
/// `BENCH_KERNEL_WARMUP` and `BENCH_KERNEL_ITERS` override the default warmup
/// (5) and iteration (50) counts.
pub fn maybe_bench(runner: &mut ModuleRunner) {
    if std::env::var_os("BENCH_KERNEL").is_none() {
        return;
    }
    let warmup: usize = std::env::var("BENCH_KERNEL_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let iters: usize = std::env::var("BENCH_KERNEL_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let t = runner.bench(warmup, iters);
    println!(
        "[bench] {name}: median={:.4} ms  q25={:.4} ms  q75={:.4} ms  (warmup={warmup}, iters={iters})",
        t.median,
        t.q25,
        t.q75,
        name = runner.name(),
    );
}
