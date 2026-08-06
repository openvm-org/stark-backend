//! Test conveniences over [`GraphModule`] / [`GraphCompiler`] / [`GraphExe`]
//! that hide the JIT boilerplate for numerical accuracy tests and
//! micro-benches.
//!
//! Tests build a [`GraphModule`] (usually wrapping a
//! single-kernel HIR module via `GraphModule::from_ir`) and feed it to
//! [`TestModuleRunner`], which owns the compiled [`GraphExe`], allocates
//! device buffers for the registered graph interface, and handles the
//! Montgomery encode/decode dance on the host boundary.
//!
//! Not gated behind a feature — the code paths it exercises are core
//! enough that keeping it always-buildable simplifies CI.
//!
//! **Two modes.** The default (pool-mode) drives the compiled
//! [`GraphExe`] through its own device pool: [`Self::set_inputs`] does
//! the H2D copy into the pool, [`Self::run`] launches, and
//! [`Self::read_outputs`] downloads from the pool. Calling
//! [`Self::set_symbol`] switches the runner to **direct-drive** mode:
//! subsequent [`Self::set_inputs`] / [`Self::run`] / [`Self::read_outputs`]
//! bypass the pool and drive the underlying [`KernelProgram`] directly,
//! with per-invocation device buffers sized off the kernel's current
//! `input_size(i)` / `output_size(i)` (which reflect the newly-bound
//! symbol). One nvcc + one dlopen serves many sizes.

use std::time::Instant;

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};

use crate::{
    graph_exe::{GraphCompiler, GraphExe},
    graph_ir::GraphModule,
    ir::ScalarType,
    CompileError,
};

/// BabyBear prime.
const BB_P: u64 = 2_013_265_921;

/// Convert canonical BabyBear `u32` `x` in `[0, P)` to Montgomery form
/// (`x * R mod P`, `R = 2^32`).
pub fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % BB_P) as u32
}

/// Convert Montgomery-form BabyBear `u32` back to canonical `[0, P)`.
pub fn from_monty(x: u32) -> u32 {
    const P: u64 = BB_P;
    const M0: u64 = 0x77ff_ffff;
    let red = (x as u64).wrapping_mul(M0) & 0xffff_ffff;
    let t = x as u64 + red * P;
    let mut out = (t >> 32) as u32;
    if out as u64 >= P {
        out = (out as u64 - P) as u32;
    }
    out
}

/// Reinterprets a `Vec<u32>` as raw bytes for H2D copy without allocating.
fn u32_slice_as_bytes(xs: &[u32]) -> &[u8] {
    // SAFETY: u32 has stricter alignment than u8; length is scaled correctly.
    unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs)) }
}

/// Montgomery-encodes BabyBear / FpExt inputs on the host boundary;
/// `U32` / `Bool` inputs pass through unchanged.
fn encode(data: &[u32], ty: ScalarType) -> Vec<u32> {
    if matches!(ty, ScalarType::BabyBear | ScalarType::FpExt) {
        data.iter().map(|&v| to_monty(v)).collect()
    } else {
        data.to_vec()
    }
}

/// Per-iteration timing summary of a benchmarked kernel, in milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct KernelBenchTime {
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
}

/// Owns a compiled [`GraphExe`] plus its [`GpuDeviceCtx`] and one device
/// buffer per registered graph input. `set_inputs` uploads host
/// `Vec<u32>` data (Montgomery-encoding BabyBear / FpExt inputs on the
/// fly), `run` launches the graph, `read_outputs` downloads and decodes.
pub struct TestModuleRunner {
    exe: GraphExe,
    ctx: GpuDeviceCtx,
    /// Byte-typed device buffer per registered graph input, in
    /// registration order. Lazily allocated on `set_inputs` from the
    /// exe's current `input_size(i)`. Only used in pool-mode.
    in_bufs: Vec<Option<DeviceBuffer<u8>>>,
    input_types: Vec<ScalarType>,
    output_types: Vec<ScalarType>,
    /// `Some` after the first [`Self::set_symbol`] call: the runner
    /// switches to direct-drive mode, allocating its own device buffers
    /// per launch and calling [`KernelProgram`]'s ABI directly.
    direct: Option<DirectDrive>,
}

/// Direct-drive state: independent per-launch device buffers, sized from
/// the kernel's current `input_size(i)` / `output_size(i)` (which change
/// with each `set_symbol`).
struct DirectDrive {
    /// Node index of the runner's kernel in `GraphExe::nodes`. Multi-size
    /// direct-drive assumes a single-kernel graph (produced by
    /// [`GraphModule::from_ir`]).
    node_idx: usize,
    in_bufs: Vec<DeviceBuffer<u8>>,
    out_bufs: Vec<DeviceBuffer<u8>>,
}

impl TestModuleRunner {
    /// JIT-compiles `graph` via [`GraphCompiler::new().compile`].
    pub fn new(graph: GraphModule) -> Result<Self, CompileError> {
        Self::with_compiler(GraphCompiler::new(), graph)
    }

    /// [`Self::new`] with a caller-provided compiler (for tests that
    /// need custom `dump_ir` / `arch` / etc. via the compiler's
    /// builder API).
    pub fn with_compiler(
        compiler: GraphCompiler,
        graph: GraphModule,
    ) -> Result<Self, CompileError> {
        let ctx = GpuDeviceCtx::for_current_device()
            .map_err(|e| CompileError::Runtime(format!("GpuDeviceCtx: {e:?}")))?;
        // Snapshot input/output scalar types from the graph before we
        // hand it to the compiler — post-compile the source `ir::Module`
        // may have been rewritten and the graph BufInfos alone don't
        // carry element type distinction between BabyBear and U32.
        let input_types = infer_iface_types(graph.as_builder(), true);
        let output_types = infer_iface_types(graph.as_builder(), false);
        let exe = compiler.compile(graph.into_builder())?;
        let n_in = exe.num_inputs();
        Ok(Self {
            exe,
            ctx,
            in_bufs: (0..n_in).map(|_| None).collect(),
            input_types,
            output_types,
            direct: None,
        })
    }

    pub fn num_inputs(&self) -> usize {
        self.in_bufs.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.output_types.len()
    }

    /// Rebinds a residual kernel parameter and switches to direct-drive
    /// mode. Subsequent [`Self::set_inputs`] / [`Self::run`] /
    /// [`Self::read_outputs`] bypass the compiled pool and drive the
    /// underlying [`KernelProgram`] directly — one nvcc + one dlopen
    /// serves many sizes.
    ///
    /// Panics if the runner's graph is not single-node (multi-size
    /// direct-drive assumes a `GraphModule::from_ir` layout).
    pub fn set_symbol(&mut self, name: &str, value: i64) {
        // Locate the sole kernel node the first time we enter direct-drive.
        let node_idx = match &self.direct {
            Some(d) => d.node_idx,
            None => {
                let mut found = None;
                for i in 0..self.exe.num_nodes() {
                    if self.exe.is_kernel_node(i) {
                        assert!(
                            found.is_none(),
                            "TestModuleRunner::set_symbol assumes one kernel node; \
                             got multiple in the graph",
                        );
                        found = Some(i);
                    }
                }
                found.expect("TestModuleRunner::set_symbol: no kernel node in graph")
            }
        };
        self.exe.kernel_program(node_idx).set_symbol(name, value);
        // Any previously-owned direct buffers are now the wrong size;
        // set_inputs below will re-allocate at the current sizes.
        self.direct = Some(DirectDrive {
            node_idx,
            in_bufs: Vec::new(),
            out_bufs: Vec::new(),
        });
    }

    /// Uploads host input tensors, Montgomery-encoding BabyBear / FpExt
    /// inputs before H2D. `U32` and `Bool` inputs pass through unchanged.
    ///
    /// In direct-drive mode (after [`Self::set_symbol`]), each input's
    /// device buffer is (re)allocated at the kernel's current
    /// `input_size(i)` and bound directly on the underlying
    /// [`KernelProgram`] rather than routed through the pool.
    pub fn set_inputs(&mut self, inputs: &[Vec<u32>]) {
        assert_eq!(
            inputs.len(),
            self.in_bufs.len(),
            "expected {} inputs, got {}",
            self.in_bufs.len(),
            inputs.len(),
        );
        match &mut self.direct {
            None => {
                for (i, data) in inputs.iter().enumerate() {
                    let want_bytes = self.exe.input_size(i);
                    let src = encode(data, self.input_types[i]);
                    let src_bytes = u32_slice_as_bytes(&src);
                    assert_eq!(
                        src_bytes.len(),
                        want_bytes,
                        "input {i}: {} bytes supplied, kernel expects {want_bytes}",
                        src_bytes.len(),
                    );
                    let buf = self.in_bufs[i].get_or_insert_with(|| {
                        DeviceBuffer::<u8>::with_capacity_on(want_bytes, &self.ctx)
                    });
                    src_bytes.copy_to_on(buf, &self.ctx).expect("H2D copy");
                    self.exe.set_input(&self.ctx, i, buf).expect("set_input");
                }
            }
            Some(direct) => {
                let node_idx = direct.node_idx;
                let km = self.exe.kernel_program(node_idx);
                // (Re)allocate in-bufs at the kernel's currently-implied
                // sizes and bind them directly.
                direct.in_bufs.clear();
                for (i, data) in inputs.iter().enumerate() {
                    let want_bytes = km.input_size(i);
                    let src = encode(data, self.input_types[i]);
                    let src_bytes = u32_slice_as_bytes(&src);
                    assert_eq!(
                        src_bytes.len(),
                        want_bytes,
                        "input {i}: {} bytes supplied, kernel expects {want_bytes}",
                        src_bytes.len(),
                    );
                    let mut buf = DeviceBuffer::<u8>::with_capacity_on(want_bytes, &self.ctx);
                    src_bytes.copy_to_on(&mut buf, &self.ctx).expect("H2D copy");
                    km.set_input(i, &buf).expect("set_input");
                    direct.in_bufs.push(buf);
                }
                // Also (re)allocate out-bufs at the current sizes so run
                // has somewhere to write.
                direct.out_bufs.clear();
                for i in 0..self.output_types.len() {
                    let want_bytes = km.output_size(i);
                    let buf = DeviceBuffer::<u8>::with_capacity_on(want_bytes, &self.ctx);
                    km.set_output(i, &buf).expect("set_output");
                    direct.out_bufs.push(buf);
                }
            }
        }
    }

    /// Launches the compiled graph (pool-mode) or the underlying kernel
    /// (direct-drive) asynchronously on the runner's stream.
    pub fn run(&mut self) {
        match &self.direct {
            None => self.exe.run(&self.ctx).expect("graph run"),
            Some(d) => {
                let node_idx = d.node_idx;
                self.exe
                    .kernel_program(node_idx)
                    .run(&self.ctx.stream)
                    .expect("kernel run");
            }
        }
    }

    /// Blocks until every launch on the runner's stream has finished.
    pub fn sync(&self) {
        self.ctx.stream.synchronize().expect("stream sync");
    }

    /// Copies every graph output back to host as canonical `u32`
    /// (Montgomery-decoding BabyBear / FpExt).
    pub fn read_outputs(&self) -> Vec<Vec<u32>> {
        (0..self.output_types.len())
            .map(|i| {
                let bytes: Vec<u8> = match &self.direct {
                    None => self.exe.get_output(i).to_host_on(&self.ctx).expect("D2H"),
                    Some(d) => d.out_bufs[i].to_host_on(&self.ctx).expect("D2H"),
                };
                assert_eq!(bytes.len() % 4, 0);
                let raw: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                if matches!(
                    self.output_types[i],
                    ScalarType::BabyBear | ScalarType::FpExt
                ) {
                    raw.into_iter().map(from_monty).collect()
                } else {
                    raw
                }
            })
            .collect()
    }

    /// Runs the graph `n_warmup + n_iter` times and returns median /
    /// q25 / q75 per-launch times in milliseconds. Each timed
    /// iteration syncs the stream so its measurement isolates a single
    /// launch.
    pub fn bench(&mut self, n_warmup: usize, n_iter: usize) -> KernelBenchTime {
        assert!(n_iter > 0, "bench: n_iter must be > 0");
        for _ in 0..n_warmup {
            self.run();
        }
        self.ctx.stream.synchronize().expect("warmup sync");

        let mut samples: Vec<f64> = Vec::with_capacity(n_iter);
        for _ in 0..n_iter {
            let t0 = Instant::now();
            self.run();
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

/// Digs into the graph builder's first Kernel node to pull each
/// registered interface buffer's element type. Assumes a single-kernel
/// graph (the common test shape) — multi-kernel graphs should build the
/// exe directly and not use `TestModuleRunner`.
fn infer_iface_types(g: &crate::graph_ir::GraphBuilder, inputs: bool) -> Vec<ScalarType> {
    use crate::graph_ir::GraphNode;
    let iface = if inputs {
        g.input_bufs()
    } else {
        g.output_bufs()
    };
    iface
        .iter()
        .map(|buf_id| {
            for node in &g.nodes {
                if let GraphNode::Kernel(k) = node {
                    if inputs {
                        if let Some(pos) = k.inputs.iter().position(|b| b == buf_id) {
                            let decl = &k.module.builder.inputs()[pos];
                            return decl.elem;
                        }
                    } else if let Some(pos) = k.outputs.iter().position(|b| b == buf_id) {
                        // Look up the top-level output type via type_infer.
                        let types = crate::passes::type_infer(&k.module)
                            .expect("interface output kernel must type-check");
                        let body_ty = types.get(k.module.body).clone();
                        let member_types: Vec<crate::ir::Type> = match body_ty {
                            crate::ir::Type::Tuple(ts) => ts,
                            other => vec![other],
                        };
                        return member_types[pos]
                            .scalar_type()
                            .expect("output must be a tensor");
                    }
                }
            }
            // Fallback: bytes-only view. u32 keeps set_inputs/read_outputs
            // in a passthrough mode.
            ScalarType::U32
        })
        .collect()
}

/// If `BENCH_KERNEL` is set, benchmarks `runner` and prints its median /
/// q25 / q75 launch time. No-op otherwise. `BENCH_KERNEL_WARMUP` and
/// `BENCH_KERNEL_ITERS` override the defaults (5 and 50).
pub fn maybe_bench(runner: &mut TestModuleRunner, name: &str) {
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
        t.median, t.q25, t.q75,
    );
}
