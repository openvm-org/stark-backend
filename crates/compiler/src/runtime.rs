//! JIT runtime: compiles generated CUDA C++ with `nvcc` into a shared
//! library, `dlopen`s it, and wraps the exported C interface in a safe
//! [`KernelModule`] that integrates with `openvm-cuda-common` buffers and
//! streams.

use std::{
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use libloading::Library;
use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    stream::{cudaStream_t, CudaStream, GpuDeviceCtx},
};
use serde::{Deserialize, Serialize};

use crate::{ir::ScalarType, kernel_ir::KernelProgram, CompileError};

/// On-disk metadata written alongside `libmodule.so` and `module.cu`. The
/// scalar types are tagged as `u8` so the JSON stays stable if ScalarType
/// gains variants (unknown tags surface as a `CompileError::Load`).
#[derive(Serialize, Deserialize)]
struct SavedMetadata {
    input_types: Vec<u8>,
    output_types: Vec<u8>,
    /// Runtime parameter names, in `KernelProgram::params` order. Defaults
    /// to empty so metadata written before the parameter ABI still loads.
    #[serde(default)]
    params: Vec<String>,
}

fn scalar_ty_to_tag(t: &ScalarType) -> u8 {
    match t {
        ScalarType::BabyBear => 0,
        ScalarType::FpExt => 1,
        ScalarType::U32 => 2,
        ScalarType::Bool => 3,
    }
}

fn scalar_ty_from_tag(t: u8) -> Result<ScalarType, CompileError> {
    Ok(match t {
        0 => ScalarType::BabyBear,
        1 => ScalarType::FpExt,
        2 => ScalarType::U32,
        3 => ScalarType::Bool,
        other => {
            return Err(CompileError::Load(format!(
                "load_from_dir: unknown scalar type tag {other}"
            )))
        }
    })
}

/// How much detail [`crate::compile_and_load`] should write into
/// [`CompileOptions::dump_ir`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Verbosity {
    /// Write nothing.
    None,
    /// Dump `{name}.hir`, `{name}.kir`, `{name}.cu` (the pre-2026 default).
    #[default]
    Basic,
    /// Dump the IR after every major pass, plus the analyses each pass
    /// produces. Files land alongside the `Basic` outputs and are prefixed
    /// with a step number so a reader sees the pipeline order.
    Verbose,
}

#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Path to the nvcc binary.
    pub nvcc: String,
    /// GPU architecture, e.g. `sm_120` or `native`.
    pub arch: String,
    pub extra_nvcc_flags: Vec<String>,
    /// Directory to write IR dumps into. Nothing is written when this is
    /// `None` regardless of [`Self::verbosity`].
    pub dump_ir: Option<PathBuf>,
    /// How much to dump when `dump_ir` is set.
    pub verbosity: Verbosity,
    /// Validate every memory access of each concrete instantiation before
    /// compiling it (see [`crate::passes::check_accesses`]): bounds for
    /// reads and writes, injectivity across parallel points for writes,
    /// plus the scatter bijectivity/inverse checks that fire during
    /// concrete canonicalization. Author-trusted (`SExpr`/`Blackbox`)
    /// accesses are skipped. Defaults to the `CRYPTO_COMPILER_CHECK_ACCESSES`
    /// environment variable.
    pub check_accesses: bool,
    /// Per-invocation wall-clock limit for nvcc. `None` disables the
    /// timeout (unbounded wait, the previous behaviour). On timeout the
    /// nvcc process group is SIGKILL'd and [`CompileError::NvccTimeout`]
    /// is returned with the module name embedded, so an aggregating
    /// caller (e.g. [`crate::graph_exe::GraphCompiler::compile`]) can
    /// surface which kernels were the slow ones.
    pub nvcc_timeout: Option<Duration>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            nvcc: std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into()),
            arch: std::env::var("CRYPTO_COMPILER_CUDA_ARCH").unwrap_or_else(|_| "native".into()),
            extra_nvcc_flags: Vec::new(),
            dump_ir: std::env::var_os("CRYPTO_COMPILER_DUMP_IR").map(PathBuf::from),
            verbosity: match std::env::var("CRYPTO_COMPILER_VERBOSITY").as_deref() {
                Ok("none") | Ok("None") | Ok("NONE") => Verbosity::None,
                Ok("verbose") | Ok("Verbose") | Ok("VERBOSE") => Verbosity::Verbose,
                _ => Verbosity::Basic,
            },
            check_accesses: std::env::var("CRYPTO_COMPILER_CHECK_ACCESSES")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(false),
            nvcc_timeout: std::env::var("NVCC_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .map(Duration::from_secs),
        }
    }
}

type MakeModuleFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyModuleFn = unsafe extern "C" fn(*mut c_void);
type QueryFn = unsafe extern "C" fn(*mut c_void) -> u64;
type QueryIdxFn = unsafe extern "C" fn(*mut c_void, u64) -> u64;
type SetIdxPtrFn = unsafe extern "C" fn(*mut c_void, u64, *mut c_void);
type SetPtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
type SetParamFn = unsafe extern "C" fn(*mut c_void, u64, i64);
type RunFn = unsafe extern "C" fn(*mut c_void, cudaStream_t) -> i32;

struct VTable {
    destroy_module: DestroyModuleFn,
    scratch_size: QueryFn,
    num_outputs: QueryFn,
    output_size: QueryIdxFn,
    num_inputs: QueryFn,
    input_size: QueryIdxFn,
    set_input: SetIdxPtrFn,
    set_output: SetIdxPtrFn,
    set_scratch_buf: SetPtrFn,
    set_param: SetParamFn,
    run: RunFn,
}

/// A compiled, dlopen-ed kernel module.
pub struct KernelModule {
    prog: *mut c_void,
    vt: VTable,
    scratch: Option<DeviceBuffer<u8>>,
    /// Whether the C ABI's scratch pointer has been bound this run, either
    /// through [`KernelModule::ensure_scratch`] (owned) or
    /// [`KernelModule::set_scratch`] (external).
    scratch_bound: bool,
    /// Runtime parameter names, in `KernelProgram::params` order. Buffer
    /// sizes and grid dims are functions of these values, so every
    /// parameter must be bound via [`KernelModule::set_param`] before the
    /// sizes are queried or the module is run.
    params: Vec<String>,
    /// Which parameters have been bound so far.
    params_set: Vec<bool>,
    source: String,
    /// Scalar element type of each module input, in declaration order.
    /// Populated from `kprog.input_bufs` so callers know which buffers hold
    /// Montgomery-encoded BabyBear vs. plain `U32`/`Bool` data.
    input_types: Vec<ScalarType>,
    /// Scalar element type of each module output, in declaration order.
    output_types: Vec<ScalarType>,
    /// The library must stay loaded while `prog` and the vtable exist.
    _lib: Library,
    /// Owned scratch directory. `None` when loaded from a persistent path
    /// (the caller owns those files).
    _dir: Option<tempfile::TempDir>,
}

// `prog` is an opaque handle allocated by C code inside the JIT'd library.
// We only touch it through the C ABI and never race launches; instances are
// externally serialized (by `graph_exe`'s single-stream execution or by an
// enclosing `Mutex`), so sharing across threads is sound as long as callers
// don't hand the same handle to two threads at once.
unsafe impl Send for KernelModule {}
unsafe impl Sync for KernelModule {}

/// Filenames used by [`KernelModule::save_artifacts`] /
/// [`KernelModule::load_from_dir`].
pub const KERNEL_MODULE_SO: &str = "libmodule.so";
pub const KERNEL_MODULE_CU: &str = "module.cu";
pub const KERNEL_MODULE_METADATA: &str = "metadata.json";

impl KernelModule {
    /// Writes `source` to a temp dir, compiles it with nvcc into a shared
    /// library, loads it and instantiates the module.
    pub fn load(
        kprog: &KernelProgram,
        source: &str,
        options: &CompileOptions,
        module_name: &str,
    ) -> Result<Self, CompileError> {
        let dir = tempfile::Builder::new()
            .prefix("crypto-compiler-")
            .tempdir()?;
        let cu_path = dir.path().join(KERNEL_MODULE_CU);
        let so_path = dir.path().join(KERNEL_MODULE_SO);
        fs::write(&cu_path, source)?;
        Self::compile_source(&cu_path, &so_path, options, module_name)?;

        let input_types: Vec<ScalarType> = kprog
            .input_bufs
            .iter()
            .map(|&b| kprog.buffer(b).elem)
            .collect();
        let output_types: Vec<ScalarType> = kprog
            .output_bufs
            .iter()
            .map(|&b| kprog.buffer(b).elem)
            .collect();
        let params = kprog.params.iter().map(|(_, n)| n.clone()).collect();
        let module = Self::load_from_so(
            &so_path,
            source.to_string(),
            input_types,
            output_types,
            params,
            Some(dir),
        )?;
        debug_assert_eq!(module.num_inputs(), kprog.input_bufs.len());
        debug_assert_eq!(module.num_outputs(), kprog.output_bufs.len());
        Ok(module)
    }

    /// Invokes `nvcc` on `cu_path` to produce a shared library at `so_path`.
    ///
    /// If [`CompileOptions::nvcc_timeout`] is set, the nvcc process (and
    /// its cicc/ptxas/cudafe children — we launch it in its own Unix
    /// process group and SIGKILL the whole group on timeout so no child
    /// escapes) is bounded to that duration; on expiration we return
    /// [`CompileError::NvccTimeout`] carrying `module_name`.
    fn compile_source(
        cu_path: &Path,
        so_path: &Path,
        options: &CompileOptions,
        module_name: &str,
    ) -> Result<(), CompileError> {
        use std::{
            io::Read,
            time::{Duration, Instant},
        };

        let mut cmd = Command::new(&options.nvcc);
        cmd.arg("-O3")
            .arg("--shared")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg(format!("-arch={}", options.arch));
        if matches!(std::env::var("CUDA_LINEINFO").as_deref(), Ok("1")) {
            cmd.arg("-lineinfo");
        }
        cmd.args(&options.extra_nvcc_flags)
            .arg("-o")
            .arg(so_path)
            .arg(cu_path);

        let timeout = options.nvcc_timeout;
        if timeout.is_none() {
            // Fast path — original behaviour, no polling.
            let out = cmd.output().map_err(|e| {
                CompileError::Nvcc(format!("failed to spawn {}: {e}", options.nvcc))
            })?;
            if !out.status.success() {
                return Err(CompileError::Nvcc(format!(
                    "{:?} exited with {}\nstdout:\n{}\nstderr:\n{}",
                    cmd.get_program(),
                    out.status,
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr),
                )));
            }
            return Ok(());
        }
        let limit = timeout.unwrap();

        // Put nvcc (and everything it forks — cicc, ptxas, cudafe, ...)
        // into its own Unix process group. On timeout we SIGKILL the
        // whole group by pid = -pgid; that leaves no orphans.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| CompileError::Nvcc(format!("failed to spawn {}: {e}", options.nvcc)))?;
        let child_pid = child.id() as i32;
        // Drain stdout/stderr on background threads while polling. nvcc can
        // emit tens of KB of warnings; if nobody reads the pipe it fills up
        // and the child blocks on write forever, which then surfaces as a
        // spurious "timeout" (pipes can be as small as one page when the
        // user's pipe-buffer soft limit is exhausted, e.g. under heavily
        // parallel compilation).
        let drain = |s: Option<Box<dyn Read + Send>>| {
            s.map(|mut s| {
                std::thread::spawn(move || {
                    let mut buf = String::new();
                    let _ = s.read_to_string(&mut buf);
                    buf
                })
            })
        };
        let stdout_thread = drain(
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        );
        let stderr_thread = drain(
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        );
        let start = Instant::now();
        let poll_interval = Duration::from_millis(200);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Ok(status),
                Ok(None) => {
                    if start.elapsed() > limit {
                        // SIGKILL the whole process group. The Rust
                        // Child::kill only SIGKILLs the direct child;
                        // cicc/ptxas siblings would leak. Best-effort:
                        // on any failure we still try Child::kill and
                        // report the timeout.
                        #[cfg(unix)]
                        unsafe {
                            libc::kill(-child_pid, libc::SIGKILL);
                        }
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(CompileError::NvccTimeout {
                            name: module_name.to_string(),
                            seconds: start.elapsed().as_secs_f64(),
                            limit: limit.as_secs_f64(),
                        });
                    }
                    std::thread::sleep(poll_interval);
                }
                Err(e) => break Err(e),
            }
        }
        .map_err(|e| CompileError::Nvcc(format!("wait failed: {e}")))?;

        if !status.success() {
            let stdout = stdout_thread
                .and_then(|t| t.join().ok())
                .unwrap_or_default();
            let stderr = stderr_thread
                .and_then(|t| t.join().ok())
                .unwrap_or_default();
            return Err(CompileError::Nvcc(format!(
                "{:?} exited with {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                cmd.get_program(),
                status,
            )));
        }
        Ok(())
    }

    /// dlopens `so_path`, invokes `make_module`, and packages the result as a
    /// `KernelModule`. The caller supplies `input_types` / `output_types` from
    /// the compiled `KernelProgram` (or from persisted metadata).
    fn load_from_so(
        so_path: &Path,
        source: String,
        input_types: Vec<ScalarType>,
        output_types: Vec<ScalarType>,
        params: Vec<String>,
        owned_dir: Option<tempfile::TempDir>,
    ) -> Result<Self, CompileError> {
        let lib = unsafe { Library::new(so_path) }
            .map_err(|e| CompileError::Load(format!("dlopen {}: {e}", so_path.display())))?;

        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                *unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| CompileError::Load(format!("symbol {}: {e}", $name)))?
            };
        }
        let make_module: MakeModuleFn = sym!("make_module", MakeModuleFn);
        let vt = VTable {
            destroy_module: sym!("destroy_module", DestroyModuleFn),
            scratch_size: sym!("scratch_size", QueryFn),
            num_outputs: sym!("num_outputs", QueryFn),
            output_size: sym!("output_size", QueryIdxFn),
            num_inputs: sym!("num_inputs", QueryFn),
            input_size: sym!("input_size", QueryIdxFn),
            set_input: sym!("set_input", SetIdxPtrFn),
            set_output: sym!("set_output", SetIdxPtrFn),
            set_scratch_buf: sym!("set_scratch_buf", SetPtrFn),
            set_param: sym!("set_param", SetParamFn),
            run: sym!("run", RunFn),
        };

        let prog = unsafe { make_module() };
        if prog.is_null() {
            return Err(CompileError::Load("make_module returned null".into()));
        }
        let params_set = vec![false; params.len()];
        Ok(Self {
            prog,
            vt,
            scratch: None,
            scratch_bound: false,
            params,
            params_set,
            source,
            input_types,
            output_types,
            _lib: lib,
            _dir: owned_dir,
        })
    }

    /// Copies the compiled `.so`, the CUDA source and a small metadata JSON
    /// into `dir` (created if needed). The layout matches
    /// [`Self::load_from_dir`] so a later process can rebuild an equivalent
    /// `KernelModule` without re-running nvcc.
    ///
    /// Fails if the module was loaded from a persistent path (in which case
    /// the caller already owns the artifacts) — we only support saving from
    /// a freshly-JIT'd temp-dir instance.
    pub fn save_artifacts(&self, dir: &Path) -> Result<(), CompileError> {
        let src_dir = self._dir.as_ref().ok_or_else(|| {
            CompileError::Runtime(
                "save_artifacts requires a freshly compiled module; \
                 this instance was loaded from disk"
                    .into(),
            )
        })?;
        fs::create_dir_all(dir)?;
        fs::copy(
            src_dir.path().join(KERNEL_MODULE_SO),
            dir.join(KERNEL_MODULE_SO),
        )?;
        fs::write(dir.join(KERNEL_MODULE_CU), &self.source)?;
        let meta = SavedMetadata {
            input_types: self.input_types.iter().map(scalar_ty_to_tag).collect(),
            output_types: self.output_types.iter().map(scalar_ty_to_tag).collect(),
            params: self.params.clone(),
        };
        fs::write(
            dir.join(KERNEL_MODULE_METADATA),
            serde_json::to_string_pretty(&meta).map_err(|e| {
                CompileError::Runtime(format!("save_artifacts: serialize metadata: {e}"))
            })?,
        )?;
        Ok(())
    }

    /// Reconstructs a `KernelModule` from artifacts previously written by
    /// [`Self::save_artifacts`] (a `libmodule.so`, `module.cu` and
    /// `metadata.json` under `dir`). Skips nvcc.
    pub fn load_from_dir(dir: &Path) -> Result<Self, CompileError> {
        let meta_bytes = fs::read(dir.join(KERNEL_MODULE_METADATA))?;
        let meta: SavedMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| CompileError::Load(format!("load_from_dir: parse metadata: {e}")))?;
        let input_types = meta
            .input_types
            .iter()
            .map(|&t| scalar_ty_from_tag(t))
            .collect::<Result<Vec<_>, _>>()?;
        let output_types = meta
            .output_types
            .iter()
            .map(|&t| scalar_ty_from_tag(t))
            .collect::<Result<Vec<_>, _>>()?;
        let source = fs::read_to_string(dir.join(KERNEL_MODULE_CU)).unwrap_or_default();
        Self::load_from_so(
            &dir.join(KERNEL_MODULE_SO),
            source,
            input_types,
            output_types,
            meta.params,
            None,
        )
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn num_inputs(&self) -> usize {
        unsafe { (self.vt.num_inputs)(self.prog) as usize }
    }

    pub fn num_outputs(&self) -> usize {
        unsafe { (self.vt.num_outputs)(self.prog) as usize }
    }

    /// Runtime parameter names, in declaration order.
    pub fn params(&self) -> &[String] {
        &self.params
    }

    /// Binds runtime parameter `i` (see [`Self::params`] for the order).
    pub fn set_param(&mut self, i: usize, v: i64) {
        assert!(i < self.params.len(), "param index out of range");
        unsafe { (self.vt.set_param)(self.prog, i as u64, v) };
        self.params_set[i] = true;
    }

    /// Whether every runtime parameter has been bound. Vacuously true for
    /// fully concrete modules.
    pub fn params_bound(&self) -> bool {
        self.params_set.iter().all(|&b| b)
    }

    /// Size of input `i` in bytes.
    pub fn input_size(&self, i: usize) -> usize {
        assert!(i < self.num_inputs(), "input index out of range");
        assert!(self.params_bound(), "input_size before params are bound");
        unsafe { (self.vt.input_size)(self.prog, i as u64) as usize }
    }

    /// Scalar element type of input `i`. `BabyBear` inputs are expected in
    /// Montgomery form (`x * R mod P`); `U32`/`Bool`/`FpExt` inputs pass
    /// through unchanged. `FpExt` inputs carry four Montgomery-encoded
    /// BabyBear coefficients per 16-byte element.
    pub fn input_type(&self, i: usize) -> ScalarType {
        self.input_types[i]
    }

    /// Scalar element type of output `i`. Mirrors [`Self::input_type`].
    pub fn output_type(&self, i: usize) -> ScalarType {
        self.output_types[i]
    }

    /// Size of output `i` in bytes.
    pub fn output_size(&self, i: usize) -> usize {
        assert!(i < self.num_outputs(), "output index out of range");
        assert!(self.params_bound(), "output_size before params are bound");
        unsafe { (self.vt.output_size)(self.prog, i as u64) as usize }
    }

    pub fn scratch_size(&self) -> usize {
        assert!(self.params_bound(), "scratch_size before params are bound");
        unsafe { (self.vt.scratch_size)(self.prog) as usize }
    }

    /// Binds a device buffer as input `i`. The buffer must stay alive until
    /// `run` completes.
    pub fn set_input<T>(&mut self, i: usize, buf: &DeviceBuffer<T>) -> Result<(), CompileError> {
        let bytes = buf.len() * size_of::<T>();
        if bytes != self.input_size(i) {
            return Err(CompileError::Runtime(format!(
                "input {i} size mismatch: buffer is {bytes} bytes, expected {}",
                self.input_size(i)
            )));
        }
        unsafe { (self.vt.set_input)(self.prog, i as u64, buf.as_mut_raw_ptr()) };
        Ok(())
    }

    /// Binds a device buffer as output `i`. The buffer must stay alive until
    /// `run` completes.
    pub fn set_output<T>(&mut self, i: usize, buf: &DeviceBuffer<T>) -> Result<(), CompileError> {
        let bytes = buf.len() * size_of::<T>();
        if bytes != self.output_size(i) {
            return Err(CompileError::Runtime(format!(
                "output {i} size mismatch: buffer is {bytes} bytes, expected {}",
                self.output_size(i)
            )));
        }
        unsafe { (self.vt.set_output)(self.prog, i as u64, buf.as_mut_raw_ptr()) };
        Ok(())
    }

    /// Allocates (if needed) and binds the scratch buffer.
    pub fn ensure_scratch(&mut self, ctx: &GpuDeviceCtx) {
        let size = self.scratch_size();
        if size == 0 || self.scratch.is_some() {
            return;
        }
        let buf = DeviceBuffer::<u8>::with_capacity_on(size, ctx);
        unsafe { (self.vt.set_scratch_buf)(self.prog, buf.as_mut_raw_ptr()) };
        self.scratch = Some(buf);
        self.scratch_bound = true;
    }

    /// Binds `buf` as the module's scratch, letting the caller share a
    /// scratch pool across kernels instead of each `KernelModule` owning
    /// its own allocation. `buf` must have at least [`scratch_size`] bytes
    /// and must stay alive until `run` completes. Any previously owned
    /// scratch is dropped (the C ABI now points at `buf`, so the owned one
    /// is unreachable and safe to free).
    pub fn set_scratch(&mut self, buf: &DeviceBuffer<u8>) -> Result<(), CompileError> {
        let want = self.scratch_size();
        if buf.len() < want {
            return Err(CompileError::Runtime(format!(
                "set_scratch: buffer is {} bytes, need at least {want}",
                buf.len()
            )));
        }
        unsafe { (self.vt.set_scratch_buf)(self.prog, buf.as_mut_raw_ptr()) };
        self.scratch = None;
        self.scratch_bound = true;
        Ok(())
    }

    /// Launches the whole kernel sequence on `stream` (asynchronous).
    pub fn run(&self, stream: &CudaStream) -> Result<(), CompileError> {
        if !self.params_bound() {
            return Err(CompileError::Runtime(
                "module parameters not bound; call set_param for every parameter first".into(),
            ));
        }
        if self.scratch_size() > 0 && !self.scratch_bound {
            return Err(CompileError::Runtime(
                "scratch buffer not set; call ensure_scratch or set_scratch first".into(),
            ));
        }
        let code = unsafe { (self.vt.run)(self.prog, stream.as_raw()) };
        if code != 0 {
            return Err(CompileError::Runtime(format!(
                "run failed with cudaError_t = {code}"
            )));
        }
        Ok(())
    }
}

impl Drop for KernelModule {
    fn drop(&mut self) {
        unsafe { (self.vt.destroy_module)(self.prog) };
    }
}
