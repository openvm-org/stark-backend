//! JIT runtime: compiles generated CUDA C++ with `nvcc` into a shared
//! library, `dlopen`s it, and wraps the exported C interface in a safe
//! [`KernelModule`] that integrates with `openvm-cuda-common` buffers and
//! streams.

use std::{ffi::c_void, fs, path::PathBuf, process::Command};

use libloading::Library;
use openvm_cuda_common::{
    d_buffer::DeviceBuffer,
    stream::{cudaStream_t, CudaStream, GpuDeviceCtx},
};

use crate::{kernel_ir::KernelProgram, CompileError};

#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// Path to the nvcc binary.
    pub nvcc: String,
    /// GPU architecture, e.g. `sm_120` or `native`.
    pub arch: String,
    pub extra_nvcc_flags: Vec<String>,
    /// Directory to write `{name}.hir` / `{name}.kir` IR dumps into.
    pub dump_ir: Option<PathBuf>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            nvcc: std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into()),
            arch: std::env::var("CRYPTO_COMPILER_CUDA_ARCH").unwrap_or_else(|_| "native".into()),
            extra_nvcc_flags: Vec::new(),
            dump_ir: std::env::var_os("CRYPTO_COMPILER_DUMP_IR").map(PathBuf::from),
        }
    }
}

type MakeModuleFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyModuleFn = unsafe extern "C" fn(*mut c_void);
type QueryFn = unsafe extern "C" fn(*mut c_void) -> u64;
type QueryIdxFn = unsafe extern "C" fn(*mut c_void, u64) -> u64;
type SetIdxPtrFn = unsafe extern "C" fn(*mut c_void, u64, *mut c_void);
type SetPtrFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
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
    run: RunFn,
}

/// A compiled, dlopen-ed kernel module.
pub struct KernelModule {
    prog: *mut c_void,
    vt: VTable,
    scratch: Option<DeviceBuffer<u8>>,
    source: String,
    /// The library must stay loaded while `prog` and the vtable exist.
    _lib: Library,
    _dir: tempfile::TempDir,
}

impl KernelModule {
    /// Writes `source` to a temp dir, compiles it with nvcc into a shared
    /// library, loads it and instantiates the module.
    pub fn load(
        kprog: &KernelProgram,
        source: &str,
        options: &CompileOptions,
    ) -> Result<Self, CompileError> {
        let dir = tempfile::Builder::new()
            .prefix("crypto-compiler-")
            .tempdir()?;
        let cu_path = dir.path().join("module.cu");
        let so_path = dir.path().join("libmodule.so");
        fs::write(&cu_path, source)?;

        let mut cmd = Command::new(&options.nvcc);
        cmd.arg("-O3")
            .arg("--shared")
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg(format!("-arch={}", options.arch))
            .args(&options.extra_nvcc_flags)
            .arg("-o")
            .arg(&so_path)
            .arg(&cu_path);
        let out = cmd
            .output()
            .map_err(|e| CompileError::Nvcc(format!("failed to spawn {}: {e}", options.nvcc)))?;
        if !out.status.success() {
            return Err(CompileError::Nvcc(format!(
                "{:?} exited with {}\nstdout:\n{}\nstderr:\n{}",
                cmd.get_program(),
                out.status,
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            )));
        }

        let lib = unsafe { Library::new(&so_path) }
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
            run: sym!("run", RunFn),
        };

        let prog = unsafe { make_module() };
        if prog.is_null() {
            return Err(CompileError::Load("make_module returned null".into()));
        }
        let module = Self {
            prog,
            vt,
            scratch: None,
            source: source.to_string(),
            _lib: lib,
            _dir: dir,
        };
        debug_assert_eq!(module.num_inputs(), kprog.input_bufs.len());
        debug_assert_eq!(module.num_outputs(), kprog.output_bufs.len());
        Ok(module)
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

    /// Size of input `i` in bytes.
    pub fn input_size(&self, i: usize) -> usize {
        assert!(i < self.num_inputs(), "input index out of range");
        unsafe { (self.vt.input_size)(self.prog, i as u64) as usize }
    }

    /// Size of output `i` in bytes.
    pub fn output_size(&self, i: usize) -> usize {
        assert!(i < self.num_outputs(), "output index out of range");
        unsafe { (self.vt.output_size)(self.prog, i as u64) as usize }
    }

    pub fn scratch_size(&self) -> usize {
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
    }

    /// Launches the whole kernel sequence on `stream` (asynchronous).
    pub fn run(&self, stream: &CudaStream) -> Result<(), CompileError> {
        if self.scratch_size() > 0 && self.scratch.is_none() {
            return Err(CompileError::Runtime(
                "scratch buffer not set; call ensure_scratch first".into(),
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
