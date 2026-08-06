//! Per-kernel backend: strict [`ir::Module`] → dlopen'd [`KernelProgram`].
//!
//! `ModuleCompiler` is the crate-facing kernel-compile surface. It is
//! intentionally *pure*: no monomorphization, no shape hints, no
//! canonicalize rewrites. It expects an already-canonical, already-
//! monomorphized single-kernel [`ir::Module`] (produced by the graph compiler
//! passes) and lowers it to a [`KernelProgram`] via:
//!
//! ```text
//! ir::Module ──lower──▶ KirProgram ──codegen──▶ KernelProgram
//!    HIR                  KIR                     dlopen'd .so
//! ```
//!
//! Multi-kernel HIR modules (post reduce-tree lowering, before graph split)
//! are rejected — splitting them into separate graph nodes is the graph
//! compiler's job.

use std::{path::PathBuf, time::Duration};

use crate::{
    ir,
    kernel_ir::KirProgram,
    passes,
    runtime::{CompileOptions, KernelProgram, Verbosity},
    CompileError,
};

/// Configuration + entry points for lowering a single HIR module to a
/// dlopen'd CUDA artifact. See the module docs for the pipeline.
#[derive(Clone, Debug)]
pub struct ModuleCompiler {
    /// Path to the nvcc binary.
    pub nvcc: String,
    /// GPU architecture, e.g. `sm_120` or `native`.
    pub arch: String,
    /// Extra nvcc flags appended after the built-in ones.
    pub extra_nvcc_flags: Vec<String>,
    /// Directory to write per-pass IR dumps into. Nothing is written when
    /// this is `None` regardless of [`Self::verbosity`].
    pub dump_dir: Option<PathBuf>,
    /// How much to dump when `dump_dir` is set.
    pub verbosity: Verbosity,
    /// The `check_accesses` flag is a compile-config bit here, but the
    /// access check itself runs graph-side (it needs concrete bindings, and
    /// the graph is where those live). `ModuleCompiler::lower` does not
    /// consult it.
    pub check_accesses: bool,
    /// Per-invocation wall-clock nvcc timeout. `None` disables the timeout.
    pub nvcc_timeout: Option<Duration>,
}

impl Default for ModuleCompiler {
    /// Env-var defaults: `NVCC`, `CRYPTO_COMPILER_CUDA_ARCH`,
    /// `CRYPTO_COMPILER_DUMP_IR`, `CRYPTO_COMPILER_VERBOSITY`,
    /// `CRYPTO_COMPILER_CHECK_ACCESSES`, `NVCC_TIMEOUT_SECS`.
    fn default() -> Self {
        // Env parsing is shared with `CompileOptions::default`.
        let opts = CompileOptions::default();
        Self {
            nvcc: opts.nvcc,
            arch: opts.arch,
            extra_nvcc_flags: opts.extra_nvcc_flags,
            dump_dir: opts.dump_ir,
            verbosity: opts.verbosity,
            check_accesses: opts.check_accesses,
            nvcc_timeout: opts.nvcc_timeout,
        }
    }
}

impl ModuleCompiler {
    /// Fresh compiler with env-var-derived defaults (see [`Self::default`]).
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_verbosity(&mut self, v: Verbosity) -> &mut Self {
        self.verbosity = v;
        self
    }

    pub fn set_check_accesses(&mut self, on: bool) -> &mut Self {
        self.check_accesses = on;
        self
    }

    pub fn set_dump_dir(&mut self, dir: impl Into<PathBuf>) -> &mut Self {
        self.dump_dir = Some(dir.into());
        self
    }

    pub fn set_arch(&mut self, arch: impl Into<String>) -> &mut Self {
        self.arch = arch.into();
        self
    }

    pub fn set_nvcc(&mut self, nvcc: impl Into<String>) -> &mut Self {
        self.nvcc = nvcc.into();
        self
    }

    pub fn add_flag(&mut self, flag: impl Into<String>) -> &mut Self {
        self.extra_nvcc_flags.push(flag.into());
        self
    }

    pub fn set_nvcc_timeout(&mut self, timeout: Option<Duration>) -> &mut Self {
        self.nvcc_timeout = timeout;
        self
    }

    /// HIR → optimized KIR. Strict: no rewriting, only checking + lowering.
    ///
    /// Preconditions (each violation returns a `CompileError`):
    /// 1. Every param must be either monomorphized away or live only in outer bounds —
    ///    [`passes::required_params`] must be empty.
    /// 2. The module must lower to exactly one kernel.
    /// 3. The module must be canonical: post-hoc single-kernel check catches non-canonical input.
    pub fn lower(&self, m: ir::Module) -> Result<KirProgram, CompileError> {
        // (1) inner bounds must be concrete
        let required = passes::required_params(&m);
        if let Some(&v) = required.first() {
            return Err(CompileError::Monomorphize(format!(
                "param `{}` appears in a must-be-concrete position; \
                 monomorphize in the graph before calling ModuleCompiler::lower",
                m.builder.param_name(v).unwrap_or("<?>")
            )));
        }

        let types = passes::type_infer(&m)?;
        // Canonicalize is idempotent: on an already-canonical input this is a
        // structural no-op that just re-derives the `Program` witness.
        let program = passes::canonicalize(m, types)?;

        // (2) exactly one kernel — splitting is the graph compiler's job.
        if program.kernels.len() != 1 {
            return Err(CompileError::Lower(format!(
                "module `{}` produced {} kernels; split it in the graph first",
                program.module.name,
                program.kernels.len()
            )));
        }

        let mut kp = passes::lower_to_kir(&program)?;
        passes::layout_infer(&mut kp);
        passes::insert_sync(&mut kp);
        Ok(kp)
    }

    /// KIR → dlopen'd CUDA artifact (source emission + verify + nvcc + dlopen).
    pub fn codegen(&self, kp: KirProgram) -> Result<KernelProgram, CompileError> {
        let source = passes::codegen(&kp)?;
        passes::verify(&kp)?;
        let opts = self.to_compile_options();
        let name = kp.name.clone();
        KernelProgram::load(&kp, &source, &opts, &name)
    }

    /// Convenience: `lower` then `codegen` in one call.
    ///
    /// Callers must supply a canonical, single-kernel, monomorphized module
    /// (the graph compiler passes produce these).
    pub fn compile(&self, m: ir::Module) -> Result<KernelProgram, CompileError> {
        let kp = self.lower(m)?;
        self.codegen(kp)
    }

    /// Internal bridge to the runtime's positional [`CompileOptions`] used
    /// by [`KernelProgram::load`].
    pub(crate) fn to_compile_options(&self) -> CompileOptions {
        CompileOptions {
            nvcc: self.nvcc.clone(),
            arch: self.arch.clone(),
            extra_nvcc_flags: self.extra_nvcc_flags.clone(),
            dump_ir: self.dump_dir.clone(),
            verbosity: self.verbosity,
            check_accesses: self.check_accesses,
            nvcc_timeout: self.nvcc_timeout,
        }
    }
}
