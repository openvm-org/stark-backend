//! A compiler that lowers a light functional DSL for cryptography compute
//! kernels (NTT, Poseidon2 Merkle trees, ...) to CUDA C++, JIT-compiles it
//! with nvcc and exposes the resulting module through a C ABI loaded via
//! `dlopen`. See `design.md` for the overall architecture.
//!
//! Two compile surfaces:
//!
//! - [`module_compiler::ModuleCompiler`] is the per-kernel backend. It expects an
//!   already-canonical, monomorphized single-kernel [`ir::Module`] and exposes `lower(ir::Module)
//!   -> KirProgram` + `codegen(KirProgram) -> KernelProgram` (or `compile` for both).
//! - [`graph_exe::GraphCompiler`] wraps a `ModuleCompiler` and drives a full
//!   [`graph_ir::GraphBuilder`] through the pass pipeline (`lower_reduce` → `monomorphize` →
//!   `canonicalize` → optional `fuse` → `dce` → `plan_memory`), then JITs every unique residual in
//!   parallel and packages everything into a [`graph_exe::GraphExe`].
//!
//! Per-kernel pipeline: [`ir::Module`] --type_infer/canonicalize-->
//! [`passes::canonicalize::Program`] --lower_to_kir-->
//! [`kernel_ir::KirProgram`] --layout_infer--> --insert_sync-->
//! --plan_shared_mem--> --codegen--> CUDA C++ --nvcc/dlopen-->
//! [`runtime::KernelProgram`].

// `kernel!`-generated code names crate items by their external path
// (`::crypto_compiler::...`); this alias makes that path resolve when the
// macro expands inside this crate too.
extern crate self as crypto_compiler;

pub use crypto_compiler_macros::kernel;

pub mod dump;
pub mod field_ext;
#[cfg(feature = "planner")]
pub mod graph_exe;
pub mod graph_ir;
pub mod ir;
#[cfg(feature = "planner")]
pub mod kernel_cache;
pub mod kernel_ir;
pub mod kernels;
pub mod module_compiler;
pub mod module_hash;
pub mod passes;
#[cfg(feature = "planner")]
pub mod planner;
pub mod poseidon2_parallel;
pub mod quast;
pub mod runtime;
pub mod test_utils;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("type error: {0}")]
    Type(String),
    #[error("canonicalization error: {0}")]
    Canonicalize(String),
    #[error("monomorphization error: {0}")]
    Monomorphize(String),
    #[error("lowering error: {0}")]
    Lower(String),
    #[error("codegen error: {0}")]
    Codegen(String),
    #[error("IR verification failed: {0}")]
    Verify(String),
    #[error("quasi-affine expression error: {0}")]
    Quast(String),
    #[error("access check failed: {0}")]
    AccessCheck(String),
    #[error("nvcc failed: {0}")]
    Nvcc(String),
    #[error("nvcc timed out compiling `{name}` after {seconds:.1}s (limit {limit:.1}s)")]
    NvccTimeout {
        name: String,
        seconds: f64,
        limit: f64,
    },
    #[error("dlopen failed: {0}")]
    Load(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
