//! A compiler that lowers a light functional DSL for cryptography compute
//! kernels (NTT, Poseidon2 Merkle trees, ...) to CUDA C++, JIT-compiles it
//! with nvcc and exposes the resulting module through a C ABI loaded via
//! `dlopen`. See `design.md` for the overall architecture.
//!
//! Pipeline: [`ir::Module`] --canonicalize--> [`canonicalize::Program`]
//! --lower--> [`kernel_ir::KernelProgram`] --layout_infer/insert_sync-->
//! --codegen--> CUDA C++ --nvcc/dlopen--> [`runtime::KernelModule`].

pub use crypto_compiler_macros::kernel;

pub mod canonicalize;
pub mod codegen;
pub mod ir;
pub mod kernel_ir;
pub mod kernels;
pub mod lower;
pub mod passes;
pub mod runtime;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CompileError {
    #[error("type error: {0}")]
    Type(String),
    #[error("canonicalization error: {0}")]
    Canonicalize(String),
    #[error("lowering error: {0}")]
    Lower(String),
    #[error("nvcc failed: {0}")]
    Nvcc(String),
    #[error("dlopen failed: {0}")]
    Load(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Compiles a module end-to-end: canonicalize, lower, generate CUDA C++,
/// build with nvcc and dlopen the result.
pub fn compile_and_load(
    module: ir::Module,
    options: &runtime::CompileOptions,
) -> Result<runtime::KernelModule, CompileError> {
    let program = canonicalize::canonicalize(module)?;
    let mut kprog = lower::lower(&program)?;
    passes::layout_infer(&mut kprog);
    passes::insert_sync(&mut kprog);
    let source = codegen::generate_cuda(&kprog);
    runtime::KernelModule::load(&kprog, &source, options)
}
