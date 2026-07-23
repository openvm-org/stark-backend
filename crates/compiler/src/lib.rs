//! A compiler that lowers a light functional DSL for cryptography compute
//! kernels (NTT, Poseidon2 Merkle trees, ...) to CUDA C++, JIT-compiles it
//! with nvcc and exposes the resulting module through a C ABI loaded via
//! `dlopen`. See `design.md` for the overall architecture.
//!
//! Pipeline: [`ir::Module`] --type_infer/canonicalize-->
//! [`passes::canonicalize::Program`] --plan_global_scratch/lower_to_kir-->
//! [`kernel_ir::KernelProgram`] --layout_infer--> --insert_sync-->
//! --plan_shared_mem--> --codegen--> CUDA C++ --nvcc/dlopen-->
//! [`runtime::KernelModule`].

pub use crypto_compiler_macros::kernel;

pub mod dump;
#[cfg(feature = "planner")]
pub mod graph_exe;
pub mod graph_ir;
pub mod ir;
pub mod kernel_ir;
pub mod kernels;
pub mod passes;
#[cfg(feature = "planner")]
pub mod planner;
pub mod quast;
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
    #[error("codegen error: {0}")]
    Codegen(String),
    #[error("IR verification failed: {0}")]
    Verify(String),
    #[error("quasi-affine expression error: {0}")]
    Quast(String),
    #[error("nvcc failed: {0}")]
    Nvcc(String),
    #[error("dlopen failed: {0}")]
    Load(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Compiles a module end-to-end: run the [`passes`] pipeline down to CUDA
/// C++, build with nvcc and dlopen the result.
///
/// With [`runtime::CompileOptions::dump_ir`] set, writes `{name}.hir` and
/// `{name}.kir` dumps of both IR levels plus the generated `{name}.cu` into
/// that directory (the IR dumps are written before codegen so they survive
/// codegen failures).
pub fn compile_and_load(
    module: ir::Module,
    options: &runtime::CompileOptions,
) -> Result<runtime::KernelModule, CompileError> {
    let hir = options.dump_ir.as_ref().map(|_| dump::dump_hir(&module));
    let types = passes::type_infer(&module)?;
    let program = passes::canonicalize(module, types)?;
    let scratch = passes::plan_global_scratch(&program)?;
    let mut kprog = passes::lower_to_kir(&program, &scratch)?;
    passes::layout_infer(&mut kprog);
    passes::insert_sync(&mut kprog);
    let shared = passes::plan_shared_mem(&kprog);
    if let Some(dir) = &options.dump_ir {
        let kir = dump::dump_kernel_ir(&kprog, &shared);
        dump::write_ir_dumps(dir, &kprog.name, &hir.unwrap(), &kir)?;
    }
    let source = passes::codegen(&kprog)?;
    if let Some(dir) = &options.dump_ir {
        dump::write_cuda_dump(dir, &kprog.name, &source)?;
    }
    passes::verify(&kprog)?;
    runtime::KernelModule::load(&kprog, &source, options)
}
