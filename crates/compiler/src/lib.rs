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
pub mod field_ext;
#[cfg(feature = "planner")]
pub mod graph_exe;
pub mod graph_ir;
pub mod ir;
#[cfg(feature = "planner")]
pub mod kernel_cache;
pub mod kernel_ir;
pub mod kernels;
pub mod module_hash;
pub mod passes;
#[cfg(feature = "planner")]
pub mod planner;
pub mod poseidon2_parallel;
pub mod quast;
pub mod runner;
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
/// With [`runtime::CompileOptions::dump_ir`] set, writes IR dumps to that
/// directory. The level of detail is controlled by
/// [`runtime::CompileOptions::verbosity`]:
///
/// - [`runtime::Verbosity::None`]: no files are written even when `dump_ir` is set.
/// - [`runtime::Verbosity::Basic`] (default): writes `{name}.hir`, `{name}.kir` and `{name}.cu`. IR
///   dumps are written before codegen so they survive a codegen failure.
/// - [`runtime::Verbosity::Verbose`]: writes `{name}.NN.<tag>.<ext>` per major pass — HIR, type
///   map, canonical program, global scratch plan, KIR after `lower_to_kir` / `layout_infer` /
///   `insert_sync`, the shared-memory plan and the final CUDA source.
pub fn compile_and_load(
    module: &ir::Module,
    options: &runtime::CompileOptions,
) -> Result<runtime::KernelModule, CompileError> {
    use runtime::Verbosity;

    let dump_dir: Option<&std::path::Path> = if options.verbosity == Verbosity::None {
        None
    } else {
        options.dump_ir.as_deref()
    };
    let verbose = matches!(options.verbosity, Verbosity::Verbose);

    let name = module.name.clone();
    let hir = dump_dir.map(|_| dump::dump_hir(module));
    if let (Some(dir), Some(hir), true) = (dump_dir, hir.as_deref(), verbose) {
        dump::write_step_dump(dir, &name, 0, "hir", "hir", hir)?;
    }

    let types = passes::type_infer(module)?;
    if let (Some(dir), true) = (dump_dir, verbose) {
        dump::write_step_dump(dir, &name, 1, "types", "txt", &dump::dump_types(&types))?;
    }

    // `canonicalize` mutates the module's builder in-place; clone so multiple
    // callers (e.g. graph deduplication) can share the same source `Module`
    // without conflict. `IRBuilder` and `Module` derive `Clone` for this.
    let program = passes::canonicalize(module.clone(), types)?;
    if let (Some(dir), true) = (dump_dir, verbose) {
        dump::write_step_dump(
            dir,
            &name,
            2,
            "canonical",
            "txt",
            &dump::dump_program(&program),
        )?;
    }

    let scratch = passes::plan_global_scratch(&program)?;
    if let (Some(dir), true) = (dump_dir, verbose) {
        dump::write_step_dump(
            dir,
            &name,
            3,
            "global_scratch",
            "txt",
            &dump::dump_global_scratch(&scratch),
        )?;
    }

    let mut kprog = passes::lower_to_kir(&program, &scratch)?;
    if let (Some(dir), true) = (dump_dir, verbose) {
        let empty_shared = passes::SharedMemPlan {
            offsets: std::collections::HashMap::new(),
            per_kernel: vec![0; kprog.kernels.len()],
        };
        dump::write_step_dump(
            dir,
            &name,
            4,
            "kir.lowered",
            "kir",
            &dump::dump_kernel_ir(&kprog, &empty_shared),
        )?;
    }

    passes::layout_infer(&mut kprog);
    if let (Some(dir), true) = (dump_dir, verbose) {
        let empty_shared = passes::SharedMemPlan {
            offsets: std::collections::HashMap::new(),
            per_kernel: vec![0; kprog.kernels.len()],
        };
        dump::write_step_dump(
            dir,
            &name,
            5,
            "kir.layout",
            "kir",
            &dump::dump_kernel_ir(&kprog, &empty_shared),
        )?;
    }

    passes::insert_sync(&mut kprog);
    let shared = passes::plan_shared_mem(&kprog);
    if let (Some(dir), true) = (dump_dir, verbose) {
        dump::write_step_dump(
            dir,
            &name,
            6,
            "kir.synced",
            "kir",
            &dump::dump_kernel_ir(&kprog, &shared),
        )?;
        dump::write_step_dump(
            dir,
            &name,
            7,
            "shared_mem",
            "txt",
            &dump::dump_shared_mem(&kprog, &shared),
        )?;
    }

    if let Some(dir) = dump_dir {
        let kir = dump::dump_kernel_ir(&kprog, &shared);
        dump::write_ir_dumps(dir, &kprog.name, hir.as_deref().unwrap(), &kir)?;
    }
    let source = passes::codegen(&kprog)?;
    if let Some(dir) = dump_dir {
        dump::write_cuda_dump(dir, &kprog.name, &source)?;
        if verbose {
            dump::write_step_dump(dir, &name, 8, "codegen", "cu", &source)?;
        }
    }
    passes::verify(&kprog)?;
    runtime::KernelModule::load(&kprog, &source, options)
}
