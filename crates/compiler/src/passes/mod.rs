//! Compiler passes, one module per pass, in pipeline order:
//!
//! - HIR: [`type_infer`], [`canonicalize`], [`lower_to_kir`];
//! - KernelIR: [`layout_infer`], [`insert_sync`], [`plan_shared_mem`], [`codegen`].
//!
//! [`verify`] structurally checks the KernelIR produced by the pipeline;
//! shared helpers live in [`utils`].
//!
//! [`split_module`] sits outside the single-module pipeline: it splits a
//! multi-kernel module into single-kernel modules for graph-level insertion
//! (see [`crate::graph_ir::GraphBuilder::insert_kernel`]).
//!
//! [`check_accesses`] is an optional debug pass (see
//! [`crate::runtime::CompileOptions::check_accesses`]): it exhaustively
//! validates the memory accesses of a concrete instantiation without
//! compiling anything.

pub mod canonicalize;
pub mod check_accesses;
pub mod codegen;
pub mod fusion;
pub mod inplace;
pub mod insert_sync;
pub mod layout_infer;
pub mod lower_to_kir;
pub mod monomorphize;
pub mod parallel_reduce_rewrite;
pub mod plan_shared_mem;
pub mod split_module;
pub mod type_infer;
pub mod utils;
pub mod verify;

pub use self::{
    canonicalize::{canonicalize, is_canonicalized},
    check_accesses::{check_module_accesses, check_program_accesses},
    codegen::codegen,
    insert_sync::insert_sync,
    layout_infer::layout_infer,
    lower_to_kir::lower_to_kir,
    monomorphize::{monomorphize, monomorphize_for_graph, required_params, GraphMono},
    parallel_reduce_rewrite::rewrite_parallel_reduce,
    plan_shared_mem::{plan_shared_mem, SharedMemPlan},
    split_module::{
        split_module, split_program, ModuleSubgraph, OutputSpec, SubgraphKernel, SubgraphValue,
    },
    type_infer::{type_check, type_infer, TypeMap},
    verify::verify,
};
