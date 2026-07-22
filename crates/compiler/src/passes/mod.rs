//! Compiler passes, one module per pass, in pipeline order:
//!
//! - HIR: [`type_infer`], [`canonicalize`], [`plan_global_scratch`], [`lower_to_kir`];
//! - KernelIR: [`layout_infer`], [`insert_sync`], [`plan_shared_mem`], [`codegen`].
//!
//! [`verify`] structurally checks the KernelIR produced by the pipeline;
//! shared helpers live in [`utils`].

pub mod canonicalize;
pub mod codegen;
pub mod insert_sync;
pub mod layout_infer;
pub mod lower_to_kir;
pub mod plan_global_scratch;
pub mod plan_shared_mem;
pub mod type_infer;
pub mod utils;
pub mod verify;

pub use self::{
    canonicalize::{canonicalize, is_canonicalized},
    codegen::codegen,
    insert_sync::insert_sync,
    layout_infer::layout_infer,
    lower_to_kir::lower_to_kir,
    plan_global_scratch::{plan_global_scratch, GlobalScratchPlan},
    plan_shared_mem::{plan_shared_mem, SharedMemPlan},
    type_infer::{type_check, type_infer, TypeMap},
    verify::verify,
};
