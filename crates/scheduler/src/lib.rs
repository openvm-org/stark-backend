//! Resource-aware scheduling of an abstract task graph.
//!
//! An [`Engine`] holds a [`Budget`] — one ceiling per resource axis — and a
//! graph of [`Node`]s, each declaring the [`ResourceProfile`] it occupies while
//! it runs. [`Engine::admit`] hands back the nodes that are both dependency-free
//! and small enough to fit what is left of the budget; the caller runs them
//! however it likes and reports each one back through [`Engine::complete`],
//! which releases its resources.
//!
//! The engine names no domain concept. What a node *is*, what "GPU bytes" cost,
//! and how work actually runs are all the caller's business.

mod engine;
mod resource;

pub use engine::{Admission, Engine, Node, SchedulerError};
pub use resource::{Budget, ResourceProfile};
