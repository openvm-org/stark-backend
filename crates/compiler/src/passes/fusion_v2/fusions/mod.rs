//! Fusion candidate-producing passes.
//!
//! Each submodule implements one pattern from
//! `detailed-fusion-plan-v2.md` §10 (producer-consumer, fanout,
//! small-kernel, epilogue, horizontal). Each pass:
//!
//! - discovers matches on the current alternative graph;
//! - synthesizes candidate HIR modules for the matched patterns;
//! - normalizes the candidates (§9);
//! - returns [`CandidateDraft`](producer_consumer::CandidateDraft) values for the saturation driver
//!   to insert.
//!
//! The M3-start slice implements a narrow producer-consumer path: only
//! identity-access, scalar-body, single-seam candidates. Legality
//! extensions (affine permutation, nested compute, reduction producer,
//! multi-seam) land in follow-up passes.

pub mod producer_consumer;
