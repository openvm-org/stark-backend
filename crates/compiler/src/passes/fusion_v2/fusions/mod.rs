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

pub mod epilogue;
pub mod fanout;
pub mod horizontal;
pub mod producer_consumer;
pub mod small_kernel;

/// `FUSION_V2_DEBUG` level: `0` unset, `1` set (aggregate per-pass
/// reject counters), `2` (value `"2"`: additionally print each
/// rejected seam).
pub(crate) fn debug_reject_level() -> u8 {
    match std::env::var("FUSION_V2_DEBUG") {
        Ok(v) if v == "2" => 2,
        Ok(_) => 1,
        Err(_) => 0,
    }
}

/// Extracts the bare variant name from a `Debug` rendering (drops any
/// payload), for aggregate reject counting.
pub(crate) fn variant_name(dbg: &impl std::fmt::Debug) -> String {
    let s = format!("{dbg:?}");
    s.split(['(', ' ', '{']).next().unwrap_or(&s).to_string()
}

/// Prints an aggregate reject-counter map for one pass invocation.
pub(crate) fn dump_rejects(pass: &str, rejects: &std::collections::BTreeMap<String, u64>) {
    if !rejects.is_empty() {
        eprintln!("[fusion-v2-debug] {pass} rejects: {rejects:?}");
    }
}

/// Per-site synthesis output: drafts in site order plus `(label, count)`
/// reject entries.
pub(crate) type SiteResult = (Vec<producer_consumer::CandidateDraft>, Vec<(String, u64)>);

/// Merged enumeration output: all drafts in sequential site order plus
/// the aggregated reject counters.
pub(crate) type EnumerateResult = (
    Vec<producer_consumer::CandidateDraft>,
    std::collections::BTreeMap<String, u64>,
);

/// Runs per-site candidate synthesis in parallel while preserving the
/// exact draft and reject ordering of the sequential loop: sites are
/// enumerated in deterministic order by the caller, mapped in parallel
/// (rayon's indexed `collect` keeps input order), and concatenated
/// sequentially. Draft order matters downstream — the driver's
/// insertion cap truncates in draft order.
pub(crate) fn par_enumerate<S, F>(sites: Vec<S>, synth: F) -> EnumerateResult
where
    S: Send,
    F: Fn(S) -> SiteResult + Send + Sync,
{
    use rayon::prelude::*;
    let per_site: Vec<_> = sites.into_par_iter().map(synth).collect();
    let mut drafts = Vec::new();
    let mut rejects = std::collections::BTreeMap::new();
    for (d, rs) in per_site {
        drafts.extend(d);
        for (k, n) in rs {
            *rejects.entry(k).or_default() += n;
        }
    }
    (drafts, rejects)
}
