//! Global-memory transaction sampling.
//!
//! `detailed-fusion-plan-v2.md` §12.5. For each global access site
//! (`Par` load or store on a [`BufferKind::Input`] / [`BufferKind::Output`]
//! buffer), the estimator counts the number of distinct
//! `global_sector_bytes` sectors touched by a representative warp. The
//! sample is deterministic — seeded by
//! `(module_hash, kernel_index, par_node, access_index, model_version)` —
//! so two identical kernels produce byte-identical estimates.

use std::collections::{BTreeMap, HashSet};

use crate::{
    ir::{ScalarType, VarId},
    kernel_ir::{Access, BufferDecl, BufferKind, IndexMap, KBound, ParAttr},
    quast::{Expr, Quast, SymConst},
};

/// One access-site cost record (§12.5).
#[derive(Clone, Debug, Default)]
pub struct AccessEst {
    /// Total bytes requested by the warp (lanes * elem bytes).
    pub requested_bytes: u64,
    /// Bytes actually pulled through the DRAM interface (distinct sectors
    /// touched * `global_sector_bytes`).
    pub transaction_bytes: u64,
    /// Average sectors touched per warp-site, across the samples.
    pub avg_sectors_per_warp: f64,
    /// Exact dynamic number of warp-site executions (grid * dynamic loop
    /// bounds).
    pub dynamic_warp_accesses: u64,
}

/// Configuration passed to [`estimate_access`].
#[derive(Clone, Debug)]
pub struct AccessSampleCfg {
    pub warp_size: u32,
    pub global_sector_bytes: u32,
    pub warp_samples_per_par: u32,
    pub unknown_global_sectors_per_warp: u32,
    /// Seeding tuple: (module_hash, kernel_index, par_node, access_index,
    /// model_version). Combined via FNV-like mix to feed the LCG.
    pub seed: u64,
}

/// Estimates one access site given the enclosing bindings.
///
/// - `enclosing`: bindings for the enclosing loops / grid / par indices; used when the access map
///   depends on them.
/// - `par_bound`: the par's logical iteration count (may be symbolic).
/// - `par_attr`: block/thread-layout attribute (`None` before `layout_infer`).
/// - `dynamic_warp_multiplier`: exact number of times the enclosing pars/loops execute this
///   warp-site, in warps. For a grid-spanning par it is the number of warps that participate; for a
///   per-block par it is `blocks * warps_per_block * ...loop bounds`.
#[allow(clippy::too_many_arguments)]
pub fn estimate_access(
    access: &Access,
    buffer: &BufferDecl,
    enclosing: &BTreeMap<VarId, i64>,
    par_bound: &KBound,
    par_attr: Option<&ParAttr>,
    block_dim: u32,
    dynamic_warp_multiplier: u64,
    cfg: &AccessSampleCfg,
) -> AccessEst {
    // Bytes each thread requests.
    let elem_bytes = buffer.elem.size_bytes() as u64;
    let lanes = cfg.warp_size as u64;
    let requested_bytes_per_warp = lanes * elem_bytes;

    let avg_sectors = sample_sectors(
        access,
        buffer.elem,
        enclosing,
        par_bound,
        par_attr,
        block_dim,
        cfg,
    );

    let sector_bytes = cfg.global_sector_bytes as u64;
    let transaction_per_warp = (avg_sectors.ceil() as u64).max(1) * sector_bytes;
    let total_warp_accesses = dynamic_warp_multiplier.max(1);
    AccessEst {
        requested_bytes: requested_bytes_per_warp.saturating_mul(total_warp_accesses),
        transaction_bytes: transaction_per_warp.saturating_mul(total_warp_accesses),
        avg_sectors_per_warp: avg_sectors,
        dynamic_warp_accesses: total_warp_accesses,
    }
}

/// Samples `cfg.warp_samples_per_par` representative warps and returns the
/// average of their distinct-sector counts.
fn sample_sectors(
    access: &Access,
    elem: ScalarType,
    enclosing: &BTreeMap<VarId, i64>,
    par_bound: &KBound,
    par_attr: Option<&ParAttr>,
    block_dim: u32,
    cfg: &AccessSampleCfg,
) -> f64 {
    // Non-analyzable index expressions get the configured worst-case.
    match &access.index {
        IndexMap::Blackbox(_) => return cfg.unknown_global_sectors_per_warp as f64,
        IndexMap::SExpr(_) => return cfg.unknown_global_sectors_per_warp as f64,
        _ => {}
    }

    let elem_bytes = elem.size_bytes() as u64;
    let logical_bound = par_bound.as_const().unwrap_or(usize::MAX) as u64;
    // The par's own logical index appears in the map as the block operand
    // that binds to lane * seq. We identify it by the convention that a
    // par map's dominating symbol is the outer par index. We treat every
    // `Sym(VarId(i))` in the expression that is not present in `enclosing`
    // as the par's own logical index (`i`).
    let sample_count = cfg.warp_samples_per_par.max(1) as u64;
    let mut rng_state = cfg.seed;
    let mut total = 0f64;
    let seq_size = par_attr.map(|a| a.seq_size).unwrap_or(1) as u64;
    for sample_i in 0..sample_count {
        rng_state = mix64(rng_state ^ sample_i);
        // Pick a random warp base index within the domain.
        let warp_base = if logical_bound > cfg.warp_size as u64 {
            let warps_in_bound = (logical_bound / cfg.warp_size as u64).max(1);
            let warp_idx = rng_state % warps_in_bound;
            warp_idx * cfg.warp_size as u64
        } else {
            0
        };
        let mut sectors: HashSet<u64> = HashSet::new();
        for lane in 0..cfg.warp_size as u64 {
            for seq in 0..seq_size {
                // Physical index of this thread's element inside a par's
                // domain: `seq * block_dim + lane` offset by the warp base.
                let phys = seq * block_dim as u64 + lane;
                let logical = warp_base.saturating_add(phys);
                if logical >= logical_bound {
                    continue;
                }
                let byte_addr = eval_addr(access, elem_bytes, logical, enclosing);
                if let Some(byte_addr) = byte_addr {
                    sectors.insert(byte_addr / cfg.global_sector_bytes as u64);
                }
            }
        }
        total += sectors.len() as f64;
    }
    total / sample_count as f64
}

/// Evaluates the byte address of one thread's element for an
/// analyzable [`IndexMap`]. Returns `None` if the expression uses symbols
/// this sampler cannot bind.
fn eval_addr(
    access: &Access,
    elem_bytes: u64,
    logical: u64,
    enclosing: &BTreeMap<VarId, i64>,
) -> Option<u64> {
    let index = match &access.index {
        IndexMap::Linear(l) => l.apply(logical),
        IndexMap::Affine { expr, .. } => eval_quast(expr, logical, enclosing)?,
        IndexMap::SExpr(_) | IndexMap::Blackbox(_) => return None,
    };
    Some((index as u64).saturating_mul(elem_bytes))
}

fn eval_quast(expr: &Quast, par_idx: u64, enclosing: &BTreeMap<VarId, i64>) -> Option<u64> {
    // Any Sym not in `enclosing` binds to par_idx. This matches lower_to_kir's
    // convention that the par's own index is the innermost symbol under an
    // IndexMap::Affine.
    fn eval(
        e: &Quast,
        par_idx: u64,
        enclosing: &BTreeMap<VarId, i64>,
        par_var: Option<VarId>,
    ) -> Option<i64> {
        match e {
            Expr::Sym(v) => {
                if let Some(&val) = enclosing.get(v) {
                    Some(val)
                } else if Some(*v) == par_var {
                    Some(par_idx as i64)
                } else {
                    // Assume this is the par index if we couldn't detect it.
                    Some(par_idx as i64)
                }
            }
            Expr::Const(c) => Some(*c),
            Expr::Add(a, b) => {
                Some(eval(a, par_idx, enclosing, par_var)? + eval(b, par_idx, enclosing, par_var)?)
            }
            Expr::Mul(a, c) => Some(eval(a, par_idx, enclosing, par_var)? * *c),
            Expr::FloorDiv(a, c) => Some(eval(a, par_idx, enclosing, par_var)?.div_euclid(*c)),
            Expr::Neg(a) => Some(-eval(a, par_idx, enclosing, par_var)?),
        }
    }
    // Detect the par's own var: the syms of `expr` minus keys of `enclosing`.
    let mut syms = std::collections::BTreeSet::new();
    expr.syms(&mut syms);
    let par_var = syms.iter().copied().find(|v| !enclosing.contains_key(v));
    let raw = eval(expr, par_idx, enclosing, par_var)?;
    if raw < 0 {
        return None;
    }
    Some(raw as u64)
}

/// Deterministic 64-bit avalanche mixer (SplitMix64-style).
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

/// Whether the access reads/writes a globally-visible buffer, i.e. one that
/// actually crosses the DRAM interface (§12.5 only tracks these). Shared and
/// register buffers contribute instruction/dependency cost but no transaction
/// bytes.
pub fn is_global(buffer: &BufferDecl) -> bool {
    matches!(buffer.kind, BufferKind::Input(_) | BufferKind::Output(_))
}

/// Byte address helper mirroring the shape check we do above; exposed so
/// tests can build expectations against it.
pub fn linear_sector(byte_addr: u64, sector_bytes: u32) -> u64 {
    byte_addr / sector_bytes as u64
}

// Convenience re-exports for the SymConst path in future revisions.
#[allow(dead_code)]
fn sym_const_lit(c: &SymConst) -> Option<i64> {
    match c {
        SymConst::Lit(l) => Some(*l),
        SymConst::Sym(_) => None,
    }
}
