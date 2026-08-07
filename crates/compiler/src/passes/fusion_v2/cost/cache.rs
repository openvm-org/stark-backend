//! Cost memoization keyed on module hash + param bindings.
//!
//! `detailed-fusion-plan-v2.md` §12.10. The estimator runs the same
//! normalized KIR many times during saturation (e.g. `(A+B)+C` and
//! `A+(B+C)` after canonicalization). Because the estimator is
//! deterministic in the module hash and the fixed
//! [`EstimateContext`](super::EstimateContext) for the run, this work is
//! cacheable.
//!
//! Key composition (§12.10):
//!
//! - `module_hash`: the alpha-normalized residual HIR hash used by
//!   [`ArtifactKey::module_hash`](super::ArtifactKey). Keying on the pre-lowering HIR hash lets a
//!   cache hit skip the KIR lowering pipeline entirely.
//! - `param_bindings_hash`: FNV hash of the sorted `(name, value)` pairs. Two candidates with the
//!   same normalized module but different concrete extents have distinct costs.

use std::collections::{BTreeMap, HashMap};

use thiserror::Error;

use crate::{
    ir,
    passes::fusion_v2::cost::{
        estimator::{estimate_kernel, EstimateContext, EstimatorConfig, KernelCostBreakdown},
        ArtifactContext, GraphNodeCost,
    },
    CompileError,
};

/// A cost-cache lookup key (§12.10).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct KirCostKey {
    /// The `ArtifactKey::module_hash` used elsewhere in the pipeline.
    pub module_hash: [u8; 32],
    /// Hash of the sorted [`EstimateContext::param_bindings`] map.
    pub param_bindings_hash: [u8; 32],
}

impl KirCostKey {
    pub fn new(module_hash: [u8; 32], param_bindings: &BTreeMap<String, i64>) -> Self {
        Self {
            module_hash,
            param_bindings_hash: hash_param_bindings(param_bindings),
        }
    }
}

/// FNV-1a hash of the sorted `(name, value)` pairs.
pub fn hash_param_bindings(bindings: &BTreeMap<String, i64>) -> [u8; 32] {
    // 32 bytes = 8 x u32 FNV; expand a 64-bit FNV by mixing it forward
    // through a SplitMix-style step, giving us a deterministic 32-byte
    // digest without an external dependency.
    let mut acc = 0xcbf29ce484222325u64;
    for (name, val) in bindings {
        for byte in name.as_bytes() {
            acc ^= *byte as u64;
            acc = acc.wrapping_mul(0x100000001b3);
        }
        acc ^= *val as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    let mut out = [0u8; 32];
    let mut state = acc;
    for chunk in out.chunks_mut(8) {
        state = splitmix(state);
        chunk.copy_from_slice(&state.to_le_bytes());
    }
    out
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

/// Estimator cache statistics.
#[derive(Copy, Clone, Debug, Default)]
pub struct CostManagerStats {
    pub hits: u64,
    pub misses: u64,
}

/// A cost error thrown when lowering or analysis fails on a candidate.
#[derive(Debug, Error)]
pub enum CostError {
    #[error(transparent)]
    Compile(#[from] CompileError),
}

/// Memoizing wrapper around [`estimate_kernel`] (§12.10).
///
/// One instance is created per `fuse_graph_v2` invocation. Enumeration,
/// boundary pruning, and multiple fusion patterns all share the same
/// cache: the same normalized module derived by different pattern paths
/// hits once and reuses the cached cost.
pub struct KernelCostManager {
    _cfg: EstimatorConfig,
    _artifact: ArtifactContext,
    graph_symbols: BTreeMap<ir::VarId, i64>,
    cache: HashMap<KirCostKey, GraphNodeCost>,
    breakdowns: HashMap<KirCostKey, KernelCostBreakdown>,
    stats: CostManagerStats,
    cfg_owned: EstimatorConfig,
    cycle_quantum: i64,
}

impl KernelCostManager {
    pub fn new(
        cfg: EstimatorConfig,
        artifact: ArtifactContext,
        graph_symbols: BTreeMap<ir::VarId, i64>,
        cycle_quantum: i64,
    ) -> Self {
        Self {
            _cfg: cfg.clone(),
            _artifact: artifact,
            graph_symbols,
            cache: HashMap::new(),
            breakdowns: HashMap::new(),
            stats: CostManagerStats::default(),
            cfg_owned: cfg,
            cycle_quantum: cycle_quantum.max(1),
        }
    }

    /// Fetches the cost for a candidate, running the estimator on a cache
    /// miss. `module_hash` is the caller-supplied
    /// [`ArtifactKey::module_hash`](super::ArtifactKey); passing it in
    /// (instead of recomputing from `module`) lets a cache hit avoid
    /// rehashing the HIR.
    pub fn cost_of(
        &mut self,
        module_hash: [u8; 32],
        module: &ir::Module,
        param_bindings: &BTreeMap<String, i64>,
    ) -> Result<GraphNodeCost, CostError> {
        let key = KirCostKey::new(module_hash, param_bindings);
        if let Some(&cost) = self.cache.get(&key) {
            self.stats.hits += 1;
            return Ok(cost);
        }
        let ctx = EstimateContext {
            graph_symbols: self.graph_symbols.clone(),
            param_bindings: param_bindings.clone(),
        };
        let (cost, breakdown) = estimate_kernel(
            module,
            module_hash,
            &ctx,
            &self.cfg_owned,
            self.cycle_quantum,
        )?;
        self.cache.insert(key, cost);
        self.breakdowns.insert(key, breakdown);
        self.stats.misses += 1;
        Ok(cost)
    }

    /// Snapshot of the cache stats.
    pub fn stats(&self) -> CostManagerStats {
        self.stats
    }

    /// Returns a reference to the cost breakdown for a previously-costed
    /// entry, useful for reporting.
    pub fn breakdown(&self, key: &KirCostKey) -> Option<&KernelCostBreakdown> {
        self.breakdowns.get(key)
    }
}
