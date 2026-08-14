//! Cost model for layout conversions.
//!
//! Centralizes the hardware-specific cost constants used by Phase B's three
//! optimization passes: par-attr inference (`layout_infer::infer_par_attr`,
//! B.6.1), shared-layout selection (`layout_infer::choose_shared_layout`,
//! B.6.2), and `ConvertLayout` decomposition (`codegen::best_decomposition`,
//! B.6.3). The fusion-v2 estimator also consumes it once B.1 lands.
//!
//! Numbers are chosen for the current RTX 5090 / GB202 target. Wrap them in
//! one type so a hardware retune (or per-op measured overrides from a
//! benchmark run) is a single construction-site change.
//!
//! Per the plan (kernel_ir_gaps.md, Phase B design 2026-08-14):
//! - `SHUFFLE_COST = 3` cycles per `__shfl_sync` round.
//! - `SHARED_COST = 30` cycles per shared round without bank conflicts; a `k`-way conflict
//!   multiplies by `k` (30·k cycles).
//! - `SYNC_COST = 100` cycles per `__syncthreads` (order-of-magnitude estimate from B.6.3;
//!   divergence-dependent 10–100 in practice).
//! - `SYMBOLIC_WEIGHT ≈ 2^30` per unknown-bound loop; concrete loops multiply their iteration
//!   count. Compounds multiplicatively for nested symbolic loops.

/// Default cost constants for the current RTX 5090 target. See the module
/// header for provenance.
#[derive(Copy, Clone, Debug)]
pub struct ConversionCostModel {
    /// Cycles per `__shfl_sync` round (one round moves one element per lane).
    pub shuffle_round: u64,
    /// Cycles per bank-conflict-free shared-memory round (128 B transaction).
    pub shared_round: u64,
    /// Cycles per `__syncthreads` block-wide barrier.
    pub sync: u64,
    /// Weight assigned to a loop whose iteration count is not statically
    /// known. See [`Self::loop_weight`].
    pub symbolic_weight: u64,
}

impl ConversionCostModel {
    /// Baseline for GB202 / SM 10.x. `shuffle_round=3`, `shared_round=30`,
    /// `sync=100`, `symbolic_weight=2^30`.
    pub const fn default_rtx5090() -> Self {
        Self {
            shuffle_round: 3,
            shared_round: 30,
            sync: 100,
            symbolic_weight: 1 << 30,
        }
    }

    /// Cost of one `__shfl_sync` round.
    pub fn shuffle_round_cost(&self) -> u64 {
        self.shuffle_round
    }

    /// Cost of one shared-memory round with `bank_conflict_factor`-way
    /// conflict (1 = no conflict).
    pub fn shared_round_cost(&self, bank_conflict_factor: u64) -> u64 {
        self.shared_round
            .saturating_mul(bank_conflict_factor.max(1))
    }

    /// Cost of one `__syncthreads`.
    pub fn sync_cost(&self) -> u64 {
        self.sync
    }

    /// Multiplicative loop weight: the product of enclosing loop iteration
    /// counts. Each `None` counts as [`Self::symbolic_weight`]. Saturating
    /// multiplication so pathological deep nesting doesn't overflow.
    pub fn loop_weight<I>(&self, iters: I) -> u64
    where
        I: IntoIterator<Item = Option<u64>>,
    {
        iters.into_iter().fold(1u64, |acc, it| {
            acc.saturating_mul(it.unwrap_or(self.symbolic_weight))
        })
    }
}

impl Default for ConversionCostModel {
    fn default() -> Self {
        Self::default_rtx5090()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_constants() {
        let m = ConversionCostModel::default();
        assert_eq!(m.shuffle_round_cost(), 3);
        assert_eq!(m.shared_round_cost(1), 30);
        assert_eq!(m.shared_round_cost(4), 120);
        assert_eq!(m.sync_cost(), 100);
        assert_eq!(m.symbolic_weight, 1 << 30);
    }

    #[test]
    fn bank_conflict_factor_of_zero_treated_as_one() {
        // Cost is never below the conflict-free baseline; a `0` factor
        // (which callers shouldn't produce) rounds up rather than dropping
        // the term to zero.
        let m = ConversionCostModel::default();
        assert_eq!(m.shared_round_cost(0), 30);
    }

    #[test]
    fn loop_weight_multiplies_concrete_iters() {
        let m = ConversionCostModel::default();
        assert_eq!(m.loop_weight([Some(4u64), Some(8), Some(2)]), 64);
        // Empty nest = weight 1.
        assert_eq!(m.loop_weight(std::iter::empty()), 1);
    }

    #[test]
    fn loop_weight_symbolic_dominates_concrete() {
        let m = ConversionCostModel::default();
        let sym = m.symbolic_weight;
        assert_eq!(m.loop_weight([None]), sym);
        assert_eq!(m.loop_weight([Some(8u64), None]), 8 * sym);
        // Nested symbolic compounds multiplicatively but stays in u64.
        let two_sym = m.loop_weight([None, None]);
        assert_eq!(two_sym, sym.saturating_mul(sym));
        assert!(two_sym > sym);
    }

    #[test]
    fn loop_weight_saturates_rather_than_overflowing() {
        let m = ConversionCostModel::default();
        // 4 symbolic loops = 2^120 conceptually — saturates to u64::MAX
        // rather than wrapping.
        let w = m.loop_weight([None, None, None, None]);
        assert_eq!(w, u64::MAX);
    }
}
