//! Shared-memory layout selection (Phase B B.6.2).
//!
//! Ports the two-access swizzle algorithm from Triton's
//! `lib/Tools/GenericSwizzling.cpp` (paper §5.4 + Appendix 9.2) to our
//! [`LinearLayout`] representation. Given two `hardware → address`
//! accesses `A` and `B` of a shared buffer plus an element width, picks a
//! partition of the shared address bits into `(vec, bank, idx)` that
//! minimizes bank conflicts across the two accesses. For `N > 2` accesses
//! we enumerate `C(N, 2)` pairs (plus the row-major identity), run the
//! two-access algorithm on each candidate, score against **all** N
//! accesses, and pick the min-weighted-conflict candidate.
//!
//! Nothing here mutates KIR; the entry points are pure functions of
//! layouts + element bytes + output dimension. Consumers wire the
//! resulting `SharedLayout` into buffer layouts (B.1
//! `choose_shared_layout`) or scratch swizzles (B.6.3
//! `pick_bounce_swizzles`).
//!
//! Terminology, adapted from the paper's `LinearLayout` (hardware →
//! address) to our labelled physical space `(slot, lane, warp)`:
//!
//! - "register" columns (`reg` in the paper) = slot input columns, i.e. `bases[thread_bits..]`;
//!   constant per lane within a warp instruction.
//! - "thread" columns (`thr` in the paper) = lane + warp input columns, i.e.
//!   `bases[..thread_bits]`.
//! - `output_dim` = `log2(shared buffer size in elements)`.
//!
//! The affine offset of the source `LinearLayout` shifts every address by
//! a constant so it doesn't affect bank-conflict analysis; only the
//! linear-column subspaces matter.

use super::layout_cost::ConversionCostModel;
use crate::kernel_ir::{f2, LinearLayout};

/// Decomposition of the shared-memory address space bits produced by
/// [`optimal_shared_swizzle`]. Each field is a subspace basis of
/// `F₂^output_dim` in reduced row echelon form:
///
/// - `vec` — vectorization bits, constant within a warp transaction (one vector load/store per
///   warp).
/// - `bank` — bank-selecting bits; distinct lane values here mean distinct banks accessed in
///   parallel.
/// - `idx` — transaction-index bits, ideally constant across the warp so all 32 lanes hit the same
///   128 B window.
///
/// The three subspaces are jointly independent and span
/// `F₂^output_dim`: `rank(vec) + rank(bank) + rank(idx) = output_dim`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedLayout {
    pub vec: Vec<u64>,
    pub bank: Vec<u64>,
    pub idx: Vec<u64>,
    pub output_dim: usize,
}

/// An access to a shared buffer for [`choose_shared_layout`] scoring.
#[derive(Clone, Debug)]
pub struct Access {
    pub layout: LinearLayout,
    /// Iteration counts of the loops enclosing this access; `None` for
    /// symbolic bounds. Multiplied into the score by
    /// [`ConversionCostModel::loop_weight`].
    pub loop_iters: Vec<Option<u64>>,
    /// `thread_bits` = `min(log2(block), bases.len())`; the split between
    /// thread and slot input columns. Callers usually pass the same value
    /// used by [`LinearLayout::phys_partition`].
    pub thread_bits: usize,
}

impl Access {
    /// Slot ("register") columns of the layout: `bases[thread_bits..]`.
    pub fn reg_columns(&self) -> Vec<u64> {
        self.layout
            .bases
            .iter()
            .skip(self.thread_bits)
            .copied()
            .collect()
    }

    /// Thread ("thr") columns of the layout: `bases[..thread_bits]`.
    pub fn thr_columns(&self) -> Vec<u64> {
        self.layout
            .bases
            .iter()
            .take(self.thread_bits)
            .copied()
            .collect()
    }
}

/// Row-major fallback: `vec = ∅`, `bank = e_0..e_{bank_bits-1}`,
/// `idx = e_{bank_bits}..e_{output_dim-1}`. Always safe; used as a
/// tie-breaker candidate in [`choose_shared_layout`].
pub fn row_major_default(output_dim: usize, element_bytes: usize) -> SharedLayout {
    let bank_bits = default_bank_bits(0, element_bytes).min(output_dim);
    let bank: Vec<u64> = (0..bank_bits).map(|i| 1u64 << i).collect();
    let idx: Vec<u64> = (bank_bits..output_dim).map(|i| 1u64 << i).collect();
    SharedLayout {
        vec: Vec::new(),
        bank,
        idx,
        output_dim,
    }
}

/// The paper's `bank_bits = log2(128 / (2^v · element_bytes))`, clamped
/// so a vector wider than the transaction gives `bank_bits = 0`.
fn default_bank_bits(v: usize, element_bytes: usize) -> usize {
    let bytes_per_lane = element_bytes << v;
    if bytes_per_lane == 0 {
        return 0;
    }
    let banks = 128usize.checked_div(bytes_per_lane).unwrap_or(0);
    if banks == 0 {
        0
    } else {
        banks.trailing_zeros() as usize
    }
}

/// The pairwise two-access algorithm (paper §5.4 + Appendix 9.2). Both
/// `a` and `b` are `hardware → address` layouts; their `thread_bits`
/// splits (passed via [`Access`]) partition columns into `(reg, thr)`.
/// `output_dim` = `log2(buffer size in elements)`; `element_bytes` fixes
/// the transaction budget.
pub fn optimal_shared_swizzle(
    a: &Access,
    b: &Access,
    output_dim: usize,
    element_bytes: usize,
) -> SharedLayout {
    let a_reg = a.reg_columns();
    let a_thr = a.thr_columns();
    let b_reg = b.reg_columns();
    let b_thr = b.thr_columns();

    // 1. Maximal common vectorization: intersection of register subspaces.
    let vec = f2::intersection(&a_reg, &b_reg);
    let v = vec.len();
    let bank_bits = default_bank_bits(v, element_bytes).min(output_dim.saturating_sub(v));
    let idx_bits = output_dim.saturating_sub(v + bank_bits);

    // 2. Dangerous subspaces (bits that vary within a warp instruction).
    let ua = f2::sum(&vec, &a_thr);
    let ub = f2::sum(&vec, &b_thr);

    // 3. Common / exclusive split within (ua, ub).
    let common = f2::intersection(&ua, &ub);
    let ea = f2::complement_within(&common, &ua);
    let eb = f2::complement_within(&common, &ub);

    // 4. Safe directions: global complement + paired XORs.
    let uaub = f2::sum(&ua, &ub);
    let ambient = f2::standard_basis(output_dim);
    let global_complement = f2::complement_within(&uaub, &ambient);
    let paired: Vec<u64> = (0..ea.len().min(eb.len())).map(|i| ea[i] ^ eb[i]).collect();
    let safe = f2::sum(&global_complement, &paired);

    // 5. Idx selection: safe first, dangerous fallback from within ua.
    let idx = if safe.len() >= idx_bits {
        f2::take_independent(&safe, idx_bits)
    } else {
        let remaining = idx_bits - safe.len();
        let current = f2::sum(&vec, &safe);
        // Draw from ua vectors that lie outside `current`; take the first
        // `remaining` that raise the rank.
        let ua_candidates: Vec<u64> = ua
            .iter()
            .copied()
            .filter(|&c| !f2::contains(&current, c))
            .collect();
        let extra = f2::extend_with(&current, &ua_candidates, remaining);
        let mut all = safe.clone();
        all.extend(extra);
        f2::reduce(all)
    };

    // 6. Bank bits: complement of (vec ∪ idx) inside the ambient space.
    let bank_ambient = f2::sum(&vec, &idx);
    let bank = f2::complement_within(&bank_ambient, &ambient);

    SharedLayout {
        vec,
        bank,
        idx,
        output_dim,
    }
}

/// Bank-conflict multiplier for one access under a candidate
/// [`SharedLayout`]. Concretely: the rank of the lane input columns'
/// projection onto `sh`'s bank subspace. Bank-conflict-free ⇒ 0; k-way
/// conflict ⇒ `k − 1` (per the plan, so `sh` with `k = 1` scores 0).
///
/// Formally: `k = 2^(bank_bits − rank_in_bank(lane_cols))`, with
/// `rank_in_bank(cols) = rank(cols ∪ vec ∪ idx) − rank(vec ∪ idx)` — the
/// dimensions that lane cols contribute outside the vec+idx subspace,
/// which is (up to basis choice) exactly their bank projection's rank.
pub fn conflict_factor(sh: &SharedLayout, layout: &LinearLayout, block: usize) -> u64 {
    let p = layout.phys_partition(block);
    if p.lane == 0 {
        return 0;
    }
    let lane_cols: Vec<u64> = layout.bases.iter().take(p.lane).copied().collect();
    let mut vec_idx: Vec<u64> = sh.vec.clone();
    vec_idx.extend(sh.idx.iter().copied());
    let bg = f2::rank(&vec_idx);
    let mut all = vec_idx;
    all.extend(lane_cols);
    let full = f2::rank(&all);
    let rank_in_bank = full - bg;
    let bank_bits = sh.bank.len();
    let deficit = bank_bits.saturating_sub(rank_in_bank);
    (1u64 << deficit).saturating_sub(1)
}

/// Total loop-weighted conflict score of a candidate against every
/// access. Zero when the candidate is conflict-free for all accesses.
fn total_conflict(
    cost: &ConversionCostModel,
    sh: &SharedLayout,
    accesses: &[Access],
    block: usize,
) -> u64 {
    accesses
        .iter()
        .map(|acc| {
            let w = cost.loop_weight(acc.loop_iters.iter().copied());
            let cf = conflict_factor(sh, &acc.layout, block);
            w.saturating_mul(cf)
        })
        .fold(0u64, |acc, x| acc.saturating_add(x))
}

/// Multi-access shared layout selection (B.6.2 last section). For
/// `accesses.len() ≥ 2` enumerate every unordered pair, run
/// [`optimal_shared_swizzle`] on it, and pick the resulting
/// [`SharedLayout`] that minimizes [`total_conflict`] across every
/// access. Falls back to [`row_major_default`] when the sweep can't
/// beat it. `block` is the CTA size (needed to partition each access's
/// lane / warp / slot columns).
pub fn choose_shared_layout(
    cost: &ConversionCostModel,
    accesses: &[Access],
    output_dim: usize,
    element_bytes: usize,
    block: usize,
) -> SharedLayout {
    let mut best = row_major_default(output_dim, element_bytes);
    let mut best_score = total_conflict(cost, &best, accesses, block);
    for i in 0..accesses.len() {
        for j in (i + 1)..accesses.len() {
            let cand =
                optimal_shared_swizzle(&accesses[i], &accesses[j], output_dim, element_bytes);
            let score = total_conflict(cost, &cand, accesses, block);
            if score < best_score {
                best_score = score;
                best = cand;
            }
        }
    }
    best
}

/// Project a [`SharedLayout`] partition to a [`LinearLayout`] mapping
/// buffer-logical index → phys address in the shared buffer.
///
/// The resulting `LinearLayout` has `output_dim` bases; each basis
/// vector is one of the partition's subspace bases, in the order
/// `[vec, bank, idx]`:
///
/// - The first `|vec|` logical bits map to the vec subspace — the "vectorization" bits that are
///   constant within a warp's transaction.
/// - The next `|bank|` bits map to the bank subspace — the bits that distinguish across the 32
///   banks in a warp transaction.
/// - The remaining `|idx|` bits map to the idx (transaction-index) subspace.
///
/// Interpretation: logical bit `j` is stored at the phys address given
/// by `bases[j]` (XOR-affine). For a canonically-chosen partition, the
/// resulting phys addresses are bank-conflict-minimizing for the
/// accesses that produced this `SharedLayout` via
/// [`choose_shared_layout`] / [`optimal_shared_swizzle`].
///
/// Callers wire this into buffer-layout assignment (Phase B.1 (b)) and
/// scratch-swizzle picking (Phase B.4-full's Bounce path).
pub fn to_linear_layout(sh: &SharedLayout) -> LinearLayout {
    let mut bases = Vec::with_capacity(sh.output_dim);
    bases.extend_from_slice(&sh.vec);
    bases.extend_from_slice(&sh.bank);
    bases.extend_from_slice(&sh.idx);
    LinearLayout { bases, offset: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn access_identity(bits: usize, block: usize) -> Access {
        Access {
            layout: LinearLayout::identity(bits),
            loop_iters: Vec::new(),
            thread_bits: bits.min(block.trailing_zeros() as usize),
        }
    }

    #[test]
    fn identity_access_row_major_is_conflict_free() {
        // A single 32-thread access to a 5-bit shared buffer with 4-byte
        // elements: row-major default should already be conflict-free.
        let acc = access_identity(5, 32);
        let sh = row_major_default(5, 4);
        assert_eq!(conflict_factor(&sh, &acc.layout, 32), 0);
    }

    #[test]
    fn broadcast_access_row_major_has_max_conflict() {
        // Access with all-zero lane columns: every lane hits address 0 →
        // 32-way conflict. Under row-major with bank_bits = 5, deficit
        // is 5, so k = 32 and k-1 = 31.
        let acc = Access {
            layout: LinearLayout {
                bases: vec![0, 0, 0, 0, 0],
                offset: 0,
            },
            loop_iters: Vec::new(),
            thread_bits: 5,
        };
        let sh = row_major_default(5, 4);
        assert_eq!(conflict_factor(&sh, &acc.layout, 32), 31);
    }

    #[test]
    fn default_bank_bits_matches_paper() {
        // element_bytes=4, no vec: 128 / (1 · 4) = 32 banks → 5 bits.
        assert_eq!(default_bank_bits(0, 4), 5);
        // element_bytes=4, v=1: 128 / (2 · 4) = 16 banks → 4 bits.
        assert_eq!(default_bank_bits(1, 4), 4);
        // element_bytes=16 (FpExt), no vec: 128 / (1 · 16) = 8 → 3 bits.
        assert_eq!(default_bank_bits(0, 16), 3);
        // Vector wider than the transaction: 0 bank bits.
        assert_eq!(default_bank_bits(6, 4), 0);
    }

    #[test]
    fn optimal_swizzle_transpose_avoids_conflict() {
        // Access A: identity on 6 bits, 5 threads bits, 1 slot bit.
        // Access B: swap the two halves — lane bits go to high output
        // positions, slot bit stays high. This is the classic
        // row-vs-column transpose pair where a swizzle helps.
        //
        //   A.bases = [1, 2, 4, 8, 16, 32]  (identity)
        //   B: bits 0..5 (lane) mapped to bits 1..6, bit 5 (slot) mapped
        //   to bit 0.
        let a = Access {
            layout: LinearLayout::identity(6),
            loop_iters: Vec::new(),
            thread_bits: 5,
        };
        let b = Access {
            layout: LinearLayout {
                bases: vec![2, 4, 8, 16, 32, 1],
                offset: 0,
            },
            loop_iters: Vec::new(),
            thread_bits: 5,
        };
        let sh = optimal_shared_swizzle(&a, &b, 6, 4);
        // Rank invariant: dims partition output_dim.
        assert_eq!(sh.vec.len() + sh.bank.len() + sh.idx.len(), 6);
        // The chosen SharedLayout should score at least as well as
        // row-major for both accesses. Here A is trivially conflict-free
        // under row-major; B under row-major hits idx bits with its lane
        // columns and gets penalized — the swizzle should reduce that.
        let rm = row_major_default(6, 4);
        let sh_total = conflict_factor(&sh, &a.layout, 32) + conflict_factor(&sh, &b.layout, 32);
        let rm_total = conflict_factor(&rm, &a.layout, 32) + conflict_factor(&rm, &b.layout, 32);
        assert!(
            sh_total <= rm_total,
            "swizzle {sh_total} should not exceed row-major {rm_total}"
        );
    }

    #[test]
    fn choose_shared_layout_prefers_conflict_free_pair() {
        // Two accesses, both identity-shaped, but one is offset into the
        // high bits so it doesn't collide with row-major on its own.
        // choose_shared_layout should at least match row-major here (both
        // are conflict-free).
        let cost = ConversionCostModel::default();
        let a = access_identity(6, 32);
        let b = access_identity(6, 32);
        let sh = choose_shared_layout(&cost, &[a.clone(), b.clone()], 6, 4, 32);
        assert_eq!(conflict_factor(&sh, &a.layout, 32), 0);
        assert_eq!(conflict_factor(&sh, &b.layout, 32), 0);
    }

    #[test]
    fn choose_shared_layout_falls_back_to_row_major_when_no_pair_helps() {
        // A single access — no pair to feed optimal_shared_swizzle.
        // Should return row-major default.
        let cost = ConversionCostModel::default();
        let a = access_identity(5, 32);
        let sh = choose_shared_layout(&cost, &[a], 5, 4, 32);
        assert_eq!(sh, row_major_default(5, 4));
    }

    #[test]
    fn to_linear_layout_of_row_major_is_identity_permutation() {
        // Row-major default with 5 output bits, 4-byte elements: vec=∅,
        // bank=e_0..e_4, idx=∅. Projection concatenates as [vec | bank |
        // idx] = [e_0..e_4] = identity(5).
        let sh = row_major_default(5, 4);
        let ll = to_linear_layout(&sh);
        assert_eq!(ll.bases.len(), 5);
        assert!(ll.is_identity());
    }

    #[test]
    fn to_linear_layout_preserves_output_dim() {
        // For any SharedLayout, the projection has `output_dim` bases.
        let sh = row_major_default(9, 4);
        let ll = to_linear_layout(&sh);
        assert_eq!(ll.bases.len(), 9);
        assert_eq!(ll.offset, 0);
    }

    #[test]
    fn to_linear_layout_of_swizzle_is_bijection() {
        // Feed the transpose case to optimal_shared_swizzle and verify the
        // projected LinearLayout is a bijection (bases span F₂^output_dim,
        // so `inverse()` succeeds).
        let a = Access {
            layout: LinearLayout::identity(6),
            loop_iters: Vec::new(),
            thread_bits: 5,
        };
        let b = Access {
            layout: LinearLayout {
                bases: vec![2, 4, 8, 16, 32, 1],
                offset: 0,
            },
            loop_iters: Vec::new(),
            thread_bits: 5,
        };
        let sh = optimal_shared_swizzle(&a, &b, 6, 4);
        let ll = to_linear_layout(&sh);
        assert_eq!(ll.bases.len(), 6);
        // Bijection: `.inverse()` succeeds iff the bases span F₂^6.
        assert!(
            ll.inverse().is_some(),
            "swizzle projection should be a bijection: bases = {:?}",
            ll.bases
        );
    }

    #[test]
    fn to_linear_layout_bases_come_from_partition_in_order() {
        // Construct a SharedLayout with distinctive vec/bank/idx bases and
        // verify the concatenation order in the projection.
        let sh = SharedLayout {
            vec: vec![0b0001],
            bank: vec![0b0010, 0b0100],
            idx: vec![0b1000],
            output_dim: 4,
        };
        let ll = to_linear_layout(&sh);
        assert_eq!(ll.bases, vec![0b0001, 0b0010, 0b0100, 0b1000]);
    }
}
