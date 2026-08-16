//! `ConvertLayout` decomposition (Phase B B.6.3).
//!
//! Chooses the emission strategy for one `ConvertLayout` op given
//! `src.layout : phys_src → logical` and `dst.layout : phys_dst →
//! logical`. The receiver-driven composite is
//!
//! ```text
//!     C = src.layout⁺ ∘ dst.layout : phys_dst → phys_src
//! ```
//!
//! (`⁺` is [`LinearLayout::right_inverse`]; requires `src` surjective
//! onto the logical space, which every distributed layout is.)
//!
//! Two strategies land in Phase B initial:
//!
//! - **A (pure shuffle)** — paper §5.4 Intra-warp Data Exchange. Applicable iff `(C)_Wrp = I`
//!   ([`LinearLayout::is_warp_column_identity`]); rounds = the exact `__shfl_sync` count
//!   `gen_shuffle` will emit ([`crate::kernel_ir::shuffle_rounds`]): `slots` on the
//!   constant-sender-slot and invertible-lane paths, `slots · 2^|dirs|` on the multi-round pull
//!   path (paper page 8's `2^|R|` exchange).
//! - **B (pure bounce)** — paper §5.4 Optimal Swizzling. Always applicable. Store to a scratch
//!   shared buffer under [`shared_swizzle::optimal_shared_swizzle`]'s partition, sync, load.
//!
//! The landing pragma from the plan restricts Strategy C (hybrid) to
//! `T ∈ {∅, all-lane-bits} × {Pre}` — those two extremes are exactly A
//! and B, so hybrid enumeration is deferred. This module can grow the
//! full hybrid search later without changing callers.
//!
//! Every emitted `__shfl_sync` uses `0xFFFFFFFFu` (Gap 5) and sender-slot
//! resolvability is guaranteed by min-weight `T⁺` (Gap 4). Codegen wires
//! in the concrete permutations `r_src`, `r_dst` from the returned
//! strategy; this module scores and picks the shape.

use super::{
    layout_cost::ConversionCostModel,
    shared_swizzle::{optimal_shared_swizzle, Access, SharedLayout},
};
use crate::kernel_ir::LinearLayout;

/// The decomposition strategy for one `ConvertLayout` op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// `src.layout` and `dst.layout` agree as functions — no data
    /// movement.
    Copy,
    /// In-thread register-slot rename: `C`'s thread-input columns are
    /// identity and no thread output depends on slot inputs. One store
    /// per destination slot at a compile-time-constant source slot
    /// index; no `__shfl_sync`. This is the paper's Intra-thread Data
    /// Exchange (§5.4) special case of Strategy A with zero shuffle
    /// rounds.
    Slot,
    /// Paper §5.4 Intra-warp Data Exchange. `rounds` is the exact number
    /// of `__shfl_sync(0xFFFFFFFFu, …)` instructions `gen_shuffle` emits
    /// ([`crate::kernel_ir::shuffle_rounds`]).
    Shuffle { rounds: u64 },
    /// Paper §5.4 Optimal Swizzling. `swizzle` is the scratch buffer's
    /// `(vec, bank, idx)` partition from
    /// [`optimal_shared_swizzle`].
    Bounce { swizzle: SharedLayout },
}

/// Whether `c : phys → phys` reduces to an in-thread slot rename under
/// a `block`-thread CTA: every thread output is exactly its own thread
/// input, and slot-input columns don't produce any thread output.
/// Under this shape, gen_convert emits one register store per
/// destination slot with a compile-time-constant source slot index —
/// no `__shfl_sync`, no shared bounce.
fn is_slot_only(c: &LinearLayout, block: usize) -> bool {
    let p = c.phys_partition(block);
    let thread_mask = p.lane_mask() | p.warp_mask();
    if c.offset & thread_mask != 0 {
        return false;
    }
    let tb = p.thread_bits();
    // Thread-input columns must be identity (no slot-output component
    // either — Def 4.10 distributedness would otherwise force column
    // duplication with the identity in thread positions).
    for i in 0..tb {
        if c.bases[i] != 1u64 << i {
            return false;
        }
    }
    // Slot-input columns must not contribute to thread outputs.
    for i in tb..c.bases.len() {
        if c.bases[i] & thread_mask != 0 {
            return false;
        }
    }
    true
}

/// Scored strategy + cycle estimate. `cost` is loop-weighted per the
/// `loop_iters` argument to [`best_decomposition`].
#[derive(Clone, Debug)]
pub struct DecompositionResult {
    pub strategy: Strategy,
    pub cost: u64,
}

/// The receiver-driven composite `C = src⁺ ∘ dst` — `None` when `src`
/// isn't surjective onto the logical space (should not happen for
/// distributed layouts).
fn composite(src: &LinearLayout, dst: &LinearLayout, logical_bits: usize) -> Option<LinearLayout> {
    let src_inv = src.right_inverse(logical_bits)?;
    Some(src_inv.compose(dst))
}

/// Estimated bounce cost: one store + one sync + one load, scaled by
/// the max bank-conflict factor across the two accesses. `Full mask,
/// no divergence guards.
fn bounce_cost(
    cost: &ConversionCostModel,
    swizzle: &SharedLayout,
    src: &LinearLayout,
    dst: &LinearLayout,
    block: usize,
) -> u64 {
    let cf_src = super::shared_swizzle::conflict_factor(swizzle, src, block);
    let cf_dst = super::shared_swizzle::conflict_factor(swizzle, dst, block);
    let store = cost.shared_round_cost(1 + cf_src);
    let load = cost.shared_round_cost(1 + cf_dst);
    store.saturating_add(load).saturating_add(cost.sync_cost())
}

/// Pick the min-cost strategy for one `ConvertLayout`. `logical_bits`
/// is the codomain width of both layouts (they must map into the same
/// logical space). `loop_iters` weights this op's cost per B.6.1's
/// convention (`None` = symbolic).
pub fn best_decomposition(
    cost: &ConversionCostModel,
    src: &LinearLayout,
    dst: &LinearLayout,
    block: usize,
    logical_bits: usize,
    loop_iters: &[Option<u64>],
) -> DecompositionResult {
    // Free case: src ≡ dst as functions.
    if crate::kernel_ir::maps_agree(src, dst) {
        return DecompositionResult {
            strategy: Strategy::Copy,
            cost: 0,
        };
    }

    let weight = cost.loop_weight(loop_iters.iter().copied());

    // Strategy A / A-slot: pure shuffle (or slot rename when C's thread
    // block is identity) when C's warp column is identity.
    let mut candidates: Vec<(Strategy, u64)> = Vec::new();
    if let Some(c) = composite(src, dst, logical_bits) {
        if is_slot_only(&c, block) {
            candidates.push((Strategy::Slot, 0));
        } else if c.is_warp_column_identity(block) {
            let rounds = crate::kernel_ir::shuffle_rounds(&c, block);
            let raw = cost.shuffle_round_cost().saturating_mul(rounds);
            candidates.push((Strategy::Shuffle { rounds }, weight.saturating_mul(raw)));
        }
    }

    // Strategy B: pure bounce via optimal_shared_swizzle. Always
    // applicable — every distributed pair has a valid shared layout.
    let src_p = src.phys_partition(block);
    let dst_p = dst.phys_partition(block);
    let src_acc = Access {
        layout: src.clone(),
        loop_iters: Vec::new(),
        thread_bits: src_p.thread_bits(),
    };
    let dst_acc = Access {
        layout: dst.clone(),
        loop_iters: Vec::new(),
        thread_bits: dst_p.thread_bits(),
    };
    // element_bytes = 4 by default (BabyBear u32); callers with wider
    // elements pass through via choose_shared_layout, but here the
    // scratch is transient and the two-access swizzle is what we want.
    let swizzle = optimal_shared_swizzle(&src_acc, &dst_acc, logical_bits, 4);
    let bounce = bounce_cost(cost, &swizzle, src, dst, block);
    candidates.push((
        Strategy::Bounce {
            swizzle: swizzle.clone(),
        },
        weight.saturating_mul(bounce),
    ));

    candidates
        .into_iter()
        .min_by_key(|(_, c)| *c)
        .map(|(s, c)| DecompositionResult {
            strategy: s,
            cost: c,
        })
        .expect("at least one candidate is always pushed")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost() -> ConversionCostModel {
        ConversionCostModel::default()
    }

    #[test]
    fn slot_swap_picks_strategy_slot() {
        // src = identity(10) (kb=10, tb=8), dst permutes the two slot
        // bits. C = src^-1 ∘ dst has identity thread inputs and swapped
        // slot inputs — best_decomposition must return Slot, not
        // Shuffle.
        let src = LinearLayout::identity(10);
        let mut dst = LinearLayout::identity(10);
        dst.bases[8] = 1 << 9;
        dst.bases[9] = 1 << 8;
        let d = best_decomposition(&cost(), &src, &dst, 256, 10, &[]);
        assert_eq!(d.strategy, Strategy::Slot);
        assert_eq!(d.cost, 0);
    }

    #[test]
    fn identical_layouts_are_free_copy() {
        let l = LinearLayout::identity(9);
        let d = best_decomposition(&cost(), &l, &l, 256, 9, &[]);
        assert_eq!(d.strategy, Strategy::Copy);
        assert_eq!(d.cost, 0);
    }

    #[test]
    fn lane_shuffle_picks_strategy_a() {
        // src = identity(9); dst = lane rotation on the low 5 bits.
        // C = src^-1 ∘ dst = dst — warp columns of dst (bits 5..8) are
        // identity, so is_warp_column_identity holds. Strategy A wins.
        let src = LinearLayout::identity(9);
        let dst = LinearLayout {
            bases: vec![2, 4, 8, 16, 1, 32, 64, 128, 256],
            offset: 0,
        };
        let d = best_decomposition(&cost(), &src, &dst, 256, 9, &[]);
        match d.strategy {
            Strategy::Shuffle { rounds } => {
                assert!(rounds >= 1, "shuffle count must be at least 1");
                // A tight rotation moves data around within the warp;
                // rounds should be small (≤ 32 for lane-only cases).
                assert!(rounds <= 32, "expected small round count, got {rounds}");
            }
            other => panic!("expected shuffle, got {other:?}"),
        }
    }

    #[test]
    fn warp_crossing_picks_strategy_b() {
        // dst mixes warp bit 5 into lane bit 0 and vice versa. C fails
        // is_warp_column_identity → only Strategy B is a candidate.
        let src = LinearLayout::identity(9);
        let mut dst = LinearLayout::identity(9);
        dst.bases[0] = 1 << 5;
        dst.bases[5] = 1;
        let d = best_decomposition(&cost(), &src, &dst, 256, 9, &[]);
        assert!(
            matches!(d.strategy, Strategy::Bounce { .. }),
            "expected bounce, got {:?}",
            d.strategy
        );
    }

    #[test]
    fn shuffle_cheaper_than_bounce_for_lane_only_mix() {
        // Cost of Strategy A (a few shuffle rounds) should beat Strategy
        // B (shared bounce + sync) whenever both apply. Verify by
        // constructing a case where both do and asserting the winner.
        let src = LinearLayout::identity(9);
        let dst = LinearLayout {
            bases: vec![2, 4, 8, 16, 1, 32, 64, 128, 256],
            offset: 0,
        };
        let d = best_decomposition(&cost(), &src, &dst, 256, 9, &[]);
        assert!(matches!(d.strategy, Strategy::Shuffle { .. }));
        // Sanity: shuffle cost < shared_round + sync + shared_round.
        let baseline_bounce = 2 * cost().shared_round_cost(1) + cost().sync_cost();
        assert!(d.cost < baseline_bounce);
    }

    #[test]
    fn loop_iters_scale_cost() {
        let src = LinearLayout::identity(9);
        let dst = LinearLayout {
            bases: vec![2, 4, 8, 16, 1, 32, 64, 128, 256],
            offset: 0,
        };
        let d1 = best_decomposition(&cost(), &src, &dst, 256, 9, &[Some(1)]);
        let d100 = best_decomposition(&cost(), &src, &dst, 256, 9, &[Some(100)]);
        assert_eq!(d1.strategy, d100.strategy);
        assert_eq!(d100.cost, d1.cost * 100);
    }

    #[test]
    fn shuffle_rounds_is_slots_for_const_sender_slot() {
        // Identity composite: constant sender slot, one shuffle per dst
        // slot. kb=9, tb=8 → slots=2.
        let c = LinearLayout::identity(9);
        assert_eq!(crate::kernel_ir::shuffle_rounds(&c, 256), 2);
    }

    #[test]
    fn multi_round_composite_picks_shuffle_with_exact_count() {
        // Lane bit 0 feeds slot bit 5 and slot bit 5 feeds lane bit 0 —
        // the lane block [0,2,4,8,16] is singular and the sender slot
        // varies per lane, so the multi-round pull path applies:
        // slots=2, |dirs|=1 → 4 shuffles. Still far cheaper than a
        // bounce (2 shared rounds + sync).
        let src = LinearLayout::identity(6);
        let dst = LinearLayout {
            bases: vec![32, 2, 4, 8, 16, 1],
            offset: 0,
        };
        assert_eq!(crate::kernel_ir::shuffle_rounds(&dst, 32), 4);
        let d = best_decomposition(&cost(), &src, &dst, 32, 6, &[]);
        assert_eq!(d.strategy, Strategy::Shuffle { rounds: 4 });
        assert!(d.cost < 2 * cost().shared_round_cost(1) + cost().sync_cost());
    }
}
