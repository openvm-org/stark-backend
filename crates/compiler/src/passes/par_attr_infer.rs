//! Par-attribute inference (Phase B B.6.1).
//!
//! Given a par `P` with pre-inferred reads `{(B_i, g_i, L_{B_i})}`, pick
//! the par-attr layout `f_par : phys → logical_P` that minimizes the
//! loop-weighted sum of conversion costs across every read. The
//! candidate set enumerates the "make this read free" ideals
//! `g_i^+ ∘ L_{B_i}` (min-Hamming-weight per Gap 4) plus a caller-
//! supplied default; each candidate is filtered by
//! [`LinearLayout::is_distributed`] (Def 4.10) before scoring.
//!
//! Cost is coarse-tiered per the plan:
//!
//! | Tier | `C = L_B^+ ∘ (g ∘ f_par)` shape | Cost |
//! |---|---|---|
//! | 0 | warp block identity AND lane block identity | 0 |
//! | 1 | warp block identity, lane block non-trivial | shuffle rounds |
//! | 2 | warp block non-identity                     | shared bounce |
//!
//! Ties are broken toward the caller's `default` layout (Hamming
//! distance in the bases), which minimizes golden churn when the winner
//! is indistinguishable from the current pipeline's choice.
//!
//! Entry points are pure functions over layouts; wiring this into
//! `layout_infer` is Phase B B.1.

use super::layout_cost::ConversionCostModel;
use crate::kernel_ir::LinearLayout;

/// One read of a par to be scored during par-attr inference.
#[derive(Clone, Debug)]
pub struct ReadDemand {
    /// The read's index map from `logical_par → logical_B`. For linear
    /// (`IndexMap::Linear`) reads this is the map; other kinds are not
    /// handled by this pass (they route to shared through the fallback
    /// candidate).
    pub g: LinearLayout,
    /// The producer layout of the buffer being read: `phys → logical_B`
    /// for register-backed buffers (which share the CTA's phys space
    /// with `P`). Shared/global buffers use a different convention and
    /// aren't scored here; callers pass only register-backed reads.
    pub producer_layout: LinearLayout,
    /// `ceil_log2(B.len())` — the codomain width for the right inverses
    /// of `g` and `producer_layout`.
    pub logical_bits: usize,
    /// Iteration counts of enclosing loops; `None` for symbolic bounds.
    /// Multiplied into the score by [`ConversionCostModel::loop_weight`].
    pub loop_iters: Vec<Option<u64>>,
}

/// Coarse cost tier of a conversion map `C : phys → phys` (both phys
/// spaces the same width) under a `block`-thread CTA. Follows B.6.1:
/// tier 0 if warp and lane blocks are both identity, tier 1 if just
/// warp is identity, tier 2 otherwise.
fn cost_tier(cost: &ConversionCostModel, c: &LinearLayout, block: usize) -> u64 {
    let warp_ok = c.warp_block(block).is_identity();
    let lane_ok = c.lane_block(block).is_identity();
    if warp_ok && lane_ok {
        0
    } else if warp_ok {
        // Approx: one shuffle round per non-identity lane input bit.
        // (Formally 2^|R| by paper §5.4; this coarse per-tier estimate
        // is enough to rank candidates against tier 2.)
        let lane = c.lane_block(block);
        let non_identity = lane
            .bases
            .iter()
            .enumerate()
            .filter(|&(i, &b)| b != (1u64 << i))
            .count() as u64;
        cost.shuffle_round_cost()
            .saturating_mul(non_identity.max(1))
    } else {
        // Full bounce: one shared round + a sync.
        cost.shared_round_cost(1).saturating_add(cost.sync_cost())
    }
}

/// Loop-weighted score of a candidate `f_par` across every read. Higher
/// = worse; `argmin` picks the winner.
fn score(
    cost: &ConversionCostModel,
    f_par: &LinearLayout,
    reads: &[ReadDemand],
    block: usize,
) -> u64 {
    reads
        .iter()
        .map(|r| {
            let e = r.g.compose(f_par);
            let Some(l_inv) = r.producer_layout.right_inverse(r.logical_bits) else {
                // Non-surjective producer: this read can't be scored as
                // an in-place conversion. Weight as full tier-2 so
                // candidates preferring surjective producers win.
                return cost
                    .loop_weight(r.loop_iters.iter().copied())
                    .saturating_mul(cost.shared_round_cost(1).saturating_add(cost.sync_cost()));
            };
            let c = l_inv.compose(&e);
            let tier = cost_tier(cost, &c, block);
            cost.loop_weight(r.loop_iters.iter().copied())
                .saturating_mul(tier)
        })
        .fold(0u64, |acc, x| acc.saturating_add(x))
}

/// Hamming distance between two layouts' bases (bit-XOR then popcount,
/// per column, summed). Used as a tie-breaker toward `default`. Bases
/// of different length count the trailing extras as full-weight
/// (compared against zero).
fn hamming_distance(a: &LinearLayout, b: &LinearLayout) -> u32 {
    let n = a.bases.len().max(b.bases.len());
    let base_diff: u32 = (0..n)
        .map(|i| {
            let ai = a.bases.get(i).copied().unwrap_or(0);
            let bi = b.bases.get(i).copied().unwrap_or(0);
            (ai ^ bi).count_ones()
        })
        .sum();
    base_diff + (a.offset ^ b.offset).count_ones()
}

/// Ideal `f_par` for a single read: the min-Hamming-weight solution of
/// `g ∘ f_par = L_B` (making this read free). `None` when no such
/// solution exists (e.g. `g` isn't surjective onto `L_B`'s codomain).
fn ideal_for_read(r: &ReadDemand) -> Option<LinearLayout> {
    let g_inv = r.g.right_inverse(r.logical_bits)?;
    Some(g_inv.compose(&r.producer_layout))
}

/// Pick the best par-attr layout for a par with the given reads. The
/// candidate set is `{default} ∪ {ideal_for_read(r) for r in reads}`
/// filtered by [`LinearLayout::is_distributed`], scored by
/// [`score`], and tie-broken by Hamming distance toward `default`.
///
/// Both `default` and every candidate must have `phys_bits` bases; the
/// caller is responsible for computing `phys_bits =
/// ceil_log2(seq_size) + ceil_log2(block)` and passing the matching
/// default (Phase A used identity + zero-column padding — see
/// `layout_infer.rs`).
pub fn infer_par_attr(
    cost: &ConversionCostModel,
    reads: &[ReadDemand],
    default: LinearLayout,
    block: usize,
    phys_bits: usize,
) -> LinearLayout {
    debug_assert_eq!(default.bases.len(), phys_bits);
    let mut candidates: Vec<LinearLayout> = vec![default.clone()];
    for r in reads {
        let Some(f) = ideal_for_read(r) else { continue };
        if f.bases.len() != phys_bits {
            continue;
        }
        if !f.is_distributed() {
            continue;
        }
        if !candidates.iter().any(|c| c == &f) {
            candidates.push(f);
        }
    }
    candidates
        .into_iter()
        .min_by_key(|f| (score(cost, f, reads, block), hamming_distance(f, &default)))
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost() -> ConversionCostModel {
        ConversionCostModel::default()
    }

    #[test]
    fn cost_tier_zero_for_identity() {
        let c = LinearLayout::identity(9);
        assert_eq!(cost_tier(&cost(), &c, 256), 0);
    }

    #[test]
    fn cost_tier_one_for_lane_shuffle() {
        // Bit-0 and bit-1 lane swap: warp block identity, lane block
        // non-identity.
        let mut c = LinearLayout::identity(9);
        c.bases[0] = 2;
        c.bases[1] = 1;
        let t = cost_tier(&cost(), &c, 256);
        assert!(t > 0);
        assert!(t < cost().shared_round_cost(1));
    }

    #[test]
    fn cost_tier_two_for_warp_crossing() {
        // Warp bit 5 → lane bit 0, and lane bit 0 → bit 5. Warp block
        // moves off the diagonal.
        let mut c = LinearLayout::identity(9);
        c.bases[0] = 1 << 5;
        c.bases[5] = 1;
        let t = cost_tier(&cost(), &c, 256);
        assert!(t >= cost().shared_round_cost(1));
    }

    #[test]
    fn infer_returns_default_when_no_reads() {
        let d = LinearLayout::identity(9);
        let got = infer_par_attr(&cost(), &[], d.clone(), 256, 9);
        assert_eq!(got, d);
    }

    #[test]
    fn infer_prefers_identity_when_producer_matches() {
        // A single read of a producer with identity layout via g =
        // identity. The ideal f_par is the identity; the default is
        // identity; both score 0 → tie, tie-breaker picks identity.
        let reads = vec![ReadDemand {
            g: LinearLayout::identity(9),
            producer_layout: LinearLayout::identity(9),
            logical_bits: 9,
            loop_iters: Vec::new(),
        }];
        let d = LinearLayout::identity(9);
        let got = infer_par_attr(&cost(), &reads, d.clone(), 256, 9);
        assert_eq!(got, d);
    }

    #[test]
    fn infer_picks_ideal_over_costly_default() {
        // Read: g = identity, producer_layout is a lane-rotation. The
        // default (identity) f_par requires a lane shuffle → tier 1.
        // The ideal f_par = g^-1 ∘ L_B = L_B itself makes C = identity →
        // tier 0. Inference should pick L_B.
        let producer = LinearLayout {
            // rotate low 5 bits: bit i → bit (i+1) % 5.
            bases: vec![2, 4, 8, 16, 1, 32, 64, 128, 256],
            offset: 0,
        };
        let reads = vec![ReadDemand {
            g: LinearLayout::identity(9),
            producer_layout: producer.clone(),
            logical_bits: 9,
            loop_iters: Vec::new(),
        }];
        let d = LinearLayout::identity(9);
        let got = infer_par_attr(&cost(), &reads, d.clone(), 256, 9);
        // Score at ideal should be 0.
        let s_ideal = score(&cost(), &got, &reads, 256);
        assert_eq!(s_ideal, 0);
        assert_ne!(got, d);
        assert_eq!(got, producer);
    }

    #[test]
    fn infer_rejects_non_distributed_ideal() {
        // Producer has non-distributed layout (two columns share a bit).
        // The ideal would inherit this shape → filtered → fall back to
        // default.
        let producer = LinearLayout {
            bases: vec![1, 1, 4, 8, 16, 32, 64, 128, 256],
            offset: 0,
        };
        let reads = vec![ReadDemand {
            g: LinearLayout::identity(9),
            producer_layout: producer,
            logical_bits: 9,
            loop_iters: Vec::new(),
        }];
        let d = LinearLayout::identity(9);
        let got = infer_par_attr(&cost(), &reads, d.clone(), 256, 9);
        assert_eq!(got, d);
    }

    #[test]
    fn infer_weights_hot_reads_higher() {
        // Two reads on the SAME producer via different `g` maps. Read 0
        // is cold (loop_iters = [Some(1)]); read 1 is hot (Some(1024)).
        // Their ideals are different f_pars; the hot read should win.
        //
        // Setup: both producers = identity(9). Read 0 has g = identity;
        // ideal f_par_0 = identity. Read 1 has g = lane-rotation; ideal
        // f_par_1 = lane-rotation (so g ∘ f = identity = L_B). Under
        // f_par = identity, read 0 tier 0, read 1 tier 1 (rotate lanes).
        // Under f_par = lane-rotation, read 0 tier 1, read 1 tier 0.
        // Hot weight ⇒ pick f_par = lane-rotation.
        let lane_rot = LinearLayout {
            bases: vec![2, 4, 8, 16, 1, 32, 64, 128, 256],
            offset: 0,
        };
        let reads = vec![
            ReadDemand {
                g: LinearLayout::identity(9),
                producer_layout: LinearLayout::identity(9),
                logical_bits: 9,
                loop_iters: vec![Some(1)],
            },
            ReadDemand {
                g: lane_rot.clone(),
                producer_layout: LinearLayout::identity(9),
                logical_bits: 9,
                loop_iters: vec![Some(1024)],
            },
        ];
        let d = LinearLayout::identity(9);
        let got = infer_par_attr(&cost(), &reads, d, 256, 9);
        // Winner should be the ideal for read 1, i.e. g_1^-1 ∘ identity.
        // g_1 = lane_rot, so g_1^-1 ∘ id = lane_rot^-1.
        let want = lane_rot.right_inverse(9).unwrap();
        assert_eq!(got, want);
    }
}
