//! KernelIR: the lower-level, imperative representation.
//!
//! A kernel is a single SSA IR: operations ([`SSAOp`]) reference values
//! ([`SSARes`]) and may own a nested region ([`SSABlock`]). The kernel body
//! is the [`Grid`]'s block, whose first operand is the grid index
//! (`blockIdx.x`); it holds the memory- and scheduling-aware ops —
//! sequential [`SSAOpCode::Loop`]s, buffer materializations
//! ([`SSAOpCode::Alloc`]) and [`SSAOpCode::Par`] blocks.
//!
//! A par is a *primitive compute block*: a block of pure single-threaded
//! math on registers that neither touches memory nor synchronizes. All
//! memory traffic of a par is declared up front as `reads` / `writes`
//! accesses; the loaded values enter its block as operands and the stored
//! values leave it as yields. Inside a par, only scalar ops appear; an
//! MLIR-style `scf.for` reuses [`SSAOpCode::Loop`] with the loop-carried
//! values threaded through operands/yields/results.
//!
//! Values form a single kernel-wide id space, so ops may reference any
//! dominating value: the grid index and enclosing loop induction variables.
//! Region ops are closed over their operands: every value a region op or its
//! block uses from an enclosing scope is listed in the op's operands.
//! Nothing here is self-referential: the [`Kernel`] owns a flat arena of
//! ops, and every block stores typed ids into it.
//!
//! Layout attributes ([`ParAttr`] on [`SSAOpCode::Par`],
//! [`BufferDecl::layout`]) are `None` right after lowering and are filled in
//! by the `layout_infer` pass. Synchronization is not represented: codegen
//! derives barriers from the pars' declared reads and writes.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
    ir::{BinOp, ScalarType, SizeExpr, VarId},
    quast::{Quast, SymConst},
};

/// A kernel-level extent: either concrete or an expression over the
/// program's runtime parameters ([`KirProgram::params`]). Only grid
/// bounds and grid-spanning par bounds may be symbolic; everything inner
/// (loops, tile shapes, `ParAttr::seq_size`) is concrete by construction
/// (guaranteed by monomorphization).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum KBound {
    Const(usize),
    Expr(SizeExpr),
}

impl KBound {
    pub fn as_const(&self) -> Option<usize> {
        match self {
            KBound::Const(c) => Some(*c),
            KBound::Expr(e) => e.as_const().map(|c| c as usize),
        }
    }

    /// The bound as a size expression (a literal when concrete).
    pub fn to_expr(&self) -> SizeExpr {
        match self {
            KBound::Const(c) => (*c).into(),
            KBound::Expr(e) => e.clone(),
        }
    }
}

impl From<usize> for KBound {
    fn from(c: usize) -> Self {
        KBound::Const(c)
    }
}

impl std::fmt::Display for KBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KBound::Const(c) => write!(f, "{c}"),
            KBound::Expr(e) => write!(f, "{e}"),
        }
    }
}

/// SSA value inside one kernel (a single kernel-wide id space).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SSARes(pub u32);

/// Id of an [`SSAOp`] in the kernel's op arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SSANode(pub u32);

/// Id of a [`BufferDecl`] in the program's buffer table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BufId(pub u32);

/// An XOR-affine map `T: Z_2^k -> Z_2^k`, `T(x) = M(x) ^ offset`, with the
/// linear part `M` represented by its images of the basis vectors:
/// `bases[i] = M(1 << i)`. `M(x)` is the XOR of the bases selected by the
/// set bits of `x`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearLayout {
    pub bases: Vec<u64>,
    pub offset: u64,
}

impl LinearLayout {
    pub fn identity(bits: usize) -> Self {
        Self {
            bases: (0..bits).map(|i| 1u64 << i).collect(),
            offset: 0,
        }
    }

    /// The linear part `M(x)`, without the offset.
    pub fn linear_apply(&self, x: u64) -> u64 {
        self.bases
            .iter()
            .enumerate()
            .filter(|&(i, _)| (x >> i) & 1 == 1)
            .fold(0, |acc, (_, &b)| acc ^ b)
    }

    pub fn apply(&self, x: u64) -> u64 {
        self.linear_apply(x) ^ self.offset
    }

    pub fn is_identity(&self) -> bool {
        self.offset == 0 && self.bases.iter().enumerate().all(|(i, &b)| b == 1u64 << i)
    }

    /// `self ∘ other`: `other` applied first.
    pub fn compose(&self, other: &LinearLayout) -> LinearLayout {
        LinearLayout {
            bases: other.bases.iter().map(|&b| self.linear_apply(b)).collect(),
            offset: self.apply(other.offset),
        }
    }

    /// The inverse map, or `None` if the map is not a bijection of
    /// `Z_2^k` (`k = bases.len()`). Gauss-Jordan elimination on the
    /// columns; the same column operations that reduce the matrix to the
    /// identity turn the identity into the inverse. `y = M(x) ^ c` inverts
    /// to `x = M^-1(y) ^ M^-1(c)`.
    pub fn inverse(&self) -> Option<LinearLayout> {
        let k = self.bases.len();
        let mut m = self.bases.clone();
        let mut inv: Vec<u64> = (0..k).map(|i| 1u64 << i).collect();
        for r in 0..k {
            let pivot = (r..k).find(|&c| (m[c] >> r) & 1 == 1)?;
            m.swap(r, pivot);
            inv.swap(r, pivot);
            for c in 0..k {
                if c != r && (m[c] >> r) & 1 == 1 {
                    m[c] ^= m[r];
                    inv[c] ^= inv[r];
                }
            }
        }
        let mut out = LinearLayout {
            bases: inv,
            offset: 0,
        };
        out.offset = out.linear_apply(self.offset);
        Some(out)
    }

    /// A right inverse `T⁺` covering `out_bits` output bits, or `None` if
    /// the linear part isn't surjective onto `F_2^{out_bits}` (some
    /// output bit isn't in the column span). For any `y < 2^out_bits`,
    /// `self.apply(result.apply(y)) == y`.
    ///
    /// Semantics: `self` is XOR-affine `T(x) = M(x) ^ offset`; the
    /// returned layout represents `T⁺(y) = M⁺(y ^ offset)`, so its
    /// `bases.len() == out_bits` and output width is `self.bases.len()`
    /// (input-side width of the original). Solutions to `M x = e_j`
    /// aren't unique when the column span has slack (zero columns or
    /// dependent columns); this picks the min-Hamming-weight preimage
    /// by using only the pivot columns (slack columns contribute zero
    /// to the preimage). That's the "canonical replica" convention —
    /// replicated inputs (zero columns of `M`) resolve to the pivot's
    /// physical index rather than being scattered arbitrarily.
    pub fn right_inverse(&self, out_bits: usize) -> Option<LinearLayout> {
        let k = self.bases.len();
        // Column-space Gaussian elimination on the linear part. `m[c]`
        // is column c of the current (post-elimination) matrix; `inv[c]`
        // tracks the column op history so `M ∘ inv[c] = m[c]` (i.e.,
        // `inv[c]` is the pre-image in the original input space of the
        // current column c).
        let mut m: Vec<u64> = self.bases.clone();
        let mut inv: Vec<u64> = (0..k).map(|i| 1u64 << i).collect();
        // Pivot column chosen for each output bit r. `None` return ⇒ some
        // bit has no pivot ⇒ not surjective.
        let mut pivot_for: Vec<usize> = Vec::with_capacity(out_bits);
        // Columns already used as pivots.
        let mut used: Vec<bool> = vec![false; k];
        for r in 0..out_bits {
            let pivot = (0..k).find(|&c| !used[c] && (m[c] >> r) & 1 == 1)?;
            used[pivot] = true;
            pivot_for.push(pivot);
            // Zero bit r out of all other columns using the pivot. The
            // pivot column itself still carries bits other than r; those
            // are cleaned by *later* rounds (each bit r' is eliminated
            // from every column except its own pivot), so the preimages
            // must be read off only after the elimination completes —
            // snapshotting `inv[pivot]` here would invert the partially
            // reduced column, not `e_r`.
            let piv_m = m[pivot];
            let piv_inv = inv[pivot];
            for c in 0..k {
                if c != pivot && (m[c] >> r) & 1 == 1 {
                    m[c] ^= piv_m;
                    inv[c] ^= piv_inv;
                }
            }
        }
        let new_bases: Vec<u64> = pivot_for.into_iter().map(|p| inv[p]).collect();
        // Affine: `T⁺(y) = M⁺(y ^ offset)`. Represent as a `LinearLayout`
        // with linear part `M⁺` and offset `M⁺(self.offset)`, so
        // `result.apply(y) = M⁺(y) ^ M⁺(self.offset) = M⁺(y ^ self.offset)`.
        let tmp = LinearLayout {
            bases: new_bases.clone(),
            offset: 0,
        };
        let new_offset = tmp.linear_apply(self.offset);
        Some(LinearLayout {
            bases: new_bases,
            offset: new_offset,
        })
    }
}

/// The `(slot, lane, warp)` partition of a `k`-bit physical index under a
/// CTA of `block` threads. `k = LinearLayout::bases.len()`, `tb = min(k,
/// log2(block))`, `lane = min(tb, 5)`, `warp = tb - lane`, `slot = k - tb`.
/// Bit positions are: lanes `0..lane`, warps `lane..lane+warp`, slots
/// `lane+warp..k`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PhysPartition {
    pub slot: usize,
    pub lane: usize,
    pub warp: usize,
}

impl PhysPartition {
    /// Thread bits total (lane + warp).
    pub fn thread_bits(&self) -> usize {
        self.lane + self.warp
    }

    /// All bits (slot + thread).
    pub fn total_bits(&self) -> usize {
        self.slot + self.lane + self.warp
    }

    /// Bit mask covering lane positions in an output `u64`.
    pub fn lane_mask(&self) -> u64 {
        if self.lane == 0 {
            0
        } else {
            (1u64 << self.lane) - 1
        }
    }

    /// Bit mask covering warp positions in an output `u64`.
    pub fn warp_mask(&self) -> u64 {
        if self.warp == 0 {
            0
        } else {
            ((1u64 << self.warp) - 1) << self.lane
        }
    }

    /// Bit mask covering slot positions in an output `u64`.
    pub fn slot_mask(&self) -> u64 {
        if self.slot == 0 {
            0
        } else {
            let tb = self.thread_bits();
            ((1u64 << self.slot) - 1) << tb
        }
    }
}

impl LinearLayout {
    /// The `(slot, lane, warp)` partition for a `block`-thread CTA.
    /// `block` must be a power of two.
    pub fn phys_partition(&self, block: usize) -> PhysPartition {
        debug_assert!(block.is_power_of_two(), "block must be pow2");
        let k = self.bases.len();
        let tb = k.min(block.trailing_zeros() as usize);
        let lane = tb.min(5);
        let warp = tb - lane;
        let slot = k - tb;
        PhysPartition { slot, lane, warp }
    }

    /// The diagonal lane sub-block `C_ll`: lane inputs projected onto
    /// lane outputs, as a `lane × lane` `LinearLayout` with offset 0.
    /// Used by `cost_tier` (B.6.1) — tier 0 requires `is_identity()`.
    pub fn lane_block(&self, block: usize) -> LinearLayout {
        let p = self.phys_partition(block);
        let mask = p.lane_mask();
        LinearLayout {
            bases: (0..p.lane).map(|i| self.bases[i] & mask).collect(),
            offset: 0,
        }
    }

    /// The diagonal warp sub-block `C_ww`: warp inputs projected onto
    /// warp outputs, shifted down so bit 0 is the first warp position.
    /// `is_identity()` on this equals the paper's `(C)_Wrp = I` **only
    /// when** the layout is distributed (Def 4.10) so cross-terms are
    /// forced to zero; use [`Self::is_warp_column_identity`] for the
    /// unconditional paper condition.
    pub fn warp_block(&self, block: usize) -> LinearLayout {
        let p = self.phys_partition(block);
        let mask = p.warp_mask();
        LinearLayout {
            bases: (0..p.warp)
                .map(|i| (self.bases[p.lane + i] & mask) >> p.lane)
                .collect(),
            offset: 0,
        }
    }

    /// The diagonal slot sub-block `C_ss`: slot inputs projected onto
    /// slot outputs, shifted down so bit 0 is the first slot position.
    pub fn slot_block(&self, block: usize) -> LinearLayout {
        let p = self.phys_partition(block);
        let mask = p.slot_mask();
        let tb = p.thread_bits();
        LinearLayout {
            bases: (0..p.slot)
                .map(|i| (self.bases[tb + i] & mask) >> tb)
                .collect(),
            offset: 0,
        }
    }

    /// Paper §5.4 condition `(C)_Wrp = I`: the warp-input column block
    /// of this layout is the identity embedding (`bases[lane + i] ==
    /// 1 << (lane + i)` for `i in 0..warp`), and the offset has no warp
    /// bits. Equivalent to `C_ww = I ∧ C_sw = 0 ∧ C_lw = 0`. Also checks
    /// that non-warp inputs contribute no warp output (`C_ws = 0 ∧
    /// C_wl = 0`) — under Def 4.10 distributedness the two follow from
    /// each other, but this defensive check keeps a non-distributed
    /// layout from silently routing through the pure-shuffle path.
    /// This is the pure-shuffle applicability condition in Strategy A of
    /// `best_decomposition` (B.6.3).
    pub fn is_warp_column_identity(&self, block: usize) -> bool {
        let p = self.phys_partition(block);
        let warp_mask = p.warp_mask();
        if self.offset & warp_mask != 0 {
            return false;
        }
        for (i, &b) in self.bases.iter().enumerate() {
            let want = if (p.lane..p.lane + p.warp).contains(&i) {
                1u64 << i
            } else {
                0
            };
            // For warp-input columns: full column must be identity
            // (`b == want`, i.e. no slot/lane output either — matches
            // classify_convert's slot_only / warp_fixed pair). For
            // non-warp inputs: no warp-output bits.
            if (p.lane..p.lane + p.warp).contains(&i) {
                if b != want {
                    return false;
                }
            } else if b & warp_mask != 0 {
                return false;
            }
        }
        true
    }

    /// Effective output width: `1 + msb(offset | max(bases))`. Zero if
    /// the layout is the zero map.
    pub fn output_bits(&self) -> usize {
        let m = self
            .bases
            .iter()
            .copied()
            .fold(self.offset, |acc, b| acc | b);
        if m == 0 {
            0
        } else {
            64 - m.leading_zeros() as usize
        }
    }

    /// Def 4.10 (distributed layout): every column has at most one
    /// non-zero bit, and all non-zero columns are distinct. Under this
    /// invariant `apply` is a permutation of `2^k` values onto its image,
    /// with the zero columns marking replicated (broadcast) input bits.
    /// The paper's shuffle construction and Strategy A of
    /// `best_decomposition` rely on this shape.
    pub fn is_distributed(&self) -> bool {
        let mut seen: u64 = 0;
        for &b in &self.bases {
            if b == 0 {
                continue;
            }
            // Exactly one bit set.
            if b & (b - 1) != 0 {
                return false;
            }
            // Distinct from every prior non-zero column.
            if seen & b != 0 {
                return false;
            }
            seen |= b;
        }
        true
    }
}

/// F₂ subspace primitives. Subspaces are represented as `Vec<u64>` where
/// each `u64` is a bit-vector in `F₂^d` (LSB = coordinate 0). The ambient
/// dim `d` is implicit — callers pass it when constructing standard-basis
/// sets or complements. Bases returned by these helpers are in **reduced
/// row echelon form**: pivots at strictly increasing bit positions, each
/// pivot bit present in exactly one basis vector.
pub mod f2 {
    /// Reduce a set of vectors to a canonical RREF basis. Drops zero
    /// vectors; the returned list is sorted by ascending pivot bit.
    pub fn reduce(mut vecs: Vec<u64>) -> Vec<u64> {
        let mut basis: Vec<u64> = Vec::new();
        for mut v in vecs.drain(..) {
            for &p in basis.iter() {
                let pb = p.trailing_zeros();
                if (v >> pb) & 1 == 1 {
                    v ^= p;
                }
            }
            if v == 0 {
                continue;
            }
            let vb = v.trailing_zeros();
            for p in basis.iter_mut() {
                if (*p >> vb) & 1 == 1 {
                    *p ^= v;
                }
            }
            let pos = basis
                .iter()
                .position(|&p| p.trailing_zeros() > vb)
                .unwrap_or(basis.len());
            basis.insert(pos, v);
        }
        basis
    }

    /// Dimension (rank) of the subspace spanned by `vecs`.
    pub fn rank(vecs: &[u64]) -> usize {
        reduce(vecs.to_vec()).len()
    }

    /// Whether `v` lies in the span of `basis`. `basis` need not be
    /// reduced.
    pub fn contains(basis: &[u64], v: u64) -> bool {
        let reduced = reduce(basis.to_vec());
        let mut cur = v;
        for &p in reduced.iter() {
            let pb = p.trailing_zeros();
            if (cur >> pb) & 1 == 1 {
                cur ^= p;
            }
        }
        cur == 0
    }

    /// Sum of two subspaces (union of bases, then reduce).
    pub fn sum(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut all = Vec::with_capacity(a.len() + b.len());
        all.extend_from_slice(a);
        all.extend_from_slice(b);
        reduce(all)
    }

    /// Intersection of two subspaces via Zassenhaus. Represents each
    /// pair `(left, right)` as a single `u128` and column-reduces on the
    /// left; rows whose left is zero after reduction give the
    /// intersection through their right component.
    pub fn intersection(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut rows: Vec<(u64, u64)> = a
            .iter()
            .map(|&x| (x, x))
            .chain(b.iter().map(|&y| (y, 0)))
            .collect();
        let mut r = 0;
        loop {
            // Find the row in `rows[r..]` with the smallest non-zero
            // trailing_zeros on its left component.
            let mut pivot: Option<usize> = None;
            let mut min_bit = u32::MAX;
            for (i, row) in rows.iter().enumerate().skip(r) {
                if row.0 != 0 {
                    let bit = row.0.trailing_zeros();
                    if bit < min_bit {
                        min_bit = bit;
                        pivot = Some(i);
                    }
                }
            }
            let Some(p) = pivot else {
                break;
            };
            rows.swap(r, p);
            let (lp, rp) = rows[r];
            for (i, row) in rows.iter_mut().enumerate() {
                if i != r && (row.0 >> min_bit) & 1 == 1 {
                    row.0 ^= lp;
                    row.1 ^= rp;
                }
            }
            r += 1;
        }
        // Rows with zero left give the intersection through their right.
        let inter: Vec<u64> = rows[r..]
            .iter()
            .map(|&(_, r_)| r_)
            .filter(|&x| x != 0)
            .collect();
        reduce(inter)
    }

    /// Standard basis of `F₂^dim`: `[1, 2, 4, …, 2^(dim-1)]`.
    pub fn standard_basis(dim: usize) -> Vec<u64> {
        (0..dim).map(|i| 1u64 << i).collect()
    }

    /// A basis for a complement of `sub` inside `F₂^ambient_dim`. Greedy:
    /// start from an empty basis, walk the standard basis of the ambient
    /// space, and keep each `e_i` that raises the current rank. The union
    /// with `sub` spans the whole ambient space; the returned basis has
    /// `ambient_dim - rank(sub)` vectors.
    pub fn complement(sub: &[u64], ambient_dim: usize) -> Vec<u64> {
        let mut current = reduce(sub.to_vec());
        let base_rank = current.len();
        let mut extra: Vec<u64> = Vec::new();
        for i in 0..ambient_dim {
            let e = 1u64 << i;
            if !contains(&current, e) {
                current.push(e);
                current = reduce(current);
                extra.push(e);
            }
        }
        debug_assert_eq!(current.len(), base_rank + extra.len());
        extra
    }

    /// A basis for a complement of `sub` **inside `containing`**. `sub`
    /// should be a subspace of `containing`; the returned basis together
    /// with `sub` spans `containing`, and its size is `rank(containing) -
    /// rank(sub)`. Vectors are drawn greedily from `containing`'s basis
    /// in the given order.
    pub fn complement_within(sub: &[u64], containing: &[u64]) -> Vec<u64> {
        let mut current = reduce(sub.to_vec());
        let mut extra: Vec<u64> = Vec::new();
        for &v in containing.iter() {
            if !contains(&current, v) {
                current.push(v);
                current = reduce(current);
                extra.push(v);
            }
        }
        extra
    }

    /// Extend `current` toward `ambient_dim` dimensions by greedily adding
    /// vectors from `candidates` (in order) that raise the rank, up to
    /// `max_extra`. Returns the added vectors (not the union).
    pub fn extend_with(current: &[u64], candidates: &[u64], max_extra: usize) -> Vec<u64> {
        let mut cur = reduce(current.to_vec());
        let mut added: Vec<u64> = Vec::new();
        for &c in candidates.iter() {
            if added.len() >= max_extra {
                break;
            }
            if !contains(&cur, c) {
                cur.push(c);
                cur = reduce(cur);
                added.push(c);
            }
        }
        added
    }

    /// The first `count` vectors of `basis` (already reduced), or all of
    /// them if there are fewer.
    pub fn take_independent(basis: &[u64], count: usize) -> Vec<u64> {
        basis.iter().take(count).copied().collect()
    }
}

/// Whether two XOR-affine maps agree as functions over the union of
/// their input widths. Missing high bases are treated as zero — inputs
/// past either domain are masked out by upstream bounds guards, so a
/// map with a shorter `bases` acts as if its high input bits are ignored.
pub fn maps_agree(a: &LinearLayout, b: &LinearLayout) -> bool {
    a.offset == b.offset
        && (0..a.bases.len().max(b.bases.len()))
            .all(|i| a.bases.get(i).copied().unwrap_or(0) == b.bases.get(i).copied().unwrap_or(0))
}

/// Whether the "sender slot" a shuffle would consume for each destination
/// slot `s'` is thread-independent. Fast path in `Strategy::Shuffle`:
/// no lane- or warp-input base of `C` writes into slot output positions,
/// so `C(s' << tb ^ tid) >> tb` collapses to `C(s' << tb) >> tb`. Under
/// this condition a shuffle can broadcast from a canonical source slot
/// even when the lane block is non-invertible (e.g. a replicated par as
/// under Gap 2) — the receivers just all read the same sender.
pub fn const_src_slot(c: &LinearLayout, tb: usize) -> bool {
    let slot_mask: u64 = if tb >= u64::BITS as usize {
        0
    } else {
        !((1u64 << tb) - 1)
    };
    c.bases[..tb.min(c.bases.len())]
        .iter()
        .all(|&b| b & slot_mask == 0)
}

/// The lane→slot mixing directions of `c : phys → phys`: the reduced
/// basis of `{ (c.bases[i] >> tb) : i < tb }` — the slot-output
/// components contributed by thread inputs. Under warp-column identity
/// the warp columns contribute zero, so this is exactly the span of
/// `C(tid) >> tb` over all `tid`. Empty iff [`const_src_slot`] holds.
pub fn lane_slot_mix_dirs(c: &LinearLayout, tb: usize) -> Vec<u64> {
    f2::reduce(
        c.bases[..tb.min(c.bases.len())]
            .iter()
            .map(|&b| b >> tb)
            .collect(),
    )
}

/// Whether the lane-input → lane-output sub-block of `c` is invertible.
/// Gates `gen_shuffle`'s sender-side ternary path (one `__shfl_sync` per
/// destination slot even when the sender slot varies per lane).
pub fn lane_block_invertible(c: &LinearLayout, tb: usize) -> bool {
    let lane_bits = 5.min(tb);
    let lane_mask = if lane_bits == 0 {
        0
    } else {
        (1u64 << lane_bits) - 1
    };
    LinearLayout {
        bases: c.bases[..lane_bits.min(c.bases.len())]
            .iter()
            .map(|&b| b & lane_mask)
            .collect(),
        offset: 0,
    }
    .inverse()
    .is_some()
}

/// Number of `__shfl_sync` instructions `gen_shuffle` emits for the
/// composite `c : phys → phys` under a `block`-thread CTA, mirroring its
/// dispatch:
///
/// - **Constant sender slot** ([`const_src_slot`]) — one shuffle per destination slot at a
///   compile-time-constant source slot: `slots`.
/// - **Invertible lane block** — sender-side ternary path, one shuffle per destination slot:
///   `slots`.
/// - **Multi-round** (everything else under warp-column identity, paper §5.4 page 8) — per
///   destination slot, one shuffle per candidate sender slot `σ` in the affine subspace `(C(s' <<
///   tb) >> tb) ^ span(dirs)`: `slots · 2^|dirs|` with `dirs =` [`lane_slot_mix_dirs`].
///
/// The caller (`best_decomposition`) checks `is_warp_column_identity`.
pub fn shuffle_rounds(c: &LinearLayout, block: usize) -> u64 {
    let kb = c.bases.len();
    let tb = kb.min(block.trailing_zeros() as usize);
    let slots = 1u64 << (kb - tb);
    if slots == 1 || const_src_slot(c, tb) || lane_block_invertible(c, tb) {
        return slots;
    }
    let dirs = lane_slot_mix_dirs(c, tb);
    slots << dirs.len().min(63)
}

/// Compute layout of a `par [N]`: a factorization of the logical iteration
/// domain into `seq_size` sequential steps (ILP) times the thread dimension.
/// `layout` maps the flattened physical index `x = s * blockDim + t` (the
/// sequential index `s` in the most significant bits) to the logical index;
/// out-of-range logical indices are masked by an `x < N` guard. The identity
/// layout is the strided factorization `i = s * blockDim + t`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParAttr {
    pub seq_size: usize,
    pub layout: LinearLayout,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AddressSpace {
    Global,
    /// Block-local shared memory.
    Shared,
    /// Thread-local registers (one slot per sequential step of the
    /// accessing pars).
    Register,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferKind {
    /// Bound at runtime via `set_input(i, ptr)`.
    Input(usize),
    /// Bound at runtime via `set_output(i, ptr)`.
    Output(usize),
    /// Kernel-local shared memory, materialized by an [`SSAOpCode::Alloc`].
    Shared,
    /// Kernel-local registers, materialized by an [`SSAOpCode::Alloc`].
    /// Only accessible at a par's own logical index (each par point owns
    /// one element).
    Register,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BufferDecl {
    pub name: String,
    pub elem: ScalarType,
    /// Logical row-major shape. Only the outermost dim of a global buffer
    /// may be symbolic (row-major strides never involve it); kernel-local
    /// buffers are fully concrete.
    pub shape: Vec<SizeExpr>,
    pub kind: BufferKind,
    pub space: AddressSpace,
    /// The alloc attribute, filled in by `layout_infer`. For shared (and
    /// global) buffers it maps the linearized logical index to the physical
    /// address; `None` (or the identity) is the row-major identity layout.
    /// For register buffers the direction is reversed: it maps the owning
    /// par's flattened physical index `slot * blockDim + thread` (which
    /// equals the par's logical index under the identity par attr) to the
    /// logical element held there. Reduce accumulators keep `None` and are
    /// only accessed at the par's own index.
    pub layout: Option<LinearLayout>,
}

impl BufferDecl {
    /// Concrete logical element count. Panics on a symbolic shape; callers
    /// that may see symbolic global buffers use [`Self::len_expr`].
    pub fn len(&self) -> usize {
        self.shape
            .iter()
            .map(|d| {
                d.as_const()
                    .unwrap_or_else(|| panic!("buffer `{}` has a symbolic shape", self.name))
                    as usize
            })
            .product()
    }

    /// Logical element count as an expression over the program params.
    /// Inner dims must be concrete (only the outermost may be symbolic).
    pub fn len_expr(&self) -> SizeExpr {
        let Some(first) = self.shape.first() else {
            return 1usize.into();
        };
        let inner: i64 = self.shape[1..]
            .iter()
            .map(|d| {
                d.as_const().unwrap_or_else(|| {
                    panic!("buffer `{}`: inner dims must be concrete", self.name)
                })
            })
            .product();
        first.mul_c(SymConst::Lit(inner)).fold_lits()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len() * self.elem.size_bytes()
    }

    /// Physical element count after applying the alloc layout (a
    /// non-identity layout maps into a power-of-two range).
    pub fn phys_len(&self) -> usize {
        match &self.layout {
            Some(l) if !l.is_identity() => 1usize << l.bases.len(),
            _ => self.len(),
        }
    }

    /// Physical size in bytes: `phys_len() * elem.size_bytes()`.
    pub fn phys_bytes(&self) -> usize {
        self.phys_len() * self.elem.size_bytes()
    }
}

/// Opcode of an [`SSAOp`]. Operands and results live in
/// [`SSAOp::operands`] / [`SSAOp::results`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SSAOpCode {
    /// Sequential loop over `0..bound`; the block's first operand is the
    /// induction variable. At the statement level (grid or loop block) it
    /// carries no values (no results), is uniform across the block, and pars
    /// inside may sync; its operands are the values captured from enclosing
    /// scopes. Inside a par it is an MLIR-style `scf.for`: the op's operands
    /// are the initial values of the loop-carried variables followed by the
    /// captures (`operands[i]` initializes `results[i]`), the block's
    /// operands are `[induction var, carried...]`, its yields are the next
    /// carried values, and the op's results are the carried values after
    /// the last iteration.
    Loop { bound: usize },
    /// Primitive compute block over `bound` logical indices: loads `reads`,
    /// runs its block per index, stores the yields to `writes`. The op's
    /// operands are the values captured from enclosing scopes (including
    /// access-index symbols other than the par's own index); its results
    /// represent the writes, one per write, in order. The block's operands
    /// are `[par index, one value per read, in order]`; its yields are one
    /// per write, in order. `attr` (from `layout_infer`) factors the domain
    /// onto sequential steps x threads.
    Par {
        /// Symbolic only for grid-spanning pars (the guard bound); per-block
        /// pars are concrete.
        bound: KBound,
        /// Grid-spanning par: the logical index is
        /// `blockIdx.x * blockDim.x + threadIdx.x` and the grid covers the
        /// whole domain. Otherwise the par iterates its domain per block.
        spans_grid: bool,
        attr: Option<ParAttr>,
        reads: Vec<Access>,
        writes: Vec<Access>,
    },
    /// Materializes a shared or register buffer in the kernel.
    Alloc { buf: BufId },
    /// Block-wide barrier (`__syncthreads()`). Statement level only; no
    /// operands, results or region. Inserted by `passes::insert_sync`
    /// before any par that reads a shared buffer written since the last
    /// barrier.
    Sync,
    /// Materializes `dst[i] = src[map(i)]` over `dst`'s logical domain,
    /// where each buffer's own layout locates its logical elements.
    /// Statement level only; no operands, results or region — codegen
    /// realizes it as a register-slot permutation, a warp shuffle or a
    /// shared-memory staging loop depending on the buffers' address
    /// spaces and [`classify_convert`]. Inserted by `passes::layout_infer`
    /// right after the op writing `src`.
    ///
    /// `scratch` names a companion [`AddressSpace::Shared`] buffer allocated
    /// by `layout_infer` alongside every `ConvertLayout`. Its byte size is
    /// filled in by `passes::allocate_convert_scratch` from the `(src, dst)`
    /// layout pair (Phase B.2); pure register→register shuffles need no
    /// scratch and the buffer stays 0-byte, but the BufId is always present
    /// so codegen and the shared-memory packer see one uniform interface.
    /// The register→register bounce and shared→shared paths (Phase B.4-full)
    /// stage through `scratch` under a bank-conflict-minimizing swizzle
    /// picked at codegen time from the two accesses that touch it.
    ConvertLayout {
        dst: BufId,
        src: BufId,
        scratch: BufId,
        map: LinearLayout,
    },
    /// No operands; one result.
    ConstU32(u32),
    /// A symbolic constant over module parameters (`SymConst::Sym`
    /// positions), read from the kernel's device parameters at runtime.
    /// No operands; one result (`U32`).
    ConstSym(SizeExpr),
    /// BabyBear constant (canonical representation); one result.
    ConstField(u32),
    /// FpExt constant `a0 + a1 x + a2 x^2 + a3 x^3` (each a canonical
    /// BabyBear `u32`); no operands; one result.
    ConstFpExt([u32; 4]),
    /// Lift a `BabyBear` value to `FpExt` as `(x, 0, 0, 0)`; one operand,
    /// one result.
    LiftFpExt,
    /// Two operands; one result. The scalar type selects field vs integer
    /// semantics.
    Bin(BinOp, ScalarType),
    /// One operand `[cond]`; one result; `SSAOp.block` is the then-body
    /// and its `yields[0]` is the then-value; the `else_block` field
    /// carries the else-body and its `yields[0]` is the else-value. Only
    /// the taken branch's body is executed, so any loads it contains are
    /// gated by `cond` — the DSL `if cond then A else B` compiles to
    /// this and never speculatively evaluates the untaken side.
    Select { else_block: SSABlock },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SSAOp {
    pub operands: SmallVec<[SSARes; 2]>,
    pub results: SmallVec<[SSARes; 1]>,
    pub opcode: SSAOpCode,
    /// Nested region; empty except for [`SSAOpCode::Loop`] and
    /// [`SSAOpCode::Par`].
    pub block: SSABlock,
}

/// A region of SSA ops. Loads are not representable inside a par's block:
/// its memory reads enter through the block operands.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SSABlock {
    /// Values bound on entry. For a par block: `[par index, one value per
    /// read, in order]`. For a loop block: `[induction var, carried...]`.
    /// For the grid block: `[grid index]`.
    pub operands: SmallVec<[SSARes; 2]>,
    pub body: SmallVec<[SSANode; 8]>,
    /// Values leaving the block. For a par block: one per write, in order.
    /// For a loop block: the next loop-carried values.
    pub yields: SmallVec<[SSARes; 1]>,
}

/// How an access maps a par's logical index to a buffer's logical index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum IndexMap {
    /// `index = layout(par index)`.
    Linear(LinearLayout),
    /// A quasi-affine expression whose symbols are kernel values by the
    /// `VarId(i) <-> SSARes(i)` convention: the par's own index, enclosing
    /// loop induction variables and the grid index.
    Affine {
        expr: Quast,
        /// Bounds of the symbols appearing in `expr`.
        bounds: BTreeMap<VarId, u64>,
    },
    /// A symbolic index expression: `Expr::Sym` positions are kernel values
    /// (the `VarId(i) <-> SSARes(i)` convention, as in [`IndexMap::Affine`])
    /// and `SymConst::Sym` positions are module parameters. Not analyzable
    /// by layout inference: a write pins the buffer to shared memory, a
    /// read gets a shared-memory mirror.
    SExpr(SizeExpr),
    /// Like [`IndexMap::SExpr`], but the expression may also reference
    /// loaded SSA values (data-dependent indexing, e.g. a gather through an
    /// index buffer). Codegen must emit the index-producing loads before
    /// the dependent access.
    Blackbox(SizeExpr),
}

/// One declared memory access of a par.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Access {
    pub buf: BufId,
    pub index: IndexMap,
}

impl Access {
    /// Kernel values used as symbols by the index expression (by the
    /// `VarId(i) <-> SSARes(i)` convention). Module parameters live in
    /// `SymConst::Sym` positions and are not kernel values, so they are
    /// never reported here.
    pub fn index_syms(&self, out: &mut BTreeSet<SSARes>) {
        let mut syms = BTreeSet::new();
        match &self.index {
            IndexMap::Linear(_) => return,
            IndexMap::Affine { expr, .. } => expr.syms(&mut syms),
            IndexMap::SExpr(e) | IndexMap::Blackbox(e) => e.syms(&mut syms),
        }
        out.extend(syms.into_iter().map(|v| SSARes(v.0)));
    }
}

/// The kernel body: `bound` blocks (`gridDim.x`), with the block's first
/// operand bound to `blockIdx.x` as a kernel-level SSA value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grid {
    pub bound: KBound,
    pub block: SSABlock,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kernel {
    pub name: String,
    pub grid: Grid,
    /// `blockDim.x`.
    pub block: usize,
    /// Buffers appearing in the kernel signature, with write flag.
    pub params: Vec<(BufId, bool)>,
    ops: Vec<SSAOp>,
    next_val: u32,
}

impl Kernel {
    /// A new kernel with the grid index bound to the first fresh value.
    pub fn new(name: String, grid_bound: impl Into<KBound>, block: usize) -> Self {
        let mut k = Kernel {
            name,
            grid: Grid {
                bound: grid_bound.into(),
                block: SSABlock::default(),
            },
            block,
            params: Vec::new(),
            ops: Vec::new(),
            next_val: 0,
        };
        let var = k.fresh_val();
        k.grid.block.operands.push(var);
        k
    }

    /// The grid index (`blockIdx.x`), the grid block's first operand.
    pub fn grid_var(&self) -> SSARes {
        self.grid.block.operands[0]
    }

    pub fn fresh_val(&mut self) -> SSARes {
        let v = SSARes(self.next_val);
        self.next_val += 1;
        v
    }

    pub fn push_op(&mut self, op: SSAOp) -> SSANode {
        let id = SSANode(self.ops.len() as u32);
        self.ops.push(op);
        id
    }

    pub fn op(&self, id: SSANode) -> &SSAOp {
        &self.ops[id.0 as usize]
    }

    pub fn ops(&self) -> &[SSAOp] {
        &self.ops
    }

    pub fn ops_mut(&mut self) -> &mut [SSAOp] {
        &mut self.ops
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KirProgram {
    pub name: String,
    pub buffers: Vec<BufferDecl>,
    pub kernels: Vec<Kernel>,
    /// Buffer id per module input index.
    pub input_bufs: Vec<BufId>,
    /// Buffer id per module output index.
    pub output_bufs: Vec<BufId>,
    /// Surviving symbolic module parameters, in declaration order. This
    /// order defines the runtime ABI: device kernels take one trailing
    /// `const uint32_t` per entry and the host `Prog` stores them as
    /// `int64_t params[]`, bound via `set_symbol(name, v)`.
    pub params: Vec<(VarId, String)>,
}

impl KirProgram {
    pub fn buffer(&self, id: BufId) -> &BufferDecl {
        &self.buffers[id.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation5() -> LinearLayout {
        // Rotate five bits left by one: bit i -> bit (i + 1) % 5.
        LinearLayout {
            bases: vec![2, 4, 8, 16, 1],
            offset: 0,
        }
    }

    #[test]
    fn inverse_of_rotation_is_reverse_rotation() {
        let rot = rotation5();
        let inv = rot.inverse().unwrap();
        for x in 0..32u64 {
            assert_eq!(inv.apply(rot.apply(x)), x);
            assert_eq!(rot.apply(inv.apply(x)), x);
        }
        assert!(rot.compose(&inv).is_identity());
        assert!(inv.compose(&rot).is_identity());
    }

    #[test]
    fn singular_maps_have_no_inverse() {
        // x -> x >> 1 drops a bit.
        let shift = LinearLayout {
            bases: vec![0, 1, 2, 4],
            offset: 0,
        };
        assert!(shift.inverse().is_none());
        // Rectangular: 2 input bits into a 4-bit space.
        let rect = LinearLayout {
            bases: vec![4, 8],
            offset: 0,
        };
        assert!(rect.inverse().is_none());
    }

    #[test]
    fn affine_offset_composes_and_inverts() {
        // x -> rot(x) ^ 5 over 5 bits.
        let mut aff = rotation5();
        aff.offset = 5;
        let inv = aff.inverse().unwrap();
        for x in 0..32u64 {
            assert_eq!(inv.apply(aff.apply(x)), x);
            assert_eq!(aff.apply(inv.apply(x)), x);
        }
        assert!(aff.compose(&inv).is_identity());
        assert!(inv.compose(&aff).is_identity());
        // Composition XORs the outer image of the inner offset.
        let c = aff.compose(&aff);
        for x in 0..32u64 {
            assert_eq!(c.apply(x), aff.apply(aff.apply(x)));
        }
    }

    #[test]
    fn compose_applies_right_map_first() {
        let rot = rotation5();
        let dbl = LinearLayout {
            bases: vec![2, 4, 8, 16, 0],
            offset: 0,
        };
        let c = dbl.compose(&rot);
        for x in 0..32u64 {
            assert_eq!(c.apply(x), dbl.apply(rot.apply(x)));
        }
    }

    #[test]
    fn right_inverse_identity() {
        let l = LinearLayout::identity(5);
        let r = l.right_inverse(5).unwrap();
        assert!(r.is_identity());
    }

    #[test]
    fn right_inverse_replicated_high_bits() {
        // bases = [1, 2, 0, 0, 0]: two live low-bit inputs, three replicated
        // (zero) columns. Right inverse must (i) exist, (ii) round-trip,
        // (iii) put the canonical replica in the pivot (low) bits.
        let l = LinearLayout {
            bases: vec![1, 2, 0, 0, 0],
            offset: 0,
        };
        let r = l.right_inverse(2).unwrap();
        assert_eq!(r.bases, vec![1, 2]);
        for y in 0..4u64 {
            assert_eq!(l.apply(r.apply(y)), y, "round-trip failed at y={y}");
        }
    }

    #[test]
    fn right_inverse_replicated_low_bits() {
        // bases = [0, 0, 0, 1, 2]: replication on the low three input bits,
        // live columns at input positions 3 and 4. Canonical replicas
        // should live at bits 3 and 4 (the pivot inputs), matching the
        // Gap 7 counterexample where canonical lanes are {0, 8, 16, 24}.
        let l = LinearLayout {
            bases: vec![0, 0, 0, 1, 2],
            offset: 0,
        };
        let r = l.right_inverse(2).unwrap();
        assert_eq!(r.bases, vec![1 << 3, 1 << 4]);
        assert_eq!(r.apply(0), 0);
        assert_eq!(r.apply(1), 1 << 3);
        assert_eq!(r.apply(2), 1 << 4);
        assert_eq!(r.apply(3), (1 << 3) | (1 << 4));
        for y in 0..4u64 {
            assert_eq!(l.apply(r.apply(y)), y);
        }
    }

    #[test]
    fn right_inverse_non_surjective() {
        // Only one output bit is in the column span; two-bit codomain
        // has no right inverse.
        let l = LinearLayout {
            bases: vec![1, 0],
            offset: 0,
        };
        assert!(l.right_inverse(2).is_none());
    }

    #[test]
    fn right_inverse_affine() {
        let l = LinearLayout {
            bases: vec![1, 2, 4],
            offset: 5,
        };
        let r = l.right_inverse(3).unwrap();
        for y in 0..8u64 {
            assert_eq!(l.apply(r.apply(y)), y);
        }
    }

    #[test]
    fn right_inverse_matches_inverse_on_square_bijections() {
        let rot = rotation5();
        let r = rot.right_inverse(5).unwrap();
        let inv = rot.inverse().unwrap();
        assert_eq!(r.bases, inv.bases);
        assert_eq!(r.offset, inv.offset);
    }

    #[test]
    fn right_inverse_of_composite_shift() {
        // The `shift` layout in singular_maps_have_no_inverse (x -> x >> 1)
        // is 4 → 3 dimensional but surjective onto its 3 low bits.
        let shift = LinearLayout {
            bases: vec![0, 1, 2, 4],
            offset: 0,
        };
        let r = shift.right_inverse(3).unwrap();
        for y in 0..8u64 {
            assert_eq!(shift.apply(r.apply(y)), y);
        }
        // Right composition is a projector on inputs, not the identity.
        let projector = r.compose(&shift);
        // shift.apply(1) = 0 (bit 0 collapsed), so projector maps bit-0
        // input to 0.
        assert_eq!(projector.apply(1), 0);
        // Bits 1, 2, 3 survive as bits 0, 1, 2 → 1, 2, 4 after reinjection.
        assert_eq!(projector.apply(2), 2);
        assert_eq!(projector.apply(4), 4);
        assert_eq!(projector.apply(8), 8);
    }

    #[test]
    fn maps_agree_treats_missing_bases_as_zero() {
        let a = LinearLayout {
            bases: vec![1, 2, 4],
            offset: 0,
        };
        let b = LinearLayout {
            bases: vec![1, 2, 4, 0, 0],
            offset: 0,
        };
        assert!(maps_agree(&a, &b));
        let c = LinearLayout {
            bases: vec![1, 2, 4, 8],
            offset: 0,
        };
        assert!(!maps_agree(&a, &c));
        let d = LinearLayout {
            bases: vec![1, 2, 4],
            offset: 5,
        };
        assert!(!maps_agree(&a, &d));
    }

    #[test]
    fn phys_partition_splits_k_into_slot_lane_warp() {
        // k=9, block=256 → tb=8, lane=5, warp=3, slot=1.
        let l = LinearLayout::identity(9);
        let p = l.phys_partition(256);
        assert_eq!(
            p,
            PhysPartition {
                slot: 1,
                lane: 5,
                warp: 3
            }
        );
        assert_eq!(p.thread_bits(), 8);
        assert_eq!(p.lane_mask(), 0b11111);
        assert_eq!(p.warp_mask(), 0b111 << 5);
        assert_eq!(p.slot_mask(), 1 << 8);
    }

    #[test]
    fn phys_partition_sub_warp_block() {
        // k=4, block=16 → tb=4, lane=4, warp=0, slot=0.
        let l = LinearLayout::identity(4);
        let p = l.phys_partition(16);
        assert_eq!(
            p,
            PhysPartition {
                slot: 0,
                lane: 4,
                warp: 0
            }
        );
        assert_eq!(p.warp_mask(), 0);
        assert_eq!(p.slot_mask(), 0);
    }

    #[test]
    fn diagonal_blocks_of_identity_are_identities() {
        let l = LinearLayout::identity(9);
        let b = 256usize;
        assert!(l.lane_block(b).is_identity());
        assert!(l.warp_block(b).is_identity());
        assert!(l.slot_block(b).is_identity());
        assert!(l.is_warp_column_identity(b));
    }

    #[test]
    fn warp_block_isolates_warp_diagonal() {
        // k=9, block=256, so warp inputs are bits 5..8 → three columns
        // bases[5], bases[6], bases[7]. Set bases[5] to touch bit 5 (warp)
        // AND bit 0 (lane): warp_block projects out the lane part.
        let mut l = LinearLayout::identity(9);
        l.bases[5] = (1 << 5) | 1; // warp bit 5 + lane bit 0
        let wb = l.warp_block(256);
        // First warp column shifted down: bit 5 → bit 0 (identity in warp
        // bit 0); lane bit 0 is masked out.
        assert_eq!(wb.bases, vec![1, 2, 4]);
        assert!(wb.is_identity());
        // The paper condition, however, is violated (lane part is non-zero).
        assert!(!l.is_warp_column_identity(256));
    }

    #[test]
    fn is_warp_column_identity_rejects_offset_in_warp_bits() {
        let mut l = LinearLayout::identity(9);
        l.offset = 1 << 6; // a warp-bit offset
        assert!(!l.is_warp_column_identity(256));
    }

    #[test]
    fn output_bits_is_top_bit_position_plus_one() {
        let l = LinearLayout::identity(5);
        assert_eq!(l.output_bits(), 5);
        let z = LinearLayout {
            bases: vec![0, 0, 0],
            offset: 0,
        };
        assert_eq!(z.output_bits(), 0);
        let off = LinearLayout {
            bases: vec![1, 2],
            offset: 1 << 7,
        };
        assert_eq!(off.output_bits(), 8);
    }

    #[test]
    fn is_distributed_flags_replicated_layouts() {
        // Identity is distributed.
        assert!(LinearLayout::identity(5).is_distributed());
        // Replicated (zero columns) is distributed.
        let repl = LinearLayout {
            bases: vec![1, 2, 0, 0, 0],
            offset: 0,
        };
        assert!(repl.is_distributed());
        // Two columns share bit 0 → not distributed.
        let dup = LinearLayout {
            bases: vec![1, 1],
            offset: 0,
        };
        assert!(!dup.is_distributed());
        // A column has two bits set → not distributed.
        let two_bits = LinearLayout {
            bases: vec![1, 3],
            offset: 0,
        };
        assert!(!two_bits.is_distributed());
        // Offset does not affect distributedness (Def 4.10 is on columns).
        let off = LinearLayout {
            bases: vec![1, 2, 4],
            offset: 7,
        };
        assert!(off.is_distributed());
    }

    #[test]
    fn f2_reduce_produces_canonical_rref() {
        // {3, 5, 6} — bits {01, 10, 11} in the low 3 positions is trivial;
        // reduce should give the standard basis of the span.
        let r = f2::reduce(vec![3, 5, 6]);
        // Rank is 2 (three vectors sum to 0 in F_2).
        assert_eq!(r.len(), 2);
        // Pivots strictly increasing.
        for w in r.windows(2) {
            assert!(w[0].trailing_zeros() < w[1].trailing_zeros());
        }
        // Each pivot bit is present in exactly one basis vector (RREF).
        for (i, &ri) in r.iter().enumerate() {
            let pb = ri.trailing_zeros();
            for (j, &rj) in r.iter().enumerate() {
                if i != j {
                    assert_eq!((rj >> pb) & 1, 0);
                }
            }
        }
    }

    #[test]
    fn f2_reduce_drops_zero_and_dependent_vectors() {
        let r = f2::reduce(vec![0, 1, 0, 2, 3, 0]);
        assert_eq!(r, vec![1, 2]);
        assert_eq!(f2::rank(&[0, 0, 0]), 0);
    }

    #[test]
    fn f2_contains_recognizes_span_membership() {
        let basis = vec![1, 2];
        assert!(f2::contains(&basis, 0));
        assert!(f2::contains(&basis, 1));
        assert!(f2::contains(&basis, 2));
        assert!(f2::contains(&basis, 3));
        assert!(!f2::contains(&basis, 4));
        assert!(!f2::contains(&basis, 7));
    }

    #[test]
    fn f2_sum_and_intersection_agree_with_dimension_formula() {
        // A = span{1, 2, 4}, B = span{2, 4, 8}. A ∩ B = span{2, 4},
        // A + B = span{1, 2, 4, 8}.
        let a = vec![1, 2, 4];
        let b = vec![2, 4, 8];
        let inter = f2::intersection(&a, &b);
        let sum = f2::sum(&a, &b);
        assert_eq!(f2::rank(&inter), 2);
        assert_eq!(f2::rank(&sum), 4);
        assert_eq!(
            f2::rank(&a) + f2::rank(&b),
            f2::rank(&inter) + f2::rank(&sum),
        );
        // Every vector in the intersection is in both A and B.
        for &v in inter.iter() {
            assert!(f2::contains(&a, v));
            assert!(f2::contains(&b, v));
        }
    }

    #[test]
    fn f2_intersection_of_disjoint_subspaces_is_empty() {
        let a = vec![1, 2];
        let b = vec![4, 8];
        let inter = f2::intersection(&a, &b);
        assert_eq!(inter.len(), 0);
    }

    #[test]
    fn f2_intersection_of_overlapping_non_axis_vectors() {
        // A = span{1 ⊕ 2, 4} = {0, 3, 4, 7}. B = span{3, 8} = {0, 3, 8, 11}.
        // A ∩ B = {0, 3}.
        let a = vec![3, 4];
        let b = vec![3, 8];
        let inter = f2::intersection(&a, &b);
        assert_eq!(inter, vec![3]);
    }

    #[test]
    fn f2_complement_spans_ambient_when_unioned() {
        // sub = {1, 4} in F_2^4. Complement should give {2, 8}.
        let sub = vec![1, 4];
        let comp = f2::complement(&sub, 4);
        assert_eq!(comp.len(), 2);
        let mut combined = sub.clone();
        combined.extend(comp.iter().copied());
        assert_eq!(f2::rank(&combined), 4);
    }

    #[test]
    fn f2_complement_of_full_space_is_empty() {
        let sub = vec![1, 2, 4];
        let comp = f2::complement(&sub, 3);
        assert!(comp.is_empty());
    }

    #[test]
    fn f2_extend_with_greedy_picks_first_rank_increasing() {
        let current = vec![1];
        let candidates = vec![1, 3, 2, 4]; // 1 dup, 3 = 1 ^ 2 (new), 2 (redundant now), 4 (new)
        let added = f2::extend_with(&current, &candidates, 3);
        assert_eq!(added, vec![3, 4]);
    }

    #[test]
    fn f2_complement_within_extracts_quotient_basis() {
        // containing = span{1, 2, 4}, sub = span{3} (= span{1 ⊕ 2}).
        // A complement of sub inside containing has dim 2.
        let sub = vec![3];
        let containing = vec![1, 2, 4];
        let comp = f2::complement_within(&sub, &containing);
        assert_eq!(comp.len(), 2);
        // union must span containing.
        let mut union = sub.clone();
        union.extend(comp.iter().copied());
        assert_eq!(f2::rank(&union), 3);
    }

    #[test]
    fn f2_extend_with_respects_max_extra() {
        let current: Vec<u64> = vec![];
        let cands = f2::standard_basis(5);
        let added = f2::extend_with(&current, &cands, 2);
        assert_eq!(added, vec![1, 2]);
    }
}
