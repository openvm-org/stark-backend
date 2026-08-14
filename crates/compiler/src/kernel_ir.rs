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
#[derive(Clone, Debug)]
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SSARes(pub u32);

/// Id of an [`SSAOp`] in the kernel's op arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SSANode(pub u32);

/// Id of a [`BufferDecl`] in the program's buffer table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufId(pub u32);

/// An XOR-affine map `T: Z_2^k -> Z_2^k`, `T(x) = M(x) ^ offset`, with the
/// linear part `M` represented by its images of the basis vectors:
/// `bases[i] = M(1 << i)`. `M(x)` is the XOR of the bases selected by the
/// set bits of `x`.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        // For each output bit r, the input-space vector whose image is
        // `e_r`. `None` means no such vector exists ⇒ not surjective.
        let mut preimage: Vec<Option<u64>> = vec![None; out_bits];
        // Columns already used as pivots.
        let mut used: Vec<bool> = vec![false; k];
        for r in 0..out_bits {
            let pivot = (0..k).find(|&c| !used[c] && (m[c] >> r) & 1 == 1)?;
            used[pivot] = true;
            preimage[r] = Some(inv[pivot]);
            // Zero bit r out of all other columns using the pivot.
            let piv_m = m[pivot];
            let piv_inv = inv[pivot];
            for c in 0..k {
                if c != pivot && (m[c] >> r) & 1 == 1 {
                    m[c] ^= piv_m;
                    inv[c] ^= piv_inv;
                }
            }
        }
        let new_bases: Vec<u64> = preimage.into_iter().map(|p| p.unwrap()).collect();
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

/// Whether two XOR-affine maps agree as functions over the union of
/// their input widths. Missing high bases are treated as zero — inputs
/// past either domain are masked out by upstream bounds guards, so a
/// map with a shorter `bases` acts as if its high input bits are ignored.
pub fn maps_agree(a: &LinearLayout, b: &LinearLayout) -> bool {
    a.offset == b.offset
        && (0..a.bases.len().max(b.bases.len()))
            .all(|i| a.bases.get(i).copied().unwrap_or(0) == b.bases.get(i).copied().unwrap_or(0))
}

/// How a layout-conversion map `C` (destination physical index to source
/// physical index, both flattened as `slot * blockDim + thread` with the
/// lane in the low 5 thread bits) can be realized. Classified by
/// [`classify_convert`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConvertKind {
    /// `C` is the identity: no data movement at all.
    Copy,
    /// `C` fixes the thread bits and its slot bits depend only on slot
    /// bits: each thread permutes its own register slots.
    Slot,
    /// `C` fixes the warp bits and its lane-to-lane block is invertible:
    /// one `__shfl_sync` per destination slot.
    Shuffle,
    /// Anything else: the conversion has to go through shared memory.
    Bounce,
}

/// Whether the "sender slot" a shuffle would consume for each destination
/// slot `s'` is thread-independent. Fast path in [`ConvertKind::Shuffle`]:
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

/// Classifies the conversion map `C` over `k = C.bases.len()` input bits for
/// a block of `block` threads. `C` maps dst physical → src physical, with
/// the low `min(k, log2(block))` bits being thread bits (lane in the low 5,
/// warp above) and the rest slot bits.
///
/// Under Gap 4 (surjective source layouts), `Shuffle` no longer requires
/// the lane block to be invertible: the fast path through `const_src_slot`
/// covers replicated senders where multiple destination lanes broadcast
/// from one canonical source.
pub fn classify_convert(c: &LinearLayout, block: usize) -> ConvertKind {
    if c.is_identity() {
        return ConvertKind::Copy;
    }
    let k = c.bases.len();
    // The bit decomposition of the physical index assumes a power-of-two
    // block; the domain must span whole warps for a shuffle.
    if !block.is_power_of_two() {
        return ConvertKind::Bounce;
    }
    let tb = k.min(block.trailing_zeros() as usize);
    let thread_mask = (1u64 << tb) - 1;

    let slot_only = c.offset & thread_mask == 0
        && c.bases.iter().enumerate().all(|(i, &b)| {
            if i < tb {
                b == 1 << i
            } else {
                b & thread_mask == 0
            }
        });
    if slot_only {
        return ConvertKind::Slot;
    }

    if tb == 0 {
        // No thread bits ⇒ nothing to shuffle; the legitimate within-thread
        // cases were caught by `slot_only` above.
        return ConvertKind::Bounce;
    }
    // Lane bits are the low `min(tb, 5)` thread bits; the remaining thread
    // bits (bits 5..tb, only if the CTA spans multiple warps) are warp
    // bits. Sub-warp CTAs (tb < 5, e.g. Poseidon2-16 at block=16) have no
    // warp bits — the checks below collapse to trivialities but the same
    // code covers them.
    let lane_bits = tb.min(5);
    let lane_mask = (1u64 << lane_bits) - 1;
    let warp_mask = thread_mask & !lane_mask;
    // Offset bits in the lane fold into a lane XOR and slot bits into the
    // per-slot constants, but warp bits would cross warps.
    if c.offset & warp_mask != 0 {
        return ConvertKind::Bounce;
    }
    let warp_fixed = c.bases.iter().enumerate().all(|(i, &b)| {
        let want = if (lane_bits..tb).contains(&i) { 1 << i } else { 0 };
        b & warp_mask == want
    });
    if !warp_fixed {
        return ConvertKind::Bounce;
    }
    let lane_block = LinearLayout {
        bases: c.bases[..lane_bits]
            .iter()
            .map(|&b| b & lane_mask)
            .collect(),
        offset: 0,
    };
    // Shuffle-fast-path: sender slot constant across the warp. Covers the
    // whole XOR-offset butterfly family AND the replicated broadcasts
    // introduced by Gap 2 (non-invertible lane blocks from zero-column
    // par-attrs), which are exactly the cases where a full-mask shuffle
    // broadcasts one source lane's value to many receivers.
    //
    // Bijective lane block AND non-const src slot is the general shuffle
    // path (transpose-style): `gen_shuffle` inverts the lane block to
    // route each destination slot from a per-tid computed source.
    if const_src_slot(c, tb) || lane_block.inverse().is_some() {
        ConvertKind::Shuffle
    } else {
        ConvertKind::Bounce
    }
}

/// Compute layout of a `par [N]`: a factorization of the logical iteration
/// domain into `seq_size` sequential steps (ILP) times the thread dimension.
/// `layout` maps the flattened physical index `x = s * blockDim + t` (the
/// sequential index `s` in the most significant bits) to the logical index;
/// out-of-range logical indices are masked by an `x < N` guard. The identity
/// layout is the strided factorization `i = s * blockDim + t`.
#[derive(Clone, Debug)]
pub struct ParAttr {
    pub seq_size: usize,
    pub layout: LinearLayout,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
    /// Block-local shared memory.
    Shared,
    /// Thread-local registers (one slot per sequential step of the
    /// accessing pars).
    Register,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
    ConvertLayout {
        dst: BufId,
        src: BufId,
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

#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Default)]
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
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct Grid {
    pub bound: KBound,
    pub block: SSABlock,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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
    fn classify_convert_cases() {
        let block = 256usize;
        assert_eq!(
            classify_convert(&LinearLayout::identity(9), block),
            ConvertKind::Copy
        );
        // Slot-bit swap over k = 10, tb = 8: threads fixed, slots permuted.
        let mut slot_swap = LinearLayout::identity(10);
        slot_swap.bases[8] = 1 << 9;
        slot_swap.bases[9] = 1 << 8;
        assert_eq!(classify_convert(&slot_swap, block), ConvertKind::Slot);
        // Lane rotation over k = 9: warps fixed, lane block invertible.
        let mut lane_rot = LinearLayout::identity(9);
        lane_rot.bases[..5].copy_from_slice(&rotation5().bases);
        assert_eq!(classify_convert(&lane_rot, block), ConvertKind::Shuffle);
        // Slot bit XOR-ed into the lane path stays a shuffle (lane block
        // is still the identity).
        let mut slot_xor = LinearLayout::identity(10);
        slot_xor.bases[0] = 1 | (1 << 9);
        assert_eq!(classify_convert(&slot_xor, block), ConvertKind::Shuffle);
        // Warp bit moved into the lanes cannot be shuffled.
        let mut warp_mix = LinearLayout::identity(9);
        warp_mix.bases[5] = 1;
        warp_mix.bases[0] = 1 << 5;
        assert_eq!(classify_convert(&warp_mix, block), ConvertKind::Bounce);
        // Singular lane block with `const_src_slot`: bit-0 lane input maps
        // to no output, so pairs of receiver lanes broadcast the same
        // sender slot. Under Gap 4 this is a Shuffle, not a Bounce —
        // `__shfl_sync` handles many-to-one lane routing natively.
        let mut fold = LinearLayout::identity(9);
        fold.bases[0] = 0;
        assert_eq!(classify_convert(&fold, block), ConvertKind::Shuffle);
        // Singular lane block *with* lane→slot mixing: a lane input maps
        // into a slot output bit, so the sender slot is thread-dependent
        // even though the lane block is non-invertible. Fast path (const
        // src slot) fails and the general path can't invert → Bounce.
        let mut fold_slot = LinearLayout::identity(9);
        fold_slot.bases[0] = 1 << 8; // bit-0 lane → bit-8 (a slot bit)
        assert_eq!(classify_convert(&fold_slot, block), ConvertKind::Bounce);
        // Pure XOR offsets: slot bits permute registers, lane bits shuffle
        // (a butterfly partner read), warp bits cross warps.
        let mut slot_off = LinearLayout::identity(10);
        slot_off.offset = 1 << 9;
        assert_eq!(classify_convert(&slot_off, block), ConvertKind::Slot);
        let mut lane_off = LinearLayout::identity(9);
        lane_off.offset = 16;
        assert_eq!(classify_convert(&lane_off, block), ConvertKind::Shuffle);
        let mut warp_off = LinearLayout::identity(9);
        warp_off.offset = 1 << 6;
        assert_eq!(classify_convert(&warp_off, block), ConvertKind::Bounce);
        // Non-power-of-two blocks cannot be bit-decomposed.
        assert_eq!(
            classify_convert(&lane_rot, 100),
            ConvertKind::Bounce,
            "non-pow2 block"
        );

        // Sub-warp block (WIDTH=16 sponge / Poseidon2-16 pattern): tb = k = 4,
        // no warp bits. XOR-stride partner reads are invertible lane
        // permutations and must classify as Shuffle.
        let sub_warp_block = 16usize;
        for stride in [1u64, 2, 4, 8] {
            let mut xor_stride = LinearLayout::identity(4);
            xor_stride.offset = stride;
            assert_eq!(
                classify_convert(&xor_stride, sub_warp_block),
                ConvertKind::Shuffle,
                "sub-warp XOR-stride {stride}",
            );
        }
        // Sub-warp non-invertible lane block (apply_mat4's `j - j%4 + r`
        // access: bits 0 and 1 collapse). Under Gap 4 this is a Shuffle —
        // 4 receivers per canonical sender, all resolved to the same
        // register slot (fast path).
        let mat4_read = LinearLayout {
            bases: vec![0, 0, 4, 8],
            offset: 1,
        };
        assert_eq!(
            classify_convert(&mat4_read, sub_warp_block),
            ConvertKind::Shuffle,
            "sub-warp non-invertible lane block with const src slot",
        );
        // Sub-warp k=6 (kb-tb=2 slot bits) with a lane→slot cross-term
        // and a singular lane block: no fast path *and* no general
        // path. Bounce.
        let mat4_hard = LinearLayout {
            bases: vec![16, 0, 4, 8, 0, 32],
            offset: 0,
        };
        assert_eq!(
            classify_convert(&mat4_hard, sub_warp_block),
            ConvertKind::Bounce,
            "sub-warp singular lane block with lane→slot cross-term",
        );
    }
}
