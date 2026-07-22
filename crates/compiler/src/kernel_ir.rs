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
    ir::{BinOp, ScalarType, VarId},
    quast::Quast,
};

/// SSA value inside one kernel (a single kernel-wide id space).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SSARes(pub u32);

/// Id of an [`SSAOp`] in the kernel's op arena.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SSANode(pub u32);

/// Id of a [`BufferDecl`] in the program's buffer table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufId(pub u32);

/// A linear map `T: Z_2^k -> Z_2^k`, represented by its images of the basis
/// vectors: `bases[i] = T(1 << i)`. `T(x)` is the XOR of the bases selected
/// by the set bits of `x`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearLayout {
    pub bases: Vec<u64>,
}

impl LinearLayout {
    pub fn identity(bits: usize) -> Self {
        Self {
            bases: (0..bits).map(|i| 1u64 << i).collect(),
        }
    }

    pub fn apply(&self, x: u64) -> u64 {
        self.bases
            .iter()
            .enumerate()
            .filter(|&(i, _)| (x >> i) & 1 == 1)
            .fold(0, |acc, (_, &b)| acc ^ b)
    }

    pub fn is_identity(&self) -> bool {
        self.bases.iter().enumerate().all(|(i, &b)| b == 1u64 << i)
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
    /// Lives at `offset` bytes inside the scratch allocation.
    Scratch { offset: usize },
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
    /// Logical row-major shape.
    pub shape: Vec<usize>,
    pub kind: BufferKind,
    pub space: AddressSpace,
    /// Map from the linearized logical index to the physical index (the
    /// alloc attribute). `None` (or the identity) is the row-major identity
    /// layout. Filled in by `layout_infer` for shared buffers.
    pub layout: Option<LinearLayout>,
}

impl BufferDecl {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        // All scalar types are stored as u32.
        self.len() * 4
    }

    /// Number of physical u32 slots after applying the alloc layout (a
    /// non-identity layout maps into a power-of-two range).
    pub fn phys_len(&self) -> usize {
        match &self.layout {
            Some(l) if !l.is_identity() => 1usize << l.bases.len(),
            _ => self.len(),
        }
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
        bound: usize,
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
    /// No operands; one result.
    ConstU32(u32),
    /// BabyBear constant (canonical representation); one result.
    ConstField(u32),
    /// Two operands; one result. The scalar type selects field vs integer
    /// semantics.
    Bin(BinOp, ScalarType),
    /// Three operands `[cond, then, else]`; one result.
    Select,
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
#[derive(Clone, Debug)]
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
}

/// One declared memory access of a par.
#[derive(Clone, Debug)]
pub struct Access {
    pub buf: BufId,
    pub index: IndexMap,
}

impl Access {
    /// Kernel values used as symbols by the index expression (by the
    /// `VarId(i) <-> SSARes(i)` convention).
    pub fn index_syms(&self, out: &mut BTreeSet<SSARes>) {
        if let IndexMap::Affine { expr, .. } = &self.index {
            let mut syms = BTreeSet::new();
            expr.syms(&mut syms);
            out.extend(syms.into_iter().map(|v| SSARes(v.0)));
        }
    }
}

/// The kernel body: `bound` blocks (`gridDim.x`), with the block's first
/// operand bound to `blockIdx.x` as a kernel-level SSA value.
#[derive(Clone, Debug)]
pub struct Grid {
    pub bound: usize,
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
    pub fn new(name: String, grid_bound: usize, block: usize) -> Self {
        let mut k = Kernel {
            name,
            grid: Grid {
                bound: grid_bound,
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
pub struct KernelProgram {
    pub name: String,
    pub buffers: Vec<BufferDecl>,
    pub kernels: Vec<Kernel>,
    pub scratch_bytes: usize,
    /// Buffer id per module input index.
    pub input_bufs: Vec<BufId>,
    /// Buffer id per module output index.
    pub output_bufs: Vec<BufId>,
}

impl KernelProgram {
    pub fn buffer(&self, id: BufId) -> &BufferDecl {
        &self.buffers[id.0 as usize]
    }
}
