//! KernelIR: the lower-level, imperative representation.
//!
//! In contrast to the purely functional HIR, KernelIR is explicit about
//! memory and mutation: every tensor lives in a declared buffer with an
//! address space and a layout, kernels are annotated with a compute layout
//! (grid/block factorization of the iteration domain), reduces have been
//! lowered to sequential `for` loops over mutable accumulators, and results
//! are written through explicit stores.
//!
//! Layout attributes (`ParAttr` on `Stmt::Par`, `BufferDecl::layout`) are
//! `None` right after lowering and are filled in by the `layout_infer` pass;
//! `Stmt::Sync` statements are inserted by the `insert_sync` pass.

use crate::ir::{BinOp, ReduceOp, ScalarType};

/// SSA-ish scalar value inside one kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValId(pub u32);

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

/// Address space of a buffer. `Register` is reserved for future passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
    /// Block-local shared memory.
    Shared,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// Bound at runtime via `set_input(i, ptr)`.
    Input(usize),
    /// Bound at runtime via `set_output(i, ptr)`.
    Output(usize),
    /// Lives at `offset` bytes inside the scratch allocation.
    Scratch { offset: usize },
    /// Kernel-local shared memory, materialized by a `Stmt::Alloc`.
    Shared,
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
}

/// Compute layout: how the logical iteration domain maps onto the physical
/// grid/block hierarchy.
#[derive(Copy, Clone, Debug)]
pub enum LaunchShape {
    /// One thread per iteration point: `idx = blockIdx.x * blockDim.x +
    /// threadIdx.x`, guarded by `idx < n`. `OuterIdx` is `idx`.
    Flat { n: usize },
    /// `OuterIdx = blockIdx.x` (grid = n); thread-level parallelism inside
    /// the body is expressed with explicit `Stmt::Par` statements.
    Grid { n: usize },
}

#[derive(Clone, Debug)]
pub enum KExpr {
    ConstU32(u32),
    /// BabyBear constant, canonical representation.
    ConstField(u32),
    /// Linear index of the outer compute.
    OuterIdx,
    Bin {
        op: BinOp,
        /// Operand scalar type (selects field vs integer semantics).
        ty: ScalarType,
        lhs: ValId,
        rhs: ValId,
    },
    Select {
        cond: ValId,
        then_val: ValId,
        else_val: ValId,
    },
    Load {
        buf: usize,
        index: ValId,
    },
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// `const uint32_t v<dst> = <expr>;`
    Def { dst: ValId, expr: KExpr },
    /// Mutable accumulator for a reduction: `uint32_t v<dst> = v<init>;`
    DefAcc { dst: ValId, init: ValId },
    /// Sequential loop `for (v<var> = 0; v<var> < bound; ++v<var>)`.
    Loop {
        var: ValId,
        bound: usize,
        body: Vec<Stmt>,
    },
    /// Thread-parallel loop over `bound` logical indices; `v<idx>` is the
    /// logical index inside `body`. `attr` (from `layout_infer`) describes
    /// the factorization onto sequential steps × threads.
    Par {
        bound: usize,
        idx: ValId,
        attr: Option<ParAttr>,
        body: Vec<Stmt>,
    },
    /// Materializes a `BufferKind::Shared` buffer in the kernel.
    Alloc { buf: usize },
    /// `__syncthreads()`: block-level barrier between `Par` statements.
    Sync,
    /// `v<acc> = v<acc> <op> v<value>;`
    AccUpdate {
        acc: ValId,
        op: ReduceOp,
        ty: ScalarType,
        value: ValId,
    },
    /// `buf[v<index>] = v<value>;`
    Store {
        buf: usize,
        index: ValId,
        value: ValId,
    },
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    pub launch: LaunchShape,
    pub grid: usize,
    pub block: usize,
    /// Buffers appearing in the kernel signature, with write flag.
    pub params: Vec<(usize, bool)>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub struct KernelProgram {
    pub name: String,
    pub buffers: Vec<BufferDecl>,
    pub kernels: Vec<Kernel>,
    pub scratch_bytes: usize,
    /// Buffer id per module input index.
    pub input_bufs: Vec<usize>,
    /// Buffer id per module output index.
    pub output_bufs: Vec<usize>,
}
