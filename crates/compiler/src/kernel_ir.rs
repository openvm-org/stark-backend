//! KernelIR: the lower-level, imperative representation.
//!
//! In contrast to the purely functional HIR, KernelIR is explicit about
//! memory and mutation: every tensor lives in a declared buffer with an
//! address space and a (row-major affine) layout, kernels are annotated with
//! a compute layout (grid/block factorization of the iteration domain),
//! reduces have been lowered to sequential `for` loops over mutable
//! accumulators, and results are written through explicit stores.

use crate::ir::{BinOp, ReduceOp, ScalarType};

/// SSA-ish scalar value inside one kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValId(pub u32);

/// Address space of a buffer. The MVP only materializes global buffers;
/// `Shared` and `Register` are reserved for the tiling/fusion passes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AddressSpace {
    Global,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// Bound at runtime via `set_input(i, ptr)`.
    Input(usize),
    /// Bound at runtime via `set_output(i, ptr)`.
    Output(usize),
    /// Lives at `offset` bytes inside the scratch allocation.
    Scratch { offset: usize },
}

#[derive(Clone, Debug)]
pub struct BufferDecl {
    pub name: String,
    pub elem: ScalarType,
    /// Logical row-major shape; the physical layout is the identity affine
    /// map over this shape.
    pub shape: Vec<usize>,
    pub kind: BufferKind,
    pub space: AddressSpace,
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
    /// `OuterIdx = blockIdx.x` (grid = n); the body runs inside
    /// `for (InnerIdx = threadIdx.x; InnerIdx < m; InnerIdx += blockDim.x)`.
    GridBlock { n: usize, m: usize },
}

#[derive(Clone, Debug)]
pub enum KExpr {
    ConstU32(u32),
    /// BabyBear constant, canonical representation.
    ConstField(u32),
    /// Linear index of the outer compute.
    OuterIdx,
    /// Index of the inner (thread-level) compute.
    InnerIdx,
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
