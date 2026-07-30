use crate::{
    graph_ir::GraphBuilder,
    ir::{BinOp, IRBuilder, Node, NodeId, ReduceOp, ScalarType, Type, VarId},
    kernel_ir::{
        Access, AddressSpace, BufId, BufferDecl, BufferKind, IndexMap, Kernel, KernelProgram,
        ParAttr, SSABlock, SSANode, SSAOp, SSAOpCode, SSARes,
    },
    passes::{
        canonicalize::{is_canonicalized, CanonKernel, CanonValue, Program, ResultExpr, TensorRef},
        plan_global_scratch::GlobalScratchPlan,
        utils::resolve_tensor_ref,
    },
    quast::{self, ParSpec, Quast, ScatterStore},
    CompileError,
};

/// this struct represents the access relations of a canonical form kernel:
/// forall i st. 0 <= i < outer_bound, to produce a set of outputs (B_1[f_1(i, j_1)], ...,
/// B_m[f_m(i, j_m)]), this kernel needs to read the set (A_1[f_{1+m}(i, j_{1+m})], ...,
/// A_n[f_{n+m}(i, j_{n+m})]). where 0 <= j_k < inner_bounds[k] forall 1 <= k < n + m
struct AccessRelation {
    outer_id: VarId,
    outer_bound: usize,

    inner_ids: Vec<VarId>,
    inner_bounds: Vec<usize>,

    write_bufs: Vec<VarId>,
    read_bufs: Vec<VarId>,

    /// let A be the ith buffer read from, represented by read_bufs[i]
    /// let f be the read expression from A, represented by read_bufs[i]
    /// then the set read from is A[f(outer_id, inner_id_{i})]
    read_accesses: Vec<Quast>, // this number of reads

    /// let B be the ith buffer written to, represented by write_bufs[i]
    /// let f be the write expression to B, represented by write_accesses[i]
    /// then the set written to is B[f(outer_id, inner_id_{i + num_reads})]
    write_accesses: Vec<Quast>, // this number of writes

    /// book-keeping
    read_nodes: Vec<NodeId>,
    last_exprs: Vec<NodeId>, // final tile before the end of the outer compute block
}

/// let a = compute [N] |i| { ...exprs; tile } // src
/// let b = compute [N] |i| { ... compute [M] |j| { ...a[f(i, j)]... } } // dst
/// -->
/// let b = compute [N] |i| { ...exprs; ... compute [M] |j| { ...tile[g(j)]... } }
struct FusionInfo {
    access_rewrites: Vec<(NodeId, NodeId, Quast)>, // (src_last_expr_node_id, dst_read_node_id, g)
}

pub fn can_fuse(
    src: &AccessRelation,
    dst: &AccessRelation,
) -> Option<(AccessRelation, FusionInfo)> {
    todo!()
}

pub fn apply_fusion(b: &mut GraphBuilder, node_a: usize, node_b: usize) {
    todo!()
}
