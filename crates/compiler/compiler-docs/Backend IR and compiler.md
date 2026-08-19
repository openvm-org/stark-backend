
## compiler high-level passes and description

The compiler has three separate IRs: the Graph IR, the HIR kernel IR and the KIR lower-level kernel IR. 

The Graph IR is supposed to represent compute and memory operations on the op/kernel granularity, for a static graph with known sizes. This means that it contains nodes which represent `memcpy`, `memset`, black box kernels, kernels in kernel IR, and constants. The edges represent buffers with a constant size and alignment. 

Black box kernels are arbitrary blackbox functions that executes rust closures, while kernels in kernel IR contain HIR kernel modules. The goal of graph level optimizations is to perform fusion on kernels we know about (HIR nodes), and scheduling on the whole graph by assigning kernel nodes streams, and abstract buffers concrete offsets in memory.

Meanwhile, HIR and KIR are supposed to represent computations at the kernel level and split responsibility as follows: HIR contains abstract control flow statements like `compute` which is semantically a parallel-for and `reduce`, an associative reduction. A kernel module declares it's own inputs and outputs and internal typed buffers. When declaring typed buffers in HIR, one does not specify how it's located on device, it's up to the compiler to decide. KIR is where the compiler decides where the buffers are located on device (global, shared, registers), and also exposes mutation semantics that requires other GPU primitives like `__syncthreads`. 

The high-level flow of compilation is as follows:
1. the graph is built by inserting various nodes and edges into it. 
2. run canonicalization
- because of the way that HIR is designed, a single HIR module could contain multiple actual kernels. Because the compiler could decide to separate a single HIR expression into multiple kernels. Canonicalization canonicalizes each HIR module by splitting it as to maintain the invariant that **each HIR module in the graph iR contains only 1 logical kernel**. Throughout compilation, each logical kernel is an outer `compute [N] |i| {...}` expression, so the canonicalization will split modules that contain multiple outer compute expressions apart. This could be avoided if I merged HIR to graph IR into one IR, but it appears that the two have different representations: graph IR represents a graph where there is no lexical scope, whereas HIR requires lexical scope to denote kernel boundaries
3. lower reduce
- The compiler lowers `reduce` into a series of `compute` and `reduce` depending on some heuristics. This is necessary when some `reduce` statements are large, so we can use parallel associative reduction.
- This once again makes the HIR modules not canonicalized, since one could contain multiple outer computes. So run canonicalization again.
4. monomorphize
- currently HIR allows for symbolic shapes in certain locations (the bound of compute, the dimensions of inputs). This is to reduce the number of kernels compiled, which can blow up due to fusion.
- however, some of the locations are not valid to be symbolic for compilation. All nested inner computes have to have constant bounds (representing a tile on a thread block), and all inner dimensions of buffers have to be constant, as well as buffer dimensions produced by inner computes. For example `compute [N] |i| { let inner_buf = compute [K] |j| {...} }`. In this example `K` has to be compile-time constant. 
- when inserting the kernel node into the graph it is required that all symbolic parameters need to have known constants. So this pass monomorphizes the inserted kernel by substituting the constant for symbolic parameter for those positions that it's required to be constant
5. fusion: [see fusion design](https://hackmd.io/bOYRlxDJStib6TXOMXyepQ)
6. dce: dead-code-elimination (eliminates provably provable dead code)
7. plan memory
- runs memory planner or memory and stream co-scheduler

After all the graph passes run, we have a list of kernels in HIR that we want to compile to CUDA. The kernels passes that run so far are necessary to deduce enough information for CUDA codegen.
1. typecheck / canonicalize (this is actually run in the graph canonicalization pass)
- gets a type map and canonicalizes the HIR into the canonical form. The canonical form is a HIR that only contains 1 level of compute or 2 level of compute. I.e. `compute [N] |i| { elemwise-expr }` or `compute [N] |i| { let a = compute [M] |j| { elemwise-expr }; let b = compute [M1] { elemwise-expr }; ... }`. Where `elemwise-expr` are indexing expressions and arithmetic operations. 
2. lower-to-kir
- lowers to KIR. Which captures the GPU parallel hierarchy strictly in it's definition, with only two-levels of granularity in the parallelism: blocks and threads.
3. infer layout
- infers the layout of the buffers. Whether to put buffers distributed among registers in a block, or on shared memory. 
- currently it's a very naive algorithm
4. insert sync
- performs an overly conservative analysis and inserts `__syncthreads` where necessary.
5. plan shared
- plans shared memory
6. codegen
- emits CUDA

### Informal abstract syntax and type inference rules

#### HIR
```
Expr = 'compute' '[' SymExpr ']' '|' IDENT '|' '{' Expr '}'
     | 'reduce' '[' SymExpr ']' '|' IDENT '|' '{' Expr '}'
     | 'let' PAT '=' Expr 'in' Expr
     | ElemwiseExpr
     | '(' Expr (',' Expr)+ ')' // tuple 
     | '[' Expr (',' Expr)+ ']' // pack

SymExpr = INTEGER | IDENT | SymExpr BINOP SymExpr | UNARYOP SymExpr | '(' SymExpr ')'

PAT = IDENT | '(' IDENT (',' IDENT)+ ')'

ElemwiseExpr = IDENT '[' SymExpr ']' | SymExpr

```

Hopefully it's clear what the binary and unary operations are. 

The types are:

```
DType = Bool | UInt32 | BabyBear | BabyBearExt4 | ...

Tensor = DType '[' SymExpr (',' SymExpr)+ ']'

Tuple = ( (DType | Tensor) (',' (Dtype | Tensor))+ )

```

Where DType are the types of scalars, obviously you could add more, like UInt64, etc.

The type inference rules for an `Expr` with type `E`, informally are 

- if `Expr` is of the form `compute [n] |i| { e }`, and `e` is judged to have type `T`
  - if `T` is `Dtype`, then `E` is `T[n]`
  - if `T` is `Tensor`, with the form `H[x1, ..., xn]`, then `E` is `H[n, x1, ..., xn]`
  - if `T` is `Tuple`, with the form `(H1, ..., Hn)`, then `E` is the previous two rules applied componentwise to each `Hi`
- if `Expr` is of the form `reduce [n] |i| { e }` and `e` has type `T`, then `E = T`
- tuple and pack type rules are fairly obvious
  - pack expressions only allows `Expr` with `Dtype` as it's operands
  - tuple expressions does not allow `Expr` with `Tuple` type operands
  - note that we don't allow nested tuples because they don't add any additional semantic meaning. But they are useful for `compute/reduce` to denote fused computations
- element-wise type inference rules are fairly obvious, following standard programming language conventions

#### KIR

KIR follows a standard SSA form that's fairly standard. At this point I think it's more helpful to just include the rust code

```rust
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
```

Here it's illegal for `Par` blocks to contain another op with `Par`. The IR is structured the same as a kernel. The outer `Kernel` denotes a grid, and inner `par` ops denote a block. 
This IR is very similar to triton or cuTile, and I think is interchangeable with them. 

It is very much possible and perhaps preferable to lower to either triton or cuTile bytecode once those become more stable. The goal of the KIR at this level of representation is to bridge the HIR functional form and the lower level details, for the sake of fusion and scheduling optimizations. For fusion and scheduling we don't want to worry about synchronization that we would if we performed those optimizations on the level of CUDA.

But for now I chose to not use those bytecodes because there's still a performance gap for some cryptography specific operations expressed in those IRs (like NTT), and should any performance gap arise, there's no control from our side to alleviate those issues. Basically control is the answer. By implementing this ourselves we get more control at the cost of implementation burden. 

**This is important**: for register/compute heavy cryptography operations like the Bn254 scalar field, those compilers don't perform any instruction-cache aware optimization as far as I know, they inline everything, which blows up the instruction cache. There's no way to control the codegen process in those cases. To be fair there's no optimization in the current compiler that does this either, but when it comes up we could.

For example, if a kernel computes something with a lot of stages, like Bn254 poseidon permutation. Naively inlining all the functions would blowup the instruction cache. Instead we can be more strategic in which functions we inline and which we don't inline. (Which is a combinatorial search problem). From my experience nvcc doesn't do this very well, likely because it's search budget is very limited or it uses some heuristics that's tuned for CPU. We an afford more compilation time for the search.



# TODO and future work

## 1
make the kernel IR backend better. Currently the heuristics there are very naive and doesn't generate performant kernels for non-trivial nested kernels (such as NTT)

roughly what's missing is: 
- automatic vectorization (detect when vectorized loads are possible and emit wide instructions)
- pipelining (detect when work within a kernel can be overlapped, emit an pipelined schedule overlapping compute and memory)
- the layout problem: we need to select layouts and layout transformations that try to satisfy shared memory requirements, register requirements while reducing the data-transfers and synchronizations

## 2 
Better scheduling algorithm for concurrency and memory. 
The problem of co-scheduling streams and memory is stated more formally as follows: given a $$G = (V, E)$$, a DAG. There's a time $$t: V \to \mathbb R$$, and edge identity $$B: E \to \textbf{Buf}$$ that associates each edge with a buffer of some size in $$\mathbb N$$. The task is to produce a satisfying assignment on $$\textbf{Buf}$$, assignment each buf an offset, on $$V$$, assigning each vertex a stream id in $$S = \{1, ..., K\}$$, and assigning each vertex a start time. Such that the 
1. if two vertices overlap, then over every one of their bufs, the assigned intervals must not overlap. 
2. for every edge $$(u,v)$$, the end time of $$u$$ (which is the start time of $$u$$ plus $$t(u)$$) must be less than the start time of $$v$$. (order of vertices is a valid topological order over the DAG)

Then the objective is to minimize the max end time of all vertices, while also maintaining that the max end interval of all buffers by under a constant `M`. 

The current algorithm is very ad-hoc and is a greedy algorithm with constant look ahead. At least that's the idea. Don't ask me too much about it, it's AI generated.

## 3 
Integrate IR framework within openvm's stark-backend

## 4 
improve perf tooling, and visibility. Get a better sense what the fusion is doing, the optimizations that happen.

## 5
Formalize a textual representation of the IRs. This is important for compilers to have, because the textual IRs are how we examine the code generated and interact with the compiler. Currently the textual format is mostly AI chosen.

## 6 
Improve the kernel cost estimator.

Step 1: setup a suite of kernels and analyze the correlation of the estimated cost and actual cost (time). Depending on how tight this correlation is, improvements may be necessary.
Step 2: If the correlation is unsatisfactory, either improve the performance model by modeling the hardware more precisely. Or introduce tunable parameters in the performance model and train on the actual cost.
