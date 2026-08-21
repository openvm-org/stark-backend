# compiler high-level passes and description

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
5. fusion: see fusion design
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

## Informal abstract syntax and type inference rules

## HIR
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

## KIR

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


# crypto-compiler file structure

`---begin AI generated description---`

## Entry points

The crate exposes two entry points:

- **`module_compiler::ModuleCompiler`** — per-kernel backend. Consumes a
  single-kernel, already-canonical, already-monomorphized `ir::Module` and
  emits a dlopen'd `runtime::KernelProgram`. Strictly pure lowering: no
  rewrites, no fusion, no monomorphization.

- **`graph_compiler::GraphCompiler`** — graph-level driver. Consumes a
  `graph_ir::GraphBuilder`, runs the graph pass pipeline
  (`lower_reduce` → `monomorphize` → `canonicalize` → optional fusion →
  dce → `plan_memory`), JITs every unique residual module in parallel
  through `ModuleCompiler`, and packages the whole thing into a
  `graph_exe::GraphExe` backed by one fixed-offset device pool
  (compatible with CUDA-graph capture). Feature-gated on `planner`.

## `src/` — core IR and infrastructure

### IR layers (three of them, not two)

- **`ir.rs`** — HIR. Pure functional expression DAG built through
  `IRBuilder`: `compute` (parallel map), `reduce`, `let`, indexing,
  elementwise ops, tuples. Hash-consed for CSE.
- **`graph_ir.rs`** — graph-level IR: kernels, memcpys, memsets, consts
  over device `BufId`s. `GraphNode::Kernel` wraps an `ir::Module` with
  explicit input/output `BufId` bindings; `BlackboxKernel` wraps opaque
  host closures for hand-written CUDA. 
- **`kernel_ir.rs`** — KIR (`KirProgram`), the lower imperative IR. SSA, memory- and schedule-aware: `par`, `for`, `alloc`,
  loads/stores, sync. Consumed by `layout_infer` → `insert_sync` →
  `plan_shared_mem` → `codegen`.

### Support

- **`quast.rs`** — `Quast` (concrete quasi-affine over `VarId`) and
  `SExpr` (symbolic module parameters). Used for buffer sizes, access
  relations, index expressions everywhere.
- **`field_ext.rs`** — DSL-side field-inverse primitives for BabyBear
  and its degree-4 binomial extension. Emits into the caller's
  `IRBuilder`; no host closure.
- **`kernels.rs`** — prebuilt DSL modules: radix-2 DIT NTT and
  Poseidon2-16 Merkle compression tree over BabyBear.
- **`poseidon2_parallel.rs`** — warp-parallel variant of Poseidon2-16
  sliced along the lane axis; cross-element reads become warp shuffles.
- **`dump.rs`** — pretty-printers. `dump_hir` mirrors DSL syntax;
  `dump_kernel_ir` prints KIR in MLIR-style with SSA names matching
  generated CUDA.
- **`test_utils.rs`** — test harness for graph compile+run: hides JIT
  boilerplate, supports pool mode and direct-drive mode, handles
  Montgomery encode/decode.

### Compile & runtime plumbing

- **`module_compiler.rs`** — `ModuleCompiler` — the per-kernel entry
  point described above.
- **`runtime.rs`** — JIT runtime: writes CUDA C++ to a temp dir, shells
  out to `nvcc`, `dlopen`s the resulting `.so`, wraps the C ABI in a
  safe `KernelProgram`. Integrates with `openvm-cuda-common` buffers /
  streams.
- **`module_hash.rs`** — deterministic SHA3-256 structural fingerprint
  of an `ir::Module` for cross-process kernel cache identity.
- **`kernel_cache.rs`** — on-disk LRU cache of JIT artifacts (`.so` +
  CUDA source + metadata), keyed by module hash. Bounded by count and
  bytes. Feature-gated on `planner`.
- **`graph_compiler.rs`** — `GraphCompiler`. Builder-pattern driver
  that runs the graph pipeline, compiles unique residual modules in
  parallel through `ModuleCompiler`, plans memory, and hands off a
  `GraphExe`. Feature-gated on `planner`.
- **`graph_compiler_config.rs`** — `GraphCompilerConfig` — the TOML
  schema behind `GraphCompiler::from_toml`. Feature-gated on `planner`.
- **`graph_exe.rs`** — `GraphExe`. Owns every JIT'd `KernelProgram`, the
  static memory plan, and the unified device pool that backs every
  buffer at a fixed offset. Supports CUDA-graph capture / replay via
  `launch_graph`. Feature-gated on `planner`.
- **`graph_info.rs`** — `GraphInfo` — per-node timings and graph hash,
  serializable, consumed by later replans (via `GraphCompiler::node_times`)
  and by the cytoscape / `sim_scheduler` overlays. Feature-gated on
  `planner`.
- **`graph_serializer.rs`** — `SerializableGraphBuilder`: bincode wire
  format for a `GraphBuilder` (pre-pass or post-fuse+dce), used by the
  sumcheck bench harness (`load_or_compile_and_dump`) and by
  `sim_scheduler`. Feature-gated on `planner`.

## `src/passes/` — compiler passes

Pipeline order (mirrors `passes/mod.rs` and the module docs there).

### Single-module (HIR → KIR → CUDA)

- **`type_infer.rs`** — bottom-up type inference over the HIR DAG.
- **`monomorphize.rs`** — substitutes concrete values for a module's
  symbolic parameters. Outer compute bounds may stay symbolic; inner
  bounds, loop sizes, and reduce lengths must be concrete before
  lowering.
- **`canonicalize.rs`** — flattens arbitrary nesting into the canonical
  form from `old_plans/design.md`: an ordered sequence of let-bound
  computes, each with at most one inner compute (or a reduce), reduces
  have scalar bodies. Merges deep nests by rewriting innermost pairs.
- **`parallel_reduce_rewrite.rs`** — block-reduce lowering for
  under-parallel reduces: rewrites to a two-stage tree (per-thread
  sequential accumulation + halving butterfly) when the outer bound is
  below saturation.
- **`lower_to_kir.rs`** — HIR (canonical) → `KirProgram`. Outputs
  become global buffers, each canonical compute becomes one kernel,
  inner let-bound computes become shared-memory buffers, par operands
  list every captured value from the enclosing scope.
- **`par_attr_infer.rs`** — assigns each par a `(spatial, lane, warp)
  → logical` par-attr by propagating read layouts forward. Sub-pass of
  `layout_infer`.
- **`layout_infer.rs`** — driver for the layout inference pipeline:
  fills in `par_attr` / `alloc_attr` left empty by lowering, promotes
  shared tiles to registers when the write access is invertible and
  non-grid-spanning, and inserts `ConvertLayout` ops where the reader's
  expected layout doesn't match the producer's. See
  `compiler-docs/Layout Inference Algorithm.md` for the algorithm.
- **`layout_cost.rs`** — hardware-specific cost constants
  (`ConversionCostModel`) shared by `par_attr_infer`, `shared_swizzle`,
  and `convert_decompose`. Currently tuned for RTX 5090 / GB202.
- **`shared_swizzle.rs`** — `choose_shared_layout` picks a
  bank-conflict-minimizing shared-memory swizzle for a given set of
  accesses (paper §5.4 + Appendix 9.2). Called by
  `allocate_convert_scratch` and by `layout_infer` for multi-access
  shared tiles.
- **`convert_decompose.rs`** — `best_decomposition` scores each
  `ConvertLayout` op across the Copy / Slot / Shuffle / Bounce
  strategies (paper §5.4) and returns the winner for codegen.
- **`allocate_convert_scratch.rs`** — sizes each `ConvertLayout`'s
  shared scratch buffer as a pure function of the `(src, dst)` layout
  pair (full tile iff `Strategy::Bounce` wins, 0 otherwise). Runs after
  `layout_infer`.
- **`insert_sync.rs`** — inserts `__syncthreads()` barriers before
  reads-after-shared-writes and around writes to aliased shared
  buffers.
- **`plan_shared_mem.rs`** — liveness-based packing of shared buffers
  onto shared memory offsets to minimize peak footprint.
- **`codegen.rs`** — `KirProgram` → CUDA C++. Emits the C ABI expected
  by `old_plans/design.md`, handles Montgomery arithmetic for BabyBear,
  uses inline PTX for mul reduction. `ConvertLayout` is realized via
  `best_decomposition`'s winning strategy.
- **`verify.rs`** — structural KIR checks: SSA, par primitiveness (no
  nesting, no statement loops, no syncs inside pars).
- **`check_accesses.rs`** — optional exhaustive access validation for a
  concrete instantiation (bounds + write injectivity). Gated by
  `CompileOptions::check_accesses`.

### Graph-level

- **`split_module.rs`** — splits a multi-kernel HIR module into
  single-kernel modules wired by edges, so each residual can go through
  `ModuleCompiler`.
- **`fusion_utils.rs`** — shared helpers for graph rewrites: `dce`
  (graph-level dead-code elimination), `FusionReport`, HIR traversal
  helpers used across `fusion/`.
- **`utils.rs`** — shared helpers used across passes.
- **`inplace.rs`** — empty placeholder.

### Fusion (`src/passes/fusion/`)

Enumerate–score–extract rewrite pipeline (design:
`compiler-docs/High-level Fusion Design.md`, long-form spec:
`old_plans/detailed-fusion-plan-v2.md`). Selectable from
`GraphCompiler::fusion_options` / `without_fusion`.

- **`mod.rs`** — module glue and public re-exports (`fuse_graph`,
  `FusionOptions`, `FusionReport`, `ArtifactKey`, ...).
- **`driver.rs`** — `fuse_graph` — top-level bounded-saturation loop:
  freeze seed → enumerate candidates per round → dedup by `CandidateKey`
  → validate acyclicity → insert accepted candidates → extract → apply.
- **`model.rs`** — `GraphFuser`, `AltGraphNode`, `ValueClassId`,
  `NodeId`. Versioned bipartite alternative graph over dense arenas.
  Seed and synthesized candidates share one namespace; sidecar state
  (origins, costs, artifact keys) lives outside the model.
- **`saturate.rs`** — saturation bookkeeping: seed-origin tracking (for
  disjointness) and cross-round dedup by `CandidateKey`.
- **`version.rs`** — versioned graph guard: wraps/unwraps a
  `GraphBuilder` into/out of the alternative graph.
- **`access.rs`** — `AccessRelation` binding reads/writes to value
  classes.
- **`validate.rs`** — acyclicity and storage-hazard checking, decoupled
  from `model` so revisions don't mutate the alternative graph.
- **`apply.rs`** — turns an `ExtractionSolution` back into a
  `GraphBuilder`; adds RAW/WAW/WAR storage-hazard edges; re-validates.
- **`tests.rs`** — module-local unit tests.

`fusions/` — candidate producers. Each pass discovers matches over the
frozen seed prefix, synthesizes candidate HIR, normalizes, and returns
draft `AltGraphNode`s to the driver:

- `producer_consumer.rs`, `fanout.rs`, `horizontal.rs`, `epilogue.rs`,
  `small_kernel.rs`, `mod.rs`.

`cost/` — cost sidecar consumed by the extractor:

- `mod.rs` (`GraphNodeCost`, `ArtifactKey`, `ArtifactContext`,
  `KernelCostManager`), `estimator.rs`, `interpreter.rs` (KIR
  interpretation for per-node cycle estimates), `liveness.rs` (register
  liveness estimate), `transactions.rs` (memory-transaction estimate),
  `cache.rs` (memoization by module hash + bindings).

`extract/` — extraction backends:

- `mod.rs` (`ExtractionData`, `ExtractionSolution`,
  `FallbackReason`), `cpsat.rs` (CP-SAT solve, gated on
  `planner-ortools`; falls back to `brute` when unavailable),
  `brute.rs` (exhaustive enumerator; correctness oracle for small
  inputs).

## `src/planner/` — memory & stream planner

Feature-gated on `planner`. Picks execution order, stream assignment,
and per-buffer byte offsets to minimize wall time under a memory bound.
Emits a unified `StreamMemoryPlan`. See
`compiler-docs/Stream & Memory co-scheduler.md` for the algorithm.

- **`mod.rs`** — public entry, `SchedulerMode` selector (`ListV1` /
  `ListV2`) and the `plan(atg, mode) -> StreamMemoryPlan` dispatcher.
- **`plan.rs`** — `StreamMemoryPlan` / `StreamInstr` output types.
- **`abstract_timing.rs`** — `AbstractTimingGraph` (the scheduler input
  layer: buffers, per-node timings, producers/consumers, alias
  classes) plus `access_from_node`, `eval_size`, `perf_est`, and
  `load_abstract_timing_graph` (the offline loader used by
  `sim_scheduler` and the abstract-planner examples).
- **`list_v1.rs`** — profile-guided list scheduler with depth-`k` beam
  look-ahead, multi-stream `WaitOn` insertion, and offline best-fit
  memory packing.
- **`list_v2.rs`** — persistent-beam list scheduler with parallel
  fan-out over ready-set × streams (uses the `im` persistent
  collections for O(1) beam clones). Post-pass replays the assignment
  into a `StreamMemoryPlan` with per-stream ordering, cross-stream
  event assignment, and best-fit-decreasing memory packing.
- **`validate.rs`** — `validate_plan`: checks the emitted schedule for
  missing cross-stream syncs, data-dep races, and pool-lifetime
  overlaps. Used by `sim_scheduler`.

## `macros/` — `kernel!` proc macro

`macros/src/lib.rs` — concrete syntax for the DSL:
`compute [n] |i| { … }`, `reduce`, `let v = e; …`, `if c then e else
e'`, tensor indexing, `#[scatter(…)]` / `#[par(…)]` / `#[grid(…)]`
compute attributes, `#c` constant splices, function calls (where `foo`
in the macro means `foo(ib, …)`). Expands to `IRBuilder` method calls
producing `NodeId`s.

## `tests/`, `examples/`, `benches/`

- **`tests/gpu.rs`** — single-kernel GPU tests.
- **`tests/gpu_graph.rs`** — multi-kernel graph compile + run tests.
- **`tests/gpu_macro.rs`** — the large macro-driven end-to-end DSL
  test suite (write these first when adding compiler features).
- **`tests/convert_layout.rs`** — layout-conversion codegen tests.
- **`tests/custom_kernels.rs`** — placeholder.
- **`tests/graph_serializer.rs`** — round-trip tests for the
  `SerializableGraphBuilder` bincode format.
- **`examples/dump_ntt_cuda.rs`** — dump generated CUDA for the NTT
  kernel.
- **`examples/ntt_scale_graph.rs`** — NTT graph construction and
  scaling (requires `planner`).
- **`examples/bench_abstract_planners.rs`** — offline planner
  benchmark: runs the heuristic + `list_v1` at several
  `max_concurrency` levels against a captured `graph.bin` +
  `timings.json` (requires `planner`).
- **`examples/bench_list_v2_ops.rs`** — micro-benchmarks `list_v2`'s
  per-step primitives (`cost`, `put_on`, `state.clone()`) (requires
  `planner`).
- **`examples/profile_ntt_supra.rs`** — profiling harness for
  Supra-compiled NTT.
- **`examples/tmp_ntt_kdist.rs`** — scratch driver for NTT
  distribution experiments.
- **`benches/ntt.rs`**, **`ntt_supra_sweep.rs`**, **`poseidon2.rs`** —
  Criterion benchmarks.
- **`src/bin/sim_scheduler.rs`** — offline replay of the memory +
  stream planner against a captured graph and timings JSON. See
  `tools.md` for usage.

## Design docs and porting guides

These docs describe intent and history; the code is the source of
truth when they disagree.

### `compiler-docs/` — current design references

- **`Backend IR and compiler.md`** — overall compile pipeline, HIR
  syntax and type-inference rules, KIR SSA layout.
- **`High-level Fusion Design.md`** — build/score/pick fusion
  architecture (matches the current `passes/fusion/` implementation).
- **`Layout Inference Algorithm.md`** — layout inference algorithm and
  `ConvertLayout` decomposition, including the shared-swizzle
  bank-conflict optimization.
- **`Stream & Memory co-scheduler.md`** — `AbstractTimingGraph` /
  `StreamMemoryPlan` interface and the `list_v2` beam search.

### Root-level guides

- **`gpu_ir_porting_guide.md`** — porting eager CUDA code to the graph
  IR (blackbox kernels, `insert_memcpy`/`insert_memset`, avoiding
  data-dependent host values).
- **`tools.md`** — user guide for `GraphCompilerConfig`, graph dumps
  (text + cytoscape), the graph visualizer, `SerializableGraphBuilder`
  format, and `sim_scheduler`.
- **`notes.md`** — miscellaneous notes: graph mutation semantics,
  benchmark harness (env vars, nsys flags, per-bench command
  templates).

### `old_plans/` — historical and reference specs

- **`design.md`** — original architecture: DSL, canonical form, KIR,
  layout system, compile flow.
- **`detailed-fusion-plan-v2.md`** — long-form spec, the operative
  reference for `passes/fusion/`.
- **`fusion-plan.md`, `fusion-plan-v2.md`, `fusion-plan-v2-agent-gen.md`,
  `fusion-v2-progress.md`, `fusion-v2-architecture.html`, `fusion_extension.md`**
  — historical fusion notes.
- **`refactor-plan.md`** — running refactor plan (planner feature
  gating, graph-exe split, etc.).
- **`mutation_semantics.md`** — semantics behind SSA restoration on
  in-place kernels. (The `restore_ssa` pass this doc describes has
  since been folded into other passes / builder-time bookkeeping.)
- **`stream-scheduling.md` / `.html`** — multi-stream list scheduler
  design.
- **`kernel_ir_porting_guide.md`** — porting notes for KIR.
- **`kernel_ir_gaps.md`**, **`kernel_ir_layout_progress.md`** —
  layout-inference work-in-progress notes.
- **`graph-ir.md`** — early graph IR spec (slightly out of date on
  field names).
- **`layout_optimization_problem.md`** — layout inference problem
  statement.
- **`gkr-small-round-overlap-plan.md`** — GKR pipelining plan.

## Feature flags

- `planner` (default) — enables `planner/`, `graph_compiler`,
  `graph_exe`, `graph_info`, `graph_serializer`, `kernel_cache`.
  Without it only the per-kernel `ModuleCompiler` surface is available.
- `planner-ortools` — enables the CP-SAT fusion extractor in
  `passes/fusion/extract/cpsat.rs`. Links against OR-Tools; see
  `Cargo.toml` for install locations / `ORTOOLS_PREFIX`. Without it,
  the fusion pass falls back to the brute-force extractor for small
  graphs and to the original graph otherwise
  (`FallbackReason::SolverUnavailable`).
  
`---end AI generated description---`

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

### Problem update:

It should be possible decompose the problem to a memory aware multi-processor scheduling problem. In this formulation we assign each vertex a stream (a processor), and start times, and capture the requirement that overlapping buffers cannot share memory (stated informally), but without specifying the scheduling on the memory.

Then once we have a satisfying assignment, the schedule, which specifies an ordering of nodes over all the streams. Then we can pack memory using cuda VMM API. To be more precise, the schedule admits a sequence of `alloc` and `free` statements at various times, and it is possible to pack all the memory without fragmentation due to paging, with some amount of internal fragmentation due to page size. So for each buffer we can reserve a VPMM address range, alias the right physical pages to it, and don't worry about freeing the virtual address range.
- a possible optimization is to first pack large buffers above the page size, then do another round either with ILP or heuristics that pack small buffers into gaps

Even with this formulation the problem is still NP-hard. 


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

