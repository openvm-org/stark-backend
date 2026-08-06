# Kernel Fusion Framework V2

This document replaces the greedy pair-at-a-time selection in
[`fusion-plan.md`](fusion-plan.md) with two separate phases:

1. enumerate and retain multiple legal implementations of each graph value; and
2. choose one globally consistent implementation with CP-SAT or an ILP solver.

The v1 access analysis and module surgery remain useful. V2 changes how candidates are
represented, composed, costed, and selected. It does not by itself make the v1
`General × General` case legal; that still needs the polyhedral machinery described in the
first plan.

The intended pipeline is:

```text
GraphBuilder
  -> graph normalization and SSA versioning
  -> import into GraphFuser
  -> alternative saturation
  -> cost every retained alternative
  -> global extraction
  -> materialize one GraphBuilder
  -> DCE, verification, and memory planning
```

## 1. Required input invariants

`GraphFuser` operates on a value-SSA graph, even though the author-facing graph API may allow
mutation:

- A registered input or a node result defines each value exactly once.
- A mutating operation consumes the pre-mutation value and produces a fresh carried result.
  Every downstream operation consumes that carried result, never the old edge.
- The versioned value graph is acyclic.
- Registered outputs refer to the final versions of their values.
- Values in the same equivalence class have identical device, element type, shape, and byte
  size. Unioning differently typed values is an error.
- Structured kernels are pure except for their declared results.
- An opaque operation with effects not represented by its data results also consumes and
  produces an artificial effect token. The final token is an extraction root. This keeps
  opaque operations live and ordered.

The graph may be refactored to establish these invariants before constructing the fuser.
In-place `BufId` reuse must not leak into the fuser: the old value and carried value are
different fuser classes even if materialization later assigns them the same physical storage.

Fusion should run after the normalization assumed by
[`refactor-plan.md`](refactor-plan.md): reduce lowering, minimal monomorphization, type
checking, and canonicalization. Every retained fused module is normalized again before it is
inserted.

## 2. Representation: a typed multi-output e-graph

This is more precisely an AND/OR hypergraph than a conventional expression e-graph:

- a `BufClassId` is an OR-node: choose one representative for this value;
- a `FuserNodeId` is an AND-node: choosing one of its results requires all of its operands;
- one fuser node may produce several result representatives and is charged once even when
  several results are used.

IDs for classes, representatives, and nodes must be separate. A class ID must not wrap a
`BufId` because union-find canonicalization can change the class while a representative keeps
its identity.

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufClassId(usize);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufRepId(usize);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct FuserNodeId(usize);

pub struct GraphFuser {
    classes: Vec<BufClass>,
    reps: Vec<BufRep>,
    nodes: Vec<FuserNode>,

    // Only needed when rules can prove two previously distinct classes equal.
    uf: UnionFind,

    // Canonical class -> result ports and operand ports.
    producers: HashMap<BufClassId, Vec<ResultUse>>,
    consumers: HashMap<BufClassId, Vec<OperandUse>>,

    // Semantic node key -> canonical node. Prevents rediscovering the same
    // fused module with the same canonical operands and result classes.
    node_intern: HashMap<NodeKey, FuserNodeId>,
    attempted_fusions: HashSet<FusionKey>,
    worklist: Vec<FuserNodeId>,

    relation_cache: HashMap<AnalysisKey, Arc<NodeAnalysis>>,

    input_roots: Vec<BufRepId>,
    output_roots: Vec<BufClassId>,
    effect_root: Option<BufClassId>,
}

pub struct BufClass {
    pub info: BufInfo,
    pub reps: Vec<BufRepId>,
}

pub struct BufRep {
    pub class: BufClassId,
    pub info: BufInfo,
    pub origin: RepOrigin,
}

pub enum RepOrigin {
    GraphInput { interface_pos: usize },
    NodeResult { node: FuserNodeId, output_pos: usize },
}

pub struct FuserNode {
    pub kind: FuserNodeKind,
    pub operands: Vec<BufClassId>,
    pub results: Vec<BufRepId>,
    pub support: OriginSet,
    pub artifact_key: Option<ArtifactKey>,
    pub analysis: Arc<NodeAnalysis>,
}

pub struct ResultUse {
    pub node: FuserNodeId,
    pub output_pos: usize,
    pub rep: BufRepId,
}

pub struct OperandUse {
    pub node: FuserNodeId,
    pub input_pos: usize,
}
```

`support` is the set of original graph nodes represented inside an alternative. It is used to
reject recursive/overlapping rewrites and to cap fusion size. `ArtifactKey` is the actual
post-normalization, post-monomorphization compilation-cache key, including relevant backend
options. It is not merely the pre-fusion module hash.

For fusion-only saturation, new result representatives can be inserted directly into the
consumer result classes. No union operation is necessary. Keep union-find only if future
rules can discover equality between two existing classes. If union is retained, every public
lookup must canonicalize its class ID, and a rebuild must canonicalize node keys and the
producer/consumer indexes after a union.

### Parallel insertion

Workers must not allocate global IDs or mutate union-find. They inspect an immutable epoch and
return pending substitutions:

```rust
pub struct PendingSubst {
    pub operands: Vec<BufClassId>,
    pub result_classes: Vec<BufClassId>,
    pub result_infos: Vec<BufInfo>,
    pub kind: FuserNodeKind,
    pub support: OriginSet,
    pub analysis: Arc<NodeAnalysis>,
}
```

A single coordinator then:

1. canonicalizes all class IDs against the current union-find;
2. rejects stale substitutions whose legality assumptions no longer hold;
3. sorts by a deterministic structural key;
4. interns the node;
5. allocates one result representative per result class;
6. updates producer/consumer indexes and the worklist; and
7. performs a rebuild if any classes were unioned.

This preserves the immutable-worker design without making insertion order affect the result.

## 3. Access analysis

The shortened `ReadRelation { read }` / `WriteRelation { write, inv }` form is insufficient.
Fusion needs to know the operand or result port, the HIR site to replace, binder bounds, and
the dynamic execution count of each read. Retain the information already discovered in v1:

```rust
pub struct NodeAnalysis {
    pub accesses: AccessRelation,
    pub exec_cost: ExecCostEstimate,
    pub kernel_class: KernelClass,
    pub artifact_key: Option<ArtifactKey>,
}

pub struct AccessRelation {
    pub domain: IterationDomain,
    pub reads: Vec<ReadRelation>,
    pub writes: Vec<WriteRelation>,
}

pub struct ReadRelation {
    pub input_pos: usize,
    pub site: ir::NodeId,
    pub index: Option<Quast>,
    pub binders: Vec<(ir::VarId, usize)>,
    pub dynamic_evals: u64,
}

pub struct WriteRelation {
    pub output_pos: usize,
    pub write: Quast,
    pub inverse: Option<Quast>,
    pub len_elems: u64,
    pub elem_bytes: u32,
}
```

An identity write gets an automatically derived inverse. An explicit scatter uses the
author-supplied inverse from the symbolic-IR refactor. The compiler validates
`inverse(write(i)) = i` and `write(inverse(p)) = p` whenever the maps concretize to a
representation for which exhaustive or algebraic validation is available. A trusted
`SExpr`/black-box inverse must be marked as such in the analysis; it must not silently look
proved.

Analysis is per node alternative, not per value class. Its cache key includes the normalized
module hash, concrete residual bindings that affect analysis, operand/result types, and the
backend cost-model version.

## 4. Saturation

### Seed graph

Import one terminal representative for each registered input and one fuser node for every
original SSA operation. A node result initially gets its own class. Registered outputs map,
in interface order, to the classes of their final SSA versions.

Opaque nodes participate in extraction but not fusion. If an opaque node has unmodelled
effects, its artificial effect-token result makes it part of the rooted value graph.

### Fusion rewrite

Consider a particular producer alternative `P`, result port `p` in class `B`, and a particular
consumer alternative `C` with one or more operand ports in `B`. This is an alternative-pair
rewrite, not a rewrite between whole classes.

A candidate is legal when:

1. both alternatives are structured kernels supported by the v1 dispatch table;
2. every consumer read through the fused operand ports has an analyzable index;
3. the producer output write map is invertible as required by the selected v1 case;
4. the producer result can be scalarized by the existing module surgery;
5. `P.support` and `C.support` are disjoint;
6. replacing the seam cannot create a value or effect-token cycle;
7. the fused body passes the body-size and per-kernel resource guards; and
8. normalization, type checking, access checking, and re-analysis of the fused module succeed.

SSA versioning subsumes the old insertion-order WAR test: the producer alternative explicitly
names the version it reads. Output aliasing is likewise checked on value versions. Effect
tokens still prohibit moving a pure-looking rewrite across an opaque effect.

The rewrite constructs:

- the fused module produced by the v1 grafting logic;
- operands equal to the external dependencies of `P ∪ C` after removing the internal seam,
  in deterministic module-input order;
- one fresh representative for each result of `C`, inserted into the corresponding existing
  result class; and
- `support = P.support ∪ C.support`.

Initially keep the v1 restriction that the producer has one scalarizable data result.
Multi-output consumers are supported because the fused node returns the same result tuple as
the consumer. Supporting a multi-output producer later requires selecting and scalarizing one
result expression while correctly charging stores and retaining the other producer results.

Every fused module is a real, validated module. Extraction never selects a symbolic
“fusion action” that still needs to be applied.

### Worklist and candidate keys

When a node is inserted, compare each result class with existing consumers and each operand
class with existing producers. Record an attempted key containing:

```text
(producer node, producer output port,
 consumer node, fused consumer input-port set,
 canonical seam class, fusion-rule version)
```

This avoids rescanning all pairs every round. A new alternative can participate in later
fusions on either side, which is what builds multi-kernel fusion chains.

### Termination and explosion control

The set of subgraphs of a finite DAG is finite but exponential. The implementation therefore
needs explicit, deterministic limits:

- maximum original nodes in `support`;
- maximum HIR body operations;
- maximum alternatives per result-class tuple;
- maximum saturation wall time and total fuser nodes;
- maximum estimated registers and shared memory per fused kernel; and
- optional beam width after dominance pruning.

A safe dominance rule may remove node `A` in favor of `B` only when they have identical
canonical operand classes, result classes, observable/effect behavior, and future-fusion
capabilities, while `B` is no worse in every cost/resource dimension. Lower runtime alone is
not enough if the alternatives have different access relations and therefore different
future fusion opportunities.

With finite limits, “optimal” means optimal over the retained alternative graph, not over all
legal fusion trees. Reports must include which cap stopped saturation.

## 5. Cost model

Cost is attached to a fully formed alternative. Do not reconstruct fused cost by summing
pairwise savings: the fused module is available and can be analyzed directly.

```rust
pub struct ExecCostEstimate {
    pub launches: u32,
    pub blocks: u64,
    pub threads_per_block: u32,
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
    pub weighted_ops: u64,
    pub critical_path_cycles: u64,
    pub body_ops: u32,
    pub registers_per_thread: Option<u32>,
    pub shared_mem_per_block: u32,
    pub runtime_ns: i64,
    pub confidence: CostConfidence,
}

pub struct ArtifactCostEstimate {
    pub compile_ns: i64,
    pub code_bytes: u64,
}
```

`grid_dim` and `block_dim` are not single `u32` values on CUDA; store either `Dim3` or the
derived block/thread counts. The dynamic work domain, rather than “number of output
elements,” is the source of truth for kernels with packs, reductions, or multiple outputs.

A first calibrated runtime estimate can be:

```text
resident_blocks_per_sm =
    min(block_limit,
        thread_limit / threads_per_block,
        register_limit / registers_per_block,
        shared_mem_limit / shared_mem_per_block)

waves       = ceil(blocks / (sm_count * resident_blocks_per_sm))
T_memory    = (dram_read_bytes + dram_write_bytes) / effective_bandwidth
T_compute   = weighted_ops / effective_weighted_ops_per_sec
T_latency   = waves * critical_path_cycles / clock_hz
T_runtime   = launches * launch_latency + max(T_memory, T_compute, T_latency)
```

This is still an approximation, but it fixes three problems in the earlier sketch:

- occupancy depends on registers, shared memory, warps, and blocks, not only
  `SM count × max threads per SM`;
- read and write traffic is counted over actual dynamic sites, not as one
  `bytes_read_per_output` scalar; and
- critical-path latency is not multiplied by the read count a second time if the path
  estimate already contains those reads.

Use measured device parameters and version the calibration. Until register estimates are
available before code generation, use body-size/resource guards and mark confidence low.

Compilation cost belongs to a unique `ArtifactKey`, not to every graph node. Two selected
nodes that deduplicate to one compiled artifact pay the compilation term once. This is
essential because reducing redundant JIT work is one of V2's primary goals.

The objective also needs a horizon. Let `expected_runs` be the expected executions of the
compiled graph and `cache_miss_probability` the probability that an artifact must be built.
Then runtime and compilation have compatible time units:

```text
expected total time =
    expected_runs * graph runtime
    + cache_miss_probability * compilation work
```

If compile wall time rather than total compiler work is important, the sum of artifact
compile times is only a proxy because compilation is parallel. A later model can approximate
`sum / workers + max_artifact_time` or explicitly schedule compile jobs.

## 6. Extraction ILP

### Sets and notation

Let:

- `E` be canonical buffer classes;
- `R_e` be the representatives in class `e`;
- `V` be fuser nodes;
- `operands(v)` be the distinct operand classes of node `v`;
- `results(v)` be its result representatives;
- `class(r)` be the class containing representative `r`;
- `owner(r)` be the node producing a non-terminal representative;
- `users(e)` be nodes that contain `e` as an operand;
- `O` be registered-output classes plus the final effect-token class; and
- `artifact(v)` be the optional compilation artifact of node `v`.

Registered input representatives are terminals and have no owner.

### Variables

- `a_e ∈ {0,1}`: class `e` is active in the extracted rooted graph.
- `z_r ∈ {0,1}`: representative `r` is the chosen implementation of its class.
- `x_v ∈ {0,1}`: node `v` is included.
- `h_q ∈ {0,1}`: compilation artifact `q` is needed.
- `d_e ∈ [0, |E|-1]`: topological rank of class `e`.

The rank variables can be omitted only if saturation construction proves that every possible
alternative dependency is acyclic.

### Constraints

#### 1. Interface roots and pinned inputs

```text
a_e = 1                                      for every e in O
z_r = 1                                      for every registered-input terminal r
```

Pinning the terminal, rather than merely activating its class, prevents extraction from
silently replacing a caller-provided input with a computed equivalent.

#### 2. Exactly one representative for every active class

```text
sum(r in R_e) z_r = a_e                      for every e in E
```

Inactive classes select no representative. Active classes select exactly one.

#### 3. Node/result ownership

```text
z_r <= x_owner(r)                            for every non-terminal r
x_v <= sum(r in results(v)) z_r              for every v
```

The first family says that selecting a result includes its producer. The second prevents
dead nodes: an included multi-output node must implement at least one active result. Together
they encode `x_v = OR(z_r for r in results(v))` without charging a multi-output node more
than once.

Every side-effecting or otherwise zero-data-result node must have an effect-token result, so
`results(v)` is never empty for a node that may need to be retained.

#### 4. Selected nodes require all operands

```text
x_v <= a_e                                   for every v and e in operands(v)
```

This is the direction the earlier encoding was missing. An active class does not force every
node that happens to consume it to be selected.

#### 5. Every active value is demanded by a root or a selected user

Let `root_or_terminal(e)` be one for an output/effect root or a registered-input class and
zero otherwise.

```text
a_e <= root_or_terminal(e)
       + sum(v in users(e)) x_v              for every e in E
```

This rules out disconnected selected subgraphs even when some costs are zero. Combined with
acyclicity, following users from any active non-root class must eventually reach a root.

#### 6. Acyclicity

Let `M = |E|`. For each possible selected result representative of a node:

```text
d_e + 1 <= d_class(r) + M * (1 - z_r)
    for every v, e in operands(v), r in results(v)
```

When `r` is selected, every operand class ranks below its result class. When it is not
selected, the constraint is relaxed. This catches accidental recursive alternatives caused
by class unions or stale rewrites.

#### 7. Compilation-artifact activation

For every artifact `q`:

```text
x_v <= h_q                                   for every v with artifact(v) = q
h_q <= sum(v: artifact(v) = q) x_v
```

Thus `h_q` is the OR of all nodes using the artifact and compilation cost is paid once.

#### 8. Optional hard budgets

Resource policies can add linear limits such as:

```text
sum(v) x_v <= max_runtime_nodes
sum(q) code_bytes(q) * h_q <= code_cache_budget
```

Per-kernel registers, shared memory, body size, and launch feasibility should normally be
used to reject an alternative before extraction rather than aggregated across unrelated
kernels.

Graph-wide peak device memory is not captured by the base formulation. If fusion choice must
obey a hard pool limit, either:

1. extend extraction with liveness/order/offset variables similar to `planner.rs`; or
2. solve extraction, run the existing memory planner, add a no-good cut when the result
   exceeds the limit, and repeat.

The second approach keeps the first implementation substantially smaller.

### Objective

Precompute non-negative integer costs in nanoseconds:

```text
minimize
    expected_runs * sum(v) runtime_ns(v) * x_v
  + cache_miss_probability * sum(q) compile_ns(q) * h_q
  + deterministic_tie_break
```

CP-SAT accepts integer coefficients, not `f64`. Represent the miss probability as a scaled
rational or fold it into the compile coefficients. Check coefficient products for `i64`
overflow. A small final objective such as node count and then stable node ID gives
deterministic choices among equal-time solutions; use sequential optimization phases if a
single weighted sum could distort the primary objective.

The model does not need a separate scalar named `C`. The solver minimizes the linear
expression directly.

### Why this formulation is complete

For a finite saturated hypergraph satisfying the invariants:

- every root selects exactly one representative;
- every selected non-terminal representative includes exactly one owner node;
- every included node recursively activates all operands;
- each active operand has exactly one selected representative;
- demand constraints exclude dead islands;
- rank constraints exclude cycles;
- effect roots preserve required opaque operations; and
- artifact variables account for deduplicated compilation.

Conversely, any rooted acyclic extraction maps directly to a satisfying assignment by setting
the variables for exactly its classes, representatives, nodes, and artifacts.

This is completeness of extraction over the retained alternative graph. It does not imply
that saturation enumerated every legal fusion, nor that the cost oracle predicts hardware
time exactly.

### Problems in the earlier encoding

The original equality

```text
N_t * (num_operands(t) + num_results(t))
  = sum C_operand + sum b_result
```

is incorrect. If an input class is active and a node consuming it is not selected, the left
side is zero while the right side is positive. It therefore forces every consumer of every
active class into the solution. It also requires all result ports of a multi-output node to
be selected even when only one is demanded.

The earlier encoding additionally lacked:

- a pinned physical representative for graph inputs;
- owner/result linkage with correct direction;
- prevention of dead disconnected selections;
- cycle prevention;
- effectful/zero-result nodes;
- registered-output materialization semantics;
- pay-once compilation artifacts;
- an actual minimization statement and integer scaling; and
- graph-wide memory handling.

Therefore the earlier ILP was not complete.

## 7. Materializing the selected graph

After solving:

1. start at registered output classes and the final effect-token class;
2. follow selected representatives to their owner nodes and operands;
3. topologically sort selected nodes, using `d_e` only as a check rather than as the sole
   ordering source;
4. assign a concrete `BufId` to every selected representative;
5. wire each node operand to the selected representative of its operand class;
6. allocate private buffers for unused result ports of an included multi-output node, since
   the executable still produces them;
7. register the selected representative of each output class in the original interface
   order;
8. preserve the original registered input representatives; and
9. run graph verification, DCE, kernel deduplication, and memory planning.

The current runtime addresses graph outputs by interface position, so remapping an output
position to an equivalent selected `BufId` is the natural policy. If any caller relies on the
pre-extraction output `BufId` itself, V2 must instead pin that representative or insert a
final copy. This policy must be decided before implementation.

## 8. Implementation sequence

1. **SSA import.** Version all writes and carried values; add effect tokens; verify the
   invariants above.
2. **Alternative graph.** Implement typed classes, representatives, multi-output nodes,
   canonical indexing, node interning, and a round-trip materializer with no fusion rules.
3. **Port v1 analysis and surgery.** Preserve port/site/binder metadata. Insert one validated
   fused alternative without changing the original alternatives.
4. **Sequential saturation.** Add the worklist, attempted keys, support tracking, limits, and
   deterministic dominance. Establish correctness before parallelizing candidate creation.
5. **Extractor.** Implement the base CP-SAT model, deterministic integer objective, and
   materialization. Keep a greedy fallback for builds without the solver.
6. **Artifact-aware cost.** Key compilation variables by the real codegen cache key and add
   the run-horizon configuration.
7. **Cost calibration.** Add device parameters, KIR-derived operation/resource estimates,
   and confidence reporting.
8. **Parallel saturation.** Move candidate construction to immutable worker epochs; retain a
   single deterministic insertion/rebuild coordinator.

## 9. Open design gaps

The extraction equations above are sufficient once their inputs exist. The following details
still need decisions or implementation-level specification:

| Gap | Why it matters | Required decision |
| --- | --- | --- |
| Output interface identity | An output class may choose a fresh fused representative. | Remap by interface ordinal, pin the original `BufId`, or insert a final copy. |
| Opaque effects | Buffer SSA alone does not preserve hidden side effects. | Confirm which nodes need an effect token and whether all opaque nodes share one total-order chain. |
| Saturation scope | Exponential alternatives can dominate analysis time. | Choose support/body/node/time caps and report truncation. |
| Safe dominance | Access-equivalent cost dominance is stricter than runtime dominance. | Define the exact future-fusability signature used for pruning. |
| Multi-output producer | V1 surgery and costs assume one producer result. | Keep the restriction initially or specify per-output scalarization/liveness. |
| Trusted scatter inverses | An invalid trusted inverse is a miscompile. | Decide whether trusted `SExpr` inverses are allowed in production and how diagnostics expose them. |
| Pass boundary | Artifact keys and legality depend on normalization order. | Freeze the pre-saturation and post-fusion pass sequence from `refactor-plan.md`. |
| Cost horizon | Runtime and compilation cannot be traded without workload context. | Expose expected runs, cache-miss probability, and device calibration in options. |
| Compile-time estimate | HIR body size is a weak predictor and artifacts compile in parallel. | Start with a versioned proxy; later fit measured compile work/wall time. |
| Register/occupancy estimate | Fusion can spill even when op counts look profitable. | Add KIR lowering as a cost oracle or conservative pre-lowering guards. |
| Peak graph memory | The base ILP does not model schedule-dependent pool size. | Ignore, add iterative no-good cuts, or integrate the existing planner model. |
| Solver availability | CP-SAT is currently optional and can time out. | Define feature gating, time limit, accepted feasible status, and deterministic greedy fallback. |
| Equality proofs | Unsound class unions make every extraction unsound. | For fusion-only V2, insert into known consumer result classes and avoid general unions initially. |

## 10. Tests

In addition to the v1 legality and numerical tests:

- round-trip a normalized graph through the fuser with no alternatives;
- verify a mutation consumes the old version and downstream nodes use only the carried
  version;
- compare the solver against brute-force enumeration on small alternative graphs;
- test a chain where the fused implementation wins;
- test a diamond where one consumer fuses and the shared producer remains for the other;
- test a multi-output node used through one and through both result ports, charging it once;
- test input pinning and output remapping;
- test that a zero-cost disconnected island is not selected;
- inject a recursive alternative and verify the rank constraints reject it;
- test an opaque side-effect node retained solely by the effect root;
- test two selected nodes sharing one `ArtifactKey` and paying compile cost once;
- force each saturation cap and assert the report identifies it;
- solve, materialize, and re-run graph SSA/type/access verification; and
- run fused/unfused GPU differential tests on the same workloads used by v1.

For tiny ILP fixtures, assert the chosen representative and node sets exactly. For larger
graphs, report retained alternatives, saturation truncation, objective components, solver
status/gap, selected artifacts, and post-materialization launch count.
