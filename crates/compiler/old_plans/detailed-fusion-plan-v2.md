# Detailed Kernel Fusion V2 Implementation Plan

This document turns `fusion-plan-v2.md` into an implementation plan for the
current `crypto-compiler` graph IR and compiler pipeline. The design keeps the
proposal's central separation:

1. Build a graph containing the original executions and synthesized fusion
   alternatives.
2. Estimate the cost of every alternative without invoking `nvcc`.
3. Select a closed set of alternatives globally with CP-SAT.
4. Materialize only the selected graph and compile only its selected module
   artifacts.

The existing fusion implementation remains available until v2 passes
correctness and measured-performance gates. V2 is opt-in throughout this plan,
but its implementation does not import, call, or extract implementation
details from the existing fusion pass.

## 1. Scope and non-goals

The first production-capable version will support:

- conversion of the current buffer graph into versioned logical values;
- the original graph as an always-available extraction;
- a new structured-kernel producer-consumer fusion implementation with
  drop-intermediate and keep-intermediate variants;
- longer producer-consumer chains via saturation composition of drop and
  keep candidates;
- multi-consumer fanout fusion;
- small-kernel block fusion for connected subgraphs of `grid_dim = 1`
  kernels;
- exact module-artifact reuse in the extraction model;
- a deterministic KIR-based static cost estimate;
- CP-SAT extraction under `planner-ortools`;
- a solver-free fallback to the original graph;
- deterministic reconstruction of a valid `GraphBuilder`.

The following are later milestones, not prerequisites for the first end-to-end
implementation:

- producer-schedule epilogue fusion;
- same-domain horizontal fusion;
- different-domain or predicated-store horizontal fusion;
- fusion through mutable or opaque-effect nodes;
- joint fusion, scheduling, and peak-memory optimization;
- compile-wall-time scheduling across parallel `nvcc` workers;
- online profiling or `ptxas` in the candidate loop.

V2 will not upgrade Plonky3 or change CUDA workspace membership.

## 2. Required invariants

These invariants are assertions in debug builds and targets for unit tests.

### 2.1 Logical-value invariants

1. A `ValueClassId` denotes one semantic value, not one physical allocation.
2. The seed graph has at most one original producer for each non-source value.
3. Candidate insertion may add more producers for an existing value. Those
   producers are alternatives, not additional sequential writes.
4. Every candidate output is semantically equal to the existing value class
   named at that output position.
5. A graph input value has no producer.
6. A selected non-input value has exactly one selected producer.
7. Positional kernel operands/results are preserved separately from set-like
   solver dependencies. Repeated use of one value at two operand positions is
   legal.

The sentence "each value has exactly one producer" therefore applies only to
the original versioned graph. After candidate enumeration, a value can have
many candidate producers and exactly one selected producer.

### 2.2 Storage and mutation invariants

The current graph IR uses `BufId` as both a physical allocation identity and a
versioned program value. `GraphFuser::take_graph` must split those concepts.

1. Every physical `BufId` has a monotonically increasing logical version.
2. Every read is bound to the version visible immediately before that graph
   node in insertion order.
3. Every write creates a new `ValueClassId`, including in-place and carried
   writes.
4. All versions of a physical buffer map to the same first instance through
   `re_exported`.
5. Registered graph outputs refer to the final version of their registered
   physical buffer.
6. The selected node order preserves RAW, WAR, and WAW hazards for versions
   sharing a physical buffer.
7. MVP fusion candidates may not cross a mutable, partial-write, black-box, or
   otherwise effectful node. Those nodes appear in the alternative graph only
   in their original form and are the sole producer of their outputs.

The seventh rule avoids needing to prove pointer-alias equivalence inside a
new fused kernel. Versioning and extraction must still handle such nodes so
the original graph remains feasible.

### 2.3 Alternative invariants

1. An alternative is one atomic graph execution.
2. Selecting an alternative produces all of its declared graph-level outputs.
3. An intermediate computed only inside a fused kernel is not an output of the
   alternative and has no materialized value in that extraction.
4. Candidate origins are a sorted set of seed `NodeId` values.
5. Generic pair composition accepts only disjoint origins. The fanout fusion
   pass is responsible for safely sharing overlapping producer work.
6. Candidate normalization leaves exactly one canonical, monomorphized HIR
   kernel with a fixed block hint.
7. The original live graph is always representable using only original
   alternatives.
8. The complete bipartite value/alternative graph is acyclic after every
   insertion.

### 2.4 Determinism invariants

1. Parallel enumeration never allocates graph IDs.
2. Candidate drafts are sorted by a stable key before serial insertion.
3. Cost sampling is seeded by the normalized module hash and estimator version.
4. CP-SAT uses integer costs, a fixed seed, and one search worker unless an
   explicit nondeterministic mode is requested.
5. Ties are broken by secondary minimization of selected executions and
   selected materialized values; stable candidate IDs and the fixed solver
   seed determine any remaining tie.

## 3. Pipeline position

V2 runs after canonicalization and initial DCE:

```text
validate interface
  -> lower_reduce
  -> monomorphize
  -> canonicalize/split
  -> initial DCE
  -> build alternative graph
  -> enumerate and normalize candidates
  -> estimate candidates
  -> extract
  -> reconstruct GraphBuilder
  -> kernel_dedup
  -> DCE
  -> existing compile and memory-plan stages
```

The input to `GraphFuser` therefore has canonical, monomorphized,
single-kernel structured nodes. Candidate synthesis must restore that same
postcondition before inserting a candidate because later saturation rounds,
module hashing, and cost estimation all consume canonical candidates.

`fuse_graph_v2` must not invoke `nvcc`, load a CUDA module, mutate the kernel
cache, or run the graph memory planner.

`GraphCompiler::fuse` passes an immutable `FusionContextV2` containing the
graph-symbol environment (`GraphCompiler::env`), target architecture, and
compile-option fingerprint. Costing uses the graph environment for symbolic
buffer sizes and per-kernel `param_bindings` for residual HIR/KIR bounds.

## 4. File layout

Add the following modules next to `passes/fusion.rs`:

```text
src/passes/
  fusion_utils.rs               independent HIR traversal/rewrite utilities
  fusion_v2/
    mod.rs                      orchestration, options, and report construction
    model.rs                    IDs, alternatives, indexes, invariants
    version.rs                  GraphBuilder -> versioned seed alternative graph
    access.rs                   binds access sites to logical values
    normalize.rs                candidate validation, normalization, artifact key
    draft.rs                    draft/finalized candidate and candidate key
    saturate.rs                 rounds, deterministic insertion, and caps
    fusions/
      producer_consumer.rs      drop/keep and multi-seam fusion
      fanout.rs                 shared-producer, multi-consumer fusion
      small_kernel.rs           single-block subgraph -> one-launch fusion
      epilogue.rs               later milestone
      horizontal.rs             later milestone
    cost/
      mod.rs                    public estimator and device model
      liveness.rs               conservative KIR register-use estimate
      transactions.rs           deterministic warp access sampling
      interpreter.rs            per-thread critical-path estimate
      cache.rs                  KernelCostManager (§12.10)
    extract/
      mod.rs                    common solution types and original fallback
      cpsat.rs                  planner-ortools implementation
    apply.rs                    selected alternatives -> GraphBuilder nodes
    tests.rs                    model, fusion-pass, solver, and reconstruction tests
```

`fusion_v2::saturate` calls the enabled fusion modules explicitly. Each module
iterates `GraphFuser.nodes`, `producers`, and `consumers` directly and matches
the few relevant `GraphNode` variants. The calls occur in a fixed order
written directly in `saturate.rs`.

`passes/mod.rs` exports `fusion_v2` unconditionally. Only
`fusion_v2::extract::cpsat` is gated by `planner-ortools`; the model,
candidate generation, estimator, fallback, and their tests build without
OR-Tools. `fusion_utils` is `pub(crate)` and has no solver dependency.

The existing Cargo feature is `planner-ortools`; do not introduce a second
feature name for the same dependency.

## 5. Core data model

`GraphFuser` uses dense arenas for logical values and graph nodes. There is
one graph `NodeId` namespace: seed nodes and fused nodes occupy the same
`nodes` arena, and a candidate inserted in round `r` is eligible for matching
in round `r + 1`.

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ValueClassId(pub usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NodeId(pub usize);

pub type ValIdMap<T> = Vec<T>; // indexed by ValueClassId
pub type NodeIdMap<T> = Vec<T>; // indexed by NodeId

pub struct AltGraphNode {
    pub inputs: Vec<ValueClassId>,
    pub outputs: Vec<ValueClassId>,
    pub node: GraphNode,
}

pub struct GraphFuser {
    pub bufs: ValIdMap<BufInfo>,
    pub nodes: NodeIdMap<AltGraphNode>,
    pub producers: ValIdMap<Vec<UseInfo>>,
    pub consumers: ValIdMap<Vec<UseInfo>>,
    /// `None` for nodes without a structured-kernel access relation.
    pub access_relations: NodeIdMap<Option<AccessRelation>>,
    /// Maps every logical version to the first ValueClassId for its physical
    /// allocation.
    pub re_exported: ValIdMap<ValueClassId>,
    pub inputs: Vec<ValueClassId>,
    pub outputs: Vec<ValueClassId>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct UseInfo {
    pub node: NodeId,
    pub pos: usize,
}
```

All `ValIdMap` fields grow in lockstep when a new `ValueClassId` is allocated.
All `NodeIdMap` fields grow in lockstep when a new `NodeId` is allocated. This
makes an absent entry an invariant violation and permits unconditional indexed
access after construction.

### 5.1 Logical port convention

`AltGraphNode.inputs` and `outputs` are positional logical ports.
`UseInfo.pos` indexes those vectors. Use the following fixed convention:

- inputs corresponding to ordinary `GraphNode` operands come first, in
  `GraphNode::get_operands` order;
- a partial write appends the previous destination value as a preservation
  input;
- explicit results come first in `outputs`, in
  `GraphNode::get_results` order;
- re-exported mutated inputs follow the explicit results, in input-position
  order.

This convention handles the two cases where logical ports are not identical
to launch-time pointer arrays without changing `AltGraphNode`:

- a black-box carried output uses an input pointer at runtime but is a new
  logical output;
- a partial memcpy/memset preserves bytes from the old destination even though
  that old value is not passed as a separate runtime argument.

Implement the positional API as graph-IR helpers, not by open-coding variant
matches in the fuser:

```rust
impl GraphNode {
    fn get_operands(&self, bufs: &[BufInfo]) -> Vec<BufId>;
    fn get_results(&self) -> Vec<BufId>;
    fn rewrite_bindings(
        &mut self,
        input_bufs: &[BufId],
        output_bufs: &[BufId],
    ) -> Result<(), CompileError>;
}
```

The `bufs` argument is required because deciding whether memcpy/memset covers
the whole destination requires its declared size.
`rewrite_bindings` centralizes the inverse positional mapping needed during
materialization. For black-box nodes it rewrites explicit inputs/outputs and
derives carried bindings from the re-export suffix. For structured candidates
it rewrites module input/output bindings.

### 5.2 Physical storage mapping

Physical storage identity is encoded as follows:

1. Seed `GraphFuser.bufs` with a clone of every original `BufInfo`, in
   `BufId` order. Therefore initial `ValueClassId(b.0)` is the first
   instance of physical `BufId(b.0)`.
2. On each write, append a clone of that physical buffer's `BufInfo`; the
   appended index is the new logical value.
3. Set `re_exported[new_value.0] = first_value`. Set identity entries for the
   initial classes as well.
4. Recover physical storage as
   `BufId(re_exported[value.0].0)`.

This makes `bufs[value.0]` the logical value's shape/device metadata and
`re_exported[value.0]` its physical-allocation identity. Logical version order
for one allocation is the increasing `ValueClassId` order among classes with
the same first instance.

### 5.3 Owning `GraphNode` directly

Change the black-box closure type before implementing the arena:

```rust
pub type KernelFn =
    Arc<dyn Fn(&[*mut ()], &[*mut ()], cudaStream_t) + Send + Sync>;

#[derive(Clone)]
pub struct KernelNode {
    // existing fields unchanged
}
```

`GraphBuilder::insert_blackbox_kernel` must require
`Fn(...) + Send + Sync + 'static` and wrap it in `Arc::new`. This makes
`KernelNode` cloneable and makes an immutably borrowed `GraphFuser`
shareable by parallel candidate workers. Workers borrow existing
`AltGraphNode`s and return newly owned `GraphNode` candidates. Constant nodes
own non-cloneable device buffers, so seed conversion moves all original
`GraphNode` values into the fuser.

### 5.4 Search and extraction sidecars

Keep data that is not part of the alternative graph outside `GraphFuser`:

```rust
struct SaturationState {
    /// origins[n.0] is the set of seed NodeIds represented by node n.
    origins: NodeIdMap<BTreeSet<NodeId>>,
    seen_candidates: HashSet<CandidateKey>,
}

struct ExtractionData {
    costs: NodeIdMap<GraphNodeCost>,
    artifact_keys: NodeIdMap<Option<ArtifactKey>>,
}
```

The sidecars have the same dense `NodeId` indexing discipline:

- `origins` prevents generic composition from computing the same seed node
  twice. It uses the same `NodeId` type; `origins[n.0] = {n}` for a seed and
  the union of parent-node origins for a fused node.
- `seen_candidates` is search bookkeeping, not graph semantics.
- costs and artifact keys are derived after saturation and belong to the ILP
  input. Keeping them out of `AltGraphNode` also permits estimator revisions
  without mutating the alternative graph.

Black-box kernel nodes never receive fusion candidates, so they remain the
sole producer of their outputs in the alternative graph. The producer
equation (§13.3) forces them to be selected whenever any of their outputs is
demanded, so no separate `mandatory` bit or ILP constraint is required.

### 5.5 Artifact identity

```rust
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ArtifactKey {
    pub module_hash: [u8; 32],
    pub target_arch: String,
    pub compiler_flags_hash: [u8; 32],
}

pub struct ArtifactContext {
    pub target_arch: String,
    pub compiler_flags_hash: [u8; 32],
}

pub struct FusionContextV2<'a> {
    pub graph_symbols: &'a BTreeMap<ir::VarId, i64>,
    pub artifact: &'a ArtifactContext,
}
```

For the first implementation, the pass can key only on the normalized
residual `module_hash` because architecture and flags are constant within one
`GraphCompiler`. Keep the wrapper type so those fields can be added without
changing the extraction model.

## 6. Building the versioned seed graph

Implement `GraphFuser::take_graph` in `version.rs`. It takes
`&mut GraphBuilder`, clones buffer metadata, and moves
`std::mem::take(&mut g.nodes)` into `GraphFuser.nodes`. The caller owns a
guard that restores the original extraction on every error path.
`ConstBuf::DeviceBuf` owns a non-cloneable allocation, so each
`AltGraphNode.node` owns the moved `GraphNode` directly.

### 6.1 Scan state

Maintain only construction-local state:

```rust
current: Vec<Option<ValueClassId>>
seed_node_count: usize
```

Initialize `GraphFuser.bufs`, `producers`, `consumers`, and identity
`re_exported` entries for every physical buffer. Initialize `nodes` and
`access_relations` empty. Set `current[b]` only for registered graph inputs;
do not infer inputs from buffers with no writers.

### 6.2 Per-node algorithm

For each moved original `GraphNode` in insertion order:

1. Call the shared positional helpers to obtain logical physical operands and
   results.
2. Resolve every operand against `current[buf]` before creating results. An
   absent current value is a read-before-write error.
3. For each distinct written physical buffer, append one `BufInfo`, an empty
   producer list, an empty consumer list, and the first-instance entry in
   `re_exported`. Their common index is the new `ValueClassId`.
4. Fill outputs in the fixed order: explicit results, then re-exported mutated
   inputs.
5. Update `current[buf]` only after all inputs have been resolved.
6. Extract `Some(AccessRelation)` for a structured kernel or `None` for every
   other graph-node variant.
7. Append `AltGraphNode { inputs, outputs, node }` and the access-relation
   option in lockstep. Their common index is the `NodeId`.
8. Add one `UseInfo { node, pos }` for every input and output position.

Variant-specific rules are:

- `Kernel`: ordinary inputs precede explicit outputs. If one physical buffer
  occurs in both lists, the input binds the old logical version and the output
  binds the new version.
- `BlackboxKernel`: ordinary inputs are logical inputs; explicit outputs are
  followed by one new output for each `carried_outputs` entry. The node is
  never a fusion candidate initially, but its versions and dependencies remain
  exact.
- `Const`: no inputs, one output.
- `Memcpy`: source is the first input and destination is the output. If the
  written range is not proven to cover the destination, old destination is a
  second preservation input.
- `Memset`: one output. If the range is not proven full, old destination is a
  preservation input.

Reject duplicate writes to one physical buffer at several result positions
unless a later explicit rule defines their ordering.

### 6.3 Interface mapping

After the scan:

- `GraphFuser.inputs` maps registered input buffers to their initial value
  classes in interface order;
- `GraphFuser.outputs` maps registered output buffers to `current[buf]` in
  interface order;
- every registered output has a current value;
- every non-input value used by a seed node has exactly one seed producer.

### 6.4 Original fallback extraction

Run existing DCE before taking the graph. Record `seed_node_count`. Selecting
`NodeId(0)..NodeId(seed_node_count)`, their outputs, and registered inputs is
the baseline solution. Store it as `ExtractionSolution::original(&gf,
seed_node_count)`.

This solution is used for:

- solver fallback;
- a correctness oracle for reconstruction;
- objective comparison;
- an optional solver hint if the `cp_sat` crate exposes stable hint support.

## 7. Storage hazard ordering

Logical producer/consumer edges encode RAW dependencies. Because several
logical versions can use the same physical `BufId`, reconstruction must also
preserve WAR and WAW dependencies.

After extraction, build a selected-node precedence graph:

1. For every selected alternative input, add the selected producer-to-consumer
   RAW edge.
2. For each physical buffer, list selected produced versions in increasing
   version order and add WAW edges between consecutive selected writers.
3. For every selected consumer of version `k`, add a WAR edge from that
   consumer to the first selected writer of a version greater than `k`.
4. Ignore self-edges when one selected alternative internally contracts both
   endpoints.

Topologically sort this precedence graph and emit selected nodes in that
order. The resulting insertion order allows the existing memory planner to
re-derive compatible RAW/WAR/WAW edges from physical `BufId`s.

For the MVP, any synthesized candidate touching a physical buffer with more
than one writer, a carried output, a partial write, or an opaque access is
rejected. Consequently, the complicated hazard cases are exercised by
original alternatives only, but the reconstruction algorithm is correct for
the whole selected graph.

The selected precedence graph is acyclic by construction: value dependencies
are a subgraph of the acyclic alternative graph, while MVP storage-hazard
edges follow seed version order and synthesized candidates do not cross the
mutable cases listed above. The topological sort computes an emission order;
failure is an invariant error that restores the original graph. With
`validate_alt_graph_acyclicity` enabled, such a failure should be unreachable;
with it disabled, the required sort remains a last-resort detector. Neither
case adds a solver constraint or triggers a re-solve.

## 8. Access relations

Access relations use the same dense-index convention as the alternative
graph:

```rust
pub struct AccessRelation {
    pub reads: Vec<ReadRelation>,
    pub writes: Vec<WriteRelation>,
    pub index_bounds: HashMap<ir::VarId, i64>,
    pub grid_index: ir::VarId,
    pub inner_indices: Vec<ir::VarId>,
}

pub struct ReadRelation {
    pub read: Quast,
    pub val: ValueClassId,
    pub node: ir::NodeId,
}

pub struct WriteRelation {
    pub write: Quast,
    pub inv: Quast,
    pub val: ValueClassId,
    pub node: ir::NodeId,
}
```

`index_bounds` is intentionally sparse: only the small, non-contiguous subset
of `ir::VarId`s used as index variables appears in it. Dense aliases apply to
`ValueClassId` and alternative-graph `NodeId`, whose complete ID spaces are
owned by `GraphFuser`.

The alternative graph uses `fusion_v2::NodeId`; the access-site fields use the
existing HIR arena index `ir::NodeId`. Always qualify the latter in
`fusion_v2` modules.

The initial access extractor supports flat computes and nested computes whose
loop bounds are all in `index_bounds`. `AccessCollector` derives the active
index scope while occurrence-walking from the module root. Because the stored
relation identifies a site by `ir::NodeId`, it accepts an access node only when
all occurrences of that node have the same ordered `IndexBinding` sequence.
It deduplicates equal occurrences after comparing their scopes. A shared
access node found under different scopes returns
`AccessError::AmbiguousAccessScope { node }`; the containing kernel remains
executable but is not eligible for structured fusion.

Passes recover a site's scope with `unique_index_scope(module, node)`, which
performs the same occurrence traversal and equality check. This is
conservative but makes the existing `AccessRelation` layout unambiguous. If a
later pass must fuse a shared access under distinct scopes, add an explicit
occurrence/scope identity to the relation rather than choosing one occurrence
arbitrarily.

Extract relations only for structured kernels. For a newly synthesized fused
node, extract the relation after normalization and bind each read/write
position to the candidate's `inputs` or `outputs`. Opaque, memcpy, memset,
and constant nodes have no relation and are not structured-fusion parents in
the initial implementation.

### 8.1 Inverses

Initially support identity, affine permutation, and `DivMod` inverses for
dense outputs. A user-supplied scatter inverse is not trusted merely because
it is a `Quast`. Candidate legality must verify over the concrete bounded
domain that:

- the forward map is in bounds;
- `inverse(forward(i)) == i` for every producer iteration;
- every consumer index fused through the seam lies in the inverse's domain;
- the write is injective where the fusion assumes one producer evaluation.

Exhaustive checking is acceptable below a configured domain size. Larger
domains require a supported symbolic inverse proof; otherwise the candidate
is rejected.

### 8.2 Shared fusion utilities

Implement `passes/fusion_utils.rs` as independent HIR infrastructure. Fusion
passes depend on this module and ordinary IR APIs; the utilities do not call
either fusion implementation.

The central traversal API is an occurrence-based visitor:

```rust
pub enum VisitControl {
    Recurse,
    SkipChildren,
}

pub struct IndexBinding {
    pub var: ir::VarId,
    pub bound: ir::SizeExpr,
    pub kind: IndexKind,
}

pub enum IndexKind {
    Compute,
    Reduce,
}

pub trait HirVisitor {
    type Error;

    fn enter(
        &mut self,
        module: &ir::Module,
        id: ir::NodeId,
        node: &ir::Node,
        index_scope: &[IndexBinding],
    ) -> Result<VisitControl, Self::Error>;

    fn leave(
        &mut self,
        _module: &ir::Module,
        _id: ir::NodeId,
        _node: &ir::Node,
        _index_scope: &[IndexBinding],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub fn visit_hir<V: HirVisitor>(
    module: &ir::Module,
    root: ir::NodeId,
    visitor: &mut V,
) -> Result<(), V::Error>;
```

`visit_hir` has these guarantees:

- children are visited in the deterministic order returned by `children_of`;
- `enter`/`leave` are balanced, including for `SkipChildren`;
- `index_scope` contains the enclosing compute/reduce indices in outer-to-
  inner order;
- `enter` on a compute/reduce node sees the outer scope; its body child sees
  that node's index appended to the scope; `leave` sees the outer scope again;
- a shared HIR node is visited once per occurrence, because the same node can
  appear under different index scopes;
- an active-recursion stack detects malformed HIR cycles;
- visitors that need node-unique behavior maintain their own dense seen set.

Build the following reusable utilities on the visitor:

- `AccessCollector`, producing `AccessRelation` and every read site for each
  module input;
- `StructureCollector`, recording compute/reduce/let nesting, scatter maps,
  parallel mappings, and block hints used by pass-specific legality checks;
- `InputUseCollector`, recording all occurrences of a selected module input;
- `ReachableNodeCounter`, enforcing candidate HIR-size limits;
- `unique_index_scope`, rejecting an access node reached under unequal index
  scopes;
- deterministic expression cloning with alpha-renaming and an explicit
  `ir::NodeId -> ir::NodeId` substitution map;
- `BoundaryBuilder` for capture-free parameter merging, logical-value input
  interning, and parent-to-candidate input remapping;
- `KernelOutputView` for identifying each top-level output expression, its
  iteration variable, domain, and write relation;
- Quast composition, bounded-domain evaluation, and inverse verification;
- tuple-output construction and final module renumbering.

The producer-consumer, fanout, small-kernel, epilogue, and horizontal fusion
modules own their legality rules and rewrite algorithms. Shared utilities
provide mechanisms only; they do not encode fusion cases or choose
candidates.

## 9. Candidate representation and insertion

Fusion passes return drafts without allocating IDs:

```rust
pub struct CandidateDraft {
    pub parents: Vec<NodeId>,
    pub alt: AltGraphNode,
}
```

`parents` may name seed nodes, fused nodes, or both. A worker reads every
parent through the same `GraphFuser.nodes[parent.0]` path. It matches
`GraphNode::Kernel`, borrows the modules, and puts a newly owned
`GraphNode::Kernel` in `alt.node`.

Candidate finalization performs, in order:

1. Union `SaturationState.origins[parent.0]` and reject overlap between
   parents. Every pass (including fanout, which references a single producer
   from multiple consumer arms) constructs its candidate from parents with
   pairwise-disjoint origin sets; this check enforces that invariant
   uniformly.
2. Type-check the synthesized HIR.
3. Canonicalize and require exactly one kernel.
4. Monomorphize with the candidate's merged parameter bindings.
5. Canonicalize again if monomorphization changed the body.
6. Validate the launch schedule chosen by the producing fusion pass. The
   "launch schedule" is the tuple
   `(grid_dim, block_hint, outer domain, inner iteration bounds,
   thread-index binding)` — everything the lowering pipeline needs to map
   `compute[...]` iterations onto a legal thread count. A mismatch (e.g.,
   the pass claims a block hint the synthesized body does not satisfy)
   rejects the candidate with `CandidateFailure::LaunchScheduleMismatch`.
7. Extract and bind the new `AccessRelation`.
8. Compute the alpha-normalized residual module hash.
9. Construct `ArtifactKey`.
10. Compute a deterministic candidate key.
11. Return the normalized `AltGraphNode`, its `AccessRelation`, origin set,
    artifact key, and candidate key as an ephemeral finalized record.

```rust
pub struct CandidateKey {
    pub inputs: Vec<ValueClassId>,
    pub outputs: Vec<ValueClassId>,
    pub artifact: ArtifactKey,
}
```

The key intentionally excludes origins: two derivations producing the same
normalized executable with the same boundary are the same alternative.

Sort finalized candidates by `CandidateKey` and process them serially. For
each candidate accepted by the validation below, assign
`NodeId(gf.nodes.len())`, push the corresponding origins and access-relation
slots in lockstep with the node, and append each input and output use to
`consumers[value.0]` and `producers[value.0]`. There is no atomic `BufId` or
node-ID allocation.

### 9.1 Optional insertion-time acyclicity validation

View `GraphFuser` as a directed bipartite graph with edges:

```text
ValueClassId -> NodeId       when the value is an input of the node
NodeId -> ValueClassId       when the value is an output of the node
```

Assume the current graph is acyclic. Inserting candidate `a` with input set
`I` and output set `O` creates a cycle if and only if the current graph has a
directed path from some `o in O` to some `i in I`. Any new cycle must contain
`a`; removing `a` from that cycle leaves exactly such a path. Conversely, such
a path is closed by `i -> a -> o`.

Check this before mutating the arenas:

```rust
fn would_create_cycle(gf: &GraphFuser, alt: &AltGraphNode) -> bool {
    let mut target: ValIdMap<bool> = vec![false; gf.bufs.len()];
    for input in &alt.inputs {
        target[input.0] = true;
    }

    let mut seen_values: ValIdMap<bool> = vec![false; gf.bufs.len()];
    let mut seen_nodes: NodeIdMap<bool> = vec![false; gf.nodes.len()];
    let mut stack = alt.outputs.clone();

    while let Some(value) = stack.pop() {
        if target[value.0] {
            return true;
        }
        if std::mem::replace(&mut seen_values[value.0], true) {
            continue;
        }
        for use_info in &gf.consumers[value.0] {
            let node = use_info.node;
            if std::mem::replace(&mut seen_nodes[node.0], true) {
                continue;
            }
            stack.extend(gf.nodes[node.0].outputs.iter().copied());
        }
    }
    false
}
```

The implementation should use reusable scratch vectors with generation
counters so validation does not clear `O(|V| + |A|)` booleans for every
candidate. The traversal cost is proportional to the portion of the graph
reachable from `O`.

Gate the check with `FusionOptionsV2::validate_alt_graph_acyclicity`. It is on
by default during v2 rollout and in all tests. A rejected candidate records
`CandidateFailure::WouldCreateCycle`; it is not inserted and does not reach
the ILP. Disabling the check is a performance option that relies on each
fusion pass's legality proof to preserve the same invariant.

## 10. Fusion passes

The saturation driver calls the following candidate-producing passes
directly. Each pass performs its own node-variant checks, graph enumeration,
legality analysis, and HIR rewrite.

There is no common pass trait or runtime dispatch table. `saturate.rs` makes
ordinary module calls so that enabling a pass and its ordering are visible at
the call site:

```rust
drafts.extend(producer_consumer::enumerate(view, options));
if options.enable_fanout {
    drafts.extend(fanout::enumerate(view, options));
}
if options.enable_small_kernel {
    drafts.extend(small_kernel::enumerate(view, options));
}
if options.enable_epilogue {
    drafts.extend(epilogue::enumerate(view, options));
}
if options.enable_horizontal {
    drafts.extend(horizontal::enumerate(view, options));
}
```

`enable_fanout`, `enable_small_kernel`, and `enable_keep_variants` default
to `true` because those passes are part of the initial version; the flags
exist to isolate them during debugging. `enable_epilogue` and
`enable_horizontal` default to `false` until their milestones land.

These functions return `CandidateDraft`s and pass-local rejection diagnostics.
They do not allocate `NodeId`s or mutate `GraphFuser`.

### 10.0 Instance-aware enumeration and compile-time cost awareness

Every fusion pass is compile-time-cost aware. A single fused module
compiled by `nvcc` is reused by every selected alternative that shares
its `ArtifactKey` (§13.3 `z_m`). The compile-time cost of a fusion pattern
is therefore amortized across its instances: a pattern with `n` instances
pays one compilation and delivers up to `n` runtime savings. Enumeration
budgets are finite, so patterns must be enumerated in an order that
spends the budget on the highest-payoff patterns first.

Passes fall into two groups depending on whether a cheap pre-synthesis
structural signature exists.

**Passes with a pattern key** (producer-consumer, fanout).

Each pass's `enumerate` proceeds in three stages:

1. **Match**: walk the alternative graph and collect every connected
   subgraph that satisfies the pass's legality requirements. Do not
   synthesize HIR — a match records only the participating `NodeId`s
   and a light-weight *pattern key*.
2. **Bucket**: group matches by pattern key. Two matches in the same
   bucket are guaranteed to normalize to the same `ArtifactKey`, so
   they share a compiled module. Sort buckets by `|bucket|` descending;
   break ties by pattern-key byte order for determinism.
3. **Synthesize**: emit `CandidateDraft`s bucket by bucket, within each
   bucket in seed-`NodeId` order. Each pass tracks its per-round emission
   count against `max_alternatives_per_pass_per_round`
   (§11 defaults: `max_total_alternatives / (max_rounds * num_enabled_passes)`,
   rounded up). When the running count would exceed that quota mid-bucket,
   finish the current bucket and stop; partial-bucket emission leaves some
   instances unfused even though the module is compiled anyway, so
   completing the current bucket is strictly better under any nonzero
   quota. The saturation-driver's global cap
   (`max_total_alternatives`, §11) is a hard stop that can fire earlier;
   it is enforced by the driver after each pass returns.

Pattern keys are computed from monomorphized seed-kernel `module_hash`
values plus the pass's structural shape. The invariant "equal pattern
key → equal fused `ArtifactKey`" is a per-pass proof obligation.
Per-pass keys:

- producer-consumer (§10.1, §10.2):
  `(producer.module_hash, consumer.module_hash, sorted seam positions)`;
- fanout (§10.5):
  `(producer.module_hash,
    sorted vector of (consumer.module_hash,
                      sorted producer-output positions read by that consumer))`.
  Including per-consumer producer-output positions is required: two
  fanouts with the same producer and the same consumer modules but
  different producer outputs at the seam inline different producer
  expressions into the fused body and therefore compile to different
  modules; omitting the positions would mis-bucket them.

**Passes without a pattern key** (small-kernel block fusion).

Small-kernel fusion (§10.7) matches connected subgraphs of arbitrary
topology. Deciding whether two matched subgraphs synthesize to the same
fused module reduces to graph isomorphism on the subgraph, and there is
no cheap structural signature that satisfies the "equal key → equal
`ArtifactKey`" invariant in general. Small-kernel therefore skips the
bucketing stage and enumerates matches directly in a deterministic
seed-`NodeId` order, relying on `CandidateKey` post-normalization
deduplication (§9) and the ILP's `z_m` semantics (§13.3) for module
reuse across any small-kernel candidates that do happen to normalize to
the same `ArtifactKey`. The trade-off is that small-kernel cannot
pre-order enumeration by predicted bucket size; if the enumeration budget
binds, it is spent in match order rather than in compile-time-payoff
order. Small-kernel candidates are larger and less numerous per graph
than producer-consumer or fanout, so this is acceptable for the initial
version.

The saturation driver records per-pass pattern-key counts (where
applicable), instance counts per pattern key, and any truncated buckets
in `FusionReport`.

### 10.1 Producer into consumer, drop seams

`fusions/producer_consumer.rs` enumerates candidates in the three stages
of §10.0. Match stage:

1. Iterate consumer `NodeId`s in increasing order and retain
   `GraphNode::Kernel` nodes with an `AccessRelation`.
2. For every distinct consumer input value, iterate
   `gf.producers[value.0]` in increasing `NodeId` order.
3. Group by `(producer, consumer)`. The group contains every value produced
   by that producer and consumed by that consumer; a match records
   `(producer NodeId, consumer NodeId, sorted seam positions)` and one
   candidate is later synthesized for the whole group rather than
   independently fusing individual seams.
4. Skip duplicate groups caused by repeated positional uses of the same
   value.

Bucket matches by
`(producer.module_hash, consumer.module_hash, sorted seam positions)`,
sort buckets by instance count descending, then synthesize.

For producer `p`, consumer `c`, and seam set `S`:

```text
inputs  = stable_unique(p.inputs followed by c.inputs minus S)
outputs = c.outputs
```

Both nodes must be pure `GraphNode::Kernel` nodes with disjoint origins and
supported access relations. For each seam output `s`, define:

```text
w_s(q)      producer write index for producer iteration q
inv_s(x)    verified inverse of w_s
r_l(j)      read index at consumer load site l under consumer indices j
sigma_l(j)  inv_s(r_l(j))
```

The candidate is legal only if, for every consumer load site of every seam:

- `r_l(j)` is in the written output domain for every bounded consumer index
  tuple `j`;
- `w_s(sigma_l(j)) == r_l(j)` over that domain;
- `sigma_l(j)` is in the producer iteration domain;
- the write is injective on the iterations reached by `sigma_l`;
- every reachable use of the seam's module input is an analyzed load site;
- neither graph node has physical mutation or partial-write semantics, and
  neither HIR module contains unsupported scatter forms;
- merged parameters, shapes, output aliases, and the configured HIR-size cap
  validate.

Use exhaustive bounded-domain verification below the configured threshold and
symbolic Quast proofs above it. Reject the candidate when neither proof is
available.

Synthesis constructs a new module from scratch:

1. Create a `BoundaryBuilder` that interns each non-seam `ValueClassId` once
   and records the new module-input position used by every parent input.
2. Keep the consumer's compute/reduce structure, grid mapping, and block hint
   as the schedule anchor.
3. For each seam load site `l`, clone the producer output expression into the
   consumer module. Substitute the producer output index variable with
   `sigma_l(j)`, remap producer inputs through `BoundaryBuilder`, and
   alpha-rename every producer-local variable.
4. Intern equal `(producer output, sigma_l, index scope)` substitutions so
   repeated identical reads share one cloned expression. Distinct read maps
   remain distinct computations and are charged by the estimator.
5. Clone the rest of the consumer with the seam-load substitutions installed.
6. Verify that no reachable reference to a removed seam input remains.
7. Return only the consumer outputs, canonicalize, type-check, monomorphize,
   and rebuild the candidate's `AccessRelation`.

The seam may remain a registered graph output or feed other alternatives. If
the extracted graph still requires it, the producer equation selects a
separate materializing alternative; the fused candidate itself drops it.

### 10.2 Producer into consumer, keep seams

For every legal drop-seams candidate, optionally synthesize:

```text
inputs  = same as the drop-seams candidate
outputs = kept seam values union consumer outputs
```

The fused module returns the producer result used at the seam as an additional
top-level output while also using it internally. A keep candidate requires the
producer seam output to share the consumer kernel's output domain and schedule;
otherwise one HIR kernel cannot materialize both shapes without introducing a
second launch. Generate a legal keep variant when at least one of these holds:

- the seam is a registered graph output;
- the seed graph has another original consumer of the seam;
- another enumerated alternative consumes the seam;
- a diagnostic option requests all keep variants.

This alternative represents "compute once, materialize for other users, and
avoid rereading in this consumer."

### 10.3 Longer producer-consumer chains

Repeated producer-consumer application forms chains. A composed pair is
eligible only when origins are disjoint and the total region remains within
caps.

Canonical origins and candidate-key deduplication ensure that `(A+B)+C`
and `A+(B+C)` do not create duplicate alternatives when they normalize to the
same boundary and module.

### 10.4 Epilogue fusion

Epilogue fusion retains the producer's launch schedule and substitutes a
pointwise consumer expression into the producer's result path before the
store.

Requirements:

- the consumer is flat, pointwise, and injective over the seam;
- every consumer output element uses exactly the corresponding producer
  output element, modulo a verified affine permutation;
- the consumer has no synchronization, reduction, scatter, or side effect;
- any consumer side inputs have compatible access domains;
- the producer may be general, including a sequential reduction, as long as
  it remains a single HIR/KIR kernel.

Generate drop and keep variants. This pass retains the producer schedule,
whereas producer-into-consumer fusion retains the consumer schedule.

### 10.5 Fanout fusion

Fanout fusion targets a producer whose seam is read by multiple consumers,
each of which would otherwise issue a separate global load after the
producer's store. The synthesized kernel computes the producer expression
once per thread and threads that value into every consumer body as an SSA
value, so the store-and-reload chain for each consumer disappears.

This differs from applying producer-into-consumer fusion (§10.1) to each
`(producer, consumer)` pair independently, which inlines the producer
expression separately in every resulting kernel and requires the producer
work to be repeated. It also differs from same-domain horizontal fusion
(§10.6), which merges kernels that share no dataflow.

The synthesized kernel has one grid launch, one shared schedule, and one
body that computes the producer once per index and then evaluates each
consumer body over the same index, reading the seam as an SSA value.
Outputs are the concatenation of all consumer outputs (drop variant), or
that concatenation prefixed by the seam values (keep variant).

Requirements for a legal fanout candidate:

- the producer is a pure `GraphNode::Kernel` and single-writer on every
  seam;
- every consumer is a pure `GraphNode::Kernel` and reads at least one
  seam through an identity access or a verified affine permutation over
  the shared outer domain;
- all consumers share the same block hint, thread geometry, and outer
  domain as the producer;
- no consumer reads another consumer's output, directly or transitively;
- origins of the producer and all consumers are pairwise disjoint;
- no storage hazard touches any seam or consumer output;
- merged HIR/KIR size and register-pressure caps validate.

Candidate composition:

```text
inputs  = stable_unique(producer.inputs
                       followed by each c_i.inputs minus all seam values)
outputs = drop:  concat(c_1.outputs, .., c_k.outputs)
          keep:  concat(seam values, c_1.outputs, .., c_k.outputs)
```

Positional operands can repeat across consumers; `stable_unique` collapses
them so the merged kernel has one operand slot per distinct logical input,
and the intra-kernel binding rewrite routes duplicate reads to the same
SSA value. Repeated positional operands do not duplicate solver
constraints (see §5.1).

Generate the keep variant when at least one of these holds:

- the seam is a registered graph output;
- the seed graph has another original consumer of the seam that is not in
  this fanout;
- another enumerated alternative consumes the seam;
- a diagnostic option requests all keep variants.

Without one of those conditions, drop is strictly better because it
retires the seam's global store.

Any `k >= 2` is admissible provided the legality requirements above hold
for every consumer and the shared block/geometry/domain is a single
equivalence class. Register pressure grows roughly linearly in `k` because
each consumer's live values accumulate through the shared body; the
estimator (§12.3, §12.4) rejects candidates whose occupancy collapses to
zero, and `max_alternatives_per_boundary` (§11) bounds the search so large
`k` does not inflate saturation without payoff.

Do not construct this pattern by taking two producer-consumer candidates
that share a producer and merging them with horizontal fusion. That
construction duplicates the producer expression in the merged HIR, and
correct sharing then depends on HIR common-subexpression elimination
happening to fold the copies. The sharing must be a construction invariant
of fanout, not a downstream optimization. Reject any candidate whose
normalized HIR contains the producer expression more than once at the
fanout root.

A fanout candidate is itself a `GraphNode::Kernel` and remains eligible
for further fusion in later saturation rounds: its outputs can serve as
seams for another fanout or an epilogue, and its origin set is the union
of the producer's and all consumers' origin sets, so subsequent
composition against any of those seed nodes fails the disjoint-origins
check.

Candidate enumeration in `fusions/fanout.rs` follows the three stages of
§10.0. Match stage: iterate producer `NodeId`s in increasing order, and
for each producer collect all pure consumers that share the producer's
block hint, thread geometry, and outer domain and read at least one
producer output through an identity or verified affine permutation.
Record each match as
`(producer NodeId, sorted vector of (consumer NodeId, producer-output positions read))`.
Its pattern key is
`(producer.module_hash,
  sorted vector of (consumer.module_hash,
                    sorted producer-output positions read by that consumer))`.
Bucket by pattern key, sort buckets by instance count descending, then
synthesize; each bucket emits its drop variant and, per the trigger list
above, its keep variant.

Later milestones may extend fanout with:

- mismatched-domain fanout where consumers cover only a subdomain of the
  producer (requires the predicated-store representation deferred with
  different-domain horizontal fusion);
- shared-producer fanout across a producer pair when the joint pattern
  beats either producer's fanout alone.

### 10.6 Same-domain horizontal fusion

Initial horizontal fusion is deliberately narrow:

- equal concrete outer domain;
- equal block hint and thread geometry;
- flat structured kernels;
- no dataflow path in either direction;
- no shared-memory allocations, syncs, atomics, mutation, scatter, or partial
  writes;
- disjoint origins and no selected storage hazard between regions.

The synthesized kernel executes both bodies at the same logical index and
returns the concatenated tuple of outputs. Identical normalized loads can be
shared only when HIR/KIR common-subexpression construction actually produces
one dominating load; cost extraction then observes that sharing.

Do not implement `compute[max(Na, Nb)]` with zero-valued masked outputs. Dense
output lowering would write the smaller buffer out of bounds. Different-domain
horizontal fusion waits for an explicit predicated-store representation.

### 10.7 Small-kernel block fusion

Small-kernel block fusion targets a connected subgraph of kernels that each
launch with `grid_dim = 1`. For such kernels the per-launch overhead
(§12.7 `launch_cycles_fit_sm`, default 800 cycles) is typically the largest
share of runtime, and neither producer-consumer, fanout, nor same-domain
horizontal fusion collapses launches unless the kernels share dataflow *and*
schedule. This pass fuses the whole subgraph into one launch by lowering
each per-kernel schedule to a per-layer partition of one shared block.

The synthesized kernel:

- has `grid_dim = 1`;
- runs one `compute[B_l]` per layer in topological order, with a
  layer-boundary `sync` between consecutive layers; `B_l` is
  `Σ_{k ∈ layer_l} it(k)`, where `it(k)` is kernel `k`'s `compute`
  iteration count and `layer_l` is the set of nodes at topological
  depth `l` within the subgraph;
- inside each layer's `compute`, dispatches with an if/else chain over
  disjoint half-open index ranges — one range per kernel in the layer,
  in seed-`NodeId` order; iterations outside every active range in a
  layer perform no work;
- routes internal producer-consumer values through shared-memory
  buffers sized to hold the seam values, so a producer iteration in one
  layer can hand off to a consumer iteration at a different index in the
  next layer.

`it(k)` and the fused `B_l` are iteration counts, not hardware thread
counts. The DSL allows an intermediate buffer or `compute[N]` where `N`
exceeds `DeviceModel::max_threads_per_block`; the lowering pipeline
maps those iterations onto a legal thread count by handling multiple
elements per thread, so small-kernel fusion imposes no direct bound on
`B_l`. Occupancy pressure from the resulting thread count is captured
by the estimator (§12.4), not by a fusion-time rejection.

Requirements for a legal small-kernel candidate:

- every kernel in the subgraph is a pure `GraphNode::Kernel` with
  `grid_dim = 1`;
- the fused shared-memory footprint (all internal seam buffers plus
  each kernel's per-block shared-memory usage) does not exceed the
  configured per-block shared-memory budget;
- every internal seam is single-writer with an access relation
  expressible as an identity or affine map over the shared block-local
  index space; cross-thread communication expressed as ordinary index
  accesses in the DSL is legal — the standard `insert_sync` lowering
  pass (§12.2) inserts intra-body syncs where the access pattern
  demands them, in addition to the layer-boundary syncs this pass
  emits;
- origins of the kernels in the subgraph are pairwise disjoint;
- the subgraph contains no black-box kernel, no partial memcpy/memset,
  and no mutable-alias node;
- no storage hazard touches an internal seam or an external output.

Candidate composition:

```text
inputs  = stable_unique(concat of k.inputs for k in S, minus internal seams)
seams   = values produced inside S and consumed inside S
outputs = drop:  values produced in S and consumed outside S, or registered
                 as graph outputs
          keep:  drop-set ∪ { seams that also have an external consumer or
                              are registered graph outputs }
```

Layering (deterministic):

1. For each node `k` in the subgraph, compute
   `depth(k) = 0` if `k` has no in-subgraph predecessor, else
   `1 + max{ depth(p) : p is an in-subgraph predecessor of k }`.
2. Bucket nodes by `depth` and sort each bucket by seed `NodeId`.
3. `layer_l` is bucket `l`; the layer count `L` is one plus the maximum
   depth.

Body construction, per layer `l` with kernels `k_1..k_{n_l}` in seed
order and widths `w_i = it(k_i)`, prefix sums `T_i = Σ_{j≤i} w_j`
(`T_0 = 0`) and `B_l = T_{n_l}`:

```text
compute[B_l] |idx| {
    if idx < T_1:
        body(k_1)[idx - T_0]
    elif idx < T_2:
        body(k_2)[idx - T_1]
    ...
    elif idx < T_{n_l}:
        body(k_{n_l})[idx - T_{n_l - 1}]
}

if l < L - 1: sync;
```

`body(k)[j]` is `k`'s HIR body with its `compute` iteration index
substituted by `j`. Seam reads inside `body(k)` are rewritten to load
from the seam's shared-memory buffer; seam writes are rewritten to store
there. External inputs and outputs retain their global-memory bindings
and are passed through the fused kernel's parameter list. `idx` is the
compute-iteration index, not a hardware thread id; the lowering pipeline
picks the actual thread count.

Shared-memory seam layout, per internal seam `s` produced by kernel
`k_p` and consumed by kernel `k_c`:

- allocate `shared_s[it(k_p)]` sized in `s`'s element type;
- the producer body, running at producer-local index `p`, writes at
  `shared_s[p]`;
- the consumer body, running at consumer-local index `c`, reads at
  `shared_s[phi_s(c)]`, where `phi_s = inv(w_s) ∘ r_s` is the composed
  seam access as in §10.1 — identity when the seam is elementwise, an
  affine permutation otherwise;
- the layer-boundary `sync` between `k_p`'s layer and `k_c`'s layer
  publishes the writes before the reads.

Multiple consumers of the same seam share one `shared_s` allocation.
When the seam is also externally consumed (keep variant), the fused
kernel additionally writes the same values through to the seam's
global-memory binding before the store.

Estimator interactions:

- `B_grid = 1` for the fused launch, so
  `launch_cycles = launch_cycles_fit_sm`. If the subgraph contained `n`
  original single-block kernels, the launch-cost saving is
  `(n − 1) · launch_cycles_fit_sm`, which is often the dominant benefit.
- The lowered KIR determines the thread count that feeds §12.4
  occupancy analysis. Small-kernel candidates whose combined register
  footprint or shared-memory usage drops `blocks_by_regs` or
  `blocks_by_smem` to zero are rejected by the estimator.
- Each layer boundary contributes `sync_latency_cycles` on the critical
  path (§12.6). The interpreter treats one layer body's critical path
  and the following sync as a single dependency edge.
- Iterations outside a layer's active range in the if/else dispatch
  contribute no operations or accesses to the estimator's counts.

Composition with other passes:

- The resulting fused kernel is a `grid_dim = 1` `GraphNode::Kernel`, so
  it can itself be a member of another small-kernel candidate in a
  later saturation round if the shared-memory budget still holds.
- Origins of the fused candidate are the union of all its source
  origins, so subsequent composition against any of those seeds fails
  the disjoint-origins check.
- Storage hazard reconstruction is unchanged: internal seams are dropped
  or kept as usual; no new `BufId` is introduced.

Candidate enumeration in `fusions/small_kernel.rs` (small-kernel does not
use pattern-key bucketing — see §10.0):

1. Filter the alternative graph to eligible seed kernels (`grid_dim = 1`,
   pure kernel, no black-box, no partial memcpy/memset, no
   mutable-alias node).
2. Iterate seed nodes in increasing `NodeId` order. For each seed, grow
   connected subgraphs by adjacency in a deterministic order (in-edges
   before out-edges, tie-broken by `NodeId`), admitting only nodes that
   preserve legality and keep the running shared-memory footprint within
   budget.
3. Bound the enumeration by `max_region_seed_nodes` (§11) and skip a
   subgraph whose seed set has already been emitted.
4. Synthesize each admitted subgraph directly, emitting the drop variant;
   emit the keep variant when at least one internal seam has an external
   consumer or is a registered graph output.

`CandidateKey` deduplication (§9) then collapses association-order
duplicates and any structurally-identical fused modules that happen to
normalize to the same `ArtifactKey`; the ILP's `z_m` semantics (§13.3)
handle module reuse across those instances.

## 11. Saturation and search-space control

The process is monotone but intentionally bounded; call it bounded saturation
in reports.

Initial defaults:

```text
max_rounds                            = 4
max_region_seed_nodes                 = 6
max_reachable_hir_nodes               = 2048
max_total_alternatives                = 5000
max_alternatives_per_boundary         = 8
max_alternatives_per_pass_per_round   = ceil(max_total_alternatives
                                             / (max_rounds
                                                * num_enabled_passes))
```

Every limit is configurable and every truncation is reported.
`max_alternatives_per_pass_per_round` is a soft cap that a pass may
overshoot to finish its current pattern-key bucket (§10.0);
`max_total_alternatives` is a hard cap enforced by the driver.

Per round:

1. Freeze a read-only view of current alternatives and indexes.
2. Call each enabled fusion pass in a fixed order. Each pass follows §10.0:
   pattern-keyed passes (producer-consumer, fanout) match every legal
   subgraph, bucket by pattern key so instances that share a fused
   `ArtifactKey` are emitted together, sort buckets by instance count
   descending, and synthesize drafts bucket by bucket in seed-`NodeId`
   order; small-kernel matches and synthesizes directly in seed-`NodeId`
   order with no bucketing.
3. Synthesize and normalize drafts in parallel.
4. Collect failures from steps 2 and 3 as counted diagnostics, not panics.
   Diagnostic categories are enumerated in `CandidateFailure`
   (`LegalityRejected`, `NormalizationFailed`, `LaunchScheduleMismatch`,
   `HirSizeCapExceeded`, `WouldCreateCycle`, `EstimationFailed`,
   `PerPassQuotaExhausted`, ...). Every failure counts against a bounded
   diagnostic buffer whose size is configurable and reported.
5. Sort successful drafts by `CandidateKey`. This sort is a
   determinism mechanism across passes — it does not override the
   compile-time-cost-aware bucket order produced within each pass by
   §10.0. Equal `CandidateKey`s from different passes collapse to one
   entry here, which subsumes step 7's structural dedup for exact
   duplicates.
6. Remove keys already present in `SaturationState::seen_candidates`.
7. Apply exact structural deduplication.
8. Apply boundary-local pruning.
9. Run the enabled acyclicity validation and insert accepted candidates
   serially.
10. Stop when zero candidates were inserted or `max_total_alternatives`
    would be exceeded by the next insertion.

Boundary pruning groups candidates by:

```text
(required value set, ordered output values, storage/effect signature)
```

Safe dominance removes a candidate only when another candidate in the group:

- has the same artifact key;
- has no greater estimated runtime;
- has no greater code size or resource estimate;
- is strictly better in at least one quantity.

Candidates with different artifact keys are not safely dominated because
global module reuse can make the locally slower candidate optimal. If the
per-boundary cap still fires, retain a deterministic Pareto/beam subset and
mark the run as heuristic rather than globally complete.

## 12. KIR-based cost estimator

The estimator analyzes the finalized candidate, not the source nodes' summed
costs. Fusion changes dynamic execution counts, CSE, memory maps, register
liveness, synchronization, and launch geometry.

### 12.1 Configuration

```rust
pub struct DeviceModel {
    pub sms: u32,
    pub warp_size: u32,
    pub max_threads_per_sm: u32,
    pub max_blocks_per_sm: u32,
    pub max_warps_per_sm: u32,
    pub registers_per_sm: u32,
    pub shared_bytes_per_sm: u32,
    pub global_sector_bytes: u32,
    pub dram_bytes_per_cycle: f64,
    pub issue_weighted_ops_per_cycle: f64,
    /// Launch overhead tiers, selected by grid size in §12.7.
    /// TODO: replace with a calibrated decision tree once measured launch
    /// data is available; these three constants are a rough starting point,
    /// not a target of accuracy.
    pub launch_cycles_fit_sm: f64,       // grid fits on one SM (default 800)
    pub launch_cycles_within_wave: f64,  // grid fits in one wave (default 1300)
    pub launch_cycles_multi_wave: f64,   // grid spans multiple waves (default 2000)
    pub global_latency_cycles: f64,
    pub sync_latency_cycles: f64,
    pub latency_saturation_warps: u32,
    pub op_latency: OpLatencyTable,
}

pub struct EstimatorConfig {
    pub device: DeviceModel,
    pub warp_samples_per_par: u32,
    pub register_fixed_overhead: u32,
    pub register_liveness_scale: f64,
    pub unknown_global_sectors_per_warp: u32,
    pub model_version: u32,
}

pub struct EstimateContext<'a> {
    pub graph_symbols: &'a BTreeMap<ir::VarId, i64>,
    pub param_bindings: &'a BTreeMap<String, i64>,
    pub artifact: &'a ArtifactContext,
}
```

Production defaults come from a named device profile or device query. Tests
use a small fixed synthetic profile.

### 12.2 Lowering

Use the pure `ModuleCompiler::lower` path:

```text
canonical HIR -> lower_to_kir -> layout_infer -> insert_sync
```

Call `plan_shared_mem(&kir)` to get the exact statically planned shared-memory
footprint. Require exactly one KIR kernel in the initial implementation.
Resolve every symbolic grid bound through `EstimateContext::param_bindings`;
an unbound launch extent is an estimation failure and makes that candidate
ineligible. Evaluate `BufInfo.size`, memcpy ranges, and memset ranges through
`EstimateContext::graph_symbols`.

### 12.3 Register estimate

Implement conservative backwards liveness over every `SSABlock`:

1. Seed live values from block yields.
2. Walk operations in reverse.
3. Remove operation results and add operation operands.
4. Include nested-region block operands, captures, yields, and loop-carried
   values.
5. Iterate a loop body's live-in/live-out sets to a fixed point.
6. Record the maximum live scalar word count at any program point.

Infer each `SSARes` type from its defining op and buffer declaration. Count
`BabyBear`/`u32` as one word and `FpExt` as four words. Treat unknown values
conservatively as four words.

```text
est_registers_per_thread =
    register_fixed_overhead
    + ceil(register_liveness_scale * max_live_words)
```

The scale and fixed overhead are calibrated offline. Reject a candidate only
when the estimate makes resident blocks zero; otherwise occupancy is a cost.

### 12.4 Occupancy

For block size `T`, estimated registers `R`, and planned shared memory `S`:

```text
blocks_by_threads = floor(max_threads_per_sm / T)
warps_per_block   = ceil(T / warp_size)
blocks_by_warps   = floor(max_warps_per_sm / warps_per_block)
blocks_by_regs    = floor(registers_per_sm / (R * T))
blocks_by_smem    = if S == 0
                      then max_blocks_per_sm
                      else floor(shared_bytes_per_sm / S)

blocks_per_sm = min(max_blocks_per_sm,
                    blocks_by_threads,
                    blocks_by_warps,
                    blocks_by_regs,
                    blocks_by_smem)

active_warps = blocks_per_sm * warps_per_block
```

Record `blocks_per_sm` as estimated occupancy and use `active_warps` for
latency hiding.

### 12.5 Global-memory transactions

Walk KIR `Par` accesses with their enclosing loop/grid domains. For each
global access site:

1. Select warp instances with a deterministic low-discrepancy sequence seeded
   by `(module_hash, kernel_index, par_node, access_index, model_version)`.
2. Evaluate the access map for every active lane:
   - `Linear`: evaluate the layout directly;
   - `Affine`: bind the par index, grid index, and enclosing loop variables;
   - `SExpr`/`Blackbox`: use the configured conservative fallback.
3. Convert element indices to byte addresses.
4. Count distinct `global_sector_bytes` sectors touched by the warp.
5. Average over samples and multiply by the exact dynamic number of warp-site
   executions.

Unknown pool-base alignment is handled by sampling every feasible sector
alignment residue when that set is small; otherwise use the worst alignment
among the deterministic samples.

`AccessEst` records at least:

```rust
pub struct AccessEst {
    pub requested_bytes: u64,
    pub transaction_bytes: u64,
    pub avg_sectors_per_warp: f64,
    pub dynamic_warp_accesses: u64,
}
```

Shared-memory bank-conflict sampling is a later estimator version. Register
and shared-memory accesses still contribute instruction and dependency costs
in v0.

### 12.6 Critical-path interpreter

Interpret KIR SSA dependencies, not numerical field values.

- A constant is ready at cycle zero.
- A scalar op result is ready at `max(operand_ready) + op_latency`.
- A global read entering a `Par` block adds base global latency, divided by a
  latency-hiding factor capped by `latency_saturation_warps`.
- A loop repeats its body critical path `bound` times while respecting
  loop-carried dependencies.
- A `Select` uses the maximum branch path until branch probabilities are
  available.
- `Sync` adds `sync_latency_cycles`.
- A write becomes ready after its yielded value and store issue latency.

Record:

- weighted dynamic operations;
- synchronization count;
- longest dependent global-load chain;
- critical cycles per block wave;
- requested and transaction bytes.

### 12.7 Aggregate cycles

For grid block count `B`:

```text
resident_blocks = sms * blocks_per_sm
block_waves     = ceil(B / resident_blocks)

latency_cycles = block_waves * critical_cycles_per_block_wave
bandwidth_cycles = total_global_transaction_bytes / dram_bytes_per_cycle
issue_cycles = total_weighted_ops / issue_weighted_ops_per_cycle

if B <= blocks_per_sm:
    launch_cycles = launch_cycles_fit_sm       // whole launch fits on one SM
elif B <= resident_blocks:
    launch_cycles = launch_cycles_within_wave  // fits in a single wave
else:
    launch_cycles = launch_cycles_multi_wave   // spans multiple waves

raw_cycles = max(latency_cycles, bandwidth_cycles, issue_cycles)
total_cycles = launch_cycles + raw_cycles
```

TODO: the three-tier launch overhead is a placeholder. Replace with a
calibrated decision tree over (grid size, block size, register pressure,
first-launch vs. warm) once measured launch data is available.

Do not multiply raw memory latency by every load count: only dependency depth
belongs on the critical path. Independent accesses contribute through
bandwidth and issue demand.

The queueing correction described in `fusion-plan-v2.md` is a later estimator
revision. Add it only after validating the roofline/critical-path model. The
proposed implementation is a bounded fixed-point calculation:

```text
utilization = min(0.95, bandwidth_cycles / estimated_total_cycles)
effective_memory_latency = base_latency / (1 - utilization)
```

Re-run the critical path and aggregate equation until relative change is below
1% or five iterations complete. Keep this disabled by default until measured
rank correlation improves on a held-out benchmark set.

### 12.8 Non-kernel costs

Non-kernel alternatives do not lower through KIR:

- `Const` has zero per-run cost unless runtime setup later proves otherwise.
- `Memcpy` and `Memset` use operation-launch cost plus transaction bytes
  divided by the configured copy/fill bandwidth.
- `BlackboxKernel` is never a fusion candidate. It appears in the alternative
  graph only as its original node and is the sole producer of its outputs, so
  the producer equation in §13.3 selects it implicitly whenever any of its
  outputs is demanded. Give it an optional caller-provided estimate;
  otherwise use zero. The zero cost cannot cause it to be dropped incorrectly
  because there is no alternative producer to compete against.

### 12.9 Estimator calibration

Add a benchmark mode that emits one JSON row per compiled kernel:

```text
artifact hash, estimator version, all estimator features,
estimated cycles, measured median cycles, nvcc wall time
```

Use warm CUDA-graph measurements for runtime and cold-cache measurements for
compile time. Fit only configuration constants; do not compile candidates
inside the optimization pass.

Acceptance for promoting an estimator version is based on decision quality:

- pairwise ranking accuracy on candidate alternatives;
- regret of the extracted graph relative to measured candidates;
- worst optimistic error for low-occupancy kernels;
- held-out kernels not used to fit constants.

### 12.10 KernelCostManager

Candidate enumeration, boundary pruning, and multiple fusion patterns all
call the estimator on the same normalized KIR kernels. Saturation rounds
compound this: the same residual module can be derived by several patterns
(e.g. `(A+B)+C` and `A+(B+C)` after canonicalization) and re-enters the cost
path each time. Because the estimator is deterministic in the KIR module and
the fixed `EstimateContext` for the run, this work should be memoized.

```rust
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct KirCostKey {
    /// Alpha-normalized residual HIR hash — the same value stored in
    /// `ArtifactKey::module_hash` (§5.5). Keying on the pre-lowering HIR
    /// hash lets a cache hit skip both lowering and interpretation; keying
    /// on the post-lowering KIR would still require running the
    /// `lower_to_kir -> layout_infer -> insert_sync` pipeline before every
    /// lookup, which is the more expensive half of estimation.
    pub module_hash: [u8; 32],
    /// Stable hash of the sorted `EstimateContext::param_bindings` map.
    /// Two candidates with the same normalized module but different concrete
    /// extents have distinct costs and distinct cache keys.
    pub param_bindings_hash: [u8; 32],
}

pub struct CostManagerStats {
    pub hits: u64,
    pub misses: u64,
}

pub struct KernelCostManager {
    device: DeviceModel,
    config: EstimatorConfig,
    graph_symbols: BTreeMap<ir::VarId, i64>,
    artifact: ArtifactContext,
    cache: HashMap<KirCostKey, GraphNodeCost>,
    stats: CostManagerStats,
}

impl KernelCostManager {
    /// `module_hash` is the caller-supplied `ArtifactKey::module_hash`;
    /// passing it in (instead of recomputing from `module`) lets a cache
    /// hit avoid rehashing the HIR.
    pub fn cost_of(
        &mut self,
        module_hash: [u8; 32],
        module: &GraphNode,
        param_bindings: &BTreeMap<String, i64>,
    ) -> Result<GraphNodeCost, CostError> {
        let key = KirCostKey {
            module_hash,
            param_bindings_hash: hash_param_bindings(param_bindings),
        };
        if let Some(cost) = self.cache.get(&key) {
            self.stats.hits += 1;
            return Ok(cost.clone());
        }
        let ctx = EstimateContext {
            graph_symbols: &self.graph_symbols,
            param_bindings,
            artifact: &self.artifact,
        };
        let cost = estimate_cost(&self.device, &self.config, module, &ctx)?;
        self.cache.insert(key, cost.clone());
        self.stats.misses += 1;
        Ok(cost)
    }

    pub fn stats(&self) -> &CostManagerStats { &self.stats }
}
```

Key composition rationale:

- `graph_symbols` and `ArtifactContext` are constant for the whole run and
  live in the `KernelCostManager` itself, so they do not need to be part of
  the key.
- `param_bindings` is per-candidate and must be part of the key because it
  resolves symbolic launch extents (§12.2) and access-map bounds; a candidate
  with the same normalized module but different bindings computes a different
  cost.
- The `module_hash` field is byte-identical to `ArtifactKey::module_hash`,
  which is already computed during candidate finalization (§9 step 8). A
  candidate that hits the compiled-module cache during extraction also hits
  the cost cache during enumeration, and a cache hit avoids the KIR
  lowering pipeline entirely.

The pass owns exactly one `KernelCostManager` for the whole `fuse_graph_v2`
invocation and passes it by `&mut` to any code that needs a cost. Report
`CostManagerStats` in `FusionReport` so hit rate is visible when tuning
saturation caps and boundary pruning.

## 13. Extraction model

### 13.1 Sets

Let:

- `A` be all alternatives;
- `V` be all logical value classes;
- `S` be graph input values;
- `D` be graph output values;
- `P(v) = { a in A | v in outputs(a) }`;
- `R(a)` be the deduplicated set of `AltGraphNode.inputs` required by
  alternative `a`;
- `K(a)` be the set of compiled artifacts required by `a`.

Black-box kernel nodes appear in `A` only as their original entry and are the
sole producer of their outputs; they are selected implicitly by the producer
equation in §13.3 whenever any of their outputs is demanded. There is no
separate mandatory-alternatives set.

### 13.2 Variables

```text
x_a in {0,1}   alternative a is selected
y_v in {0,1}   logical value v is available/materialized
z_m in {0,1}   compiled artifact m is required
```

There is no concrete-`BufId` variable and no per-result variable.

### 13.3 Constraints

Sources and demanded outputs:

```text
y_v = 1                                      for v in S union D
```

Exactly one producer for every selected non-source value:

```text
sum_{a in P(v)} x_a = y_v                    for v in V minus S
```

Selected alternatives require all boundary inputs:

```text
x_a <= y_v                                   for a in A, v in R(a)
```

Artifact activation:

```text
x_a <= z_m                                   for a in A, m in K(a)
z_m <= sum_{a: m in K(a)} x_a                for every artifact m
```

The second constraint gives `z_m` exact OR semantics even for cache hits or a
zero compile-cost weight.

Optional budgets:

```text
sum_m z_m <= max_modules
sum_{m not in original_artifacts} z_m <= max_new_modules
sum_m code_size_m * z_m <= code_size_budget
```

Default to `max_new_modules`, because it leaves the original extraction
feasible. If `max_modules` is configured below the original graph's required
artifact count, reject the configuration before solving.

### 13.4 Why multi-output all-or-none is implicit

If alternative `a` produces values `u` and `v`, then `x_a` occurs in both
producer equations. Selecting `x_a` forces both `y_u` and `y_v` to one and
prevents another selected producer for either value. No additional
all-results constraint is needed.

### 13.5 Objective

CP-SAT accepts integer coefficients. Convert estimated cycles to `i64`
objective units with a recorded `cycle_quantum`:

```text
runtime_units(a) = max(1, round(total_cycles(a) / cycle_quantum))
```

Choose the smallest power-of-ten quantum for which the worst-case sum fits in
`i64`; calculate the bound in `i128` before constructing the model.

The initial optimization policy is lexicographic in four stages, each
motivated by a distinct cost dimension:

1. **Runtime.** Minimize total predicted runtime:

   ```text
   sum_a runtime_units(a) * x_a
   ```

   The primary optimization target.

2. **Compile cost.** Holding runtime at its optimum, minimize the number
   of selected artifacts (`sum_m z_m`). This is the standing proxy for
   compile-wall-clock until §13.5's calibrated compile-cost term lands;
   each unique `ArtifactKey` costs one `nvcc` invocation.

3. **Graph size.** Holding runtime and artifact count, minimize
   `sum_a x_a`. Fewer selected alternatives means a smaller emitted graph,
   which reduces launcher overhead, planner work, and reporting noise;
   it also prefers the more consolidated of two runtime-equivalent
   plans.

4. **Peak-memory proxy.** Holding all of the above, minimize `sum_v y_v`.
   Sources and demanded outputs are pinned to 1 by §13.3 constraints, so
   this stage only bites on optional intermediates — dropping an
   otherwise-unused seam materialization reduces peak buffer live-range
   count, which the existing planner turns into a peak-memory saving.

If the Rust CP-SAT binding does not support native multiple objectives,
solve sequentially and add an equality or configured tolerance constraint
for each completed stage.

Do not assign uncalibrated seconds to compilation. Once compile-time
calibration exists, replace stage 2 with either a hard budget:

```text
sum_m compile_units(m) * z_m <= compile_budget
```

or an explicit expected-run objective folded into stage 1:

```text
expected_runs * runtime_units
    + compile_weight * sum_m compile_units(m) * z_m
```

`compile_units(m)` is calibrated per-artifact `nvcc` wall time and
`expected_runs` is a caller-provided workload characteristic (number of
times the compiled graph will run before recompilation); both live in
`FusionOptionsV2` once the calibration data set exists.

### 13.6 Solver behavior

Use the existing `cp_sat` crate under `planner-ortools`.

- Set a configurable wall-time limit, initially five seconds.
- Accept `Optimal` and `Feasible`.
- On `Unknown` or another no-solution status, return the original extraction.
- `Infeasible` is an internal error when no user budget excludes the original
  solution; report diagnostics and fall back safely.
- Record status, objective, best bound where available, solve time, and model
  size in `FusionReport`.

The original extraction is the incumbent/fallback. Solver hints are optional
and must not be required for correctness.

### 13.7 Why extraction needs no acyclicity constraints

The complete alternative graph is a DAG by the insertion invariant in
Section 9.1. A solver solution retains a subset of its alternative nodes,
value nodes, and incident edges. Every subgraph of a DAG is a DAG, so the
selected execution graph is acyclic independently of the values assigned to
`x`, `y`, and `z`.

Therefore the CP-SAT model has no topological-order variables, cycle cuts, or
solve/check/re-solve loop. A topological sort still runs during reconstruction
because emission needs a concrete order, but it is not part of optimization.

### 13.8 Partitioning

Do not partition the first implementation. Value constraints, shared artifact
variables, horizontal candidates, and global module budgets all couple
components.

A later exact partitioner operates on the full constraint factor graph,
including artifact variables. With a remaining global module budget, each
component must expose a runtime-versus-module-count frontier and a master
knapsack must allocate the budget. Partitioning only the dataflow graph is not
exact.

## 14. Reconstructing `GraphBuilder`

`apply.rs` accepts a validated `ExtractionSolution` and mutates the existing
builder only after all fallible checks pass.

### 14.1 Pre-commit preparation

1. Verify every demanded value and selected input constraint.
2. Resolve the unique selected producer of every selected non-source value.
3. Build RAW/WAR/WAW precedence edges as described above.
4. Topologically sort selected alternatives.
5. For every selected node, compare `inputs` and `outputs` with its
   positional graph-node API and validate the physical binding derived from
   `re_exported`. On any mismatch:
   - abort reconstruction before mutating `GraphBuilder`;
   - let the graph-take guard (§14.3) restore the seed-node prefix so the
     original extraction becomes the emitted plan;
   - record the failure as
     `FusionReport::FallbackReason::ReconstructBindingMismatch { node, kind }`
     with `kind` distinguishing input-arity mismatch, output-arity mismatch,
     unregistered physical `BufId`, and re-exported-version conflict.
   Any step-5 failure is a bug in the pass that produced the candidate, not
   a user-visible error; the fallback path preserves correctness while
   the report drives diagnosis.
6. Retain the topological `Vec<NodeId>` as the emission plan. Do not move
   nodes during validation. The black-box kernel is cloneable after the
   `Arc` change, but a constant may own a non-cloneable `DeviceBuffer`, so
   the general path still moves selected `GraphNode` values only at commit.

### 14.2 Buffer mapping

Every logical value maps back through:

```rust
let first = gf.re_exported[value.0];
let physical = BufId(first.0);
```

Candidate kernels therefore write the original physical buffer associated
with the semantic output class they replace. Dropped intermediates remain as
unused entries in `g.bufs`; the existing planner already skips completely
orphaned buffers. The duplicate `BufInfo` entries used for logical versions
exist only in `GraphFuser.bufs` and are not appended to `GraphBuilder.bufs`.

Multiple logical mutation versions intentionally map to the same original
`BufId`. MVP candidates never fuse through those versions, and the emitted
node order preserves their hazards.

No new `BufId` or `ValueClassId` is necessary for the initial fusion passes:
every alternative produces semantic values that already exist. A future
fusion pass that introduces a genuinely new materialized value must add one
physical `BufId`, its first logical class, and its `BufInfo` explicitly
rather than borrowing an arbitrary representative.

### 14.3 Moving selected nodes

After preparation succeeds:

```rust
let mut arena: NodeIdMap<Option<AltGraphNode>> =
    gf.nodes.into_iter().map(Some).collect();
let mut emitted = Vec::with_capacity(order.len());
for id in order {
    emitted.push(arena[id.0].take().expect("a selected NodeId is emitted once").node);
}
```

There is deliberately no original/candidate branch. Both are
`AltGraphNode { inputs, outputs, node }` addressed by the same `NodeId`.
Candidate finalization already created the normalized `GraphNode::Kernel`;
commit only moves it.

Then:

- assign `g.nodes = emitted`;
- set `g.plan = None`;
- run `kernel_dedup`;
- run existing DCE;
- run graph-interface and write-before-read validation;
- assert that post-fusion canonicalize/monomorphize does not change candidate
  artifact hashes.

The graph-take guard restores the seed-node prefix on every failure before
commit. After commit, validation failures are internal bugs; keep the arena
alive until validation succeeds so the guard can still reconstruct the
original seed prefix.

Move the current private interface/order validation out of `graph_exe.rs`
into a `pub(crate)` graph-validation helper so both compilation and v2 apply
call exactly the same implementation.

## 15. Options and reporting

```rust
pub struct FusionOptionsV2 {
    pub max_rounds: usize,
    pub max_region_seed_nodes: usize,
    pub max_reachable_hir_nodes: usize,
    pub max_total_alternatives: usize,
    pub max_alternatives_per_boundary: usize,
    pub max_alternatives_per_pass_per_round: usize,
    pub max_modules: Option<usize>,
    pub max_new_modules: Option<usize>,
    pub validate_alt_graph_acyclicity: bool,
    pub solver_time_limit_secs: f64,
    pub runtime_tolerance_ppm: u32,
    pub estimator: EstimatorConfig,
    pub enable_keep_variants: bool,
    pub enable_fanout: bool,
    pub enable_small_kernel: bool,
    pub enable_epilogue: bool,
    pub enable_horizontal: bool,
    pub verbose: bool,
}
```

`enable_keep_variants`, `enable_fanout`, and `enable_small_kernel` default
to `true` because those passes are part of the initial version.
`enable_epilogue` and `enable_horizontal` default to `false` until their
milestones pass measured gates. `validate_alt_graph_acyclicity` defaults
to true; turning it off removes validation work but does not relax
fusion-pass legality.

Extend the existing fusion report rather than changing `GraphExe`'s report
type immediately. Add optional/defaulted v2 fields:

```text
planner kind
alternatives generated/inserted/pruned
per-pass pattern-key count, instances per bucket, truncated bucket count
candidate failures by reason (see CandidateFailure enum in §11)
acyclicity checks and rejected candidates
saturation rounds and truncation flags
cost-cache hits/misses (from KernelCostManager, §12.10)
model x/y/z variable and constraint counts
solver status, time, objective, bound
estimated original and selected cycles
original and selected artifact counts
selected candidate count
fallback reason (FallbackReason enum below)
```

`FallbackReason` enumerates every path that causes v2 to emit the original
extraction instead of the solver's answer:

- `SolverUnavailable` — built without `planner-ortools`.
- `SolverStatusUnknown` — CP-SAT returned `Unknown` within the wall-time
  limit.
- `SolverStatusInfeasible` — CP-SAT proved infeasible; this is a v2 bug
  when no user budget excludes the original solution and is reported
  with a copy of the constraint violation.
- `SolverTimeout` — wall-time limit expired with no `Feasible` or better
  status.
- `ReconstructBindingMismatch { node, kind }` — §14.1 step 5 rejected
  the solver solution.
- `NoImprovementOverOriginal` — solver's objective did not beat the
  original within `runtime_tolerance_ppm`.
- `InternalError { message }` — catch-all for any other v2-side failure
  before commit.

Preserve existing report fields and semantics.

## 16. GraphCompiler integration

Keep the existing implementation as the default. Add an internal strategy enum:

```rust
enum FusionStrategy {
    Existing(FusionOptions),
    V2(FusionOptionsV2),
}
```

`GraphCompiler::fusion_options` continues to select the existing
implementation. Add:

```rust
pub fn fusion_v2_options(mut self, opts: FusionOptionsV2) -> Self;
```

When built without `planner-ortools`, requesting v2 runs enumeration and cost
analysis if desired but extracts the original graph and reports
`FallbackReason::SolverUnavailable` (see §15). Do not silently run the
existing implementation under the name v2.

Only change the default strategy after:

- correctness parity on all graph tests;
- no material compile-time regression under configured module budgets;
- measured runtime parity or improvement on the agreed golden suite;
- stable estimator-regret results.

## 17. Implementation milestones

Each milestone is independently reviewable and keeps the existing compiler
path passing.

### M0: independent shared fusion utilities

- Implement `HirVisitor`, deterministic occurrence traversal, scope tracking,
  and malformed-cycle detection in `fusion_utils.rs`.
- Implement `AccessCollector`, `StructureCollector`, `InputUseCollector`, and
  `ReachableNodeCounter` on the visitor.
- Make access extraction reject a hash-consed access node reached under two
  unequal index scopes, and expose `unique_index_scope` to fusion passes.
- Implement independent alpha-renamed expression cloning, boundary input
  interning, Quast composition, and inverse verification.
- Do not modify or import implementation code from the existing fusion pass.

Exit gate: focused visitor/utility tests pass, and the existing compiler path
is unchanged.

### M1: versioned seed model and original extraction

- Implement the ID aliases, dense arenas, seed conversion, re-exported version
  mapping, and dense producer/consumer indexes.
- Implement the toggleable insertion-time DAG validator. When enabled,
  validate the seed graph once after conversion.
- Implement original fallback solution and reconstruction.
- Round-trip representative graphs without adding candidates.

Exit gate: graph dump, interface, execution result, and physical hazard order
match the input after round-trip, modulo dead-node removal and module dedup.

### M2: exact extraction model

- Implement x/y/z CP-SAT model with integer objectives.
- Add module keys and budgets.
- Add original fallback and status reporting.
- Verify CP-SAT results against exhaustive enumeration for small generated
  alternative graphs.

Exit gate: exhaustive and CP-SAT optima agree for all toy/property cases.

### M3: producer-consumer drop-seams candidates

- Bind independently collected access relations to logical values.
- Implement direct producer/consumer-index enumeration.
- Implement multi-seam grouping, relation composition, bounded legality
  proofs, from-scratch synthesis, normalization, deduplication, and caps.
- Estimate original and candidate kernels with a temporary simple KIR cost if
  the full estimator is not yet landed.
- Reconstruct selected fused graphs.

Exit gate: identity, affine-permutation, nested-index, reduction-producer, and
multi-seam fixtures produce correct outputs against the unfused graph.

### M4: KIR estimator v0

- Implement shared-memory accounting, SSA type inference, liveness/register
  estimate, occupancy, transaction sampling, critical path, and aggregate
  cycles.
- Add deterministic golden feature snapshots.
- Add benchmark JSON export and calibration profiles.

Exit gate: estimator is deterministic and passes the ranking/regret threshold
chosen from the golden benchmark suite.

### M5: keep-seam variants

- Implement the producer-consumer keep-seam synthesis path.
- Add interface-output and fanout examples where keep beats materialize or
  duplicate.

Exit gate: solver chooses all expected materialize/drop/keep outcomes in
hand-costed tests.

### M6: bounded saturation and chain extraction

- Compose alternatives across rounds.
- Add persistent candidate keys, canonical origins, exact dominance, and
  heuristic cap diagnostics.
- Test association-order deduplication.

Exit gate: chain candidates terminate at a fixpoint or declared cap and are
deterministic across repeated runs.

### M7: fanout pass

- Implement multi-consumer (any `k >= 2`) fanout drop/keep candidates as
  described in §10.5.
- Extend `FusionHistory` with an n-ary fusion/origin variant and update
  dump serialization; the current binary producer/consumer variant
  remains the representation for producer-consumer fusion.
- Add pass-specific legality proofs and cost tests, including the
  "producer expression appears more than once at the fanout root"
  rejection check.

Exit gate: fanout candidates compute shared producer expressions once,
and the solver prefers fanout over duplicated producer-consumer candidates
when the seam is expensive.

### M8: small-kernel block fusion

- Implement `fusions/small_kernel.rs` per §10.7.
- Add topological layering, if/else dispatch synthesis, layer-boundary
  sync insertion, and shared-memory seam routing.
- Emit drop and keep variants; enforce block-size and shared-memory caps
  before enumeration commits.
- Add cost tests covering the launch-overhead saving, occupancy
  regression at a large fused iteration count, and layer-boundary sync
  accounting.

Exit gate: connected subgraphs of single-block kernels fuse to a single
launch with correct outputs, and the solver prefers the fused candidate
over the original chain when the launch overhead dominates the
**estimated** runtime. Measured-runtime confirmation happens in M11's
golden comparison, not at the M8 exit — M8 is a pass-correctness and
estimator-agreement milestone.

### M9: same-domain horizontal fusion

- Implement the restricted same-domain horizontal pass.
- Add reachability/convexity checks.
- Reject shared-memory and synchronization cases.
- Measure register/occupancy regressions.

Exit gate: no shape-changing or out-of-bounds output is possible, and the
estimator rejects or prices low-occupancy merges consistently with measurement.

### M10: epilogue pass

- Implement producer-schedule epilogues.
- Add pass-specific legality proofs and cost tests.

Exit gate: reductions followed by pointwise work retain the producer
schedule.

### M11: opt-in integration and golden comparison

- Add `GraphCompiler::fusion_v2_options`.
- Add reports/dumps visualizing alternatives and the selected extraction.
- Run compile-time, estimated-runtime, measured-runtime, and module-count
  comparisons against the existing and unfused baselines.

Exit gate: v2 remains opt-in but is usable by benchmarks and downstream test
graphs.

### M12: Numerical accuracy on fractional_sumcheck IR variants

Once M11 lands and the pass runs end-to-end, re-run the pairwise DSL/eager
tests in `crates/cuda-backend/src/logup_zerocheck/fractional_ir_dsl.rs`
(`dsl_port_tests`) with fusion v2 enabled. These tests already cover
every `fractional_sumcheck_gpu_ir` module — extract-root,
reduce-to-single-evaluation, compute-round, and the eq-hypercube stage
chain — pairwise against the eager `fractional_sumcheck_gpu` reference,
so they are the correct oracle for a numerical-accuracy validation of
fusion v2.

- Configure a run with `fusion_v2_options(..)` enabled and default
  patterns (producer-consumer drop/keep, fanout, small-kernel).
- Run every fixture in `dsl_port_tests` under both v2 and the existing
  fusion pipeline; compare BabyBear/FpExt outputs bit-for-bit against
  the eager reference.
- Cover the full parameter matrix (fraction count, extension depth,
  transcript seed) currently exercised by the eager tests.
- Record, per fixture: v1/v2 selected artifact count, cold compile time,
  measured runtime, and fusion report; include these in the M11
  comparison report.

Exit gate: every `dsl_port_tests` fixture matches the eager reference
bit-for-bit with fusion v2 enabled, on both the DSL-only and end-to-end
`fractional_sumcheck_gpu_ir` paths.

## 18. Test plan

### 18.1 Versioning and storage tests

- registered input read by several nodes;
- one writer followed by several readers;
- structured kernel reading and writing the same `BufId`;
- black-box carried output;
- partial memcpy and partial memset preserving the old value dependency;
- WAW chain with a reader between writers;
- registered output resolving to the final version;
- invalid unregistered read-before-write rejected.

### 18.2 Alternative-graph tests

- source values have no producers;
- every seed non-source value has one original producer;
- inserting two candidate producers updates the same value class;
- a candidate with the same value in its inputs and outputs is rejected;
- a candidate whose output reaches an input through several existing nodes is
  rejected;
- a candidate with no output-to-input path is accepted;
- disabling insertion validation skips the traversal without changing
  accepted valid candidates;
- positional duplicate operands do not duplicate solver constraints;
- multi-output candidate selection forces all output values;
- black-box kernel node cannot become a fusion candidate and is selected
  implicitly by the producer equation when any of its outputs is demanded;
- persistent candidate dedup works across rounds.

### 18.3 Solver tests

- simple chain chooses original, pair, or full-chain alternative by cost;
- fanout chooses among materialize, duplicate drop candidates, keep, and
  shared fanout;
- one artifact used by several selected alternatives is charged once;
- different residual artifacts are charged separately;
- `max_new_modules` leaves the original solution feasible;
- zero-second/unknown solver status returns the original extraction;
- randomized models with at most 12 alternatives match brute-force optimum.

### 18.4 Fusion-pass tests

- visitor child order, balanced callbacks, nested compute/reduce scopes, and
  malformed-cycle rejection;
- a shared access node under equal scopes is accepted and deduplicated, while
  the same node under unequal scopes returns `AmbiguousAccessScope`;
- independent identity, affine-permutation, nested-index, and reduction
  producer-consumer fixtures;
- a graph with `n` structurally-identical producer-consumer pairs produces
  one pattern-key bucket of size `n`, buckets are emitted in descending
  instance-count order, and every instance in the bucket is emitted before
  the next bucket starts;
- a producer-consumer bucket sharing an `ArtifactKey` charges a single
  `z_m` when the ILP selects any of its instances;
- all read sites for a removed seam are rewritten;
- two seams between the same nodes fuse atomically;
- keep variant returns a byte-for-byte equivalent intermediate;
- registered-output seam forces materialization elsewhere or a keep variant;
- overlapping origins rejected by generic composition;
- multi-consumer fanout computes the producer expression once and threads
  it into every consumer body for `k = 2` and `k = 3` fixtures; the
  "producer appears more than once at fanout root" rejection fires when a
  producer-consumer × producer-consumer merge is attempted;
- small-kernel block fusion collapses a chain of three `grid_dim = 1`
  kernels into one launch; parallel-eligible siblings share one
  `compute[B]` via if/else dispatch, and layer-boundary syncs correctly
  order cross-layer seam handoffs through shared memory;
- small-kernel candidate is rejected when the combined shared-memory
  footprint exceeds the configured per-block shared-memory budget;
- small-kernel accepts a subgraph whose per-layer `compute[N]` iteration
  count exceeds `max_threads_per_block` and whose kernel bodies contain
  index-access cross-thread communication; the lowering pipeline handles
  the thread-count mapping and `insert_sync` places intra-body syncs;
- epilogue retains producer launch shape;
- horizontal fusion requires equal domains and has no masked out-of-range
  stores.

### 18.5 Estimator tests

- contiguous warp access uses fewer sectors than a fixed-stride access;
- broadcast, coalesced, strided, and black-box access fallbacks;
- increased register liveness lowers estimated resident blocks at the expected
  threshold;
- increased shared memory lowers resident blocks;
- independent loads affect bandwidth but not dependent-load depth;
- reduction loop increases critical path by its bound;
- sync contributes configured latency;
- fixed module/config produces bit-for-bit identical estimates.

### 18.6 Reconstruction and end-to-end tests

- original extraction round-trip;
- drop-intermediate removes the selected materialization;
- keep-intermediate retains it for another consumer;
- mutable original chains preserve execution results and planner order;
- module count matches selected artifact variables;
- post-apply graph passes are idempotent;
- Poseidon2 and NTT small cases match unfused outputs;
- `fractional_ir_dsl.rs::dsl_port_tests` fixtures match the eager
  `fractional_sumcheck_gpu` reference bit-for-bit with fusion v2 enabled
  (M12 numerical accuracy check).

## 19. Benchmarks and validation

Add `benches/fusion_v2_comparison.rs` after M3. For each workload record:

```text
enumeration time
normalization/estimation time
solver time and status
total optimization time
alternatives before/after pruning
selected graph nodes
unique residual modules
nvcc cold compile wall time
kernel-cache hit count
estimated runtime
measured warm runtime
peak graph memory after the existing planner
```

Golden workloads initially include:

- small and medium Poseidon2 graphs;
- NTT sizes covering one block and multiple blocks;
- a vertical pointwise chain;
- a producer feeding two consumers;
- a fold/reduce case known to punish producer inlining;
- repeated structurally identical kernels to exercise module reuse;
- a register-heavy candidate that loses occupancy;
- the `fractional_sumcheck_gpu_ir` DSL modules (extract-root,
  reduce-to-single-evaluation, compute-round, eq-hypercube stage chain)
  as the primary end-to-end numerical-accuracy workload.

Use CUDA profiler start/stop around only the core measured graph computation,
with one descriptive NVTX range per workload. When profiling with NSYS, use
`--cuda-graph-trace=node` and `--gpu-metrics-devices=visible`.

## 20. Build and verification commands

Run checks for the affected crate only:

```bash
cargo check -p crypto-compiler
cargo nextest run -p crypto-compiler
cargo test -p crypto-compiler -- <single_fusion_v2_test>
```

When OR-Tools is available:

```bash
cargo check -p crypto-compiler --features planner-ortools
cargo nextest run -p crypto-compiler --features planner-ortools
cargo test -p crypto-compiler --features planner-ortools -- <single_fusion_v2_test>
```

Before a commit or PR:

```bash
cargo clippy -p crypto-compiler --all-targets --tests -- -D warnings
cargo +nightly fmt -- --check
```

CUDA benchmark execution requires the CUDA toolchain/GPU and is a separate
validation step from the crate's default check.

## 21. Completion criteria

V2 is ready to replace the existing implementation as the default only when
all of the following hold:

1. Versioning, effects, and reconstruction have property-test coverage and no
   known semantic exceptions on supported graph nodes.
2. The original extraction is always available and fallback is tested.
3. CP-SAT matches brute force on small generated models.
4. Candidate normalization hashes match the modules eventually compiled.
5. No candidate requires `nvcc` or GPU execution to be priced.
6. Search caps and solver timeouts are visible in reports.
7. The measured golden suite shows no correctness failures and acceptable
   runtime regret relative to the best measured enumerated alternative.
8. Cold compile time and selected unique-module count stay within configured
   budgets.
9. `fractional_ir_dsl.rs::dsl_port_tests` pass bit-for-bit against the eager
   `fractional_sumcheck_gpu` reference with fusion v2 enabled, on both the
   DSL-only and end-to-end `fractional_sumcheck_gpu_ir` paths (M12).
10. The existing implementation remains selectable for at least one release
    or migration window.

## 22. Explicitly deferred design decisions

The following should not be decided implicitly while implementing an earlier
milestone:

- whether mutable fusion can preserve same-pointer alias requirements;
- whether partial/scatter writes need general Presburger relations;
- how predicated stores are represented for different-domain horizontal
  fusion;
- whether compile cost is total compiler work or parallel wall-clock makespan;
- whether peak graph memory becomes a joint extraction objective;
- whether large models use factor-graph partitioning, column generation, or a
  solver-independent heuristic;
- whether calibrated estimator uncertainty is optimized as mean, upper
  confidence bound, or a robust interval objective.

Until those decisions are made, the implementation must reject the affected
candidate class or leave the corresponding objective disabled rather than
silently relying on an optimistic approximation.
