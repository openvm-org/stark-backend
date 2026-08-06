# Kernel Fusion Framework V2

In the first version of the kernel fusion pass pairs of kernels are eagerly fused together, leading to sub-optimal kernel fusions and excessive compilation times.

In this version we are going for something more clever, by using a combination of equality saturation and combinatorial optimization with CP-SAT or an ILP.

We are going to separate the task of fusing kernels from the task of figuring out which set of fusions is optimal.

Therefore we need a `GraphFuser` that can represent multiple potential sub-graphs as a hypergraph:


```rust 

/// represents the class of bufs (OR node)
pub struct BufClassId(usize); 

/// represents the compute node (AND node)
pub struct NodeId(usize); 

pub struct GraphFuser {
  /// BufClassId indexes into this
  bufs: Vec<BufInfo>,
  /// inputs and outputs are BufIds, NodeId indexes into this
  nodes: Vec<GraphNode>,

  /// hypergraph edges from BufClassId to BufId
  producers: HashMap<BufClassId, Vec<UseInfo>>,
  /// hypergraph edges from BufClassId to their users
  consumers: HashMap<BufClassId, Vec<UseInfo>>,
  /// maps to the canonical BufClassId
  canon_id: HashMap<BufId, BufClassId>,

  access_relations: HashMap<NodeId, AccessRelation>,

  inputs: Vec<BufClassId>,
  outputs: Vec<BufClassId>,
}

pub struct UseInfo {
  node: NodeId,
  pos: usize
}

pub struct SubGraphSubst {
  inputs: Vec<(BufId, BufClassId)>, // ties the particular buffer to the canonical class
  outputs: Vec<(BufId, BufClassId)>,
  node: GraphNode,
}

```

This design is so the fusion can happen in parallel, `GraphFuser` gets borrowed immutably, and fusion logic produces `SubGraphSubst`, which can be collected into a vec of updates. Then you'd insert `SubGraphSubst` into `GraphFuser` mutably and apply canonicalization as done in equality saturation.

Since the graph has the SSA property: each buffer is the output of exactly one node (modulo equivalences), and we may assume that each node supports the following positional api:

```rust 
impl GraphNode {
  fn get_operands(&self) -> Vec<BufId> {...}
  fn get_results(&self) -> Vec<BufId> {...}
}
```

the access relations needs to be simplified to the following:

```rust

struct AccessRelation {
  reads: Vec<ReadRelation>,
  writes: Vec<WriteRelation>,
  index_bounds: HashMap<VarId, i64>,
}

struct ReadRelation {
  read: Quast,
}

struct WriteRelation {
  write: Quast,
  inv: Quast,
}
```
since most writes are identities, the only writes that are not are those with scatter maps, on which the user supplies the inverse.

Note that graph nodes that mutate inputs keep the SSA property by carrying the mutated input in its output; future readers read from the carried output and for buffers that gets mutated, there is only allowed to be one consumer.

## Extraction

After a series of subgraphs are inserted into the `GraphFuser`, it now contains a bunch of subgraphs that needs to be extracted. We first need a cost function:

```rust
pub type Cycles = f64;
pub type Secs = f64;

pub struct GraphNodeCost {
  grid_dim: u32,
  block_dim: u32,
  bytes_read_per_output: u64,
  flops_per_output: u64,
  code_size: u64,
  
  raw_cycles: Cycles, // not counting kernel launch
  kernel_launch: Cycles,
  compilation_time: Secs,

  total_cost: f64
}

pub fn estimate_cost(module: &GraphNode) -> GraphNodeCost {
...
}
```

to be clear, estimate_cost is only a rough approximation. The algorithm is a data-flow analysis: the number of memory reads (of global memory) is the amount of memory read to produce each output element. The number of flops is the number of operations needed to produce each output element, not the total amount. The number of cycles is the maximum of the bandwidth and access latency bounds. If there are `N` elements produced and each element takes `k` memory reads and `t` flops, and the total parallelism of the GPU is `G` (the product of the sm count and max threads per sm count). Then there are `W=ceildiv(N, G)` waves.
The latency limit is `L = W * cycles_per_elem * k`, the bandwidth limit is `B = N * k * bytes_per_elem / bytes_per_cycle`, the flop limit is `F = t * N / flops_per_cycle`. Then `raw_cycles = max(L, B, F)`.
And `kernel_launch = 2000` cycles for simplicity. 

For the sake of simplicity compilation_time is a linear model of the op count.



We'll need to encode the extraction problem as an ILP. 

- Let `C_i in {0,1}` denote that the ith `BufClassId` is chosen or not. 
- Let `b_j in {0,1}` denote that the jth `BufId` is chosen
- Let `N_t in {0,1}` denote that the tth `NodeId` is chosen
- Let `M_l in {0,1}` denote that the lth module is compiled (zero or more `Node`s correspond to a compiled module.)
- Let `O` be the maximum number of modules we are willing to compile

Then we need the following constraints:

For every `i in outputs`, `C_i = 1`: every output `BufClassId` must be chosen
For every `i in inputs, C_i = 1`: every input `BufClassId` must be chosen 

For every `i`, `C_i == sum_{j in producers(i)} b_j`:  `C_i` is chosen iff one of the bufids that produces it is chosen
For every `t`, `N_t * num_results(t) == sum_{j in results(t)} b_j`: a node `t` is chosen iff of all of it's results is chosen
For every `t`, `N_t * num_operands(t) <= sum_{j in operands(t)} b_j`: a node `t` is chosen implies all of it's operands are chosen

- `sum_l M_l <= O`
- For every `l`, `sum_{t in nodes_of(l)} N_t <= M_l * num_nodes_of(l)`: if any node `t` uses module `l`, it's compiled


Then the cost is `C == sum_t cost(t) * N_t + compile_time_discount * (sum_l cost(l) * M_l)`. 


## Detailed plan

New pass lives in `src/passes/fusion_v2/` alongside (not replacing) the current `passes/fusion.rs`. Existing pass stays wired into `runner.rs` until v2 reaches parity on the golden benches.

### Files

```
src/passes/fusion_v2/
  mod.rs           // public entry: fuse_graph_v2(g, opts) -> FusionReport
  graph_fuser.rs   // GraphFuser data structure + BufId -> BufClassId map
  access.rs        // AccessRelation (thin wrapper reusing passes::fusion types)
  candidates.rs    // enumeration -> Vec<SubGraphSubst>
  cost.rs          // per-node/per-module cost model (reuses KernelCost)
  extract.rs       // ILP / CP-SAT extraction using or-tools bindings
  apply.rs         // rewrite GraphBuilder from an extraction solution
  tests.rs         // unit tests (behind planner feature)
```

Only the extraction step depends on the `planner` cargo feature (or-tools). Everything else is unconditional so `cargo check -p openvm-compiler` stays working without or-tools installed.

### Step 1 — `GraphFuser` data structure

Matches the shape at the top of the doc verbatim; the additions here are only bookkeeping the pass needs (per-node access relations, input BufId tracking, cost cache).

`graph_fuser.rs`:

```rust
pub struct GraphFuser {
    bufs:  Vec<BufInfo>,        // indexed by BufId
    nodes: Vec<GraphNode>,      // indexed by NodeId

    producers: HashMap<BufClassId, Vec<UseInfo>>,
    consumers: HashMap<BufClassId, Vec<UseInfo>>,
    canon_id:  HashMap<BufId, BufClassId>,

    access_relations: HashMap<NodeId, AccessRelation>,

    inputs:      Vec<BufClassId>,
    outputs:     Vec<BufClassId>,
    input_bufs:  Vec<BufId>,    // parallel to `inputs`; the concrete BufId
                                // seeded into each input class at from_graph time
                                // (input classes have no producer UseInfos, so
                                // members() needs this to enumerate them).

    next_class:  usize,         // monotone class-id allocator
    cost_cache:  RefCell<HashMap<NodeId, GraphNodeCost>>,
}
```

`BufClassId(pub usize)`, `NodeId(pub usize)`, and `UseInfo { node: NodeId, pos: usize }` are as declared at the top.

**`class_members` is a derived function, not a stored field.** Every concrete BufId is either a graph input (tracked via `input_bufs`) or the result of exactly one node — SSA plus the invariant that mutated buffers carry through as outputs (top-of-doc line 84). So the members of a class are exactly the union of:

- for each `UseInfo { node, pos }` in `producers[c]`: `nodes[node.0].get_results()[pos]`
- if `c` is an input class: the corresponding entry in `input_bufs`

```rust
impl GraphFuser {
    pub fn class_of(&self, b: BufId) -> BufClassId { self.canon_id[&b] }

    pub fn members<'a>(&'a self, c: BufClassId) -> impl Iterator<Item = BufId> + 'a {
        let from_producers = self.producers.get(&c).into_iter().flatten()
            .map(move |u| self.nodes[u.node.0].get_results()[u.pos]);
        let from_inputs = self.inputs.iter().zip(&self.input_bufs)
            .filter_map(move |(&ic, &ib)| (ic == c).then_some(ib));
        from_producers.chain(from_inputs)
    }

    pub fn producers(&self, c: BufClassId) -> &[UseInfo] {
        self.producers.get(&c).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn consumers(&self, c: BufClassId) -> &[UseInfo] {
        self.consumers.get(&c).map(|v| v.as_slice()).unwrap_or(&[])
    }
}
```

This also means the ILP's `sum_{j ∈ class_members(c)} b_j` (constraint (2)) is materialized at solver-encoding time by iterating `members(c)` — no extra data structure to keep in sync.

`GraphFuser::from_graph(g: &GraphBuilder) -> GraphFuser` seeds 1:1: allocate a fresh singleton `BufClassId` per source BufId, populate `canon_id` and the hyper-edges by scanning `g.nodes`, record `input_bufs` parallel to `inputs`. Access relations are populated by `access::extract` (Step 2).

Merging classes never happens explicitly. When Step 4 inserts a `SubGraphSubst`, the substitution's `outputs` tell us "this fresh BufId belongs in *this existing class*"; `canon_id.insert(fresh_bid, existing_class)` is the only class-side write. Classes are only ever *extended* — never merged, never renamed — so no union-find.

### Step 2 — `AccessRelation`

`access.rs` defines the simplified shape from the top of this doc:

```rust
pub struct AccessRelation {
    pub reads:  Vec<ReadRelation>,
    pub writes: Vec<WriteRelation>,
    pub index_bounds: HashMap<VarId, i64>,
}

pub struct ReadRelation  { pub read:  Quast }
pub struct WriteRelation { pub write: Quast, pub inv: Quast }
```

Writes are identities in the common case; the `inv: Quast` is only meaningful for scatter writes and is supplied by whoever built the node.

`fn extract(node: &GraphNode) -> AccessRelation` builds this by walking the kernel body. It is a fresh, small walker — NOT a wrapper around `passes::fusion::extract_one` — because that module's `AccessRelation`/`ReadAccess`/`WriteAccess`/`KernelClass`/`InverseMap` chain carries baggage (case A/B kernel-class dispatch, tree-lowered-reduce detection, per-site sigma bookkeeping) that fusion v2 does not need: v2's ILP-based extractor operates over whole subgraphs at once, so the per-read-site plumbing that v1 needs disappears.

Non-kernel nodes (memcpy, const) get a hand-written `AccessRelation`: identity `Quast` over the buffer extent, empty inverse.

### Step 3 — Candidate enumeration (saturation loop)

Every fusion pattern is a **two-phase transform** operating on the concrete graph, not on classes:

```rust
pub trait FusionPattern: Sync {
    /// Phase A — Extract. Scan concrete NodeIds / BufIds in `gf` and yield
    /// zero or more matched instances. A match is a concrete slice of the
    /// graph (specific producer + consumer NodeIds, specific intermediate
    /// BufIds) that this pattern knows how to rewrite. Reads `gf` immutably.
    fn extract(&self, gf: &GraphFuser) -> Vec<Self::Match>;

    /// Phase B — Apply. Given a concrete match, synthesize a replacement
    /// GraphNode (with a fresh ir::Module, fresh output BufIds) plus the
    /// canonicalization info needed to plug it back into the hypergraph.
    fn apply(&self, gf: &GraphFuser, m: &Self::Match) -> SubGraphSubst;

    type Match: Send + Sync;
}
```

**Why the split.** Fusion rewriting fundamentally operates on *concrete* SSA nodes and BufIds — an `ir::Module` is grafted by α-renaming variables against the producer's actual output binder, walking the actual consumer's read sites, and emitting a new module body. Classes are a hypergraph abstraction that only makes sense at extraction time (the ILP picks *one* concrete producer per class). But candidate enumeration and synthesis must see the concrete world:

- Matching a "kernel-kernel fusible pair" needs concrete access relations, concrete `Quast` expressions, concrete binder counts. Two `BufId`s in the same class may have *equivalent* content but be produced by structurally different modules — pattern matching happens per-`(producer NodeId, consumer NodeId)` pair, not per-class.
- Synthesis produces fresh `BufId`s for the new module's outputs. Those fresh BufIds are then declared equivalent to an existing class (the class the fused output would have replaced), which is a pure map-write in Step 4.

The hypergraph shows up only in *where* patterns look:

- To find fusion opportunities across a buffer, iterate `producers[c]` × `consumers[c]` for each class `c`. Both endpoints are concrete `NodeId`s. Since fusion is destructive at the concrete level (produces a new module, doesn't rewrite the old one), we don't care that `c` may already have alternative producers — we just add ours as another option.
- When a pattern's applicability depends on "is buffer B consumed only once?", it queries `gf.consumers(gf.class_of(b))`. That count is class-scoped, meaning it counts consumers of any equivalent producer — which is what fusion legality requires (if any variant is read twice, materialization can't be skipped).

`candidates.rs`:

```rust
pub struct SubGraphSubst {
    /// For each input BufId of the new GraphNode: the existing class it aliases.
    /// Insertion rewires the new node's operands to any concrete representative
    /// of that class (the smallest BufId — deterministic and stable).
    pub inputs:  Vec<(BufId, BufClassId)>,

    /// For each output BufId of the new GraphNode (freshly allocated by apply()):
    /// the existing class this fresh BufId joins. Fusion rewrites always
    /// produce outputs equivalent to a class that already exists in the fuser
    /// (the class of the buffer the fusion replaces), so no None case.
    pub outputs: Vec<(BufId, BufClassId)>,

    pub node: GraphNode,
    pub access: AccessRelation,
    pub provenance: Provenance,
}

pub enum Provenance {
    PairFuse  { src: NodeId, dst: NodeId, via_buf: BufId },
    WidthFuse { a: NodeId, b: NodeId },
}
```

The saturation loop lives in `passes/fusion_v2/mod.rs`:

```rust
pub fn fuse_graph_v2(g: &mut GraphBuilder, opts: &FusionOptionsV2) -> FusionReport {
    let mut gf = GraphFuser::from_graph(g);
    let patterns: Vec<Box<dyn FusionPattern>> = vec![
        Box::new(PairFusePattern),
        Box::new(WidthFusePattern),
    ];

    for round in 0..opts.sat_rounds {
        // Phase A: parallel extraction over the immutable fuser.
        let matches: Vec<SubGraphSubst> = patterns.par_iter()
            .flat_map(|p| p.extract(&gf).into_par_iter()
                .map(move |m| p.apply(&gf, &m)))
            .collect();

        // Deduplicate: two patterns may synthesize structurally identical
        // modules from different matches — keep one representative per
        // (node_hash, inputs_class_tuple, outputs_class_tuple).
        let matches = dedup_substs(matches);

        if matches.is_empty() { break; }

        // Phase C: canonical insertion (Step 4).
        gf.insert_all(matches);
    }

    // Extract (Step 6) and apply (Step 7).
    let sol = extract::solve(&gf, opts);
    apply::apply(g, gf, sol)?;
    ...
}
```

Key properties:

- Enumeration is over `&gf` — patterns can run in parallel with no synchronization; `SubGraphSubst`s are collected into a `Vec` and applied serially in Phase C.
- The loop is monotone: `insert_all` only *adds* nodes and BufIds (`class_members` grows; `producers`/`consumers` gain entries; `canon` gains keys). Nothing is ever removed. Termination comes from bounding `sat_rounds` and from patterns being idempotent modulo dedup (a pattern that already fired on a concrete pair produces a match with the same `(node_hash, class tuple)`; the deduper drops it).
- No pattern needs mutable access to `gf` mid-round.

**MVP patterns:**

Only two patterns for MVP: **PairFuse** (vertical, dataflow-connected) and **WidthFuse** (horizontal, dataflow-independent). Longer vertical chains fall out of repeated `PairFuse` rounds inside the saturation loop, and rematerialization is what happens when extraction picks a fused variant *and* the un-fused shared-producer variant simultaneously (constraint (5) allows only one producer per class, but both fused and un-fused consumers coexist in the fuser after Phase C — the ILP just picks the cheaper set). Rematerialization as a distinct pattern isn't needed once the two `SubGraphSubst`s (fused-into-consumer-A and fused-into-consumer-B) both live in the hypergraph.

**1. PairFusePattern — vertical fusion.**

`extract`: scan each class `c` and yield one match per `(prod_use, cons_use)` in `producers[c] × consumers[c]` where both endpoints are `Kernel` and the pair passes hard preconditions:

- WAR safety: consumer doesn't write a buffer the producer still reads through some other path.
- Output aliasing: fused kernel's outputs don't overlap its own inputs.
- Case A / Case B dispatch (from v1): identify whether the consumer reads the producer's output at an affine offset (Case A — direct substitution) or via a DivMod seam (Case B — inner binder split). Reject if neither applies.
- Body-ops budget: combined HIR node count under `opts.max_body_ops`.
- Interface exposure: `c` is not a graph output (extraction can drop it, but interface classes can't be dropped).

The v1 *scoring* is not reused — v2 doesn't need per-candidate profitability at enumeration time because extraction (Step 6) globally optimizes over the resulting hypergraph. But v1's *anti-patterns* are baked in as extra preconditions in `extract` since letting them into the fuser bloats the ILP without upside:

- **Fold-into-reduce hazard.** If the consumer's read site is inside a `reduce[R]` loop and the producer has `k` DRAM loads per output, inlining puts `k·R` sequential DRAM ops on the thread's critical path. Reject when `k · R > opts.max_per_thread_loads` (default 8). This is what v1's `per_thread_bw` term detected; v2 makes it a hard prune since a candidate that will always lose is a waste to encode.
- **Flop-blowup rematerialization.** Case B fusions may inline the producer at `m` consumer sites, multiplying producer flops by `m`. Reject when `m · producer_flops_per_output > opts.max_flop_blowup · producer_bandwidth_savings` (rough per-candidate check — no compile required, both terms derive from the access relation).
- **Parallelism collapse.** If the fused kernel would launch strictly fewer waves than the un-fused consumer (e.g. the fusion serializes a reduce that used to have grid-level parallelism), reject.
- **Register pressure proxy.** Op count is a rough register-count proxy; the body-ops budget doubles as a register-pressure guard. No compile.

`apply`: graft the producer module into the consumer via α-renamed σ substitution (mechanics reused from v1's `apply_fusion` / `GraftCx` in `passes::fusion:1171-...` — the graft *logic* is well-tested; only the enumeration/scoring shell is new). Returns a fresh `Kernel` node with:
- inputs = union of producer inputs and consumer inputs (with `c` removed);
- outputs = consumer's outputs (each fresh BufId mapped to the consumer's corresponding output class).

**2. WidthFusePattern — horizontal fusion.**

Two `Kernel` nodes with no dataflow relationship (neither is a transitive ancestor of the other) and structurally similar shapes get fused into one wider kernel: one launch instead of two, one compile instead of two, at the cost of some register pressure.

`extract` iterates unordered `Kernel × Kernel` node pairs `(a, b)` with `a.NodeId < b.NodeId` (dedup) and yields a match when all of the following hold:

- **Independence.** Neither is a transitive ancestor of the other. Computed once per round by a reachability scan; O(|nodes|²) worst-case but the pair space is already O(|nodes|²) so it's not the bottleneck.
- **Shape compatibility.** Both kernels have a top-level compute; extract the outer par bound. If either kernel is *tiled* (nested compute, has an inner par), require identical inner block sizes across the two. If both are flat, always compatible on the shape axis. Different-outer-bound kernels are still allowed — the fused kernel's grid is `max(N_a, N_b)` with the shorter kernel masked past its bound.
- **Op-count similarity.** `min(ops_a, ops_b) / max(ops_a, ops_b) ≥ opts.width_op_ratio` (default 0.5). Ensures we don't glue a huge kernel to a tiny one — the tiny one would inflate register pressure across the whole grid for no launch-count savings on the huge one.
- **Combined body-ops budget.** `ops_a + ops_b ≤ opts.max_body_ops` (same guard as PairFuse).
- **No shared inputs required.** Not a hard rule, but width fusion with overlapping input classes is a bonus (one DRAM read serves both bodies). No preconditions on it — extraction cost model already prices it via `bytes_dram_per_launch`.

`apply` synthesizes a `Kernel` whose body is a `select` on the grid index picking body A or body B:

```
compute[max(N_a, N_b)] |g|
  if g < N_a then body_a(g) else scalar_zero    // A's output
  if g < N_b then body_b(g) else scalar_zero    // B's output
```

The two logical outputs stay separate BufIds (fresh, mapped to A's and B's output classes respectively). The scatter map (top-level `#[scatter]`) is a diagonal identity per output — trivially bijective. Op-count of the width-fused kernel is `ops_a + ops_b + O(1)` for the mask; combined DRAM traffic is the union of A's and B's read/write sets (no duplication if input classes overlap, since we rewire to the class rep in Step 4).

Termination guard for the pattern: emitting one width-fuse per unordered pair per round means O(|Kernel_nodes|²) matches per round in the worst case. Cap via `opts.width_pairs_per_round` (default 128, tuneable) — take the top-K by *rough* affinity (op-count similarity × shared-input class count) and let extraction sort out which are actually profitable.

### Step 4 — Canonical insertion

```rust
impl GraphFuser {
    pub fn insert_all(&mut self, substs: Vec<SubGraphSubst>);
}
```

For each `SubGraphSubst` (the fresh output BufIds inside `subst.node` were allocated by `apply()` from a shared counter — see below):

1. Push each `(fresh_bid, BufInfo)` into `self.bufs`.
2. For each `(fresh_bid, class)` in `subst.outputs`: `canon_id.insert(fresh_bid, class)`. This is the only class-side write; `members(class)` will now pick the fresh BufId up automatically via the new node's producer UseInfo (see step 5).
3. Rewire `subst.node`'s input `BufId`s to a canonical representative of the mapped input class (the smallest existing `BufId` in that class — deterministic and stable).
4. Push `subst.node` into `self.nodes` at a fresh `NodeId t`; insert `access_relations.insert(t, subst.access)`.
5. Register `t` as a producer in `producers[class]` for each output class (pushing `UseInfo { node: t, pos }`), and as a consumer in `consumers[class]` for each input class.

Fresh-BufId allocation. `apply()` needs to return a node containing concrete BufIds it hasn't yet reserved in `gf.bufs`. A single `AtomicUsize` shared across the parallel `apply` calls hands out reservations; `insert_all` pushes into `bufs` in id order, so the ids match.

No dead-node removal, no class merging — extraction (Step 6) prunes.

### Step 5 — Cost model

Runtime cost is in **cycles**; compile-time is in seconds. The objective mixes them via a single `compile_time_discount` factor (cycles per second) — set from the target GPU's clock rate and how much compile time the caller is willing to spend per cycle saved.

**Constraint: no compiler in the loop.** The whole point of the module ILP is to *avoid* compiling everything just to price alternatives. So the cost model must be a pure function of `(ir::Module, access relation)` — no ptxas probe, no register count, no compiled shared-memory footprint. We keep only quantities derivable by walking the HIR + access relation:

- DRAM byte traffic (from `AccessRelation.reads/writes` + `index_bounds`)
- Field-weighted op count (from HIR walker, using `bin_weight`/`scalar_op_weight` in `passes::fusion:374-398`)
- Output element count (product of output extents)
- Kernel launch grid/block dimensions declared by the module

`cost.rs`:

```rust
pub type Cycles = f64;
pub type Secs   = f64;

#[derive(Clone, Debug)]
pub struct GraphNodeCost {
    pub grid_dim: u32,
    pub block_dim: u32,
    pub bytes_dram_per_launch: u64,     // producer write + consumer read summed
    pub weighted_flops_per_output: f64, // uses fusion::bin_weight
    pub k_reads_per_output: f64,        // # DRAM loads per output element (avg)

    pub raw_cycles:    Cycles,          // max(bw, flop, latency)
    pub launch_cycles: Cycles,          // constant, default 2000
}

pub struct ModuleCost {
    pub op_count: u64,
    pub compile_secs: Secs,             // seconds (nvcc); NOT convertible to cycles here
}

pub fn estimate_node(
    node: &GraphNode,
    rel: Option<&access::AccessRelation>,
    opts: &FusionOptionsV2,
) -> GraphNodeCost;

pub fn estimate_module(
    nodes: &[&GraphNode],
    opts: &FusionOptionsV2,
) -> ModuleCost;
```

`FusionOptionsV2` carries the machine model in cycle-native units:

```rust
pub struct FusionOptionsV2 {
    // machine (all derivable from device query without invoking nvcc)
    pub parallelism: u32,          // sms * max_threads_per_sm, treated as G
    pub bytes_per_cycle: f64,      // aggregate DRAM bw / clock
    pub flops_per_cycle: f64,      // aggregate ALU throughput / clock (field-weighted)
    pub mem_latency_cycles: f64,   // avg DRAM access latency
    pub launch_cycles: Cycles,     // default 2000

    // pattern preconditions
    pub max_body_ops: u64,            // combined-HIR-op ceiling for PairFuse / WidthFuse
    pub max_per_thread_loads: u64,    // fold-into-reduce prune (k · R threshold)
    pub max_flop_blowup: f64,         // Case-B rematerialization ratio ceiling
    pub width_op_ratio: f64,          // min(ops_a, ops_b) / max ≥ this for WidthFuse
    pub width_pairs_per_round: usize, // cap on width-fuse matches emitted per round

    // solver knobs
    pub sat_rounds: usize,
    pub max_ilp_size: usize,
    pub solver_time_budget_secs: Secs,
    pub max_modules: u32,             // O in the ILP
    pub compile_time_discount: f64,   // cycles per second of compile time
}
```

`parallelism` (`G`) collapses "SMs × threads/SM × occupancy" into one number so we don't have to reason about occupancy. Set it to the device's nominal max threads-in-flight — this makes `raw_cycles` a *lower* bound on the memory-latency term, which is the direction we want when comparing fusion candidates (over-fused kernels don't get penalized for unknown occupancy loss; if they should be, they'll lose on bandwidth/flops).

Formulas (all outputs in cycles):

- `N = output element count`
- `k = k_reads_per_output` (DRAM loads per output; from `AccessRelation.reads`)
- `t = weighted_flops_per_output`
- `waves = ceildiv(N, parallelism)`
- `bw_cycles   = N * k * bytes_per_elem / bytes_per_cycle`
- `flop_cycles = N * t / flops_per_cycle`
- `lat_cycles  = waves * mem_latency_cycles * k`
- `raw_cycles  = max(bw_cycles, flop_cycles, lat_cycles)`
- `launch_cycles = opts.launch_cycles` (default 2000)

No shared-memory, register, or occupancy terms. Fusion decisions that would exceed hardware limits (too many registers, shared-mem overflow) are handled by pruning at *enumeration time* — the enumerator uses cheap HIR-side heuristics (live-value count, declared shared-mem allocs) and drops candidates that clearly won't fit. If a surviving candidate happens to be slow due to occupancy loss, we accept the modeling error; the alternative (compile-to-measure) defeats the purpose.

Non-kernel node cycles: `Memcpy = bytes / bytes_per_cycle`; `Const = 0`.

Per-module `compile_secs`: `a + b·ops + c·ops·log2(ops)` — a pure function of op count. `(a,b,c)` fitted from CI timing data (initial guess `a=0.5, b=1e-4, c=1e-5`, TODO calibration bench). Compile cost is the *only* quantity in the model that has to stay in seconds; it enters the objective through `compile_time_discount`.

`GraphFuser::cost(&self, t: NodeId)` memoizes `GraphNodeCost`; module costs are memoized per module hash.

### Step 6 — Extraction (ILP via or-tools CP-SAT)

`extract.rs`, only compiled with `--features planner`. Uses `google_or_tools` crate (already in workspace per `reference_ortools_install.md`).

The inserted hypergraph is a DAG by construction — every `SubGraphSubst` names its input classes (existing) and creates fresh output BufIds — so no acyclicity constraint is needed in the ILP.

Variables:
- `C[c] ∈ {0,1}` for each `BufClassId c`
- `b[j] ∈ {0,1}` for each `BufId j`
- `N[t] ∈ {0,1}` for each `NodeId t`
- `M[l] ∈ {0,1}` for each *candidate* module `l` (see step 6.5)

Constraints (corrected from the header):

```
(1) C[c] = 1                      for c ∈ inputs ∪ outputs
(2) C[c] = sum_{j ∈ class_members(c)} b[j]                (SSA canonical pick)
(3) sum_{j ∈ results(t)} b[j] = N[t] * |results(t)|       (node's outputs are all-or-nothing)
(4) sum_{c ∈ operands(t)} C[c] >= N[t] * |operands(t)|    (operands materialized)   ← was b_j; corrected
(5) sum_{t ∈ producers_of(c)} b[j∈results(t)] = C[c]      (a chosen class needs exactly one producer node)
(6) sum_l M[l] <= O
(7) N[t] <= sum_{l ∋ t} M[l]                              (chosen node has a module)  ← missing in header
(8) M[l] * |nodes_of(l)| >= sum_{t ∈ nodes_of(l)} N[t]    (module compiled if any node uses it)
(9) N[t] = 0 for t with infeasible resource footprint     (pruned before solve)
```

Objective (cycles):
```
min   sum_t N[t] * (cost.raw_cycles[t] + cost.launch_cycles[t])
    + opts.compile_time_discount * sum_l M[l] * cost.compile_secs[l]
```

`opts.compile_time_discount` is in **cycles per second** — it says "how many runtime cycles are we willing to pay to save one second of compile time." Default `0.0` (ignore compile time); typical: for a graph that runs once, set to the target clock rate (e.g. `1.5e9`) so each launched cycle counts once against each second of compile.

Solver setup:
- Set `parameters.num_search_workers = 1` and fixed random seed → deterministic solutions.
- Warm-start incumbent = the greedy result from running the existing `passes::fusion::select_candidates` on the enumerated subset.
- Time budget `opts.solver_time_budget_secs` (default 5s). If solver returns without proof of optimality, take best feasible. If no feasible solution, fall back to the greedy incumbent (which is always feasible by construction).

**Sub-problem partitioning.** If `|N| > opts.max_ilp_size` (default 2000), partition the graph by weakly-connected components in the *induced fusion candidate graph* (nodes connected by a shared candidate), and solve each independently. Global inputs/outputs anchor cross-component boundaries.

#### Step 6.5 — Module candidates

A "module" is a set of nodes that produce a single compiled kernel. Two nodes share a module iff their kernel-`Module`s are structurally identical modulo parameters (existing `passes::fusion::renumber_module` gives the canonical form; SHA-256 of the canonical form is the module key).

Before extraction, group all `Kernel` nodes by module key → each group is a candidate `M[l]`. Non-kernel nodes get a dummy module with `compile_secs = 0`. Constraint (7)/(8) then handle "one compile amortized over N launches" correctly.

### Step 7 — Applying the extraction

`apply.rs`:

```rust
pub fn apply(
    g: &mut GraphBuilder,
    gf: GraphFuser,
    sol: ExtractionSolution,
) -> Result<(), CompileError>;
```

1. Build a new `GraphBuilder` by walking chosen `N[t]` in topological order (the induced sub-hypergraph is a DAG since the whole hypergraph is; a Kahn's-algorithm sweep over chosen nodes suffices).
2. For each chosen node, allocate `BufId`s in the target for its result classes (one per class, since `C[c]=1 ⇒ exactly one bufid`). Look up operand classes and reuse the already-allocated target `BufId` for the class rep.
3. Copy over `GraphNode` (with input/output `BufId`s rewritten), preserving `FusionHistory` provenance.
4. Register input/output classes as builder inputs/outputs.
5. Replace `*g = new_g` and run existing `dedup_modules(g)` + `dce(g)`.

`FusionHistory` on each chosen node records the merged provenance (list of source `NodeId`s), so downstream `dump.rs` can visualize.

### Step 8 — Integration & tests

1. `FusionOptionsV2` defined in step 5 above lives in `passes/fusion_v2/mod.rs`.
2. Wire `fuse_graph_v2` behind a boolean in `runner.rs` (default off). Existing tests keep the v1 path.
3. Test plan (`tests.rs`):
   - **T1** GraphFuser initialization: a small hand-built `GraphBuilder` → check hyper-edges match.
   - **T2** Insert-and-canonicalize: two `SubGraphSubst`s equate the same class → union-find has one rep.
   - **T3** DAG-invariant check: assert that after any sequence of `insert_all` calls a topological sort of `nodes` succeeds (cheap runtime check gated on `debug_assertions`).
   - **T4** Cost model regression: golden `GraphNodeCost` snapshots for one NTT, one Poseidon2, one fused chain. Update-golden on intentional change.
   - **T5** Extraction optimality on a hand-optimum toy graph (5 nodes, 3 candidate fusions, known best cost).
   - **T6** Solver-timeout fallback: force `time_budget = 0` → assert we get a feasible greedy solution.
   - **T7** End-to-end parity vs v1 on the small benches (Poseidon2 4-round, NTT n=16). Expect v2 ≤ v1 cost.
4. Bench under `benches/fusion_v2_vs_v1.rs` measuring compile time + estimated runtime + measured runtime.

### Open TODOs (deliberately deferred)

- Compile-time model calibration bench (fit `(a, b, c)` in `a + b·ops + c·ops·log2(ops)`).
- Enumeration-time hardware feasibility pruning: HIR-side heuristics for register / shared-memory blowup; keeps the "no compiler in the loop" invariant.
- Width-fuse pair selection: MVP takes top-K by rough affinity per round. A smarter approach uses class-graph clustering (kernels sharing input classes and shape are grouped, then O(|group|²) enumeration per cluster).
- Cross-stream / concurrent-kernel scheduling — out of scope; current model assumes serial launches.
