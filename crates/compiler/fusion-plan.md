# Kernel Fusion Plan

Graph-level kernel fusion pass over `GraphBuilder` (`src/graph_ir.rs`), operating on the HIR
modules held by `GraphNode::Kernel` nodes. Motivating workload:
`crates/cuda-backend/src/logup_zerocheck/fractional_ir.rs`, whose prologue/epilogue is a chain of
micro-kernels (eq-hypercube stages, scalar epilogue chains, constant-index extracts) that today
each cost a full launch + global-memory round trip.

The pass is general: any kernel whose access relations are quasi-affine (`Quast`) is analyzable,
not just elementwise or constant-index kernels.

---

## 1. Positioning: why HIR, not KernelIR

Fusion runs as a graph pass **on HIR modules**, before any per-module JIT compilation:

```rust
pub fn fuse_graph(g: &mut GraphBuilder, opts: &FusionOptions)
    -> Result<FusionReport, CompileError>
```

invoked at the top of `GraphCompiler::compile`, behind a `CompileOptions` flag.

Rationale for HIR over KernelIR:

- The only place where one HIR kernel expands into multiple CUDA kernels is
  `rewrite_parallel_reduce` (`passes/parallel_reduce_rewrite.rs`), and that rewrite is itself
  **HIR → HIR** (`Module → Module`), running inside `compile_and_load` (`lib.rs:100`) *before*
  canonicalize/lower. KernelIR never splits kernels. So the "one node may become several
  kernels" hazard is fully decidable at HIR level by consulting the same predicate the rewrite
  uses (`should_tree_lower`).
- All the rewrite machinery lives at HIR: `subst` (canonicalize.rs), hash-consing/CSE in
  `IRBuilder`, `NodeEmitter` for emitting Quasts back as HIR, `canonicalize` as a structural
  validator, and `GraphBuilder::dedup_module` for module-hash based JIT caching. Fusing at KIR
  would require splitting `compile_and_load` apart and breaks module-hash cache keying.
- Pre-`layout_infer` KernelIR is essentially canonicalized HIR anyway; there is no extra
  information available at that point that the cost model needs.

KernelIR is reserved as a **v2 cost oracle**: speculatively lower a fusion candidate to KIR and
read real SSA op counts / register-pressure proxies instead of the HIR weight table.

Fusion model: **substitution / recompute**. The consumer is always the skeleton — its outer
compute structure, scatter map, `par`/`threads`, and pack shape are untouched. The producer
contributes only scalar expressions grafted at the consumer's read sites. This preserves the
graph invariant (one `ir::Module` per `Kernel` node contains exactly one top-level compute) by
construction.

---

## 2. Definitions

For each kernel node, canonicalize gives a `CanonKernel` with outer var `o` over
`0 <= o < outer_bound`, optional inner tile computes, `inline_lets`, results (scalar or `Pack`),
and optional `scatter_store`.

**Write map** `w` per output, linearized into the output buffer:

| shape                     | write map                                   |
|---------------------------|---------------------------------------------|
| simple scalar             | `w(o) = o`                                  |
| pack-k scalar             | `w(o, l) = o*k + l`,  `0 <= l < k`          |
| general (tile `M`, pack k)| `w(o, j, l) = sigma(o*M*k + j*k + l)`       |

where `sigma` is the `ScatterStore` expression if present (identity otherwise).

**Read map** per `Index` site `s` in the body: `r_s : D x U_s -> Z`, a `Quast` over the outer
var plus all binder vars in scope at the site (inner compute var, reduce vars, tile iteration
vars), linearized over the input's shape. `U_s` is the box of those binders' bounds.

**Kernel classes**:

- **Simple** — no inner compute, no `inner_lets`, no `scatter_store`, and **no `Reduce`
  anywhere in the body**. Reduce is excluded because `rewrite_parallel_reduce` may lower a
  reduce into a multi-stage compute/reduce combo emitting multiple CUDA kernels; a "simple"
  producer must be guaranteed to stay a single scalar expression.

  **Perfect-nest normalization** (stage 0e): reduce lowering (and hand-written DSL) produces
  many small perfect nests, `compute [2] |i| { compute [8] |j| { e } }`. Canonicalize must
  normalize these to one flat compute, `compute [16] |t| { e[i := t/8, j := t%8] }`, so they
  classify as Simple and trigger case A — otherwise every such micro-stage lands in General
  as a spurious "tile producer". See stage 0e for conditions.
- **General** — anything else that is still a single kernel after `rewrite_parallel_reduce`
  (i.e., contains inner computes / scatter / non-tree-lowered reduces).
- **MultiKernel** — contains a reduce for which `should_tree_lower(k, outer_par)` returns true
  (exported from `parallel_reduce_rewrite`). Never a producer. Valid *consumer* for case A
  only: grafting into the reduce body preserves the `compute [M] |t| reduce [K]` shape, so the
  downstream rewrite composes unchanged, and total launch count still drops
  (P + C(2 stages) = 3 → C'(2 stages) = 2).

---

## 3. Graph-level preconditions (all cases)

For producer node `P` writing buffer `B`, consumer node `C` reading `B`:

1. **Unique writer**: `P` is the only writer of `B` (`classify_buf_uses`, graph_ir.rs:995), and
   every read site of `B` in `C` extracts to a `Quast`. This is the same quasi-affinity
   requirement `lower_to_kir` already imposes; note DSL out-of-bounds clamping
   (`Select`-guarded indices) can make specific sites non-affine and veto fusion at those sites.
2. **WAR safety**: no writer of any input of `P` sits strictly between `P` and `C` in topo
   order (insertion order is a valid topo order). Otherwise recomputing `P`'s expression at
   `C`'s position would read clobbered inputs.
3. **No output aliasing**: `writes(C) ∩ reads(P) = ∅`.
4. **Elimination eligibility** (to delete `P` and `B` afterwards): `C` is `B`'s only reader and
   `B` is internal (allocated by `insert_subgraph_impl`, not a user/planner buffer). Fusion
   while *retaining* `P` is a v2 extension (see restrictions table).

---

## 4. Fusion cases

### Case A — prologue fusion: Simple producer → any consumer (always legal)

Producer `P` is simple: its module is `compute [N_P] |o_P| { E_P(o_P, inputs) }` with scalar (or
pack) result `E_P`.

For each read site `B[r_s]` in `C`:

```text
scalar producer:
    sigma_A = { o_P <- r_s }                 # substitution, r_s emitted via NodeEmitter
    graft:   B[r_s]  ==>  alpha(E_P)[o_P := emit(r_s)]

pack-k producer:
    c = simplify(r_s mod k, bounds)
    if c is Const(c0):
        graft member c0 of E_P with  o_P <- r_s div k
    else:
        reject in v1                          # v2: Select chain over members
```

`alpha(·)` is the α-renaming cross-copy (section 7). Consumers may contain reduces — including
tree-lowering ones — because the graft happens inside the reduce body and leaves the
`compute/reduce` shape (and the `should_tree_lower` verdict) intact.

### Case B — epilogue fusion: General producer → Simple consumer, invertible write map

Producer `P` is general (tile size `M`, pack `k`, write map `w`); consumer `C` is simple with
read sites `B[f(o_C)]`.

**Invertibility of `w`:**

```text
no scatter:
    w^-1(p) = ( p div (M*k),          # o_P
                (p div k) mod M,      # j_P (tile index)
                p mod k )             # l_P (pack member)

scatter sigma present:
    try L = sigma.expr.to_linear_layout(bounds)     # quast.rs:543, pow-2 domains
    if L exists and its GF(2) bit-matrix is invertible:
        w^-1 = reconstitute Quast from L^-1
               (bit extract = div/mod by powers of 2; recombine = weighted sum)
    else: reject
```

**Algorithm:**

```text
for each read site B[f(o_C)] in C:
    (o_P, j_P, l_P) = simplify(w^-1(f), C bounds)   # aligned cases collapse to identity
    require l_P is Const                             # pack member must be static
    E = P.result expr at (o_P, j_P), member l_P

    # recursive tile scalarization:
    # E may reference producer InnerLet tiles t via t[g(j_P)]
    worklist in inner-let binding order:
        replace every  t[g]  in E  by  body(t)[iter := g]
        (recurse; terminates because inner-lets form a binding-order DAG)

    graft as in case A (alpha-rename, NodeEmitter for o_P/j_P exprs, replace site)
```

Legality is easy; **profitability rarely holds** for heavy tiles: scalarization discards the
producer's cooperative shared-memory structure, so the cost model must charge the full
per-element recompute flops of the scalarized tile chain.

### Case C — general × general: `TODO(polyhedral)`

Written in polyhedral terms for the record; **not implemented in v1** — the required machinery
does not exist in this codebase yet.

```text
# per consumer outer block b_c, the producer elements needed:
Need(b_c)  = union over read sites s of  r_s( {b_c} x U_s )     # image of a box
                                                                 # under a quasi-affine map
# the producer outer blocks that produce them:
NeedP(b_c) = project_o( w^-1( Need(b_c) ) )                      # preimage + existential
                                                                 # elimination with divisors

legality / profitability tests:
    single-valued:  NeedP is a function h(b_c)
        h = id        -> classic aligned fusion
        h affine      -> shifted / permuted fusion
        |NeedP(b_c)| <= K  -> stencil-like; needs parametric cardinality (Barvinok-style)
    footprint:      sum over b_p in NeedP(b_c) of |Tile(b_p)|  <=  shared-mem budget

rewrite:
    synthesize an InnerLet in C per needed producer tile, re-indexed by h(b_c)
    redirect C reads  B[f]  ->  tile[loc(f)]
    non-singleton NeedP requires parametric loop generation over the set (isl-style codegen)
```

Missing machinery: a Presburger set/relation type; image/preimage/composition of quasi-affine
maps; projection with existential divisors; single-valuedness test; parametric counting;
set-scanning code generation. Options when we get there: bind `isl`, or build a minimal
Presburger kernel specialized to boxes + quasi-affine maps.

### Dispatch

```rust
match (class(P), class(C)) {
    (Simple, _)                          => case A,   // modulo pack-member-const check
    (General, Simple) if invertible(w_P) => case B,
    (General, _) | (MultiKernel, _)      => None,     // v1; (General,General) = TODO(polyhedral)
}
```

---

## 5. Stage 0 — prerequisites (shared infrastructure)

a. **Factor the affine recognizer**: extract `extract_quast` (lower_to_kir.rs:1049) out of the
   lowering context into a standalone
   `hir_to_quast(b: &IRBuilder, id: NodeId, env: &BTreeMap<VarId, Quast>, inline_lets) -> Option<Quast>`
   in `passes/utils.rs`; `lower_to_kir` calls it too.
b. **Node replacement**: `replace_nodes(b: &mut IRBuilder, root: NodeId, map: &HashMap<NodeId, NodeId>) -> NodeId`
   — memoized bottom-up rebuild. Because the IR is hash-consed, syntactically identical `Index`
   nodes are the *same* `NodeId`, so a NodeId-keyed map replaces all occurrences of a read site
   at once.
c. **Buffer provenance**: add `internal: bool` to `BufInfo`, set by `insert_subgraph_impl` for
   auto-allocated intermediates (`{subgraph}.k{ki}.o{oi}`). Used for elimination eligibility
   (precondition 4) and as DCE roots (non-internal = live).
d. **Export `should_tree_lower`** from `parallel_reduce_rewrite` for the MultiKernel classifier.
   Caveat: the classifier must evaluate it against the *pre-canonicalize* nest shape, since
   `rewrite_parallel_reduce` runs before canonicalize in `compile_and_load` and never sees the
   flattened form.
e. **Flatten perfect nests in canonicalize.** Today `flatten_nests` (canonicalize.rs:612) only
   collapses nests *below* the first inner level; the depth-2 nest itself is kept as
   `CanonKernel.inner` (canonicalize.rs:440). Extend `emit_kernel` to absorb the first level
   too, producing `CanonKernel { outer_bound: N*M, inner: None }`, when all of:

   - `inner_par.is_none()` — an inner `#[par(...)]` pins the thread-level element mapping;
     flattening would destroy it (same reason line 412 already errors on deep-flatten + par);
   - `inner_lets.is_empty()` — let-bound tiles are per-outer-iteration shared memory;
     flattening the outer domain would scalarize/recompute them;
   - outer `threads.is_none()` — an explicit `#[grid(threads = ...)]` is keyed to the outer
     domain size.

   Scatters are no obstacle: both outer and inner scatters are already composed into a
   flat-domain `scatter_store` before flattening (canonicalize.rs:362, :417), and a flat
   kernel with `scatter_store` is valid canonical form. Implementation: substitute
   `outer_var := t div M`, `j := t mod M` (via `subst` + `NodeEmitter`, exactly as
   `flatten_nests` does) into results *and* the peeled `inline_lets` values.

   Note this changes launch geometry in the real pipeline for affected kernels (N*M threads
   writing 1 element each, instead of N threads writing M contiguous elements each) — desired:
   better coalescing and occupancy for the reduce-lowered micro-stages.

---

## 6. Stage 1 — access-relation extraction + cost estimates

### Data structures

```rust
struct AccessRelation {
    outer_var: VarId,
    outer_bound: usize,
    reads: Vec<ReadAccess>,
    writes: Vec<WriteAccess>,
    cost: KernelCost,
    class: KernelClass,                  // Simple | General | MultiKernel
    write_inverse: Option<InverseMap>,   // for case B
}

enum InverseMap {
    DivMod { tile: usize, pack: usize },         // no scatter
    Layout(/* inverted LinearLayout, as Quast coordinate exprs */),
}

struct ReadAccess {
    input_pos: usize,                    // which module input
    site: NodeId,                        // the Index node (hash-consed => unique per shape)
    expr: Option<Quast>,                 // None => non-affine site, vetoes fusion through it
    inner: Vec<(VarId, usize)>,          // binders in scope at the site + bounds (U_s)
    site_env: BTreeMap<VarId, NodeId>,   // for re-emission via NodeEmitter
}

struct WriteAccess {
    out_pos: usize,
    len_elems: usize,
    inner_factor: usize,                 // M*k (tile * pack)
}

struct KernelCost {
    flops_per_elem: f64,      // weighted op count of the result expr (equivalent-u32-op units)
    bytes_in_per_elem: f64,   // global-load bytes per output element
    bytes_out_per_elem: f64,  // global-store bytes per output element (elem size * pack k)
    body_ops: usize,          // raw node count, guards code-size blowup
}
```

### Algorithm

```text
extract(g: &GraphBuilder) -> HashMap<node_idx, Arc<AccessRelation>>:
    cache: HashMap<[u8;32], Arc<AccessRelation>>   # keyed by module_hash, survives rounds
    for each GraphNode::Kernel node:
        if cache hit: reuse
        else:
            type_infer + canonicalize  ->  CanonKernel
            classify: Simple / General / MultiKernel (via should_tree_lower on reduces)
            walk results + inline_lets with a binder scope stack
                (Compute and Reduce push (var, bound))
            per Index { tensor: Input(k), index }:
                expr = hir_to_quast(index) then linearize over input shape
            write maps + write_inverse per section 4B
            cost: memoized DFS over body
                BinOp x ScalarType weight table (FpExt mul >> BabyBear mul >> add)
                Index adds elem-size bytes
                Reduce multiplies its subtree cost by the reduce bound
    per round: (writers, readers) = classify_buf_uses(nodes, n_bufs)
```

---

## 7. Stage 2 — candidate enumeration and selection

### Data structures

```rust
struct FusionCandidate {
    src: usize,             // producer node idx
    dst: usize,             // consumer node idx
    buf: BufId,
    score: f64,
    info: FusionInfo,
}

struct FusionInfo {
    rewrites: Vec<SiteRewrite>,
}

struct SiteRewrite {
    index_node: NodeId,                 // read site in dst to replace
    sigma: BTreeMap<VarId, Quast>,      // producer binder -> index expr
    pack_elem: Option<usize>,           // which pack member to graft
    site_env: BTreeMap<VarId, NodeId>,
}
```

### Algorithm

```text
enumerate:
    for each internal buffer B with writers[B] == [src] and readers[B] == [dst]:
        run dispatch (section 4); if Some -> build FusionInfo
        WAR check: binary-search sorted writer lists of src's inputs
                   for a writer index in (src, dst)

score:
    see "Cost model" below; reject if src.body_ops + dst.body_ops > max_body_ops
    keep candidates with score > 0

select (disjoint pairs):
    greedy max-weight matching:
        sort candidates by score desc
        accept candidate iff neither src nor dst already used (FixedBitSet)
    # 1/2-approximation is fine; the fixpoint loop mops up chains across rounds
```

### Cost model

All quantities derive from the stage-1 `AccessRelation`s. Everything is converted to time via
three machine constants: `BW_eff` (effective DRAM bytes/sec), `FLOP_eff` (weighted ops/sec),
and `LAUNCH_COST` (seconds per eliminated launch, ~3–10 µs without CUDA graphs). These live in
`FusionOptions`.

**Producer per-element costs** (stage 1, memoized DFS over the result expression):

```text
flops_per_elem(P)     = weighted op count of P's scalar body
bytes_in_per_elem(P)  = sum over P's read sites of bytes(input elem)
bytes_out_per_elem(P) = bytes(B.elem) * pack_k
```

The flop weight table is indexed by `BinOp x ScalarType`, in equivalent-u32-op units
(illustrative; calibrate once with a microbench):

| op                          | weight |
|-----------------------------|--------|
| u32 add / shift / mask      | 1      |
| div/mod by pow2 const       | 1      |
| Select                      | 2      |
| BabyBear add/sub            | 2      |
| BabyBear mul (Montgomery)   | 8      |
| FpExt add                   | 8      |
| FpExt mul                   | ~50    |

`Reduce` multiplies its subtree's weight by the reduce bound. Weight precision only matters
near the decision boundary, and the boundary is dominated by the memory and launch terms.

**Site volumes.** For each read site `s` of `B` in `C`, with binders
`inner(s) = [(v_1, b_1), ...]` in scope at the site (inner compute var, reduce vars):

```text
vol(s) = C.outer_bound * prod_k(b_k)     # how many times the site executes per launch
```

**Replication.** Fusion is recompute-based: the grafted producer expression is evaluated once
per executed graft site, with no caching across sites.

```text
evals_before = |B|
evals_after  = sum over sites s reading B of vol(s)
R            = evals_after / |B|          # average recomputes per producer element
```

- `evals_after > |B|`: overlapping reads recompute the producer (the example below has R = 2).
- `evals_after < |B|`: C subsamples B — the sole-reader precondition means the untouched part
  of B was dead work, so fusion *saves* producer flops; the delta goes negative on its own.
- A site under `reduce [K]` has `vol = C.outer_bound * K`: fusing into reduce bodies is legal
  (case A admits MultiKernel consumers) but the K-fold recompute usually kills profitability
  unless the producer is near-free (broadcast, cheap affine gather).
- Two sites with the *same* index expression are the same hash-consed `NodeId`, hence one
  site — the model never double-charges reads that CSE will merge anyway.

**Deltas.** Three terms, all in bytes or weighted ops:

```text
mem_saved   = (|B| + evals_after) * bytes(B.elem)
              # B's store (|B| elems) + every load of B in C (evals_after elems)

mem_added   = gamma * (evals_after - |B|) * bytes_in_per_elem(P)
              # each graft re-reads P's inputs; only the replication excess is charged

extra_flops = (evals_after - |B|) * flops_per_elem(P)
```

`gamma ∈ [0, 1]` is a cache discount on the replicated producer loads: grafted sites re-read
the same input buffers through shifted/overlapping windows, so most replicated loads hit L2
rather than DRAM. v1: fixed `gamma = 0.5`. v2: compute the unique footprint of each input as
the image of its read Quasts over the site domains (`Quast::range`) and derive gamma from the
actual overlap ratio.

**Score** (time domain):

```text
score = (mem_saved - mem_added) / BW_eff
      + LAUNCH_COST
      - extra_flops / FLOP_eff
```

Two regimes fall out naturally:

- **Tiny kernels** (fractional_ir scalar epilogue chains, N in the tens/hundreds): every
  byte/flop term is negligible against `LAUNCH_COST`, so fusion is essentially always
  accepted — correct, since launch overhead is the entire cost of those kernels.
- **Large-N kernels**: the bandwidth term dominates; the flop term only vetoes when an
  expensive producer (FpExt muls, reduce subtrees) is replicated across many high-volume
  sites.

### Worked example

```text
let a = compute [N] |i| { x[i] * 2 };              # P: Simple; 1 BabyBear mul,
                                                    #    4 B in, 4 B out per element
let b = compute [N] |i| { a[i] + a[(i+1)%N] };     # C: 2 read sites of a, vol(s) = N each

evals_after = 2N     |B| = N     R = 2

mem_saved   = (N + 2N) * 4      = 12N bytes        # a's store + both load streams of a
mem_added   = 0.5 * (2N-N) * 4  =  2N bytes        # extra x reads; x[i] and x[(i+1)%N]
                                                    # footprints overlap almost entirely,
                                                    # so mostly L2 hits (gamma = 0.5)
extra_flops = (2N-N) * 8        =  8N weighted ops # the "N more muls"

score = 10N bytes / BW_eff + LAUNCH_COST - 8N ops / FLOP_eff   > 0
```

At a machine balance of ~10+ weighted ops per byte, saving 10N bytes buys ≥ 100N ops of
headroom against the 8N spent — a clear win before even counting the launch. The index
arithmetic `(i+1)%N` is *not* charged: it executes either way (as a load address before
fusion, as part of the grafted expression after); address computation is never part of
`flops_per_elem`.

Counter-example the model correctly rejects: the same producer with
`E_P = fpext_mul(x[i], y[i])` (weight ~50, 32 B in per element) consumed at 4 sites under a
`reduce [K]` gives `evals_after = 4NK`, so `extra_flops ≈ 200·NK` and
`mem_added ≈ 16·NK` bytes against a fixed `mem_saved = (1 + 4K)·16N` bytes of B traffic —
for K beyond a handful the replicated FpExt work swamps the savings.

---

## 8. Stage 3 — applying a fusion (module surgery)

```text
apply_fusion(g, cand):
    mb = clone of dst's IRBuilder                     # IRBuilder: Clone

    # cross-copy of producer body — the main correctness trap:
    # src and dst modules come from different builders, so their VarIds collide
    # (split_module preserves VarIds verbatim). Unlike split_module::Extract,
    # the walker here must ALPHA-RENAME EVERY src binder
    # (Compute / Reduce / Let vars -> mb.fresh_var()).
    CrossCopy walker (template: split_module.rs:180 Extract):
        src Input(k)  ->  if dst already binds the same BufId: reuse that input NodeId
                          else: fresh mb.input(); record appended BufIds in first-use order
        src outer/inner vars -> NodeEmitter-emitted sigma exprs per site (typed U32)
        pack_elem selects the Pack member

    for each SiteRewrite: inlined[site.index_node] = grafted expr
    body2 = replace_nodes(mb, dst_body, inlined)
    merged = mb.finish(format!("{dst_name}__f__{src_name}"), body2)

    # validation — error => skip candidate, do not panic:
    type_infer(merged)?
    debug_assert!(split_module(&merged)?.kernels.len() == 1)

    # graph surgery:
    nodes[dst] = Kernel(KernelModuleNode {
        module:  g.dedup_module(Arc::new(merged)),    # exact-hash dedup only; alpha-variant
                                                      # duplicates are collapsed by stage 6
        inputs:  dst_inputs_without_B ++ appended_src_inputs,
        outputs: dst.outputs,                          # unchanged
    })
    # nodes[src] untouched here; stage 4 DCE removes it
```

The single-top-level-compute invariant holds by construction: only scalar expressions are
inlined into the consumer's existing compute; no new top-level bindings are created.

---

## 9. Stage 4 — graph canonicalization (v1: dead-node elimination)

```text
dce(g):
    needed: FixedBitSet over BufIds      # init: all non-internal buffers
    live:   FixedBitSet over nodes
    for n in nodes.rev():                                  # reverse topo order
        (reads, writes) = node_reads_writes(n)             # graph_ir.rs:806
        live[n] = is_blackbox(n) || writes.any(|b| needed[b])
        if live[n]:
            needed |= reads                                # never clear bits:
                                                           # partial writes mean an earlier
                                                           # writer of a needed buf stays live
    retain live nodes; leave bufs table untouched
    # (verify the planner ignores unreferenced bufs; if not, add buf compaction later)
```

`BlackboxKernel` nodes are unconditionally live (opaque effects). `Const`/`Memset`/`Memcpy` are
live iff a written buffer is needed.

---

## 10. Stage 5 — fixpoint driver

```rust
pub struct FusionOptions {
    pub max_iterations: usize,   // default 10
    pub max_body_ops: usize,     // code-size / register-pressure guard
    pub launch_cost: f64,        // seconds per eliminated launch (~3-10 us)
    pub bw_eff: f64,             // effective DRAM bytes/sec
    pub flop_eff: f64,           // weighted ops/sec (equivalent-u32-op units)
    pub gamma: f64,              // cache discount on replicated producer loads, default 0.5
}

pub struct FusionReport {
    pub rounds: usize,
    pub fused: Vec<(String, String)>,    // (src_name, dst_name)
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub deduped: usize,                  // modules collapsed by stage 6
}
```

```text
fuse_graph(g, opts):
    stage4_dce(g)        # DCE FIRST, before any analysis: a dead reader of B would
                         # otherwise fail the single-reader precondition and inflate
                         # evals_after in the cost model; dead producers waste extraction
    for round in 0..opts.max_iterations:
        (writers, readers) = classify_buf_uses(...)
        rels    = stage1(g)              # module_hash-cached across rounds
        matched = stage2(g, rels, writers, readers)
        if matched.is_empty(): break
        for cand in matched: stage3(g, cand)     # skip on validation failure
        stage4_dce(g)                            # clean up fused-away producers + B
    stage6_dedup(g)      # after all fusion runs: alpha-normalize + dedup fused modules
                         # so identical patterns JIT-compile once (section 11)
    return report
```

Termination: every applied fusion removes ≥ 1 node (single-reader precondition guarantees the
producer dies in DCE), so the node count strictly decreases per productive round;
`max_iterations` is a backstop.

---

## 11. Stage 6 — post-fusion module deduplication (α-normalization)

After all fusion rounds, run a deduplication pass over the fused kernel modules so that
structurally identical fused kernels JIT-compile once.

**Why `dedup_module` alone is not enough.** `module_hash` (module_hash.rs) canonicalizes
NodeIds (post-order DAG renumbering) but hashes **raw VarIds**: `Node::Var(v)` feeds `v.0`
directly, and so do `Quast::Sym` in scatters, scatter params/bounds, and `ParSpec`. Fusion
surgery α-renames producer binders with `fresh_var()` from the *cloned consumer builder*, so
two fusions of the same logical (P, C) pattern — e.g. the same pair instantiated at different
graph locations, or built by separate macro expansions with different var counters — produce
α-equivalent modules with *different* hashes. Both the in-memory `module_dedup` map
(graph_ir.rs:441) and the on-disk JIT cache (keyed by `module_hash_hex`, kernel_cache.rs:102)
would miss, and each copy would compile separately.

**Algorithm:**

```text
stage6_dedup(g):
    for each GraphNode::Kernel n:
        m2 = renumber(module(n)):
            rebuild into a fresh IRBuilder:
            - walk body in the same canonical post-order module_hash uses
            - intern nodes in that order (canonical NodeIds for free via hash-consing)
            - renumber every VarId in first-binder-occurrence order
              (Compute/Reduce/Let binders, scatter params, ParSpec vars)
        nodes[n].module = g.dedup_module(Arc::new(m2))
    report.deduped = #(distinct Arcs before) - #(distinct Arcs after)
```

After renumbering, α-equivalent modules are byte-identical to the hasher, `dedup_module`
collapses them to one `Arc<Module>`, and `compile_and_load` compiles each distinct kernel
once (this also feeds the parallel-compilation batcher one copy instead of N). Side benefit:
the sweep runs over *all* kernel nodes, so pre-existing α-variant duplicates in the input
graph (independent macro expansions of the same kernel) get deduped too.

**Caveat — module names.** The hash covers `module.name`. Merged names
(`{dst_name}__f__{src_name}`) are derived deterministically from constituent names, which are
themselves deduped, so equal patterns normally get equal names. If multi-round chains ever
produce name-divergent but structurally identical modules, either canonicalize merged names or
split the dedup key from the display name — deferred until observed.

## 12. v1 restrictions and how they lift

| restriction (v1)                          | how it lifts later                                        |
|-------------------------------------------|-----------------------------------------------------------|
| producer single-output                    | per-output liveness; fuse one output, retain node          |
| producer identity write / no scatter      | `to_linear_layout` inversion (case B scatter path)         |
| producer no inner_lets / par / threads    | tile-aware fusion (case C machinery)                       |
| single reader of B                        | retained-producer fusion (fuse into one reader, keep P)    |
| pack member must be constant              | Select chain over pack members                             |
| no Reduce anywhere in producers           | never for tree-lowering reduces; possibly sequential reduces with recompute cost gating |
| general × general                         | TODO(polyhedral), section 4C                               |

---

## 13. Testing plan (test-first)

CPU-only unit tests (no GPU), in `passes/fusion.rs` + `tests/`:

1. **Extraction**: build small modules via the DSL, assert extracted read/write Quasts equal
   expected expressions (`a[i]`, `a[2*i]`, `a[0]` broadcast, `a[i/2]`, `a[i % c]`, pack writes).
2. **Perfect-nest normalization** (stage 0e): `compute [2] |i| { compute [8] |j| e }`
   canonicalizes to `outer_bound = 16, inner = None` with `t/8`, `t%8` index recovery and
   classifies Simple; with an inner `#[par]`, a let-bound tile, or an outer
   `#[grid(threads=…)]` the nest is preserved; scattered variants keep a correct flat
   `scatter_store`; gpu_macro numeric equality flat-vs-nested.
3. **`can_fuse` verdicts**: elementwise chain fuses; `a[2i]` gather fuses (case A);
   pack-constant member fuses; pack non-constant rejected; reduce-containing producer rejected;
   WAR-blocked pair rejected; multi-consumer buffer rejected; case B div/mod inversion on an
   aligned tile writer.
4. **Merged module structure**: `dump_hir` snapshot of fused module; `split_module` reports one
   kernel; type_infer passes; α-renaming produces no VarId collisions.
5. **DCE**: dead producer + internal buffer removed; blackbox and non-internal-output writers
   retained.
6. **Dedup (stage 6)**: fuse the same (P, C) pattern at two graph locations built from
   independently constructed builders (different VarId counters); assert `module_hash`
   equality and pointer-equal `Arc<Module>`s after `stage6_dedup`; assert renumbering is
   idempotent and numerics unchanged.

GPU tests (`tests/gpu_macro.rs`, per test-first workflow): fused-vs-unfused numeric equality on
representative chains; assert kernel-launch-count reduction on the fractional_ir DSL graph.

---

## 14. Tracked follow-ups (not in v1)

- **Hoist `rewrite_parallel_reduce` to graph-insertion time**, before `split_module`. Upgrades
  the invariant to *1 graph node = 1 CUDA kernel*, makes reduce stages individually fusable and
  plannable. The perfect-nest normalization (stage 0e) is what makes the resulting small stage
  kernels classify Simple and actually fuse. Moves inter-stage buffers from
  `plan_global_scratch` to planner buffers — per AGENTS.md this requires updating
  `crates/stark-backend/src/memory_metering.rs` and `docs/cuda-backend/gkr-prover.md`. Own PR.
- **v2 cost oracle**: speculatively lower fusion candidates to KernelIR and read real SSA op
  counts / register-pressure proxies instead of the HIR weight table.
- **Polyhedral machinery** for case C (bind isl, or minimal Presburger kernel over boxes +
  quasi-affine maps).

## 15. Expected payoff on fractional_ir

Immediate wins once ports land in `fractional_ir_dsl.rs`: eq_hypercube stage chains
(`i % step` / `i / step` accesses — case A gathers), scalar epilogue chains
(`reconstruct_s_evals → update_running_scalars`, `reduce_to_single_evaluation → claim_combine`),
and `extract_root` / `extract_claim` constant-index gathers. Most prologue/epilogue nodes are
still `BlackboxKernel` (opaque, unfusable) — the larger wins are gated on continuing the
DSL-port migration.
