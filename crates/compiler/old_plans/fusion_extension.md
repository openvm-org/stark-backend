# Fusion v2 extension plan: toward more general fusion

Status: proposal (2026-08-11). Companion to `detailed-fusion-plan-v2.md` (the
base design) and `fusion-v2-progress.md` (what has landed through M13 /
session 18). Each section below quotes the code that currently blocks a
generalization, sketches the proposed change, and explains why it is worth
doing. Sections are ordered by expected unblocking value.

Context for the "why" columns: on the GKR fractional-sumcheck bench
(`FRAC_V2_BENCH_FUSION_V2=1`), saturation currently *generates* thousands of
candidates that are then wasted at the costing stage (3,123 sentinel-excluded
at n=2^16; 1,656 at LOG_N=24), and the committed graph still carries a ~3.2×
kernel-work amplification versus the theoretical fused minimum. The items
below attack both losses: §1 recovers already-synthesized candidates, §§2–9
let synthesis produce candidates it currently rejects.

Examples use the test-suite shorthand (`tests.rs::general_pc_tests`):
`y = compute [n] |k| expr` is a module whose body is a single `Compute`
with bound `n`, binder `k`, and scalar body `expr`; `bindings {n: 8}`
are the node's `param_bindings`; `pack(a, b)` is a rank-extending
`Node::Pack`.

---

## 1. Canonicalization of composed keep candidates (plan §9 gap)

**The biggest single win: it wastes candidates we already pay to synthesize.**

### Current state

The keep variant emits a `Tuple` body so the seam is materialized alongside
the consumer output (`fusions/producer_consumer.rs:618-635`):

```rust
let compute_body = match variant {
    FusionVariant::Drop => cloned_body,
    FusionVariant::Keep => {
        let mut producer_vars: HashMap<VarId, HirNodeId> = HashMap::new();
        producer_vars.insert(p_shape.outer_var, k_var_node);
        let seam_body = clone_expr_with_params(
            &p_module, p_shape.body_root, &mut fb,
            &producer_subst, &producer_vars, &p_param_map,
            |_, _, _| Ok(None),
        )
        .map_err(|e| SynthesisFailure::CloneError(format!("{e:?}")))?;
        fb.intern(Node::Tuple(vec![cloned_body, seam_body]))
    }
};
```

When such a keep node is *itself* fused again in a later saturation round
(a composed candidate), the resulting body can contain a `Tuple` whose
elements embed inner-let computes. `canonicalize` accepts the module, but
`is_canonicalized` (`passes/canonicalize.rs:779-827`) requires result/tile
bodies to be **pure scalar expressions**:

```rust
for &n in scalars {
    if !is_scalar_form(b, program, ck, n, &mut visited) {
        return false;
    }
}
```

and `lower_to_kir` asserts it (`passes/lower_to_kir.rs:70`):

```rust
debug_assert!(is_canonicalized(program));
```

The KIR cost estimator catches the resulting panic and prices the node at
the `GraphNodeCost::FAILED` sentinel (`fusion_v2/cost/mod.rs:90`), so the
extractor can never select it. Net effect on the GKR bench: 3,123 candidates
at n=2^16 (1,656 at LOG_N=24) are synthesized, hashed, deduped, and then
discarded at costing.

### Example

The recorded failures are composed `epilogue_keep` candidates (progress
doc, session 16). To be precise about what is and is not handled, both
of these shapes canonicalize fine today:

```text
# (a) root-level let-bound compute: peel_body_lets hoists it into an
#     InnerLet shared-memory tile (canonicalize.rs:605-661):
compute [n] |k| ( let t = compute [8] |j| f(x[j]) in g(t[k]) )

# (b) two-level nest at the body root: the `inner` path + flatten_nests
#     (canonicalize.rs:399-467, :705-772), speculatively absorbed into
#     one flat compute (:471-539):
compute [n] |k| compute [8] |j| f(x[k*8 + j])
```

Round-1 small-kernel fused modules have shape (a) — a `Let`-bound tile
chain at the body root (`small_kernel.rs:537-566`) plus a block hint,
which keeps them out of producer-consumer (`producer_consumer_covers`,
`epilogue.rs:138-141`), so epilogue is their composition path. Epilogue
clones the producer body *wholesale* — "inputs may be used in any form"
(`epilogue.rs:102-105`) — so such a node qualifies as a round-2
producer.

The round-2 keep body wraps `Tuple(cloned_c, cloned_p)` around
wholesale clones of the producer's body root — **Let chain included**
(`epilogue.rs:410`). The `Let`s now sit *inside* the tuple elements,
one level below the body root:

```text
# c: z = compute [8] |k| y[k] * inv[k], seam y read at identity
(z, y) = compute [8] |k|
    Tuple( ( let t  = compute [8] |j| f(x[j]) in g(t[k]) ) * inv[k],  # z: Let mid-expression
           ( let t' = compute [8] |j| f(x[j]) in g(t'[k]) ) )         # y: Let-rooted element
```

**Exactly what marks this non-canonical:**

1. `peel_body_lets` only peels the body *root* — its `while let
   Node::Let` loop runs on `id` itself (`canonicalize.rs:605`). The
   root is now `Tuple`, so the loop never fires and no tile is hoisted.
2. The result splitter accepts it anyway: `classify_result`'s catch-all
   (`:592`) classifies a `Let`-rooted tuple element as
   `ResultExpr::Scalar`. Only a *bare* `Compute` element errors
   gracefully (`:587-591`, "compute nested inside a tuple result is not
   supported"). So `canonicalize` returns `Ok`.
3. `is_canonicalized` walks each result with `is_scalar_form`, which
   recurses into a `Let`'s *value* as a scalar position (`:883-886`);
   the value is a `Compute`, and `Compute | Tuple | Proj | Pack` in
   scalar positions are categorically false (`:887`). The
   `lower_to_kir.rs:70` `debug_assert!(is_canonicalized(program))`
   fires during *costing*.

The gap is the disagreement between `classify_result` (accepts
`Let{Compute}` elements) and `is_scalar_form` (rejects them). Every
composed candidate that buries a producer's tile chain under the keep
`Tuple` — or inlines it mid-expression on the drop path, as in element
`z` above — lands in it: synthesized, deduped, inserted into the alt
graph, then priced `FAILED`, invisible to the extractor.

### Proposed change

Teach `canonicalize` to fully normalize composed keep bodies instead of
passing them through in a shape `is_canonicalized` rejects. Concretely:

1. Generalize the `Let{Compute}` peel from the body root to embedded
   positions: after the existing root peel (`emit_kernel`,
   `canonicalize.rs:396`), hoist `Let`-bound computes out of tuple
   elements and out of scalar positions into the same `inner_lets`,
   replacing each `Let` node with its (recursively processed) body.

   ```rust
   // Sketch — emit_kernel, after the root peel at canonicalize.rs:396:
   let body = self.peel_body_lets(body, &mut inline_lets, &mut inner_lets)?;
   let body = match self.b.node(body).clone() {
       Node::Tuple(elems) => {
           let elems = elems
               .iter()
               .map(|&e| self.hoist_embedded_lets(e, &mut inline_lets, &mut inner_lets))
               .collect::<Result<Vec<_>, _>>()?;
           self.b.intern(Node::Tuple(elems))
       }
       _ => self.hoist_embedded_lets(body, &mut inline_lets, &mut inner_lets)?,
   };
   // hoist_embedded_lets: walk scalar positions; on Let{value: Compute},
   // push the (flatten_nests'd, scalar-peeled) tile into inner_lets —
   // the same steps peel_body_lets runs at :641-660 — and replace the
   // Let node with its recursively-processed body.
   ```

   Soundness: an `InnerLet` tile is materialized per outer iteration, so
   moving a `Let{Compute}` from mid-expression to the kernel's tile list
   preserves per-iteration semantics; binders are globally fresh
   (`clone_expr_with_params` allocates fresh vars), so no capture is
   possible.

2. Make the fix *closed under composition*: a hoisted tile body may itself
   contain an embedded `Let{Compute}` — run the hoist to fixpoint (bounded
   by body size, so termination is trivial). Duplicate tiles from the
   drop/keep double-clone (`t`/`t'` in the example) can be CSE'd by value
   NodeId, since hash-consing makes identical tile computes share a node.

3. Restore the invariant `canonicalize(m).is_ok() ⇒ is_canonicalized(p)`:
   after the hoist, `classify_result` and `is_scalar_form` agree by
   construction — any residual non-scalar shape should become a graceful
   `CompileError::Canonicalize`, never an `Ok` that later trips the
   `lower_to_kir.rs:70` assert.

4. Exit gate: a unit test that composes a tile-chain producer under
   `synthesize_epilogue` keep and asserts `is_canonicalized(&lowered)` —
   i.e. `cost_of` returns a real cost, not `FAILED`. Plus a driver-level
   test asserting the sentinel-exclusion count drops to zero on a small
   halving-chain graph.

### Why

- No new synthesis logic is needed — the candidates already exist; they are
  purely lost between `canonicalize` and `lower_to_kir`.
- Keep-composed candidates are exactly the ones that let the extractor trade
  seam materialization against artifact reuse, which is the lever for the
  remaining kernel-work amplification on the GKR graph.
- Risk is contained: the rewrite only fires on bodies `is_canonicalized`
  currently rejects, so already-working candidates lower byte-identically.

---

## 2. Multi-output producers / multi-seam fusion

### Current state

Producer-consumer hard-gates on a single producer output
(`fusions/producer_consumer.rs:345-347`):

```rust
if p_alt.outputs.len() != 1 {
    return Err(SynthesisFailure::ProducerNotSingleOutput);
}
```

and fanout does the same (`fusions/fanout.rs:103-108`):

```rust
// For M7 we handle single-output producers only. Multi-output
// producers (e.g. from a keep-variant fused kernel) are the
// multi-seam case, deferred.
if outputs.len() != 1 {
    continue;
}
```

This means every keep-variant node (which by construction has ≥2 outputs:
consumer output + materialized seam) is a dead end as a producer — it can be
selected, but nothing can fuse *through* it. Combined with §1 this caps
chain depth at exactly two layers.

### Example

Round 1 keep-fuses `f` into `g` because the seam `y` is also read
elsewhere:

```text
# Round-1 keep node — 2 outputs:
p:  (z, y) = compute [n] |k| ( g(f(x[k])), f(x[k]) )

# Next consumer, reading only z:
c:  w = compute [n] |k| h(z[k])
```

Desired round-2 drop fusion through output 0 (`z`), with `y` still
materialized:

```text
fused:  (w, y) = compute [n] |k| ( h(g(f(x[k]))), f(x[k]) )
```

**Why it is rejected today:** `p_alt.outputs.len() == 2` →
`SynthesisFailure::ProducerNotSingleOutput`
(`producer_consumer.rs:345`). Fanout skips `p` at enumeration for the
same reason (`fanout.rs:106`). Nothing is wrong with the fused module —
the gate is purely structural on the producer's output arity.

### Proposed change

Generalize the seam from "the producer's output" to "one designated output
of the producer", threading an `out_idx` through synthesis:

1. `synthesize_producer_consumer` takes `seam_out_idx: usize`; the enumerator
   in `fusions/mod.rs` iterates over `p_alt.outputs.iter().enumerate()`
   instead of assuming index 0.
2. The producer body for a multi-output kernel is a `Tuple`; the seam body
   root becomes `tuple.elems[seam_out_idx]` (a `Proj` in HIR terms). The
   inline hook substitutes that element; the *other* elements remain
   materialized, so the fused node's outputs are
   `p_alt.outputs \ {seam} ++ c_alt.outputs` for drop, and all of them for
   keep.

   ```rust
   // Sketch — seam body selection:
   let p_body_root = match tuple_elems(&p_module, p_shape.body_root) {
       Some(elems) => elems[seam_out_idx],
       None => {
           debug_assert_eq!(seam_out_idx, 0);
           p_shape.body_root
       }
   };
   ```

3. Fanout drops its `outputs.len() != 1` skip and instead groups consumers
   per `(producer, output)` pair, reusing the same `out_idx` plumbing.
4. Graph bookkeeping: `CandidateDraft` output lists and
   `apply::apply_solution` rebinding already handle multi-output nodes
   (keep proves this); only the *enumeration* and *seam selection* change.

### Why

- Unlocks fusing through keep nodes — the composition §1 makes costable.
  Without this, keep is only ever a leaf optimization.
- The GKR graph's fold chains produce multi-use intermediates; today those
  force a materialize-everything cut at every multi-output point.
- The σ-composition machinery (read coords → linearize → delinearize →
  scatter inverse, `producer_consumer.rs:936-1048`) is per-read and already
  indifferent to *which* output the read targets; the gate is purely
  structural.

---

## 3. Keep-variant generalization

### Current state

Keep requires plain shapes, an empty spine, and equal outer bounds
(`fusions/producer_consumer.rs:438-448`):

```rust
if variant == FusionVariant::Keep {
    // Keep materializes the seam at the fused domain's index, so it
    // requires plain kernels, a scalar producer body, and equal
    // outer bounds.
    if !p_shape.is_plain() || !c_shape.is_plain() || !spine.is_empty() {
        return Err(SynthesisFailure::UnsupportedShape);
    }
    if !two_tier_eq(&p_shape.outer_bound, &p_ctx, &c_shape.outer_bound, &c_ctx) {
        return Err(SynthesisFailure::OuterBoundMismatch);
    }
}
```

Drop already supports scatter/attrs on the producer, rank-extending spines
(Pack / inner-Compute drilling), and unequal bounds (σ handles the index
mapping). Keep supports none of these.

### Example

The GKR halving step with a surviving intermediate:

```text
p:  y = compute [q]   |i| x[i] + x[i+q]       # bindings {q: 8}
c:  z = compute [q/2] |i| y[i] + y[i+q/2]
# y is also read by a later verifier-side kernel → keep variant needed
```

Desired keep fusion — the seam recomputed by its own loop over the
*producer's* domain (proposed step 3):

```text
fused:  (z, y) = module {
    y = compute [q]   |i| x[i] + x[i+q]       # inner let, own bound
    z = compute [q/2] |i| y'(i) + y'(i+q/2)   # σ-inlined, drop-style
}
```

**Why it is rejected today:** the keep gate demands equal outer bounds —
`two_tier_eq(q, q/2, …)` fails → `OuterBoundMismatch`
(`producer_consumer.rs:445-447`). The drop sibling *is* emitted, but
since `y` is externally demanded the extractor must then also select
the original `p`, so the fold body runs twice. Likewise a producer
carrying `par`/`threads` launch attrs fails the `is_plain()` gate
(`:442`) even when the bounds match exactly — keep never fires on
attributed kernels at all.

### Proposed change

Lift the three restrictions in order of difficulty:

1. **Producer attrs (par/threads/block hints).** Launch hints vanish under
   inlining for drop; for keep the *seam element itself* is recomputed at
   the fused index, so hints are equally irrelevant. Delete `is_plain()`
   for attrs-only deviations (keep the scatter restriction initially):

   ```rust
   // is_plain() splits into is_plain_layout() (no scatter) and
   // has_launch_attrs(); keep only requires the former.
   if !p_shape.is_plain_layout() || !c_shape.is_plain_layout() { ... }
   ```

2. **Producer scatter.** The materialized seam must land in the layout
   downstream consumers expect. Emit the seam element under the *producer's*
   scatter by attaching the producer scatter to the seam output of the fused
   compute. This requires per-output scatter on `Node::Compute` — today
   `scatter` is a single per-compute field (`producer_consumer.rs:643-650`):

   ```rust
   let fused_body_id = fb.intern(Node::Compute {
       bound: outer_bound_fb,
       var: k_var,
       body: compute_body,
       scatter: c_shape.scatter.clone(),   // <- one scatter for all outputs
       par: c_shape.par.clone(),
       threads: c_shape.threads,
   });
   ```

   Proposed: `scatter: Vec<Option<Scatter>>` in lockstep with the tuple
   arity (single-output stays a 1-vector; codegen picks per-output write
   maps). This is an IR change and needs a lower_to_kir + codegen audit.

3. **Unequal bounds / non-empty spine.** Materialize the seam at the
   *producer's* index domain by emitting a second output whose write index
   is σ⁻¹ — but σ⁻¹ is not generally available (we only trust provided
   scatter inverses, never derive them). Instead, when bounds differ, emit
   the keep seam as a *separate inner-let compute* over the producer's own
   bound inside the fused module (the small-kernel pass already builds
   multi-layer bodies this way). The seam is then written by its own loop,
   not the consumer's.

### Why

- Today keep fires only on exactly-conformable pointwise pairs. The GKR
  fold chains are conformable *per level* but halve between levels — every
  cross-level pair with a surviving reader of the intermediate falls back
  to unfused because keep can't express "recompute at producer's domain".
- Step 1 is nearly free and immediately widens keep to attributed kernels
  (most real kernels carry launch hints).
- Steps 2–3 are ordered so each is independently shippable and testable.

---

## 4. Param-split (`normalize_params`) adoption in the other four passes

### Current state

Producer-consumer unifies parameters by *(name, bound value)* with `#k`
splitting, so a chain of one symbolic module at shrinking sizes shares one
artifact (`fusions/producer_consumer.rs:803-838`):

```rust
fn normalize_params(
    fb: &mut IRBuilder,
    claimed: &mut HashMap<String, (VarId, Option<i64>)>,
    merged_bindings: &mut BTreeMap<String, i64>,
    module: &Module,
    binding: &BTreeMap<String, i64>,
) -> HashMap<VarId, VarId> {
    // Each source param claims the first `base`, `base#1`, `base#2`, …
    // slot whose bound value matches (an unbound param only unifies
    // with another unbound one).
    ...
}
```

The other four passes still use the old name-only merge and *reject* on
value conflicts. Fanout (`fusions/fanout.rs:330-339`):

```rust
for (name, val) in &rec.bindings {
    match merged_bindings.get(name) {
        Some(existing) if existing != val => {
            return Err(FanoutFailure::ParamNameConflict);
        }
        _ => { merged_bindings.insert(name.clone(), *val); }
    }
}
```

Epilogue has the identical pattern with `EpilogueFailure::ParamConflict`
(`fusions/epilogue.rs:298-307`), horizontal with
`HorizontalFailure::ParamNameConflict` (`fusions/horizontal.rs:297-317`),
and small-kernel with `SmallKernelFailure::ParamNameConflict`
(`fusions/small_kernel.rs:381` area).

### Example

One seam, two consumers instantiated from symbolic modules that both
name their size `n` — at different values:

```text
p:   y = compute [8] |i| f(x[i])
c1:  u = scale(y)   bindings {n: 8}   # compute [n] |i| 2 * y[i]
c2:  v = fold(y)    bindings {n: 4}   # compute [n] |i| y[i] + y[i+n]
```

Desired fanout group (one launch, both consumers inline the producer):

```text
fused:  (u, v) = fanout module with params n=8, n#1=4:
    u = compute [n]   |i| 2 * f(x[i])
    v = compute [n#1] |i| f(x[i]) + f(x[i + n#1])
```

**Why it is rejected today:** fanout's merge sees `n=8` from `c1`, then
`n=4` from `c2` → `FanoutFailure::ParamNameConflict` (`fanout.rs:333`).
The identical pair fed to producer-consumer *does* fuse — its
`normalize_params` splits the name to `{n: 8, n#1: 4}` (proven by
`same_module_chain_splits_params`). Horizontal (`horizontal.rs:301`),
epilogue (`epilogue.rs:301`), and small-kernel hit the same wall on the
same inputs.

### Proposed change

Extract `normalize_params` + `strip_split_suffix` from `producer_consumer.rs`
into `fusions/mod.rs` (or a `params.rs` sibling) unchanged, and replace each
pass's name-only loop with two calls:

```rust
// fanout.rs sketch — replaces the seen_names loops at :318-350:
let mut claimed = HashMap::new();
let mut merged_bindings = BTreeMap::new();
let p_param_map =
    normalize_params(&mut fb, &mut claimed, &mut merged_bindings, &p_module, &p_binding);
let consumer_param_maps: Vec<_> = consumer_recs
    .iter()
    .map(|rec| normalize_params(&mut fb, &mut claimed, &mut merged_bindings,
                                &rec.module, &rec.bindings))
    .collect();
```

One subtlety: fanout/horizontal allocate params via
`fb.var_watermark()`/`raise_var_watermark` while `normalize_params` uses
`fb.fresh_var()` — these must produce the same VarId sequence because
`module_hash` hashes param VarIds raw (see the comment at
`producer_consumer.rs:374-382`). Verify with the existing
hash-identity tests (`chain_association_orders_hash_identically` pattern)
extended to each pass.

Then delete the now-unreachable `ParamNameConflict`/`ParamConflict` variants
(or keep them for genuinely un-unifiable cases, e.g. one side bound and the
other unbound at the same probe — `normalize_params` handles that by
splitting, so the variants should become dead).

### Why

- The GKR halving chains hit exactly this: two levels of the same module
  with `n=8` vs `n=4` cannot participate in fanout/horizontal/small-kernel
  groups today even when the group is otherwise legal. Every
  `ParamNameConflict` reject in `FUSION_V2_DEBUG=1` dumps is this bug.
- The mechanism is already proven in producer-consumer (tests:
  `same_module_chain_splits_params`, `chain_levels_share_one_artifact` —
  bindings `{n:8, n#1:4}` sharing one artifact hash).
- Pure refactor + adoption; no new theory.

---

## 5. Epilogue: affine seam access (drop the identity-read requirement)

### Current state

Epilogue only fuses when every consumer seam read is literally `y[k]`
(`fusions/epilogue.rs:286-292`):

```rust
let identity = [SExpr::sym(c_shape.outer_var)];
if seam_reads
    .iter()
    .any(|r| r.index_exprs.as_deref() != Some(&identity[..]))
{
    return Err(EpilogueFailure::SeamReadNotIdentity);
}
```

plus strict `p_shape.outer_bound != c_shape.outer_bound` equality (:266).

### Example

A block-hinted fold producer — epilogue is its only fusion path, since
producer-consumer skips non-plain producers
(`producer_consumer_covers`, `epilogue.rs:138-141`) — feeding a cheap
reversal:

```text
p:  y = compute [n] |k| fold_body(x, k)     # carries block_hint / par
c:  z = compute [n] |k| y[n-1-k] * w[k]
```

Desired fusion (producer schedule retained, σ = `n-1-k`):

```text
fused:  z = compute [n] |k| fold_body(x, n-1-k) * w[k]
```

**Why it is rejected today:** the seam read `y[n-1-k]` is not literally
`y[k]` → `index_exprs != [c_shape.outer_var]` → `SeamReadNotIdentity`
(`epilogue.rs:286-292`). Any stride or offset (`y[2k]`, `y[k+c]`) dies
the same way, even though producer-consumer's `ReadPlan` already knows
how to build exactly this σ.

### Proposed change

Reuse the producer-consumer σ machinery. Epilogue is producer-consumer
specialized to "consumer is pointwise and cheap"; the identity requirement
predates the general `ReadPlan` (`producer_consumer.rs:899-1048`). Replace
the check with a call to the shared read-plan builder:

```rust
// Sketch — epilogue.rs:
let plan = plan_seam_read(read, decl_dims, &p_phys, &spine,
                          p_shape.scatter.as_ref(), &p_ctx, &c_ctx)?;
// Inline producer body at σ(k) instead of k:
producer_vars.insert(p_shape.outer_var, emit_sexpr(&mut fb, &plan.sigma, k_var_node));
```

Keep the epilogue-specific gates that actually define the pass
(`ConsumerNotPointwise`, `ConsumerHasBlockHint`, cost threshold) and drop
only the access-pattern gate. Bound equality relaxes to the same drop gate
producer-consumer uses (element-count equality via `counts_eq`,
`producer_consumer.rs:882-897`).

### Why

- Post-fold epilogues in the GKR graph read with a stride or offset
  (`y[2k]`, `y[k + c]`) — all currently `SeamReadNotIdentity` rejects.
- Zero new machinery: this is convergence toward code that already exists
  and is tested in producer-consumer; long-term it shrinks epilogue to a
  thin policy layer over the shared synthesis core.

---

## 6. Small-kernel: parallel siblings and keep variant

### Current state

Chain growth stops at any multi-consumer seam
(`fusions/small_kernel.rs:203-214`):

```rust
// The seam must have exactly one consumer within the frozen
// prefix. Multiple consumers => fanout territory, deferred.
let consumers: Vec<NodeId> = gf.consumers[seam.0] ...;
if consumers.len() != 1 {
    break;
}
```

and the keep variant is rejected outright (`small_kernel.rs:284-288`):

```rust
if variant != FusionVariant::Drop {
    // Keep variant deferred (would require materializing every
    // internal seam alongside the last-layer outputs).
    return Err(SmallKernelFailure::UnsupportedShape);
}
```

Parallel siblings within a layer are documented as deferred (module doc,
`small_kernel.rs:57` area).

### Example

A diamond of tiny kernels (bound 8 — launch overhead dominates compute):

```text
t = compute [8] |i| f(x[i])
a = compute [8] |i| g(t[i])
b = compute [8] |i| h(t[i])
r = compute [8] |i| a[i] + b[i]
```

Desired: one fused launch with `t`, `a`, `b` as shared-memory tiles:

```text
fused:  r = small-kernel module {
    tile t = compute [8] |i| f(x[i])
    tile a = compute [8] |i| g(t[i])
    tile b = compute [8] |i| h(t[i])
    r      = compute [8] |i| a[i] + b[i]
}
```

**Why it is rejected today:** `find_chain` starting at `t` breaks
immediately — seam `t` has two consumers (`consumers.len() != 1`,
`small_kernel.rs:212`). Only the sub-chains `[a, r]` and `[b, r]` are
found; each still needs 3 launches, and since both contain `r` at most
one can be selected. Separately, if an internal seam (say `a`) were
also externally read, the keep variant that would cover it is rejected
outright (`variant != Drop` → `UnsupportedShape`,
`small_kernel.rs:284-288`), so the chain must cut there instead.

### Proposed change

1. **Layer DAGs instead of chains.** Replace the linear `find_chain` walk
   with a small-DAG collector: seed at a tiny kernel, greedily absorb
   frozen-prefix neighbors (producers and consumers) that satisfy
   `is_eligible_kernel_node`, subject to the same origin-disjointness union
   check (`small_kernel.rs:222-238`) and `max_chain_length` (renamed
   `max_group_size`). Body synthesis already lays out per-link inner lets;
   generalizing the topological order from a path to a DAG is mechanical
   (the links are emitted in topo order either way).
2. **Keep variant.** The deferral comment states the requirement precisely:
   materialize every internal seam alongside the last-layer outputs. With
   §3's per-output scatter and multi-output bodies this becomes: fused
   outputs = all layer outputs (not just the last layer's), each internal
   seam written from its own inner let. Gate keep on §3 landing.

### Why

- Small kernels in the GKR prologue form diamonds (one tiny producer, two
  tiny consumers, one tiny join) — today the chain walk breaks at the fork
  and each side launches separately, which is exactly the launch-overhead
  regime this pass exists to eliminate.
- The single-consumer break is a *search* limitation, not a soundness one:
  synthesis for a DAG group needs no new correctness argument beyond what
  the chain already has (origin disjointness + topo emission).

---

## 7. Horizontal: unequal / symbolic domains via `compute[max]` masking

### Current state

Both partners must have equal *concrete* bounds and equal block hints —
symbolic bounds are rejected in the prefilter (`fusions/horizontal.rs:160`,
`:255-257`) and pairs are gated at `:128`:

```rust
if ea.outer_bound != eb.outer_bound || ea.block_hint != eb.block_hint {
    continue;
}
```

```rust
if shape.outer_bound.as_const().is_none() {
    return Err(HorizontalFailure::NonConstantBound);
}
```

### Example

Two independent symbolic siblings over the same domain:

```text
a = compute [n] |i| f(x[i])     bindings {n: 8}
b = compute [n] |i| g(w[i])     bindings {n: 8}
```

Desired (one launch, and — because the module stays symbolic — one
artifact serving every `n`):

```text
fused:  (a, b) = compute [n] |i| ( f(x[i]), g(w[i]) )
```

**Why it is rejected today:** the prefilter demands a concrete bound —
`outer_bound.as_const()` is `None` → the node is never `Eligibility`
(`horizontal.rs:160`, `:255-257`) — so the pair loop never sees these
kernels, despite `n == n` being trivially provable after param
unification.

The masking case (proposed step 2) — adjacent fold levels:

```text
a = compute [n]   |i| f(x[i])
b = compute [n/2] |i| g(w[i])

fused:  (a, b) = compute [n] |i| ( f(x[i]),
                                   select(i < n/2, g(w[i]), undef) )
        # b's store predicated on i < n/2; b keeps logical length n/2
```

is rejected earlier still, at the equal-bound pair gate
(`ea.outer_bound != eb.outer_bound`, `:128`).

### Proposed change

1. **Symbolic-equal bounds** (low risk, high value): replace the
   `as_const()` requirement + raw `SExpr` equality with `two_tier_eq` from
   producer-consumer (post-§4, both sides share the fused param namespace,
   so `n == n` proves symbolically). This alone admits symbolic kernels
   with identical bounds.
2. **Unequal bounds via masking** (bigger step): fuse domains of sizes
   `m ≤ n` into one compute over `max = n`, guarding the smaller body:

   ```rust
   // body sketch: tuple element for the smaller partner becomes
   //   select(k < m, body_b(k), undef_of_type)
   // and its output keeps logical length m via a shape annotation.
   ```

   This requires (a) a `select`-guarded tuple element whose store is
   suppressed when the guard is false — i.e. predicated stores in codegen —
   and (b) cost-model awareness that the fused kernel does `max` work.
   The codegen read-sinking machinery (innermost-crossing sinks,
   `codegen.rs::compute_read_sinks`) already handles guarded *loads*;
   guarded *stores* are new.

Do step 1 immediately; gate step 2 on a demonstrated bench need — masking
pays only when launch overhead dominates the wasted `(n - m)` lanes, which
the KIR estimator can decide per candidate.

### Why

- Step 1: the GKR graph is symbolic-first; requiring concrete bounds means
  horizontal fires only after monomorphization, missing the shared-artifact
  window that makes fusion cheap across sizes.
- Step 2: sibling folds at adjacent levels (sizes `n`, `n/2`) are the
  dominant unfused pair-pattern left on the bench after producer-consumer;
  masking is the only way horizontal can touch them.

---

## 8. Pack drilling: non-constant component selection

### Current state

Drilling a seam read through a `Pack` spine step requires the component
coordinate to fold to a literal (`fusions/producer_consumer.rs:1027-1039`):

```rust
SpineKind::Pack { len } => {
    // Pack components are positional, so the coordinate
    // must fold to an in-range literal. Deliberately NOT
    // certified via `concretize`: a binding-dependent
    // component pick would bake this candidate's sizes into
    // the module text and break one-artifact-per-chain.
    let k = coord
        .as_const()
        .filter(|&k| k >= 0 && (k as usize) < len)
        .ok_or(SynthesisFailure::SeamComponentNotConst)?;
    DrillStepPlan::PackElem(k as usize)
}
```

### Example

An interleaving producer whose consumer picks the Pack component by
parity:

```text
p:  y = compute [n]  |k| pack( x[2k], x[2k+1] )      # y: [n, 2]
c:  z = compute [2n] |i| y[i/2, i%2] * s[i]
```

σ-composition itself succeeds: outer coordinate `i/2`, component
coordinate `i%2`. The *literal*-component sibling read `y[k, 0]` drills
fine today (`DrillStepPlan::PackElem(0)`).

Desired select-chain drill:

```text
fused:  z = compute [2n] |i|
            select( i%2 == 0, x[2*(i/2)], x[2*(i/2)+1] ) * s[i]
```

**Why it is rejected today:** the component coordinate `i%2` does not
fold to a literal → `coord.as_const()` is `None` →
`SeamComponentNotConst` (`producer_consumer.rs:1034-1037`). Note the
reject is *not* about analyzability — the coordinate is affine and
in-range by construction; the only missing piece is an encoding for a
positional pick at a symbolic position.

### Proposed change

When the coordinate is not literal, synthesize a select chain over the
(statically known, small) component count instead of failing:

```rust
SpineKind::Pack { len } => match coord.as_const() {
    Some(k) if k >= 0 && (k as usize) < len => DrillStepPlan::PackElem(k as usize),
    Some(_) => return Err(SynthesisFailure::SeamComponentNotConst), // out of range: real error
    None => DrillStepPlan::PackSelect { len, coord: coord.clone() },
    //      ^ new: emit select(coord == 0, elem0, select(coord == 1, elem1, ...))
};
```

The emitted selects stay symbolic in the module text, so artifact identity
is preserved (this respects the "deliberately NOT certified" rationale —
we never bake candidate bindings in). `len` is a structural property of the
producer body, identical across the chain. Codegen already supports nested
selects with sunk guarded reads, so each `elemK` subtree evaluates only in
its branch. Guard with a small `len` cap (e.g. ≤ 4) since the body grows
linearly in `len`.

### Why

- Fold kernels that pick even/odd halves via `k % 2` index a Pack with a
  non-literal coordinate — a `SeamComponentNotConst` reject today even
  though the access is perfectly analyzable.
- The select-chain encoding is exactly the shape the recent read-sink
  codegen fix was built to handle safely (guarded loads sink to their
  branch, both-sides uses stay eager and bounds-checked), so the backend
  risk is already paid for.

---

## 9. Reshape views: symbolic inner extents

### Current state

Linearize/delinearize across a reshape view requires every inner extent to
be a positive *literal* (`fusions/producer_consumer.rs:952-996`):

```rust
// Reshape view: linearize row-major over the declared view
// (Horner), then delinearize over the physical shape. All inner
// extents must be positive literals for strides to be
// expressible.
...
let d = d
    .as_const()
    .filter(|&d| d > 0)
    .ok_or(SynthesisFailure::SeamShapeMismatch)?;
flat = flat.mul_c(SymConst::Lit(d)).add(c);
```

and likewise for the physical trailing dims (`:970-981`), whose strides are
folded into `i64` literals.

### Example

A fold producer whose consumer declares the seam under a symbolic
reshaped view:

```text
p:  y = compute [n]   |k| x[k] + x[k+n]              # y stored flat: [n]
c:  z = compute [n/2] |i| yv[0, i] * yv[1, i]        # seam declared yv: [2, n/2]
```

The drop gate passes (`2 * n/2 == n` element-count equality, tier 1).
Linearizing the read `yv[1, i]` gives `1*(n/2) + i` — but the inner
extent `n/2` is symbolic.

Desired (symbolic Horner + strides; the composed σ stays symbolic, so
the halving chain keeps one artifact):

```text
fused:  z = compute [n/2] |i| (x[i] + x[i+n]) * (x[n/2 + i] + x[n/2 + i + n])
```

**Why it is rejected today:** every inner extent must be a positive
literal — `d.as_const().filter(|&d| d > 0)` fails on `n/2` →
`SeamShapeMismatch` (`producer_consumer.rs:960-964`). The concrete tier
cannot rescue it: folding `n/2` at this candidate's bindings would bake
sizes into the module text and break one-artifact-per-chain, so the
reject is unconditional for symbolic kernels.

### Proposed change

`SExpr` already supports symbolic multiplication, floordiv, and rem with
symbolic operands (σ composition uses them). Lift the literal requirement
by carrying strides as `SExpr` instead of `i64`:

```rust
// Sketch:
let mut flat = coords[0].clone();
for (c, d) in coords[1..].iter().zip(&decl_dims[1..]) {
    flat = flat.mul(d).add(c);        // d: SExpr, not folded literal
}
let mut strides: Vec<SExpr> = vec![SExpr::one(); p_phys.len()];
for j in (0..p_phys.len() - 1).rev() {
    strides[j] = strides[j + 1].mul(&trailing[j]);
}
// delinearize with SExpr floordiv/rem instead of Lit divisors
```

Two prerequisites:

1. `SExpr::floordiv`/`rem_c` must accept `SExpr` divisors (today they take
   `SymConst`); the KIR emitter must lower symbolic div/rem — check the
   `emit` path in canonicalize's flattening machinery
   (`canonicalize.rs:763-764` already emits `floordiv`/`rem_c` from
   flattened bounds, so the lowering exists for the const case; extend it).
2. Positivity: literal extents were filtered `> 0`; symbolic extents need
   the same guarantee. Shape declarations already imply positive extents,
   so record this as an invariant rather than a runtime check, but keep the
   tier-2 concrete certification (`counts_eq`) as the safety net — a
   candidate whose bindings make an extent zero fails the drop gate before
   reaching the read plan.

### Why

- This is the last purely-symbolic gap in the σ pipeline: reads, scatter
  inverses, and bound comparisons are all symbolic already; only reshape
  strides force concreteness. Fixing it keeps halving-chain candidates
  binding-independent (one artifact per chain) even when the consumer views
  the seam at a different rank.
- Without it, any symbolic kernel that declares a reshaped seam view
  (`[n] → [n/2, 2]`) is a `SeamShapeMismatch` reject regardless of how
  simple the actual access is.

---

## Suggested sequencing

| Phase | Items | Rationale |
|-------|-------|-----------|
| A | §1, §4 | Recover wasted candidates + unlock cross-size grouping; no new fusion theory, both land as independent PR-sized changes with existing test patterns. |
| B | §2, §5 | Multi-seam plumbing and epilogue-σ convergence; both reuse the read-plan core, and §2 makes §1's recovered keep nodes composable. |
| C | §3 (steps 1–2), §8 | Keep generalization through per-output scatter; Pack select-chains. IR-touching but bounded. |
| D | §6, §7, §9, §3 (step 3) | Search-space widening (DAG groups, masking, symbolic strides) — best justified by re-profiling the GKR bench after phases A–C, since A–C may already close most of the 3.2× gap. |

Each phase's exit gate mirrors the M-milestone pattern in
`fusion-v2-progress.md`: pass-level unit tests, driver saturation test,
brute/CP-SAT agreement where extraction semantics change (none of the items
above change extraction), and a GKR bench replay with
`FUSION_V2_DEBUG=1` reject-histogram deltas recorded in the progress doc.
