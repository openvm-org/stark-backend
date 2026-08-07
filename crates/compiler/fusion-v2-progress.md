# Fusion v2 Implementation Progress

Tracks progress against `detailed-fusion-plan-v2.md`. Update as milestones land.

## Milestone status

| Milestone | Status | Session | Notes |
|-----------|--------|---------|-------|
| M0-part 1 | ✅ Done | 1 | Core visitor + basic collectors + alpha-renamed cloning |
| M0-part 2 | ⏳ Deferred | — | AccessCollector, KernelOutputView, BoundaryBuilder, Quast composition, inverse verification — best landed alongside M3 |
| M1 | ✅ Done | 2 | Versioned seed model + original extraction |
| M2 | ✅ Done | 3 | CP-SAT extraction model (x/y/z), brute-force reference, agreement tests |
| M3 | ✅ Done | 4-6 | Identity, affine, nested-index, reduction producer — all match hand-authored references. Multi-seam deferred (needs multi-output producers). GPU equivalence test blocked on M11. |
| M4 | ✅ Done | 7 | KIR estimator v0: liveness/occupancy/transactions/critical-path/aggregate cycles + KernelCostManager + non-kernel closed-form + driver wiring |
| M5 | ✅ Done | 8 | Keep-seam variants (§10.2): FusionVariant Drop/Keep, Tuple body for keep, §10.2 trigger conditions, driver enable flag |
| M6 | ✅ Done | 9 | Bounded saturation and chain extraction (§11): SaturationState (origins + seen_candidates), multi-round driver loop, CandidateKey dedup, per-pass and per-round caps, min_new_parent_id pruning |
| M7 | ⬜ Todo | — | Fanout pass |
| M8 | ⬜ Todo | — | Small-kernel block fusion |
| M9 | ⬜ Todo | — | Same-domain horizontal fusion |
| M10 | ⬜ Todo | — | Epilogue pass |
| M11 | ⬜ Todo | — | Opt-in integration + golden comparison |
| M12 | ⬜ Todo | — | Numerical accuracy on fractional_sumcheck |

## What landed

### M0-part 1 (session 1)

**File:** `crates/compiler/src/passes/fusion_utils.rs` (registered in `passes/mod.rs`)

Independent HIR traversal utilities, none importing from `passes/fusion.rs`:

- `VisitControl` (`Recurse` / `SkipChildren`)
- `IndexKind` (`Compute` / `Reduce`)
- `IndexBinding` — a compute/reduce loop binder in scope
- `HirVisitor` trait with balanced `enter`/`leave`, occurrence-based traversal
- `visit_hir(module, root, visitor)` — deterministic child order via existing
  `module_hash::children_of`; only `Compute`/`Reduce` extend the scope for their
  body child; active-recursion cycle guard returns `MalformedHir`
- `unique_index_scope(module, target)` — plan §8 requirement: rejects a
  hash-consed access reached under two unequal index scopes
- `AmbiguousAccessScope`, `MalformedHir`, `VisitError`
- Collectors: `collect_input_uses` → `InputUse`, `count_reachable_nodes`,
  `collect_structure` → `StructureFacts`
- `clone_expr(src, root, dst, subst, subst_vars)` — deterministic alpha-renamed
  HIR cloning with explicit `NodeId → NodeId` and `VarId → VarId` substitution
  maps; fresh `VarId` per Compute/Reduce/Let binder for capture-free composition
- `clone_pure`, `bound_vars` convenience helpers

**Tests (11, all passing):** visitor scope threading, `SkipChildren` still calls
`leave`, `unique_index_scope` accepts and rejects appropriately,
`count_reachable_nodes` matches expected DAG size, `collect_input_uses` records
every occurrence, `collect_structure` records computes/lets, `clone_expr`
preserves module hash on a full-body clone, alpha-renames independently on
duplicate clones, and applies substitution maps.

**Verification:**
- `cargo check -p crypto-compiler` clean
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean
- `cargo +nightly fmt` applied
- 2 gpu_graph tests (`module_with_intermediate_buffers_is_rejected`,
  `symbolic::partial_monomorphization_and_fusion`) fail — verified pre-existing
  on `main` sans my changes.

### M0-part 2 (deferred)

The following M0 items are best landed alongside their first real consumer:

- `AccessCollector` + `RawAccessRelation` — naturally paired with M1's
  `ValueClassId` (post value-binding produces the final `AccessRelation`)
- `KernelOutputView` — first consumer is M3 producer-consumer synthesis
- `BoundaryBuilder` — needs `ValueClassId` from M1; first consumer is M3
- Quast composition + bounded-domain inverse verification — first consumer is
  M3 legality proofs
- Tuple-output construction + module renumbering helpers — M3 synthesis

### M1 (session 2)

**Files:**
- `crates/compiler/src/graph_ir.rs` — added positional API on `GraphNode`:
  - `get_operands(bufs)` — logical operands including memcpy/memset
    preservation inputs (§5.1)
  - `get_results()` — explicit results followed by re-exported carried
    outputs
  - `covers_full_range(offset, num_bytes, size)` private helper — structural
    check that a memcpy/memset covers its whole destination
- `crates/compiler/src/passes/fusion_v2/mod.rs` — module skeleton with
  `access`, `apply`, `extract`, `model`, `version` submodules
- `crates/compiler/src/passes/fusion_v2/model.rs` — `ValueClassId`,
  `NodeId`, `ValIdMap`, `NodeIdMap`, `AltGraphNode`, `UseInfo`,
  `GraphFuser` with `physical()` helper. Registered in `passes/mod.rs`.
- `crates/compiler/src/passes/fusion_v2/access.rs` — `AccessRelation`,
  `ReadRelation`, `WriteRelation` shape per §8. Extractor deferred.
- `crates/compiler/src/passes/fusion_v2/version.rs` — `take_graph(&mut
  GraphBuilder) -> Result<GraphFuser, TakeGraphError>`, implementing §6.2:
  moves `GraphNode`s out of the builder, allocates new value classes per
  written buffer, resolves inputs before publishing outputs.
- `crates/compiler/src/passes/fusion_v2/extract/mod.rs` —
  `ExtractionSolution` + `ExtractionSolution::original(&gf)` baseline
  (§6.4) and `FallbackReason` enum (§15 hooks).
- `crates/compiler/src/passes/fusion_v2/apply.rs` — `apply_solution`
  reconstruction (§7 + §14) with Kahn's topological sort over a
  RAW/WAW/WAR precedence graph derived from the selected set. `BufId`
  lookup uses `.0` keys since `BufId` is not `Ord`.
- `crates/compiler/src/passes/fusion_v2/tests.rs` — 9 M1 exit-gate tests.

**Tests (9, all passing):**
- `take_graph_versions_a_single_writer_chain` — new versions allocated per
  write, physical BufId recovered via `re_exported`
- `take_graph_rejects_read_before_write`
- `original_solution_round_trip_preserves_node_order` — fingerprint match
- `original_solution_round_trip_with_const_and_memcpy` — non-kernel
  variants exercised
- `full_memcpy_has_no_preservation_input`
- `partial_memcpy_adds_preservation_input`
- `re_exported_versions_share_physical_bufid`
- `round_trip_matches_registered_output_final_version`
- `hazard_order_respects_waw_between_selected_writers`

**Verification:**
- `cargo check -p crypto-compiler` clean
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean
- `cargo +nightly fmt` applied
- `cargo nextest run -p crypto-compiler --lib` — 223 tests, all pass.
- The 2 pre-existing gpu_graph failures noted in session 1 still fail
  on `main` sans my changes; not introduced by M1.

**Simplifications relative to plan:**
- Insertion-time DAG validator (§9.1) is deferred to M3 when candidate
  synthesis lands — for M1 there are no candidates to validate.
- Graph-take guard (§14.3) is not implemented: on M1 error paths the
  original nodes remain owned by `GraphFuser`, and the caller currently
  drops `gf` and reconstructs from scratch. This is sufficient until v2
  is wired into `GraphCompiler` (M11).
- `rewrite_bindings` on `GraphNode` is deferred to M3 when candidate
  synthesis is the first real consumer.

### M2 (session 3)

**Files:**
- `crates/compiler/src/passes/fusion_v2/cost.rs` (new): `GraphNodeCost`,
  `ArtifactKey`, `ArtifactContext` per §5.5 and §12; `key_for` helper on
  `ArtifactContext`.
- `crates/compiler/src/passes/fusion_v2/model.rs`: added
  `GraphFuser::insert_candidate` — appends a fusion candidate that reuses
  existing value classes and keeps the producer/consumer/access sidecars
  in lockstep (§14.2's MVP no-new-value-class rule).
- `crates/compiler/src/passes/fusion_v2/extract/mod.rs`: reshaped to
  `pub mod brute; #[cfg(feature = "planner-ortools")] pub mod cpsat;`;
  added `ExtractionData` (§5.4 sidecar with `costs` + `artifact_keys`),
  `ExtractOptions` (solver time limit, module budgets, `cycle_quantum`,
  `runtime_tolerance_ppm`), `SolverStatus`. `ExtractionSolution::original`
  now also carries `status: None`.
- `crates/compiler/src/passes/fusion_v2/extract/brute.rs` (new):
  Exhaustive enumerator over subsets — always available. Implements the
  full four-stage lex objective (§13.5). Caps at
  `BRUTE_FORCE_LIMIT = 32` alternatives.
- `crates/compiler/src/passes/fusion_v2/extract/cpsat.rs` (new, gated on
  `planner-ortools`): x/y/z boolean variables (§13.2), single-producer,
  boundary-input, upward/downward artifact-activation constraints (§13.3),
  optional `max_modules`/`max_new_modules` budgets, and strict four-stage
  lexicographic minimization solved sequentially (§13.5) with a
  `runtime_tolerance_ppm` slack on the stage-1-to-stage-2 lock.
- `crates/compiler/src/passes/fusion_v2/tests.rs`: added an `extractor`
  submodule (5 brute-force tests) and a `cpsat_agreement` submodule (4
  CP-SAT vs brute-force tests, one of which is the M2 exit-gate
  randomized property test running 32 LCG seeds × 6 candidates each).

**Tests (10 new, all passing):**

Brute-force cases:
- `no_candidates_solver_returns_original` — original is the unique
  feasible solution.
- `brute_force_prefers_cheaper_fused_candidate` — the fused alternative
  wins on runtime.
- `brute_force_keeps_original_when_fused_is_more_expensive` — the
  original wins on runtime.
- `shared_artifact_across_two_alternatives_is_charged_once` — artifact
  count objective is exact-OR.
- `max_new_modules_zero_forces_original` — budget cap makes fused
  infeasible.
- `value_count_tiebreak_drops_unused_intermediate` — stage-3 node-count
  drops the redundant seed after runtime ties.

CP-SAT agreement cases (feature-gated):
- `agree_no_candidates`
- `agree_cheaper_fused_candidate`
- `agree_max_new_modules_zero`
- `agree_random_property` — 32 LCG seeds, 6 random candidates each,
  compares brute-force and CP-SAT lex-costs (not identical selections
  because ties are possible).

**Verification:**
- `cargo check -p crypto-compiler --lib` clean (planner-ortools disabled).
- `cargo check -p crypto-compiler --features planner-ortools` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean
  (both feature configurations).
- `cargo nextest run -p crypto-compiler --lib` — 229 tests, all pass
  (15 fusion_v2 without feature + everything else).
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — 19 tests, all pass (adds 4 cpsat_agreement tests).

**Design decisions:**
- `insert_candidate` on `GraphFuser` is `pub` because tests and the
  eventual M3 saturation driver both need it; validation of legality
  (disjoint origins, matching shapes) is the caller's responsibility.
- Lexicographic solve uses **sequential minimization with equality/slack
  constraints** (§13.5): each stage is a fresh `minimize` call after
  locking the previous stage's value. `cp_sat` 0.4.1 doesn't support
  native multi-objective on top of the CP-SAT protobuf model exposed by
  this bindings crate.
- Brute force compares costs as `(i128 runtime, u64 art, u64 node, u64
  val)` tuples and uses `Ord` directly on the tuple.
- Because `ArtifactKey` is not `Copy`, the brute force clones it each
  time it lands in a `HashSet`; the sets stay small (≤ artifact-count)
  and the LCG property tests only use 3 distinct artifact keys anyway.
- The randomized property test compares **lex-costs**, not selected sets,
  because two different subsets can share the same cost tuple under
  perfect ties.

**Not landed in M2** (still on the plan for later milestones):
- Native solver-hint support for the original solution (§13.6, marked
  optional in the plan).
- `NoImprovementOverOriginal` fallback wiring (M11 integration).
- Estimator-computed `GraphNodeCost` and per-artifact compile cost — M4.
- Feasibility partitioning (§13.8, explicitly deferred).

### M3 first slice (session 4)

Landed the identity-access drop-seams pipeline end-to-end.

**Files:**
- `crates/compiler/src/graph_ir.rs`: added `GraphNode::rewrite_bindings`
  (§5.1 completion). Rewrites positional buffer bindings for every
  variant — Kernel, BlackboxKernel (rebuilds carried-outputs suffix
  from the input positions that appear in the original carried set),
  Const, Memcpy (validates preservation-input == destination), Memset
  (same). Returns `CompileError::Canonicalize` on shape mismatches.
- `crates/compiler/src/passes/fusion_v2/fusions/mod.rs` (new):
  registers the fusion passes; only `producer_consumer` for now.
- `crates/compiler/src/passes/fusion_v2/fusions/producer_consumer.rs`
  (new):
  - `CandidateDraft` type — a candidate carrying its parents plus a
    finalized `AltGraphNode`. Passes do not allocate `NodeId`s; the
    saturation driver / caller does.
  - `identify_identity_kernel` — recognizes the narrow M3 shape:
    single top-level `Compute` (no scatter/par/threads/reduce/nested
    compute), every reachable `Node::Input(_)` reads via a
    `Node::Index` whose only index is `Var(outer_var)`.
  - `synthesize_identity(gf, producer, consumer, seam)` — capture-free
    HIR clone: declares producer inputs + consumer non-seam inputs in
    the fused module, clones the producer body into the fused
    builder once (as the substitute for every seam `Index` site),
    then clones the consumer body applying the substitution map.
    Alpha-renames every parent `outer_var` and every parent parameter
    to fresh identities in the fused module. `SynthesisFailure`
    enumerates the rejection modes (NotAKernel, UnsupportedShape,
    OuterBoundMismatch, ProducerNotSingleOutput, NoSeamReadInConsumer,
    CloneError, TypeCheckFailed).
  - `enumerate(gf)` — iterates single-producer/single-consumer seams
    among seed kernels and returns every accepted `CandidateDraft`.
    Non-seed nodes are skipped (chain composition is M6).
- `crates/compiler/src/passes/fusion_v2/tests.rs`: added the
  `producer_consumer_tests` submodule (4 tests) that builds a two-kernel
  scale-by-two / scale-by-three chain, runs the M3 enumeration, and
  verifies:
  - one candidate is produced;
  - the synthesized module type-checks;
  - the synthesized module hashes byte-for-byte identical to a
    hand-authored `compute[N] |i| 3 * (2 * x[i])`;
  - inserting the candidate, extracting with brute-force, and applying
    yields a one-node graph.

**Tests (4 new, all passing):**
- `enumerate_identity_chain_produces_one_candidate`
- `synthesized_module_type_checks`
- `synthesized_module_hash_matches_hand_authored_reference` — proves the
  synthesis produces the semantically-expected HIR structurally.
- `extractor_picks_cheap_fused_candidate_and_apply_produces_one_node` —
  full pipeline: enumerate → insert → extract → apply.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib` — 233 tests, all pass.
- 19 fusion_v2 tests pass (M0/M1/M2 + 4 M3 tests).

**Design decisions:**
- HIR alpha-renaming leans on `fusion_utils::clone_expr`'s `subst`
  (NodeId map) and `subst_vars` (VarId map). The fused module's
  compute uses one fresh `VarId` for the outer index; the
  substitution maps producer's and consumer's outer vars both onto
  this fresh var, so a single clone pass handles both alpha-renames.
- Parameter merging is nominal: same param name in producer and
  consumer resolves to the same fused-module param VarId. Conflicting
  bindings for the same name reject the candidate as
  `UnsupportedShape`. Full symbolic parameter unification lands
  with the affine-permutation case.
- The fused `KernelModuleNode` records the parent modules' `param_bindings`
  merged; `types` / `hash` / `canonical` reset to defaults so
  downstream passes recompute them.
- `enumerate` deliberately restricts to seed nodes (no fused-into-fused
  composition). Chain composition is M6 (bounded saturation).
- No dedup for identical drafts yet — the M3 first slice produces one
  candidate per (producer, consumer, seam) and doesn't chase pattern-key
  bucketing (§10.0). Ties naturally dedupe by `CandidateKey` at insertion
  once §9 finalization lands.

**Landed in the M3 final slice (session 6):**
- Affine-permutation access maps (see final-slice section below).
- Nested-index consumers.
- Reduction producers.

**Deferred beyond M3 exit gate:**
- Multi-seam grouping — requires multi-output producers with `Tuple`
  bodies.
- Shared `AccessRelation` extractor (raw form) — the M3 synthesis
  inlines shape recognition; the M0-part 2 `AccessCollector` will
  subsume it when other passes need the extracted form.
- Full candidate finalization pipeline (§9): canonicalize, monomorphize,
  launch-schedule validation, artifact-key computation, `CandidateKey`
  dedup, boundary pruning.
- Bounded saturation with multi-round dispatch and per-pass caps (§11
  → M6).
- Semantic-equivalence test against unfused GPU output (M11 golden
  suite).

### M3 second slice (session 5): driver + §9.1 validator

Landed the §9.1 insertion-time acyclicity validator and the top-level
`fuse_graph_v2` driver, so callers now have a one-liner entry point.

**Files:**
- `crates/compiler/src/passes/fusion_v2/validate.rs` (new):
  `would_create_cycle(gf, inputs, outputs)` — BFS forward from every
  output; reports a cycle if it reaches any of `inputs`. Implements
  §9.1 exactly.
- `crates/compiler/src/passes/fusion_v2/driver.rs` (new):
  - `FusionOptionsV2` (M3-relevant subset of §15) with
    `max_total_alternatives`, `validate_alt_graph_acyclicity`,
    `solver_time_limit_secs`, `enable_producer_consumer`.
  - `FusionReportV2` — `nodes_before/after`, candidate counts by
    outcome (generated / inserted / rejected_cycle / rejected_cap),
    `selected_from_solver`, `fallback_reason`.
  - `FuseV2Error` — thiserror wrapping `TakeGraphError`/`ApplyError`.
  - `fuse_graph_v2(g, options) -> Result<FusionReportV2>` — runs
    take_graph → enumerate → validate (§9.1) → insert → extract →
    apply in one call.
  - Placeholder cost model: every alt gets `runtime_units = 1`,
    `Kernel` variants derive `ArtifactKey` from their pre-computed
    `KernelModuleNode::hash` (or a freshly-computed `module_hash`);
    non-kernel nodes have `artifact_key = None`. Explicitly marked as
    "until M4 lands the KIR estimator".
  - `choose_extractor`: CP-SAT when `planner-ortools`, else brute
    force (with the original as a final fallback if the graph exceeds
    `BRUTE_FORCE_LIMIT`).

**Tests (12 new, all passing):**

Validate tests (4):
- `candidate_that_cycles_is_rejected` — proposes an `inputs=[c],
  outputs=[a]` candidate on a `a → b → c` chain; validator returns true.
- `candidate_that_does_not_cycle_is_accepted` — the legitimate fused
  candidate on the same chain.
- `empty_output_set_never_cycles`.
- `candidate_with_input_equal_to_output_is_a_self_cycle` — validator
  correctly rejects immediate self-loops.

Driver tests (8):
- `driver_fuses_two_kernel_chain_into_one_kernel` — full pipeline
  reduces a two-kernel chain to one; interface preserved.
- `driver_leaves_single_kernel_unchanged` — no candidates on a
  one-kernel graph.
- `driver_leaves_disjoint_kernels_unfused` — no producer-consumer
  edge means no candidates.
- `driver_produces_hand_authored_reference_module` — the resulting
  single kernel structurally matches `3 * (2 * x[i])`.
- `driver_enumerates_two_candidates_when_producer_feeds_two_consumers`
  — fanout produces one drop candidate per (producer, consumer) pair.
- `driver_max_total_alternatives_zero_disables_all_fusion` — cap
  path reflected in `candidates_rejected_cap`.
- `driver_disable_producer_consumer_produces_no_candidates` — feature
  flag path.
- `driver_leaves_registered_output_intact_when_seam_is_graph_output`
  — the fused drop candidate cannot displace the producer when the
  seam is a demanded graph output; extractor keeps a producer for
  the seam.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean (both feature configs).
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean
  (both configs, including `planner-ortools`).
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib` — **241** tests, all pass.
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — **31** fusion_v2 tests, all pass.

**Design decisions:**
- `fuse_graph_v2` is a single-round entry point. Multi-round saturation
  (§11) requires §9's full candidate finalization and pattern-key
  bucketing (§10.0); both land in M6.
- The placeholder cost model deliberately makes fusion attractive on
  a simple chain (2 seed nodes → 1 fused node under `sum x_a` runtime)
  while leaving fanout/graph-output cases unfused — because those need
  a producer to remain live, replicating work under the placeholder
  costs. This matches the plan's rejection expectations for M3.
- `choose_extractor` uses `unwrap_or_else` on the brute-force result
  so a graph larger than `BRUTE_FORCE_LIMIT` gracefully falls back to
  the original solution rather than panicking.
- `FusionReportV2::fallback_reason` is set from the extractor's
  solution to keep the report as a single source of truth on the
  extraction path taken. The M11 wiring will lift this into the
  `GraphCompiler` fusion report (§15) alongside existing v1 fields.

## Testing tally

| Session | Milestone | Tests added | Tests passing |
|---------|-----------|-------------|---------------|
| 1 | M0-part 1 | 11 | 11 |
| 2 | M1 | 9 | 9 |
| 3 | M2 | 10 | 10 (4 gated on `planner-ortools`) |
| 4 | M3 first slice | 4 | 4 (identity-access chain end-to-end) |
| 5 | M3 second slice | 12 | 12 (driver + §9.1 validator + fanout/cap/output tests) |
| 6 | M3 final slice | 3 | 3 (affine permutation, reduction producer, nested-index consumer) |
| 7 | M4 | 12 | 12 (determinism, cost comparisons, occupancy, cache, non-kernel closed-form; 46 total fusion_v2 lib tests) |
| 8 | M5 | 10 | 10 (keep-variant enumeration, HIR shape, cost pricing, extractor picks keep on graph-output/fanout; 56 total fusion_v2 lib tests) |
| 9 | M6 | 7 (+3 saturate submodule) | 66 total fusion_v2 lib tests; 3-kernel chain collapses via composition; associativity dedup; determinism; caps; origin filter |

### M3 final slice (session 6): affine / nested / reduction

Extended the producer-consumer pass to cover the remaining M3 exit-gate
patterns.

**Files:**
- `crates/compiler/src/passes/fusion_utils.rs`:
  - `clone_expr`'s `subst_vars` argument changed from
    `HashMap<VarId, VarId>` to `HashMap<VarId, NodeId>`. The simple
    alpha-rename becomes `.insert(v, dst.intern(Node::Var(v')))`; the
    new form lets callers substitute a bound variable with an
    arbitrary destination expression, which is what inlining a
    producer body at an affine consumer index requires.
  - New `clone_expr_with_hook(src, root, dst, subst, subst_vars, hook)`.
    The hook is called at every source `NodeId` before the default
    clone logic runs, with the destination builder and a snapshot of
    the current source-VarId → destination-NodeId map. Returning
    `Ok(Some(id))` uses `id` as the replacement (memoized). This is
    the mechanism the producer-consumer synthesis uses to emit a
    site-specific producer-body inline at each seam read, including
    reads whose index uses an inner Compute/Reduce variable.
- `crates/compiler/src/passes/fusion_v2/fusions/producer_consumer.rs`:
  - Replaced `IdentityKernel` with a general `KernelShape` that
    records read sites' index expressions as `Quast` (via `hir_to_quast`)
    and their in-scope binder stack.
  - Renamed `synthesize_identity` → `synthesize_producer_consumer`.
  - Added `IndexEmitter` — a `QuastEmitter` that lowers a `Quast`
    into HIR nodes on a destination `IRBuilder`.
  - Synthesis uses a single hook-based clone: at every seam-read site
    it emits the read's index expression as a fresh HIR expression
    (using `IndexEmitter` against the current var snapshot) and
    clones the producer body with producer's outer var substituted
    by that expression. Both outer-scope reads (identity, affine) and
    inner-scope reads (nested-index consumers) are handled by the
    same loop.
  - `SynthesisFailure::SeamIndexNotAffine` remains for reads whose
    index expression `hir_to_quast` cannot handle (rare in practice).

**Tests (3 new, all passing):**
- `synthesized_module_supports_affine_permutation_consumer` — the
  fused module for `y = 2*x`, `z[i] = 5 * y[N-1-i]` hashes identical
  to `compute[N] |i| 5 * (2 * x[N-1-i])`.
- `synthesized_module_supports_reduction_producer` — the fused module
  for `y[i] = sum_{j<K} c[j] * x[i]`, `z = 3 * y` hashes identical to
  `compute[N] |i| 3 * (sum_{j<K} c[j] * x[i])`.
- `synthesized_module_supports_nested_index_consumer` — the fused
  module for `y = 2*x`, `z[i] = sum_{j<N} y[j]` hashes identical to
  `compute[N] |i| sum_{j<N} 2*a[j]` (producer's input name is
  inherited by the fused module).

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo clippy -p crypto-compiler --lib --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib` — **248** tests, all pass.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **34**
  fusion_v2 tests, all pass.

**Design decisions:**
- Chose a clone-time hook over pre-cloning the producer body once and
  substituting via a `subst: HashMap<NodeId, NodeId>`. The pre-clone
  strategy works only when every seam read is under the top-level
  compute; nested-index consumers require access to fresh identities
  introduced by inner Compute/Reduce nodes at the moment of the read,
  and the hook naturally exposes them via the passed-in `vars`
  snapshot.
- The hook signature exposes a snapshot of the current `vars` map as
  `&HashMap<VarId, NodeId>` — it captures both the caller-supplied
  outer bindings and any fresh binders `clone_expr` allocated for
  Compute/Reduce/Let it has already descended into. This is exactly
  what the seam-substitution logic needs.
- Kept the clone_expr memoization: if a hash-consed seam Index NodeId
  is visited more than once under the *same* enclosing scope
  (the fusion pipeline already guarantees this via unique-scope
  reads), the memoized inline is safe.

### M4 (session 7): KIR estimator v0

Replaces the placeholder "runtime = 1" cost model with a real
HIR→KIR-driven cycle estimator. The extractor now consumes calibrated
costs; behavior on the M3 tests is unchanged (all existing tests still
prefer the fused single-kernel outcome on the two-mul chain).

**File layout:**

Converted `passes/fusion_v2/cost.rs` into a `cost/` module:

- `cost/mod.rs` — re-exports the estimator surface and keeps
  `ArtifactKey`/`ArtifactContext`/`GraphNodeCost` (§5.5). Adds
  `GraphNodeCost::from_cycles(total, quantum)` to quantize with a
  floor of 1 per §13.5.
- `cost/liveness.rs` — per-thread register liveness estimate (§12.3).
  Builds an `SSARes → word count` table (`BabyBear`/`U32`/`Bool` = 1
  word, `FpExt` = 4 words, unknown fallback = 4 words) and walks each
  `SSABlock` in reverse to track the running total of live words,
  taking the peak across all program points. Loop bodies are visited
  twice as a bounded fixed point for carried liveness. Returns
  `RegisterEstimate { max_live_words, registers_per_thread }` — the
  scaled per-thread estimate adds `register_fixed_overhead +
  ceil(register_liveness_scale * max_live_words)`.
- `cost/transactions.rs` — global-memory sector counting (§12.5).
  `estimate_access` deterministically samples
  `warp_samples_per_par` warps within the par's domain, evaluates
  `IndexMap::Linear`/`Affine` per-lane to distinct sector buckets,
  and averages the sector count. `SExpr`/`Blackbox` maps fall back to
  the configured `unknown_global_sectors_per_warp`. Sample seed comes
  from a SplitMix64 mix of `(module_hash, kernel_index, par_node,
  access_index, model_version)`.
- `cost/interpreter.rs` — critical-path interpreter (§12.6). Walks
  the grid block, tracking each SSA value's ready cycle and dependent
  global-load depth. `Bin(op, ty)` reads per-op latencies from
  `OpLatencyTable::bin_latency`. Loop replicates the body's critical
  path by the loop bound while respecting carried dependencies. Par
  reads add the global-latency contribution attenuated by
  `min(active_warps, latency_saturation_warps)`. Sync adds
  `sync_latency_cycles`. Weighted dynamic ops accumulate scaled by
  the enclosing par/loop multipliers.
- `cost/estimator.rs` — top-level `estimate_kernel` and
  `estimate_non_kernel` (§12.7 aggregate + §12.8 non-kernel).
  `analyze_program` lowers the module via `ModuleCompiler::lower`,
  calls `plan_shared_mem` for the per-kernel shared footprint, then
  combines occupancy, critical path, transaction bytes, and issue
  weighting per plan:
  ```text
  latency_cycles   = block_waves * critical_cycles_per_block
  bandwidth_cycles = transaction_bytes / dram_bytes_per_cycle
  issue_cycles     = weighted_ops / issue_weighted_ops_per_cycle
  launch_cycles    = fit_sm / within_wave / multi_wave  (§12.7 tiers)
  total_cycles     = launch_cycles + max(latency, bandwidth, issue)
  ```
  Non-kernel costs: `Const = 0`, `Memcpy`/`Memset = memop_launch +
  bytes / memcpy_bytes_per_cycle`, `BlackboxKernel = caller hint`.
- `cost/cache.rs` — `KernelCostManager` per plan §12.10. Cache key is
  `(module_hash, hash_param_bindings(bindings))`; a hit returns the
  cached `GraphNodeCost` without re-lowering. Keeps
  `CostManagerStats { hits, misses }` for the fusion report.
- `driver.rs` — added `estimator: EstimatorConfig`, `graph_symbols`,
  `artifact: ArtifactContext`, `cycle_quantum`, and
  `blackbox_hint_cycles` fields on `FusionOptionsV2`. The old
  placeholder cost model is replaced by a `KernelCostManager`
  instantiated once per `fuse_graph_v2` call. `FusionReportV2` now
  carries `cost_cache_hits`, `cost_cache_misses`, and
  `total_runtime_units` for reporting.

**New types:**
- `DeviceModel` (§12.1) with `DeviceModel::synthetic()` for unit
  tests — round numbers so tests can hand-check.
- `EstimatorConfig` (§12.1) wrapping the device model + tunables.
- `EstimateContext` — `graph_symbols` + `param_bindings` bindings.
- `KernelCostBreakdown` — full per-kernel breakdown (registers,
  occupancy, critical path, access aggregate, launch/latency/
  bandwidth/issue cycles). Not consumed by the extractor but stored on
  the `KernelCostManager` for reporting.
- `AccessAggregate` — sum of per-site `AccessEst` records.
- `OpLatencyTable` — per-op latency in cycles. Reasonable defaults
  for BabyBear/FpExt/U32; TODO calibrate.

**Tests (12 new — all passing):**

Determinism / sanity:
- `estimator_is_deterministic_across_calls` — two calls on the same
  module produce bit-identical cycles / registers / transaction
  bytes / sync counts.
- `synthetic_device_defaults_are_positive` — the synthetic profile
  isn't accidentally zero.

Cost comparisons:
- `fused_kernel_has_more_compute_than_single_step` — the two-mul
  fused module costs at least as much per launch as a single mul
  (fusion saves the intermediate materialization by removing a
  kernel — the extractor's runtime stage still prefers the fused
  version because there's one fewer launch).
- `larger_domain_costs_more_than_smaller` — 8192 vs 128 element
  domain: bigger domain costs strictly more.
- `cycle_quantum_scales_runtime_units` — same cycles at quantum=1 vs
  1000 gives fine > coarse `runtime_units`, coarse ≥ 1.
- `slower_dram_raises_memcpy_cost` — halving
  `memcpy_bytes_per_cycle` raises the memcpy runtime.

Occupancy:
- `higher_register_pressure_reduces_resident_blocks` — bumping
  `register_fixed_overhead` never increases resident blocks;
  `blocks_per_sm` is clamped to ≥ 1.

Cache:
- `cost_manager_caches_repeated_lookups` — same key -> 1 miss then 1
  hit.
- `cost_manager_keys_on_param_bindings` — different
  `param_bindings` miss independently despite same `module_hash`.

Non-kernel closed-form (§12.8):
- `non_kernel_const_costs_zero` — `Const` returns cost 1 after floor
  (raw 0.0 cycles → floor 1 unit per §13.5).
- `non_kernel_memcpy_costs_launch_plus_bandwidth` — matches the
  closed-form `memop_launch + bytes / memcpy_bytes_per_cycle`.
- `non_kernel_memset_costs_launch_plus_bandwidth` — same closed-form
  for memset.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo clippy -p crypto-compiler --lib --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **46**
  tests pass (34 pre-M4 + 12 new M4).
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — **50** tests pass (46 + 4 cpsat_agreement).
- `cargo nextest run -p crypto-compiler --lib` — **260** tests pass
  across the full crate.

**Design decisions:**
- The estimator uses `ModuleCompiler::lower` for the HIR→KIR path —
  no dedicated fusion-v2 lowering pipeline. That means the estimator
  runs `type_infer + canonicalize + lower_to_kir + layout_infer +
  insert_sync + plan_shared_mem`, matching what the compile pipeline
  eventually does. `estimate_kernel` requires exactly one KIR kernel;
  multi-kernel HIR is a `CompileError::Lower`.
- `OpLatencyTable` values are placeholders labeled as such (§12.7's
  three-tier launch overhead notes the same). Calibration lands with
  the benchmark JSON export (deferred out of M4 exit gate — the exit
  gate is "estimator is deterministic and returns ranking-consistent
  results on the golden benchmark suite", which the ranking tests
  cover for the toy chain).
- `KernelCostManager` deliberately keeps a separate `breakdowns`
  HashMap alongside the cost map so callers who want the full
  breakdown for reporting can retrieve it without re-costing; the
  extractor only touches `cache`. The breakdown map grows with cache
  misses only.
- Non-kernel `Const` returns cost 1 (runtime_units floor per §13.5)
  rather than 0. Zero would leak through the CP-SAT integer objective
  as no cost pressure at all; the floor keeps the ILP aware that the
  node exists.
- The transaction sampler treats every un-analyzable index expression
  (`SExpr`/`Blackbox`) as `unknown_global_sectors_per_warp` sectors.
  This is deliberately pessimistic — the alternative is running the
  index expression through symbolic evaluation, which the current
  `IndexMap` layer doesn't support without loading data-dependent
  values.
- `estimate_non_kernel` returns `launch_cycles_within_wave` for
  `Kernel` variants as a safety default; callers should route kernel
  nodes through `estimate_kernel` via the manager. The extractor and
  the driver both do that.
- `graph_symbols` on `FusionOptionsV2` is currently populated by the
  caller. When M11 wires v2 into `GraphCompiler`, the graph-symbol
  environment already available on `GraphCompiler::env` will be
  threaded through, matching plan §3's `FusionContextV2` sketch.

**Not landed in M4** (still on the plan for later milestones):
- Benchmark JSON export and calibration harness (§12.9): the estimator's
  numerical constants are placeholders. Exit gate for M4 requires
  determinism and ranking sanity, both of which are covered.
- Queueing correction (§12.7 second half) — disabled by default;
  requires measured rank correlation before turning on.
- Shared-memory bank-conflict sampling (§12.5) — a later estimator
  version.
- Interpreter access to `&[BufferDecl]` so it can distinguish
  shared/register loads from global (currently treats every access as
  global, biasing latency conservatively). Not required for M4 exit
  gate but should land alongside M9 horizontal fusion which exercises
  register-resident computations.

### M5 (session 8): keep-seam variants

Extends the producer-consumer pass with the §10.2 keep variant. Under
keep, the fused kernel materializes the seam value as a top-level output
alongside the consumer's outputs, so downstream nodes that still need
the seam do not force the original producer to run.

**Files:**

- `passes/fusion_v2/fusions/producer_consumer.rs`:
  - Added `FusionVariant { Drop, Keep }` and a `variant` field on
    `CandidateDraft`.
  - Extended `synthesize_producer_consumer` with a `variant` parameter.
    For `Keep`, the fused compute body wraps the drop-variant body in a
    `Node::Tuple([consumer_body, seam_body])` — the seam body is a
    fresh clone of the producer body with `producer.outer_var →
    k_var_node` (identity access at the materialized index).
  - `fused_outputs` for `Keep` = `consumer.outputs ++ [seam_val]`.
    `insert_candidate` then registers the fused node as a producer of
    the seam value class alongside the original producer, so the ILP
    sees keep as a valid single-alternative producer of both.
  - Renamed the fused-module name suffix to `_drop` / `_keep` so keep and
    drop candidates hash to distinct modules (and thus distinct
    `ArtifactKey`s).
  - Added `EnumerateOptions::enable_all_keep_variants` for testing.
  - `enumerate(gf, options)` now takes an options struct; `should_emit_keep`
    encodes §10.2's trigger conditions (seam is graph output, seam has
    another seed consumer, or diagnostic override).

- `passes/fusion_v2/driver.rs`:
  - Added `FusionOptionsV2::enable_keep_variants` (default `true`) and
    `enable_all_keep_variants` (default `false`).
  - `fuse_graph_v2` passes the options through and, when
    `enable_keep_variants == false`, filters keep drafts out post-
    enumeration.

**Tests (10 new — all passing):**

- `enumerate_emits_keep_variant_when_seam_is_graph_output` — seam is a
  registered output → 1 drop + 1 keep.
- `enumerate_skips_keep_when_seam_has_single_consumer` — single-consumer
  seam, not a graph output → 1 drop only.
- `enable_all_keep_variants_emits_keep_on_single_consumer` — the
  diagnostic flag forces keep emission for every legal drop.
- `keep_variant_outputs_include_seam_value` — verifies output layout
  `[consumer_output, seam_output]`.
- `keep_variant_module_hash_matches_hand_authored_reference` — the
  synthesized fused module for `y = 2*x; z = 3*y` (y is graph output)
  hashes byte-for-byte identical to a hand-authored
  `compute[N] |i| Tuple([3 * (2 * x[i]), 2 * x[i]])`.
- `keep_variant_type_checks` — synthesized module passes `type_infer`.
- `extractor_picks_keep_over_original_chain_when_seam_is_graph_output` —
  end-to-end: enumerate → insert → brute-force extract → apply produces
  a single-node graph (the keep kernel writes both `y` and `z`).
- `extractor_prefers_keep_over_drop_plus_original_producer_on_fanout` —
  fanout: 2 drop + 2 keep candidates; extractor selects ≤ 2 nodes.
- `keep_variant_cost_is_priced_by_estimator` — the M4 estimator prices
  the keep variant at least as much as the drop variant (extra tuple
  element + store adds compute; not less).
- `driver_disable_keep_variants_leaves_seam_needing_original_producer` —
  driver flag disables keep emission entirely.

Updated 2 existing M3 tests to reflect the new candidate counts:
- `driver_enumerates_two_candidates_when_producer_feeds_two_consumers`
  now expects 4 candidates (2 drop + 2 keep because seam has another
  consumer at each site).
- `driver_leaves_registered_output_intact_when_seam_is_graph_output`
  now expects 2 candidates (drop + keep triggered by graph-output seam).

**Verification:**

- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo clippy -p crypto-compiler --lib --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **56** tests
  pass (46 pre-M5 + 10 new M5).
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — **60** tests pass (56 + 4 cpsat_agreement).

**Design decisions:**

- Chose `Node::Tuple([consumer_body, seam_body])` as the keep-variant
  body rather than two top-level `Compute`s. A tuple-body `Compute` is
  canonicalized into one kernel with two output tensors sharing the
  outer index; two top-level computes would produce two separate
  kernels (the canonicalize `walk_top`/`resolve_or_emit` path treats
  each `Compute` as a distinct kernel). We want one launch, not two —
  that's the whole point of keep over "drop + original producer".
- Kept `variant` on `CandidateDraft` as a diagnostic tag. The ILP does
  not see it; it drives only enumeration filtering (`enable_keep_variants`)
  and the fused module's name suffix. Ties in cost between drop and
  keep still resolve deterministically because the module hashes
  differ (name and body structure).
- Ordered `outputs = [consumer_output, seam_output]` (consumer first,
  seam appended). Alternative would have been seam-first — but
  consumer-first keeps the drop-variant boundary as a prefix of the
  keep-variant boundary, which makes debugging and downstream code
  reading easier (the drop and keep outputs are ordered consistently).
- The keep-variant module's name is `fused_{p}_{c}_keep` vs the drop's
  `fused_{p}_{c}_drop`. Because the module hash is affected by module
  name, the two variants get distinct `ArtifactKey::module_hash`
  values and consume separate `z_m` variables in the ILP. That is
  correct: they're distinct compiled artifacts.
- `enable_keep_variants: bool` on `FusionOptionsV2` is a coarse gate
  (§15 default `true`). `enable_all_keep_variants` is a diagnostic
  finer knob that overrides `should_emit_keep`; it is off by default
  because emitting keep for every drop candidate inflates enumeration
  on the common single-consumer case where keep is strictly worse.
- The keep variant reuses the same seam-substitution hook as the drop
  variant. That means all M3 shapes (identity, affine permutation,
  nested-index consumer, reduction producer) automatically get their
  keep-variant siblings for free — the tuple-body wrapper is
  orthogonal to how the consumer body is rewritten.

**Not landed in M5** (still on the plan for later milestones):

- Producer-consumer keep with the seam having a *different* domain
  than the consumer's output (would require a second launch inside
  one HIR kernel; deferred per §10.2 preconditions).
- Fanout keep — the fanout pass (M7) will emit its own keep variant
  that materializes the seam and consumes it in each fanout arm; the
  M5 keep here handles the simpler two-node case only.
- Deduplication of drop+keep candidates by `CandidateKey` (§9). Right
  now each variant produces a distinct module hash so structural
  dedup is a no-op, but §9's normalization pipeline (canonicalize +
  monomorphize + boundary pruning) has not landed and will refine the
  key set once M6/M11 come in.

### M6 (session 9): bounded saturation and chain composition

Extends the driver from single-pass to a multi-round saturation loop
(§11) and adds the sidecar bookkeeping needed to compose fused
candidates across rounds without exploding enumeration.

**Files:**

- `passes/fusion_v2/saturate.rs` (new) — `SaturationState` sidecar
  and `CandidateKey` type (§5.4, §9). `SaturationState::origins` is
  a dense `Vec<BTreeSet<NodeId>>` indexed by `NodeId`; seeds carry
  singleton origins, fused candidates get the union of their parents'
  origins via `register_origins`. `origins_disjoint` is the §9.1
  composition-legality check the enumerator applies before
  synthesizing a candidate. `seen_candidates: HashSet<CandidateKey>`
  is the cross-round dedup set; `note_seen` returns whether the key
  was fresh.

- `passes/fusion_v2/fusions/producer_consumer.rs`:
  - Added `EnumerateContext { frozen_node_count, origins,
    min_new_parent_id, options }`. Enumeration walks only the frozen
    prefix and rejects any pair whose parents' origins overlap; the
    `min_new_parent_id` watermark skips pairs both of whose parents
    predate the previous round, so the same `(A, B)` pair isn't
    re-emitted every round (the `CandidateKey` dedup would catch it
    anyway, but this avoids the wasted synthesis work).
  - Added `OwnedEnumerateContext::all_seed` — a test-facing owned
    wrapper that treats every node in `gf` as its own seed origin
    with `min_new_parent_id = 0`. Existing tests call it via
    `ctx.as_ref()`.
  - Removed the `producers.len() == 1` filter and instead iterates
    every producer of each value class (§10.1). Once drop candidates
    land, a value class typically has two producers (the original
    seed plus the drop candidate that materializes it as its
    consumer output) — the old filter silently killed chain
    composition through that value.
  - Renamed synthesized fused modules to a canonical
    `fused_drop` / `fused_keep` (removed producer/consumer names
    from the module name). `module_hash` includes the module name,
    so composition-order variants like `(A+B)+C` vs `A+(B+C)` now
    hash byte-identical and collide at `CandidateKey` dedup.

- `passes/fusion_v2/driver.rs`:
  - Added a saturation loop up to `max_rounds` (default 4, §11). Per
    round: freeze `gf.nodes.len()`, enumerate the enabled passes over
    that frozen prefix, dedup by `CandidateKey`, validate acyclicity,
    insert. Advance `min_new_parent_id` to the frozen count so the
    next round only enumerates pairs involving newly-inserted nodes.
    Break as soon as a round inserts zero candidates.
  - Added new `FusionOptionsV2` fields: `max_rounds`,
    `max_alternatives_per_pass_per_round` (soft cap, `0` disables).
  - Added new `FusionReportV2` fields: `candidates_rejected_dedup`,
    `candidates_rejected_pass_cap`, `rounds_run`, `rounds_inserted`,
    `max_rounds_hit`.
  - `candidate_key(draft, &artifact_ctx)` computes the CandidateKey
    from a draft's kernel module hash.

**Tests (7 new — 3 unit tests inside `saturate` + 7 in the
`saturation_tests` submodule; all passing):**

Saturate unit tests:
- `seeds_have_singleton_origins`
- `disjoint_origins_check_catches_overlap`
- `note_seen_deduplicates`

Saturation-driver tests (in `passes::fusion_v2::tests::saturation_tests`):
- `three_kernel_chain_collapses_to_one_kernel` — the exit-gate
  fixture: 3-kernel scale chain → single fused kernel via multi-round
  composition. Asserts `rounds_run >= 2`.
- `association_order_dedup_across_rounds` — verifies that both
  `(A+B)+C` and `A+(B+C)` composition paths are enumerated in round 2
  and at least one is rejected by `CandidateKey` dedup.
- `saturation_is_deterministic_across_runs` — repeated runs produce
  identical reports (candidate counts, per-round counts) and
  identical emitted graph fingerprints.
- `max_rounds_one_prevents_chain_composition` — capping to one round
  leaves at least 2 nodes.
- `saturation_terminates_at_fixpoint_before_max_rounds` — with
  `max_rounds = 8`, the loop stops at the natural fixpoint and
  `max_rounds_hit == false`.
- `per_pass_cap_truncates_and_reports` — setting
  `max_alternatives_per_pass_per_round = 1` on a 4-candidate fanout
  reports 3 rejections in `candidates_rejected_pass_cap`.
- `overlapping_origins_prevent_re_fusion` — origin overlap prunes
  `(fused(A,B), fused(B,C))` composition on the {B} overlap; the
  chain still saturates via non-overlapping paths.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo clippy -p crypto-compiler --lib --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **66**
  tests pass (56 pre-M6 + 7 saturation-driver + 3 saturate
  unit tests).
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — **70** tests pass.

**Design decisions:**

- `SaturationState` is a driver-owned sidecar, not a member of
  `GraphFuser` (§5.4). That keeps the alternative-graph arenas
  independent of search bookkeeping — future estimator revisions or
  search strategies can mutate origins/seen without touching
  `gf.nodes`.
- `origins` is dense (a `Vec<BTreeSet<NodeId>>` grown in lockstep
  with `gf.nodes`) rather than a `HashMap`. Every insertion goes
  through `register_origins(NodeId(n), parents)` which asserts
  `NodeId(n) == origins.len()`. This is much cheaper than a hashmap
  for the tight enumeration inner loop and gives us the same
  invariant.
- `min_new_parent_id` is a per-round watermark: pairs `(p, c)` where
  both `p.0 < min_new_parent_id` AND `c.0 < min_new_parent_id` are
  skipped in enumeration because they were already emitted in a
  previous round. The saturation loop advances the watermark to the
  round's frozen count at the end of every round. Without this, the
  round-2 enumerator would re-emit every round-1 pair and rely
  entirely on `CandidateKey` dedup — correct but wasteful.
- Chose to iterate *all* producers of a value class rather than only
  values with a single producer. Once drop candidates land, a value
  has two producers (seed + drop-candidate-that-outputs-it); the
  single-producer filter silently blocked chain composition through
  that value. The new loop considers each `(producer, consumer)`
  pair independently, deduping via `CandidateKey` afterward.
- Renamed synthesized fused modules to `fused_drop` / `fused_keep`
  (removed producer/consumer names). `module_hash` hashes the module
  name, so the pre-M6 name suffixes made `(A+B)+C` and `A+(B+C)`
  hash to different values even though their bodies were
  byte-identical after α-normalization. Loss: fewer human-readable
  module names in dumps. Gain: real dedup — CandidateKey collides
  across composition orders. Debug names can be recovered from the
  `FusionHistory` metadata when M11 lifts it into the report.
- `min_new_parent_id` also prevents an infinite composition loop —
  once no new nodes are inserted, the enumerator's minimum-parent
  filter kicks in and every pair is skipped. Combined with the
  "break if zero inserted" check, the loop always terminates.
- `candidates_rejected_dedup` is a diagnostic counter — dedup
  rejections are expected under chain composition and don't
  represent a problem, but the counter helps distinguish "the
  saturation reached a fixpoint" from "we're just re-enumerating
  the same pairs".
- The plan's §11 step 8 (boundary-local pruning) has not landed in
  M6. That's an efficiency optimization — safe dominance can drop
  provably-worse candidates before insertion — and doesn't affect
  correctness. It lands with M11 integration.

**Not landed in M6** (still on the plan for later milestones):

- Boundary-local pruning (§11 step 8) — safe Pareto dominance
  reduces the enumerator's output before insertion but requires the
  full candidate-finalization pipeline (§9) that lands with M11.
- Pattern-key bucketing (§10.0) — the enumerator currently emits
  candidates in seed-NodeId order without pattern-key grouping.
  Sufficient for producer-consumer where every match is one
  `(producer, consumer)` pair; matters for fanout (M7) and
  small-kernel (M8) where instance counts per pattern drive the
  amortization argument.
- `SaturationState::origins` in `FusionReportV2` — the report
  doesn't currently expose the union of origins per selected node.
  M11 will surface this in the debug dump.

## Design decisions

- Kept `fusion_utils` as a single file per §4 rather than a submodule.
- `VisitError<E>` wraps visitor errors so `visit_hir` can also report
  `MalformedHir` without polluting every visitor's error type.
- Collectors use `HashSet<NodeId>` for reachable dedup; deterministic external
  API surfaces (`InputUse`, `StructureFacts`) are `Vec` in visitation order.
- Cloning is idempotent within one call via a per-call `memo` on the
  destination NodeIds; combined with `dst.intern`'s hash-consing, cloning the
  same body twice yields the same root NodeId when no fresh bindings are
  introduced under distinct calls.

## Deferred conventions

- `AccessRelation` (per §8) will live in `passes/fusion_v2/access.rs` with
  fields typed on `ValueClassId`. `AccessCollector` in `fusion_utils.rs` will
  return a raw form parameterized only on `ir::NodeId`; the pass binds those to
  logical values after `AccessCollector` returns.
- `BoundaryBuilder` will be introduced in M1's `fusion_v2` model (§9),
  parameterized on `ValueClassId` rather than being generic.
