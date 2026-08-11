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
| M7 | ✅ Done | 10 | Fanout pass (§10.5): k≥2-consumer fanout with hash-consed shared producer expression, drop and keep variants, identity-access-only seam reads, consumer-to-consumer dataflow rejection |
| M8 | ✅ Done (first slice) | 11 | Small-kernel block fusion (§10.7): linear chain of concrete-bound kernels fuses via let-bound inner-compute tiles. Handles **different domain sizes** unlike M3/M7. Uses the existing DSL's inner_let shared-memory tile lowering. Only linear chains for now (§10.7's parallel-siblings-within-layer deferred). |
| M9 | ✅ Done | 12 | Same-domain horizontal fusion (§10.6): pairwise merge of dataflow-independent equal-domain flat kernels into one multi-output kernel; three-way merges compose across saturation rounds via Tuple splicing. No `compute[max]` masking; concrete bounds only. |
| M10 | ✅ Done | 13 | Epilogue fusion (§10.4): flat pointwise consumer substituted into the producer's result path; producer schedule (bound, `par`, `threads`, block hint) retained verbatim. Identity seam reads only; producers already covered by producer-consumer are skipped in enumeration. |
| M11 | ✅ Done | 14 | Opt-in `GraphCompiler` integration (§16): internal `FusionStrategy` enum, `fusion_v2_options` setter, env→`graph_symbols` threading, v2 report embedded in `FusionReport.v2` (§15), `verbose` saturation/extraction dump. Module-count and estimated-runtime comparisons CPU-side; measured-runtime/compile-time comparisons land with M12's `dsl_port_tests` replay per plan. |
| M12 | ✅ Done | 15 | Numerical accuracy on fractional_sumcheck: 8/8 `dsl_port_tests` fixtures bit-for-bit vs eager under `FRAC_DSL_FUSION=v2` (CP-SAT Optimal, no fallback). Two estimator bugs fixed (block-hint stamping, param threading in the transaction sampler). CP-SAT seed-solution hints unblock the 7.6k-var bench model. Perf compared v1 vs v2 at n=2^16. |

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
| 10 | M7 | 10 | 76 total fusion_v2 lib tests (80 with `planner-ortools`); k=2 & k=3 fanout, hand-authored HIR match, extractor prefers fanout, consumer-to-consumer rejection |
| 11 | M8 | 10 | 86 total fusion_v2 lib tests (90 with `planner-ortools`); same-domain and different-domain chains, 3-kernel chain, symbolic-bound rejection, branching-intermediate rejection, shared-mem budget, driver end-to-end, coexistence with producer-consumer |
| 12 | M9 | 12 | 100 total fusion_v2 lib tests (104 with `planner-ortools`); independent-pair fusion, shared-input hash-consing vs hand-authored reference, dataflow/transitive-dataflow/domain/symbolic/non-flat/WAW/WAR/block-hint rejections, multi-output Tuple splicing, driver end-to-end, multi-round three-way composition |
| 13 | M10 | 11 | 111 total fusion_v2 lib tests (115 with `planner-ortools`); driver end-to-end block-hinted reduction + pointwise (exit gate), hand-authored HIR reference match, `threads`/`par` retention, keep variant on seam output, producer-consumer coverage skip, identity/flat/bound/hinted-consumer/tuple-producer rejections |
| 14 | M11 | 6 | 117 total fusion_v2 lib tests (120 with `planner-ortools`); GraphCompiler v2-strategy end-to-end fuse with report embedding, v1-default-unchanged, `without_fusion` disables both, module-count parity v2 vs v1 vs unfused, env→`graph_symbols` threading observable via symbolic memcpy estimate, `SolverUnavailable` fallback beyond brute-force cap (non-ortools only) |
| 15 | M12 | 1 | 118 total fusion_v2 lib tests (121 with `planner-ortools`); symbolic-outer-bound costing via stamped block hint. GPU oracle: 8/8 `dsl_port_tests` fixtures bit-for-bit identical to eager under `FRAC_DSL_FUSION=v2` (and 8/8 under v1 baseline); full cuda-backend suite 411/413 (2 pre-existing failures reproduce on clean HEAD `3650dc5b`) |
| 16 | M12 follow-up | 0 | 118 total fusion_v2 lib tests (121 with `planner-ortools`) unchanged; parallel enumeration is draft-order-identical to sequential, sentinel-exclusion covered by existing CP-SAT/brute agreement tests |
| 17 | solver workers + LOG_N=24 nsys | 0 | 121 lib tests pass post-rebase onto `feat/stream-scheduler`; `solver_num_workers` plumbed (default 1 = deterministic per §2.4); nsys LOG_N=24: v2+capture within 7.7% of eager |

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

### M7 (session 10): fanout pass

Lands the second fusion pass — fanout targets a producer whose seam
value is read by two or more consumers, materializing the producer
expression once (via HIR hash-consing) and threading its NodeId into
every consumer body.

**Files:**

- `passes/fusion_v2/fusions/fanout.rs` (new) — `enumerate(gf, ctx)`,
  `synthesize_fanout(gf, producer, consumers, seam, variant)`, and
  `FanoutFailure`. Iterates producers in the frozen prefix; for each
  seam value collects consumers within the frozen prefix; checks
  origin-disjointness of the full group and rejects if any consumer
  reads another consumer's output. Emits one drop candidate plus
  (§10.2) a keep candidate when the seam has other users. Uses the
  `clone_expr_with_hook` machinery from `fusion_utils` — the hook
  returns the shared producer body NodeId at every seam-read site,
  and hash-consing collapses all references to that single instance.

- `passes/fusion_v2/fusions/producer_consumer.rs` — exposed
  `KernelShape`, `ReadSite`, and `identify_kernel_shape` as
  `pub(super)` so `fanout.rs` reuses the same shape recognizer as
  producer-consumer. No behavior change to producer-consumer.

- `passes/fusion_v2/fusions/mod.rs` — declared `pub mod fanout;`.

- `passes/fusion_v2/driver.rs`:
  - Added `FusionOptionsV2::enable_fanout` (default `true`, §15).
  - Per-round loop calls `fanout::enumerate` after `producer_consumer::enumerate`; the two lists
    concatenate before the cap/dedup pipeline, so `CandidateKey` dedup handles any collision
    between a fanout candidate and a producer-consumer keep candidate on the same boundary.

**Tests (10 new — all passing):**

Enumeration:
- `fanout_k_equals_2_emits_one_drop_candidate` — one drop candidate,
  parents = producer + 2 consumers.
- `fanout_k_equals_3_emits_one_drop_candidate` — one drop candidate,
  parents = 1 + 3 = 4.

HIR shape:
- `fanout_module_hash_matches_hand_authored_reference` — synthesized
  fanout body hashes byte-identical to a hand-authored
  `compute[N] |i| Tuple([3 * (2*a[i]), 5 * (2*a[i])])` in which the
  Rust bindings share `b.mul(ai, two)` at both call sites. Because
  IRBuilder hash-conses, the shared `2*a[i]` sub-expression is one
  NodeId used twice — matching the fanout body's single-instance
  invariant.
- `fanout_body_is_a_tuple_at_the_compute_root` — the fused compute's
  body is a `Node::Tuple` over `k` elements.
- `fanout_module_type_checks` — `passes::type_infer` accepts the
  synthesized module.

Extractor behavior:
- `extractor_prefers_fanout_over_duplicated_producer_consumer_candidates` — end-to-end via
  `fuse_graph_v2`: the two-consumer fanout collapses to a single fused kernel.
- `fanout_drop_apply_produces_single_node_graph` — enumerate → insert → brute-force extract → apply
  produces a single-node graph.

Legality rejections:
- `fanout_rejects_when_a_consumer_shape_is_unsupported` — one
  consumer has a `#[grid(threads = N)]` hint; `identify_kernel_shape`
  returns `None` and the fanout group is rejected.
- `fanout_rejects_consumer_reads_another_consumer_output` — a
  `y = 2*x; z1 = 3*y; z2 = 4*z1` chain has only one consumer of
  `y` (`z1`) since `z2` reads `z1` not `y`; the enumerator emits
  zero fanout candidates.

Driver flag:
- `disable_fanout_flag_suppresses_fanout_candidates` — with
  `enable_fanout = false` and `enable_keep_variants = false`, only
  producer-consumer drops are emitted.

Updated existing tests that count candidates on the fanout fixture:
- `driver_enumerates_two_candidates_when_producer_feeds_two_consumers` — added
  `enable_fanout: false` so the assertion still counts only producer-consumer candidates.
- `driver_disable_keep_variants_leaves_seam_needing_original_producer` — same isolation.
- `saturation_tests::per_pass_cap_truncates_and_reports` — now expects 5 generated
  candidates (2 pc-drops + 2 pc-keeps + 1 fanout-drop) and 4 pass-cap rejections at
  `max_alternatives_per_pass_per_round = 1`.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo clippy -p crypto-compiler --lib --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **76**
  tests pass (66 pre-M7 + 10 new M7).
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — **80** tests pass.

**Design decisions:**

- Chose hash-consing over an explicit `Node::Let` for the shared
  seam expression. Both give the "compute once" property downstream,
  but the hand-authored reference (using Rust-level `let seam = ...`)
  matches the hash-consing approach byte-for-byte via `module_hash`.
  An explicit `Node::Let` in the fused body would still lower
  correctly (canonicalize's `peel_body_lets` inlines scalar-typed
  Lets), but its `module_hash` differs from the reference — which
  makes the anti-pattern rejection test harder to spec cleanly.
- Restricted M7 to producers with a single output value. Multi-output
  producers (e.g. M5 keep-variant kernels) become fanout producers
  in later saturation rounds via chain composition, but a
  multi-seam-in-one-shot fanout would need per-output analysis. That
  extension lands with M8/M11 rather than complicating M7.
- Restricted seam reads to identity access. Any consumer with an
  affine-permuted read of the seam falls back to producer-consumer
  fusion for that individual `(p, c)` pair. This is stricter than
  the plan text (which allows affine permutation), but preserves the
  "compute once" invariant unconditionally — with a non-identity
  permutation the producer would need re-evaluation at every distinct
  index.
- The consumer-to-consumer dataflow check is a **direct dependency**
  check only: consumer `i`'s outputs must not appear as consumer
  `j`'s inputs. Transitive dependencies through non-fanout nodes are
  fine — the fanout candidate still safely computes the producer
  once and threads it into the fanout consumers. Full transitive
  reachability analysis is deferred until §14.3 storage-hazard
  ordering (where it becomes an emission-order concern rather than
  a legality one).
- Fanout drop and keep both use the fused module name
  `fanout_drop` / `fanout_keep` — same rationale as M6's canonical
  producer-consumer names, giving cross-derivation `CandidateKey`
  dedup.
- Fanout produces a single candidate per `(producer, consumer set)`
  grouping — we don't enumerate subsets. A producer with 3 consumers
  emits *one* 3-way fanout, not `C(3,2) = 3` 2-way fanouts plus one
  3-way. Selecting a subset would strictly worsen the extractor's
  best case: fewer consumers means more launches. If a legality
  problem excludes a specific consumer, the whole group is rejected
  and the individual `(p, c)` producer-consumer candidates carry
  the load.

**Not landed in M7** (still on the plan for later milestones):

- Fanout keep with the seam feeding a chain of downstream users
  (the M5 keep-variant construction handles the shape but M7 doesn't
  drive it into the fanout enumeration flow explicitly).
- Multi-output producers as fanout roots — deferred to when
  multi-output fanout patterns land alongside the M11 golden suite.
- Affine-permuted seam reads inside fanout — requires per-site
  producer re-evaluation and a proof that the total work is still
  cheaper than duplicating producer-consumer candidates; not on the
  critical path for the M11 golden suite.
- `FusionHistory` n-ary variant + dump serialization (per §M7 exit
  gate) — the current binary-fusion variant of `FusionHistory` in
  `passes/fusion.rs` still fits producer-consumer; extending it for
  n-ary fanout output lands with M11 integration when dumps become
  externally visible.

### M8 (session 11): small-kernel block fusion (first slice)

Lands the third fusion pass. Small-kernel fusion collapses a **linear
chain** of concrete-bound pure kernels into a single fused kernel that
routes each intermediate seam through a let-bound inner-compute tile
(the DSL's existing shared-memory-tile pattern).

**Key distinction vs M3/M7:** small-kernel is the first pass that
handles **different domain sizes** across the chain. Each source
kernel keeps its own iteration count; each becomes an inner-let
compute at its own domain. The fused kernel's outer compute takes the
last kernel's domain; the block is sized to the maximum tile bound by
`lower_to_kir`'s `max_par` policy.

**Files:**

- `passes/fusion_v2/fusions/small_kernel.rs` (new) — `enumerate(gf, ctx, options)`,
  `synthesize_small_kernel(gf, chain, variant, options)`, `identify_chain`,
  `SmallKernelOptions`, `SmallKernelFailure`. Identifies maximal linear chains of
  concrete-bound `KernelShape`-recognizable kernels; synthesizes the fused module
  with nested `Let { tile_i = Compute[N_i] |j| ...; ... }` bindings inside a
  top-level `Compute[N_L]`. Rejects any chain whose combined tile bytes exceed
  `SmallKernelOptions::max_shared_bytes`. Runs `type_infer` + `canonicalize`
  on the synthesized module as a sanity gate before returning the draft.

- `passes/fusion_v2/fusions/mod.rs` — declared `pub mod small_kernel`.

- `passes/fusion_v2/driver.rs`:
  - Added `enable_small_kernel` (default `true`), `small_kernel_shared_bytes` (48 KiB), and
    `small_kernel_max_chain` (6) fields to `FusionOptionsV2`.
  - Per-round loop calls `small_kernel::enumerate` after `fanout::enumerate`; drafts flow through
    the same `CandidateKey` dedup and cap pipeline.

- `passes/fusion_v2/cost/cache.rs` — wraps `estimate_kernel` in
  `std::panic::catch_unwind` and adds
  `CostError::LoweringPanicked` for the case where a debug-assertion
  in `lower_to_kir` fires on a synthesized module. Driver treats this
  as an infinite-cost fallback (`GraphNodeCost::new(i64::MAX / 4)`), so
  the extractor never picks a broken candidate.

**Tests (10 new — all passing):**

Enumeration:
- `two_kernel_same_domain_chain_fuses` — baseline 2-kernel chain.
- `two_kernel_different_domain_chain_fuses` — M8's headline case:
  `scale(N=16)` → `take_half(→N=8)` chain fuses.
- `three_kernel_chain_fuses` — 3-kernel chain emits a 3-parent
  candidate.

HIR / lowering:
- `fused_module_type_checks_and_lowers` — the synthesized module
  passes `type_infer`.

Legality rejections:
- `rejects_symbolic_bounds` — chain with a symbolic outer bound is
  skipped (M8 requires all bounds constant per user request).
- `rejects_branching_intermediate` — an intermediate kernel with two
  downstream consumers falls out of the linear-chain requirement and
  no candidate emits.
- `rejects_when_shared_mem_budget_exceeded` — chain whose combined
  tile bytes exceed `max_shared_bytes` is rejected before synthesis.

Driver:
- `driver_end_to_end_fuses_two_kernel_chain` — 2-kernel chain fuses
  with M8 alone (producer_consumer/fanout off).
- `small_kernel_and_producer_consumer_coexist` — 2-kernel chain
  emits both a producer-consumer drop and a small-kernel candidate;
  extractor picks whichever wins on cost.
- `three_kernel_chain_with_small_kernel_and_producer_consumer` —
  regression guard for the 3-chain + M8 + multi-round path that
  previously panicked in `lower_to_kir`.

Updated 5 existing tests to isolate producer-consumer counting via
`enable_small_kernel: false`. One saturation test
(`max_rounds_one_prevents_chain_composition`) explicitly disables M8
because M8's single-round chain candidate collapses the 3-chain in
one round.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --lib --tests -- -D warnings` clean.
- `cargo clippy -p crypto-compiler --lib --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **86**
  tests pass (76 pre-M8 + 10 new M8).
- `cargo nextest run -p crypto-compiler --lib fusion_v2 --features
  planner-ortools` — **90** tests pass.

**Design decisions:**

- **Chose the DSL's existing inner-let tile pattern** rather than the
  plan's §10.7 layered `compute[B_l]` + layer-boundary-sync
  structure. The plan's ideal shape requires HIR extensions (multiple
  top-level computes in one kernel, explicit inter-compute syncs) not
  currently supported. The tile approach compiles through the same
  DSL surface as existing kernels — no compiler changes required —
  and preserves the launch-overhead saving because each tile is
  computed once per block in shared memory rather than per outer
  iteration.
- **Grid launches with `outer_bound = N_L` blocks, not `1`.** This
  differs from §10.7's pure "grid_dim = 1" vision. When `N_L` is
  small (typical for the fused pattern's use case), the launch fits
  on one SM and the launch-overhead saving vs `L` separate launches
  is still real. When `N_L` is large, the fusion still wins over
  separate launches because the L tile phases share the DRAM
  round-trip, but the per-block redundant tile recomputation is a
  real cost that the M4 estimator can price.
- **Linear chains only in the M8 first slice.** Parallel siblings
  within a layer (§10.7's `B_l = Σ it(k)` case where multiple kernels
  at the same topological depth get merged via if/else dispatch)
  require the if/else HIR emission from §10.7. Deferred to a follow-
  up slice.
- **Concrete outer bounds required for every kernel in the chain.**
  This is the user's explicit request ("make sure that all
  block/grids are constant"). Symbolic bounds hit
  `SizeExpr::as_const() == None` and are rejected at chain
  identification.
- **Different domain sizes are the headline capability.** The tile
  representation naturally handles this: `tile_i = compute[N_i] |j|
  ...` allocates a shared-memory buffer of shape `[N_i]`; the outer
  compute of shape `[N_L]` reads any subset of a tile's indices via
  ordinary `Index(Var(tile_i), [expr])` — non-identity access
  patterns are legal because the tile is a full shared-memory
  allocation of the producer's shape.
- **The shared-mem budget check happens before synthesis.** For a
  chain that would exceed `max_shared_bytes`, `synthesize_small_kernel`
  returns `SmallKernelFailure::SharedMemoryBudgetExceeded` before
  running the expensive HIR clone. Callers can tune this via
  `SmallKernelOptions::max_shared_bytes`.
- **Post-synthesis canonicalize check.** After `type_infer`, we also
  run `passes::canonicalize` on the synthesized module. Any failure
  here (e.g. an unexpected let-binding shape) rejects the candidate
  before it enters the alternative graph. This is defensive — the
  estimator's `lower_to_kir` also runs canonicalize, but its
  `debug_assert!(is_canonicalized(program))` fires with a panic
  rather than a graceful error, so catching structural issues
  earlier keeps the driver panic-free.
- **`catch_unwind` in the cost cache.** As a belt-and-suspenders
  guard, `KernelCostManager::cost_of` wraps the estimator in
  `std::panic::catch_unwind`. If a synthesized module still slips
  past the pre-checks and panics inside `lower_to_kir`, the driver
  substitutes `GraphNodeCost::new(i64::MAX / 4)` as the fallback
  cost so the ILP never picks the broken candidate. This is the
  standard defense against synthesis bugs — the pass emits its own
  candidates and cannot rely on downstream passes catching every
  malformed shape.
- **`identify_chain` finds maximal chains from each starting node.**
  A 3-chain [A, B, C] emits both `[A, B, C]` (starting from A) and
  `[B, C]` (starting from B). Each is a valid candidate with
  distinct value-classes on its boundary; `CandidateKey` dedup
  would merge them only if they normalize to the same module hash.
  Since the chains differ in structure, they don't dedup.

**Not landed in M8** (deferred):

- Parallel siblings within a layer (§10.7's Σ-iteration if/else
  dispatch case) — requires either an HIR extension or an
  ir-level `Select` chain over disjoint ranges. Both are viable;
  deferred to a follow-up slice.
- Keep variants — M8 first slice only emits drop candidates. Keep
  would need to materialize every internal seam as an additional
  output, which requires either a `Tuple` at the module body (with
  the last-layer output plus each tile) or a scatter to route tile
  values into external buffers. The `Tuple`-at-body approach is
  straightforward but wasn't required for the M8 exit gate.
- §10.7's *grid_dim = 1* invariant. My current shape has
  `outer_bound = N_L` blocks. Achieving `grid_dim = 1` would require
  reshaping the fused module so the outer `compute[N_L]` becomes an
  inner-let too, and the module-level compute is `compute[1] |_|
  {...}`. Doable but requires more of the DSL's Tuple/Proj machinery
  than is currently exercised.
- The plan's shared-memory seam routing with `phi_s = inv(w_s) ∘
  r_s`. Instead I lean on the tile's full shape covering the seam
  domain, which handles identity, affine permutation, and any
  bounded index expression uniformly. Explicit `phi_s` composition
  will matter when we start emitting affine-permuted writes into
  smaller-than-source tiles (a shared-memory optimization).

### M9 (session 12): same-domain horizontal fusion

Lands the fourth fusion pass. Horizontal fusion merges two kernels
with **no dataflow relation** into a single kernel that executes both
bodies at the same logical index and returns the concatenated Tuple of
outputs. There is no seam, hence no drop/keep distinction — the draft
carries `FusionVariant::Drop` as a diagnostic placeholder.

**Scope:** pairwise-only per invocation. Larger groups compose across
saturation rounds: a horizontally-fused pair is a multi-output kernel
whose `Tuple` body elements are spliced positionally into the next
merge, so `{A,B}` + `C` yields a flat 3-element Tuple, not a nested
one. Disjoint-origins tracking (M6) prevents overlapping re-merges.

**Files:**

- `passes/fusion_v2/fusions/horizontal.rs` (new) — `enumerate(gf, ctx)`,
  `synthesize_horizontal(gf, a, b)`, `HorizontalFailure`. Enumeration prefilters per-node
  eligibility once per round (kernel, `identify_kernel_shape`, concrete outer bound, flat body,
  body/output-arity consistency, block hint), then walks unordered pairs `a < b` over the frozen
  prefix with the `min_new_parent_id` watermark and origin-disjointness checks.

- `passes/fusion_v2/fusions/producer_consumer.rs` — `find_input_nodes` and `remap_size_expr`
  visibility raised to `pub(super)` for reuse.

- `passes/fusion_v2/fusions/mod.rs` — declared `pub mod horizontal`.

- `passes/fusion_v2/driver.rs` — added `enable_horizontal` (default `true`, per §15) to
  `FusionOptionsV2`; per-round loop calls `horizontal::enumerate` after `small_kernel`. Drafts flow
  through the same `CandidateKey` dedup and cap pipeline.

**Legality (§10.6), all enforced in `synthesize_horizontal`:**

- **Equal concrete outer domain.** Symbolic bounds rejected
  (consistent with M8's all-bounds-constant requirement). No
  `compute[max(Na, Nb)]` masking — dense output lowering would write
  the smaller buffer out of bounds, which is exactly the M9 exit
  gate's "no shape-changing or out-of-bounds output" condition.
- **Equal block hint / thread geometry.** `identify_kernel_shape`
  already rejects `scatter`/`par`/`threads` on the outer compute, so
  geometry reduces to builder-level block-hint equality; the hint is
  propagated to the fused module.
- **Flat structured kernels.** No inner `Compute`/`Reduce` anywhere in
  either body (also excludes shared-memory tiles and their syncs).
- **No dataflow path in either direction.** Reuses the §9.1
  reachability primitive `would_create_cycle` on the union boundary:
  a path from `{a,b}`'s outputs to `{a,b}`'s inputs is exactly a path
  `a → b` or `b → a` (a node cannot reach its own inputs in a DAG).
  Catches transitive paths through any alternative, which matters
  because extraction has no acyclicity constraints (§13.7).
- **No storage hazard.** Physical `BufId` footprints must satisfy
  write∩write = ∅ and write∩read = ∅ in both directions. A fused node
  runs both regions concurrently, so cross-region WAW/WAR ordering
  cannot be recovered by the §7 hazard sort. Dataflow is checked
  first: a direct producer→consumer pair also overlaps on physical
  storage, and `DataflowPath` is the more precise diagnosis.
- **Disjoint origins** — enforced by the enumeration context.

**Synthesis:** merged param bindings with conflict rejection,
name-keyed param remap, stable-unique input boundary
(`stable_unique(a.inputs ++ b.inputs)`) with per-part position maps
and deterministic input-decl selection (ordered scan, not HashMap
iteration), one fresh outer var shared by both cloned bodies,
`type_infer` sanity gate. Shared loads dedup purely through
hash-consing — when both bodies read the same fused input at the same
index the `Index` node interns to one `NodeId` (verified by
module-hash equality with a hand-authored reference).

**Tests (12 new — all passing, `mod horizontal_tests`):**

Enumeration + HIR:
- `two_independent_kernels_fuse` — baseline pair; 2 inputs, 2 outputs,
  type-checks.
- `shared_input_is_deduped_and_hash_consed` — both kernels read the
  same `x`; fused boundary has 1 input and the module hash matches the
  hand-authored `compute[n] |k| Tuple(2*a[k], 3*a[k])` reference.
- `multi_output_parent_splices_tuple_elements` — a fused pair merged
  with a third kernel yields a flat 3-element Tuple matching the
  hand-authored triple reference.
- `matching_block_hint_propagates` — equal hints fuse and the hint
  lands on the fused module.

Legality rejections:
- `rejects_dataflow_pair`, `rejects_transitive_dataflow_pair` —
  direct edge and 2-hop path both yield `DataflowPath`.
- `rejects_different_domains`, `rejects_symbolic_bounds` — bound
  mismatch / non-constant bound.
- `rejects_non_flat_kernels` — inner `Reduce` yields `NotFlat`.
- `rejects_waw_hazard`, `rejects_war_hazard` — overwrite pattern and
  read-old-version/write-new-version pattern both yield
  `StorageHazard`.
- `rejects_block_hint_mismatch`.

Driver:
- `driver_end_to_end_fuses_independent_kernels` — horizontal alone
  collapses an independent pair to one node with outputs preserved.
- `horizontal_composes_across_rounds` — three independent kernels
  merge to a single node across ≥2 saturation rounds (exercises the
  multi-output splice path under the driver).

Updated 6 existing driver tests with `enable_horizontal: false` where
the fixtures contain dataflow-independent same-domain pairs (fanout
arms, disjoint kernels) that would otherwise change
`candidates_generated` counts.

**Verification:**
- `cargo check -p crypto-compiler --lib` clean.
- `cargo clippy -p crypto-compiler --all-targets --tests -- -D warnings` clean
  (also fixed one pre-existing `needless_range_loop` in `benches/poseidon2.rs`).
- `cargo clippy -p crypto-compiler --all-targets --tests --features
  planner-ortools -- -D warnings` clean.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib -E 'test(fusion_v2)'` —
  **100** tests pass (88 pre-M9 + 12 new).
- Same with `--features planner-ortools` — **104** tests pass.
- Full `cargo nextest run -p crypto-compiler --lib` — 314 tests pass
  (horizontal enabled by default did not regress anything outside
  fusion_v2).

**Design decisions:**

- **Pairwise enumeration only.** §10.6's k-way groups fall out of
  bounded saturation instead of a dedicated grouping search: round 1
  emits pairs, round 2 merges a pair-node with a third kernel, etc.
  This keeps the pass O(n²) per round and reuses the M6 dedup/caps
  machinery for the combinatorial control §11 prescribes.
- **`FusionVariant::Drop` as placeholder.** Horizontal has no seam to
  drop or keep. Reusing `Drop` avoids widening `FusionVariant` for a
  case with only one variant; the parents' outputs are all preserved
  on the fused node's boundary regardless.
- **Dataflow via `would_create_cycle` on the union boundary** rather
  than a bespoke pair-reachability walk — one primitive, already
  tested by the §9.1 validator suite.
- **Storage hazards on physical BufIds, not value classes.**
  Same-version RAW appears as a dataflow path; cross-version WAW/WAR
  (overwrite patterns) are invisible at the value-class level and
  must be caught on the physical footprint.
- **Estimator prices low-occupancy merges** (M9 exit gate second
  half): the fused kernel is lowered by the M4 estimator like any
  other candidate; no special-case pricing was added. The
  `catch_unwind` + infinite-cost fallback from M8 guards the lowering
  path.

**Not landed in M9** (deferred):

- Unequal-domain horizontal via masking — deliberately excluded by
  §10.6; different-size independent kernels remain separate launches
  (or fuse through M8 small-kernel when they form a chain).
- Reduction/non-flat horizontal merges — would need per-region thread
  geometry reconciliation.
- Grouped >2-way single-round enumeration — compositional rounds
  cover it; a dedicated grouping heuristic is only worth adding if
  round counts become a compile-time problem.
- The M4 deferral note suggested landing interpreter `&[BufferDecl]`
  access (shared/register vs global load classification) alongside
  M9. Not done — M9's flat kernels are global-load dominated, so the
  estimator's conservative all-global bias affects both the fused and
  unfused alternatives symmetrically. Revisit with the §12.9
  calibration harness.

### M10 (session 13): epilogue fusion

Lands the fifth fusion pass. Epilogue fusion is producer-consumer's
dual: instead of rebuilding the fused kernel on the consumer's
schedule, it retains the **producer's** top-level `Compute` verbatim —
bound, `par`, `threads`, and block hint — and substitutes the
consumer's pointwise expression into the result path right before the
store. This is the pass that serves the M10 exit gate: "reductions
followed by pointwise work retain the producer schedule."

**Files:**

- `passes/fusion_v2/fusions/epilogue.rs` (new) — `enumerate(gf, ctx)`,
  `synthesize_epilogue(gf, p, c, seam, variant)`, `EpilogueFailure`,
  `identify_epilogue_producer` (top-level `Compute`, no `scatter`, non-`Tuple` body; `par`/
  `threads` accepted and carried), `producer_consumer_covers` skip filter. Enumeration mirrors the
  producer-consumer seam loop (frozen prefix, `min_new_parent_id`, disjoint origins, §10.2
  keep-trigger via the shared `should_emit_keep`).
- `passes/fusion_v2/fusions/producer_consumer.rs` — `should_emit_keep` raised to `pub(super)`.
- `passes/fusion_v2/fusions/horizontal.rs` — `body_is_flat` raised to `pub(super)`.
- `passes/fusion_v2/fusions/mod.rs` — declared `pub mod epilogue`.
- `passes/fusion_v2/driver.rs` — added `enable_epilogue` (default `true`) to `FusionOptionsV2`;
  per-round loop calls `epilogue::enumerate` after `horizontal`, with the same
  `enable_keep_variants` / `enable_all_keep_variants` gating as producer-consumer.

**Legality (§10.4), enforced in `synthesize_epilogue`:**

- **Producer:** single-output kernel, top-level `Compute` with no
  `scatter` (partial/permuted writes break "consumer element k =
  f(producer element k)") and a non-`Tuple` body. `par`/`threads`/
  block hints are allowed — retaining them is the point. No read-site
  analysis: the producer body is cloned wholesale with `Input`-node
  substitution, so inputs may be used in any form, not only under
  `Index`.
- **Consumer:** single-output flat pointwise kernel — recognizable by
  `identify_kernel_shape` (which already rejects `scatter`/`par`/
  `threads` on its outer compute), no inner `Compute`/`Reduce`
  (`body_is_flat`), no block hint of its own (it would be silently
  discarded), and **every seam read at the identity index** `y[k]`.
  Affine-permutation seams are deferred.
- **Equal outer bound**, compared symbolically (producer-consumer
  precedent) — the producer schedule is reused unchanged, so M8/M9's
  concrete-bound requirement does not apply.
- **Disjoint origins** — enforced by the enumeration context.

**Synthesis:** producer-consumer's conventions throughout — merged
param bindings with conflict rejection, name-keyed param remap,
producer inputs declared first then consumer non-seam inputs, one
fresh outer var `k`, `type_infer` gate, boundary BufIds via
`gf.physical`. The producer body is cloned **once** at the identity
index and every seam `Index` node is mapped to that clone through the
plain `subst` map — no clone-time hook needed, because `clone_expr`
consults `subst` before descending. Keep variant wraps
`Tuple(consumer_body, producer_body)` and appends the seam to the
outputs (§10.2). Canonical names `epilogue_drop`/`epilogue_keep`.

**Overlap with producer-consumer:** for a flat, unhinted producer both
passes would synthesize identical HIR differing only in module name —
and the name participates in `module_hash`, so `CandidateKey` dedup
would *not* collapse the pair, inflating candidate counts in every
existing driver test. `enumerate` therefore skips producers where
`identify_kernel_shape(..).is_some() && block_hint().is_none()`
(producer-consumer's territory). The skip is a dedup measure, not a
legality constraint — `synthesize_epilogue` still succeeds on such
pairs when called directly (covered by test). Epilogue's territory:
block-hinted producers (producer-consumer drops hints), `par`/
`threads` producers and non-Index input use (its recognizer rejects
them). Note a block-hinted *flat* producer is enumerated by both
passes — producer-consumer emits the hintless rebuild, epilogue the
hint-retaining one — and the estimator/ILP arbitrates.

**Tests (11 new — all passing, `mod epilogue_tests`):**

Exit gate + HIR:
- `driver_end_to_end_retains_producer_schedule` — block-hinted
  row-sum reduction + pointwise scale collapses to one
  `epilogue_drop` kernel with the hint retained (M10 exit gate).
- `hinted_flat_producer_matches_reference` — module hash equals the
  hand-authored `compute[n] |k| (2*a[k])*3` reference with hint 128.

Schedule retention:
- `threads_producer_retains_threads` — `#[grid(threads = 64)]`
  carries to the fused top-level `Compute`.
- `par_producer_retains_par` — the producer's `ParSpec` is copied
  verbatim (its `expr` only references its own `thread`/`seq`
  binders, so no alpha-renaming is needed).

Variants + coverage:
- `keep_variant_when_seam_is_output` — drop + keep emitted; keep has
  2 outputs, `epilogue_keep` name, hint retained.
- `covered_producer_is_skipped` — flat unhinted producer yields zero
  drafts from `enumerate` but `synthesize_epilogue` succeeds.

Legality rejections:
- `rejects_non_identity_seam_read` (`y[n-1-k]` →
  `SeamReadNotIdentity`), `rejects_non_flat_consumer` (inner `Reduce`
  → `ConsumerNotPointwise`), `rejects_bound_mismatch`,
  `rejects_hinted_consumer`, `rejects_tuple_body_producer`.

**Verification:**
- `cargo check -p crypto-compiler` clean.
- `cargo clippy -p crypto-compiler --all-targets --tests -- -D warnings` clean;
  same with `--features planner-ortools`.
- `cargo +nightly fmt` applied.
- `cargo nextest run -p crypto-compiler --lib fusion_v2` — **111**
  tests pass (100 pre-M10 + 11 new). With `--features planner-ortools`
  — **115** tests pass.
- Full `cargo nextest run -p crypto-compiler --lib` — 325 tests pass.
  Epilogue enabled by default caused **zero candidate-count churn** in
  existing driver tests, confirming the coverage skip works.

**Design decisions:**

- **Single producer clone at the identity index.** Because every seam
  read is identity, one clone serves all read sites via the `subst`
  map; producer-consumer's per-site re-clone hook machinery is not
  needed. Affine-permutation seams (which would need per-site
  substitution of the producer's outer var) are deferred.
- **Consumer block hints reject rather than reconcile.** Even an
  equal hint is rejected — a consumer with an explicit hint signals a
  deliberate schedule this pass would override; producer-consumer can
  still fuse the pair on the consumer's terms.
- **Keep gating reuses `should_emit_keep`** rather than duplicating
  the §10.2 trigger logic.

**Not landed in M10** (deferred):

- Affine-permutation seam reads (per-site producer inlining, as in
  producer-consumer's hook path).
- Multi-output producers (first-slice restriction shared with
  producer-consumer).
- Scatter-carrying producers — needs the §8 access-relation machinery
  to prove the consumer's identity read matches the permuted store.
- Driver-level tests for `par`/`threads` producers — enumeration-level
  only, since the estimator's KIR lowering of bare `par` fixtures is
  exercised separately and a lowering panic would price the candidate
  at infinite cost (M8 `catch_unwind` guard), silently deselecting it.

### M11 (session 14): opt-in GraphCompiler integration

**Files:** `src/graph_exe.rs`, `src/passes/fusion.rs` (one field),
`src/passes/fusion_v2/driver.rs` (verbose), `src/passes/fusion_v2/tests.rs`
(`graph_compiler_tests` module).

Plan §16's opt-in wiring. v2 is now reachable from the public compile
pipeline while the existing pass stays the default:

- **`FusionStrategy` enum** (private, plan §16 sketch):
  `Existing(FusionOptions) | V2(Box<FusionOptionsV2>)` (boxed for
  clippy's `large_enum_variant`), stored as
  `GraphCompiler.fusion: Option<FusionStrategy>` so `without_fusion`
  keeps its `None` semantics. `new()` defaults to
  `Existing(FusionOptions::default())`; `fusion_options` keeps
  selecting the existing pass; new `fusion_v2_options` selects v2.
- **`fuse()` routing**: same normalize prelude/postlude for both
  strategies (`lower_reduce → monomorphize → canonicalize → fuse →
  canonicalize → monomorphize → plan = None`). The v2 arm clones the
  options, merges `GraphCompiler::env` into
  `FusionOptionsV2::graph_symbols` (env wins on conflict — it is
  already authoritative for memory planning and size evaluation; M4
  deferral closed), runs `fuse_graph_v2`, and maps `FuseV2Error` to
  `CompileError::Verify`.
- **Report embedding** (plan §15 "extend the existing fusion report
  rather than changing `GraphExe`'s report type"): `FusionReport` gains
  a defaulted `pub v2: Option<FusionReportV2>` field. When v2 ran, the
  wrapper carries only `nodes_before`/`nodes_after` and the embedded v2
  report holds the §15 counters (candidates, rounds, cost-cache stats,
  `total_runtime_units`, `fallback_reason`). `GraphExe::fusion_report()`
  and every existing caller (gpu_graph tests, cuda-backend dumps) are
  unchanged.
- **`FusionOptionsV2::verbose`** (§15): per-round saturation counters
  (`generated`/`inserted`/`alt_nodes`) and a selected-extraction dump —
  one line per selected node with id, seed/alt provenance, kind +
  module name, `runtime_units`, and value-class ports — mirroring the
  existing pass's `FusionOptions::verbose`.
- **No-solver semantics verified** (§16 "do not silently run the
  existing implementation under the name v2"): without
  `planner-ortools` the driver already routes to the brute-force
  extractor (≤ `BRUTE_FORCE_LIMIT` = 32 alt nodes, a genuine v2
  extraction with `fallback: None`) and otherwise returns the original
  extraction with `FallbackReason::SolverUnavailable`. Covered by a
  34-disjoint-kernel test gated `#[cfg(not(feature =
  "planner-ortools"))]`.

**Tests** (6, in `graph_compiler_tests`, gated on default `planner`
feature): v2 strategy fuses a two-kernel chain through
`GraphCompiler::fuse` and embeds the v2 report with `fallback_reason:
None`; default strategy still runs v1 (`report.v2.is_none()`);
`without_fusion` disables both; three-chain module-count golden
comparison (unfused 3 → v1 1 = v2 1); env→`graph_symbols` threading
(unbound symbolic memcpy size falls back to a 1 KiB estimate, so
binding the symbol must strictly raise `total_runtime_units`);
`SolverUnavailable` beyond the brute-force cap.

**Verification:** 117/117 fusion_v2 lib tests default (120/120 with
`planner-ortools`), full lib 331/331, clippy `-D warnings` clean both
configs, fmt clean, `cargo check -p openvm-cuda-backend` clean.
gpu_graph integration suite: 10/12 pass; the 2 failures
(`module_with_intermediate_buffers_is_rejected`,
`symbolic::partial_monomorphization_and_fusion`) reproduce on clean
HEAD `3650dc5b` — pre-existing, unrelated to M11.

**Design decisions:**

- **env overrides `opts.graph_symbols` on merge.** A stale caller-set
  binding that disagreed with `GraphCompiler::env` would make the
  estimator price kernels against sizes the planner never compiles
  for; `symbol()` is the one API for graph sizes.
- **`FuseV2Error` → `CompileError::Verify`.** Both variants
  (`TakeGraph`, `Apply`) are structural-invariant failures, matching
  the existing pass's use of `verify()`-sourced errors.
- **v1 wrapper fields stay defaulted under v2** rather than being
  synthesized (e.g. faking `fused` pairs from v2 history): consumers
  that want v2 detail should read `report.v2`, and fabricated v1
  counters would corrupt existing dashboards silently.

**Not landed in M11** (deferred):

- Measured-runtime and cold-compile-time comparisons on real workloads
  — plan M12 collects these per `dsl_port_tests` fixture and folds
  them into the M11 comparison report.
- GPU semantic-equivalence tests (fused vs unfused output) — unblocked
  now that v2 is reachable from `GraphCompiler`; they land with M12's
  bit-for-bit oracle runs.
- `NoImprovementOverOriginal` fallback wiring (needs
  `runtime_tolerance_ppm` in `FusionOptionsV2` and the original-cost
  comparison in the extractors).
- Graph-take guard on error paths (§14.3) — `fuse_graph_v2` errors
  currently leave `g` drained by `take_graph`.
- Origins/`FusionHistory` in the verbose dump; per-node dump files
  under `dump_dir` (the module-level dumps only cover compiled
  kernels).
- Boundary-local pruning (§11 step 8, efficiency-only).

### M12 (session 15): numerical accuracy + perf on fractional_sumcheck

**Files:** `crates/cuda-backend/src/logup_zerocheck/fractional_ir_dsl.rs`,
`crates/cuda-backend/src/logup_zerocheck/fractional_sumcheck_gpu_irv2.rs`,
`crates/compiler/src/passes/fusion_v2/cost/estimator.rs`,
`crates/compiler/src/passes/fusion_v2/cost/transactions.rs`,
`crates/compiler/src/passes/fusion_v2/extract/cpsat.rs`,
`crates/compiler/src/passes/monomorphize.rs` (`outer_bounds` → `pub(crate)`)

**Wiring.** `run_graph_read_bufs` in `fractional_ir_dsl.rs` reads
`FRAC_DSL_FUSION` (`v1` default / `v2` / `off`) and configures the
`GraphCompiler` accordingly; under v2 it prints a `[dsl-fusion-v2]`
report line (generated/inserted/selected/fallback) after compile. The
`bench_fractional_sumcheck_eager_vs_irv2` bench gained
`FRAC_V2_BENCH_FUSION_V2=1` plus `FRAC_V2_BENCH_SOLVER_SECS` (default
60) and `FRAC_V2_BENCH_MAX_ALTS` (default 5000) knobs.

**Correctness gate (exit criterion): PASSED.** All 8 `dsl_port_tests`
fixtures are bit-for-bit identical to the eager CUDA reference under
`FRAC_DSL_FUSION=v2` with CP-SAT extraction — every compile reported
`status=Some(Optimal)`, `fallback=None`. The v1 baseline also passes
8/8. Full cuda-backend suite: 411/413 (the 2 failures are
pre-existing on clean HEAD `3650dc5b`, unrelated).

Note: on the fixtures the saturation generates 0 candidates. That is
legitimate, not a bug: kernel→kernel seams in the fractional-sumcheck
fold chains have *halving* outer bounds (producer iterates `n`,
consumer `n/2`), which producer-consumer rejects with
`OuterBoundMismatch` (M3-slice restriction), and const/memcpy parents
fail `NotAKernel`. The correctness gate therefore exercises the full
v2 pipeline (saturation, costing, CP-SAT, apply) but selects the seed
extraction. The bench graph (below) does produce real fusions.

**Estimator bug 1 — stripped block hints.** `GraphCompiler::fuse()`
runs `lower_reduce → monomorphize → canonicalize → fuse → canonicalize
→ monomorphize`. Monomorphize stamps per-group block hints, but the
canonicalize *before* fuse rebuilds modules and drops them, so any
kernel with a symbolic outer bound hit `lower_to_kir`'s "a block hint
is required" error inside the estimator → sentinel cost
(`i64::MAX/4`). Never visible under v1 (which doesn't lower for
costing); the final monomorphize re-stamps before real compilation, so
compiled artifacts were always fine. Fix: `estimate_kernel` now calls
a local `stamp_block_hint` that mirrors monomorphize's
`block_size_policy(max_outer)` using bounds concretized against
`graph_symbols` + `param_bindings`. Covered by new lib test
`symbolic_outer_bound_is_costed_via_stamped_block_hint`.

**Estimator bug 2 — sampler bound named params to the par index.**
`analyze_program` threaded only VarId-keyed `ctx.graph_symbols` into
the transaction sampler; name-keyed `param_bindings` (e.g. `q`) were
never bound, and `eval_quast` treats any unbound symbol as the par
index — producing garbage addresses and, post-fix-1, a debug multiply
overflow swallowed by `catch_unwind` into `CostError::LoweringPanicked`
→ sentinel cost. Fix: merge `kp.params` × `param_bindings` into an
augmented context in `analyze_program`, and make `eval_quast` use
checked arithmetic (overflow → `None` → worst-case sectors instead of
panic). After both fixes, `fold_ef_frac_columns_dsl` costs scale with
`q` (3928/7056/13312 units) instead of the sentinel.

**CP-SAT at scale — seed-solution hints.** The bench graph (n=2^16)
has 2649 seed nodes; saturation generated 60,938 candidates and
inserted 5000 (`max_total_alternatives` cap hit in round 0), giving a
7,649-node model (~7.6k bool vars). A cold single-worker CP-SAT could
not find *any* feasible solution within 5 s or even 60 s per stage →
`SolverStatusUnknown` → fallback to the unfused original. Fix in
`cpsat.rs`: hint the always-feasible all-seeds solution (§13.2) before
stage 1, and re-hint each stage's solution before stages 2–4 (it stays
feasible under the added objective-lock constraint). With hints the
60 s solve returns `Feasible`, `fallback=None`, selecting 2226/7649
nodes (1969 seeds + 257 fused kernels replacing 680 seeds). Verified
against the existing CP-SAT/brute-force agreement tests (121/121 with
`planner-ortools`).

**Performance, n=2^16 (compare ratios within-run — the eager baseline
varies 4.3–6.4 ms across runs with GPU state):**

| metric | v1 | v2 unfused (pre-hints fallback) | v2 hinted CP-SAT (60 s) |
|--------|----|--------------------------------|--------------------------|
| eager median | 6.43 ms | 4.33 ms | 4.36 ms |
| exec ratio vs eager | 1.674× | 2.150× | 1.978× |
| **capture ratio vs eager** | **1.367×** | 1.307× | **1.251×** |
| nodes | 2649 → 1428 | 2649 (no fusion) | 2649 → 2226 |
| unique modules | 455 (445 nvcc'd) | 38 | 48 (14 nvcc'd, 11.9 s) |
| build+compile wall | 420.6 s (nvcc-dominated) | 488 s | 566 s (generation-dominated) |

Takeaways:

- v2's captured graph is *relatively* faster than v1's (1.251× vs
  1.367× eager) while compiling **9.5× fewer unique modules** — nvcc
  work drops from ~420 s to ~12 s. v2's cost model concentrates fusion
  where it pays instead of v1's 5,832 indiscriminate merges.
- Fusion itself buys ~4% capture time over unfused v2 (5.66 → 5.45 ms
  at matched eager baselines) — bounded by the 5000-alternative
  insertion cap (8% of generated) and the Feasible-not-Optimal solve.
- Neither v1 nor v2 beats eager at this size; the captured graph is
  the right comparison point and it is 1.25× eager under v2.
- v2's wall-clock cost is now dominated by candidate *generation*
  (~460 s): enumeration does not early-stop once the insertion cap is
  reached. This is the top efficiency item.

**Not landed in M12** (deferred):

- Early-stop of candidate enumeration at `max_total_alternatives`
  (60,938 generated vs 5,000 inserted; ~460 s wasted).
- `num_search_workers` is hardcoded to 1; parallel solve would likely
  reach Optimal within the stage budget.
- Producer-consumer across *halving* outer bounds (the dominant seam
  shape in fractional-sumcheck fold chains) — would let the fixtures
  generate real candidates.
- Larger-size sweeps (LOG_N 20/22/24) once generation is cheap enough
  to iterate.

### M12 follow-up (session 16): parallel enumeration + sentinel-cost exclusion

**Files:** `fusions/mod.rs` (new `par_enumerate` helper),
`fusions/{producer_consumer,fanout,small_kernel,epilogue,horizontal}.rs`,
`cost/mod.rs` (`GraphNodeCost::FAILED` + `is_failure`),
`extract/cpsat.rs`, `driver.rs`,
`fractional_sumcheck_gpu_irv2.rs` (bench defaults).

**Bench defaults changed.** Horizontal fusion off by default
(`FRAC_V2_BENCH_HORIZONTAL=1` re-enables — on this graph it cost
~99% of enumeration time for a handful of launch-quantum savings);
solver budget 120 s/stage (`FRAC_V2_BENCH_SOLVER_SECS`); alternatives
cap 10,000 (`FRAC_V2_BENCH_MAX_ALTS`).

**Parallel enumeration.** All five passes now collect their sites
sequentially in deterministic order, then run per-site synthesis via
rayon (`par_enumerate` in `fusions/mod.rs`); indexed `collect`
preserves draft order bit-for-bit vs the sequential loop (draft order
matters: the insertion cap truncates in draft order). Reject counters
merge per-site `(label, count)` lists. Round-0 pass timings on the
n=2^16 bench: producer-consumer 276 ms → 27 ms; saturation total
272.9 s → 23.0 s (with horizontal off), and the budget now sustains
**3 rounds** of composition (inserted per round: 2088/5224/2688)
instead of capping out in round 0.

**Regression found + fixed: sentinel costs poisoned CP-SAT.** The
3-round graph produces composed `epilogue_keep` candidates whose
lowering panics (`is_canonicalized` assert, lower_to_kir.rs:70 — §9
candidate canonicalization is a later milestone). 3,123 such nodes
each got the `i64::MAX/4` sentinel cost; those coefficients overflowed
CP-SAT's int64 objective validation → stage 1 `Unknown` in 0.1 s →
silent fallback to the unfused seeds. Fix: named
`GraphNodeCost::FAILED` sentinel + `is_failure()`; `cpsat.rs`
force-excludes failed *alternatives* from the model (`x_a = 0`,
omitted from the objective — seeds always cover the graph, so
feasibility holds) and clamps a hypothetical failed *seed*'s
coefficient so the objective sum cannot overflow; the stage-1 lock
uses the same clamped coefficients. `driver.rs`'s
`total_runtime_units` diagnostic skips sentinels. Brute-force
extractor already summed in i128 and needed no change.

**Bench, n=2^16, horizontal off / 120 s / 10k cap:** generated
72,351, inserted 10,000, alt graph 12,649 nodes; costing 1.5 s
(3,123 failures excluded); model x=12649 y=5299 z=792; all 4 stages
Feasible at 120 s (runtime 14,143,519 vs 14,165,924 in the 60 s
horizontal-on run; artifacts 67; nodes 2227); selected 2223/12649,
`fallback=None`. Perf: capture 5.54 ms = 1.226× eager (prior best
1.251×); exec 8.56 ms = 1.893×. Compile wall 613 s (28 modules
nvcc'd).

**New/remaining efficiency items:** insertion (`would_create_cycle`
per candidate) now dominates saturation — 14.5 s in round 2 vs 2.2 s
of enumeration; the `is_canonicalized` lowering panic on composed
epilogue_keep candidates wastes 3,123 candidates (real fusion
opportunities lost, not just noise); enumeration still lacks
early-stop at the cap (72,351 generated vs 10,000 inserted); solver
stages are Feasible-not-Optimal even at 120 s (num_search_workers
still 1).

### Session 17: solver workers, stream-scheduler rebase, LOG_N=24 nsys

**Solver workers.** `solver_num_workers` plumbed through
`FusionOptionsV2` → `ExtractOptions` → CP-SAT `SatParameters`
(`num_search_workers`). Default 1 keeps the solve deterministic per
plan §2.4; the bench defaults to all cores
(`FRAC_V2_BENCH_SOLVER_WORKERS`). Effect at 16 workers on the
LOG_N=24 model (x=15,965, y=11,931, z=191): stage 1 (runtime) went
Feasible-at-120 s → **Optimal in 1.3 s**; stages 3/4 Optimal in
15.7 s/40.2 s; only stage 2 (artifacts, objective 50) still hits the
120 s cap Feasible. Total solve 177 s (vs 480 s all-Feasible at 1
worker on the smaller n=2^16 model).

**Rebase onto `feat/stream-scheduler`** (multi-stream co-scheduling:
`SchedulerMode::ListV1`, `StreamInstr::WaitOn`, multi-stream graph
capture). Clean rebase; fixed 5 clippy lints in the incoming
feature-gated planner code (`planner/heuristic.rs`,
`planner/list_v1.rs`); all 121 fusion_v2 lib tests pass on the
rebased tree.

**nsys LOG_N=24** (single profiler window, per-iteration nvtx
ranges; v2 fusion at 120 s/10 k/horizontal-off defaults + list_v1
scheduler with 8 streams;
`target/nsys/frac_v2_log24_fusionv2_streams8.nsys-rep`):

| mode | wall median | kernels/iter | kernel GPU time |
|---|---|---|---|
| eager | 17.07 ms | 841 (1 stream) | 6.20 ms |
| v2 graph exec | 45.15 ms | 5,003 (5 streams used) | 19.74 ms |
| v2 graph capture | 18.38 ms | 5,003 (graph replay) | 19.66 ms |

Saturation 22.0 s / 2 rounds (generated 30,626, inserted 10,000,
5,965 seeds → 5,029 selected); costing 1.2 s (14,007 cache hits,
1,656 sentinel failures — same epilogue_keep canonicalization
panic); nvcc 9 s (43/50 modules from disk cache).

Findings: (1) captured v2 graph is within **7.7% of eager**
(18.38 ms vs 17.07 ms; was 22.6% at n=2^16) despite executing
3.2× the kernel-time — CUDA-graph replay keeps the GPU essentially
saturated (19.66 ms kernel time in an 18.4 ms window). (2) Un-captured
graph exec is launch-bound: ~9 µs/launch × 5,029 launches ≈ the 45 ms
wall; list_v1 also skews placement (2,774/1,347/781/71/30 kernels on
the 5 streams it used of 8). (3) Eager itself is gap-bound on one
stream (6.2 ms GPU time in 17.1 ms wall), so beating eager is within
reach if the 3.2× work amplification drops — the [n,2] Frac spine
seam legality (rank-2 reads → SeamIndexNotAffine) remains the
blocker. Caveat: nsys `--capture-range` defaults to
`stop-shutdown`, which SIGTERMs the test at `cudaProfilerStop`
(stdout summary lost, profile intact); pass
`--capture-range-end=stop` to keep the process alive.

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
