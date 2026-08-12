# Mutation semantics & SSA restoration — progress

Status doc for the work adding an SSA-restoration pass + alias-aware planner so
`verify_graph`'s single-writer / SSA invariant can coexist with `BlackboxKernel`'s
in-place-mutation contract.

**Status: complete.** All e2e variants pass; the final bug was an
alias-blind WAR guard in fusion v1 (see "Root cause" below).

## Background

`GraphBuilder::insert_blackbox_kernel(inputs, outputs, modifies)` takes a per-input
`modifies: bool` flag. Any input with `modifies=true` is folded into
`KernelNode::carried_outputs` — the same `BufId` flows in as a read and out as a
re-write of that buffer. This is convenient for callers who want to mutate a
long-lived scratchpad in place (the fractional-GKR driver, for example, mutates a
single `layer` buffer across a dozen kernel launches).

`verify_graph` (added in `feat: big refactor part 2, decouple graph passes`)
enforces two rules:

1. **Shape consistency** for `Kernel` nodes.
2. **Single-writer + write-before-read.** `classify_buf_uses` counts writers per
   buffer; any buffer with more than one entry in `writers[b]` fails the check.

`classify_buf_uses` (used by *both* the planner and the verifier) treats
`carried_outputs` as writes — necessary so downstream dep tracking sequences
readers after each mutation. But this makes the verifier reject any buffer that
gets mutated more than once, blocking every `fractional_ir` end-to-end test.

The chosen fix (from the user):

> Not necessary to mark edges as being mutated so long as every mutated edge is
> used once, and there's a substitution pass later that makes later uses of the
> mutated edge use the carried output edge instead.

Concretely: SSA-rename every `carried_output` to a fresh `BufId`, rewrite
downstream reads to the new id, and record an alias so the planner packs both
onto the same pool slot.

## Design

Two-part change:

1. **`passes::restore_ssa`** rewrites the graph so `carried_outputs` becomes
   empty and every mutation produces a fresh SSA output. Aliases record the
   equivalence class.
2. **Alias-aware planner** (`PlanCtx::build_with_aliases`) canonicalizes every
   buffer reference, packs only canonicals, and propagates the canonical's
   offset to every member so the runtime's `bufid_ptr` resolves aliased members
   to the same pool address.

The mutating blackbox closure operates on `inputs[0]` (the read side); with the
alias in place, `inputs[0]`, `outputs[0]`, and every downstream reader of the
fresh SSA output all resolve to the same offset — the mutation lands where
downstream reads look for it.

## What's implemented

### `GraphBuilder::aliases` (crates/compiler/src/graph_ir.rs)

- `pub aliases: Vec<Option<BufId>>` field.
- `add_buf` grows it in lockstep with `bufs`.
- `canonical_buf(id)` walks the alias chain (path-shorter loop).
- `alias_bufs(child, parent)` unions `child` under `canonical_buf(parent)`,
  panicking on self-alias or double-alias.

### `passes::restore_ssa` (crates/compiler/src/passes/restore_ssa.rs)

Walks nodes in insertion order, maintains `current: HashMap<BufId, BufId>` from
original → latest SSA version.

- `BlackboxKernel`: remap `inputs` through `current`; for each `carried_output`,
  allocate a fresh `BufId` cloning the source `BufInfo`, alias it to the
  canonical, push it onto `outputs`, and update `current`. Clear
  `carried_outputs`.
- `Kernel`: remap `inputs` only (outputs are already SSA producers).
- `Memcpy`: remap `src`.
- `Memset`: if the target was mutated earlier, allocate a fresh SSA version and
  alias it (a memset is a fresh definition of the canonical).
- `Const`: no inputs to remap.

Returns `RestoreSsaReport { renamed_carried, aliases_added }`.

Idempotent: a second run over an already-SSA graph does nothing.

### `PlanCtx::build_with_aliases` (crates/compiler/src/planner/ctx.rs)

- New `canon: Vec<usize>` field, path-compressed at build time.
- Writers/readers accumulate onto the canonical; sizes take the max across the
  class; `on_device` and `pinned` fold onto the canonical.
- `packable(b)` returns `false` for non-canonical members — backends only pack
  canonicals.
- `PlanCtx::build` delegates with empty aliases (backward compatible).

### `propagate_alias_offsets` (crates/compiler/src/planner/ctx.rs)

After the backend fills offsets for canonicals, this walks `canon` and copies
`offsets[canonical]` to every alias member. Called once inside `plan_raw`.

### `plan_raw` signature change (crates/compiler/src/planner/mod.rs)

Added an `aliases: &[Option<BufId>]` parameter (empty slice = no aliases).
`plan()` threads `graph.aliases` through automatically.

### Compile pipeline hook (crates/compiler/src/graph_exe.rs)

`GraphCompiler::compile` invokes `restore_ssa` at stage 0 — before `fuse` /
`lower_reduce` / `canonicalize` — so every downstream pass, including the
verifier, sees SSA. `plan_memory` forwards `graph.aliases` to `plan_raw`.

## Testing status

### Unit tests — `passes::restore_ssa::tests`

All pass:

- `empty_graph_is_noop`
- `single_mutation_chain_produces_alias_class` (Memcpy → k0(mut) → k1(mut) →
  reader; asserts 3-member alias class and that the reader reads the last SSA
  version)
- `intermediate_reader_sees_correct_version` (regression for the deleted
  `current[latest] = fresh` fallback)
- `independent_buffers_stay_in_separate_classes`
- `idempotent_when_no_carried_outputs`

### Existing planner tests

All 4 pass (`packs_disjoint_lifetimes`, `overlapping_lifetimes_do_not_share_memory`,
`respects_symbol_assignment`, `unbound_symbol_is_reported`) — the alias-free
delegation path is unchanged.

### `fractional_ir` non-e2e tests (openvm-cuda-backend)

All 9 pass:

- `build_segment_tree_ir_matches_host_dense_small`
- `build_segment_tree_ir_matches_host_dense_large`
- `build_segment_tree_ir_matches_host_virtual`
- `sqrt_eq_layers_ir_matches_host`
- `reduce_to_single_evaluation_ir_matches_host`
- `reconstruct_s_evals_ir_matches_host`
- `update_running_scalars_ir_matches_host`
- `eq_mle_table_ir_matches_host`
- `dev_challenge_entry_points_match_host_value`

### Round composites — all 3 pass

- `do_sumcheck_round_and_revert_ir_matches_eager`
- `do_fused_sumcheck_round_ir_matches_eager`
- `do_fused_sumcheck_round_inplace_ir_matches_eager`

### End-to-end — **all 6 variants pass**

`fractional_sumcheck_gpu_ir_matches_eager_{dense, virtual, virtual_half_edge,
precompute_m_dense, precompute_m_virtual, precompute_m_multi_window}` all
pass (verified with the on-disk kernel cache cleared first —
`rm -rf ~/.openvm/kernel_cache`).

Plumbing observations at `n=16` (dense):

- `restore_ssa` renamed 12 carried outputs (matches count of mutating blackboxes in the driver).
- 10 aliases point to the canonical `BufId(2)` (`"leaves"` — the layer buffer) at pool offset 512.
- 2 aliases point to work buffers `gkr_work_2` (`BufId(76)`) and `gkr_work_3` (`BufId(131)`) — round-2 and round-3 in-place folds on the FoldEval work slot.
- `writers[canonical layer] = [2, 3, 4, 5, 6, 7, 10, 24, 47, 80, 89]` — 1 memcpy + 10 mutations in insertion order.
- Peak pool: 2096 bytes with aliases vs 2256 bytes without — packing correctly shares the mutation-chain slot.

## Root cause of the layer-1 divergence (fixed)

Two suspects were eliminated first:

- The `current[latest] = fresh` fallback in `rewrite_blackbox` was deleted
  (with regression test `intermediate_reader_sees_correct_version`) — the
  e2e still failed, so `restore_ssa` was not the bug.
- The planner was verified sound: `PlanCtx::build_with_aliases`
  canonicalizes every read/write onto the class root, so `edges()` derives
  WAW/RAW/WAR precedence over the *class*, fully re-serializing the
  mutation chain. Neutralizing fusion (`FusionOptions { max_iterations: 0 }`)
  made the e2e pass — isolating the bug to fusion v1.

**The bug**: the WAR guard in `enumerate_candidates`
(`passes/fusion.rs`) checked `writers[input_buf]` — writers of the *exact*
`BufId`. Post-`restore_ssa`, every `BufId` has exactly one writer by
construction; an in-place mutation clobbering a slot writes a *fresh SSA
sibling* of the buffer it overwrites. So the guard never fired, and fusion
inlined producers into consumers positioned *after* a mutation of the
producer-input's alias class — moving the producer's read past the clobber
of its pool slot. First divergence: `sumcheck_polys[0][0]` (round 1's first
compute), which then cascaded through the transcript.

**The fix**: the guard now checks class-level writers — `writers[m]`
unioned over every `m` with the same `canonical_buf` as the producer input.
Regression test `alias_sibling_war_hazard_blocks_fusion` (blackbox writes
an alias sibling of the producer's input between producer and consumer →
candidate rejected).

Note: `passes/fusion_v2` will need the same alias-class WAR treatment when
it lands; only v1 was fixed here.

## Pre-existing failures on this branch (not related to this work)

`cargo nextest run -p crypto-compiler`: 424/426 pass. The 2 failures occur
identically with the fusion fix stashed, i.e. they pre-date it:

- `gpu_graph::module_with_intermediate_buffers_is_rejected` — compile now
  *succeeds* where the test expects a lowering-time rejection of
  multi-kernel modules.
- `gpu_graph::symbolic::partial_monomorphization_and_fusion` — "kernel k0
  has a symbolic outer bound (s0 * 8); a block hint is required". The same
  error appears when compiling with `.without_fusion()`: the no-fusion
  branch of `GraphCompiler::compile` runs `monomorphize` *before*
  `canonicalize` (the fuse path ends `canonicalize → monomorphize`), so
  block hints are lost.

## Files touched

- `crates/compiler/src/graph_ir.rs` — `aliases` field, `canonical_buf`, `alias_bufs`, `add_buf` grow.
- `crates/compiler/src/passes/mod.rs` — expose `restore_ssa` module.
- `crates/compiler/src/passes/restore_ssa.rs` — new pass + 5 unit tests.
- `crates/compiler/src/passes/fusion.rs` — alias-class WAR guard in `enumerate_candidates` + regression test.
- `crates/compiler/src/planner/ctx.rs` — `build_with_aliases`, `canon` field, `packable` canonical guard, `propagate_alias_offsets`.
- `crates/compiler/src/planner/mod.rs` — `plan_raw` alias parameter, propagation call.
- `crates/compiler/src/graph_exe.rs` — stage-0 `restore_ssa` invocation, `plan_memory` alias threading.

## Retrospective notes

Diagnosis discipline that would have shortened this loop:

- Write the `rewrite_blackbox` correctness proof as a unit test *before*
  wiring it into the planner. A 20-line test on a hand-built
  `Memcpy → mut → mut → read` graph asserting the exact final `k.inputs` and
  `k.outputs` would have caught (or ruled out) the `current[latest] = fresh`
  line before any e2e run.
- Reach for a scratch `#[cfg(test)]` unit test, not `env::var` prints in
  shared infrastructure, when a specific hypothesis needs checking.
- After the first debug print confirmed the aliases resolved as expected,
  stop and re-focus on the SSA rewrite logic — that's where the remaining
  bug must live if the planner side is verified correct.
