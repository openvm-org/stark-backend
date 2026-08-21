# GKR Small-Round Overlap (Precompute-M) — Plan & Status

Status tracker at the bottom. Owner: crypto-compiler / fractional GKR IR prover.

## Goal

Add a new graph-IR driver variant in
`crates/cuda-backend/src/logup_zerocheck/fractional_ir.rs` that, for small
outer GKR rounds (pq buffer ≤ 4096 `Frac<EF>` elements), uses the
precompute-M strategy restructured so that the **next** round's
"invert-pq + compute-M" stage has **zero transcript dependencies** and can
therefore overlap the **current** round's transcript sampling and s-poly
sumcheck evals. The driver only *exposes* the parallelism in the graph;
the graph scheduler is responsible for placing the independent chains on
different streams. Any new kernels are written in the kernel DSL
(`fractional_ir_dsl.rs` style), not raw CUDA.

Terminology: "invert pq" = the GKR tree-layer *revert*
(`frac_build_tree_layer(..., revert=true)`, whose core is
`rhs_q_inv = inv(rhs.q)` — see `gkr.cu::compute_round_and_revert_kernel`,
~line 676, and the DSL port `build_frac_build_tree_layer_revert_module`).

## Reference material

- Eager prover (semantics source of truth):
  `fractional.rs::fractional_sumcheck_gpu` (line ~621), strategy selection
  `choose_round_strategy` / `choose_precompute_m_window_w` (~179–289).
- Existing IR driver: `fractional_ir.rs::fractional_sumcheck_gpu_ir`
  (~2431–3047): FoldEval arm ~2584–2780, PrecomputeM arm ~2781–3016.
- Precompute-M derivation: `docs/cuda-backend/gkr-prover.md`
  §"Precompute-M strategy" (eprint 2025/1473 §4).
- DSL module conventions: `fractional_ir_dsl.rs` header (dense-only,
  challenges bind as `[D_EF] BabyBear`, `Frac<EF>` binds as `[n,2] FpExt`,
  quasi-affine indices only, two-stage reduce pattern at ~699–737).

## Why stock precompute-M cannot overlap

In the existing PrecomputeM arm, the M build for outer round R
(`frac_precompute_m_build_raw` / graph equivalent) depends on the
transcript in three ways:

1. **pending fold**: the window starts at inner round `base ≥ 1`, so the
   build consumes `r_prev` from the fused round-0 compute+revert
   (`do_sumcheck_round_and_revert_ir`) — a sampled challenge.
2. **eq tail**: windows with `base + w < R` leave tail variables `b`, and
   the build weights by `eq(b, z_b)` where `z_b` ⊂ `xi_prev` — which
   contains `mu` and `r_vec` from the *previous* round's transcript.
3. **λ**: the eager build folds `λ` into the left factor
   (`sh_left0 = wt*(p0 + λ*q0)`, `gkr.cu::precompute_m_build_partial_kernel`
   ~810) — `λ` is sampled at the top of round R.

So the M build can only start after round R's λ sample, which serializes
it behind the whole transcript chain.

## Key design: challenge-free M

Three restructurings remove every transcript input from the build stage.
They are only affordable because the regime is small (pq_size ≤ 4096,
i.e. R ≤ 11, M has 4^R ≤ 4.2M entries):

1. **Full-round window, w = R, no tail.** The window covers *all* R inner
   sumcheck rounds of outer round R. The tail `b` is empty, so the
   `eq(b, z_b)` factor vanishes (weight 1). M entries become pure outer
   products (no reduction over b).
2. **base = 0, no pending fold.** Round 0 is inside the window, so there
   is no fused fold-by-`r_prev`; M is built directly from the reverted
   pq(R). The revert becomes a standalone stage instead of being fused
   into round-0 compute (`do_sumcheck_round_and_revert_ir` is skipped
   entirely in the small regime, as is `SqrtEqLayersIR`).
3. **λ-separation.** Split `M = M_a + λ·M_b`:
   - `M_a[u,v] = p0[u]·q1[v] + p1[u]·q0[v]`
   - `M_b[u,v] = q0[u]·q1[v]`
   where `p0/q0` = first half of pq(R) (poly 0) and `p1/q1` = second half
   (poly 1), `u, v ∈ [0, m)`, `m = 2^R = pq_size/2`. λ is combined inside
   the eval kernel (`m = m_a + λ·m_b`) from a device scalar. With an empty
   tail this is exactly the eager `M[u,v] = (p0[u]+λ·q0[u])·q1[v] +
   p1[u]·q0[v]` decomposed by powers of λ.

### The two chains

Per small round R:

- **A(R) — challenge-free:** out-of-place revert producing side buffer
  pq(R) (parents = pq(R−1), rhs = stored tree level), then M_ab build from
  pq(R). Depends only on A(R−1) and the segment tree — never on the
  transcript.
- **B(R) — transcript-serial:** sample λ; seed `prev_s_eval` from previous
  claims (`reduce_to_single_evaluation_ir` + `claim_combine_ir`); R eval
  rounds t = 0..R, each: eq tables (`eq_mle_table_ir`), eval kernel →
  `d_sum` (s'(1), s'(2)), `observe_and_update_ir` (observe evals, sample
  r_t, update `prev_s_eval`/`eq_r_acc`; `xi_j = xi_prev[t]`, t=0 → mu);
  finally claims fold (eq_r contraction of pq(R)) → observe 4 claims →
  sample mu.

The whole A-chain (A(1) → A(2) → … → A(R_max)) hangs off the segment-tree
build and is independent of every B(R). B(R) reads pq(R) (from A(R)) and
M_ab(R). The graph scheduler sees A(R+1) as concurrent with B(R) — the
requested overlap — and can in fact run the entire A-chain ahead.

### Correctness / transcript equivalence

- Fold-vs-contract: the eager path folds pq by r_t each round and reads
  claims at indices 0 and pq_size/2; the overlap path contracts once at
  the end: `claim_α = Σ_u eq_r[u]·poly_α[u]` with
  `eq_r = eq_mle_table_ir([r_0..r_{R-1}])` (big-endian: r_0 owns the top
  within-poly bit = round-0's variable — matches the bit-reversed layout).
  This is a reordering of finite-field sums — *exact*, so the transcript
  (all observed values) is bit-identical to the eager prover. Tests
  compare proofs directly.
- Eval-round index math replicates
  `gkr.cu::precompute_m_eval_round_kernel` (~1072) and the existing DSL
  port `build_frac_precompute_m_eval_round_module` (w, t): decomposition
  `[prefix t bits | X | suffix w−t−1 bits]`,
  `beta1_0 = (b1 << (suffix_bits+1)) | suffix`, `cur_bit = 1 <<
  suffix_bits`; `s'(1) = Σ w·m11`, `s'(2) = Σ w·(m00 − 2(m01 + m10 − 2·m11))`
  with `w = eq_r_prefix[b1]·eq_r_prefix[b2]·eq_suffix[s]`,
  `eq_r_prefix = eq_mle_table_ir(window_rs so far)`,
  `eq_suffix = eq_mle_table_ir(&xi_prev[t+1..R])` (empty → `[EF::ONE]`).
  The current-variable affine eq factor and prefix accumulator are handled
  by `observe_and_update_ir` exactly as in the existing PrecomputeM arm.

## Buffer plan

- **Side buffers pq(R)**: one fresh `add_frac_ef_buf` per small round,
  2^(R+1) Fracs. Total ≤ Σ 2^(R+1) < 2·4096 Fracs = 256 KB — negligible.
- **M_ab(R)**: interleaved `[m·m, 2] FpExt` (m_a at col 0, m_b at col 1),
  one per small round. Dominant cost: 4^R·32 B = 134 MB at R = 11.
  If GraphBuilder multi-output kernels turn out to be well supported, two
  separate `[m·m]` buffers are an acceptable alternative; interleaved is
  the default plan.
- **Layer is read-only during the small regime.** In-place revert(R+1)
  would clobber layer[0..2^(R+1)] = the region round R still conceptually
  owns; more importantly it would serialize the A-chain against B-chain
  reads. All small-regime kernels read the layer, never write it.
- **Root/round-1 seeding**: the root revert (out-of-place via the existing
  `build_frac_build_tree_layer_revert_module`-style path, or a 2-Frac
  materialization of the existing root-revert output) produces pq(0) as a
  2-element side buffer so rounds R ≥ 1 use the uniform two-input revert.
  Exact indices to be confirmed against the IR driver's root-revert block
  during implementation.
- **Transition at regime exit** (R_max < total_rounds − 1): D2D copy
  pq(R_max) → layer[0..2^(R_max+1)] as a blackbox node (raw-pointer view
  pattern, cf. `frac_add_alpha_ir` at fractional_ir.rs:436 — blackbox, so
  the "kernels must use the DSL" rule doesn't apply). Graph WAR hazards on
  the layer buffer order this copy after all small-regime layer readers.
  Later large rounds' rhs regions layer[2^R..2^(R+1)], R > R_max, were
  never touched. From the copy onward the existing large-round arms run
  unchanged.
- **Virtual input**: if `virtual_input` (real_len < total_leaves), the
  small regime is **disabled entirely** for v1 — the driver falls back to
  the existing path. Rationale: the compact tree layout does not guarantee
  dense heap-prefix storage even for upper levels at boundary sizes (the
  `real_len == total_leaves/2` first-claims special case in
  `fractional_sumcheck_gpu_ir` shows layer[1] itself can be virtual), and
  the DSL revert/M kernels are dense-only. Possible v2 extension:
  dense-with-padding revert variant that selects stored values vs the
  host-constant `(0, virtual_padding_q(alpha, ..))` by index threshold.
- **Threshold knob**: `SWIRL_CUDA_GKR_SMALL_M_MAX_PQ` (default 4096),
  mirroring the existing precompute-M env-knob style. Read once at driver
  entry (size/env-dependent only — Principle-4-safe).
- Per AGENTS.md: new scratch allocations ⇒ update the interaction memory
  estimate in `crates/stark-backend/src/memory_metering.rs` and the
  accounting in `docs/cuda-backend/gkr-prover.md`.

### Perf caveat

At R = 11 the eval rounds scan up to 4^t·2^(R−t−1) ≤ 4^10 ≈ 1M entries per
round and the build writes 4.2M entries. This is *more* raw work than
FoldEval for large R — the bet is that small-round kernels are
latency-bound (tiny grids, serial transcript chain) and the overlap +
challenge-free A-chain wins wall-clock. The knob lets us tune the
crossover; benchmarks decide the default.

## New DSL kernels (`fractional_ir_dsl.rs`)

All follow file conventions: dense-only, `Frac<EF>` as `[n,2] FpExt`,
challenges as `[D_EF] BabyBear` via `bind_challenge_as_fpext`, EF
inversion needs `[n,2,D_EF] BabyBear` binding, quasi-affine indices
(power-of-two `rem` for clamping, cf. revert module ~line 985).

1. **`build_frac_tree_revert_two_input_module(half, layer_rows)`** —
   out-of-place revert with separate parent buffer.
   - Inputs: `parents: [half, 2] FpExt` (pq(R−1), no inversion needed);
     `layer_in: [layer_rows, 2, D_EF] BabyBear`, concrete row count
     (whole layer buffer; only [half, 2·half) is read; BabyBear binding
     for `ef_inverse_coeffs`). Initially bound with a symbolic row
     count, but layout inference needs concrete global shapes — the JIT
     panicked with "buffer `layer_in` has a symbolic shape" at e2e
     sizes.
   - Output: `pq_out: [2·half, 2] FpExt`.
   - Body: for k in [0, 2·half): rhs_row = (half + k) mod (2·half) (mod
     keeps the discarded-branch load in-bounds); parent_row = k mod half;
     `unadd`: `new_q = parents[parent_row].q · inv(layer[rhs_row].q)`,
     `new_p = (parents[parent_row].p − new_q·layer[rhs_row].p) ·
     inv(layer[rhs_row].q)`; `out[k] = select(k < half, unadd,
     layer[k])`.
   - Note: rhs region for round R is layer[2^R..2^(R+1)) — the module
     reads `layer[half + k]` (k < half) and `layer[k]` (k ≥ half), both in
     that range.
2. **`build_frac_m_outer_product_module(m)`** — M_ab build.
   - Input: `pq: [2·m, 2] FpExt`. Output: `m_ab: [m·m, 2] FpExt`.
   - Body: `u = i / m`, `v = i % m` (power-of-two quasi-affine);
     `m_a = pq[u].p·pq[m+v].q + pq[m+u].p·pq[v].q`,
     `m_b = pq[v].q·pq[m+v].q` — careful: `m_b[u,v] = q0[u]·q1[v]`, i.e.
     `pq[u].q·pq[m+v].q`; pack `[m_a, m_b]`.
3. **`build_frac_precompute_m_eval_lambda_module(w, t)`** — adaptation of
   `build_frac_precompute_m_eval_round_module` (~298):
   - Inputs: `m_ab: [m·m, 2] FpExt`; `lambda: [D_EF] BabyBear`;
     `eq_r_prefix: [2^t] FpExt`; `eq_suffix: [2^(w−t−1)] FpExt`.
   - Output: `d_sum: [2] FpExt` (s'(1), s'(2)).
   - Body: identical index math to the existing module, but each of the 4
     M loads becomes `m = m_a + λ·m_b`; λ once via
     `load_ext_coeffs`/`fpext_from_coeffs` (hash-consing CSEs it).
   - Reduce domain total = 4^t·2^(w−t−1) — up to ~1M. Follow the
     multi-stage pattern (~699–737): if `reduce_lowers_multi_stage(total,
     2)`, split into `g_blocks = (total/256).min(64)` block partials +
     `build_ef_rowsum_module(2, g_blocks)`.
4. **`build_frac_claims_fold_module(m)`** — final claims contraction.
   - Inputs: `pq: [2·m, 2] FpExt`; `eq_r: [m] FpExt`.
   - Output: `claims: [2, 2] FpExt` — Frac 0 = (Σ eq_r[u]·p0[u], Σ
     eq_r[u]·q0[u]), Frac 1 = (Σ·p1, Σ·q1). Then reuse
     `extract_claim_pair_ir(g, claims, 2, 1, name, device)` for the
     observe path. Same multi-stage-reduce guard (total = m ≤ 2048 —
     single stage almost certainly fine, but assert).

Per-round concrete sizes give ≤ ~11 JIT'd variants per module family; the
kernel cache handles reuse across runs (do not clear it).

## Driver structure (`fractional_ir.rs`)

New pub fn `fractional_sumcheck_gpu_ir_overlap` (same signature as
`fractional_sumcheck_gpu_ir`):

1. Shared prologue (unchanged): segment tree build, root revert, first
   claims, mu sample, `xi_prev` init.
2. Compute `small_rounds`: 0 if `virtual_input`, else the maximal prefix
   `R ∈ 1..=total_rounds−1` with `2^(R+1) ≤ max_pq` (knob).
3. **A-chain emission**: emit *all* small-round reverts + M builds up
   front (graph order ≠ execution order; emitting early keeps the code
   simple and the dependencies are what matter). pq(0) seeded from the
   root-revert output.
4. **B-chain loop** over small rounds: λ sample, prev_s_eval seed, t =
   0..R eval rounds (eq tables + eval kernel + `observe_and_update_ir`),
   claims fold kernel + `extract_claim_pair_ir` + observe + mu sample,
   `xi_prev = [mu] ++ r_vec`.
5. Transition copy (if any large rounds remain), then the large-round
   loop. **Refactor**: extract the existing driver's per-round body
   (round-0 fused compute+revert, FoldEval arm, PrecomputeM arm, claims,
   observe/mu) into a helper (working name `gkr_outer_round_ir`) used by
   both `fractional_sumcheck_gpu_ir` (all rounds) and the overlap driver
   (rounds > R_max). Existing driver behavior must be bit-identical after
   the refactor.
6. Epilogue unchanged (final xi/claims packaging).

## Test plan (test-first — write tests before compiler/driver changes)

In `fractional_ir.rs` tests module (templates:
`fractional_sumcheck_gpu_irv2.rs::assert_irv2_matches_eager` and the
existing graph-vs-eager helpers at ~3052+):

- `overlap_matches_eager_all_small`: n = 10 (every round small) —
  proof/transcript bit-identical to `fractional_sumcheck_gpu`.
- `overlap_matches_eager_mixed`: n = 14 (rounds 1–11 small, 12–13 large) —
  exercises transition copy + refactored large-round path.
- `overlap_virtual_falls_back`: virtual input → 0 overlap nodes in the
  graph, proof still matches eager (fallback path check).
- `overlap_knob_disable`: `SWIRL_CUDA_GKR_SMALL_M_MAX_PQ=0` → 0 overlap
  nodes, matches eager (pure refactor check).
- Node-count assertions mirror `assert_e2e_matches_eager_precompute_m`:
  host-side graph build, count `GraphNode::Kernel` whose
  `module.name` starts with `frac_m_outer_product_dsl`.
- Per-kernel DSL unit tests (CPU-check where possible, GPU compare vs
  eager raw kernels): revert-two-input, m-outer-product, eval-λ, claims-fold.

Commands (AGENTS.md + memory):
`cargo check -p openvm-cuda-backend`;
`cargo nextest run -p openvm-cuda-backend --test-threads=4`;
`cargo clippy -p openvm-cuda-backend --all-targets --tests -- -D warnings`;
`cargo +nightly fmt`.

Optional follow-up: nsys profile (nvtx ranges, `--cuda-graph-trace=node`,
`--gpu-metrics-devices=visible`, workloads set up before
`cudaProfilerStart`) to confirm the scheduler actually multi-streams the
A-chain; bench template `bench_fractional_sumcheck_all_schedulers_nsys`
in `fractional_sumcheck_gpu_irv2.rs`.

## Open questions — resolved

- pq(0) materialization: 2-Frac side buffer, seeded by a
  `insert_memcpy_range` D2D copy of `layer[0..2)` (post root-revert). All
  small rounds use the uniform two-input revert.
- Same-BufId-twice `insert_kernel`: moot (pq(0) is a side buffer).
- Refactor granularity: whole-round helper `gkr_outer_round_ir` with a
  `Copy` context struct `GkrOuterRoundCtx` (sizes + env knobs) and output
  struct `GkrOuterRoundOut` (claim, round_polys, r_vec, mu). The loop body
  was moved verbatim; only `claims_per_layer.last()` → `prev_claim`
  parameter and the loop-tail xi/push bookkeeping moved to the caller.
- Transition copy: implemented as a `GraphBuilder::insert_memcpy_range`
  node (dedicated Memcpy graph node, planner-visible) instead of a
  blackbox raw-pointer kernel — strictly better: the memory planner sees
  the byte ranges, and WAR/RAW hazards on `layer` order it exactly as
  planned.

## Pre-existing compiler bug found and fixed en route

`fractional_sumcheck_gpu_ir_overlap_knob_disable_matches_eager` runs the
*base* driver graph at `(1<<10, 1<<10)` — a size no prior test covered —
and hit `verify_graph: buffer BufId(2) written by 4 nodes`. Root cause
(pre-existing, reproduces on `fractional_sumcheck_gpu_ir` alone): the
FoldEval scheduler ping-pong (`LayerToWork`/`WorkToLayer`) re-writes
`layer` and `gkr_work_N` as *declared* blackbox outputs, but
`passes/restore_ssa.rs` only SSA-renamed *carried* (in-place) outputs, so
repeated declared writes stayed multi-writer. Fix (compiler crate):
`restore_ssa` now versions **any** write to an already-written buffer —
declared kernel/blackbox outputs, memcpy destinations, memsets — with a
fresh aliased `BufId` (planner orders alias classes via canonical-id
insertion-order WAW/WAR/RAW edges in `PlanCtx::edges`), and
`fusion::dce` seeds interface liveness through the alias table. This
also covers the overlap driver's transition `insert_memcpy_range` into
`layer` (a second write of `layer`). Unit tests:
`repeated_declared_output_writes_renamed`, `memcpy_dst_rewrite_renamed`
in `passes/restore_ssa.rs`.

## Status

- [x] Plan reviewed / open questions resolved against code
- [x] Tests written first (eager-match harness for overlap driver, per-kernel unit tests) — expected to fail/not compile initially
- [x] DSL kernel: two-input out-of-place revert
- [x] DSL kernel: M_ab outer-product build
- [x] DSL kernel: eval round with λ-combine (+ multi-stage reduce)
- [x] DSL kernel: claims fold (eq_r contraction)
- [x] Refactor: extract shared outer-round helper from `fractional_sumcheck_gpu_ir`; existing tests still pass bit-identical
- [x] Overlap driver: small-regime A-chain + B-chain emission
- [x] Transition copy node (memcpy_range, see resolved questions) + large-round reuse
- [x] Virtual-input fallback + `SWIRL_CUDA_GKR_SMALL_M_MAX_PQ` knob
- [x] All tests green: full `openvm-cuda-backend` suite 156/156 (`--test-threads=4`), full `crypto-compiler` suite 524/524 (`--test-threads=8`). One flaky failure of the two big overlap e2e tests was observed once during a cold-cache 4-way parallel run but did not reproduce in three attempts (isolation, warm-cache 43-test parallel run, exact cold-cache 16-test parallel rerun); error output from the original occurrence was lost.
- [x] clippy `-D warnings` + `cargo +nightly fmt` clean (both crates)
- [x] Accounting: new subsection in `docs/cuda-backend/gkr-prover.md`; `memory_metering.rs` unaffected (overlap driver is graph-IR-only, not on the metered eager path)

## Benchmark results (2026-08-17)

`bench_fractional_sumcheck_eager_vs_ir_overlap` in `fractional_ir.rs`
(eager vs base IR vs overlap IR; captured-CUDA-graph replays, 2-stream
ListV1, ITERS=3, e2e-checked against the eager proof each run):

| config                     | eager    | ir (base) | ir_overlap | overlap vs ir |
|----------------------------|----------|-----------|------------|---------------|
| 2^16, pq ≤ 4096 (11 M)     | 11.33 ms | 6.22 ms   | 8.76 ms    | 1.408x slower |
| 2^24, pq ≤ 4096 (11 M)     | 41.24 ms | 21.16 ms  | 24.65 ms   | 1.165x slower |
| 2^24, pq ≤ 512  (8 M)      | 38.91 ms | 21.11 ms  | 22.07 ms   | 1.045x slower |

Takeaways:
- Vs **eager**, the overlap driver wins comfortably (1.6–1.8x at 2^24),
  but that gain comes from the graph pipeline itself — the base IR
  driver is already 1.9x faster than eager.
- Vs the **base IR driver**, overlap is a net loss at these sizes. At
  the default cutoff the O(m^2) M builds/evals dominate (m=2048 alone
  is a 128 MiB `M_ab`): +3.5 ms at 2^24. Shrinking the regime to
  pq ≤ 512 removes almost all M cost but ~1 ms of structural overhead
  remains (+129 exe nodes, extra cross-stream events, transition copy).
- Root cause: with captured CUDA graphs, the per-round transcript
  latency the A/B split was meant to hide is already tiny — replay
  eliminates host launch overhead, so there is little serial latency
  left to overlap, while the challenge-free M restructuring adds real
  compute. The overlap structure is exposed and correctly
  multi-streamed (scheduler emits 2 streams / ~3.4k events); it is the
  workload balance that does not pay off at these sizes.

## Follow-up: pipelined windowed driver (blackbox CUDA kernels only)

`fractional_sumcheck_gpu_ir_pipelined` in `fractional_ir.rs` — a second
overlap variant that keeps the pipelining idea but drops the DSL M
kernels entirely (the DSL driver JIT'd ~800 modules at 2^24; this one
JIT's only the fixed-name scalar/transcript helpers plus one new
symbolic-size `frac_m_lerp` module, independent of input size).

Design (base skeleton = `fractional_sumcheck_gpu_ir`, not the DSL
driver):

- **Every dense outer round `j`** runs precompute-M windows from
  `base = 0` with `w = min(W, j − base)`; `W` is the
  `SWIRL_CUDA_GKR_PIPELINE_WINDOW` knob (default 5 = the CUDA build
  kernel's template cap). The pipelining is window-size-agnostic.
- **Per inner round `t`:** eval on M (blackbox
  `frac_precompute_m_eval_round`), observe/sample, then an
  out-of-place `fold_ef_frac_columns` into a fresh exact-size buffer
  that runs concurrently with the next eval/observe.
- **Window boundaries:** the next window's blackbox
  `frac_precompute_m_build_dev_challenge` absorbs the just-sampled
  challenge via its inline fold (reads the pre-fold buffer), so no
  fold bubble.
- **Round boundaries:** round `j+1`'s tree revert (in place on
  `layer`; ordering vs readers = graph insertion order) and its
  window-0 M are emitted inside round `j`. λ is not yet sampled, so
  window-0 M is built twice at host λ = 0 and λ = 1 (challenge-free
  blackbox `frac_precompute_m_build` pair) and lerped after λ is
  sampled via the `frac_m_lerp` DSL module — exact because the build
  is affine in λ. Both overlap round `j`'s final fold, claim observes,
  and `mu` sample.
- **Fallbacks:** virtual-compact inputs fall back to
  `fractional_sumcheck_gpu_ir` (dense revert/M addressing does not
  apply).

Node census (dense, `R = total_rounds`): `2(R−1)` λ-split build nodes,
`Σ_{j=1}^{R−1}(⌈j/W⌉−1)` dev-challenge builds, one lerp per round.

Tests (`fractional_sumcheck_gpu_ir_pipelined_*` in `fractional_ir.rs`):
e2e transcript match vs eager at 2^10 and 2^14, virtual fallback,
W ∈ {1, 3} knob variants (each also asserts the exact M-build node
census), and a `num_unique_modules() <= 50` compile assertion at 2^12.

Benchmark: third driver row (`ir_pipelined`) in
`bench_fractional_sumcheck_eager_vs_ir_overlap`; tune `W` by rerunning
with `SWIRL_CUDA_GKR_PIPELINE_WINDOW=1..5` (use
`FRAC_BENCH_DRIVERS=ir_pipelined` for the sweep — the three drivers
together exceed the kernel cache's 300-entry eviction cap, so full runs
re-JIT for minutes).

### Pipelined driver results (2026-08-17)

**Module count / compile time (the driver's whole point).** At 2^24:
15 unique JIT modules, compile ~33 s (vs `ir` 641 modules / ~381 s and
the DSL overlap driver's ~800). The pipelined harness compiles
**without fusion** by default (`pipelined_compiler_from_env`): all
heavy compute is blackbox, and v1 fusion of the tiny per-inner-round
scalar chains bakes round-specific structure (eq-table stage counts,
neighborhood shapes) into each cluster — 173 unique modules at 2^12
fused vs 15 unfused, roughly one fused variant per inner round.
`FRAC_IR_FUSION` still overrides.

**Window sweep** (captured-graph replays, ITERS=3, means;
`FRAC_BENCH_DRIVERS=ir_pipelined` processes):

| W | 2^16 | 2^24 |
|---|------|------|
| 1 | 12.40 ms | — |
| 2 | 7.68 ms | — |
| 3 | 6.77 ms | **26.51 ms** |
| 4 | **6.68 ms** | 27.65 ms |
| 5 | 7.51 ms | 33.40 ms |

W = 3–4 is the sweet spot; W = 5's `(32, 32)` builds are slower than
two smaller windows (the dev-challenge build at 1024 threads is
register-limited — it needs `__launch_bounds__` just to launch, see
below).

**Stream sweep** (`FRAC_BENCH_STREAMS=8`, i.e.
`ListSchedulerV1::max_concurrency = 8`, vs the default 2; 2^24,
pipelined-only processes, ITERS=3 means):

| W | 2 streams | 8 streams |
|---|-----------|-----------|
| 2 | — | 28.71 ms |
| 3 | 26.51 ms | **23.54 ms** |
| 4 | 27.65 ms | 24.43 ms |
| 5 | 33.40 ms | 30.41 ms |

8 streams is a consistent ~9–12% win across windows (more of the eq
chains, λ-split halves, and next-window builds run concurrently) and
does not change the W ordering — W=3 remains best, now 23.5 ms vs
eager's ~13.3 ms steady in the same process. Peak pool grows ~6%
(1.87 GB → 2.0 GB) from the extra concurrency, and the memory plan
takes ~25 s vs ~28 s (same 6997 events).

**Planner tiebreak fix (post-sweep).** The 8-stream NSYS profile
showed transcript sponge ops strictly serialized after the
`precompute_m_build` kernels despite having no data dependency. Root
cause: `plan_raw` passes a uniform `est_time = 1.0` to `ListV1`, whose
stream pick minimizes estimated start time with ties broken by lowest
stream index — with uniform costs, start times tie constantly, so the
serial sponge chain was repeatedly co-scheduled onto (and FIFO-queued
behind) the stream holding a heavy build.

A bytes-touched cost proxy was tried first and *fixed the boundary
overlap but regressed the 2^24 replay ~5%* (in-profile pipelined
iters 24.28 ms → 25.47 ms, `frac_pipelined_24_s8_costfix.nsys-rep`):
aligning the two profiles' transcript chains op-by-op showed the
regression concentrated in `sponge_observe_ext` waits at round
boundaries — a bytes-weighted bottom level rates bulk prefetch work
(eq chains, next-window builds) as critical and delays the
claim-producing folds the transcript actually waits on, whereas
uniform cost ≈ chain depth matches the true (launch-latency-bound,
serial-transcript) critical path.

Final fix: keep uniform costs, change only the stream-pick tiebreak in
`ListSchedulerV1::commit` to prefer the chain predecessor's stream
(the producer of the latest-arriving input, tracked in `buf_ready`).
`GRAPH_EXE_DUMP_SCHEDULE=1` (new knob in `graph_exe.rs`) confirms the
sponge chain now stays back-to-back on its own stream while builds
land elsewhere with only true-dependency (`frac_m_lerp`) waits.
Measured on an uncontended GPU
(`frac_pipelined_24_s8_affinity.nsys-rep`, 2^24, 8 streams, W=3):
replay 24.03–24.15 ms vs 24.28–24.46 ms lowest-index / 25.47–25.53 ms
bytes-cost; sponge∩build kernel overlap 228 µs vs 85 / 46 µs;
in-range kernel concurrency 1.42x vs 1.40x / 1.32x; transcript
inter-op wait sum 14.57 ms vs 14.90 / 15.98 ms. Peak pool unchanged
(2.0 GB, same 6997 events).

**Process-hygiene finding (invalidates the earlier eager baseline).**
Eager timings are strongly inflated by resident graph-exe state:
steady-state eager at 2^24 is **~13.4 ms** in a lean process (matching
the ~15 ms expectation) but 19.7 ms with the `ir` graph (3.8 GiB
scratch pool) resident and ~41 ms with all three drivers resident —
the earlier "ir 1.9x faster than eager" table row was an artifact of
that inflation. Captured-graph replay times are stable across process
states (<5% shift), so replay-vs-replay comparisons remain valid.

**Cross-driver comparison at 2^24** (each replay measured in its own
lean process; eager = lean steady state): eager ~13.4 ms < `ir`
21.1 ms < `ir_pipelined` (W=3) 26.5 ms. At 2^16: eager ~4.4 ms < `ir`
6.06 ms < `ir_pipelined` (W=4) 6.68 ms. The pipelined driver trades
~25% replay time vs the base IR driver for a 43x module-count / ~10x
compile-time reduction; with the corrected eager baseline, no graph
driver currently beats lean eager at these sizes.

**Launch-bounds fix.** The dev-challenge M build (`DEV_CH=true`) at
w=5 launches `(32, 32)` = 1024 threads and exceeded the register file
(`cudaErrorLaunchOutOfResources`); `precompute_m_build_partial_kernel`
now carries `__launch_bounds__(2^(2W))` (gkr.cu), which caps registers
for exactly the block shape each instantiation uses.

**NSYS profile** (`target/nsys/frac_pipelined_24.nsys-rep`; LOG_N=24,
W=3, all four workloads in one capture range, 3 NVTX-wrapped iters
each, `--cuda-graph-trace=node --gpu-metrics-devices=0`). Same-process
per-iter NVTX ranges: eager 55.4–64.9 ms (all three graph exes
resident — the inflation above, plus gpu-metrics sampling overhead),
`ir` ~21.7 ms, `ir_overlap` ~24.1 ms, `ir_pipelined` ~28.0 ms. The
graph replays match their lean-process numbers within ~6%, again
confirming replays are robust to process state while eager is not.
Note nsys's default `--capture-range-end=stop-shutdown` kills the test
at `cudaProfilerStop`, so the bench's printed summary is lost under
nsys (pass `--capture-range-end=stop` to keep it).
