# Porting the logup-zerocheck phase to the Graph IR

This is the phase-specific companion to
[`crates/compiler/gpu_ir_porting_guide.md`](../../crates/compiler/gpu_ir_porting_guide.md).
That guide states the four principles and uses the fractional-GKR port as its
running example; it is the law and this document does not restate it. What
follows is what those principles mean for the **logup-zerocheck** phase —
the stage map, the conventions this port adds, and the traps that actually cost
us commits.

Milestone status and evidence live in
[`logup-zerocheck-graph-ir-progress.md`](./logup-zerocheck-graph-ir-progress.md).
Nothing here is a status claim; read the checklist for that.

Source of truth for every statement below: `83ae7e1b`.

- Eager phase: `crates/cuda-backend/src/logup_zerocheck/mod.rs`,
  `prove_zerocheck_and_logup_gpu`.
- Graph-IR mirror: `crates/cuda-backend/src/logup_zerocheck/zerocheck_ir.rs`,
  `logup_zerocheck_gpu_ir`, declared `#[cfg(feature = "graph-ir")] pub mod
  zerocheck_ir;` (`graph-ir` is a default feature, so a plain `cargo check -p
  openvm-cuda-backend` compiles it).

---

## Scope and non-goals

**In scope.** Stages C (univariate round 0), D (the `n_max` MLE rounds) and E
(column openings) of `prove_zerocheck_and_logup_gpu`.

**Not in scope, and deliberately so.**

- Stage A (grinding) has no graph-IR counterpart at all —
  `FiatShamirTranscriptGraphIR` has no `grind`.
- Stage B (`alpha_logup`/`beta_logup`, GKR input evaluation, fractional
  sumcheck) belongs to `fractional_ir.rs`. Seed this phase's transcript with
  `DuplexSpongeGpuIR::from_live` so the graph chains onto the live Fiat–Shamir
  stream instead of restarting it.
- **Wiring.** `logup_zerocheck_gpu_ir` is *enablement*, exactly as
  `fractional_ir.rs` was: nothing in production calls it, and `mod.rs`'s eager
  call is untouched. The eager path is the oracle and must stay behaviourally
  unchanged; widening an eager item to `pub(super)` so the mirror can reuse it
  is the author's own practice and is fine, changing eager logic is not.
- **Tier 1 (blackbox-first).** Every kernel wrapper inserts a
  `GraphNode::BlackboxKernel` that calls the same
  `crate::cuda::logup_zerocheck` entry point the eager path calls. No
  `ir::Module` DSL ports live in this module yet. That is the same order the
  fractional port went in — blackbox mirror (`fractional_ir.rs`), then
  structured DSL modules (`fractional_ir_utils.rs`), then a DSL-first driver
  (`fractional_sumcheck_gpu_irv2.rs`) — and it is what makes the
  byte-equality oracle possible at all: a blackbox and its eager twin launch
  the *same* kernel, so a mismatch is a wiring bug and never a numerics
  difference.

---

## 1. Eager → IR stage map

| Stage | Eager (`mod.rs`) | IR mirror (`zerocheck_ir.rs`) | Where the values come from |
|---|---|---|---|
| C.0 | `lambda = sample_ext()` | `transcript.sample_ext(g)`; node emitted so the sponge advances | value from `plan.lambda_pows` |
| C.1 | `precompute_lambda_combinations` | `precompute_lambda_combinations_ir` | graph inputs (keygen-static monomial streams) |
| C.2 | `EqEvalLayers::new_rev` eq tree | `eq_hypercube_interleaved_stage_ext_ir` per layer | `xi` still a host `EF` — see §3 |
| C.3 | logup numer/denom combinations | `precompute_logup_{numer,denom}_combinations_ir` | graph inputs |
| C.4 | round-0 evaluators | `zerocheck_ntt_eval_constraints_ir`, `logup_bary_eval_interactions_round0_ir` | descriptor arrays + graph inputs |
| C.5 | **host seam**: D2H, transpose, `from_geometric_cosets_evals_idft`, `s_0_poly` assembly | not ported | `plan.logup_sum_claims`, `plan.s_0_coeffs` enter as `Const` |
| C.6 | `fold_ple_evals` (+rotate), `fold_selectors_round0` | `fold_ple_from_evals_ir`, `fold_selectors_round0_ir` | `r_0`-derived scalars by value |
| D | `n_max` MLE rounds: `interpolate_columns`, evaluator dispatch, `compute_batch_s_poly`, observe/sample, two folds | `interpolate_columns_ir`, four `*_batch*_ir` evaluators, `observe_and_update_zerocheck_round_ir` (`batch_s_ring_pre`/`_post`), `batch_fold_mle_ir` | **computed on device**; `r_1..r_n` never resolve on the host |
| E | D2H every folded matrix, split doubled width into `(orig, rot)`, reorder common-main-first, observe | not ported | `plan.opening_claims` enter as `Const`; `column_openings` are the raw folded buffers in build order |

The round loop is unrolled with an ordinary Rust `for` at graph-build time —
the author's structure for `fractional_sumcheck_gpu_ir`.

---

## 2. The transcript: control on the host, value on the device

This split is the single idea that makes the port tractable, and it is worth
stating explicitly because it looks like a Principle-4 violation and is not.

- The transcript's **control** state (`absorb_idx` / `sample_idx`) lives on the
  host and selects *which module* each `observe_ext` / `sample_ext` emits. It
  is a function of how many observes have happened, which is a graph *shape*
  property known at build time — not a function of any buffer's contents.
- The transcript's **value** state (the sponge) threads through the graph as a
  `BufId`. It is never read to the host.

The final sponge state must be registered as a graph output. Without it the
whole Fiat–Shamir chain is dead code and DCE deletes every transcript node —
a graph that compiles, runs, and proves nothing.

---

## 3. Challenge ledger

Principle 4's question — *"does this host value depend on a buffer in the
graph?"* — applied to every scalar this phase captures by value. Legitimate
host values (dimensions, `step`, `round`, `g_shift`, widths, offsets, shape
booleans) are not listed; they depend on the plan, not on a kernel output.

| Origin | Current representation | Required representation | Consumer | Closing test |
|---|---|---|---|---|
| `xi` — fractional sumcheck, padded to `l_skip + n_global` | host `EF` in `plan.xi`, captured by value | `BufId` per `xi[j]`, ideally a structured module like the fractional twin | `eq_hypercube_interleaved_stage_ext_ir`, eq-tree layers, the ring | `eq_hypercube_interleaved_stage_ir_matches_eager` extended to two runtime `x_i` values on one compiled graph, plus a constant-challenge sabotage |
| `denom_sum_init` — accumulated from `eq_3bs` and powers of the logup challenge | host `EF` in `plan.round0_denom_sum_init` | `BufId` through a `_dev_challenge` entry point | `logup_bary_eval_interactions_round0_ir` | round-0 logup equality on two runtime values |
| `is_first` / `is_last` — `eval_eq_uni_at_one(l, r_0, …)` | host `EF` pair in `plan.fold_selector_scalars` | `BufId`; also needs a device `eval_eq_uni_at_one` | `fold_selectors_round0_ir` | selector-fold equality on two runtime `r_0` |
| `bus_term_sum` — logup challenge, riding **inside** an uploaded ctx struct | host `EF` field of `LogupMonomialCommonCtx` | needs a monomial-ctx descriptor writer taking it as a `BufId` | `logup_monomial_batched_ir` | monomial evaluator equality on two runtime values |
| `r_0` | host `EF` in `plan.r_0`; also pre-baked into `inv_lagrange_denoms_r0` | `BufId` (the sampled buffer `r0_buf` already exists and is retained) | round-0 launchers | round-0 chain on two runtime samples |
| `lambda_pows`, `mu_pows` | `Const` producers | `BufId` when stage A/B joins the graph | round-0 and eq tree | out of this module's scope |
| `s_0_coeffs`, `logup_sum_claims`, `opening_claims` | `Const` producers — these are **proof messages** | derived by the graph, not supplied | the transcript | the phase-proof oracle (§12) |
| `r_1 .. r_n` | **closed** — sampled into device buffers, never resolved on the host | — | `batch_fold_mle_dev_challenge`, the ring | `observe_and_update_zerocheck_round_ir_matches_eager` |
| batched sumcheck polynomial `s_round` | **closed** — computed on device by `batch_s_ring_pre` | — | the transcript | `batch_s_ring_pre_post_matches_eager`, `observe_and_fold_zerocheck_round_ir_matches_eager` |

Every open row above carries a `TODO(cc-ir)` at its site in `zerocheck_ir.rs`
with a `WHY:` and a `RISK:` line. Keep that discipline: a by-value challenge
with no TODO is indistinguishable from an oversight.

---

## 4. Eager entry point → `*_ir` inserter

The phase has **17** `extern "C"` entry points. Fourteen `*_ir` inserters cover
fifteen of them (one inserter serves two entry points); two further inserters
drive kernels this port added. Each inserter makes exactly one
`insert_blackbox_kernel` call — 16 inserters, 16 call sites in the module's
production code.

| Eager entry point | Inserter | Note |
|---|---|---|
| `_precompute_lambda_combinations` | `precompute_lambda_combinations_ir` | |
| `_precompute_logup_numer_combinations` | `precompute_logup_numer_combinations_ir` | |
| `_precompute_logup_denom_combinations` | `precompute_logup_denom_combinations_ir` | |
| `_eq_hypercube_interleaved_stage_ext` | `eq_hypercube_interleaved_stage_ext_ir` | interleaved map, **not** the fractional non-overlapping one |
| `_zerocheck_ntt_eval_constraints` | `zerocheck_ntt_eval_constraints_ir` | Principle-1 exception (2 kernels) |
| `_logup_bary_eval_interactions_round0` | `logup_bary_eval_interactions_round0_ir` | Principle-1 exception (2 kernels) |
| `_fold_ple_from_evals` | `fold_ple_from_evals_ir` | takes `FoldPleDst`, see §8 |
| `_fold_selectors_round0` | `fold_selectors_round0_ir` | |
| `_interpolate_columns` | `interpolate_columns_ir` | column table order matters, see §11 T6 |
| `_zerocheck_batch_eval_mle` | `zerocheck_batch_eval_mle_ir` | Principle-1 exception |
| `_logup_batch_eval_mle` | `logup_batch_eval_mle_ir` | Principle-1 exception |
| `_zerocheck_monomial_batched` | `zerocheck_monomial_batched_ir` | one inserter, `par_y: bool` selects the entry point; Principle-1 exception |
| `_zerocheck_monomial_par_y_batched` | ″ | ″ |
| `_logup_monomial_batched` | `logup_monomial_batched_ir` | Principle-1 exception (3 kernels) |
| `_batch_fold_mle` | `batch_fold_mle_ir` | routed to `batch_fold_mle_dev_challenge` |
| `_zerocheck_eval_mle` (single-AIR) | **absent** | single-AIR fallback, not ported |
| `_logup_eval_mle` (single-AIR) | **absent** | single-AIR fallback, not ported |
| — | `batch_s_ring_pre_ir` | **new** kernel (`cuda/src/logup_zerocheck/ring.cu`); replaces host `compute_batch_s_poly` |
| — | `batch_s_ring_post_ir` | **new** kernel; updates `{tilde, prev_s_eval, eq_n, eq_sharp_n}` on device |

### Principle-1 exceptions, declared

Seven of this phase's entry points launch more than one CUDA kernel behind one
`extern "C"` symbol (an evaluator pass plus `final_reduce_block_sums`, or the
logup monomial's numer/denom/reduce triple). Splitting them into one-kernel
symbols means going from 17 to 27 `extern "C"` entry points, which is a much
larger change than the port.

The exception is safe **only** because the node declares the complete
read/write union across every kernel it launches — which is what §7 is about.
Each exception is named at its type with that rationale. Do not add a new one
silently: if a wrapper grows a second launch, either split it or write the
union down.

---

## 5. Strategy dispatch: a seven-way eager choice, mirrored three ways

The eager path chooses its stage-D evaluator per trace, and the choice is
richer than the mirror admits:

| Eager arm | Predicate | Mirror (`RoundEvalKind`) |
|---|---|---|
| late logup monomial | `num_y == 1`, `has_interactions` | `Monomial` |
| late zerocheck monomial | `num_y == 1`, has monomials | `Monomial` |
| early logup monomial | `num_y <= monomial_num_y_threshold` | `Monomial` |
| early logup batched DAG | `num_y > monomial_num_y_threshold` | `Dag` |
| early zerocheck monomial | `num_y <= monomial_num_y_threshold` | `Monomial` |
| early zerocheck batched DAG | `num_y >` threshold **and** `num_monomials >= DAG_FALLBACK_MONOMIAL_RATIO * rules_len` | `Dag` |
| early zerocheck monomial par-Y | `num_y >` threshold **and** the ratio test fails | `MonomialParY` |
| single-AIR fallback launcher | a memory-limit bin ends up holding one trace | **not represented** |
| FFD bin-packing | sort by intermediates size descending, then `find_batch_end` under `memory_limit_bytes` | **not represented** |

`monomial_num_y_threshold` is a *production* value: 512 for app proofs
(`log_blowup == 1`) and 64 otherwise. It is not a test knob, which is why a
production-faithful whole-phase fixture cannot claim coverage from the DAG arm
alone — the final early rounds fall below it.

`RoundEvalKind` is a three-way *summary*. The missing branches are choices
about how many AIRs share one launch, not about what is computed.

> **RISK.** A real port must emit the same partition the eager path picks, or
> the graph's launch geometry (`num_blocks`, `air_block_offsets`) will not
> match the ctx arrays staged for it. Classifying independently in the mirror
> is a stand-in; the fix is a **shared** pure planner that returns launch-order
> batch records, consumed by eager and IR alike. Copying the seven-way decision
> into `zerocheck_ir.rs` reproduces the drift, it does not remove it.

The current fusion frontend cannot express this selection: candidates are
compiler-synthesized equivalents over the seed graph's existing value classes,
and opaque blackboxes carry no per-candidate access or cost detail. Different
strategies need different launchers, scratch, contexts and batch partitions.
Treat an `alternative_region` API as a separate RFC, not as part of this port.

---

## 6. Conventions this port follows

1. **Inserter name = eager name + `_ir`.** One `insert_blackbox_kernel` per
   inserter. Callers compose inserters; callers never call
   `insert_blackbox_kernel` directly.
2. **The closure only launches.** No host math, no `synchronize`, no D2H, no
   transcript access, no branch on runtime data. It reconstructs borrowed
   `DeviceBuffer` views with `from_raw_parts`, calls the safe wrapper, and
   `mem::forget`s **every** view so the runtime keeps ownership. There are
   zero `to_host` / `copy_from_device` / `synchronize` calls in this module's
   production code; keep it that way.
3. **One mutation keeps one `BufId`.** §8.
4. **A descriptor's pointee set is declared by its consumer.** §7.
5. **Positional outputs; the caller registers them.** §9.
6. **Equality oracles are PoW-controlled.** §10.
7. **Every unresolved thing carries `TODO(cc-ir)` + `WHY:` + `RISK:`** at its
   site.

---

## 7. The descriptor / liveness contract

Seven of this phase's entry points are runtime *interpreters* driven by arrays
of `#[repr(C)]` context structs that used to embed raw device pointers. A
struct of host-baked addresses is an opaque leaf: the planner cannot see
through it, so alias analysis is impossible and every pointee is invisible.

`MainMatrixDesc`, `EvalCoreCtx`, `ZerocheckCtx`, `LogupCtx` and
`BatchSRingTraceDesc` therefore hold **no pointers**: every device-pointer
field is a `BaseOff` byte offset into the `GraphExe`'s unified pool, and the
launcher takes the pool base as a kernel argument. Offsets come from
`GraphExe::plan().offsets`, so an array is host-computable after `compile()`
and uploaded once as a registered graph input.

**Base+offset does not retire the liveness obligation.** An offset into a pool
slot the planner has since reassigned is exactly as wrong as a stale pointer.
What it retires is the *materializer* node whose own access set could be
under-declared. The obligation that remains is:

> Every buffer reachable by dereferencing a descriptor the node consumes must
> appear in that node's declared access set — as a read, or (for a write-only
> destination table) as an explicit output.

The mechanism that keeps this honest is a **single writer**. Each descriptor
variant has exactly one encoder (`write_main_matrix_desc`,
`write_eval_core_ctx`, `write_zerocheck_ctx`, `write_logup_ctx`,
`write_batch_s_ring_trace_desc`), generic over an `OffSink`. Running it with a
`ReadCollector` sink *derives* the read set; running it with the byte encoder
*produces* the bytes. Both traverse the same fields, nested descriptor arrays
included. A pointee therefore cannot reach the device without its buffer
flowing into the declared access set — you would have to add a field to the
writer to leak one, and that same edit adds it to the read set.

Consumers then append what the writer derived: round-0 main-matrix pointees to
both evaluator input lists, stage-D context reads *plus* nested main-descriptor
reads before `eval_node_bindings`, ring evaluator outputs to `batch_s_ring_pre`,
interpolation tables their source buffers, fold input tables their sources as
inputs and fold output tables their destinations as **outputs**.

That last case is the documented exception, not a bug: `batch_fold_mle`'s
output table points at destinations the kernel writes and never reads.
Declaring them as outputs gives the planner the write lifetime it needs without
pretending the kernel reads them.

`eval_node_bindings(fixed, written, read)` assembles `(inputs, modifies)` from
a positional prefix, a written set and a read set; later duplicates fold into
the first binding, and a buffer that is both read and written comes out with
`modifies = true`.

**Still opaque, and therefore still a gap:** the three *monomial* ctx structs
(`MonomialAirCtx`, `LogupMonomialCommonCtx`, `LogupMonomialCtx`) are not on the
base+offset ABI, their arrays are zero-filled placeholders, and their
prospective reads are still hand-written (`push_monomial_reads`, the last
hand-written read list in the phase). That is an incomplete strategy port and a
drift risk — not an omitted read from an encoded descriptor.

---

## 8. One mutation keeps one `BufId`

`crates/compiler/notes.md` states the rule: a real buffer is *mutated* iff some
node has it in both its consumed and produced sets, and the ATG then inserts
the ordering edges. A blackbox's `carried_outputs` (⊆ `inputs`) is the
representation of a read/write on the same allocation.

The corollary, learned the hard way (§11 T1): **a partial write into an
existing buffer must be a carried input, never a fresh `BufId`.** Renaming the
destination — even with an alias recorded — gives the two ids different pool
offsets, so half the data lands in one allocation and half in another.

The port encodes this in a type rather than a comment:

```rust
pub enum FoldPleDst {
    /// A fresh buffer this launch is the sole producer of: declared as the
    /// node's output.
    Fresh(BufId),
    /// An existing buffer this launch overwrites part of: declared as a
    /// carried input (`modifies = true`), never as a renamed output.
    InPlace(BufId),
}
```

Round 0's rotated fold, which writes the upper half of a doubled-width buffer
the previous launch produced, is `InPlace`. There is no `alias_bufs` call
anywhere in the crate, and a test asserts there is none.

---

## 9. Positional outputs; the caller registers them

`logup_zerocheck_gpu_ir` returns a `ZerocheckPhaseProofIR`:
`round0_zc_evals`, `round0_logup_evals`, `evaluator_outputs`,
`sumcheck_round_polys`, `r`, `column_openings`, `transcript_state`,
`descriptors`.

The direction of travel, matching the author's `FracSumcheckProofIR`, is that
the builder returns positions and the **caller** decides what to export.
Registering a buffer as a graph output pins it — it must be materialized and
survive to the end of the run — so which intermediates become outputs is a
caller's optimization decision, not the phase's. The author's fractional
harness copies to fresh outputs in fusion-sensitive equality tests rather than
registering the source buffer; prefer that shape. Direct registration stays
fine for dumps.

Note for reviewers: the builder at `83ae7e1b` still registers some artifacts
itself. That is tracked as `A3-OUTPUTS` in the checklist and is not yet done.

Two ordering facts a consumer must respect:

- `column_openings` are the **raw folded buffers in build order**, not the
  proof's order. The eager exit splits the doubled width into `(orig, rot)` and
  reorders common-main-first before observing. Both transforms are
  keygen-static permutations, but the permutation is not carried by the plan
  yet, so a consumer must apply it. The *observes* use the plan's already
  ordered claim list, so the transcript is right regardless.
- `descriptors` must be bound to the compiled `GraphExe` before `run`. The
  arrays are registered graph inputs, so `run` refuses otherwise.

---

## 10. Test and RED protocol

Tests live in an in-file `mod zerocheck_ir_tests`, like the author's in-file
test modules on the fractional side. The shape of a good one:

1. **Seed deterministically** (`StdRng`), build ragged shapes on purpose —
   `(log_height, width) = (4,3),(5,2),(3,5),(6,1)` catches more than a square
   fixture ever will.
2. **Compare raw device bytes** against the eager path, not derived summaries.
3. **Build the eager reference independently.** If the test constructs the
   reference *from* the mirror's own data structures, it cannot fail for the
   class of bug that matters (§11 T4, T6).
4. **Include a sabotage leg.** Perturb one element of one input and assert the
   bytes differ. A test with no teeth passes on an all-zero graph.
5. **Show the RED.** Revert the fix, run, and paste the failure into the
   report/checklist. A fix whose RED was never observed is a claim, not
   evidence. The checklist's `done` status requires a SHA *and* the exact
   passing command and result.
6. **Re-upload descriptor inputs before every run.** Graph inputs are not
   preserved across an execution under the shipped scheduler; a second `run`
   that skips `DescriptorPlan::upload` reads whatever the planner has since put
   in that slot (§11 T5b).

### PoW-controlled equality oracles

Any fixture that drives a real prove must set `params.logup.pow_bits = 0`
before proving. Both grinds are nondeterministic — the CPU transcript uses
`find_any`, and the GPU sponge takes the first `atomicCAS` winner — so a
nonzero `pow_bits` makes byte equality impossible. Both short-circuit to
`F::ZERO` at `bits == 0` and `check_witness` accepts unconditionally there, so
the verifier still passes.

**Accuracy note for reviewers:** no fixture in `zerocheck_ir_tests` at
`83ae7e1b` needs this yet. They drive the transcript only through
`observe_ext` / `sample_ext`, and both PoW paths are reachable only through
`grind`, which is never called — there is no nondeterminism to remove. The rule
above binds the first whole-prove fixture, which does not exist yet.

---

## 11. Traps

These are the expensive part of this document. Each one shipped, or nearly
shipped, on this branch.

### T1 — a rename split an in-place fold across two allocations

Round 0 folds a `need_rot` matrix with two launches into one doubled-width
buffer. The mirror spelled the second launch as a fresh `BufId` plus
`GraphBuilder::alias_bufs`. `alias_bufs` only records a parent id, and
`plan_memory` never reads the alias table — so the two ids got different pool
offsets. The lower half was written into one allocation, the upper half into
another, and every later round read the second one, whose lower half no launch
ever wrote.

Reproduced on GPU with exactly that signature:

```
RED PROBE: whole-buffer match=false lower-half match=false upper-half match=true
```

**Rule:** §8. One mutation, one `BufId`; carried input, never a rename.
`alias_bufs`'s doc promises a runtime pool-slot guarantee that no code
implements — do not build on it.

### T2 — control tables memset to zero fail *silently*

`batch_fold_mle`'s four control tables (`in_ptrs`, `out_ptrs`, `widths`,
`log_output_heights`) were `insert_memset(_, 0)` — the highest-frequency node
in stage D, with no TODO on them. `fold_mle` reads `width = widths[mat_idx]`
and every thread returns once `output_height * width == 0`. So: the kernel
launches, dereferences no null, raises no error, writes nothing, and the
destinations keep whatever the pool slot held. The graph produced **all zeros**
and nothing said so.

```
assertion `left == right` failed: batch_fold_mle_ir mismatch on matrix 0
  left:  [0, 0, 0, 0, 0, 0, 0, ... ]      <- the graph: ALL ZEROS
  right: [245, 200, 132, 41, 171, ... ]   <- the eager kernel
```

**Rule:** a zero-filled placeholder is not a neutral stand-in. Kernels that
early-return on a zero extent turn it into a silent no-op. Either fill the
table or make its absence a hard error — `PhaseInputBinder::bind` now refuses
and *names* the unbound buffers instead of zero-filling. Graph dumps do not
help here: they show the `Memset` node faithfully and cannot know the zeros are
invalid.

*Found while filling those tables:* the selector fold was folding **every**
trace every round, including traces already exhausted, where the output height
saturates to 1 and the kernel reads `input[1]` of a height-1 buffer. Invisible
while the tables were zeroed; an out-of-bounds read the moment they were real.

### T3 — the serializer read uninitialized struct padding

`DescElem::encode` built a `#[repr(C)]` value and took
`slice::from_raw_parts(v as *const T as *const u8, size_of::<T>())`. Every one
of these structs has padding. Rust never initializes padding, so the read is UB
*and* the encoded bytes are not a function of the descriptor's fields — which
matters doubly here, because nearly every oracle on this branch compares
encoded descriptor bytes.

```
assertion `left == right` failed: `MainMatrixDesc` encoded to different bytes
  left:  [0, 0, 173, 222, 0, 0, 0, 0, 120, 86, 52, 18, 0, 0, 0, 0]
  right: [0, 0, 173, 222, 0, 0, 0, 0, 120, 86, 52, 18, 165, 165, 165, 165]
```

The trailing bytes came back as the `0xA5` stack-poison pattern on the second
encode: observed, not inferred.

**Rule:** encode field-wise at `offset_of!` positions into a zero-filled buffer
of the struct's exact `size_of`. Never `bytes_of` a struct with padding. Assert
in the test that the struct *has* padding, so the test cannot pass vacuously if
the layout changes.

### T4 — eager and graph shared one decoder, so eager stopped being an oracle

Putting eager on the same `BaseOff` context structs and the same C++ `resolve_*`
decoders as the graph path — differing only by passing `pool_base = nullptr` —
looks like healthy deduplication. It is not. Every equality test on this branch
compares graph against eager, so a defect in the shared decoder, or a Rust/C++
layout drift, is applied identically to both sides and **no test can see it**.

Measured, by injecting a 4-byte skew into the shared `base_off_ptr`:

```
PASS  round0_one_element_descriptor_sabotage_changes_output   <- shared ABI: BLIND
PASS  round0_pool_offsets_match_absolute_addresses            <- shared ABI: BLIND
FAIL  test_monomial_vs_dag_equivalence                        <- independent: CATCHES IT
```

**Rule:** share the evaluator *body* — both sides must compute the same math —
but never the *decode*. The fix templates the kernels on the context type and
gives the eager path `*Raw` context structs and `_raw` entry points, so the two
sides share no encoder and no decoder. Three of the four evaluator families are
converted; the round-0 pair is not, and that gap is a `TODO(cc-ir)` with the
run above as its cost.

### T5 — two compiler docs that are wrong, in opposite directions

**(a) `GraphBuilder::register_input`'s doc comment says "Inputs must not be
written by any graph node (validated at compile time)". That is false.** The
validator in `graph_compiler.rs` says the opposite in its own comment —
*"Inputs may be written in-place by graph kernels (blackbox `carried_outputs`)"*
— and permits it.

What the validator *does* enforce is the pair of guards either side: a
registered input that no node reads is an error, and a buffer that is read but
never written and is not a registered input is an error. So when converting a
zeroed buffer into a registered input, **delete** the memset rather than
supplementing it, and sequence delete-then-register — the second guard turns
the compiler into your checklist for the ones you miss.

**(b) `notes.md` is right that inputs are not preserved, and it is easy to
build an API that cannot honour it.** `DescriptorPlan::bind` uploaded once and
offered no way to do it again: it begins with `set_scratch`, which rejects an
exe that already owns a pool. A second execution read whatever the planner had
put in those slots, and the decoder adds that to the pool base — wrong in-pool
data, or an invalid device address. Two in-tree sites depended on this and
worked only because they used ListV1, which pins graph inputs through the
schedule; the shipped default (ListV2) frees them. `bind` is now split into
one-shot `install_pool` and repeatable `upload`.

**Rule:** verify a compiler doc against the compiler's own validator before
building on it. Both of these cost a debugging session.

### T6 — a table whose order is a wrong answer with no error

The eager `interpolate_columns` table is `iter::once(sels).chain(mats)` —
**selectors first**. The mirror's `srcs` dependency list is
matrices-then-selectors. `srcs` order is harmless because it is only a
dependency declaration; copying it into the *table* is a wrong answer with no
error anywhere.

**Rule:** a dependency list and a device-visible table are different objects
even when they hold the same ids. Build the table from an explicit order, and
have the test construct the eager reference **independently** so a permuted
builder cannot pass.

---

## 12. What "verified" means here

Two labels, deliberately separate, because collapsing them either understates
real evaluator work or overstates the phase.

**`strategy-verified`** — for a *forced* stage-D strategy, real trace and
keygen inputs flow through the evaluator, the ring, the challenge and the fold,
and the result is **byte-equal to eager**. That is a genuine claim about the
machinery, and it is the bar each strategy slice must clear.

**`phase-proof-verified`** — the graph *derives and observes* the round-0
messages, all challenges, and the ordered openings, instead of inserting plan
constants.

At `83ae7e1b` the phase is **not** `phase-proof-verified`, and no amount of
green evaluator tests would make it so. `plan.s_0_coeffs`,
`plan.logup_sum_claims` and `plan.opening_claims` still enter as `Const`
producers because the round-0 iDFT chain and the stage-E permutation are still
on the host. Read the graph accordingly: it is not device-input-only. The
checklist keeps `PHASE-PROOF-ORACLE` red until those plan fields are gone.

A related bar worth stating once: a *whole-phase* fixture is a different thing
again from a strategy slice, because it must cover every arm the production
dispatch actually reaches (§5).

---

## 13. Known gaps

Tracked with status and evidence in
[`logup-zerocheck-graph-ir-progress.md`](./logup-zerocheck-graph-ir-progress.md).
In short, at `83ae7e1b`:

- Round 0's iDFT chain (stage C.5) and stage E's D2H + split + reorder are host
  seams; the messages either side of them are plan constants.
- `xi`, `denom_sum_init`, `is_first`/`is_last`, `bus_term_sum` and `r_0` are
  still challenges captured by value.
- The three monomial ctx structs are host-assembled zero placeholders, so the
  monomial strategy arms are not runnable.
- `RoundEvalKind` is a three-way summary of a seven-way eager dispatch, and the
  single-AIR fallbacks and FFD bin-packing are unrepresented.
- The round-0 entry points still share their decoder with eager (T4).
- Proving-key rule tables enter as `Static` addresses, valid only while that
  exact `DeviceMultiStarkProvingKey` lives — do not cache a compiled graph
  beyond the pk's lifetime.
- Nothing in production calls `logup_zerocheck_gpu_ir`.
