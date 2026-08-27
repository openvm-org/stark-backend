# logup-zerocheck Graph-IR Port — Progress

Tracks the port of the logup-zerocheck phase onto the Graph IR. Conventions,
stage map and traps are in
[`logup-zerocheck-graph-ir-porting.md`](./logup-zerocheck-graph-ir-porting.md);
this file is status and evidence only. Update it in the same commit as the
change it describes.

**Baseline for every claim below: `83ae7e1b`** (8 commits on top of
`feat/crypto-compiler`).

## How to read this

`status` is one of `not started`, `in progress`, `blocked`, `done`.

**`done` requires a commit SHA and the exact passing command and result.** A
row without both is not done, however finished the code looks.

Two verification labels, kept deliberately apart:

- **`strategy-verified`** — for a *forced* stage-D strategy, real trace and
  keygen inputs flow through the evaluator, the ring, the challenge and the
  fold, and the result is byte-equal to eager.
- **`phase-proof-verified`** — the graph *derives and observes* the round-0
  messages, all challenges and the ordered openings, instead of inserting plan
  constants.

**At `83ae7e1b` the phase is not `phase-proof-verified`, and no strategy row
turning green will make it so.** `plan.s_0_coeffs`, `plan.logup_sum_claims` and
`plan.opening_claims` still enter as `Const` producers; the round-0 iDFT chain
and the stage-E opening permutation are still host seams. This graph is not
device-input-only. `PHASE-PROOF-ORACLE` stays red until those plan fields are
gone.

Two further facts a reviewer should have before reading the table:

- **The phase is enablement-only.** Nothing in production calls
  `logup_zerocheck_gpu_ir`; `mod.rs`'s eager call is untouched. The eager path
  is the oracle and its behaviour is unchanged.
- **The port is tier 1 (blackbox-first).** Every wrapper launches the same
  `crate::cuda::logup_zerocheck` entry point the eager path launches. No
  `ir::Module` DSL ports live in this module yet.

## Milestone status

| ID | scope | kind | status | dependency | exit oracle | landed SHA |
|---|---|---|---|---|---|---|
| M0-MIRROR | phase expressed as a graph-IR module (stages C/D/E, 16 inserters) | code | done | — | module compiles + shape tests | `1dc5ea5c` |
| M1-DEVCTX | evaluator contexts materialized on device; device-challenge fold; cacheable transcript seeding | code | done | M0-MIRROR | full suite green | `94a64e9f` |
| M2-ACCESS-SET | declared access set derived from the pointers actually bound | code | done | M1-DEVCTX | under-declaration sabotage fails | `6398ff48` |
| M3-REBASE | mirror adapted to the rewritten compiler API | code | done | M2-ACCESS-SET | full suite green | `1fa2a0d5` |
| M4-BASE-OFF | embedded device pointers replaced by a base+offset context ABI | code | done | M3-REBASE | CUDA layout probes + pool-reuse byte probe | `f31e750f` |
| M5-REAL-INPUTS | real trace/keygen data bound into the graph; zero placeholders removed | code | done | M4-BASE-OFF | graph-vs-eager raw bytes on filled tables | `ffb3d999` |
| M6-REVIEW-FIXES | five external-review defects (T1–T5 in the guide) | code | done | M5-REAL-INPUTS | five red-green pairs, each run on GPU | `ad22ba77` |
| M7-RING | steady-round sumcheck polynomial computed on device; `r_1..r_n` device-resident | code | done | M6-REVIEW-FIXES | ring vs host `compute_batch_s_poly`, raw bytes | `83ae7e1b` |
| A1-LIVENESS | descriptor pointees are declared by their consumer | code | done | M2-ACCESS-SET | single-writer/collector audit + 4 structural tests | `83ae7e1b` |
| A1-UNIVERSAL-TEST | one structural assertion over *every* descriptor-root consumer | code | not started | A1-LIVENESS | all five sabotage cases fail | — |
| B3-GUIDE | phase-specific porting guide, checked in | process | in progress | — | linked from `docs/README.md` | — |
| B4-CHECKLIST | this file | process | in progress | — | linked from `docs/README.md` | — |
| B5-PLAN-REVIEW | IR author reviews guide + checklist **before** further code commits | process | blocked | B3-GUIDE, B4-CHECKLIST | author's review recorded in this file | — |
| A2-XI | every `xi[j]` becomes a graph `BufId` | code | not started | B5-PLAN-REVIEW | two runtime `x_i` on one compiled graph; constant-challenge sabotage fails | — |
| A3-OUTPUTS | builder returns positions; the caller registers outputs | code | not started | B5-PLAN-REVIEW | builder leaves `output_bufs` untouched; caller positions match the documented order | — |
| B1-FRONTEND | additive typed buffer/blackbox frontend in the compiler | code | not started | B5-PLAN-REVIEW | emitted `KernelNode` identical to the raw API; serializer round-trips green | — |
| B2-STRATEGY-PLAN | one shared pure planner for eager and IR strategy policy | code | not started | B5-PLAN-REVIEW | planner characterization matches eager decisions on every fixture | — |
| MONO-CTX-DESC | monomial ctx structs onto the descriptor ABI (they are zero placeholders today) | code | not started | B2-STRATEGY-PLAN | encoded bytes match an independently built eager ctx | — |
| R0-RAW-TWIN | round-0 entry points get a `_raw` twin so eager stops sharing the decoder | code | not started | — | injected decoder skew fails a round-0 differential | — |
| S-ZC-DAG | early zerocheck batched-DAG strategy slice | code | not started | B2-STRATEGY-PLAN | `strategy-verified` | — |
| S-LG-DAG | early logup batched-DAG strategy slice | code | not started | S-ZC-DAG | `strategy-verified` | — |
| S-ZC-MONO | early zerocheck monomial strategy slice | code | not started | MONO-CTX-DESC | `strategy-verified` | — |
| S-ZC-PARY | early zerocheck monomial par-Y strategy slice | code | not started | S-ZC-MONO | `strategy-verified` | — |
| S-LG-MONO | early logup monomial strategy slice | code | not started | MONO-CTX-DESC | `strategy-verified` | — |
| S-LATE | late traces (`num_y = 1`) launched separately | code | not started | B2-STRATEGY-PLAN | mixed-height fixture, `strategy-verified` | — |
| S-SINGLE-FFD | single-AIR fallback launchers + FFD bin-packing | code | not started | B2-STRATEGY-PLAN | partition matches eager exactly | — |
| R0-MESSAGES | round-0 messages derived by the graph, not `insert_const` | code | not started | R0-RAW-TWIN | `s_0_coeffs` / `logup_sum_claims` leave `ZerocheckPhasePlan` | — |
| R0-DERIVED-SCALARS | `denom_sum_init`, `is_first`/`is_last`, `bus_term_sum`, `r_0` become `BufId` | code | not started | A2-XI | round-0 chain on two runtime samples | — |
| E-OPENINGS | stage-E split + reorder expressed in the graph | code | not started | — | `column_openings` in proof order; `opening_claims` leave the plan | — |
| PHASE-PROOF-ORACLE | `phase-proof-verified` for the whole phase | code | not started | R0-MESSAGES, R0-DERIVED-SCALARS, E-OPENINGS, all `S-*` | whole-prove fixture, `pow_bits = 0`, byte-equal to eager | — |

---

## What landed

### M0-MIRROR — `1dc5ea5c`

**Plan.** Express stages C, D and E of `prove_zerocheck_and_logup_gpu` as a
graph-IR module mirroring `fractional_ir.rs`: one `*_ir` inserter per eager
entry point, one `insert_blackbox_kernel` per inserter, the round loop unrolled
at graph-build time, the transcript through `FiatShamirTranscriptGraphIR`.

**Acceptance.** The module compiles under the default `graph-ir` feature and
the phase graph builds and compiles for a synthetic plan.

**RED evidence.** Not applicable — new module.

**GREEN evidence.**

```
Summary [   4.083s] 5 tests run: 5 passed, 158 skipped
```

**Deviations.** Tier 1 only (blackbox, no DSL modules) — deliberate, see the
guide's *Scope and non-goals*. Seven inserters are declared Principle-1
exceptions because their eager entry point launches more than one kernel.

### M1-DEVCTX — `94a64e9f`

**Plan.** Materialize the evaluator context structs on device, add the
device-challenge fold entry point, and make transcript seeding cacheable
(`DuplexSpongeGpuIR::from_live`).

**Acceptance.** Full crate suite green; ABI probes agree with the C++ truth.

**GREEN evidence.**

```
Summary [ 273.672s] 161 tests run: 161 passed (6 slow), 8 skipped
```

**Deviations.** None recorded.

### M2-ACCESS-SET — `6398ff48`

**Plan.** Stop hand-writing the declared read set. Derive it from the same
writer that encodes the descriptor bytes, so a pointee cannot reach the device
without its buffer entering the access set.

**Acceptance.** An under-declared access set must fail a test, not merely look
wrong.

**RED evidence.** `evaluator_declares_every_referenced_buffer` and the
pool-reuse probe each have an explicit sabotage handle; the reports record
three separate `1 test run: 0 passed, 1 failed` states before the fix.

**GREEN evidence.**

```
Summary [ 251.476s] 164 tests run: 164 passed (5 slow), 8 skipped
```

**Deviations.** `push_monomial_reads` remains the one hand-written read list —
the monomial ctx structs are not descriptors yet (see `MONO-CTX-DESC`).

### M3-REBASE — `1fa2a0d5`

**Plan.** Adapt the mirror to the rewritten compiler API after
`feat/crypto-compiler` moved underneath the branch.

**GREEN evidence.**

```
Summary [ 418.058s] 166 tests run: 166 passed (7 slow), 12 skipped
```

**Deviations.** None recorded.

### M4-BASE-OFF — `f31e750f`

**Plan.** Replace embedded device pointers in `MainMatrixDesc`, `EvalCoreCtx`,
`ZerocheckCtx` and `LogupCtx` with byte offsets into the `GraphExe` pool, with
the launcher taking the pool base as an argument.

**Acceptance.** Rust and CUDA agree on layout by construction (probes, not
assumption), and offsets survive planner pool reuse.

**GREEN evidence.**

```
Summary [ 403.160s] 168 tests run: 168 passed (8 slow), 12 skipped
```

**Deviations — two, both material and both corrected later.**

1. The commit message claimed *every* device-pointer field was converted. It is
   not: the three monomial ctx structs, `GkrInputCtx` and three bare `T *const *`
   tables keep raw pointers, because they are keygen-static tables rather than
   graph buffers. The claim was narrowed in `ad22ba77` and the `BaseOff` doc
   now carries the exhaustive two-column enumeration.
2. This commit also put the **eager** path on the same context structs and the
   same C++ decoders as the graph path. That silently retired eager as an
   independent oracle; see `M6-REVIEW-FIXES` finding 3 and guide trap T4.

### M5-REAL-INPUTS — `ffb3d999`

**Plan.** Remove the zero placeholders. Register every keygen- and
challenge-derived buffer as a graph input; fill `batch_fold_mle`'s four control
tables and the `interpolate_columns` column table for real.

**Acceptance.** Graph output bytes equal the eager kernel's on ragged shapes,
with a sabotage leg; a missing input is a hard error naming the buffer, not a
silent all-zero prove.

**RED evidence.** Reverting `fold_ptr_tables` to the pre-fix memset:

```
FAIL  batch_fold_mle_ir_ptr_tables_match_eager
assertion `left == right` failed: batch_fold_mle_ir mismatch on matrix 0
  left:  [0, 0, 0, 0, 0, 0, 0, ... ]      <- the graph: ALL ZEROS
  right: [245, 200, 132, 41, 171, ... ]   <- the eager kernel
```

Swapping the column table to matrices-first:

```
FAIL  interpolate_columns_ir_column_table_matches_eager
assertion `left == right` failed: interpolate_columns column table mismatch
```

**GREEN evidence.**

```
Summary [ 564.869s] 177 tests run: 177 passed (8 slow), 12 skipped
```

**Deviations.** One bonus defect fixed in the same commit: the selector fold was
folding traces already exhausted, where the output height saturates to 1 and the
kernel reads `input[1]` of a height-1 buffer. Invisible while the tables were
zeroed.

### M6-REVIEW-FIXES — `ad22ba77`

**Plan.** Fix the five defects raised by external review. They are written up as
traps T1–T5 in the guide because the lesson generalizes past this port.

**Acceptance.** Every fix carries a red-green pair actually run on GPU 7.

**RED evidence.**

*T1, the `alias_bufs` rename* — restoring the pre-fix spelling:

```
RED PROBE: whole-buffer match=false lower-half match=false upper-half match=true
panicked: RED PROBE: aliased spelling does not reproduce the eager bytes
```

*T3, uninitialized padding* — restoring `bytes_of`:

```
assertion `left == right` failed: `MainMatrixDesc` encoded to different bytes on round 0
  left:  [0, 0, 173, 222, 0, 0, 0, 0, 120, 86, 52, 18, 0, 0, 0, 0]
  right: [0, 0, 173, 222, 0, 0, 0, 0, 120, 86, 52, 18, 165, 165, 165, 165]
```

*T4, the shared decoder* — injecting a 4-byte skew into `base_off_ptr`:

```
PASS  round0_one_element_descriptor_sabotage_changes_output   <- shared ABI: BLIND
PASS  round0_pool_offsets_match_absolute_addresses            <- shared ABI: BLIND
FAIL  test_monomial_vs_dag_equivalence                        <- independent: CATCHES IT
```

**GREEN evidence.**

```
Summary [ 403.617s] 180 tests run: 180 passed (7 slow), 12 skipped
```

**Deviations — one scoped gap, tracked as `R0-RAW-TWIN`.** Three of the four
evaluator families got independent `*Raw` contexts and `_raw` entry points; the
round-0 pair did not, because round 0 dispatches through a macro that would need
a third template parameter threaded through it. The injected-skew run above is
the live measurement of what that gap costs. Partly mitigated:
`assert_ctx_abi_matches_cuda` pins `MainMatrixDesc`'s size and both field
offsets against the C++ truth and pins the `BASE_OFF_NULL` sentinel, so both
layout-drift classes have a direct guard even without an independent oracle.
What stays uncovered is a decode bug those asserts do not express.

### M7-RING — `83ae7e1b`

**Plan.** Stop supplying the steady round's answer. Compute the batched sumcheck
polynomial on device from graph-resident evaluator outputs, observe it from
device buffers, sample the round challenge into a device buffer, and hand that
same buffer to both folds.

**Acceptance.** Ring output byte-equal to the unchanged eager
`compute_batch_s_poly` + `observe_ext`/`sample_ext` chain, with a sabotage leg;
`r_1..r_n` never resolved on the host.

**GREEN evidence.**

```
Summary [ 355.779s] 189 tests run: 189 passed (7 slow), 12 skipped
```

with the ten new tests in it, including:

```
PASS  zerocheck_ir_tests::batch_s_ring_pre_post_matches_eager
PASS  zerocheck_ir_tests::batch_s_ring_oracle_detects_sabotage
PASS  zerocheck_ir_tests::observe_and_update_zerocheck_round_ir_matches_eager
PASS  zerocheck_ir_tests::observe_and_fold_zerocheck_round_ir_matches_eager
PASS  zerocheck_ir_tests::batch_fold_mle_ir_uses_sampled_device_challenge
PASS  zerocheck_ir_tests::zerocheck_ring_node_budget
```

**Count reconciliation:** 180 (`ad22ba77`) + 10 new − 1 renamed = **189**. The
renamed test (`logup_zerocheck_phase_graph_compiles` →
`…_without_round_messages`) is a strict superset of the one it replaced. No test
regressed.

**Deviations.** `ZerocheckPhasePlan::round_evals` was deleted and `r: Vec<EF>`
replaced by `r_0: EF`; no compatibility copy was kept. `mod.rs` received an
**extract-only** refactor of `compute_batch_s_poly` so the host oracle and the
production path stay one function — eager behaviour unchanged.

**What this commit did *not* close, so nobody reads more into the graph than is
there:** round 0 and stage E remain plan inputs. `xi`, `lambda_pows`, `mu_pows`,
`s_0_coeffs`, `logup_sum_claims` and `opening_claims` are still constants. The
phase is not device-input-only.

### A1-LIVENESS — `83ae7e1b`

**Plan.** Close the descriptor-liveness question raised in review: an offset into
a pool slot the planner has reassigned is as stale as a pointer, so the
*consuming node* must declare the descriptor's reachable buffers.

**Acceptance.** Every descriptor encoding path has exactly one build-time
collector; every current descriptor consumer has the matching access
declaration; the one write-only raw-pointer table is declared as outputs rather
than reads.

**RED evidence.** The pool-reuse test carries an explicit under-declaration
sabotage handle; `M2-ACCESS-SET` records it failing before the fix.

**GREEN evidence.** Covered by the `83ae7e1b` full-suite run above. The four
structural tests are:

```
zerocheck_ir_tests::evaluator_declares_every_referenced_buffer
zerocheck_ir_tests::descriptor_offsets_survive_pool_reuse
zerocheck_ir_tests::every_descriptor_array_is_a_registered_input_and_fully_filled
zerocheck_ir_tests::main_matrix_descs_shared_per_trace_across_families
```

**Deviations / scope.** This row is `done` as a **design invariant with a
source audit**, not as a universal machine-checked property. The four tests
cover the important pieces (DAG transitive closure, pool reuse, bare pointer
tables, ring read set) but there is no single assertion over *every*
descriptor-root consumer. That is `A1-UNIVERSAL-TEST`, and it is open. No
production defect was found at `83ae7e1b`. Monomial contexts do not contradict
this: they are not descriptors yet, so there is no encoded descriptor to
under-declare — they are an incomplete port (`MONO-CTX-DESC`), not an omitted
read.

---

## Gates at `83ae7e1b`

Environment: `ORTOOLS_PREFIX=$HOME/opt/ortools`,
`LD_LIBRARY_PATH=$ORTOOLS_PREFIX/lib:$LD_LIBRARY_PATH`,
`CUDA_VISIBLE_DEVICES=7`, tests serialized on one GPU.

| Gate | Command | Result |
|---|---|---|
| check | `cargo check -p openvm-cuda-backend --all-targets` | clean |
| fmt | `cargo +nightly fmt -- --check` | clean on every file this branch touched |
| clippy | `cargo clippy -p openvm-cuda-backend --all-targets --tests --no-deps -- -D warnings` | no findings in files this branch touched |
| tests | `cargo nextest run -p openvm-cuda-backend --test-threads=4` | **189/189 passed**, 12 skipped |

The test gate reproduced on a clean `83ae7e1b` worktree:

```
Summary [ 411.878s] 189 tests run: 189 passed (7 slow), 12 skipped
```

`cargo nextest list -p openvm-cuda-backend` enumerates the same 189 tests, 29
of them in `zerocheck_ir_tests`.

Two pre-existing conditions a reviewer will hit and should not chase:

- Plain `cargo clippy -p openvm-cuda-backend ... -- -D warnings` fails inside
  `crates/compiler` (12 upstream findings) before it reaches this crate, so
  `--no-deps` is required to gate this crate at all.
- `cargo +nightly fmt -p openvm-cuda-backend` reformats five upstream
  `fractional_*` files as a side effect. They are not part of this branch's
  diff.

## Open TODOs in the source

Every one carries `WHY:` and `RISK:` lines at its site in `zerocheck_ir.rs`.
Grep `TODO(cc-ir` for the current set. At `83ae7e1b` they are, by theme:

| Theme | Sites | Risk if left |
|---|---|---|
| challenge captured by value | `x_i`, `denom_sum_init`, `is_first`/`is_last`, `bus_term_sum` | blocks a device-resident Fiat–Shamir chain; bit-exact today because eager does the same |
| host seams | round-0 iDFT chain (stage C.5), stage-E D2H + split + reorder | the phase is not device-resident across these points; messages either side are plan constants |
| strategy mirror | `RoundEvalKind` is three-way, eager is seven-way; late traces merged into the early batch | launch geometry can disagree with the staged ctx arrays; a mixed-height plan currently panics at build time rather than proving something wrong |
| descriptor coverage | monomial ctx structs are zero placeholders; `push_monomial_reads` is hand-written | monomial arms are not runnable; the hand-written list can drift |
| lifetime | proving-key rule tables enter as `Static` addresses | a `GraphExe` cached beyond the pk's lifetime replays dangling pointers |
| planner-dependent test | one pool-reuse test's teeth depend on planner behaviour it cannot pin | the sabotage could stop biting after a planner change without the test failing |

## Author review

*(B5-PLAN-REVIEW — not yet recorded. This section is where the IR author's
review of the guide and this checklist goes, before further code commits stack
on top.)*
