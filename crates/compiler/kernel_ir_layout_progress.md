# kernel_ir layout progress

Tracks execution of the plan in `kernel_ir_gaps.md`. Phase A is the
minimal deadlock unblock (Gaps 1–5); Phase B is the general
linear-layout machinery (single-primitive `ConvertLayout` pipeline,
par-attr inference, shared-layout selection, decomposition
optimization). Each item lists **status**, the **files touched**, and
any **deviations from the plan**.

## Phase A — unblock the fused_drop deadlock

Status: **complete, verified**. All 365 unit tests + 446 integration
tests green. Frac_v2 bench at LOG_N=16 (FUSION_V2=1) runs end-to-end
with no shuffle deadlock.

Overall design note: the plan originally called for Gap 2 (zero-column
replication in the par-attr for sub-block bounds) alongside Gap 1
(warp-aligned launch). We landed Gap 1 but **not** Gap 2 for Phase A —
Gap 2 makes replicated register writes non-idempotent (each thread's
`b[v]` slot is distinct, but the *value* computed depends on
thread-local state), which breaks reduce accumulator chains (par[k/2]
reads par[k]'s register tile at v=0 on a replica thread that never
wrote v=0). Under Phase A's revised model the par-attr stays identity
on `log2(block) + log2(seq_size)` bits, the par body tail-masks on
`phys >= bound`, and shuffle correctness comes from Gap 1's warp-
alignment alone — `__shfl_sync` mask lanes are all resident, and
statement-level `ConvertLayout` ops sit outside par bodies so their
full-mask shuffles get 32-lane participation regardless of the par's
tail-mask. Gap 2 remains a Phase B design item for the cases where
producer replication genuinely simplifies inference.

### A1 · `LinearLayout::right_inverse(&self, out_bits: usize)`

**Status:** landed. `crates/compiler/src/kernel_ir.rs`.

Column-space Gaussian elimination over 𝔽₂ that returns `None` if the
column span doesn't cover all `out_bits`. Per output bit picks the
min-Hamming-weight preimage (pivot columns only, slack columns zero) —
this is the canonical-replica convention: replicated inputs resolve
to the pivot's physical index instead of scattering arbitrarily. The
returned layout's `bases.len() == out_bits` (input width shrinks to
the codomain); its output values live in the original input space.
Affine handled: `T⁺(y) = M⁺(y ^ self.offset)`.

CPU tests exhaustively verify:
- Identity round-trips.
- Replicated-high-bits (`bases = [1, 2, 0, 0, 0]`) → canonical pivots
  in low input positions.
- Replicated-low-bits (`bases = [0, 0, 0, 1, 2]`, the Gap 7
  counterexample) → canonical pivots in high input positions `{0, 8,
  16, 24}`.
- Non-surjective inputs (`bases = [1, 0]`, `out_bits = 2`) → `None`.
- Affine offsets.
- Degeneracy to `inverse()` on square bijections.
- `right_inverse ∘ self` is the projector on inputs (not the
  identity), matching the paper's `L⁺ ∘ L`.

### A2 · Hoist `maps_agree` into `kernel_ir.rs`

**Status:** landed. Removed from `crates/compiler/src/passes/codegen.rs`
(previously a private fn there), moved next to `LinearLayout`.
Codegen re-imports. Also used in `layout_infer::promote_tiles` for the
Gap 3 amendment (fast-path Direct test).

### A3 · Warp-aligned launch (Gap 1)

**Status:** landed. `crates/compiler/src/passes/lower_to_kir.rs`.

Every block-choice site (`flat` / `!flat` / symbolic-outer-bound) now
ends in `.max(32)`. Deviation from the initial cautious refinement:
**explicit `#[grid(threads = X)]` hints do NOT get to opt out of
warp-alignment** — the initial refinement respected `threads=X`
verbatim, but that left `frac_compute_round_dsl_n8` deadlocking at
block=4 (fused kernel with explicit `threads=4`). Rewarding user
intent for sub-warp launches keeps the mask-vs-resident-lane bug
alive; correctness wins. Poseidon2-16's `threads=WIDTH=16` was the
motivating case for the refinement — it now lifts to block=32
without regressions because `spec_attr` accepts sub-block bounds
(see below) and tail-masking handles the extras.

`spec_attr` (`lower_to_kir.rs:494-560`):

- Old: rejected `bound < block` outright.
- New: accepts sub-block bounds. `seq_size = max(1, bound/block)`,
  `thread_bound = min(bound, block)`. The spec's own `to_linear_layout`
  runs on `(thread_bound, seq_size)` and yields a layout on
  `log2(bound)` input bits. That gets padded with **identity** bases
  on positions `log2(thread_bound)..log2(block)` — not zero columns
  (Gap 2). Rationale: identity padding keeps `attr.layout.is_identity()`
  true whenever the spec was identity, which routes through codegen's
  fast identity-strided-loop path where the tail-mask
  (`v < bound` loop condition) falls out for free. Zero-column
  padding would force the non-identity path (with the
  `phys >= bound` guard change described in A5's fallout).

### A4 · Replicated par layouts (Gap 2) — **deferred to Phase B**

**Status:** not landed. Attempted then reverted (see design note above).
Default par-attr in `layout_infer.rs:66` stays
`LinearLayout::identity(ceil_log2(seq_size) + ceil_log2(block))`.
Replication as a first-class layout property is a Phase B design item
requiring coordinated changes to par-body tail-masking, register
write semantics, and the reduce accumulator lowering path.

### A5 · Migrate six bijectivity call sites (Gaps 3–4)

**Status:** landed across `kernel_ir.rs`, `layout_infer.rs`,
`codegen.rs`.

1. **`kernel_ir.rs:243`** (`classify_convert`): the lane-block
   bijectivity check now accepts either `lane_block.inverse().is_some()`
   OR `const_src_slot(c, tb)`. The latter is the fast-path
   broadcast case where the sender slot is thread-independent —
   e.g., a replicated writer's tile read via a non-invertible
   composite `C = l⁺ ∘ E` still shuffles as one-canonical-source-to-
   many-receivers.
2. **`kernel_ir.rs`** new `pub fn const_src_slot(c: &LinearLayout,
   tb: usize) -> bool`: hoisted from `codegen.rs:1135` (previously
   an inline `let`). One predicate, shared between classifier and
   emitter — classifier can no longer accept a case emitter can't
   handle, and vice versa.
3. **`layout_infer.rs:208`** (`promote_tiles` promotion gate):
   `l.inverse()` → `l.right_inverse(kb)`. Sub-block writer layouts
   (`l` has `block_bits` input, `kb` output, non-square) survive
   promotion instead of silently degrading to shared mirror.
4. **`layout_infer.rs:263-264`** (Plan selection): `maps_agree(&eff,
   &l)` fast-path before classifying `l_inv.compose(&eff)`. Under
   replication `C = l⁺ ∘ l` is the *projector* onto canonical
   replicas, not the identity; without this test the promoted tile
   emits a wasteful self-broadcast shuffle where a direct register
   read would suffice.
5. **`codegen.rs:1010`** (reg→reg source layout): `f.inverse()` →
   `f.right_inverse(kb)`. Sub-block source tiles convert correctly.
6. **`codegen.rs:1148-1165`** (`gen_shuffle` general path): keeps
   the strict `.inverse().expect(...)` — reachable only when
   `!src_slot_const AND slots > 1`, which requires the classifier
   to have admitted via the invertible-lane path. Renamed the local
   `const_src_slot` variable to `src_slot_const` (the hoisted
   function shadowed it).
7. **`codegen.rs:1044`** (reg→shared `map.inverse()`) — kept strict
   per plan; memory layouts stay injective (broadcasting consequence
   (d) in `kernel_ir_gaps.md`).

The `Promotion` struct gained an explicit `kb: usize` field: the
tile's logical bit count. Previously `let kb = pr.l.bases.len()` in
the second half of `promote_tiles` computed kb from the register
layout's *input* width, which used to equal the tile's *output*
width under identity par-attrs but diverges under Gap 1's padded
layouts (input = 5, output = `log2(tile_size)`). Codegen indexes
buffers on the output width.

### A6 · Delete the divergent guard (Gap 5)

**Status:** landed. Removed `if (threadIdx.x < n) { … }` wrapper
around the shuffle body at `codegen.rs:1117-1121` and its matching
close-brace at the end of `gen_shuffle`. `pad` is now `let pad`
instead of `let mut pad`. The stale `classify_convert guarantees kb
>= 5` comment at `:1115` was rewritten to explain the Gap 1+2
invariant that keeps the full-mask shuffle valid even when the
logical domain is smaller than the block.

### A7 · Fallout

**Status:** landed. Tests and goldens updated to reflect the new
classifier and warp-align behavior.

- `kernel_ir::tests::classify_convert_cases`: the singular-lane-block
  cases (`fold`, `mat4_read`) now expect `Shuffle` under the
  const-src-slot fast path. Added two new counterexamples that still
  Bounce: `fold_slot` (bit-0 lane maps into a slot output bit — no
  fast path, no general path) and `mat4_hard` (k=6 sub-warp with
  lane→slot cross-term and singular lane block).
- `passes::layout_infer::tests::non_linear_reader_gets_shuffle_view`
  (renamed from `non_linear_reader_gets_shared_mirror`): the many-to-
  one fold `tile[j % (t/2)]` now becomes a register `View` +
  `__shfl_sync` broadcast, not a shared mirror. Assertions verify no
  `__shared__` allocation and no `__syncthreads`.
- `passes::insert_sync::tests::shared_tile_gets_alloc_and_sync` and
  `independent_tiles_share_one_sync`: reworked to use a symbolic-
  modulus read (`(j + #m) % t`) so `linearize_accesses` can't
  linearize it — the read stays `SExpr` and routes through the
  shared-memory mirror, exercising the same sync path the old fold
  did. Kernel block also updated to 32 (warp-aligned).
- `passes::lower_to_kir::tests::par_bound_smaller_than_block_pads_with_identity`
  (renamed from `..._is_rejected`): now asserts the padded layout
  and block=32.
- `dump::tests::kir_dump_shows_tile_kernel`: expects `block[32]
  shared=0B` (no more mirror), `register layout=…` for the tile and
  its view, no `sync`, and a `convert_layout` line.
- `passes::fusion_v2::tests::driver_tests::driver_produces_hand_authored_reference_module`:
  matches on `dump_hir` output with a manual α-normalization
  (`v\d+` → `v`) rather than `module_hash` equality. Under warp-
  align the cost model prefers the nested `small_kernel` synthesis
  over the flat vertical fuse for the compute[8]-chain shape
  (block=32 with 24 replica lanes either way, so keeping the
  producer as a tile slightly under-costs the flat version); the
  new assertion codifies that outcome.

**Codegen guard change (extra fallout):** `if (v >= bound) continue;`
in the non-identity par path at `codegen.rs:658` changed to
`if (phys >= bound) continue;`. Under identity layouts `v == phys`
so the guard is unchanged; under non-identity layouts (Poseidon2's
padded spec) `v = layout.apply(phys)` doesn't fire on replicated
threads, whereas `phys >= bound` correctly guards the extras. This
preserves the tail-mask semantics the reduce accumulator chain
depends on.

### Verification

- **Unit tests**: 365/365 pass (`cargo nextest run -p crypto-compiler
  --lib`).
- **Integration tests**: 446/446 pass (`cargo nextest run -p
  crypto-compiler --tests`), including all Poseidon2, NTT, and
  fusion-v2 GPU end-to-end tests.
- **Frac_v2 bench at LOG_N=16** (env: `FRAC_V2_BENCH_FUSION_V2=1`,
  `GRAPH_EXE_DISPATCH_WATCHDOG_MS=30000`): PASS in 264s. All 3345
  stream instructions dispatch successfully; graph capture and
  execution complete. No shuffle deadlock.
- The `frac_compute_round_dsl_n8` kernel that previously deadlocked
  (launched at `dim3(2u), dim3(4u)`) now launches at `dim3(_, 32u)`
  under Gap 1's unconditional warp-alignment.

### Landing sequence taken

Roughly matched the plan's A0-A7 ordering:

1. A1 (right_inverse) + A2 (maps_agree hoist) landed together as
   pure additions, exhaustive CPU tests green.
2. A3 (warp-align lowering) landed; initial refinement respected
   explicit `threads=X`, later reverted to unconditional after the
   bench watchdog surfaced `frac_compute_round_dsl_n8` still
   deadlocking at block=4.
3. A5 (six call sites) landed as one changeset; Poseidon2's tests
   required `spec_attr` to accept sub-block bounds and pad with
   identity.
4. A6 (guard deletion) landed with the gen_shuffle rewrite.
5. A7 (goldens/tests) landed as the failing tests surfaced during the
   test-run passes.
6. A4 (Gap 2) attempted but reverted — see design note above.

## Phase B — general linear-layout machinery

Status: **not started**. Waiting on Phase A soak.

The plan sub-items (B.1 layout_infer rewrite, B.2 allocate_convert_scratch,
B.3 insert_sync extension, B.4 codegen `best_decomposition`, B.5
landing sequence, B.6 the three key algorithms, B.7 order) are all
still speculative — no code exists for any of them yet.

Known Phase B enablers already in Phase A:
- `right_inverse` (B.4's `factor_reg_to_reg` needs it; A1 provides).
- `maps_agree` in `kernel_ir.rs` (B.1's step 2a Direct fast path uses
  it; A2 provides).
- `const_src_slot` hoisted (B.4's `best_decomposition` legality check
  reuses it; A5 provides).
- Warp-aligned launch (B's whole model assumes it; A3 provides).

Deferred from Phase A to Phase B:
- **Gap 2 replicated par-attr layouts** — needed only when producer
  replication genuinely simplifies inference (e.g. sub-warp compute
  chains whose downstream ConvertLayouts want full-warp shuffles
  without redundant tail-mask arithmetic in the par body).
  Coordination with reduce-accumulator lowering required.
- **Store-guard from `T⁺∘T`** (Gap 7) — Phase A leaves
  `Shared`/`Global` stores duplicated `32/T`× under replicated par-
  attrs (idempotent same-value same-address, benign). Phase B B.2
  makes this an explicit codegen guard once Gap 2 lands.
- **Consumer-side broadcasts** (`A[i%c]` / `A[0]` / reduce results)
  — Phase A already handles these via the const-src-slot fast path
  in classify_convert, but B.1 makes them first-class in
  `linearize_accesses` so they don't need the promote_tiles detour.
- **`ConvertLayout` as sole layout-change primitive** — Phase A still
  routes through `Plan::{Direct, View, Mirror}` in promote_tiles.
  B.1 collapses all three into `ConvertLayout` insertions.
- **Multi-stage `gen_convert` decomposition** — Phase A's
  `gen_convert` still fails on `Bounce` for reg→reg and doesn't
  support Shared→Register / Shared→Shared. B.4 factors
  `C = R_dst ∘ S ∘ R_src` and picks strategy A (pure shuffle) / B
  (pure bounce) / C (hybrid) by cost estimate.
- **Shared layout selection via Triton `GenericSwizzling`** — Phase
  A uses row-major shared layouts throughout. B.6.2 implements the
  paper's Appendix §9.2 algorithm.

## Open questions for Phase B

- What's the right cost-model constant for `SHUFFLE_COST` vs
  `SHARED_COST` on the target hardware (RTX 5090 / GB202)? Phase A
  didn't need one; Phase B's `best_decomposition` argmin depends on
  it.
- Does the fusion-v2 driver's cost estimator (`fusion_v2/cost/`) need
  to be aware of `ConvertLayout` costs when scoring candidates? Right
  now `stamp_block_hint` bakes `block_size_policy(max_bound)` into
  the module before lowering — under B.1's ConvertLayout-first
  design, the *number* of ConvertLayouts becomes a first-class cost
  term, not just launch-geometry.
- The `Promotion { kb }` field added in A5 lives on the old
  `promote_tiles` path; B.1 dissolves that path entirely. Whatever
  replaces it needs the same `kb`-vs-`bases.len()` distinction —
  logical bit count is *not* the same as physical input width once
  Gap 1's warp-alignment kicks in.
