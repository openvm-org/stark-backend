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

Status: **foundations + three core algorithms landed, unwired.**
The B.7 landing sequence (B.4-partial → B.2+B.3 → B.1 → B.4-full) still
lies ahead; what's below is the algorithmic scaffolding those steps will
consume.

### B.foundations · cost model + F₂ + block decomposition

- **`ConversionCostModel`** — `crates/compiler/src/passes/layout_cost.rs`
  (new file). Wraps the constants from the Phase B cost-model decision
  (SHUFFLE_COST=3, SHARED_COST=30·k, SYNC_COST=100, SYMBOLIC_WEIGHT=2³⁰)
  behind `shuffle_round_cost`, `shared_round_cost(bank_conflict_factor)`,
  `sync_cost`, and `loop_weight` (saturating multiplication so nested
  symbolic loops don't overflow). 5 CPU tests.
- **F₂ subspace primitives** — `kernel_ir::f2` submodule. `reduce`
  (canonical RREF), `rank`, `contains`, `sum`, `intersection`
  (Zassenhaus), `standard_basis`, `complement` (inside ambient),
  `complement_within` (inside a containing subspace), `extend_with`,
  `take_independent`. All operate on `Vec<u64>` — each `u64` a bit vector
  over the ambient dim. Consumed by B.6.2 and B.6.3.
- **`LinearLayout::is_distributed`** — Def 4.10 check (≤1 non-zero bit
  per column, distinct non-zero columns). Filters candidate par-attrs in
  B.6.1 and guards Strategy A's paper-condition-suffices claim in B.6.3.
- **Block decomposition** — `PhysPartition { slot, lane, warp }` +
  `LinearLayout::{phys_partition, lane_block, warp_block, slot_block,
  is_warp_column_identity, output_bits}`. Deduplicates the ad-hoc
  slot/lane/warp split inside `classify_convert`; used pervasively by
  the three algorithms. 12 CPU tests across the new helpers.

### B.6.1 · par-attr inference

**Status:** landed as a pure function, unwired.
`crates/compiler/src/passes/par_attr_infer.rs` (new file).
`infer_par_attr(cost, reads, default, block, phys_bits) -> LinearLayout`
implements the algorithm from the plan verbatim: candidate set =
`{default} ∪ {g⁺ ∘ L_B for each read}` filtered by `is_distributed`,
scored by `cost_tier` (0/1/2 by warp/lane block identity, weighted by
loop iteration counts), tie-broken by Hamming distance toward `default`.
8 CPU tests including the "hot read wins" weighting scenario.

Wiring into `layout_infer.rs` is B.1; the current default-par-attr
computation at `layout_infer.rs:66` becomes the `default` argument.

### B.6.2 · shared layout selection

**Status:** landed as pure functions, unwired.
`crates/compiler/src/passes/shared_swizzle.rs` (new file). Ports
Triton's `GenericSwizzling.cpp` two-access algorithm to our
`LinearLayout`: maximal common vec, dangerous subspaces `U_A`/`U_B`,
common/exclusive split, safe directions (global complement + paired
XORs `E_A[i] ⊕ E_B[i]`), idx selection with fallback into `U_A`, bank
as the remaining complement. `SharedLayout { vec, bank, idx,
output_dim }` returned as F₂ subspace bases in RREF.

Multi-access `choose_shared_layout(cost, accesses, output_dim,
element_bytes, block)` enumerates C(N,2) pairs and picks the min
loop-weighted `conflict_factor` candidate; `row_major_default` is the
safety-net baseline. `conflict_factor` computes rank deficiency of lane
columns' bank-projection — bank-conflict-free ⇒ 0, k-way ⇒ k−1. 6 CPU
tests including the classic transpose case where the swizzle beats
row-major.

Wiring into `layout_infer.rs` (as `choose_target`'s
`choose_shared_layout` call) and `codegen.rs` (as
`best_decomposition`'s scratch swizzle picker) is B.1 / B.4.

### B.6.3 · convert decomposition

**Status:** landed as pure function, unwired.
`crates/compiler/src/passes/convert_decompose.rs` (new file).
`best_decomposition(cost, src, dst, block, logical_bits, loop_iters) ->
DecompositionResult` scores two strategies and returns the min:

- **A (pure shuffle)** — paper §5.4. Applicable iff
  `C.is_warp_column_identity(block)` (the `(C)_Wrp = I` condition).
  Round count = `2^|R|` where R extends `V ∪ I ∪ G` to `F₂^logical_bits`
  per page 8 of the paper (V, I, E, F, G computed from
  intersections/complements of `src.reg/thr` and `dst.reg/thr`).
- **B (pure bounce)** — always applicable. Delegates to
  `optimal_shared_swizzle` for the scratch buffer's `SharedLayout`; cost
  = store + sync + load, each shared round scaled by `conflict_factor`.

`Strategy::Copy` short-circuits when `maps_agree(src, dst)`. The hybrid
Strategy C (per the plan's landing pragma restricted to
`T ∈ {∅, all-lane-bits} × {Pre}`) is deferred to the full B.4 landing
— its two extremes are exactly A and B, so nothing is missed. 6 CPU
tests: identity ⇒ Copy, lane-only mix ⇒ Shuffle beats Bounce,
warp-crossing ⇒ Bounce, `loop_iters` scales cost linearly, matching
partitions give |R|=0.

### B.2 · `allocate_convert_scratch` pass + scratch: BufId in ConvertLayout

**Status:** landed. `crates/compiler/src/passes/allocate_convert_scratch.rs`
(new file), `kernel_ir.rs`, `layout_infer.rs`, `plan_shared_mem.rs`,
`insert_sync.rs`, `codegen.rs`, `dump.rs`, `module_compiler.rs`.

**Op-shape change.** `SSAOpCode::ConvertLayout` grows a third BufId:

```rust
ConvertLayout {
    dst: BufId,
    src: BufId,
    scratch: BufId,   // NEW
    map: LinearLayout,
}
```

`layout_infer` emits every ConvertLayout with a companion Shared
`Alloc` at shape `[0]` (`new_scratch_buffer` helper). The two allocs
land next to the tile's own alloc, before the writer par; the convert
lands after the writer, unchanged. Post-B.2 stmt_kinds sequence for a
tile+view is `["alloc", "alloc", "alloc", "par", "convert", "par"]`
(one extra `alloc` per ConvertLayout).

**Pass wire-up.** `module_compiler::lower` runs
`allocate_convert_scratch` after `layout_infer` and before `insert_sync`
so the sync walk sees final scratch shapes. The pass walks each
ConvertLayout, computes required bytes via `convert_scratch_bytes`
(pure), and resizes the scratch `Alloc`. Buffers whose requirement is 0
stay at shape `[0]` and are filtered out of `plan_shared_mem`'s
liveness (no first write → no interval).

**`convert_scratch_bytes` matrix** (per `(src.space, dst.space)`, pure):

| src → dst        | Bytes                                                  |
|------------------|--------------------------------------------------------|
| Reg → Reg        | 0 iff `C = src⁺∘dst` has `is_warp_column_identity(block)`; else full-tile bounce (`block × ceil(dst_len/block) × elem_bytes`) |
| Reg → Shared     | 0 (dst is already shared)                              |
| Shared → Reg     | 0 (src is already shared)                              |
| Shared → Shared  | full-tile bounce                                       |
| anything Global  | 0 (direct load/store)                                  |

Consumer notes for today: all current layout_infer emissions are
Reg→Reg with warp-column-identity composites (View plan) or Reg→Shared
(Mirror plan), so every scratch buffer stays at 0 bytes. The pass is
still meaningful — it locks in the invariant B.4-full will consume,
and it will surface any future warp-crossing Reg→Reg that the current
codegen would silently reject.

7 CPU tests on `convert_scratch_bytes` (identity → 0, warp-crossing →
full tile, Shared arms, Global arms, sub-block tile that fits in lanes
stays 0), plus one integration-style test that runs the whole
`layout_infer → allocate_convert_scratch` pipeline on a tile+view
kernel and asserts every scratch buffer stays 0-byte.

### B.3 · `insert_sync` + `plan_shared_mem` scratch tracking

**Status:** landed. `crates/compiler/src/passes/insert_sync.rs`,
`plan_shared_mem.rs`.

`insert_sync`'s dirty walk now treats a ConvertLayout's non-zero
scratch as both a read and a write of shared memory — a sync fires
before the convert if the current `reads_since_sync` state would
otherwise let a downstream reader see a stale scratch region reused by
the packer. The inner store→sync→load sequence stays codegen-internal
per the B.4-full plan; this outer walk just protects the scratch from
aliasing hazards. `collect_shared_writes` (loop pre-extend) picks up
scratch on the same conditions.

`plan_shared_mem::compute_liveness` `note`s scratch as read+written at
the ConvertLayout's position, giving it a single-point liveness
interval — enough for the packer to reserve non-aliasing space when
scratch is non-zero. Zero-byte scratch stays filtered out via the
`start.is_some()` filter (`note` never sets `start` because the buffer
kind check now sees a Shared buffer, but the interval collector
already filters no-first-write buffers).

Both extensions gate on `!p.buffer(*scratch).is_empty()`, so today's
0-byte scratch buffers are no-ops in the walk — golden parity holds
for every existing kernel.

New test: `insert_sync::tests::nonzero_scratch_survives_dirty_walk`
forces a scratch buffer's shape to non-zero after `layout_infer`,
runs `insert_sync`, and confirms the walk terminates cleanly, verify
passes, and codegen still emits a `__shfl_sync` (i.e. no spurious
extra syncs from the scratch handling).

### B.4-full · complete

**Status:** landed. Every ConvertLayout kind-pair works uniformly through
`gen_convert`, and every `best_decomposition` outcome is lowered by
codegen. `classify_convert` and the reg→reg Bounce compile-error path
are gone. The `layout_infer`-side `shuffle_emittable` gate is removed
(routing falls back to Bounce via scratch instead of Mirror).
`crates/compiler/src/passes/codegen.rs`, `layout_infer.rs`,
`allocate_convert_scratch.rs`, `kernel_ir.rs`.

**Landed:**

1. **`classify_convert` + `ConvertKind` deleted** from `kernel_ir.rs`;
   `classify_convert_cases` test removed. `shuffle_emittable` (formerly
   private in `layout_infer.rs`) moved to `kernel_ir.rs` as the single
   emittability predicate, alongside `const_src_slot`.

2. **Reg→Reg Bounce** (paper §5.4 Optimal Swizzling): `gen_convert`
   emits store → `__syncthreads` → load through the ConvertLayout's
   `scratch` buffer when `best_decomposition` picks `Strategy::Bounce`
   or when `Strategy::Shuffle` isn't `shuffle_emittable`. New helper
   `gen_reg_bounce`.

3. **Shared→Register** ConvertLayout arm: `gen_convert` case
   `(Register, Shared)`. Uses `g = f ∘ map ∘ ld` (src.layout ∘ map ∘
   dst.layout) as the read address in shared memory. Symmetric to the
   Register→Shared (Mirror) arm.

4. **Shared→Shared** ConvertLayout arm: `gen_convert` case
   `(Shared, Shared)`. Reads-then-writes per physical index; codegen-
   internal without a scratch or explicit sync (outer sync handled by
   `insert_sync` if the dst is downstream-aliased).

5. **Multi-round Shuffle fallback**: `classify_reader` no longer
   filters `Strategy::Shuffle` by `shuffle_emittable`. All Shuffle
   outcomes flow to a Register alias; `gen_convert`'s Shuffle arm
   checks emittability at emission time and falls back to
   `gen_reg_bounce` for non-emittable composites (the true multi-round
   emitter is deferred, but functionally the Bounce fallback covers
   the same cases). `allocate_convert_scratch` now sizes scratch based
   on `is_warp_column_identity(C) && shuffle_emittable(C)` — 0 iff
   both hold, full-tile otherwise — so codegen always has enough
   space for whichever emission it chose.

6. **`choose_shared_layout` wired into scratch selection** (B.1 (b)
   partial): `allocate_convert_scratch::pick_scratch_layout` calls
   B.6.2's `choose_shared_layout` on the two accesses touching each
   scratch (src.layout as writer access, dst.layout as reader access)
   and projects the resulting `SharedLayout` to a `LinearLayout` via
   `shared_swizzle::to_linear_layout`. `gen_reg_bounce` then uses the
   bank-conflict-minimizing swizzle for its store/load addresses.

**Golden test updates:**

`par_layouts_compose_in_promotion` used to assert one shared Mirror
alias for the `(th, s) -> th*16 + s` reader against the identity gather.
Under B.4-full, that path becomes Register (view via Bounce), with a
non-zero scratch buffer. Test rewritten to check:
- no `_sm` (Mirror) buffer exists;
- at least one `_cs` (scratch) buffer is non-zero;
- gather + promoted tile Register buffers still have the expected
  layouts;
- `__syncthreads()` still appears (from `gen_reg_bounce`'s internal
  sync, not from a Mirror consumer).

New test: `warp_crossing_reader_bounces_via_scratch` exercises the
Bounce path directly with `block=128` and an explicit non-inferred
par-attr.

**Where the plan's B.1 pipeline table now stands:**

- ✓ `layout_infer` is the sole layout-choice site — one call to
  `classify_reader` per read, with `best_decomposition` scoring.
- ✓ `allocate_convert_scratch` sizes scratch as a pure function of the
  layout pair and picks swizzles via `choose_shared_layout`.
- ✓ `insert_sync`'s dirty walk sees the scratch as a shared read+write.
- ✓ `gen_convert` factors `C = dst.layout⁺ ∘ src.layout` and emits
  reg-perm / shuffle / bounce as the strategy directs, across all four
  `(dst_kind, src_kind)` pairs uniformly.
- ✓ Shared memory swizzled optimally (for scratch buffers; the same
  wire-up for persistent Shared aliases is a follow-up since today's
  tests only produce trivial Mirror layouts).
- ✓ Every ConvertLayout lowered per paper spec.

**Deferred:**

- **True multi-round `gen_shuffle`**: the paper's `2^|R|` round
  emitter. Today's fallback (`gen_reg_bounce` for non-emittable
  Shuffles) is functionally equivalent; the shuffle version would be a
  cycle-count optimization for high-round-count cases.
- **`choose_shared_layout` for Mirror aliases**: today's Mirror
  buffers come from readers with non-linear access maps (SExpr /
  grid-spanning), where B.6.2 can't produce a useful swizzle. Wire-up
  is trivial when a test case exercises it.

**What each "optimality" claim is actually tested by:**

- *"ConvertLayout lowered per paper spec":*
  - **Strategy decision** (unit): `convert_decompose.rs` — 7 tests
    that `best_decomposition` returns the correct `Strategy` for
    Copy / Slot / Shuffle / Bounce inputs, weighted by loop iters.
  - **Scratch sizing** (unit): `allocate_convert_scratch.rs` — 8
    tests per `(src.space, dst.space)` pair, verifying that the
    conservative full-tile sizing kicks in when
    `is_warp_column_identity(C) && shuffle_emittable(C)` fails.
  - **Reg→Reg Copy / Slot / Shuffle emissions** (runtime, on-device):
    `tests/gpu_macro.rs::macro_register_shuffle_lane_rotation`,
    `macro_register_shuffle_slot_xor`, and every macro-suite test
    that transitively uses these paths — outputs are compared
    against a CPU reference.
  - **Reg→Reg Bounce emission** (runtime, on-device):
    `tests/gpu_macro.rs::macro_reg_reg_bounce_warp_crossing` — new
    test with `block=128`, explicit `#[par((th,s)->th+s*128)]`, and
    a warp-crossing index. Verifies `gen_reg_bounce`'s store-sync-load
    against a CPU reference.
  - **Shuffle-not-emittable → Bounce fallback in `gen_convert`**
    (structural): `passes::layout_infer::tests::par_layouts_compose_in_promotion`
    — verifies no Mirror alias is emitted, at least one non-zero
    `_cs` scratch exists, `__syncthreads()` appears in the output.
  - **NOT DIRECTLY TESTED**: the `(Register, Shared)` and
    `(Shared, Shared)` `gen_convert` arms — the code is in place but
    nothing in `layout_infer` currently emits ConvertLayouts with
    those `(dst_kind, src_kind)` shapes (all reads of Shared/Global
    sources go direct today). Those arms are dead code paths guarded
    by symmetric address arithmetic that mirrors the tested
    `(Register, Register)` and `(Shared, Register)` paths.

- *"Shared memory swizzled optimally":*
  - **Swizzle algorithm** (unit): `shared_swizzle.rs` — 10 tests
    that `optimal_shared_swizzle` and `choose_shared_layout` return
    bank-conflict-free partitions when they exist (`identity_access_row_major_is_conflict_free`,
    `optimal_swizzle_transpose_avoids_conflict`,
    `choose_shared_layout_prefers_conflict_free_pair`), and fall
    back to row-major otherwise.
  - **`SharedLayout → LinearLayout` projection** (unit): 4 tests
    that `to_linear_layout` returns identity for row-major inputs,
    preserves output_dim, and produces a bijection for transpose
    swizzles.
  - **Wire-up in `pick_scratch_layout`** (integration):
    `par_layouts_compose_in_promotion` — asserts the scratch layout
    is a bijection AND has at least one non-power-of-two basis (i.e.
    a real XOR-swizzle, not identity). Verified output: `[1, 2, 4,
    8, 16, 33, 66, 132, 264]` — bank bits 5..9 XOR-mixed with slot
    bits, the classic paper §5.4 paired-XOR pattern.
  - **NOT DIRECTLY TESTED**: that the bank-conflict-avoidance is
    *observed at runtime* (no NCU-based profiling harness). The
    swizzle is *emitted* into address arithmetic and validated by
    the runtime Bounce test above, but bank-conflict count itself
    isn't measured.
  - **NOT WIRED**: persistent Mirror alias layouts. Today's Mirror
    aliases come from readers with non-linear index maps (SExpr /
    grid-spanning) where B.6.2 has no useful information to score,
    so wiring would just produce row-major identity. The wire-up
    site is `emit_convert_layouts`' `Insert{space: Shared, ...}`
    branch; feeding it is a follow-up when a test case exercises it.

**Verification:**

- Unit tests: **422/422** pass (removed the 1 `classify_convert_cases`
  test that no longer applies).
- Integration tests: **504/504** pass (added
  `macro_reg_reg_bounce_warp_crossing` runtime GPU test).
- Clippy: clean (added `#[allow(clippy::too_many_arguments)]` on
  `gen_convert`).
- Rustfmt clean.

### B.4-full partial · Reg→Reg Bounce codegen

**Status:** landed. `crates/compiler/src/passes/codegen.rs`,
`layout_infer.rs`.

Adds the first B.4-full codegen arm: reg→reg conversions whose
composite fails `is_warp_column_identity` now emit a
store→`__syncthreads`→load through the ConvertLayout's `scratch`
buffer (paper §5.4 Optimal Swizzling), instead of falling to the
Mirror (Shared-alias) path.

**Landed:**

1. **`gen_convert` signature widened** to accept `scratch: BufId` from
   the emission site (previously `scratch: _`).

2. **`gen_reg_bounce` helper** emitted between `gen_shuffle` and
   `gen_convert`. Sequence:
   - For each src slot: `b<scratch>[scratch_layout(f(phys))] = src[slot];`
   - `__syncthreads();`
   - For each dst slot: `dst[slot] = b<scratch>[scratch_layout(ld(phys))];`
   
   The scratch buffer's layout defaults to identity when
   `scratch.layout` is `None` (today's placeholder). Wire-up with
   `choose_shared_layout` for bank-conflict-minimizing swizzles is
   a B.4-full follow-up.

3. **`classify_reader` admits Bounce as Register** — was routing to
   Mirror, now returns `Insert{Register, layout=eff, shape=[1<<kb]}`
   for `Strategy::Bounce`. Consistent with the plan's unified
   ConvertLayout-only design; the Bounce → shared-mirror split (an
   inference-time policy) is gone.

4. **`allocate_convert_scratch` alignment**: its existing
   `is_warp_column_identity` check already sizes scratch to a full
   tile whenever the composite fails, matching exactly the case
   `best_decomposition` picks Bounce. Scratch is correctly non-zero
   under Bounce; zero for Shuffle/Slot/Copy.

**Test coverage:**

`warp_crossing_reader_bounces_via_scratch` — new integration-style
test with `block=128` (warp bits > 0) and a linearized reader index
`(j & 31) * 4 + (j / 32)` that permutes warp bits into lane positions.
An explicit `#[par((th, s) -> th + s * 128)]` par-attr bypasses B.6.1's
inference (which would otherwise rotate the par-attr to make the
composite identity). Verifies:
- `block == 128` (non-standard partition).
- One non-zero scratch buffer sized to the tile.
- Emitted CUDA contains `__syncthreads()` and a `_sh_pool`
  declaration.
- No `__shfl_sync(0xFFFFFFFFu` in the kernel body — Bounce path,
  not Shuffle.

**What's still on B.4-full:**

- **Multi-round shuffle emitter**: removes `shuffle_emittable` gate
  in `classify_reader`, expanding cases where warp-column-identity
  composites go through Shuffle (currently fall to Mirror when
  gen_shuffle can't emit).
- **Shared→Register** ConvertLayout arm (Shared-source load into
  Register via configured swizzle).
- **Shared→Shared** ConvertLayout arm (paper's general case; needs
  register intermediate).
- **Wire `choose_shared_layout`** into scratch buffer selection for
  Bounce (bank-conflict-minimizing swizzle).
- **Delete `classify_convert`** — the fast-path classifier is
  superseded by `best_decomposition` at every call site.

**Verification:**

- Unit tests: **423/423** pass (up from 422 with the new
  `warp_crossing_reader_bounces_via_scratch` test).
- Integration tests: **504/504** pass (up from 503).
- Clippy: clean.
- Rustfmt clean.

### B.1 (b) partial · `SharedLayout → LinearLayout` projection

**Status:** landed as pure function, unwired.
`crates/compiler/src/passes/shared_swizzle.rs`.

Adds `to_linear_layout(sh: &SharedLayout) -> LinearLayout` — projects a
`SharedLayout` partition to a `LinearLayout` mapping buffer-logical
index → phys address. The bases are concatenated `[vec | bank | idx]`
in that order (`output_dim` bases total), so logical bits 0..|vec| map
to the vec subspace, then |vec|..|vec|+|bank| to the bank subspace,
then the rest to idx.

4 CPU tests:
- Row-major default → identity permutation.
- Preserves `output_dim` and offset=0.
- Swizzle result is a bijection (`inverse()` succeeds).
- Partition-ordering: distinctive vec/bank/idx bases appear in the
  correct concat order.

Wiring into `layout_infer`'s Shared fallback in `decide_target` and
into every `Insert{space:Shared,...}` site is the next step; today's
kernels don't benefit (their Mirror aliases use trivial access patterns
where row-major identity is already conflict-free), so the wire-up
would change golden layouts without moving the runtime needle. Held
back until either (a) a test case exercises a non-trivial swizzle, or
(b) B.4-full's scratch-swizzle picker starts consuming this projection.

### B.1 (0b) · Universal reader walker

**Status:** landed. `crates/compiler/src/passes/layout_infer.rs`.

**No emitted-IR changes** — same golden IR on every test.

**Landed:**

1. **`emit_convert_layouts` restructured** as the plan's universal
   walker: outer loop iterates pars in program order (via
   `walk_par_order`); inner loop iterates the par's reads. Was: outer
   loop over promoted tiles, inner over their readers. Structurally
   matches the plan's B.1 step (2) pseudocode.

2. **Kernel-scoped `views_of(source_buf)`** — a `BTreeMap<BufId,
   Vec<Alias>>` accumulated across all pars. Multiple readers of the
   same source with the same `(space, layout, shape)` demand share one
   alias, even if they're in different pars. Previously the dedup
   cache was local to each tile's promotion block.

3. **Per-source write-map pre-cache** (`write_map_of`). Computed once
   from `tiles` before the mutation loop so we don't need to
   re-borrow `p.kernels[ki].op(wnode)` inside the loop that mutates
   `p.buffers`.

4. **Removed `Promotion` struct** — the intermediate collection of
   per-tile classification results is gone; readers flow directly
   from `classify_reader` into `rewrites` and `inserts`.

5. **Guard: only Register-source reads are analyzed** —
   `if source_decl.space != AddressSpace::Register { continue; }`.
   Reads of Shared/Global tiles (and Input/Output) go directly via
   the buffer's own layout at codegen time; the plan's
   Shared→Register load path is a B.4-full item (needs codegen
   support for the Shared→Register `ConvertLayout` arm).

**Verification:**

- Unit tests: **418/418** pass.
- Integration tests: **499/499** pass.
- Clippy: clean (added `type Alias = ...` alias for the complex-type
  warning).
- Rustfmt clean.

### B.1 (0) · Interleaved par-attr + writer-layout assignment

**Status:** landed (partial). `crates/compiler/src/passes/layout_infer.rs`.

**No emitted-IR changes** — the same golden IR as pre-rewrite on every
existing test; behavioral parity preserved.

**Landed:**

1. **`layout_infer` restructured** into three explicit phases matching
   kernel_ir_gaps.md's B.1 pipeline:
   ```
   linearize_accesses          # unchanged — Affine → Linear
   assign_par_attrs_and_writes # NEW — interleaved par-attr + writer-layout
   emit_convert_layouts        # was promote_tiles — now uses pre-set layouts
   ```

2. **`assign_par_attrs_and_writes`** — a single program-order walk
   over pars via `walk_par_order` (produces DFS order across
   loop-nested pars). Per par P:
   - **(a) `infer_and_set_par_attr(P)`**: if `P.attr` is `None`, calls
     `infer_par_attr` (B.6.1) with `ReadDemand`s whose
     `producer_layout` is read from the **buffer table** — no longer
     identity-defaulted. Producer layouts are set by earlier iterations
     of the same walk, so downstream consumers see real (non-identity)
     producer layouts. The plan's step 0/1 chicken-and-egg is
     resolved.
   - **(b) `assign_layouts_for_par_writes(P)`**: for each buffer P
     writes that's still `BufferKind::Shared` (i.e. the tile hasn't
     been touched yet), call `decide_target` to pick
     `(space, kind, layout)`.

3. **`decide_target`** — the single "register-eligible?" check that
   replaces the seven-precondition bailout in the old `promote_tiles`.
   Register iff every gate passes:
   - writer non-grid-spanning
   - buffer size is a power of two
   - single writer (from `analyze_buffers`)
   - no back-edge reader (from `analyze_buffers`)
   - no writer self-read with mismatched map (from `analyze_buffers`)
   - linear write map with `f.bases.len() == kb`
   - composite `l = f ∘ par_attr.layout` is surjective onto kb bits
     (`right_inverse` exists)

   Falls to `(Shared, identity(kb))` otherwise — a placeholder that
   B.1 (b) will replace with `choose_shared_layout`'s output.

4. **`analyze_buffers`** — kernel-wide pre-analysis of writers +
   readers + statement sequence numbers. Extracts the three inputs to
   `decide_target`'s register check that need whole-kernel visibility.

5. **`emit_convert_layouts`** (was `promote_tiles`) — per-tile reader
   analysis using the pre-set buffer layouts. Preconditions are gone
   (they've moved upstream); the per-tile loop just:
   - reads `buf.layout` (set by step 2b);
   - iterates readers and calls `classify_reader` per-reader;
   - materializes the alias allocs + ConvertLayouts.

6. **`walk_stmts` kind filter widened** from `BufferKind::Shared` only
   to `Shared | Register` — some tiles are now `Register` by the time
   `walk_stmts` runs, and we still need their alloc/writer/reader
   positions.

7. **Safety-net loop scope narrowed**: was `buf.space != Register`
   (which panicked on symbolic-shape Input/Output via `.len()`); now
   `buf.space == Shared` only. Input/Output keep whatever layout
   lowering supplied.

**What's still on today's B.1 path (not yet landed):**

- **Universal reader walker**: `emit_convert_layouts` only emits
  ConvertLayouts for Register-promoted tiles' readers. Buffers that
  stayed Shared (or Global Input/Output) don't get their readers
  analyzed for layout mismatches. For today's kernels this doesn't
  produce different IR — no such buffer has readers with mismatched
  demands. The universal walker (all reads through the same
  `classify_reader` + emission machinery) is the next step.
- **Cross-buffer `views_of`**: still per-tile scoped in
  `emit_convert_layouts`. Same as before this landing.
- **B.6.2 (`choose_shared_layout`)**: `decide_target`'s Shared
  fallback still uses `identity(kb)`. Wire-up requires the
  `SharedLayout → LinearLayout` projection.
- **`gen_convert`'s `best_decomposition` call**: still a second cost
  policy site at emission time. B.4-full's rewrite collapses these.

**Verification:**

- Unit tests: **418/418** pass.
- Integration tests: **499/499** pass.
- Clippy: clean.
- Rustfmt clean.

### B.1 · `layout_infer` structural + `ReaderTarget` unification

**Status:** landed. `crates/compiler/src/passes/layout_infer.rs`,
`allocate_convert_scratch.rs`, `codegen.rs`.

**No emitted-IR changes** — the kernel program `layout_infer`
produces is byte-identical to pre-B.1 output on every existing test.
The changes here reshape internals and wire in B.6.1 / B.6.3 for
future use.

**Code that landed:**

1. **`infer_par_attrs` sub-pass**, run between `linearize_accesses`
   and `promote_tiles`. For every par lacking an explicit `#[par]`,
   calls B.6.1's `infer_par_attr` with the strided-identity default
   and the par's linear reads of Shared buffers as `ReadDemand`s.

   *Behavioral effect today: none observed.* The wire-up feeds
   `identity(logical_bits)` as every `producer_layout` (the plan's
   step 0/1 chicken-and-egg — real producer layouts aren't
   materialized when this runs). Under identity producers every
   candidate ties with the default; tie-break by Hamming distance
   picks the default. Every emitted par-attr is byte-identical to
   the pre-B.1 identity fallback. Effective wire-up requires
   interleaving par-attr assignment with buffer-layout assignment;
   see the next-list below.

2. **`ReaderPlan` enum removed.** The `Direct(Option<LinearLayout>)`
   / `View { eff, g_padded }` / `Mirror` three-way that was the ad-hoc
   inference-time classification (kernel_ir_gaps.md Gap 8) is gone.
   Replaced with a `ReaderTarget` enum expressing the plan's binary
   2a/2c split literally:

   ```rust
   enum ReaderTarget {
       Existing(BufId),                    // 2a: match a live alias
       Insert { space, layout, shape },    // 2c: fresh insertion
   }
   ```

   The Register-view-vs-Shared-mirror decision is now just the value
   of `space` — a cost-model output, not an IR-level classification.
   Emission dedups by `(space, layout, shape)`, so Mirror
   (`Shared, None, tile.shape`) still collapses to one alloc per
   tile and Register views (`Register, Some(eff), [1<<kb]`) still
   dedup by `eff`.

3. **`classify_reader` replaces `plan_reader`.** Returns
   `(ReaderTarget, Option<IndexMap>)` directly. The plan-side
   decisions map to:
   - `maps_agree(&eff, &l)` (2a self-match) → `Existing(tile)` with
     the index rewritten to `padded_g`. No emission.
   - `best_decomposition` picks `Copy`/`Slot`/emittable-`Shuffle` →
     `Insert { space: Register, layout: Some(eff), shape: [1<<kb] }`.
   - `Bounce` or non-emittable `Shuffle` or the pre-scoring bailouts
     (non-linear index / grid-spanning / non-pow2 bound) → `Insert {
     space: Shared, layout: None, shape: tile.shape }`.

   The `shuffle_emittable(c, block)` gate stays — `gen_shuffle`
   still needs `const_src_slot` OR an invertible lane block, and the
   broader `is_warp_column_identity` cases (which
   `best_decomposition` admits) need B.4-full's multi-round emitter.

4. **Unified emission loop.** One code path handles all readers:
   `Existing` → just rewire `reads[rai].buf`; `Insert` → dedup, and on
   first sight of a `(space, layout, shape)` tuple allocate a buffer
   + companion scratch (Phase B.2) and emit one `ConvertLayout` with
   `map = identity(kb)`. The old three-branch match on `Direct`/`View`/
   `Mirror` is gone; nothing else in codegen or plan_shared_mem sees
   the difference.

5. **`convert_scratch_bytes` codomain fix** (B.2 bug exposed by B.1's
   layout representation). Layouts now have `phys_bits` bases with
   trailing zero columns (Gap 2 model) — `right_inverse(max(bases.len))`
   fails on the zero-column bits, driving the composite through the
   pessimistic full-tile-bounce branch. Fix: pass `logical_bits =
   log2(dst_len)` (the buffer's actual codomain width).

6. **Codegen skips 0-byte shared `Alloc`s** (`codegen.rs:465`). Under
   B.2 every `ConvertLayout` carries a companion scratch alloc; when
   its size stays 0 the emitted `_sh_pool` reference was undefined
   (`_sh_pool` isn't declared when peak shared is 0). Fix: `continue`
   on `decl.is_empty()` at the shared-Alloc branch.

**One golden updated:**

`own_index_tile_promotes_to_registers` asserted
`regs[0].layout.is_identity()` on a 3-bit tile. Under B.1 the
producer's inferred par-attr is now `Some(identity(5))` at the point
`promote_tiles` reads it (previously `None` and lazily-set later), so
`l = f.compose(&par_attr.layout)` produces `[1, 2, 4, 0, 0]` instead
of `[1, 2, 4]`. Semantically identical (Gap 2 zero-column form) but
no longer matches the strict `is_identity()` check. Test updated to
compare against the exact layout.

**IR gaps in the plan pseudocode (not resolved this session):**

These are places where the plan's B.1 pseudocode assumes machinery
the IR doesn't have. Each one is a real gap that a future landing
needs to address:

- **`views_of(B)` isn't a first-class IR concept.** The plan's
  "aliases of buffer B" is tracked as a local `inserted:
  Vec<(space, layout, shape, BufId)>` in `promote_tiles`, scoped to
  one tile's promotion. Cross-tile view sharing (a view of tile1
  also serving as a view of tile2) isn't representable in the IR.
- **"Earliest program point dominating this read"** — no dominator
  machinery. Current insert point is right after the writer par (a
  valid dominator, but not always earliest).
- **`phys_bits` varies per par**. Different pars have different
  seq_size, so cross-par layout comparisons need width
  reconciliation. `pad_layout_identity` papers over this.
- **`IndexMap` doesn't have composition.** The plan's 2b path
  `adjusted = v.layout.compose(&required)` implies rewriting a
  reader's index via a layout compose, but the IR's `IndexMap` has
  fixed variants (Linear/Affine/SExpr/Blackbox); arbitrary
  composition isn't representable. Today we side-step this by only
  matching 2a (register aliases with byte-identical layouts) and
  keeping 2b as a wildcard "reuse the mirror by matching Shared /
  None / tile.shape".
- **`all_uses(B)` for `choose_target`** requires kernel-wide use
  analysis. Current per-tile analysis is buffer-scoped only.
- **`pick_source_view` policy** — plan silent. Today's src is always
  the tile.

**Next, in order:**

1. **Interleave par-attr assignment with buffer-layout assignment**
   so `infer_par_attrs` sees real producer layouts. Program-order
   pass: for each par P, infer P.attr → propagate P's writes' buffer
   layouts. This unblocks B.6.1.
2. **Wire B.6.2's `choose_shared_layout`** into the mirror insertion
   and the final-loop shared-buffer layout assignment. Needs a
   `SharedLayout` → `LinearLayout` projection (subspace bases →
   XOR-affine).
3. **Cross-tile `views_of`** — promote the per-tile `inserted` list
   to a kernel-scoped map so a view emitted for tile1's promotion
   can be reused when scoring tile2's readers.
4. **B.4-full**: reg→reg Bounce codegen (`scratch` stores + sync +
   loads with receiver swizzle) + multi-round `gen_shuffle` (removes
   `shuffle_emittable` gate) + Shared→Register / Shared→Shared arms.
   Deletes `classify_convert` and the reg→reg-Bounce compile-error.

Items previously listed as "non-power-of-two" and "multi-writer" are
moot in this codebase: all tiles are powers of two, and every tile
has exactly one writer. The `n.is_power_of_two()` and `bound ==
Some(n)` gates in `promote_tiles` never fire negatively.

**Verification:**

- Unit tests: **418/418** pass.
- Integration tests: **499/499** pass.
- Clippy: my code clean (kept `#[allow(clippy::too_many_arguments)]`
  on `classify_reader`); pre-existing warnings unchanged.
- Rustfmt clean.

### B.4-partial · `gen_convert` routes through `best_decomposition`

**Status:** landed. `crates/compiler/src/passes/codegen.rs`.

`gen_convert`'s Register→Register arm no longer calls
`classify_convert`; it computes `best_decomposition(cost, f,
map∘ld, block, kb)` and dispatches on `Strategy::{Copy, Slot, Shuffle,
Bounce}`. Emission paths are unchanged — the goal is golden parity
against the classify_convert-driven flow. Two implementation notes:

- **`Strategy::Slot`** added to the enum with detection matching
  `classify_convert`'s `slot_only` block (thread-input columns pure
  identity, slot-input columns don't contribute thread outputs, no
  thread offset).
- **`is_warp_column_identity` tightened** — now also rejects
  non-warp inputs whose image touches warp output bits, matching
  `classify_convert::warp_fixed`. Under Def 4.10 distributedness the
  two directions imply each other; the tighter check is defense in
  depth against non-distributed inputs slipping into the pure-shuffle
  path (where `gen_shuffle`'s `expect` on the lane-block inverse
  would panic).

`Strategy::Bounce` still errors at codegen; that's B.4-full. The
composite `C = f_inv.compose(map.compose(ld))` is recomputed inside
gen_convert for the Slot / Shuffle emitters, keeping the current
byte-for-byte output.

### Verification

- Unit tests: **417/417** pass (`cargo nextest run -p crypto-compiler
  --lib`), up from B.4-partial's 409 by 8 new tests (7 pure
  `convert_scratch_bytes` cases + 1 `nonzero_scratch_survives_dirty_walk`).
- Integration tests: **498/498** pass (`cargo nextest run -p
  crypto-compiler --tests`) — golden parity confirmed for every
  ConvertLayout emission the current pipeline produces (all keep
  0-byte scratch under today's shuffle-view and mirror plans).
- Clippy: my code clean (`-D warnings`); two pre-existing warnings in
  `kernel_ir.rs::right_inverse` and `planner/list_v1.rs` are unchanged.
- Rustfmt clean (`cargo +nightly fmt -- --check`).

### What lands next

The plan's target pipeline (verbatim from kernel_ir_gaps.md B.1):

```
KIR (par-attrs partial; buffer layouts empty)
   │
   ▼
[layout_infer]           assigns buffer layouts from writers; inserts
                         ConvertLayout wherever producer.layout ≠
                         consumer.required; chooses reg vs shared
                         intermediate. (Only cost policy site.)
   │
   ▼
[allocate_convert_scratch]  per-op scratch requirement is a pure
                            function of (src, dst) layouts+spaces;
                            materializes a scratch BufId, joins the
                            shared-mem packer's liveness pool.
   │
   ▼
[insert_sync]            existing dirty-shared walk consumes the
                         scratch reads/writes uniformly; no code
                         change beyond registering them.
   │
   ▼
[codegen]                per ConvertLayout: factor
                         C = dst.layout⁺ ∘ src.layout into
                         (reg-perm ∘ shuffle ∘ reg-perm) when the warp
                         block of C is identity, else (store → sync
                         → load) through op.scratch.
```

To land this shape, the remaining steps are:

1. **B.4-partial** ✓ landed.
2. **B.2 + B.3** ✓ landed.
3. **B.1 structural + `ReaderTarget`** ✓ landed — `ReaderPlan` removed;
   B.6.1 and B.6.3 wired in; emission collapsed to one uniform 2a/2c
   loop. **No emitted-IR changes on any existing test.**
4. **B.1 (0) — interleaved par-attr + writer-layout** ✓ landed.
   `layout_infer` now runs a program-order walk that infers each par's
   attr with real producer layouts (from the buffer table) and then
   sets writer-derived buffer layouts. The seven-precondition bailout
   collapses into a single `decide_target` "register-eligible?" check.
   `emit_convert_layouts` uses the pre-set layouts. **No emitted-IR
   changes on any existing test.**
5. **B.1 (0b) — universal reader walker** ✓ landed. Program-order
   walk over pars/reads with kernel-scoped `views_of(B)`. Still gated
   on `source.space == Register` — reads of Shared/Global buffers go
   direct today; the Shared→Register load path is a B.4-full item.
6. **B.1 (b) partial — `SharedLayout → LinearLayout` projection** ✓
   landed as pure function `shared_swizzle::to_linear_layout`. The
   wire-up into `decide_target`'s Shared fallback + `Insert{space:
   Shared,...}` sites is held until a test case exercises a
   non-trivial swizzle (today's Mirror aliases use trivial identity
   accesses).
7. **B.4-full** ✓ landed (complete). `gen_convert` covers every
   `(dst_kind, src_kind)` pair via `best_decomposition`; Reg→Reg Bounce
   goes through `scratch`; `shuffle_emittable` gate removed from
   `classify_reader` (non-emittable Shuffles fall back to Bounce);
   `choose_shared_layout` wired into scratch swizzle selection;
   `classify_convert` deleted. Multi-round `gen_shuffle` deferred as
   a cycle-count optimization; today's fallback is functionally
   equivalent via Bounce.
8. **B.5 — true multi-round `gen_shuffle` emitter** ✓ landed
   (2026-08-15, section below). Every `is_warp_column_identity`
   composite is now shuffle-emittable; `shuffle_emittable` deleted.
9. **Deferred**: wiring `choose_shared_layout` for persistent
   Shared/Mirror aliases (today only trivial identity accesses reach
   that path).

After 4–6, the pipeline table matches:

- `layout_infer` is the only cost policy site — chooses reg vs shared
  intermediate uniformly for every layout mismatch.
- Mirror decisions are represented in the IR as `AddressSpace::Shared`
  aliases with `choose_shared_layout`-picked layouts.
- Shared memory is swizzled optimally (bank-conflict-minimizing).
- Every ConvertLayout lowers per paper §5.4: `C = dst.layout⁺ ∘
  src.layout` factored into rename ∘ shuffle ∘ scratch-bounce ∘
  rename, with strategy picked by `best_decomposition`.
- The `promote_tiles`-vs-`layout_infer` distinction is gone;
  `promote_tiles` is deleted.

The tiles-in-this-codebase constraints (all powers of two, exactly
one writer) don't lift the requirement for step 4 — the point of step
4 is IR uniformity, not just handling more cases.

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

### B.5 · true multi-round `gen_shuffle` emitter (2026-08-15)

**Status:** landed. `gen_shuffle` now lowers **every**
`is_warp_column_identity` composite — the paper §5.4 page 8 multi-round
exchange included — so `Strategy::Shuffle` never falls back to a shared
bounce. `shuffle_emittable` is deleted.
`crates/compiler/src/kernel_ir.rs`, `passes/codegen.rs`,
`passes/convert_decompose.rs`, `passes/allocate_convert_scratch.rs`.

**The multi-round pull scheme.** For the composite
`C : phys_dst → phys_src` with identity warp column, at dst slot `s'`
thread `tid` needs src physical `q = C(s' << tb) ^ C_lin(tid)`
(`C_lin` = C without offset; XOR-affinity splits the constant out).
Warp-column identity gives `q`'s warp bits = `tid`'s warp bits (the
value lives in this warp) and

```
q >> tb  ∈  (C(s' << tb) >> tb) ^ span(dirs),
dirs = f2::reduce({ C.bases[i] >> tb : i < tb })
```

(warp columns contribute 0 to `dirs` under warp-column identity). Per
`(s', σ)` with `σ` ranging over that affine subspace, emit one
**unconditional** `__shfl_sync(0xffffffffu, src_reg[σ], q & 31u)` into a
`const auto` temp, then guard only the register write with
`if ((q >> tb) == σ)`. Exactly one `σ` matches per `(tid, s')` — the
receiver pulls; no lane-block inversion needed (forward map only), so
singular lane blocks (replicated lane inputs feeding slot outputs) are
covered. `slots · 2^|dirs|` shuffles. Shuffles never sit inside a
divergent branch.

**Dispatch in `gen_shuffle`** (cheapest first):
1. `slots == 1 || const_src_slot(C)` — fast path, `slots` shuffles at compile-time-constant sender
   slots (unchanged).
2. invertible lane block — sender-side ternary path, `slots` shuffles (unchanged).
3. everything else — the new multi-round pull path.

**Landed:**

1. `kernel_ir.rs`: `shuffle_emittable` **deleted**; new
   `lane_slot_mix_dirs(c, tb) -> Vec<u64>` (the `dirs` basis),
   `lane_block_invertible(c, tb)`, and `shuffle_rounds(c, block) -> u64`
   returning the **actual** `__shfl_sync` emission count per the
   dispatch above (`slots` on paths 1–2, `slots << |dirs|` on path 3).
2. `convert_decompose.rs`: the V∪I∪G basis-extension `shuffle_rounds`
   proxy (and `extension_rank`) replaced by
   `kernel_ir::shuffle_rounds(&C, block)` — `Strategy::Shuffle{rounds}`
   now carries the exact instruction count that codegen will emit, so
   the Shuffle-vs-Bounce price comparison is honest.
3. `codegen.rs`: `gen_convert`'s Shuffle arm unconditionally calls
   `gen_shuffle` (the `shuffle_emittable`-gated `gen_reg_bounce`
   fallback is gone); `gen_shuffle` grows the multi-round third path and
   its `.expect()` on the lane-block inverse is gone.
4. `allocate_convert_scratch.rs`: Reg→Reg sizing now runs the **same
   `best_decomposition`** codegen re-runs (full tile iff
   `Strategy::Bounce` wins, else 0) instead of the structural
   `warp_column_identity && shuffle_emittable` predicate — this closes
   the latent inconsistency where a cost-preferred Bounce on an
   emittable composite would have hit a 0-byte-scratch compile error.
   The walk now also folds the op's `map` into the dst layout
   (`dst_eff = map ∘ ld`), mirroring `gen_convert`. Shared→Shared
   scratch drops to 0 — codegen's arm is a single read-then-write loop
   and never touched the tile that was being allocated.

**Verification (this step):**

- Unit tests: **424/424** pass (`--lib --test-threads=8`); new/updated:
  `shuffle_rounds_is_slots_for_const_sender_slot`,
  `multi_round_composite_picks_shuffle_with_exact_count` (replaces
  `extension_rank_zero_for_matching_thread_partitions`),
  `shared_to_shared_needs_no_scratch` (was `…_needs_full_tile_intermediate`),
  `multi_round_shuffle_composite_needs_no_scratch`.
- Integration suite (`--tests --test-threads=4`) rerun for golden
  parity.
- Clippy `-D warnings` clean — including fixing the two pre-existing
  `needless_range_loop` warnings (`kernel_ir::right_inverse`,
  `planner/list_v1.rs`) noted in the B.4-full verification.

**Next (per 2026-08-15 directive):** edge-case + prop tests exercising
swizzling/codegen for general reg/reg shuffles, shared→shared, and
shared→register with randomly generated linear layouts; then the
greedy chaining rewrite of `layout_infer` (readers processed in order,
converts chain — a tile converted to layout B feeds subsequent readers
at B; shared only materializes for writers whose access function isn't
expressible as a linear layout).

### B.5.1 · ConvertLayout edge-case + prop test suite (2026-08-15)

**Status:** landed. New integration suite
`crates/compiler/tests/convert_layout.rs` (15 tests) plus a DSL-level
end-to-end test in `gpu_macro.rs`. The random-layout prop tests
immediately caught **two real compiler bugs** (below) that every
deterministic layout in the existing suites had masked.

**Test machinery.** Programs are hand-built at the KIR level (bypassing
`layout_infer`, exercised separately) so each test pins the exact
`(src, dst, map)` triple fed to `gen_convert`. A `Chain` is
`input → t0 → … → tN → output`: global-load copy par into `t0`, one
`ConvertLayout` per link, copy-par store from `tN`. Tiles pick
Register/Shared freely; `maps[i]` is the i-th link's logical map
(`dst[v] = src[map(v)]`). Distinct input values (`v*7 + i`) make any
misrouted element visible. `run_chains` bundles many chains into one
`KirProgram` (one nvcc compile per test). Randomness is seeded
splitmix64; `rand_bijective` rejection-samples invertible XOR-affine
maps, 50% with a random offset.

- CPU prop tests (no GPU): pull-scheme simulation (every `(tid, slot)`
  covered exactly once, σ in range, warp bits preserved),
  `shuffle_rounds` ↔ dispatch-path agreement (incl. `dirs` is a
  canonical RREF basis), scratch sizing ↔ `best_decomposition`
  agreement, and `right_inverse`/`inverse` roundtrip on dense random
  affine maps.
- CPU source-marker tests: exact 16-shuffle multi-round composite,
  Slot strategy emits zero shuffles, Bounce stages through a
  full-tile scratch with exactly one `__syncthreads`, Shared→Shared
  is a direct loop with zero scratch. (Marker counts are taken from
  `__global__` onward — the arithmetic prelude's FpExt `__shfl_sync`
  overload otherwise pollutes them.)
- GPU edge cases: reg/reg (copy, slot swap, lane rotation, affine
  offsets, 1- and 2-dir multi-round, single-slot, two-hop chain),
  warp-crossing at block=64 (bounce, warp-identity multi-round,
  bounce roundtrip), shared endpoints (reg→shared plain/swizzled,
  shared→shared swizzled, shared-swizzle→reg, reg→shared→reg
  sandwich), non-identity `map`s on all three space pairs.
- GPU prop tests: random 2–4-tile chains with random spaces + random
  bijective affine layouts at block=32 (shuffle-only) and block=64
  (bounce-triggering), plus random non-identity maps.
- DSL end-to-end (`gpu_macro.rs::multi_round_shuffle_transpose_read`):
  producer+reader pars under `(th, s) -> th*4 + s` with a
  `tile[j % 32 * 4 + j / 32]` transpose read — layout_infer promotes
  it to the exact 16-shuffle multi-round pull (asserted on source,
  no `__syncthreads`), and the GPU result matches the host model.

**Bug 1 — `LinearLayout::right_inverse` inverted partially-reduced
columns** (`kernel_ir.rs`). The Gauss–Jordan loop snapshotted
`preimage[r] = inv[pivot]` *during* elimination round `r`, but the
pivot column still carries other output bits that only later rounds
clean (each bit r' is zeroed from every column except its own pivot).
So for dense matrices `M(pre[r]) = e_r ^ (not-yet-eliminated bits)`,
i.e. the "right inverse" wasn't one. Permutation and triangular
layouts (all the deterministic tests: swaps, bit reversal,
xor-swizzles, rotations) get cleaned in processing order, which is why
this survived every existing suite; a random dense bijective layout
broke immediately (`gpu_prop_random_chains_block32`, chain
`rnd_b32_1`). Fix: record pivot indices during elimination and read
the preimages off `inv[]` only after elimination completes. The
min-weight/canonical-replica property is unchanged (pivot columns only
ever mix with other pivot columns).

**Bug 2 — Shared→Shared `gen_convert` applied the dst layout twice.**
The arm emitted `b_dst[ld(x)] = b_src[f(map(ld(x)))]` — `x` treated as
logical on the write side but `ld(x)` fed to the read side as if it
were logical too. Net effect: the dst tile held `src[map(ld(x))]` at
logical `x` instead of `src[map(x)]` (observed as "swizzle applied
once" in `gpu_shared_edge_cases::shared_to_shared_swz` and as
`R∘S` instead of `R` in `gpu_non_identity_maps::map_rev_shared`).
Fix: `b_dst[ld(x)] = b_src[f(map(x))]`, loop over logical `x` —
consistent with the Reg↔Shared arms and `access_str` (shared layout =
logical→phys everywhere).

**Verification (this step):** `convert_layout` 15/15;
`gpu_macro` 63/63 incl. the new DSL test; full `--lib` 424/424
(`--test-threads=8`); full `--tests` suite green
(`--test-threads=4`); clippy `-D warnings` + nightly fmt clean.

### B.7 · greedy chained layout inference (2026-08-15)

**Status:** landed. `layout_infer` rewritten around a single
program-order walk per kernel (`greedy_chain` → `walk_chain_block`),
replacing the old two-phase `assign_par_attrs_and_writes` +
`emit_convert_layouts` split (both deleted, along with
`classify_reader`/`ReaderTarget`).

**Algorithm.** Per par statement, in program order:

1. **Par-attr inference** (`infer_and_set_par_attr`, unchanged B.6.1
   scoring except producer layouts now resolve through the chain tip):
   user-specified attrs are kept; otherwise pick the attr minimizing
   convert cost against the current effective layouts of the read
   tiles; non-linear reads fall back to the standard
   `tid + slot * blockDim`.
2. **Read chaining** (`chain_par_reads`): for each kernel-local tile
   read, resolve the buffer through `tip: BTreeMap<BufId, BufId>`
   (absent = original) and classify:
   - *Scoreable* (Linear index, `!spans_grid`, concrete pow2 `n`,
     const pow2 bound ≤ n): if the current version is Register and
     `maps_agree(eff, layout)` — rewire to it, no convert. Otherwise
     emit a ConvertLayout to a fresh Register version with layout
     `eff = pad(g) ∘ pad(attr.layout)` and rewire the read to
     `index = Linear(g_padded)`. Shared tips always convert down to
     registers — linear readers live on registers, per the plan.
   - *Non-scoreable*: original Shared → read the original in place
     (converts never mutate their source). Current version Register →
     emit a Register→Shared mirror (`_sm`) with the original's shape
     and no layout, and rewire (index kept). Current version already
     Shared → rewire to it.
3. **Write layout assignment** (`assign_layouts_for_par_writes`,
   gates unchanged): a tile is promoted to Register with layout
   `l = f ∘ attr` only when the writer's access function is a
   right-invertible linear layout over all `kb` phys bits, there is a
   single writer, no grid span, no back-edge reader, and no
   self-read mismatch — otherwise Shared + identity. This is exactly
   "shared only materializes when the writer's access function is not
   convertible to a linear layout".
4. **Tip maintenance**: writes reset the written buffers' tips (all
   later readers see the original again). Chained versions become the
   new tip, so if buffer A gets converted to layout B, subsequent
   readers wanting B rewire for free — the greedy chain.

**Semantics decisions (deviations worth recording):**

- **Per-par snapshot, not intra-par chaining.** All reads of one par
  classify against the tip *at par entry*; the chain only advances
  across statements. Strict sequential chaining within a par would
  make `f(tile[perm(j)], tile[j])` pay a convert *back* for the
  second read (the first read's version has the permuted layout).
  Intra-par versions dedup by `(orig, space, layout, shape)`; the
  last chainable version becomes the tip after the par.
- **Dead-end rule.** A Register version whose layout has no
  `right_inverse(kb)` (folding reads, e.g. `tile[j % 4]`) is emitted
  but never becomes a tip — converting *from* a non-surjective layout
  has no well-defined pull scheme. Shared versions always chain.
- **Loop back edges.** Tips for every buffer written anywhere inside
  a loop body are cleared at loop entry (a version created before the
  loop may predate an iteration's write), and the whole tip map is
  restored to its entry snapshot at loop exit: versions created
  *inside* the body don't dominate the statements after the loop (it
  may execute zero times), and body-written buffers' in-loop tips may
  predate the last iteration's write.
- **Placement.** Version + scratch Allocs go right after the original
  tile's Alloc. The convert goes: (a) right after the source
  version's own convert (`def_group`) when chaining off a version —
  same execution frequency by construction; (b) right after the
  writer par when the source is a Register original (old behavior);
  (c) immediately before the reader par when the source is a Shared
  original — safe for multi-writer/back-edge tiles because it
  re-executes each iteration, after all preceding writes.
  `insert_sync` then fences the dirty shared source before the
  convert (statement-level, so `verify`'s no-convert-inside-par rule
  holds for every placement).
- **Shared→Register versions need no new scratch or codegen.**
  `convert_scratch_bytes` is 0 for every Shared-touching pair, and
  the Reg←Shared `gen_convert` arm already handles sub-block
  replicated dst layouts: writer-par guards zero exactly the phys
  range that `right_inverse`'s canonical-replica preimages avoid, so
  population range and read range coincide.

**Test fallout:** only `multi_writer_tile_stays_shared` changed — its
`tile[row, col]` reader linearizes to the identity, so it now lands on
registers via a Shared→Register convert
(`["alloc","alloc","alloc","par","sync","convert","par"]` after
insert_sync); the tile itself still stays Shared + identity. The other
6 layout_infer unit tests, the 3 insert_sync stmt-sequence tests, and
the full integration suites are unchanged by the per-par snapshot
decision.

**Future work (delegated, per plan):** replace the greedy chain with a
search over conversion placements (e.g. hoisting converts out of
loops, choosing between chaining off a version vs. the original, or
global ILP over the conversion DAG) at the cost of more compile-time
resources.

**Verification (this step):** `layout_infer` unit tests 7/7; full
`--lib` 424/424 (`--test-threads=8`); full `--tests` suite green
(`--test-threads=4`); clippy `-D warnings` + nightly fmt clean.

## Phase B design decisions

Resolved from the "open questions" list. Marked with a date so the
doc doubles as the rationale trail.

### Cost model (2026-08-14)

**Constants:**
- `SHUFFLE_COST = 3` cycles per `__shfl_sync` round.
- `SHARED_COST = 30` cycles per shared round without bank conflicts.
- `k`-way bank conflict multiplies `SHARED_COST` by `k` (so 30·k
  cycles for a k-way conflict).

**Encapsulation.** These numbers are hardware-specific (chosen for
the current RTX 5090 / GB202 target). Wrap them in a single
`ConversionCostModel` (or `struct LayoutCostConstants`) type in
`crates/compiler/src/passes/layout_cost.rs` (new file) with methods
`shuffle_round_cost(&self) -> u64`, `shared_round_cost(&self,
bank_conflict_factor: u64) -> u64`, and `sync_cost(&self) -> u64`.
Instantiate once at pass-driver entry and thread through
`best_decomposition`, `choose_target`, `optimal_shared_swizzle`
scorers, and the fusion-v2 estimator. Rationale: when we retune per
hardware (or add per-op measured overrides from a benchmark run),
the change lives at one construction site — no scattered constants
to hunt.

**Loop weighting.** Multiplicative on the raw cycle count: per the
plan's B.6.1/B.6.2/B.6.3 convention, each cost is multiplied by
`loop_weight(op) = product of enclosing loop iteration counts`, with
`SYMBOLIC_WEIGHT ≈ 2^30` per unknown-bound loop. Symbolic bounds
compound multiplicatively — a symbolic loop nested inside another
symbolic loop weights `2^60`, which stays well below `u64::MAX`
even for reasonable depths.

### Fusion-v2 cost estimator awareness (2026-08-14)

**Ruling:** yes, the estimator must be `ConvertLayout`-aware once
B.1 lands. Under B.1's design, the number and shape of ConvertLayout
ops varies per candidate — a candidate that fuses a cross-warp
consumer into a producer's tile may need a costly shared-bounce
ConvertLayout that a keep-unfused variant avoids entirely. Scoring
that difference is exactly the estimator's job.

**How it plugs in:** the estimator already calls
`ModuleCompiler::lower(stamped_module)` at `fusion_v2/cost/estimator.rs:171`
and gets a `KirProgram` back. Under B.1 the KIR will contain
ConvertLayout ops as first-class statements. Walk them at
estimator time, run `best_decomposition` with the shared cost
constants, sum the resulting cycle counts (loop-weighted), and add
to the estimator's existing launch-cost term. Do NOT try to model
this without lowering — the layout choices are lowering artifacts,
and any pre-lowering heuristic would be less accurate than just
lowering and inspecting. The estimator's per-run overhead
(~0.3 ms/run per the current logs) grows a bit; the `kernel cost
cache` at the estimator entry point already dedupes by module hash,
so repeat costs are still O(1).

### `Promotion { kb }` — derive from buffer shape (2026-08-14)

**Ruling:** the `kb: usize` field added in A5 is a workaround, not a
design constraint. Under the paper's model a `LinearLayout` maps
`(Reg, Lane, Thr) → logical` — the *logical* space is external to
the layout, tracked by the buffer's shape / the compute def's tensor
type. Layout output width need not equal `bases.len()`; it's
determined by the tensor.

**Follow-up:** during B.1 (which dissolves `promote_tiles` entirely
into the ConvertLayout-first flow), derive `kb` inline from the
target buffer's shape (`log2(buffer.len())`) at the point of use —
`gen_convert`, `promotion.kb` reads, ConvertLayout emission. Drop
the cached `kb` field. This matches the physical / logical
separation the paper makes explicit.

Phase A left the field in place because it was the smallest change
to keep the codegen path indexing correctly under warp-aligned
input widths; the "correct" fix is a Phase B refactor along with
the rest of the promote_tiles → ConvertLayout migration.
