# kernel_ir codegen gaps

Gap analysis between the current `kernel_ir` → `codegen` pipeline and the
design target: a generic linear-layout emission model built around the
Triton-style "linear layouts" framework (*Linear Layouts: Robust Code
Generation of Efficient Tensor Computation Using 𝔽₂*, arXiv:2505.23819;
paper quotes below verified against the v5 HTML rendering).

Trigger: the `fused_drop` deadlock uncovered by the frac bench at LOG_N=26 —
fusion produced a kernel launched with `blockDim.x = 4` containing
`__shfl_sync(0xFFFFFFFFu, …)` inside `if (threadIdx.x < 2u)`. Two distinct
mask violations compound there:

1. the mask names lanes that don't exist (`blockDim.x = 4 < 32`) — Gap 1;
2. the mask names lanes that took a divergent branch (`gen_shuffle`'s
   `n < block` guard) — Gap 5.

Either alone hangs the shuffle. The second is reachable even at
warp-multiple blocks: `classify_convert` returns `Shuffle` for a `kb = 4`
conversion at `block = 32`, and `gen_shuffle` wraps it in
`if (threadIdx.x < 16u)` under a full mask. (The comment at
`codegen.rs:1115` claiming "`classify_convert` guarantees kb >= 5" is
stale — `classify_convert_cases` in `kernel_ir.rs` asserts `Shuffle` for
`kb = 4` sub-warp cases.)

## The paper's model, stated precisely

- **Def 4.1** — a linear layout is a linear map between *labeled* vector
  spaces over 𝔽₂, canonically `L: Reg × Thr × Wrp → 𝔽₂ⁿ × 𝔽₂ᵐ`. Direction
  is **hardware → tensor**. Memory layouts likewise map offsets → tensor
  coordinates (§4.3). Least-significant bits come first.
- **Matrix orientation** — columns are images of *input* (hardware) basis
  vectors; rows are output (tensor) bits. Replication/broadcast is **zero
  columns** (§5.1: "identifying threads and warps with duplicated data
  reduces to detecting zero columns in the layout matrix"). Not zero rows.
- **Def 4.10** — a distributed layout is a **surjective** linear layout
  from Reg × Thr × Wrp onto the logical tensor where each column has at
  most one non-zero bit and no two non-zero columns repeat ("a permutation
  matrix that may have additional zero columns interleaved").
- **Def 4.5** — a right inverse exists iff the layout is surjective **onto
  its full codomain** (not "onto its image", which is vacuous); computed by
  Gaussian elimination over 𝔽₂ as the least-squares solution of `MX = I`.
- **§5.4** — the conversion from source layout `A` to destination layout
  `B` is `B⁻¹ ∘ A` (hardware_src → hardware_dst), with `B⁻¹` a *right*
  inverse ("While B need not be invertible, it is surjective as it
  represents the entire logical tensor"). Selection criterion (2)
  ("Promoting broadcasting"): among solutions of `BX = A`, set the slack
  variables to zero to get the minimal-Hamming-weight solution, so that
  "all the elements pointing to the same value in the logical tensor read
  from the same input execution unit". Note the min-weight condition is on
  the *composite* solution `X`, not on `B⁻¹` in isolation.
- **§5.4 classification** — intra-thread when the composite is a pure
  register permutation; intra-warp "if `(B⁻¹∘A)_Wrp` is the identity";
  otherwise shared memory (with optimal swizzling, Appendix 9.2). In the
  shuffle construction every thread sends and receives exactly one element
  per round — **no divergence guards**; rounds = `2^|R|` where `R` extends
  `V ∪ I ∪ G` (shared vectorization bits, shared thread bits, paired
  exchange bits) to a basis. The construction *assumes no broadcasting* in
  `A`/`B`; broadcasting is resolved upstream by criterion (2).
- **Hardware space** — warp size is a hardware constant (`Thr = 𝔽₂⁵` for
  32 lanes; §4.1, §2.1 — *not* §5.2, which is mixed-precision matmul). The
  paper never models sub-warp compute domains.
- **What the paper does NOT contain**: affine layouts (`y = Ax ⊕ b` is
  future work, §8); any store-predication rule for replicated layouts (§6
  mentions avoiding "redundant load and store instructions" but states no
  condition — the guard rule in Gap 7 is our design, not paper text); any
  multi-stage decomposition of one `convert_layout` into shuffle + shared
  stages; non-power-of-two shapes (§8: pad and mask).

## Conventions in this codebase vs. the paper

- `LinearLayout` (`kernel_ir.rs:94-102`) is **XOR-affine**: `T(x) = M(x) ^
  offset`, `bases[i] = M(1 << i)` — so `bases` are the paper's *columns*.
  The affine offset is already load-bearing (butterfly partner reads,
  `Plan::View` layouts), i.e. we already exceed the paper's linear model;
  every inverse/right-inverse routine must handle the offset.
- There are no labeled dims and **no explicit codomain**: `bases.len()`
  fixes the input width, the output width is implicit. `inverse()`
  (`kernel_ir.rs:142`) therefore only handles square bijections, and
  "surjective" is not even expressible without passing a codomain width.
- Physical space is `x = slot * blockDim + thread` — a `(Reg, Thr)` pair
  with **no structural lane/warp split**; the split is re-derived ad hoc
  inside `classify_convert` (`tb = k.min(log2(block))`, `lane_bits =
  tb.min(5)`, `kernel_ir.rs:196,221`). `block` is a workload parameter,
  not a hardware constant, which is exactly what the paper's fixed
  `Reg × Thr × Wrp` space forbids.
- **Two direction conventions coexist** (`kernel_ir.rs:296-304`): register
  buffers and `ParAttr` layouts map physical → logical (the paper's
  direction); shared/global `BufferDecl::layout` maps logical → physical
  (the reverse of the paper's memory-layout convention). `gen_convert`
  composes inverses differently per buffer-kind pair as a result.
- The conversion map is oriented **destination → source**: `layout_infer`
  classifies `C = L⁻¹ ∘ E` (`layout_infer.rs:263`) where `L` is the
  producer's layout and `E = g ∘ f_reader` the reader's effective map, so
  `C : phys_dst → phys_src`. This is the *inverse orientation* of the
  paper's `B⁻¹ ∘ A` — receiver-driven, matching `__shfl_sync` (each
  destination lane names its source). Consequence: it is the **source**
  layout that must be surjective (right-invertible) for `C` to exist,
  which holds for any distributed layout by Def 4.10.

## Target model

Every codegen unit launches with `blockDim.x = k.block`, a compile-time
constant **multiple of 32**. For a `compute[T]` with `T = 2^t ≤ 32`,
`k.block = 32` and the par uses a *replicated* layout on the full 5 lane
bits (`seq_size = 1`, so no slot bits):

    bases = [1, 2, …, 2^(t-1), 0, …, 0]   (length 5), offset = 0

The zero **columns** in positions `t..5` express the replication: `32/T`
register-identical replicas of each logical index across the warp.
`logical_id = threadIdx.x & (T-1)` falls out of forward evaluation — no
codegen special case emits the `& (T-1)`.

For a producer `A` at layout `T_A` and a reader
`B = compute[T'] |i| { A[g(i)] }` with par layout `f_B`, the reader needs
`A`'s data at `E = g ∘ f_B` (phys_B → logical_A). The conversion map is

    C = T_A⁺ ∘ E : phys_B → phys_A

with `T_A⁺` the minimal-weight right inverse, so replicated sources
collapse to one canonical replica (paper §5.4 criterion (2) ⇒ broadcasts,
minimal shuffle traffic). Codegen decomposes `C` into intra-thread
(register permutation), intra-warp (`__shfl_sync`, all lanes, full mask),
and inter-warp (shared bounce) stages.

## Broadcasting without broadcast ops

The DSL has no broadcast op, but broadcasts still arise — as *layout
facts*, visible only as zero columns in maps the pipeline already
computes. Four op-less sources:

1. **Producer-side replication** (Gap 2): `compute[T]` with `T < 32`
   launched on a full warp. The par layout has zero columns on lanes
   `log2(T)..5`; every logical element lives in `32/T` replica lanes.
2. **Consumer-side folds**: a reader indexing `A[i % c]`, `A[i / c]`, or
   `A[0]` (power-of-two `c`) has a *linear but non-injective* access map
   `g` — a projection whose dropped input bits are zero columns of
   `E = g ∘ f_reader`, even when the par layout `f_reader` is bijective.
   Today these route to `Plan::Mirror` (the
   `non_linear_reader_gets_shared_mirror` test pins `tile[j % (t/2)]` →
   shared mirror); in the target model they are ordinary conversions
   whose `C` happens to be many-to-one — `__shfl_sync` broadcasts
   natively, no shared memory needed.
3. **Sub-domain readers** (paper: slicing, Prop 4.8): a reader whose
   domain is smaller than the producer's tile reads a slice; the pad
   bits are dead. Zero-column padding (Gap 2's `pad_layout_identity`
   fix) is the honest encoding — today's identity-padding fabricates
   live bits precisely so the bijectivity checks pass.
4. **Reduce results**: a fully reduced value logically lives in a 0-bit
   space; its distributed layout is all zero columns (every lane holds
   it). Any consumer's `C` is then the zero map — classified `Copy`,
   consumed for free from the local register. This is the mechanism
   behind §6's observation that zero-column detection "avoid[s]
   redundant load and store instructions".

Consequences for inference (the first is a correction to Gap 3 as
previously stated; the rest are constraints the generic model must
keep):

- **(a) The Direct-plan test must be function equality, not
  `classify(C) == Copy`.** Under replication `C = L⁺ ∘ L` is a
  projector, not the identity — see the amendment in Gap 3.
- **(b) One fixed min-weight `T⁺` per source.** All destinations wanting
  logical `i` then name the same canonical source phys — one sender per
  element, natively broadcast by `__shfl_sync`, and Gap 3's sender-slot
  fiber condition holds by construction.
- **(c) Read/write asymmetry.** Broadcasting is legal for reads
  (non-injective `E` is fine) and illegal for logical writes:
  `check_accesses.rs:210`'s injectivity check on logical write maps is
  not a casualty of replication — it stays.
- **(d) Memory layouts stay injective.** Zero columns belong to
  *distributed* (register/par) layouts only. `Shared`/`Global`
  `BufferDecl::layout` maps logical → physical; a zero column there
  would alias distinct logical elements to one address. Replication of
  data in memory is expressed on the *access* side (`E`), never in the
  buffer layout.

## Ad-hoc machinery in today's pipeline

An inventory of the special cases that the generic model would subsume.
Each is a place where inference either bails to shared memory or bakes in
a bijectivity assumption that replication violates.

**Lowering (`lower_to_kir.rs`)**
- `block` policy: `min(bound, 256)` / `threads` hint / monomorphize block
  hint (`:170-218`); `#[grid(threads=t)]` accepts any power of two in
  `1..=1024` including sub-warp values (`:162-168`).
- `#[par]` spec validation requires `bound ≥ block`, both powers of two,
  and a **bijective** layout on exactly `log2(bound)` bits
  (`spec_attr`, `:494-512`). A replicated par (`bound < block`) is a hard
  error on this path today.

**Layout inference (`layout_infer.rs`)**
- Default par attr: `identity(ceil_log2(seq_size) + ceil_log2(block))`
  (`:66`) — injective by construction, sized to the workload, not to the
  hardware.
- `linearize_accesses` (`:90-114`): only accesses depending on *exactly*
  the par's own index become `Linear`; anything touching a loop var or a
  second symbol stays `Affine` → unanalyzable → mirror.
- `promote_tiles` preconditions, each a silent bail to shared memory:
  single writer; writer non-grid-spanning with `bound == len ==` power of
  two; write map `Linear` on exactly `kb` bases; `l = g_w ∘ f_w`
  **invertible** (`:208`); writer self-reads must equal the write map
  *exactly* — a shuffleable self-read aborts the whole promotion
  (`:226-234`); no back-edge reads (`:234-238`); readers non-grid-spanning,
  power-of-two bound `≤ n`, `Linear` (`:239-267`).
- `pad_layout_identity` (`:80`): sub-domain readers are padded with
  **identity** bases to force a square bijective composite. The padded
  input bits are dead (always zero at runtime), but the classifier treats
  them as live identity bits — the honest model is a rectangular map
  classified by surjectivity; zero-column padding is the approximation
  consistent with replication (see Gap 2).
- `Plan::Direct` requires `classify(C) == Copy` — a test that turns
  false-negative under replication, where `C = L⁺∘L` is a projector,
  not the identity (see the Gap 3 amendment);
  `Plan::View` allocates one view buffer per distinct effective layout;
  `Plan::Mirror` is the catch-all. `ConvertLayout` ops are only ever
  inserted with `map = identity(kb)` (`:323-330`, `:346-350`) — the
  general `map` in the opcode is unexercised.

**Classifier (`kernel_ir.rs:186-248`)**
- Non-power-of-two `block` → `Bounce` (`:193`); `tb == 0` → `Bounce`
  (`:211`); offset bits in the warp positions → `Bounce` (`:226`); warp
  block must be exactly the identity (`:229-234`); lane block must be
  **bijective** (`:243`). Slot→lane mixing is allowed; lane→slot mixing
  is legal but routes `gen_shuffle` to its general path.

**Codegen (`codegen.rs`)**
- Launch: `__launch_bounds__(k.block)` (`:370`) and `dim3(k.block)`
  (`:1589`) take whatever lowering picked, sub-warp included.
- Bounds guards conflate three roles: tail masking (`v < bound` loop
  condition / `if (v >= bound) continue;` at `:657`), replica dedup
  (implicit — surplus lanes simply never run), and store dedup. The
  target model separates them: compute runs on every lane (replicas
  included), only stores are predicated (Gap 7), and non-power-of-two
  tails keep a mask *on the stores*.
- `gen_convert` supports only Register→Register (`Copy`/`Slot`/`Shuffle`;
  `Bounce` is a **compile error**, `:1030-1035`) and Shared←Register
  (requires `map` invertible, `:1044`); the source register layout must
  be invertible (`:1010`); Shared→Register and Shared→Shared are
  unsupported (`:1068-1073`). No conversion ever emits more than one
  stage.
- `gen_shuffle`: the `n < block` guard (`:1117-1121`, Gap 5); the general
  path takes a strict inverse of the lane block (`:1148-1165`, Gap 4);
  the fast path's receiver-driven forward formula (`(cs ^ ct) & 31`)
  never needed an inverse.
- `check_accesses` (`check_accesses.rs:210`) checks injectivity of
  *logical* write maps (par index → buffer index); this is about write
  races in the logical domain and stays valid under replication — no
  change needed there.

## Gaps

Numbered in pipeline order. Gaps 1–5 are the deadlock fix; 6–7 are
emission quality; 8 is the broader refactor.

### Gap 1 — Physical launch dimension isn't warp-aligned

**Where:** `lower_to_kir.rs:170-218` (block selection: `threads` hint,
`min(bound, BLOCK_SIZE)`, or monomorphize block hint);
`codegen.rs:370` (`__launch_bounds__`), `:1589` (`dim3(k.block)`).

**Current:** `compute_with(WIDTH, …, threads: Some(WIDTH))` with WIDTH=16
gives `kernel.block = 16`; the fused_drop fusion gives `kernel.block = 4`.
Any `__shfl_sync(0xFFFFFFFFu, …)` in such a kernel is malformed per se:
the mask names lanes that are never resident.

**Paper (§4.1, §2.1):** layouts live in the fixed physical space
`Reg × Thr × Wrp`; the lane bit-count is a hardware constant (5), not a
workload parameter. Sub-warp physical dimensions aren't part of the model.

**Fix:** in lowering, set `k.block = max(32, block)` (round up to a
multiple of 32; all current paths already produce powers of two). Sub-warp
semantics move *inside* the layout as zero columns (Gap 2). Knock-on
effects the previous revision missed:

- `spec_attr` (`lower_to_kir.rs:494-512`) errors on `bound < block` and
  requires a bijective spec layout — the `#[par]` path needs
  replication-aware validation (a spec over `thread < T` must be extended
  with zero columns on lanes `log2(T)..5`, not rejected).
- `layout_infer`'s default attr and `promote_tiles` composition consume
  `k.block`; without Gap 2 they'd produce `identity(5)` layouts plus a
  `x < T` bounds guard — which merely relocates the divergence, it does
  not remove it.

### Gap 2 — Par layouts are `identity(log2 T)`; should be 5-bit-wide with zero columns on replicated positions

**Where:** `layout_infer.rs:66`:

    *attr = Some(ParAttr {
        seq_size,
        layout: LinearLayout::identity(ceil_log2(seq_size) + ceil_log2(block)),
    });

and `pad_layout_identity` (`layout_infer.rs:80`).

**Current:** for `compute[T]` at `block = T`, the par layout is a
`log2(T)`-bit identity — a bijection of a workload-sized space. The
lane/warp interpretation of those bits is derived later inside
`classify_convert`.

**Paper (Def 4.1, Def 4.10, §5.1):** the matrix's columns are the
*hardware's* input bits; replication is zero **columns** (the previous
revision said "zero rows" — wrong orientation for both the paper and our
`bases` representation, where `bases[i]` is the image of input bit `i`,
i.e. a column).

**Target:** for `compute[T]`, `T = 2^t ≤ 32`, emit the 5-column layout
`bases = [1, 2, …, 2^(t-1), 0, …, 0]`, `offset = 0`. For `T > 32` extend
with slot/warp columns as today. `pad_layout_identity` must pad with
**zero** columns when the extended range crosses the replication boundary
(bits `≥ log2(T)`), identity below it — zero-padding is also the honest
semantics for the reader-smaller-than-tile case (those inputs are dead),
whereas today's identity-padding fabricates live bits to satisfy the
bijectivity checks that Gaps 3–4 remove.

**Knock-on (missed previously):** with a replicated (non-injective)
`a.layout`, `promote_tiles`' `l = f.compose(&a.layout)` is non-invertible
and the `l.inverse()` gate at `layout_infer.rs:208` silently demotes every
such tile to a shared mirror — a performance cliff, not an error. That
gate must become "surjective onto the tile's `kb` bits" with the layout
stored as a rectangular map (Gap 4's right inverse replaces `l_inv` in
the `C = L⁺ ∘ E` composition at `:263`).

### Gap 3 — Classifier requires a bijective lane block; the real conditions are weaker

**Where:** `kernel_ir.rs:243`:

    if lane_block.inverse().is_some() {
        ConvertKind::Shuffle
    } else {
        ConvertKind::Bounce
    }

**Current:** the lane→lane sub-block of `C` must be a bijection, so any
replicated layout falls to `Bounce` — which `gen_convert` then rejects as
a compile error for register pairs (`codegen.rs:1030`).

**Paper (Def 4.5, Def 4.10, §5.4):** distributed layouts are surjective,
not injective; `B⁻¹` in `B⁻¹∘A` is a *right* inverse, existing precisely
because the layout is surjective onto the full logical tensor.

**Precise requirements** (for `C : phys_dst → phys_src`, receiver-driven):

1. `C` must exist — i.e. the *source* layout is surjective onto the
   logical space, so `C = T_src⁺ ∘ E` is total. Every destination then
   has a well-defined source; `Shuffle` never needs the lane block to be
   injective *receiver-side* (multiple destinations naming one source is
   exactly a broadcast, which `__shfl_sync` supports natively).
2. Warp component of `C` identity, no offset bits in warp positions
   (unchanged from today, `kernel_ir.rs:226-234`).
3. Sender-slot resolvability, per destination slot `s'`: every source
   lane must be able to present a single register slot. Fast path: the
   needed source slot `C(s' << tb ^ tid) >> tb` is tid-independent
   (`const_src_slot`, `codegen.rs:1135`) — holds for the whole
   butterfly/XOR-offset family. General path: the source-slot function
   must be constant on the fibers of `tid ↦ source-lane` — guaranteed
   when `C` is built through a *fixed minimal-weight* `T_src⁺` (all
   replicas of a logical element resolve to the same source phys), and
   checkable by comparing `C` on fiber representatives otherwise.

**Amendment — the Direct test comes *before* the composition.**
`layout_infer.rs:263-264` plans `Direct` iff
`classify_convert(l_inv.compose(&eff)) == Copy`. Under replication this
is a false negative: when the reader wants exactly the producer's layout
(`E ≡ L` as functions — equal on every physical index, offset included),
the composite `C = L⁺ ∘ L` is **not** the identity but the idempotent
projector onto canonical replicas (`L⁺` fixes one preimage per logical
element; every other replica maps onto it). `classify(C)` then reports
`Shuffle`, and inference emits a self-broadcast for data every lane
already holds. The correct pipeline:

1. `maps_agree(E, L)` (function equality) → `Plan::Direct`, no data
   movement;
2. otherwise compose `C = T_src⁺ ∘ E` and classify the *residue*.

Codegen already owns the right primitive — `maps_agree`
(`codegen.rs:1382-1386`), used for the own-slot store check at
`:1404-1414`; inference must apply it *before* composing with the right
inverse, not inspect the composite afterwards. (Today the two tests
coincide because every layout is bijective and `C = L⁻¹ ∘ E` is the
identity exactly when `E ≡ L`; replication is what splits them.)

Under fused_drop's `compute[4] → compute[2]` butterflies
(`T_c = T_p = T`), condition 1 holds trivially and the classification is
`Shuffle` with broadcasts.

The previous revision's fix sketch ("extract the low `log2(T_c)` rows × 5
columns … verify the row-span covers the non-zero output columns") mixed
rows and columns; the correct formulation is above: surjectivity is a
property of the source layout's column span covering the logical bits,
checked when the right inverse is computed (Gap 4), not a post-hoc rank
test on the composite's lane block.

### Gap 4 — `LinearLayout::right_inverse` doesn't exist

**Where:** `kernel_ir.rs` (`LinearLayout` has only square `inverse()`,
`:142-163`). Call sites that assume bijectivity and must migrate:

- `layout_infer.rs:208` — promotion gate (`l.inverse()`);
- `layout_infer.rs:263` — `C = l_inv.compose(&eff)`;
- `codegen.rs:1010` — source register layout in reg→reg converts;
- `codegen.rs:1044` — `map.inverse()` in reg→shared converts;
- `codegen.rs:1160` — `gen_shuffle` general path
  (`.inverse().expect("classify_convert checked the lane block")` — the
  panic the previous revision cited at `:1148-1165`);
- `kernel_ir.rs:243` — the classifier check itself (Gap 3).

**Paper (Def 4.5, §5.4 criterion (2)):** right inverse = least-squares
solution of `MX = I` by Gaussian elimination over 𝔽₂; uniqueness is
resolved by zeroing the slack variables, giving the minimal-Hamming-weight
solution so replicas "read from the same input execution unit". (The
paper's min-weight condition is stated on the composite solution of
`BX = A`; computing a min-weight `B⁺` and composing achieves the same
canonical-replica effect for our layouts, whose non-zero columns are
distinct powers of two.)

**Fix:** add

    LinearLayout::right_inverse(&self, out_bits: usize) -> Option<LinearLayout>

1. Gaussian-eliminate `M` (the `bases` as columns) over 𝔽₂; return `None`
   unless the column span covers all `out_bits` (surjectivity onto the
   stated codomain — the codomain must be a parameter because
   `LinearLayout` carries none).
2. For each output bit choose the minimal-weight preimage (zero slack
   variables); zero columns contribute nothing, so replicated inputs
   resolve to the canonical replica.
3. Affine handling (beyond the paper, which defers affine to §8): for
   `T(x) = M(x) ^ c`, return `T⁺(y) = M⁺(y ^ c)`; then `T(T⁺(y)) = y`
   for all `y` in the codomain.

`inverse()` remains for the square-bijective fast paths; `right_inverse`
with `out_bits = bases.len()` degenerates to it.

### Gap 5 — `gen_shuffle`'s divergent lane guard breaks `__shfl_sync`

**Where:** `codegen.rs:1117-1121`:

    let guard = n < block;
    if guard {
        writeln!(s, "{pad}if (threadIdx.x < {n}u) {{").unwrap();
        pad.push_str("    ");
    }

**Current:** when the conversion domain is smaller than the CTA, the
shuffles are wrapped in `if (threadIdx.x < n)` while the mask stays
`0xFFFFFFFFu` — the mask names lanes that skipped the branch. Direct
cause of the fused_drop hang (together with the sub-warp launch, Gap 1),
and reachable at `block = 32` whenever `kb < 5` classifies `Shuffle`.

**Paper (§5.4):** in every shuffle round "every thread sends and receives
only one element" — all lanes participate, no guards; broadcasting is
handled by the min-weight right-inverse solution, never by masking
destinations out.

**Fix:** delete the guard; mask stays `0xFFFFFFFFu`. Ordering constraint:
this is only sound once Gap 1 (all 32 lanes resident) *and* Gap 2 (every
lane holds a defined replica, so surplus lanes have valid values to
present and valid sources to read) are in. Gap 1 + guard deletion alone
would already un-hang — the `& 31` in the source-lane formula keeps
sources in range — but lanes `≥ n` would then shuffle uninitialized
registers (formally UB, practically dead values); the replicated layouts
make every lane's value defined, which is the paper's model.

### Gap 6 — `blockDim.x` is a runtime variable in emitted `.cu`

**Where:** `codegen.rs:584` (grid-span index `blockIdx.x * blockDim.x +
threadIdx.x`), `:619`/`:626` (strided-loop `v += blockDim.x`), `:645`
(non-identity phys index `v_s * blockDim.x + threadIdx.x`), `:1055`
(convert staging loop).

**Current:** `blockDim.x` is a runtime uniform, so ptxas can't
constant-fold it against literal loop bounds (e.g. a `bound ≤ block` loop
provably runs once per thread, but the emitted loop still materializes
the compare and add).

**Fix:** `k.block` is a compile-time constant; substitute the literal
`{k.block}u` at every emission site (helper `blockdim_lit(&Kernel)`).
Pure emission change, no IR impact.

### Gap 7 — Writes under replication: the store guard must come from the right inverse

**Where:** all `Shared`/`Global` store paths — `gen_par_body`'s write
loop (`codegen.rs:978-981`), the reg→shared staging loop (`:1053-1066`).

**Current:** no replication exists yet, so dedup is implicit: surplus
physical indices fail the bounds guard and never run. Once pars are
replicated (Gap 2), every replica executes the body and each
`Shared`/`Global` store would be written `32/T`× (idempotent but `32/T`×
the traffic, and a real hazard the moment stores stop being idempotent).

**Wrong fix (previous revision):** guard with `threadIdx.x < T`. That is
correct only for the special case "identity on the low `log2(T)` lane
bits, zero columns above, any offset". It is wrong in general — e.g.
`bases = [0, 0, 0, 1, 2]` (T = 4 replicated across the *low* three lane
bits) has canonical storing lanes `{0, 8, 16, 24}`, not `{0..3}`.

**General rule:** let `T : phys → logical` be the par's (affine) layout,
surjective onto a logical domain of size `N` with `N | P`,
`P = seq_slots · 32 · num_warps`, and fix a right inverse `T⁺` (Gap 4).
The replica at physical index `x` stores logical element `i = T(x)` iff

    thread_component(x) == thread_component(T⁺(i))

i.e. `(lane, warp)(x) == (lane, warp)(T⁺(T(x)))`. The spatial (register
slot) component is **not** checked: a canonical thread whose slot bits are
replicated re-stores from the same thread — idempotent, no cross-thread
traffic, and predicating on the slot would buy nothing. Since `T⁺∘T` is
affine, the guard compiles to a mask-and-compare on `threadIdx.x`
(`((T⁺T)(x) ^ x) & thread_mask == 0`), constant-foldable per slot
iteration; in the canonical low-bit-identity case it folds to exactly
`threadIdx.x < T`.

Register stores must stay **unguarded** — not merely as an optimization:
every replica lane must write its own slot or later shuffles broadcast
stale/undefined values from the surplus lanes.

Non-power-of-two logical bounds are outside the 𝔽₂ model (paper §8: pad
and mask); they keep today's `x < N` tail mask, applied at the store,
composed with the replica guard.

(The paper states no such condition anywhere — §6 only reports that zero-
column detection "avoid[s] redundant load and store instructions". The
rule above is our design, consistent with the implementation the paper
describes.)

### Gap 8 — `ConvertLayout` exists but is narrow: promotion-driven, single-stage, register-centric

**Where:** `kernel_ir.rs:405-414` (`SSAOpCode::ConvertLayout`),
`layout_infer.rs` (`promote_tiles`, `Plan::{Direct,View,Mirror}`),
`codegen.rs:990-1076` (`gen_convert`).

**Current (the previous revision wrongly claimed the op doesn't exist):**
`ConvertLayout { dst, src, map }` is a first-class statement op, inserted
by `promote_tiles` for views and mirrors. The actual gaps are narrower
and sharper:

1. **Insertion is promotion-driven, not propagation-driven.** Conversions
   only arise from the shared-tile→register promotion path; the producer's
   layout is pinned by its write access, readers adapt or mirror. There is
   no anchor-layout assignment, no forward/backward propagation, and no
   rematerialization (paper §4.4: anchors from load/store and
   layout-constrained ops, forward propagation along use chains with
   conversions inserted at conflicts, backward rematerialization of cheap
   chains to eliminate conversions).
2. **`map` is always the identity.** Both insertion sites pass
   `identity(kb)` (`layout_infer.rs:323-330`, `:346-350`); the general
   `dst[i] = src[map(i)]` semantics of the opcode are unexercised and
   untested.
3. **Kind-pair coverage:** Register→Register handles `Copy`/`Slot`/
   `Shuffle` but makes `Bounce` a compile error (`codegen.rs:1030-1035`);
   Shared←Register requires an invertible `map` (`:1044`);
   Shared→Register and Shared→Shared are unsupported (`:1068-1073`). The
   Bounce fallback exists only as `Plan::Mirror` — a *different buffer*
   with a barrier, chosen at inference time, not a lowering of the same
   conversion.
4. **No composition.** Each conversion must fall entirely into one
   category; the paper's `B⁻¹A` factorization composes stages (register
   permutation ∘ shuffle ∘ shared bounce with optimal swizzling) for a
   single conversion.

**Fix (deferred):** re-plumb `layout_infer` to walk producer→consumer
edges, compute the required layout `E = g ∘ f_reader`, and emit
`ConvertLayout` wherever `E` differs from the producer layout; give
`gen_convert` the three-stage decomposition driven by Gap 4's right
inverse. The fused_drop deadlock does not need any of this — Gaps 1–5
suffice.

## Summary

| # | Gap | Needed for the deadlock? |
| --- | --- | --- |
| 1 | `k.block` not warp-aligned (mask names non-resident lanes) | yes |
| 2 | Par layouts workload-sized/injective; need 5-bit + zero columns | yes (soundness of 5; avoids silent mirror-demotion cliff) |
| 3 | Classifier demands bijective lane block; needs `maps_agree`-first Direct test + source-surjectivity + sender-slot rules | yes |
| 4 | `right_inverse` missing; six call sites assume bijectivity | yes |
| 5 | `gen_shuffle` divergent guard under full mask | yes |
| 6 | `blockDim.x` runtime var vs. compile-time literal | no (emission quality) |
| 7 | Replica store guard — from `T⁺`, not `threadIdx.x < T` | no (traffic; correctness once stores aren't idempotent) |
| 8 | `ConvertLayout` promotion-driven, identity-map-only, single-stage | no (broader refactor) |

## Corrections vs. the previous revision of this doc

- Replication is zero **columns**, not zero rows (paper §5.1; our `bases`
  are columns).
- Fixed-hardware-space claims live in §4.1/§2.1, not §5.2 (which is
  mixed-precision matmul).
- Def 4.5: right inverse exists iff surjective onto the full **codomain**
  ("onto its image" was vacuous). `LinearLayout` has no codomain field, so
  `right_inverse` must take one.
- The min-weight/broadcast rule is §5.4 criterion (2), phrased on the
  solution of `BX = A` with zeroed slack variables; "min-norm right
  inverse promotes broadcasting (page 10)" was a paraphrase.
- The paper contains **no** store-predication rule; Gap 7's guard is our
  design, and the previous `threadIdx.x < T` form was wrong for layouts
  that replicate on low lane bits (correct rule: thread component of
  `T⁺(i)`).
- `SSAOpCode::ConvertLayout` **does** exist and is inserted today
  (`kernel_ir.rs:405`, `layout_infer.rs`); Gap 8 restated as
  promotion-driven / identity-map / single-stage / kind-pair gaps.
- `classify_convert`'s bijective check is at `kernel_ir.rs:243` (not
  `:231`), and the doc's path for lowering is
  `passes/lower_to_kir.rs` (not `passes/lowering/`).
- The trigger is two independent mask violations (non-resident lanes +
  diverged lanes); the guard bug reproduces at `block = 32` with
  `kb = 4`, so Gap 1 alone is not sufficient. The `gen_shuffle` comment
  "classify_convert guarantees kb >= 5" is stale.
- The paper's conversion `B⁻¹∘A` maps source→destination hardware; our
  `C = L⁻¹∘E` is destination→source (receiver-driven). Equivalent
  information, opposite orientation — the surjectivity requirement lands
  on the *source* layout in our convention.
- The codebase is already XOR-**affine** (offset); the paper is purely
  linear (affine deferred to §8). All right-inverse machinery must carry
  the offset.
- Gap 3 as first rewritten said "classify `C = T_src⁺ ∘ E`" without
  qualification — incomplete: with `E ≡ L`, `C = L⁺∘L` is the projector
  onto canonical replicas, so `classify(C) == Copy` is a false negative
  for `Plan::Direct`. The `maps_agree(E, L)` function-equality test must
  come *before* the composition (Gap 3 amendment; broadcasting
  consequence (a)).

## References

- Linear Layouts: Robust Code Generation of Efficient Tensor Computation
  Using 𝔽₂. arXiv:2505.23819 (v5).
  - Def 4.1 — linear layouts as maps between labeled 𝔽₂ spaces,
    hardware → tensor; columns = input bits, LSB first.
  - Def 4.2 — composition as label-wise matrix multiplication.
  - Def 4.5 — right inverse via Gaussian elimination; exists iff
    surjective onto the codomain.
  - Def 4.10 — distributed layout: surjective; columns have ≤ 1 non-zero
    bit; non-zero columns distinct.
  - §4.4 — anchor layouts, forward/backward propagation,
    `convert_layout` insertion and rematerialization.
  - §5.1 — broadcasting = zero columns.
  - §5.4 — conversion `B⁻¹∘A`; criteria (1) identity blocks / (2)
    zero-slack min-weight solution; intra-thread / intra-warp
    (`(B⁻¹∘A)_Wrp = I`) / shared-memory classification; shuffle rounds
    `2^|R|` from the `V ∪ I ∪ G` basis, all lanes participating.
  - §8 — affine layouts and non-power-of-two shapes as future work.

## Implementation plan

Two phases. Phase A is the minimal, jointly-landing change set that fixes
the deadlock (Gaps 1–5); Phase B is the general machinery (Gaps 6–8 +
the broadcasting consequences). The phase split is forced by a
dependency chain, not preference:

    right_inverse (pure)  ──►  warp-aligned launch  ──►  replicated layouts
                                                             │
    classifier/inference migration  ◄────────────────────────┘
                │
    guard deletion (only sound once every lane is resident AND defined)

Landing Gap 1 or Gap 2 without Gap 3–4 turns today's working kernels
into `Bounce` compile errors or silent mirror demotions (the old
classifier rejects every replicated composite); landing Gap 5 first
shuffles uninitialized registers. So A3–A6 below ship as one change.

### Phase A — unblock the deadlock (Gaps 1–5)

**A0. Reproducer tests first** (test-first workflow):

- `gpu_macro.rs`: a `compute[4] → compute[2]` butterfly chain matching
  fused_drop's fused shape (today lowers to `blockDim.x = 4` + guarded
  full-mask shuffles — hangs). Asserts values, not just termination.
- `gpu_macro.rs`: a `kb = 4` register conversion at `block = 32` (the
  guard-only bug, Gap 5 without Gap 1).
- CPU unit tests for A1/A5 as listed per step.

**A1. `LinearLayout::right_inverse(&self, out_bits: usize) ->
Option<LinearLayout>`** (`kernel_ir.rs`, beside `inverse()` at `:142`).
Pure addition, no call-site changes yet.

Algorithm (Def 4.5 + §5.4 criterion (2), affine-extended):

1. Column-reduce `M` (`bases` as columns) over 𝔽₂, tracking pivot input
   bits. Return `None` if rank < `out_bits` (not surjective onto the
   stated codomain).
2. For each output bit `e_j`, solve `M·x = e_j` using **pivot columns
   only** (slack inputs = 0) → column `j` of `M⁺`. Zero columns are
   never pivots, so `M⁺`'s image avoids replicated inputs — the
   canonical-replica property falls out.
3. Affine: `T⁺(y) = M⁺(y ^ c)`, i.e. returned `bases` = columns of
   `M⁺`, returned `offset` = `M⁺(self.offset)`.

CPU tests (exhaustive over small widths, like `to_linear_layout`'s):
`T(T⁺(y)) == y` for all `y < 2^out_bits`; `T⁺∘T` idempotent;
degenerates to `inverse()` on square bijections; canonical replicas —
`bases = [1,2,0,0,0]` → `T⁺` image `{0,1,2,3}`, `bases = [0,0,0,1,2]`
→ image `{0,8,16,24}`.

**A2. Hoist `maps_agree`** from `codegen.rs:1382-1386` into
`kernel_ir.rs` (function equality over `2^in_bits`, offsets included);
codegen re-imports. Needed by A5's Direct test.

**A3. Warp-aligned launch (Gap 1)** — `lower_to_kir.rs:170-218`: after
the existing selection, `block = block.max(32)` (all paths already
yield powers of two). `spec_attr` (`:494-512`): drop the
`bound ≥ block` error; a spec over `bound < block` is accepted by
zero-extending the spec layout on bits `log2(bound)..log2(block)`
(bijectivity still required on the low `log2(bound)` bits).

**A4. Replicated par layouts (Gap 2)** — `layout_infer.rs:66`: the
default attr becomes

    bases[i] = 1 << i   for i < ceil_log2(min(bound, block))
    bases[i] = 0        for the remaining columns
    width    = ceil_log2(seq_size) + log2(block)

(degenerates to today's identity when `bound ≥ block`; replication only
occurs at `seq_size = 1`). `pad_layout_identity` (`:80`) → `pad_layout`:
identity columns below the source's logical width, **zero** columns
above. The `if (v >= bound) continue;` guard (`codegen.rs:657`) is dead
for replicated layouts (`T(x) < bound` always) — harmless, no change.

**A5. Classifier + inference migration (Gaps 3–4)** — the six call
sites:

- `layout_infer.rs:208`: promotion gate `l.inverse()` →
  `l.right_inverse(kb)`; keep the rectangular result for reuse below.
- `layout_infer.rs:263-264`: maps_agree-first (Gap 3 amendment):

      if maps_agree(&eff, &l)      → Plan::Direct
      else classify_convert(&l_rinv.compose(&eff), k.block) → …

- `kernel_ir.rs:243` (`classify_convert`): keep the warp-identity and
  no-offset-in-warp-bits conditions; replace lane-block bijectivity
  with: `Shuffle` iff lane block bijective **or** `const_src_slot`
  holds for every destination slot. Hoist the `const_src_slot`
  computation (`codegen.rs:1135`) into a shared predicate so classifier
  and emitter agree by construction — this retires the
  `.expect("classify_convert checked the lane block")` coupling at
  `codegen.rs:1160`.
- `codegen.rs:1010` (reg→reg source layout): `inverse` →
  `right_inverse(kb)`.
- `codegen.rs:1044` (reg→shared `map`): **stays strict** — memory
  layouts remain injective (broadcasting consequence (d)).
- Everything failing the new Shuffle test still classifies `Bounce` →
  `Plan::Mirror` at inference; the reg→reg Bounce compile error stays
  unreachable.

Phase A scope cut: `gen_shuffle`'s general path keeps its strict
lane-block inverse. Replicated conversions ride the `const_src_slot`
fast path, which covers the whole butterfly/XOR-offset family including
fused_drop. The generalized-inverse path is B5.

CPU tests: `classify_convert_cases` extended with replicated composites
(projector → after A5's maps_agree ordering, never reaches classify;
replicated-with-XOR-offset → `Shuffle`; fiber-inconsistent slot →
`Bounce`); `promote_tiles` plan snapshots for a replicated producer.

**A6. Delete the divergent guard (Gap 5)** — remove
`codegen.rs:1117-1121`; fix the stale `kb >= 5` comment at `:1115`;
mask stays `0xFFFFFFFFu`. Sound only now: all 32 lanes resident (A3),
every lane's registers defined (A4).

**A7. Fallout.** Re-baseline golden CUDA (`v\d+` normalization;
sub-warp kernels now emit `block = 32` and `threadIdx.x & (T-1)`
indexing); update `classify_convert_cases`'s sub-warp `block` arguments
(no longer produced by lowering). Replicated pars now duplicate
`Shared`/`Global` stores `32/T`× — idempotent (same value, same
address), benign until Gap 7 lands in B2; leave a code comment pointing
here.

**Acceptance:** A0 tests green; frac bench fused_drop at `LOG_N = 26`
completes; full crypto-compiler suite green.

### Phase B — full general linear-layout machinery

**Goal.** Collapse `Plan::{Direct, View, Mirror}`,
`ConvertKind::{Copy, Slot, Shuffle, Bounce}`, `promote_tiles`'s
seven-precondition bailout, and `gen_convert`'s kind-pair matrix into
one lowering primitive — `ConvertLayout` — and localize all layout-
choice policy to a single site.

**Design.** Every par has a par-attr layout `f_par : phys → logical`
(register direction), user-partial or inferred (see B.1 step (0):
inference picks the par-attr that minimizes total ConvertLayout cost
across the par's reads). Every buffer carries a layout (register:
phys → logical, matching its writer's par-attr; shared/global:
logical → phys, injective; shared layouts additionally chosen to
minimize bank conflicts, see B.1 `choose_shared_layout`). A layout
or address-space mismatch between a producer and a consumer is
bridged by exactly one op — `ConvertLayout` — the sole primitive that
changes layout or space.

`ConvertLayout` carries three BufIds:

    ConvertLayout { dst: BufId, src: BufId, scratch: BufId }

Both `dst` and `scratch` are results of ordinary `SSAOpCode::Alloc`
ops emitted by `layout_infer` alongside each ConvertLayout — there is
**no** private scratch pool. `scratch` is always an `AddressSpace::
Shared` buffer; its byte size may be zero (pure reg→reg shuffle needs
no scratch, but the BufId is still present so codegen and the shared-
memory packer see one uniform interface). A *shared mirror* is
`ConvertLayout(dst=Shared, src=Register)` whose result multiple
downstream consumers read. A *non-mirror* reg→reg adaptation is
`ConvertLayout(dst=Register, src=Register)`; if the warp block of the
composite is non-identity, `scratch` is sized for a shared bounce and
codegen picks the optimal shuffle + swizzled-shared combination
(paper §5.4 + Appendix 9.2).

**Pipeline.**

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
                         → load) through op.scratch. Shared/Global
                         cases are the same primitive with different
                         factors dropping out.
```

Everything below is stated as pseudocode for these four stages, plus
per-stage notes on what's removed, what's absorbed, and where
Phase-A-vintage gap items land.

#### B.1 layout_infer (rewrite)

```
fn layout_infer(k: &mut Kernel):
  # (0) Infer par-attr for pars that don't already have one. Producers
  # first so consumers can see producer layouts when they infer.
  for par P in producer→consumer topological order:
    if P.attr.is_none():
      P.attr = infer_par_attr(P, k)

  # (1) Fill each buffer's layout from its writer.
  for par P in program order:
    for (buf B, write_map w) in P.writes:
      match B.space:
        Register:
          # check_accesses guarantees w agrees with P.attr.layout
          # (own-slot); the buffer inherits the par-attr layout.
          B.layout = P.attr.layout
        Shared | Global:
          # w is logical → phys; must be injective (broadcasting
          # consequence (d)); becomes the buffer's layout.
          B.layout = w

  # (2) For each read, insert ConvertLayout when the producer's layout
  # doesn't agree with what the consumer needs. `views_of(B)` is the
  # program-wide list of aliases produced by earlier ConvertLayouts.
  for par C in program order:
    for (buf B, read_map g) in C.reads:
      required = g.compose(&C.attr.layout)   # phys_C → logical

      # (2a) Free case — some existing register view already matches.
      # This is the maps_agree(E, L) test hoisted from Phase A, now
      # in its natural place.
      if let Some(v) = views_of(B).find(|v|
            v.space == Register && maps_agree(&v.layout, &required)):
        rewire_read_to(v)
        continue

      # (2b) Existing shared view — reuse it, compose its layout into
      # the read's index expression so the load addresses the shared
      # buffer with its own layout.
      if let Some(v) = views_of(B).find(|v| v.space == Shared):
        adjusted = v.layout.compose(&required)   # phys_C → phys_shared
        rewire_read_to(v, adjusted)
        continue

      # (2c) Insert a fresh ConvertLayout. `choose_target` is the ONE
      # policy call — the entire reg-vs-shared / mirror-vs-adapt
      # decision lives here.
      (dst_space, dst_layout) = choose_target(B, required, all_uses(B))

      # Allocate BOTH dst and scratch through the ordinary Alloc
      # machinery. Scratch is always Shared; its byte size stays 0
      # here and is filled in by B.2. Both BufIds are wired into the
      # ConvertLayout so codegen and the shared-mem packer see one
      # uniform interface — no private scratch pool.
      dst     = insert Alloc { space=dst_space, layout=dst_layout,
                               shape=B.shape, elem=B.elem }
      scratch = insert Alloc { space=Shared, layout=None,
                               shape=[0], elem=B.elem }
      src_view = pick_source_view(B, dst_space)
      insert ConvertLayout { dst, src: src_view, scratch }
             at the earliest program point dominating this read
             (after src_view's producer).
      views_of(B).push(dst)
      rewire_read_to(dst)

# ---------------------------------------------------------------
# (0) par-attr inference — minimize aggregate ConvertLayout cost
# ---------------------------------------------------------------
fn infer_par_attr(P, k):
  # Cost tiers (paper §5.4 + our conversion orientation):
  #   0  free    — composite C is a pure register permutation
  #                (warp and lane blocks both identity; only slot
  #                bits mix — in-thread rename, no code emitted)
  #   1  cheap   — warp block identity, lane block non-trivial
  #                (one or more __shfl_sync rounds, no shared)
  #   2  costly  — warp block non-identity (shared bounce required)
  #
  # For each read (B, g), the *ideal* f_par is one that makes the
  # composite C = L_B⁺ ∘ (g ∘ f_par) intra-thread — i.e. the read
  # sees producer values already in-lane. Solving g ∘ f_par = L_B:
  #
  #     f_par_candidate = g.right_inverse(logical_bits).compose(&L_B)
  #
  # Non-unique when g has slack (non-injective reads / broadcasts) —
  # we take the min-Hamming-weight solution (Gap 4 semantics), which
  # zeros irrelevant hardware bits and tends to preserve coalescing.
  candidates = { default_par_attr(P) }   # identity + zero-column
                                         # replication baseline
  for (buf B, read_map g) in P.reads:
    L_B = producer_layout_of(B)          # already assigned by step (1)
                                         # for producers preceding P in
                                         # topological order
    if let Some(f) = g.right_inverse(k.logical_bits).compose(&L_B):
      candidates.insert(canonicalize(f))

  # Score each candidate over ALL reads; the winner minimizes total
  # cost. Ties broken toward the default (avoid churning goldens).
  fn score(f):
    sum over (B, g) in P.reads:
      C = L_B.right_inverse(k.logical_bits).compose(&g.compose(&f))
      cost_tier(C)

  return ParAttr { layout: argmin(candidates, score), … }

fn cost_tier(C):
  if warp_block(C).is_identity() && lane_block(C).is_identity(): 0
  else if warp_block(C).is_identity():                            1
  else:                                                           2

# ---------------------------------------------------------------
# (2c) target selection — reg vs shared, and shared bank-conflict
#      minimization
# ---------------------------------------------------------------
fn choose_target(B, required, uses):
  # Reg-vs-shared: the only cost site. All prior Plan-selection
  # heuristics collapse here. Under this policy `promote_tiles`
  # disappears — the "promote a shared tile to registers" decision
  # is just: producer wrote to Register (via its par-attr),
  # consumers read via ConvertLayout.
  if distinct_required_layouts(uses).len() >= REUSE_THRESHOLD
     || B.is_persistent_across_kernels():
    (Shared, choose_shared_layout(required, uses))
  else:
    (Register, required)

fn choose_shared_layout(required, uses):
  # Reduces to the two-access swizzle algorithm from Triton
  # (paper §5.4 + Appendix 9.2, lib/Tools/GenericSwizzling.cpp);
  # full spec at B.6.2. For N > 2 accesses, enumerate pairs, run
  # optimal_shared_swizzle on each, and take the min-cost
  # SharedLayout across all uses. Loop-weighted per B.6.2.
  candidates = pairs(uses).map(|(a, b)|
      optimal_shared_swizzle(a.layout, b.layout, element_bytes))
    + [row_major_default()]                # safety net
  return argmin over candidates:
    Σ over acc in uses: loop_weight(acc)
                      * conflict_factor(candidate, acc.layout)
```

**What it replaces / absorbs:**

- Kills `promote_tiles` (`layout_infer.rs:150+`), `Plan::{Direct,
  View, Mirror}`, and every precondition bailout at `:208`, `:226`,
  `:234`, `:239`. `l.inverse()` (`:208`) becomes
  `l.right_inverse(kb)` inside `pick_source_view` when needed for
  domain widening, never as a rejection.
- Kills `classify_convert` (`kernel_ir.rs:186-248`) at the inference
  layer. The Copy/Slot/Shuffle/Bounce distinction moves to codegen's
  `factor` (B.4) and stops leaking into IR shape.
- Absorbs Phase-A's `maps_agree`-first Direct test naturally (2a).
- Absorbs consumer-side broadcasts (`A[i%c]`, `A[0]`, reduce results
  as all-zero-column layouts): `linearize_accesses`
  (`layout_infer.rs:90-114`) accepts non-injective linear maps; the
  resulting many-to-one `required` flows through 2a-2c unchanged and
  codegen resolves via `T⁺` (broadcasting consequence (b)). The
  `non_linear_reader_gets_shared_mirror` test flips to expect a
  reg→reg ConvertLayout, not a mirror.
- Anchors (paper §4.4): pinned-layout accesses (scatter maps, global
  coalescing constraints) enter as `required`s that `choose_target`
  respects verbatim; forward propagation is exactly the program-order
  walk above; rematerialization is a future addition (recompute cheap
  producer chains at consumer layout when cheaper than converting).

#### B.2 allocate_convert_scratch (new pass — fills in Alloc sizes)

The scratch `Alloc` ops are already present in the IR (inserted by
B.1 with a placeholder shape `[0]`). This pass just walks each
ConvertLayout, computes the required scratch bytes from the
`(src, dst)` layout pair, and updates the corresponding `Alloc`'s
shape. Everything downstream — `plan_shared_mem`'s first-fit liveness
packer, `insert_sync`'s dirty-shared walk, codegen's `__shared__`
declarations — treats the scratch buffer as any other shared
`Alloc`. No private pool, no special-case.

```
fn allocate_convert_scratch(k: &mut Kernel):
  for op in k.ops.iter() where op.opcode is ConvertLayout:
    bytes = convert_scratch_bytes(op.src, op.dst, k.block)
    if bytes > 0:
      # Resize the existing scratch Alloc in place.
      scratch_alloc = k.alloc_of(op.scratch)
      scratch_alloc.shape = [bytes / sizeof(op.src.elem)]

fn convert_scratch_bytes(src, dst, block):
  # Pure function of the layout pair. No global reasoning.
  match (src.space, dst.space):
    (Register, Register):
      # Composite C = dst.layout⁺ ∘ src.layout.
      # If C's warp component is identity, one or more shuffle
      # rounds suffice — no shared bounce, scratch stays 0.
      # Otherwise size for a full-tile bounce; codegen may
      # actually use less (see B.4: shuffle + swizzled-shared
      # combinations reduce the shared traffic below tile size),
      # but the allocation is an upper bound so packing is
      # conservative.
      C = dst.layout.right_inverse(logical_bits).compose(&src.layout)
      if warp_block(C).is_identity(): 0
      else: elems_per_thread(C) * block * sizeof(elem)
    (Register, Shared) | (Shared, Register): 0  # dst / src IS shared
    (Shared, Shared): elem_count * sizeof(elem) # via reg intermediate
    (_, Global) | (Global, _): 0                # direct load/store
```

**Notes:**

- Standalone pass because scratch sizing needs `k.block` and the
  full layout pair, but doesn't need consumer/producer graph
  reasoning — cheaper to separate from `layout_infer`'s worklist
  walk and gives codegen a fixed upper bound to plan against.
- Reuses the existing shared-alloc packing verbatim: no BufId
  creation, no new access registration.
- Determinism by construction: `convert_scratch_bytes` is pure.
- Codegen may choose a decomposition that uses less scratch than
  the upper bound (e.g. one shuffle round + a small swizzled
  exchange — B.4); the packer just sees a smaller `__shared__`
  region. Codegen must not exceed the pass's byte count.

#### B.3 insert_sync (extension)

No new pass. The existing dirty-shared walk (`insert_sync.rs`, dirty
set + loop back-edge pre-extend) consumes the ConvertLayout accesses
populated by B.2 unchanged. Two boundaries matter:

1. **Outer syncs** — between a ConvertLayout's scratch/shared writes
   and any downstream reader of the same buffer (another
   ConvertLayout, or a par read from a shared mirror). Falls out of
   the standard dirty walk; no code change beyond making sure the
   walk enumerates ConvertLayout as both a reader and a writer.
2. **Inner sync** — inside a reg→reg ConvertLayout with a bounce, the
   store → `__syncthreads` → load sequence is entirely codegen-
   internal (B.4). `insert_sync` does not emit it, because it doesn't
   see inside the op; codegen emits an unconditional `__syncthreads`
   between the two internal stages, and the outer walk's post-write
   sync coalesces with the next dirty-consumer's pre-read sync via
   the existing dedup logic.

Existing invariants preserved: Sync is a statement-level op only;
verifier rejects Sync inside pars; the dirty-walk-in-reverse two-phase
pattern remains.

#### B.4 codegen (rewrite of `gen_convert`)

```
fn gen_convert(op: &ConvertLayout, k: &Kernel) -> String:
  src, dst = op.src, op.dst
  # Receiver-driven composite: for each destination phys, name the
  # source phys. Requires src's right inverse (Gap 4).
  C = src.layout.right_inverse(logical_bits).compose(&dst.layout)
  #   C : phys_dst → phys_src, affine

  match (src.space, dst.space):

    (Register, Register):
      # Search over decomposition strategies; each is a tuple
      #   (reg_perm_src, [shuffle rounds], scratch_layout,
      #    shared_write_layout, shared_read_layout, reg_perm_dst)
      # some components may be identity/empty. Score = static cycle
      # estimate: shuffle rounds + shared_traffic_bytes + bank_
      # conflict_factor * shared_ops. Pick the min.
      #
      # Extremes of the search space:
      #   (a) Pure shuffle: warp block of C is identity ⇒ one
      #       __shfl_sync round per non-identity lane bit; scratch
      #       stays unused (BufId still present, size 0 in B.2).
      #   (b) Pure bounce: store all elements to `scratch` under a
      #       bank-conflict-minimizing swizzle, __syncthreads, load
      #       under the reader's swizzle. Uses full scratch.
      #   (c) Hybrid (our extension, builds on paper primitives —
      #       the paper §5.4 does not compose shuffle+shared for one
      #       conversion): one or two shuffle rounds handle the
      #       lane-level mixing; remaining warp-level mixing goes
      #       through a *smaller* scratch with an XOR swizzle
      #       (Appendix §9.2) picked so both the store and the load
      #       are bank-conflict-free. Beats (b) whenever any lane
      #       bits of C are cheap and the swizzle search finds a
      #       mutual optimum.
      #
      # Legality throughout: mask = 0xFFFFFFFFu (Gap 5, no
      # divergent guards); generalized sender-slot handled by
      # preimage-consistency under min-weight T⁺ (B5).
      Strategy { r_src, shuffles, scratch_layout,
                 sh_write, sh_read, r_dst } =
          best_decomposition(C, src.layout, dst.layout,
                             op.scratch, k.block)

      emit register permutation r_src
      for round in shuffles:
        emit __shfl_sync(0xFFFFFFFFu, ..., round.src_lane_expr)
      if !shuffles_alone:
        emit stores to `op.scratch` under sh_write
              (with Gap-7 replica guard where applicable)
        emit "__syncthreads();"
        emit loads from `op.scratch` under sh_read
      emit register permutation r_dst

    (Register, Shared):
      # This IS the reg→shared store path today.
      # Emitted per src slot with the Gap-7 replica guard
      #   π_thr(x ^ (T⁺∘T)(x)) == 0
      # composed with the non-pow2 tail mask (paper §8: pad and mask,
      # store-side only).

    (Shared, Register):
      # Ordinary shared load, materialized here as a ConvertLayout
      # for uniformity with the rest of the pipeline. Per-slot:
      #   dst[slot] = src[dst.layout ∘ src.layout⁻¹(slot idx)]

    (Shared, Shared) | (Global, _) | (_, Global):
      # Realize via a register intermediate under two ConvertLayouts
      # (canonical form emitted by layout_infer). Direct memcpy only
      # if the composite reduces to the identity permutation.
```

**Absorbs, verbatim:**

- **Gap 6** (literal `blockDim`): the mechanical substitution happens
  in this rewrite — `k.block` is a compile-time constant at every
  emission site.
- **Gap 7** (replica store guard): only the Register → Shared arm
  emits under the guard. Formulated generically as
  `π_thr(x ^ P(x)) == 0` where `π_thr` is the phys-to-thread
  projection and `P = T⁺∘T`; under today's `x = slot * blockDim +
  threadIdx.x` encoding the mask is `k.block - 1u`; a user-permuted
  par-attr just relocates the projector, keeps the form. Register
  writes stay unguarded (replicas must stay defined for downstream
  shuffles).
- **Gap 8** (general `map`, propagation, kind-pair coverage,
  multi-stage): this arm-by-arm rewrite IS the fix. Shared→Register
  and Shared→Shared are no longer "unsupported" (`codegen.rs:1068-
  1073` deleted).
- **B5 sender-slot** in `best_decomposition`: generalized lane-block
  inverse (`MXM = M`); legality = source-slot function constant on
  each preimage of the lane map; guaranteed when `C` is built through
  min-weight `T⁺` (Gap 4).
- **Non-pow2 shapes** (paper §8): store-side mask on the Register →
  Shared arm, composed with the replica guard. No new machinery.
- **Bank-conflict minimization at emission** (paper Appendix 9.2):
  `best_decomposition`'s scratch layout is not required to match the
  B.1-chosen shared layout — B.1 optimizes the *persistent* shared
  buffer for its long-lived readers/writers, whereas this transient
  scratch is single-op and can pick an XOR-swizzle tailored to just
  the two access maps that touch it. The search may find that a
  small shared exchange under a bespoke swizzle plus one shuffle
  round beats either extreme; the scratch upper bound from B.2
  bounds it from above so no re-planning is needed.

#### B.6 The three key algorithms in depth

Three algorithms carry the entire Phase B design: par-attr inference
(B.1 step 0), shared layout selection (B.1 `choose_shared_layout`),
and ConvertLayout decomposition (B.4 `best_decomposition`). Each is
sketched inline above; this section states them precisely.

##### B.6.1 Par-attr inference

**Inputs.** A par `P` with reads `{(B_i, g_i)}`. Producer layouts
`L_{B_i}` are already fixed by topological order (producers precede
consumers in the walk).

**Cost model.** Each read `(B, g)` induces a demand
`E = g ∘ f_par : phys_P → logical_B`; the receiver-driven conversion
map is `C = L_B⁺ ∘ E : phys_P → phys_B` (Gap 4's right inverse
handles the surjective `L_B`). Decompose C's linear part over the
physical bit basis `(slot, lane, warp)` and classify:

| Tier | Condition on C | Emission | Cost |
|---|---|---|---|
| 0 (free)   | warp block identity **and** lane block identity | per-thread register rename; slot bits mix in-place | 0 |
| 1 (cheap)  | warp block identity, lane block non-trivial      | `__shfl_sync` round(s)                              | ~1 cycle / non-identity lane bit |
| 2 (costly) | warp block non-identity                          | store → sync → load through shared scratch          | 10s of cycles |

**Search space.** All distributed `f_par` is combinatorial; the key
reduction is that a read is *free* iff `g ∘ f_par ≡ L_B` as
functions, which is a **linear equation in `f_par`** with solution

    f_ideal(B, g) := g.right_inverse(logical_bits).compose(&L_B)

(min-Hamming-weight when `g` has slack — Gap 4 semantics; non-uniqueness
is resolved by zeroing slack bits so the resulting `f_par` stays
distributed).

**Algorithm.**

```
candidates = { default_par_attr(P) }
for each read (B, g):
  if let Some(f) = g.right_inverse(logical_bits).compose(&L_B):
    if is_distributed(f):                # Def 4.10: ≤1 bit per
                                         # column, distinct non-zero
      candidates.insert(canonicalize(f))

fn score(f) =
  Σ over reads (B_i, g_i):
    weight_i * cost_tier(
      L_{B_i}.right_inverse(logical_bits)
             .compose(&g_i.compose(&f)))

return argmin(candidates, score)
```

`cost_tier` inspects C's block decomposition (warp block identity?
lane block identity?) via 𝔽₂ Gaussian elimination on the appropriate
sub-matrices.

**Near-optimality argument.** The candidate set enumerates every
`f_par` that makes *some* read free. Any `f_par` outside the set
produces tier-1-or-2 on every read. An `f_par` with no free reads
can beat the best candidate only when it makes multiple reads tier-1
simultaneously in a way none of the free-per-read candidates do —
formally possible, empirically rare because tier-1 requires warp
block identity, and any `f_par` achieving it for read 2 without being
`g_2⁺ ∘ L_{B_2}` typically also fails read 1.

**Weights.** `weight_i` = product of enclosing loop iteration counts.
Symbolic loop bounds contribute `SYMBOLIC_WEIGHT ≈ 2^30` so any
bounded-but-symbolic loop dominates known-small constants — this
biases the optimizer toward cheap conversions on symbolic-hot paths
even without concrete iteration counts. Shared across all three
algorithms below (B.6.2, B.6.3); the same loop-weight function
computes weights everywhere.

**Distributedness filter.** `g⁺ ∘ L_B` may not be a legal par-attr
(two phys points to one logical, violating own-slot). Non-distributed
candidates are discarded before scoring.

**Tie-breaking.** Toward `default_par_attr(P)`, measured by Hamming
distance in the bases — minimizes golden churn.

**Failure mode.** All-tier-2 outcome is legal; the algorithm's job is
minimization, not free-guarantee.

##### B.6.2 Shared layout selection (`choose_shared_layout`)

**Reference implementation.** Triton's
`lib/Tools/GenericSwizzling.cpp` (paper §5.4 + Appendix 9.2). The
sketch below matches that implementation; ours differs only in the
codomain-carrying `LinearLayout` type and the multi-access
generalization (last section).

**Data types.** All bases are `Vec<BitVec>` over 𝔽₂. Each `BitVec`
is `output_dim` bits wide; a basis is a list of such vectors.

```rust
type BitVec = Vec<bool>;

struct Basis { vecs: Vec<BitVec> }

struct LinearLayout {           // hardware → address
    reg: Basis,                 // columns for register-slot input bits
    thr: Basis,                 // columns for thread input bits (lane ∪ warp)
    output_dim: usize,          // = log2(shared buffer size in elements)
}

struct SharedLayout {           // decomposition of the shared address space
    vec:  Basis,                // vectorization bits (constant within a warp
                                //   transaction — one vector load/store)
    bank: Basis,                // bank-selecting bits (must vary across lanes
                                //   for conflict-free access)
    idx:  Basis,                // transaction-index bits (ideally constant
                                //   across lanes so all lanes hit one txn)
}
```

For our codebase: to feed a `crate::LinearLayout` into this algorithm,
project its columns by input-bit origin — columns indexed by slot
input bits become `reg`; columns indexed by lane+warp bits become
`thr`; `output_dim = log2(buffer_elems)`. The affine offset is
irrelevant to bank-conflict analysis (it shifts every address by a
constant); it survives untouched into codegen.

**Bank memory model.** Shared memory serves a fixed **128-byte
transaction** per warp instruction (32 lanes × 4-byte word = 128 B).
Bank-conflict-free iff, within one transaction, all lanes address the
same 128 B window (idx constant across the warp) and 32 distinct
banks (bank bits cover the lane variation).

**Setup.**

    vec       = A.reg  ∩  B.reg              # maximal common vectorization
    v         = |vec|
    bank_bits = log2(128 / (2^v · element_bytes))
    idx_bits  = output_dim − v − bank_bits

The common register basis is the largest set of address bits that
both A and B keep constant per thread — i.e., the widest vector both
accesses can load/store. Given that, the 128 B transaction budgets
`bank_bits` bits for bank selection.

**Dangerous subspaces.** Bits that vary within a warp instruction:

    U_A = span(vec ∪ A.thr)
    U_B = span(vec ∪ B.thr)

If any `idx` bit lands inside `U_A`, then within A's warp the
transaction-index changes across lanes — different lanes hit different
128 B windows and compete for banks. Symmetric for `U_B`. Conflict-
free access requires `idx ∩ (U_A ∪ U_B) = 0`.

**Common / exclusive split.**

    common = U_A ∩ U_B
    E_A    = complement(common, U_A)         # in U_A but not U_B
    E_B    = complement(common, U_B)         # in U_B but not U_A

**Safe directions.**

Two sources of directions outside both `U_A` and `U_B`:

1. **Global complement:** `C = complement(U_A + U_B, ambient)` —
   dimensions that appear in neither dangerous subspace.
2. **Paired XORs:** for `i in 0..min(|E_A|, |E_B|)`,
   `g_i = E_A[i] XOR E_B[i]`. Each `g_i` lies outside both `U_A` and
   `U_B`.

   *Proof.* If `g_i ∈ U_A`, then `E_B[i] = E_A[i] XOR g_i ∈ U_A`
   (both terms in `U_A`), but `E_B[i]` is by construction in
   `U_B \ common`, i.e. in `U_B` but not `U_A`. Contradiction.
   Symmetric for `U_B`. ∎

The paired-XOR trick is the reason a two-access swizzle can be
conflict-free where a single-access one can't: exclusive-to-A and
exclusive-to-B directions cancel each other out.

    safe = C ∪ { g_i }

**Idx selection.** Prefer `idx` bits from `safe`:

- If `|safe| ≥ idx_bits`: **fully bank-conflict-free**. Take any
  `idx_bits` of them.
- Otherwise: some conflicts are mathematically unavoidable. Fill
  `safe` first, then extend with directions from `U_A` (chosen to
  increase basis rank). The choice of `U_A` over `U_B` is arbitrary
  and can be flipped per pair to pick whichever gives fewer conflicts.

**Bank selection.** Everything not in `vec` or `idx`:

    bank = complement(vec ∪ idx, ambient)

Guaranteed `|bank| = bank_bits` by dimension counting.

**Algorithm.**

```rust
fn optimal_shared_swizzle(
    a: &LinearLayout,
    b: &LinearLayout,
    element_bytes: usize,
) -> SharedLayout {
    let d       = a.output_dim;
    let ambient = standard_basis(d);

    // 1. Maximal common vectorization.
    let vec       = intersection(&span(&a.reg), &span(&b.reg));
    let v         = vec.vecs.len();
    let bank_bits = exact_log2(128 / ((1 << v) * element_bytes));
    let idx_bits  = d - v - bank_bits;

    // 2. Dangerous subspaces.
    let ua = union_basis(&[&vec, &a.thr]);
    let ub = union_basis(&[&vec, &b.thr]);

    // 3. Common / exclusive.
    let common = intersection(&ua, &ub);
    let ea = complement_basis(&common, &ua);
    let eb = complement_basis(&common, &ub);

    // 4. Safe directions.
    let c = complement_basis(&sum_subspaces(&ua, &ub), &ambient);
    let g: Basis = (0..ea.vecs.len().min(eb.vecs.len()))
        .map(|i| xor(&ea.vecs[i], &eb.vecs[i]))
        .collect();
    let safe = union_basis(&[&c, &g]);

    // 5. Idx bits: safe first, dangerous fallback.
    let idx = if safe.vecs.len() >= idx_bits {
        take_independent(&safe, idx_bits)
    } else {
        let remaining = idx_bits - safe.vecs.len();
        let current   = union_basis(&[&vec, &safe]);
        let cands     = complement_basis(&intersection(&current, &ua), &ua);
        let extra     = extend_with_candidates(&current, &cands, remaining);
        union_basis(&[&safe, &extra])
    };

    // 6. Bank bits: complement of (vec ∪ idx).
    let bank = complement_basis(&union_basis(&[&vec, &idx]), &ambient);

    SharedLayout { vec, bank, idx }
}
```

Helpers (`span`, `intersection`, `sum_subspaces`, `complement_basis`,
`union_basis`, `extend_with_candidates`, `xor`) are standard 𝔽₂
linear-algebra primitives — reduce to Gaussian elimination on
bit-matrices.

**Multi-access generalization (v1 heuristic).** For N accesses (M
writers + K readers, N > 2), enumerate all `C(N, 2)` pairs, run
`optimal_shared_swizzle` on each, and pick the resulting `SharedLayout`
that minimizes total weighted conflict across *all* N accesses:

```rust
fn choose_shared_layout(accesses: &[Access], element_bytes: usize)
    -> SharedLayout
{
    let candidates = accesses.iter().tuple_combinations()
        .map(|(a, b)| optimal_shared_swizzle(&a.layout, &b.layout,
                                             element_bytes))
        .chain(once(row_major_default()));  // safety net

    fn cost(sh: &SharedLayout, accesses: &[Access]) -> u64 {
        accesses.iter()
            .map(|acc| loop_weight(acc) *
                       conflict_factor(sh, &acc.layout))
            .sum()
    }

    candidates.min_by_key(|sh| cost(sh, accesses)).unwrap()
}
```

The optimality guarantee of `optimal_shared_swizzle` is per-pair; the
pairwise-min heuristic doesn't guarantee a global optimum for N > 2,
but empirically dominates the identity and single-swizzle baselines
because at least one pair covers the two hottest accesses. If cost
model complexity grows (multi-way XOR pairings, higher-order safe
subspaces), replace with a proper N-way solver — the algorithm
generalizes, we just haven't implemented it.

**Loop weighting.**
`loop_weight(access) = product of enclosing loop iteration counts`
(concrete when known; `SYMBOLIC_WEIGHT ≈ 2^30` per unknown-bound
loop). Same convention as B.6.1 and B.6.3.

**Conflict factor.** `conflict_factor(sh, layout)` counts residual
conflicts of `layout` against `sh`: the rank deficiency of the
lane-bits → bank-bits map after projecting through `sh`'s
partition. Bank-conflict-free ⇒ 0; k-way conflict ⇒ `k − 1`.

**Element width.** `element_bytes` handles the extension-field case
directly (BabyBear = 4, ext = 16, etc.); the 128-byte transaction
budget adjusts `bank_bits` automatically without algorithm changes.

**Padding as alternative.** Row-padding is available as a fallback
when the swizzle can't achieve conflict-free access AND shared memory
budget is tight (padding wastes shared but is simpler to reason
about under symbolic sizes). Not v1.

##### B.6.3 ConvertLayout decomposition (`best_decomposition`)

**Inputs.** A ConvertLayout with `C = src.layout⁺ ∘ dst.layout :
phys_dst → phys_src`, a pre-allocated `scratch` BufId sized by B.2,
and `k.block`.

**Primitives.**

| Primitive | Cost | Constraint |
|---|---|---|
| Per-thread register rename | 0            | slot bits only |
| `__shfl_sync` round        | ~1 cycle     | full mask; sender-slot constant on preimages |
| Shared store               | 1 × conflict | swizzle picked here |
| `__syncthreads`            | fixed        | block-wide |
| Shared load                | 1 × conflict | swizzle picked here |

**Structural pipeline.** Every ConvertLayout factors as some prefix
of

    src regs → rename → shuffle₁ → shared store → sync
             → shared load → shuffle₂ → rename → dst regs

with stages emitted only when the corresponding block of C is
non-trivial.

**Block decomposition of C (our exposition).** The paper works with
column projections `(C)_Reg`, `(C)_Thr`, `(C)_Wrp` — the columns of
the composite acting on each labeled input space. For talking through
strategies we find it clearer to write C's linear part as the 3×3
block matrix in the (slot, lane, warp) basis with rows = outputs,
columns = inputs:

               →slot  →lane  →warp
       slot→ [ C_ss   C_sl   C_sw ]
       lane→ [ C_ls   C_ll   C_lw ]
       warp→ [ C_ws   C_wl   C_ww ]

The paper's `(C)_Wrp` = the middle-column block `(C_sw, C_lw, C_ww)`
stacked vertically; `(C)_Wrp = I` in the paper's sense means
`C_ww = I` (warp→warp identity) and warp inputs produce no output
in slot or lane positions (`C_ws = 0`, `C_wl = 0`).

Per-block emission:

- `C_ss`: free rename.
- `C_ll`: warp shuffle (round count derived below).
- `C_ww`: needs shared (warp-crossing).
- `C_sl`, `C_ls`: shuffle + rename combo.
- Any warp-touching cross-term (`C_sw`, `C_ws`, `C_lw`, `C_wl`): shared.

**Strategies.**

- **A (pure shuffle)** — paper §5.4 Intra-warp Data Exchange:
  applicable iff `(C)_Wrp = I` (page 8: *"If `(B⁻¹∘A)_Wrp` is the
  identity, data exchange can be performed using warp shuffles"*).
  In our block form this is
  `C_ww = I ∧ C_ws = 0 ∧ C_wl = 0` — the warp-input column stays
  identity. Def 4.10 does the rest: since our whole pipeline
  maintains distributed layouts (columns with ≤1 non-zero bit,
  distinct non-zero columns — enforced by min-weight `T⁺` and
  Gap 4's `is_distributed` check), any slot/lane input column
  whose non-zero bit landed in a warp output row would collide
  with one of the identity warp columns, violating distinctness.
  So `C_sw = 0 ∧ C_lw = 0` fall out automatically — the paper's
  one condition is sufficient. Emit rename + shuffle round(s) +
  rename. Zero shared traffic.

  Defense-in-depth: `debug_assert!` the two implied cross-terms
  are zero. If a future non-distributed ConvertLayout (e.g., a
  hand-authored anchor) violates the invariant, we want a loud
  failure rather than silent data corruption.
- **B (pure bounce)** — paper §5.4 Optimal Swizzling: always
  applicable. Store all elements to scratch under `sh_write`; sync;
  load under `sh_read`. Swizzles picked by B.6.2's
  `optimal_shared_swizzle` specialized to the two accesses — which
  is *exactly* the paper's Appendix §9.2 algorithm applied to the
  scratch buffer.
- **C (hybrid) — our extension.** The paper does not describe
  combining a partial shuffle stage with a shared bounce for one
  conversion; §5.4's three components (intra-thread rename, intra-
  warp shuffle, swizzled shared) are presented as independent
  primitives, not composed. We propose: absorb a subset `T` of
  lane bits into a pre-store and/or post-load shuffle so the shared
  bounce handles only `C_ww` plus residual cross-terms. Search
  space `2^5 × {Pre, Post, Both}` — bounded, exhaustive enumeration
  is fine.

**Shuffle round count (paper §5.4, page 8).** Given `A = src.layout`
and `B = dst.layout`, define:

    V = A.Reg ∩ B.Reg              # maximal common vectorization
    I = A.Thr ∩ B.Thr              # thread bits requiring no exchange
    E = A.Thr \ I                  # thread bits exclusive to A
    F = B.Thr \ I                  # thread bits exclusive to B
    G = { e_i ⊕ f_i | 1 ≤ i ≤ |E| }   # paired exchange bits
    R = basis extension of (V ∪ I ∪ G) to F₂^d   # d = shuffle output dim

Then **rounds = `2^|R|`** ("*we can exchange the elements in `2^|R|`
rounds, shuffling the elements in each round*"). Each round moves
`2^|V|` elements per thread (one vectorized shuffle op), and every
thread sends+receives one element (page 8: *"In each round, every
thread sends and receives only one element"*). All lanes participate
under full mask — an *implication* of the construction, not a paper
quotation.

Note V, I, E, F, G reappear in B.6.2 (our shared-swizzle algorithm
is the paper's Appendix §9.2 applied to the shared/scratch buffer
using the same primitives) — the `G = {e_i ⊕ f_i}` construction is
used by both.

**Algorithm.**

```
strategies = []

# Strategy A — paper §5.4 Intra-warp Data Exchange.
# Paper's condition: (C)_Wrp = I. Under Def 4.10 distributedness
# this implies the two output-row cross-terms automatically.
if C_ww == I and C_ws == 0 and C_wl == 0:    # paper's (C)_Wrp = I
  debug_assert!(C_sw == 0 and C_lw == 0)     # implied by Def 4.10
  # Factor as three stages: rename → shuffle → rename. Paper only
  # gives (C)_Reg as a single register permutation observation
  # (page 8, Intra-thread Data Exchange); the R_dst ∘ S ∘ R_src
  # sandwich is our derivation — the pre-shuffle rename maximizes
  # V = A.Reg ∩ B.Reg for the round formula, the post-shuffle
  # rename fixes up (C)_Reg on the output side.
  (r_src, S, r_dst) = factor_pure_shuffle(C)
  # Round count from paper: 2^|R| where R extends V ∪ I ∪ G to a
  # basis of F₂^d.
  rounds = 1 << basis_extension_rank(V, I, G, d)
  strategies.push(("shuffle",
                   rounds * SHUFFLE_COST,
                   { r_src, S, r_dst }))

# Strategy B
(sh_write, sh_read, scratch_swizzle) =
    pick_bounce_swizzles(src.layout, dst.layout, block)
strategies.push(("bounce",
                 tile_bytes * SHARED_COST *
                     conflict_factor(sh_write, sh_read, scratch_swizzle)
                 + SYNC_COST,
                 { sh_write, sh_read, scratch_swizzle }))

# Strategy C
for T in lane_bit_subsets():                   # 32 subsets
  for placement in {Pre, Post, Both}:
    (r_src, S_pre, sh_write, sh_read, S_post, r_dst, scratch_swizzle) =
        factor_hybrid(C, T, placement, block)
    strategies.push(("hybrid_" + T + "_" + placement,
                     (S_pre.rounds() + S_post.rounds()) * SHUFFLE_COST
                     + residual_bytes(C, T) * SHARED_COST *
                         conflict_factor(sh_write, sh_read, scratch_swizzle)
                     + SYNC_COST,
                     { ... }))

return argmin(strategies, cost)
```

**`pick_bounce_swizzles`.** Delegates to B.6.2's
`optimal_shared_swizzle(src.layout, dst.layout, element_bytes)` —
this is the paper's Appendix §9.2 algorithm applied to the two-access
scratch, its canonical use case. Returns a `SharedLayout` with
`vec`/`bank`/`idx` decomposition; the scratch write uses
`src.layout ∘ M` and the read uses `dst.layout ∘ M` where `M`
reconstructs the shared address from the decomposition. Bank-
conflict-free when `|safe| ≥ idx_bits` in B.6.2's terms; residual
conflicts enter the strategy's cost otherwise.

**`factor_hybrid`.** For each `T` and placement, algebraically split
C's mixing: bits in `T` absorbed into shuffle stages (as one or two
`__shfl_sync` rounds), the residual `C / T` handled by the shared
bounce. Solve for the scratch swizzles that keep the residual
conflict-free.

**Cost constants.** Approximately, on TU104: `SHUFFLE_COST ≈ 4`,
`SHARED_COST ≈ 4 cycles/bank/warp`, `SYNC_COST ≈ 10–100` (divergence-
dependent). Kept tunable — the boundary cases (3 shuffle rounds vs.
one shared exchange) are sensitive to these.

**Loop weighting.** Every cost above is multiplied by
`loop_weight(op) = product of enclosing loop iteration counts`
(concrete when known; `SYMBOLIC_WEIGHT ≈ 2^30` per symbolic-bound
loop — same convention as B.6.1 and B.6.2). Within `best_decomposition`
the weight is a constant multiplier that doesn't change the argmin,
but the *scaled* cost matters upstream: `choose_target` in B.1 uses
`best_decomposition`'s cost to decide reg-vs-shared mirror, and a
ConvertLayout inside a hot loop is exactly the case that should tip
the choice toward the amortized shared mirror.

**Bank-conflict scoring within decomposition.** `conflict_factor` in
each strategy is computed by feeding the strategy's `sh_write` and
`sh_read` layouts (paired with `op.scratch`'s decomposition) into
B.6.2's `optimal_shared_swizzle` as the 2-access case — this is the
canonical two-access instance, so it applies cleanly and produces the
bank-conflict-free swizzle when one exists. When it doesn't, the
residual conflict count enters the strategy's cost. `pick_bounce_swizzles`
in Strategy B and `factor_hybrid`'s residual-solve in Strategy C both
delegate to `optimal_shared_swizzle`; no separate bank-conflict logic
lives in codegen.

**Legality invariants preserved by every strategy.**

- `__shfl_sync` uses `0xFFFFFFFFu`, no divergent guard (Gap 5).
- Sender-slot function constant on preimages of the lane map (B5);
  guaranteed under min-weight `T⁺`.
- Scratch bytes emitted ≤ B.2's upper bound. Every hybrid strategy
  uses ≤ pure-bounce bytes, so B.2's Alloc always fits.
- Replica store guard (Gap 7) wraps any shared store when `src`'s
  par-attr is replicated: `π_thr(x ^ P(x)) == 0`.

**Register-pressure escape.** Pure bounce holds `elems_per_thread`
values briefly (between prefix shuffle and store); pure shuffle holds
them through all rounds. Under heavy live-set pressure, shorter
register lifetime wins independent of raw cycles. Cost model
extension: penalize peak register-lifetime × count.

**Landing pragma.** For an initial implementation, restrict Strategy
C to `T ∈ {∅, all-lane-bits} × {Pre}` — this covers pure shuffle,
pure bounce, and one middle ground. Already strictly better than the
current pipeline (where Bounce is a compile error). Full hybrid
enumeration lands as an optimization once the primary rewrite proves
correct.

#### B.7 Order and landing sequence

Landings are independent as long as B.1's rewrite ships alongside a
codegen `gen_convert` that at least matches Phase A's capability set;
otherwise the plan enum removal breaks working kernels.

1. **B.4-partial**: rewrite `gen_convert` behind a feature flag,
   passing today's `Plan::View` conversions through it. Verify golden
   parity.
2. **B.2 + B.3**: land the scratch pass and confirm `insert_sync`
   picks up the new accesses. No IR-shape change to inference yet.
3. **B.1**: switch `layout_infer` to the ConvertLayout-only model.
   `promote_tiles` deletion, Plan enum deletion, and consumer-fold
   broadcasting all land together (they're the same rewrite).
4. **B.4-full**: expand `factor_reg_to_reg` to the multi-stage
   decomposition; delete `classify_convert` and the Bounce error
   path.
5. **Codomain-aware `LinearLayout` type** (optional cleanup): add
   `out_bits` to the type so `right_inverse` no longer takes it as a
   parameter, and expose `is_distributed()` (Def 4.10 check: ≤1 bit
   per column, distinct non-zero columns) as a method. Purely
   internal; no callers change semantics.

Acceptance: `Plan`, `ConvertKind`, `promote_tiles`, and
`classify_convert` are gone; consumer folds (`A[i%c]`) route through
reg→reg conversions without shared mirrors; reduce-result consumption
compiles to an all-zero-column source with `C` the zero map (free
local read); the fused_drop chain and every current golden retain
byte-parity up to `v\d+` normalization.
