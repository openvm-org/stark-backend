
A layout is something that describes how a particular buffer is represented on device. For a buffer of type `T` and shape `[N]`, we need to know if it's represented as shared memory, as the collection of a bunch of registers, or global memory.

If we decide it's represented as a bunch of registers, the descriptor need to map physical hardware indices `(spatial_id, lane_id, warp_id) -> logical_id` to the logical index (a lane is `threadIdx.x % WARP_SIZE` and a warp is `threadIdx.x / WARP_SIZE`, on Nvidia hardware `WARP_SIZE=32` and on AMD, `WARP_SIZE=64`; `spatial_id` represents multiple elements on the same thread).

If we decide it's represented as shared memory, then we need to decide the physical to logical mapping `(vectorized, bank, outer) -> logical_offset`. GPU shared memory is organized in banks of 32 bits. During execution, each bank can only service one lane per access, so if multiple lanes within the same warp access the same bank, the access becomes serialized.

The choice of the family of access functions is entirely arbitrary, I use [Linear Layouts](https://arxiv.org/pdf/2505.23819) from the paper, because it's simple enough to optimize and expressive enough for most GPU stuff.

The access function is the family of linear functions over $\mathbb Z_2$, where indices are represented as bit vectors. This means that all shapes has to be powers of 2.

I extend the paper slightly to allow for affine functions over $\mathbb Z_2$. And all results from the paper should generalize (affine functions are useful when we want to slice buffers).

For example, say the warp size is 2 and we have 2 warps. A particular encoding of a `u32[8]` buffer on registers is `(spatial_id, lane_id, warp_id) -> spatial_id + 2 * lane_id + 4 * warp_id`. This represents the buffer represented as `[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]`, where the first logical element is put on register 0 of lane 0 and warp 0, and the second is put on register 1 of lane 0 and warp 0, and so on...

The linear map of this corresponds to $$\begin{bmatrix}
& \text{spatial} & \text{lane} & \text{warp}\\
\text{logical least significant bit} & 1 & 0 & 0 \\ &0 & 1 & 0 \\ \text{logical most significant bit} &0 & 0 & 1
\end{bmatrix}$$
here I'm assuming that the linear map is $\mathbb Z_2^{\log_2(N_\text{spatial})} \times \mathbb Z_2^{\log_2(N_\text{lane})} \times \mathbb Z_2^{\log_2(N_\text{warp})} \to \mathbb Z_2^{\log_2(N_\text{logical})}$.

## Mapping to compilation 

The input to the layout inference pass is given an SSA-form IR that essentially consists of a sequence of parallel loops (denoted `par`) and abstract buffers. Each `par` has a sequence of reads and write accesses as well as a concrete iteration bound and loop inductive variable. Each `par` can be thought as `par(i, N, reads=[A_1[f_1(i,...),...], writes=[B_1[g_1(i,...)]])`. Where `N` is the concrete iteration bound, `i` is the induction variable. 

This sequence of `par` ops has the property that every buffer is written to exactly once by 1 `par` op, and every buf has concrete bounds that's a power of 2. 

The job of the layout infer pass is to assign each buffer a concrete layout that's either `Reg LinearLayout` or `Shared LinearLayout`, inserting `ConvertLayout` op as necessary that converts one layout to another. As well as determine the each `par` op's `par-attr`, which is the map from `(spatial_id, lane_id, warp_id) -> logical_id`, that determines how a generic `par` parallel loop is scheduled on the device. The logical id corresponds to the logical loop induction var `i`. This is similar to the register layout except this time `spatial_id` represents the loop iterations performed sequentially. For example, a  `par [8] |i| e(i)` with `par-attr = (spatial_id, lane_id, warp_id) -> spatial_id + 2 * lane_id + 4 * warp_id` would be scheduled like the following:
```
for spatial_id in 0..2 {
	e(spatial_id + 2 * lane_id + 4 * warp_id)
}
```


The algorithm for layout inference is as follows:

```
for each source par_i (par ops that reads from global buffers not produced by any other par), with concrete bound N_i, the `par-attr` is (s, l, w) -> S * s + w * WARP_SIZE + l. Where S = WARP_SIZE * NUM_WARPS, and the spatial repetition is N_i / S. 

deduce the layouts of the written to buffers once the par-attr is known (find_output_layout).

walk the list of pars in topological order. For each par, assume that the read buffers have known layouts. 

Then, call find_opt_par, to find the optimal par-attr given the read expressions. 
Once the par-attr for the current par op is known, call insert_convert_layout for each read from buffer A to convert the layout of the buffer to the right one:
Let P = (spatial, lane, warp) -> logical_idx be the par-attr.
let R be the layout of the buffer A.
let f be the read function. Then here are the cases:
- f is representable as an affine function.
  then required layout is Reg f ∘ P
- f is not an affine function
  then the required layout is Shared id
Then if the required layout is not equal to R, then we insert ConvertLayout(A, R, <required_layout>)

call find_output_layout to infer written buffers.

```

*note: if f is not an affine function then determining the required layout on registers may not be possible, because the analysis is either impossible if f is data-dependent or intractable for general f, since we need to optimize the convert layout*

There are three core pieces to this algorithm:
- `optimal_convert_layout_codegen`: the optimal codegen that uses a combination of warp shuffles and shared shuffles to convert one layout to another
- `find_output_layout`: deduce the output layout given `par-attr` and write expressions
- `find_opt_par`: deduce the optimal `par-attr` that minimizes convert layout cost

For `find_output_layout`, the reasoning is similar to the cases for reads. Let `P` be the determined `par-attr` and let `g` be the write expression for `B`. If `g` is affine then `g ∘ P` is a linear layout and the resulting layout is `Reg g ∘ P`, otherwise it's `Shared id`.

For `find_opt_par`, we assume that all reads expressions are convertible to affine maps, otherwise the default is `(s, l, w) -> S * s + w * WARP_SIZE + l` (threads are contiguous). For each read, `g : logical -> logical`, we get the producer layout either `Shared f` or `Reg f`. 

First initialize a list of potential `par-attr` layouts `L`

- If it's shared, find the optimal swizzled access `par-attr`, `P : (s, l, w) -> i` such that `g ∘ P` minimizes the band conflicts reading `Shared f`. Add `P` to list `L`
- If it's reg, the `par-attr` `P` which doesn't require a layout transformation is `g ∘ f`, so `P = g ∘ f`, and add `P` to `L`

For each `P in L`, compute the cost of picking `P`, which is the cost of all the convert-layouts or bank-conflicts that is necessary to commit this `par-attr` to `P`. Pick the minimum `P` such that this cost is minimized.

## optimal_convert_layout_codegen

There are three cases:
1. `ConvertLayout(Reg f, Reg g)`
2. `ConvertLayout(Shared f, Reg g)`
3. `ConvertLayout(Reg f, Shared g)`

In every case:
- `f` is the *source* buffer's layout,
- `g` is the *destination* buffer's layout,
- The `ConvertLayout` op carries a `map : logical_dst → logical_src` reindexing (identity for the "same logical space" case `layout_infer` emits today).

### Case 1: `ConvertLayout(Reg f, Reg g)`

Both buffers live in the CTA's phys space `(spatial, lane, warp)`. I use the *receiver-driven* composite

```
C = f⁺ ∘ (map ∘ g) : phys_dst → phys_src
```

i.e. "for each destination slot on each thread, which source phys position provides its data". I use this receiver-driven direction because on the GPU the receiving thread specifies which lane to pull from, so it maps directly to how `__shfl_sync` is emitted.

`f⁺` is `LinearLayout::right_inverse` — the min-Hamming-weight preimage (see paper for more details). C can be decomposed as:


```
                →slot  →lane  →warp
    slot in→ [ C_ss   C_sl   C_sw ]
    lane in→ [ C_ls   C_ll   C_lw ]
    warp in→ [ C_ws   C_wl   C_ww ]
```


#### 1.a Copy (`Strategy::Copy`, cost 0)

Free case: `f` and `map ∘ g` agree as functions (`maps_agree`). Every dst slot aliases its src slot with no address changes and I emit one `dst_reg[i] = src_reg[i]` per slot. Falls out of e.g. two consecutive pars with equal par-attr and no reindexing.

#### 1.b Slot (`Strategy::Slot`, cost 0)

In-thread register rename — no cross-thread traffic:

```
C_ll = I,  C_ww = I     (thread inputs stay on their own thread outputs)
C_ls = C_lw = 0
C_ws = C_wl = 0
C_sl = C_sw = 0         (slot inputs contribute no thread output)
```

Under this shape:
- No data crosses threads — each dst slot on each thread pulls from a compile-time-constant src slot on the *same* thread.
- The slot block `C_ss` may still permute or XOR-mix slot bits.

Emit one `dst_reg[i] = src_reg[from]` per dst slot, where `from = C(i << tb) >> tb` is a compile-time constant.

#### 1.c Shuffle (`Strategy::Shuffle { rounds }`, paper §5.4 Intra-warp Data Exchange)

Applicable iff C's warp column is identity — `LinearLayout::is_warp_column_identity`:

```
C_ww = I  ∧  C_sw = 0  ∧  C_lw = 0        (paper's (C)_Wrp = I)
C_ws = 0  ∧  C_wl = 0                     (nothing contributes to warp outputs from non-warp inputs)
```

Under this condition warps don't cross — every element that needs to move stays inside its warp — so the exchange lowers to `__shfl_sync(0xffffffffu, …)` with a full mask. The rough shape is a register rename → warp shuffle → register rename sandwich, but the number of shuffles emitted depends on C.

`shuffle_rounds(C, block)` — mirrored by `gen_shuffle`'s three code paths — is:

- **Constant sender slot** (`const_src_slot`). No lane- or warp-input column of C has bits in slot positions, so `(C(s'<<tb ^ tid)) >> tb = C(s'<<tb) >> tb` is a compile-time constant — every receiver on every lane pulls from the same sender slot. `rounds = slots`, one shuffle per dst slot at a constant source-slot index. This is the whole butterfly-partner family, including every stage of the register NTT.
- **Sender-side ternary** (`lane_block_invertible`). Sender slot varies per receiver but the 5×5 lane block `M` is invertible, so the sender computes which receiver needs its slot via `l = M⁻¹(…)` and offers the right slot through a nested `?:` chain. `rounds = slots`.
- **Multi-round pull** (paper §5.4 page 8's `2^|R|` exchange). Lane block singular — the sender can't resolve a unique receiver, so receivers pull. For each dst slot the required sender slot ranges over an affine subspace of size `2^|dirs|`, where `dirs = lane_slot_mix_dirs(C)` is the reduced basis of lane→slot mixing directions. I emit one unconditional shuffle per candidate sender slot with a guarded register write: `rounds = slots · 2^|dirs|`.

In the general case, using the paper's notation, splitting `f`'s columns by input space:

```
     slot   lane   warp
f = [A_reg, A_thr, A_wrp] logical
```

and similarly `g = [B_reg, B_thr, B_wrp] logical`. The number of elements that can be exchanged simultaneously is `n = |span(A_reg) ∩ span(B_reg)|`. Let `V` be the largest basis such that `span(V) ⊆ span(A_reg) ∩ span(B_reg)`, subject to $2^{|V|} \times \text{bits\_per\_elem} = \text{WARP\_SIZE} \times \text{SHUFFLE\_BITS}$.

Let $I = \text{span}(A_\text{thr}) \cap \text{span}(B_\text{thr})$ and $E, F$ be bases such that $\text{span}(A_\text{thr}) = \text{span}(E) \oplus I$, $\text{span}(B_\text{thr}) = \text{span}(F) \oplus I$.

Let $G = \{e_i \oplus f_i : e_i \in E, f_i \in F\}$.

$I$ is the subspace of elements that belong to the same thread in both A and B (no exchange needed), and $G$ is the basis of the subspace where every element belongs to a different thread of A and B (a single paired shuffle exchanges these). The paper's round count is $2^{|R|}$ where $R$ extends $V \cup I \cup G$ to a basis of $\mathbb F_2^d$.

TODO: this is not how the code currently does it, because it assumes that every element is 32 bits, so the `V/R` tiling is not relevant.  

**Example — lane rotation** (`convert_decompose::tests::lane_shuffle_picks_strategy_a`). `src = identity(9)` and

```
dst.bases = [2, 4, 8, 16, 1, 32, 64, 128, 256]
```

rotates the five lane bits (`i → (i+1) mod 5`). `C = src⁺ ∘ dst = dst`; warp columns are identity, the lane block is invertible → sender-side ternary path with `rounds = slots = 2`.

**Example — multi-round** (`convert_decompose::tests::multi_round_composite_picks_shuffle_with_exact_count`). `block = 32`, `src = identity(6)`, and

```
dst.bases = [32, 2, 4, 8, 16, 1]
```

Lane bit 0 feeds slot bit 5 and slot bit 5 feeds lane bit 0. The lane block is singular and the sender slot varies per lane, so I take the multi-round path: `slots = 2`, `|dirs| = 1`, `rounds = 4`.

#### 1.d Bounce (`Strategy::Bounce`, paper §5.4 Optimal Swizzling)

Everything else: `is_warp_column_identity` fails. C has some warp cross-term (`C_ws ≠ 0`, `C_wl ≠ 0`, `C_ww ≠ I`, `C_sw ≠ 0`, or `C_lw ≠ 0`) — a single-pass warp shuffle can't move the data, so I route through shared memory (`gen_reg_bounce`):

1. **Store**: for each src phys `x = s · blockDim + t`, `scratch[sh(f(x))] = src_reg[s]`.
2. `__syncthreads()`.
3. **Load**: for each dst phys `x = s · blockDim + t`, `dst_reg[s] = scratch[sh(g(x))]`.

`sh` is the scratch buffer's own `logical → phys` layout, picked by `choose_shared_layout` (see the shared-swizzle subsection below) to minimize bank conflicts across both accesses. `allocate_convert_scratch` sizes the scratch to a full tile.

Bounce is *always* applicable, so it's always in `best_decomposition`'s candidate set; it wins whenever Shuffle doesn't apply, or when the shuffle-round count is high enough that `rounds · shuffle_round_cost > 2 · shared_round + sync`.

**Example — warp-crossing composite** (`layout_infer::tests::warp_crossing_reader_bounces_via_scratch`). `t = 128, block = 128` (5 lane bits + 2 warp bits), reader `#[par((th, s) → th)]` (identity), index `(j & 31) · 4 + j / 32` — moves warp bits 5, 6 into lane positions 0, 1. `C`'s warp column isn't identity → Bounce. Scratch is sized to the full tile (128 elements) and `pick_scratch_layout` picks a bank-conflict-minimizing swizzle.

### Case 2: `ConvertLayout(Shared f, Reg g)`

Here `f : logical_src → phys_shared_addr` and `g : phys_dst → logical_dst` (register-buffer layouts point the opposite direction from shared-buffer layouts; see `BufferDecl::layout`'s doc comment). Codegen (the `(BufferKind::Register, BufferKind::Shared)` arm of `gen_convert`) emits a straight strided load:

```c
for (uint32_t x = threadIdx.x, s = 0; x < n; x += blockDim.x, ++s) {
    dst_reg[s] = b_src[G(x)];         // G = f ∘ map ∘ g : phys_dst → phys_shared_addr
}
```

Each thread loads from `b_src[G(x)]`. No scratch buffer, no `__syncthreads` from this op (the shared source's dirty write is fenced upstream by `insert_sync`).

**Bank-conflict optimization.** There's no receiver-driven C decomposition here — the source lives in shared memory and the conflict pattern is fixed by `f`. I set that once, either in `layout_infer` (`decide_target` falls back to identity for shared tiles) or in `choose_shared_layout` when the tile has multiple accesses. The load pattern `G` inherits it.

**Example** (`layout_infer::tests::multi_writer_tile_stays_shared`). A butterfly-style tile is written twice per point — `decide_target` blocks register promotion because `writer_count > 1`, so the tile stays shared with identity layout. Its linear reader gets a Shared→Reg convert into a fresh register version, and `insert_sync` inserts one `__syncthreads` between the last shared write and the convert.

### Case 3: `ConvertLayout(Reg f, Shared g)`

The mirror direction: `f : phys_src → logical_src`, `g : logical_dst → phys_shared_addr`. Codegen (the `(BufferKind::Shared, BufferKind::Register)` arm of `gen_convert`) emits a strided store:

```c
for (uint32_t x = threadIdx.x, s = 0; x < n; x += blockDim.x, ++s) {
    b_dst[G(x)] = src_reg[s];         // G = g ∘ map⁻¹ ∘ f : phys_src → phys_shared_addr
}
```

Each src slot `s` on thread `t` writes its content to `b_dst[G(x)]`. `map⁻¹` requires `map` to be invertible (a bijection between the two logical spaces); the current `layout_infer` only emits identity `map` here, so this is trivially satisfied.

**When I emit it.** A par with a non-analyzable read (`IndexMap::SExpr`, or an `IndexMap::Affine` that can't be linearized) into a register tile forces a shared mirror — register buffers are per-thread and non-linear reads can't be routed through per-slot indexing. `layout_infer` mirrors the tile into a shared version and rewires the reader to that. Register→register conversions with a warp-crossing composite go through the Bounce path of Case 1, not this arm, so Reg→Shared is only emitted for genuinely non-analyzable indices.

**Example** (`layout_infer::tests::sexpr_reader_gets_shared_mirror`). A tile is promoted to registers by its own-index writer, then read via `(j + #m) % t` (a symbolic offset — `IndexMap::SExpr`). `layout_infer` emits a Reg→Shared convert that mirrors the tile to a `_sm` buffer; `insert_sync` inserts one `__syncthreads` between the mirror store and the symbolic read.

### Bank-conflict optimization for shared buffers (`choose_shared_layout`)

**Where this is applied.** The 1.d Bounce path routes data through a shared scratch buffer, and *how* that scratch is addressed is what determines whether the store/load pair hits bank conflicts. `allocate_convert_scratch::pick_scratch_layout` runs `choose_shared_layout` for exactly this: it looks at the two register accesses (the store from `src.layout` and the load into `dst.layout`) and picks the scratch's `logical → phys` layout to minimize conflicts across both. `gen_reg_bounce` then uses the result as `sh` in the store/load address computation. (The same machinery is designed to run for user-facing shared buffers too — a shared tile with multiple accesses — but that path isn't wired in yet; `decide_target` currently falls back to identity for shared tiles.)

**Setup.** Following paper §5.4 + Appendix 9.2, each access is treated as a `hardware → address` linear layout, with its columns split by input space:

```
       slot (=Reg)   thread (=Thr)
A = [   A.Reg    |    A.Thr   ]  → shared address, output_dim bits
B = [   B.Reg    |    B.Thr   ]
```

`A.Reg` = the columns for slot-input bits (constant per lane within a warp instruction — one thread's slot is a register), `A.Thr` = the columns for thread inputs (lane ∪ warp). `output_dim = log2(buffer size in elements)`. In the code these come from `Access::reg_columns()` and `Access::thr_columns()`.

The goal is to partition the `output_dim` address bits into three 𝔽₂ subspaces:

- `vec` — vectorization bits, constant within a warp transaction. One vectorized load/store per warp.
- `bank` — bank-selecting bits: must vary across lanes so 32 lanes hit 32 distinct banks in parallel.
- `idx` — transaction-index bits: ideally constant across the warp so all 32 lanes hit the same 128-byte window.

with `|vec| + |bank| + |idx| = output_dim` and `|bank| = log2(128 / (2^|vec| · element_bytes))` (the 128-byte transaction budget; element width handles BabyBear=4, FpExt=16 automatically).

**Paper's construction.** Each letter below is a *basis* (in reduced row-echelon form, per `f2::reduce`) for the described 𝔽₂ subspace of `F₂^{output_dim}`; the code stores them as `Vec<u64>`. The operators `∩`, `+`, and "complement of X in Y" are the subspace operations on the spans, computed via Gaussian elimination in the `f2` module (`f2::intersection` uses Zassenhaus; `f2::sum` reduces the union; `f2::complement_within` extends greedily). `A.Reg`, `A.Thr` are the raw column lists of A's layout; subspaces enter by taking their span.

```
V   := basis of span(A.Reg) ∩ span(B.Reg)          maximal common vectorization       (vec)
U_A := basis of span(V ∪ A.Thr)                    varies within A's warp instruction (ua)
U_B := basis of span(V ∪ B.Thr)                    varies within B's warp instruction (ub)
I   := basis of span(U_A) ∩ span(U_B)              varies in both                     (common)
E   := basis of complement of span(I) in span(U_A) varies only in A                   (ea)
F   := basis of complement of span(I) in span(U_B) varies only in B                   (eb)
G   := [ E_i ⊕ F_i  for i = 1..min(|E|,|F|) ]      paired-XOR basis                   (paired)
C   := basis of complement of span(U_A + U_B)      global complement                  (global_complement)
       in F₂^{output_dim}
```

The size `|V|`, `|U_A|`, etc. denotes the dimension of the corresponding subspace (equivalently: the length of the reduced basis).

Note V is a basis of `span(A.Reg) ∩ span(B.Reg)`, not the column-wise intersection: two accesses can share a full-rank register subspace even when no individual column matches. E.g. if `A.Reg = [1, 2]` and `B.Reg = [3]` with `3 = 1 ⊕ 2`, both span the same 2-dimensional subspace, so `|V| = 2` and V's basis is (say) `[1, 2]`. That's what `f2::intersection` computes with Zassenhaus.

Bank-conflict-free access means: within one warp instruction, all 32 lanes give the same `idx` (address the same 128 B window) and 32 distinct `bank` values (so no bank serves two lanes). "Same `idx` across the warp" means every `idx` basis vector lies outside U_A (for A) and outside U_B (for B). So `idx` should be drawn from `complement(U_A ∪ U_B)` — that is exactly `C`.

If `|C| ≥ |idx|` a conflict-free swizzle exists; otherwise `G` covers the shortfall: each `g_i = e_i ⊕ f_i` lies outside both U_A and U_B — if `g_i ∈ U_A`, then `f_i = e_i ⊕ g_i ∈ U_A` (both terms in U_A), contradicting `f_i ∈ F = U_B \ I`; symmetric for U_B. This paired-XOR trick is *why* a two-access swizzle can be conflict-free where either alone can't: exclusive-to-A and exclusive-to-B directions cancel each other out. (The same `G` appears in the shuffle-round derivation of 1.c, for the same reason — paired exchange between two labeled thread bases.)

**Algorithm** (`optimal_shared_swizzle`):

1. Compute V, then `|bank|`, then `|idx| = output_dim − |V| − |bank|`.
2. Compute U_A, U_B, I, E, F.
3. Compute the safe subspace `safe = C ∪ G` (code: `safe = f2::sum(&global_complement, &paired)`).
4. Pick `idx`:
   - if `|safe| ≥ |idx|`: take `|idx|` independent vectors from `safe` — **fully conflict-free**.
   - else: take everything in `safe`, then fill the rest from U_A vectors that raise the rank (`ua_candidates` filtered by `!contains(current, c)`). These extra `idx` bits *are* dangerous — they will cause conflicts — but the swizzle is otherwise unavoidable.
5. `bank = complement(V ∪ idx, F₂^{output_dim})` — everything left over. `|bank|` comes out right by dimension counting.

Return `SharedLayout { vec, bank, idx, output_dim }`.

**Scoring residual conflicts.** `conflict_factor(sh, layout, block)` measures how badly `layout`'s lane columns collide under `sh`'s partition: it is `2^(|bank| − rank_in_bank(lane_cols)) − 1`, i.e. 0 when the lane bits span the full bank subspace (no serialization) and `k − 1` for `k`-way conflict. Zero for both accesses ⇒ fully conflict-free swizzle.

**Multi-access generalization.** For `N > 2` accesses (e.g. one writer + several readers), `choose_shared_layout` enumerates every unordered pair, runs `optimal_shared_swizzle` on it, scores each result by the loop-weighted sum of `conflict_factor` across *all* N accesses, and picks the winner. `row_major_default` (bank at low bits, idx at high bits, no vec) is included as a safety net so the sweep can never regress.

**Projection to a `LinearLayout`.** `to_linear_layout(sh)` concatenates the bases as `[vec | bank | idx]` — logical bit `j` is stored at phys address `bases[j]`. That `LinearLayout` becomes the scratch buffer's `layout` attribute, and `gen_reg_bounce` computes the store address as `sh.compose(src_layout)` and the load address as `sh.compose(dst_layout)`.

### Cost model (`layout_cost::ConversionCostModel`)

`best_decomposition` scores each candidate strategy with:

| Primitive | Cost |
|---|---|
| Register rename (Copy / Slot) | 0 |
| `__shfl_sync` round           | `shuffle_round = 3` |
| Shared round, k-way conflict  | `shared_round · max(k, 1) = 30 · k` |
| `__syncthreads`               | `sync = 100` |

Bounce cost is `store + sync + load` scaled by the max bank-conflict factor across the two accesses (`bounce_cost`); Shuffle cost is `shuffle_round · rounds`. Every cost is multiplied by the loop weight — the product of enclosing trip counts, with `symbolic_weight = 2^30` per unknown-bound loop, so any bounded-but-symbolic loop dominates known-small constants. The same cost model is shared by `find_opt_par` (coarse tier scoring) and `choose_shared_layout` (conflict-factor weighting).

