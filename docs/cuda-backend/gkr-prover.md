# GKR Fractional Sumcheck Prover

## Part 1: Protocol

Part 1 follows Protocol 3.4.5 (fractional sumcheck) in the SWIRL whitepaper.

### Overview

We begin with functions `p, q : {0,1}^n -> F_ext`. The goal is to reduce a claim about the rational hypercube sum

```
S = sum_{x in {0,1}^n} p(x) / q(x)
```

to evaluations of the multilinear extensions `p̂` and `q̂` at a random point. The GKR protocol applies to the fractional summation tree (segment tree) built from the `2^n` leaf fractions `(p(x), q(x))`.

### Fraction tree and layer polynomials

The tree has `n + 1` layers. Layer `0` is the root. Layer `n` is the leaves. Layer `j` has `2^j` nodes, each storing a pair of extension field elements `(p, q)`. The layer's witness functions `p_j, q_j : {0,1}^j -> F_ext` are defined recursively.

Starting from the leaves `(p_n, q_n)` (the input fractions), each parent layer is defined by fractional addition:

```
p_{j-1}(y) = p_j(0, y) * q_j(1, y) + p_j(1, y) * q_j(0, y)
q_{j-1}(y) = q_j(0, y) * q_j(1, y)
```

where `p_j(0, y)` and `p_j(1, y)` are the two halves of layer `j` (the first variable selects the half). The convention of evaluating 0,1 from the left leads to a better memory layout where hypercube coordinates are little-endian integers.

At the root (layer 0): `p_0` and `q_0` are scalars. The claimed fractional sum is `S = p_0 / q_0`.

### Polynomial identities

The recursive definition gives rise to polynomial identities between layers. Let `p̂_j, q̂_j` denote the multilinear extensions (MLEs) of `p_j, q_j`. Both sides below are multilinear and agree on the hypercube, so they agree everywhere:

```
p̂_{j-1}(Y) = sum_{y in {0,1}^{j-1}} eq(Y, y) * [p̂_j(0, y) * q̂_j(1, y) + p̂_j(1, y) * q̂_j(0, y)]
q̂_{j-1}(Y) = sum_{y in {0,1}^{j-1}} eq(Y, y) * [q̂_j(0, y) * q̂_j(1, y)]
```

These identities are the basis of the GKR reduction: a claim about `p̂_{j-1}` and `q̂_{j-1}` at a point can be reduced via sumcheck to claims about `p̂_j` and `q̂_j` at a new random point.

### GKR reduction (layer by layer)

The GKR protocol proceeds in rounds `j = 1, ..., n`. In round `j`, the prover starts with the MLEs `p̂_{j-1}, q̂_{j-1}`. The verifier holds evaluation claims `p̂_{j-1}(ξ^{(j-1)})` and `q̂_{j-1}(ξ^{(j-1)})` for a random point `ξ^{(j-1)} in F_ext^{j-1}` from the last round. (For the root, `ξ^{(0)}` is empty and `p̂_0, q̂_0` are constants.)

1. **Batch sumcheck.** Sample `λ_j` for batching. Using the polynomial identities, reduce the claims `p̂_{j-1}(ξ^{(j-1)})` and `q̂_{j-1}(ξ^{(j-1)})` to a single sumcheck:

   ```
   sum_{y in {0,1}^{j-1}} eq(ξ^{(j-1)}, y) * h(y) = p̂_{j-1}(ξ^{(j-1)}) + λ_j * q̂_{j-1}(ξ^{(j-1)})
   ```

   where `h` is the combined tree relation:

   ```
   h(y) = p̂_j(0, y) * q̂_j(1, y) + p̂_j(1, y) * q̂_j(0, y)
          + λ_j * q̂_j(0, y) * q̂_j(1, y)
        = [p̂_j(0, y) + λ_j * q̂_j(0, y)] * q̂_j(1, y) + p̂_j(1, y) * q̂_j(0, y)
   ```

   The sumcheck runs `j - 1` rounds with challenges `ρ^{(j-1)} = (ρ_0, ..., ρ_{j-2})`, reducing the sum to evaluation claims:

   ```
   p̂_j(0, ρ^{(j-1)}),  p̂_j(1, ρ^{(j-1)}),  q̂_j(0, ρ^{(j-1)}),  q̂_j(1, ρ^{(j-1)})
   ```

2. **Reveal claims.** The prover sends the four evaluation claims above. Note `p̂_j(·, ρ^{(j-1)})` and `q̂_j(·, ρ^{(j-1)})` are linear polynomials (in the first variable), so these two evaluations each determine a unique linear polynomial.

3. **Reduce with `μ`.** Sample `μ_j`. Reduce the four claims to two by evaluating the linear polynomials at `μ_j`:

   ```
   p̂_j(ξ^{(j)}) = (1 - μ_j) * p̂_j(0, ρ^{(j-1)}) + μ_j * p̂_j(1, ρ^{(j-1)})
   q̂_j(ξ^{(j)}) = (1 - μ_j) * q̂_j(0, ρ^{(j-1)}) + μ_j * q̂_j(1, ρ^{(j-1)})
   ```

   where `ξ^{(j)} = (μ_j, ρ^{(j-1)})`. These become the evaluation claims for the next round.

### Sumcheck round details

The sumcheck at GKR round `j` has `j - 1` sumcheck rounds (indexed `t = 0, ..., j - 2`). At sumcheck round `t`, we have already fixed variables to challenges `ρ_0, ..., ρ_{t-1}` from previous rounds. The round polynomial sums over the remaining `j - 2 - t` variables:

```
s_t(X) = sum_{y in {0,1}^{j-2-t}} eq(ξ^{(j-1)}, (ρ_{<t}, X, y)) * h_t(X, y)
```

where `h_t(X, y) = h(ρ_0, ..., ρ_{t-1}, X, y)` is `h` partially evaluated at the previous sumcheck challenges. Since `h` is degree 2 in each variable, `s_t` is degree 3 (the extra degree comes from `eq`).

---

## Part 2: Implementation

Implementation: `crates/cuda-backend/src/logup_zerocheck/fractional.rs` (`fractional_sumcheck_gpu`)
CUDA kernels: `crates/cuda-backend/cuda/src/logup_zerocheck/gkr.cu`

### Input preparation

In our use case (LogUp), the denominators have the form `q(x) = alpha + f(x)` where `alpha` is a random challenge. The `alpha` addition is done on the GPU (`add_alpha_kernel`) in `log_gkr_input_evals` (`gkr_input.rs`) before entering the GKR protocol. The GKR prover itself is generic over any `(p, q)` leaf fractions.

### Tree build

Input: `2^n` leaf pairs in a `DeviceBuffer<Frac<EF>>` called `layer`. `Frac<EF>` is a struct `{ p: EF, q: EF }` (`FracExt` on the CUDA side), so the buffer is an array of structs: `[(p_0, q_0), (p_1, q_1), ...]`. Leaf at index `i` holds `(p(x), q(x))` for hypercube point `x = (x_1, ..., x_n)` where `i = x_1 * 2^{n-1} + ... + x_n * 2^0` (big-endian).

#### Bit-reversal layout

In the natural (big-endian) ordering, leaf at index `i` corresponds to hypercube point `(x_1, ..., x_n)` with `i = x_1 * 2^{n-1} + ... + x_n * 2^0`. In a standard segment tree, the children of node `i` are at positions `2i` and `2i+1` — adjacent elements — so each layer pairs neighbors at stride 1.

Before building the tree, the leaves are permuted into bit-reversed order. This changes the tree wiring so that each layer's children are at stride `half` (a power of 2) apart in memory, which gives coalesced warp accesses since adjacent threads read adjacent positions.

The tree kernel pairs `layer[idx]` with `layer[idx + half]` where `half` is halved each layer. In bit-reversed order, this means the tree sums variables from last to first (x_n, x_{n-1}, ..., x_1).

Concrete example with `n = 3` (8 leaves):

```
Position:  0       1       2       3       4       5       6       7
Coords:    (0,0,0) (1,0,0) (0,1,0) (1,1,0) (0,0,1) (1,0,1) (0,1,1) (1,1,1)

Layer 0 (half=4): pairs at stride 4 -> sums over x_3
  [0]+[4]: (0,0,_)    [1]+[5]: (1,0,_)    [2]+[6]: (0,1,_)    [3]+[7]: (1,1,_)

Layer 1 (half=2): pairs at stride 2 -> sums over x_2
  [0]+[2]: (0,_,_)    [1]+[3]: (1,_,_)

Layer 2 (half=1): pairs at stride 1 -> sums over x_1
  [0]+[1]: root
```

The children of node `idx` in a layer of size `half` are at positions `idx` and `idx + half` in the layer below (size `2*half`). In hypercube terms, these children differ in the variable being summed at that level.

#### In-place build with revert

The tree is never materialized in full. Instead, we maintain a single buffer and repeatedly sum in-place: for each layer, `layer[idx] += layer[idx + half]` for `idx < half`. This overwrites the left half with parent values, but the right half (`layer[idx + half]`) is preserved unchanged. Because fractional addition is reversible (`frac_unadd` inverts `frac_add` given the right child, requiring the right child's denominator to be nonzero), we can recover any layer's left-half values later by reverting from the preserved right half.

This means only a single buffer of `2^n` elements is needed (the size of the leaf layer) rather than `2^{n+1} - 1` elements for a fully materialized tree. The GKR rounds revert one layer at a time as needed.

Steps:

1. **Bit-reverse** the leaves (`bit_rev_frac_ext`).
2. **Build tree** by `n` rounds of in-place fractional addition, each halving the active size:
   ```
   for idx < half:
       layer[idx] = frac_add(layer[idx], layer[idx + half])
   // layer[idx + half] is untouched — preserved for later revert
   ```
   where `frac_add(L, R) = (L.p * R.q + R.p * L.q, L.q * R.q)`.
3. Read the root `(p_0, q_0)` — this is the claimed fractional sum. Observe `p_0` (unless `assert_zero`) and `q_0` in the transcript.
4. **Revert** the root layer (`frac_build_tree_layer(revert=true)` on size 2) to recover the two children `p̂_1(0), q̂_1(0), p̂_1(1), q̂_1(1)` as the first layer's claims (`claims_per_layer[0]`).
5. Observe the four claim values and sample `μ_1` from the transcript. Set `ξ^{(1)} = [μ_1]`.

This completes GKR round `j = 1` (which has a trivial 0-round sumcheck). The remaining GKR rounds `j = 2, ..., n` each revert the next tree layer and run a `j - 1` round sumcheck — see "Sumcheck round strategies" below.

Kernels:
- `bit_rev_permutation_z` — bit-reversal permutation of `(EF, EF)` pairs
- `frac_build_tree_layer_kernel<false>` — tree build (`frac_add_inplace`)
- `frac_build_tree_layer_kernel<true>` — tree revert (`frac_unadd_inplace`)

### Sumcheck round implementation

Since `eq` factorizes over its variables, the round polynomial splits into:

```
s_t(X) = eq_r_acc_t * eq(ξ_t, X) * s'_t(X)
```

where:
- `ξ_t = ξ^{(j-1)}[t]`
- `eq_r_acc_t = prod_{i<t} eq(ξ_i, ρ_i)` — accumulated scalar from previous rounds
- `s'_t(X) = sum_y eq(ξ_{>t}, y) * h_t(X, y)` — the **reduced round polynomial** (degree 2, `GKR_SP_DEG = 2`)

Factoring out `eq_r_acc_t` and `eq(ξ_t, X)` reduces what the GPU must compute from degree 3 to degree 2, requiring evaluations at only 3 points for interpolation.

**Round mechanics** (`reconstruct_s_evals`):
1. GPU computes `s'_t(1)` and `s'_t(2)` (via `accumulate_compute_contributions`).
2. CPU scales by `eq_r_acc_t`: `s'_t(i) *= eq_r_acc_t`.
3. CPU derives `s'_t(0)` from the sumcheck identity `s_t(0) + s_t(1) = s_{t-1}(ρ_{t-1})` and the relation `s_t(X) = eq(ξ_t, X) * s'_t(X)`.
4. Now `s'_t(0), s'_t(1), s'_t(2)` are known — 3 evaluations of a degree-2 polynomial. CPU interpolates `s'_t(3)`.
5. CPU computes `s_t(i) = eq(ξ_t, i) * s'_t(i)` for `i = 1, 2, 3`.
6. The prover observes `s_t(1), s_t(2), s_t(3)` in the transcript. The verifier samples challenge `ρ_t`.

### Eq buffer: sqrt decomposition

The reduced round polynomial `s'_t(X) = sum_y eq(ξ_{>t}, y) * h_t(X, y)` needs `eq(ξ_{>t}, y)` for all tail points `y`. We store this with `SqrtEqLayers` instead of a single full table.

At each GKR round, `SqrtEqLayers::from_xi` is built from `xi_prev[1..]` (length `round - 1`) with an initial split:
- `low_n = floor((round - 1) / 2)`
- `high_n = (round - 1) - low_n`

During sumcheck rounds, `drop_layer()` updates this split by dropping from `high` first, then `low`. For the current inner round, let `k = low_n` and `m = low_n + high_n`. Then:

```
eq(ξ_{>t}, y) = eq(ξ_low, y_low) * eq(ξ_high, y_high)
```

where `y_low = y & (2^k - 1)` and `y_high = y >> k`. Kernels reconstruct the full weight from two tables (`sqrt_buffer_get` in `gkr.cu`):

```
eq_weight[y] = eq_xi_low[y & (2^k - 1)] * eq_xi_high[y >> k]
```

The two tables have sizes `2^k` and `2^{m-k}` instead of one table of size `2^m`.

**Advancing rounds.** Eq layers are precomputed upfront; each sumcheck round calls `drop_layer()` (no GPU eq-fold kernel). Precompute-M uses `eq_tail_ptrs` to select low/high pointers for the tail part after skipping window variables.

### Sumcheck round strategies

In the code, the loop `for round in 1..total_rounds` handles GKR rounds `j = 2, ..., n` with `j = round + 1`. GKR round `j = 1` (trivial: 0-round sumcheck) is handled by the initialization above. At the start of code `round`, `xi_prev` holds `ξ^{(round)}` (= `ξ^{(j-1)}`), and `lambda` holds `λ_j` (= `λ_{round+1}`).

The first sumcheck round (round 0) of each GKR round is always handled by `do_sumcheck_round_and_revert`, which fuses the tree layer revert with computing `s'_0(1)` and `s'_0(2)`. Subsequent sumcheck rounds use one of two strategies:

#### Fold-eval strategy

Processes one variable per sumcheck round. At the start of round `t`, the buffer holds `2 * pq_size` evaluations of `p̂_j` and `q̂_j` partially evaluated at the previous challenges `ρ_0, ..., ρ_{t-2}`, but **not** yet folded by the latest challenge `ρ_{t-1}`. A single fused kernel does two things simultaneously:

1. **Fold** the buffer by challenge `ρ_{t-1}`: for each pair, compute `even + ρ_{t-1} * (odd - even)`, halving `pq_size`. This produces the partially-evaluated multilinear values at `(ρ_0, ..., ρ_{t-1}, ·)`.
2. **Compute** `s'_t(1)` and `s'_t(2)` from the freshly folded values, multiplying by `eq` weights from `eq_buffer` and reducing via block sums.

**Buffer strategy:** The `layer` buffer (size `2^n`) holds the tree and is needed for reverts in future GKR rounds. During most GKR rounds, the first fold copies data from `layer` into a secondary `work_buffer` (size `2^{n-2}`, which suffices since GKR round `j` starts with `2^{j-1}` evaluations and `j < n`), and all subsequent folds operate in-place on `work_buffer`. On the **last GKR round** (`j = n`) there are no future reverts, so all folds operate in-place on `layer` directly — no `work_buffer` is needed. The final fold after the last sumcheck round is standalone (no next round to compute).

Kernels:
- `compute_round_and_revert_kernel` — fused revert + compute (sumcheck round 0 only)
- `compute_round_and_fold_inplace_kernel` — fused compute + fold (in-place)
- `compute_round_and_fold_kernel` — fused compute + fold (out-of-place, ping-pong)
- `fold_ef_columns_kernel` — standalone fold (final inner round)
- `static_final_reduce_block_sums` — reduce per-block partial sums to `s'(1)` and `s'(2)`

#### Precompute-M strategy

Processes `w` variables at once (`w = GKR_WINDOW_SIZE = 3`). It is not work-efficient: M-build is `O(2^{n+w})` vs `O(w * 2^n)` for `w` fold-eval rounds (a `2^w / w` arithmetic overhead), but it reduces global-memory passes by evaluating `w` rounds from a small `4^w` matrix.

We follow ideas in [Section 4, https://eprint.iacr.org/2025/1473] as applied to sumcheck on the GKR layer polynomial.

**Derivation.** The GKR round polynomial has the form (see Part 1):

```
s_i(X) = eq((r, X), z_r) * sum_{u,v in {0,1}^{i-1}} eq(u, r) * eq(v, r)
          * [sum_{b in {0,1}^{n-i-1}} (p_0(u,X,b)*q_1(v,X,b) + p_1(u,X,b)*q_0(v,X,b)
              + lambda * q_0(u,X,b)*q_1(v,X,b)) * eq(b, z_b)]
```

where `p_0, q_0` and `p_1, q_1` are the two halves (first variable = 0 or 1) and `z_r, z_b` are the corresponding parts of `z = xi^{(j-1)}`. The `eq((r,X), z_r)` factor is separated because it only depends on `r` and `X`, not on `u, v, b`.

We precompute:

```
M_{αβ}[u, v] = sum_{b in {0,1}^{n-i-1}} (p_0(u,α,b)*q_1(v,β,b) + p_1(u,α,b)*q_0(v,β,b)
                + lambda * q_0(u,α,b)*q_1(v,β,b)) * eq(b, z_b)
```

for `α, β in {0, 1}`. Then `s'_i(X)` (with the `eq((r,X), z_r)` factor removed) is a degree-2 polynomial in `X` expressible in terms of the four `M_{αβ}` tables via multilinearity:

```
s'_i(X) = sum_{u,v} eq(u,r) * eq(v,r) * [(1-X)^2 * M_{00}[u,v] + (1-X)*X * (M_{01}[u,v] + M_{10}[u,v]) + X^2 * M_{11}[u,v]]
```

**Windowed implementation.** In the code, `w = GKR_WINDOW_SIZE` sumcheck rounds are handled per window. The M matrix is `2^w x 2^w`, where the row index `u` and column index `v` are each `w`-bit numbers corresponding to the window variables.

At sumcheck round `t` within the window (`t = 0, ..., w - 1`), each `w`-bit index is decomposed as `[prefix (t bits)] [current X (1 bit)] [suffix (w - t - 1 bits)]`. The prefix bits correspond to challenges already sampled in this window; the suffix bits correspond to window positions not yet reached. The round polynomial evaluation becomes:

```
s'_t(X) = sum_{b1,b2 in {0,1}^t} sum_{s in {0,1}^{w-t-1}}
          eq_r_prefix[b1] * eq_r_prefix[b2] * eq_suffix[s] * f(X, M[b1|X|s, b2|X|s])
```

where:
- `eq_r_prefix` = MLE table of the `t` challenges sampled so far in this window, i.e. `eq_r_prefix[b] = eq(b, (r_0, ..., r_{t-1}))`
- `eq_suffix` = MLE table of the remaining `xi_prev` values for suffix positions, i.e. `eq_suffix[s] = eq(s, (z_{t+1}, ..., z_{w-1}))`
- `f(X, ...)` combines the four `M_{αβ}` entries using multilinearity in `X` as above

Note that `u` and `v` share the same suffix bits `s` — this is because both sides of the product (`p_0 * q_1`, etc.) share the same hypercube variables `b` in the M definition, so the suffix contraction is shared.

At round `t = 0`: `eq_r_prefix` is trivial (size 1), `eq_suffix` has size `2^{w-1}`. At round `t = w - 1`: `eq_r_prefix` has size `2^{w-1}`, `eq_suffix` is trivial (size 1). The total work per round is `O(4^w)` — scanning the M matrix — regardless of the evaluation buffer size.

The M build can start from two different states, corresponding to two kernel variants (`precompute_m_build_partial_kernel<inline_fold>`):

- **After a multifold** (or at the start of a subsequent window): the evaluation buffer is already folded and has `2^{rem_n}` elements. The kernel reads directly from the buffer (`inline_fold = false`).
- **After a compute-and-revert round** (first window of each GKR round): the buffer still has `2^{rem_n+1}` elements because the fold by the previous challenge `r_prev` has not yet been applied. The kernel folds inline while building M (`inline_fold = true`), computing `val = lo + r_prev * (hi - lo)` on the fly.

Steps for each window of `w` rounds:

1. **Build M** (`frac_precompute_m_build_raw`) — For each tail point `b`, compute the contributions to `M[u,v]` weighted by `eq(b, z_b)` and `lambda`, and accumulate. Parallelized over tail points: each CUDA block processes a tile of `tail_tile` tail points, producing a partial M matrix. Partials are reduced by `precompute_m_reduce_partials_kernel`.

2. **Eval rounds** (repeat `w` times) — For each round `t`, compute `s'_t(1)` and `s'_t(2)` from M by contracting with `eq_r_prefix` and `eq_suffix` as described above. This is a small kernel (`precompute_m_eval_round_kernel`) operating on the `4^w`-sized M matrix. Each round calls `observe_and_update` to sample `r_t` and update accumulators. No buffer folding happens during these rounds.

3. **Multifold** (`frac_multifold_raw`) — After the window, fold the evaluation buffer by all deferred challenges at once. When `pending_fold` is true (first window), this includes the pending `r_prev` plus the `w` window challenges, folding `w + 1` coordinates total (`pq_size >>= w + 1`). Otherwise it folds only the `w` window challenges (`pq_size >>= w`). Computes `eq_r_window` = MLE table of all deferred challenges.

After all windows, any remaining rounds fall back to the fold-eval path.

Kernels:
- `precompute_m_build_partial_kernel<false>` — build partial M blocks from an already-folded buffer
- `precompute_m_build_partial_kernel<true>` — same, but folds inline by `r_prev` (first window)
- `precompute_m_reduce_partials_kernel` — reduce partial M blocks into final M
- `precompute_m_eval_round_kernel` — extract `s'(1)`, `s'(2)` from M
- `multifold_kernel<W>` — multi-coordinate fold (templated on fold width)

#### Strategy selection

`choose_round_strategy()` picks precompute-M when all of the following hold:
- Not disabled via `SWIRL_CUDA_GKR_PRECOMPUTE_M=0`
- A valid window width `w` exists: `rem_n >= GKR_WINDOW_MIN_LOG_N` (22), at least `w` sumcheck rounds remain (`base + w <= stop` where `stop = (round + 1) / 2`), and enough tail points (`2^(rem_n - w)`) to fill `min_blocks` CUDA blocks

Otherwise it falls back to fold-eval.

### Additional buffers
- `m_buffer` — the `4^w`-sized M matrix (`DeviceBuffer<EF>`)
- `m_partial_buffer` — partial M blocks from each CUDA thread block
- `eq_r_prefix_buffer`, `eq_suffix_buffer` — small device buffers for MLE evaluation tables (size `<= 2^w`)
- `d_sum_evals` — two `EF` values holding `s'(1)` and `s'(2)` from each round
- `tmp_block_sums` — block-level partial sums for reduction; reused for `eq_r_window` upload during multifold (safe because the two uses don't overlap temporally)

### GPU memory accounting

Let:
- `n = log2(total_leaves)`
- `w = GKR_WINDOW_SIZE` (default `w = 3`)
- `|EF| = sizeof(EF)` bytes
- `|Frac| = sizeof(Frac<EF>) = 2 * |EF|`
- Assume `n` is large
- Assume precompute-M is active

Buffers used at peak (workspace only; excludes input `layer`/leaves):
- `work_buffer`: `2^(n-w-2) * |Frac|` (precompute-M defers `w+1` folds before first write)
- `copy_scratch`: `1 * |Frac|`
- `d_sum_evals`: `2 * |EF|`
- `tmp_block_sums`: `< 2^(n-7) * |EF|`
- `eq_buffer` (`SqrtEqLayers`): `(2^floor(n/2) + 2^ceil(n/2) - 2) * |EF|`
- `m_buffer`: `4^w * |EF|` (default `w=3` gives `64 * |EF|`)
- `m_partial_buffer`: `num_blocks * 4^w * |EF|`, with `num_blocks <= 2^(n-w-10)` for large `n`, so `< 2^(n+w-10) * |EF|`
- `eq_r_prefix_buffer` + `eq_suffix_buffer`: `2^(w+1) * |EF|` (default `w=3` gives `16 * |EF|`)

Workspace upper bound (sum of the buffers above), using `|Frac| = 2 * |EF|`:
- `W < (2^(n-w-1) + 2^(n-7) + 2^(n+w-10) + 2^floor(n/2) + 2^ceil(n/2) + 4^w + 2^(w+1) + 2) * |EF|`

Total GPU memory required (workspace + input leaves `2^n * |Frac| = 2^(n+1) * |EF|`):
- `M_total < (2^(n+1) + 2^(n-w-1) + 2^(n-7) + 2^(n+w-10) + 2^floor(n/2) + 2^ceil(n/2) + 4^w + 2^(w+1) + 2) * |EF|`
- With default `w = 3` and `|EF| = 16` bytes:
  - `M_total < (2^(n+1) + 2^(n-4) + 2^(n-6) + 2^floor(n/2) + 2^ceil(n/2) + 82) / 2^26 GiB`
  - `n = 27`: `M_total < 4.16 GiB`
  - `n = 28`: `M_total < 8.31 GiB`
  - `n = 29`: `M_total < 16.63 GiB`
  - `n = 30`: `M_total < 33.25 GiB`

Dominant term is the input leaves (`2^(n+1) * |EF|`). Workspace overhead is ~4%.
