# Pipelined fractional sumcheck with precompute-M

At round $j$ of sumcheck, there's $j-1$ internal rounds with $t \in \{0, ..., j-2\}$. At round $t$ there's already sampled challenges $\rho_0,\dots,\rho_{t-1}$, the prover needs to compute $$s_t(X)=\sum_{y\in \{0,1\}^{j-2-t}} \text{eq}(ξ^{(j-1)}, (ρ_{<t}, X, y)) * h_t(\rho_{<t}, X, y)$$

With the precompute-M strategy, at round $t$ we can process $w$ variables at once.

$$\begin{align} s_i(X) = \text{eq}((\rho_{<t}, X), ξ^{(0..t-1)}) \sum_{u,v \in \{0,1\}^{t-1}}&
\text{eq}(u, \rho_{<t}) \text{eq}(v, \rho_{<t})
          \\&[\sum_{b \in \{0,1\}^{j-t}} \text{eq}(b, ξ^{(t..j-1)})(p_0(u,X,b)q_1(v,X,b) + 
          \\ & \qquad\qquad p_1(u,X,b)q_0(v,X,b)
              + \lambda q_0(u,X,b)q_1(v,X,b)) 
              ]\end{align}$$

First, precompute $$M_{αβ}[u, v] = \sum_{b \in \{0,1\}^{j-t}} \text{eq}(b, ξ^{(t..j-1)})(p_0(u,α,b)q_1(v,β,b) + p_1(u,α,b)q_0(v,β,b)
                + \lambda q_0(u,α,b)q_1(v,β,b))
$$

Then let $$\begin{align} s'_i(X) = \sum_{u,v}& \text{eq}(u,\rho_{<t}) \text{eq}(v,\rho_{<t})\\&[(1-X)^2 M_{00}[u,v] + (1-X)X (M_{01}[u,v] + M_{10}[u,v]) + X^2 M_{11}[u,v]]\end{align}
$$
Let $f(X, M, u, v) = (1-X)^2 M_{00}[u,v] + (1-X)X (M_{01}[u,v] + M_{10}[u,v]) + X^2 M_{11}[u,v]$ 

Then 

$$s'_t(X) = \sum_{b_1,b_2 \in \{0,1\}^t} \sum_{s \in \{0,1\}^{w-t-1}}
          \text{eq\_r\_prefix}[b_1]  \text{eq\_r\_prefix}[b_2] \text{eq\_suffix}[s] f(X, M, b_1|X|s, b_2|X|s)$$
where:
- `eq_r_prefix` = MLE table of the `t` challenges sampled so far in this window, i.e. $\text{eq\_r\_prefix}[b] = \text{eq}(b, (\rho_0, ..., \rho_{t-1})$
- `eq_suffix` = MLE table of the remaining `xi_prev` values for suffix positions, i.e. $\text{eq\_suffix}[s] = \text{eq}(s, ξ^{(t..j-1)})$

**Original Strategy**:

Steps for each window of `w` rounds:

1. **Build M** (`frac_precompute_m_build_raw`) — For each tail point `b`, compute the contributions to `M[u,v]` weighted by `eq(b, z_b)` and `lambda`, and accumulate. Parallelized over tail points: each CUDA block processes a tile of `tail_tile` tail points, producing a partial M matrix. Partials are reduced by `precompute_m_reduce_partials_kernel`.

2. **Eval rounds** (repeat `w` times) — For each round `t`, compute `s'_t(1)` and `s'_t(2)` from M by contracting with `eq_r_prefix` and `eq_suffix` as described above. This is a small kernel (`precompute_m_eval_round_kernel`) operating on the `4^w`-sized M matrix. Each round calls `observe_and_update` to sample `r_t` and update accumulators. No buffer folding happens during these rounds.

3. **Multifold** (`frac_multifold_raw`) — After the window, fold the evaluation buffer by all deferred challenges at once. When `pending_fold` is true (first window), this includes the pending `r_prev` plus the `w` window challenges, folding `w + 1` coordinates total (`pq_size >>= w + 1`). Otherwise it folds only the `w` window challenges (`pq_size >>= w`). Computes `eq_r_window` = MLE table of all deferred challenges.

**Pipelined Strategy**:

For round $j$ assume we already sampled $\lambda$ and hold `xi_prev` ($ξ^{j-1}$). 

For the first round, precompute M with window $w$, so we can eval rounds $[0,w)$ (eval $s'$ and sample from transcript.) Denote $M^{[i,j)}$ to mean that we hold precomputed $M$ with window $[i,j)$. When $i > 0$ this means that $M$ was evaluated on the folded $p_0, p_1, q_0, q_1$ i.e.

$$M_{αβ}^{[i,j)}[u, v] = \sum_{b \in \{0,1\}^{j-t}} \text{eq}(b, ξ^{(t..j-1)})(p_0(\rho_{\leq i}, u,α,b)q_1(\rho_{\leq i}, v,β,b) + p_1(\rho_{\leq i}, u,α,b)q_0(\rho_{\leq i}, v,β,b)
                + \lambda q_0(\rho_{\leq i}, u,α,b)q_1(\rho_{\leq i}, v,β,b))$$

Pick some $\alpha \in \mathbb Z$ such that $0< \alpha < w$, say $\alpha = \text{floor}(w/2)$ (assuming $w > 2$)

So in the first round we compute $M^{[0,w)}$.

In stream 1: eval $s'$ from $M^{[0,w)}$ for bits $t \in [0,\alpha)$. Sample $\rho_0,\dots,\rho_{\alpha-1}$. Fold $p_0,q_0,p_1,q_1$ with the previous samples.
In stream 2: Receive folded $p_0,q_0,p_1,q_1$, precompute $M^{[\alpha, w+\alpha)}$. 
In stream 1: eval $s'$ from $M^{[0,w)}$ for bits $t \in [\alpha,w)$. Sample $\rho_\alpha,\dots,\rho_{w-1}$. Fold $p_0,q_0,p_1,q_1$ with the previous samples.
In stream 2: Receive folded $p_0,q_0,p_1,q_1$, precompute $M^{[w, 2w)}$

...

In stage $i$ of this process:

In stream 1: eval $s'$ from $M^{[(i-1)w+\alpha, iw+\alpha)}$, for bits $t \in [iw, iw+\alpha)$. Sample $\rho_{iw},\dots,\rho_{iw + \alpha - 1}$. Fold $p_0,q_0,p_1,q_1$ with the previous samples.
In stream 2: Receive folded $p_0,q_0,p_1,q_1$, precompute $M^{[iw+\alpha, (i+1)w+\alpha)}$. 
In stream 1: eval $s'$ from $M^{[iw,(i+1)w)}$ for bits $t \in [iw+\alpha,(i+1)w)$. Sample $\rho_{iw+\alpha},\dots,\rho_{(i+1)w-1}$. Fold $p_0,q_0,p_1,q_1$ with the previous samples.
In stream 2: Receive folded $p_0,q_0,p_1,q_1$, precompute $M^{[(i+1)w, (i+2)w)}$
In stream 1: eval $s'$ from $M^{[iw+\alpha,(i+1)w+\alpha)}$ for bits $t \in [(i+1)w,(i+1)w+\alpha)$. Sample $\rho_{(i+1)w},\dots,\rho_{(i+1)w+\alpha-1}$. Fold $p_0,q_0,p_1,q_1$ with the previous samples.

etc...

## Concrete schedule

Indexing for this section: inner rounds of the windowed region are $t = 0, 1, \dots$
with challenges $\rho_t$. (In code these sit at absolute inner rounds
$\text{base} + t$ with $\text{base} = 1$; inner round 0 is the fused revert round
and its challenge `r_prev` is inline-folded by the startup M-build, exactly like
`pending_fold = true` in the eager driver.) $M^{[a,a+w)}$ is the M matrix over
window variables $[a, a+w)$, built from the pq buffer folded by $\rho_{<a}$.

Time is divided into **half-slots** of alternating length $\alpha, w-\alpha$:

- half-slot $k$ covers rounds $[\text{lo}(k), \text{lo}(k+1))$, where
  $\text{lo}(0) = 0$ and $\text{lo}(k+1) = \text{lo}(k) + (\alpha$ if $k$ even,
  else $w - \alpha)$. Note $\text{lo}(k+1) = \text{lo}(k-1) + w$.
- M windows are built at every base $\text{lo}(k)$, i.e. $0, \alpha, w, w+\alpha, 2w, \dots$
- $M^{[0,w)}$ serves half-slots 0 **and** 1 (rounds $[0,\alpha)$ and $[\alpha,w)$);
  for $k \ge 2$, half-slot $k$ is served by $M^{[\text{lo}(k-1),\,\text{lo}(k-1)+w)}$,
  built during half-slot $k-1$. Since $\text{lo}(k+1) = \text{lo}(k-1) + w$, that
  M covers half-slot $k$ exactly (its second half).

Evaluating round $t$ from $M^{[a,a+w)}$ contracts with
$\text{eq\_r\_prefix} = \text{MLE}(\rho_a, \dots, \rho_{t-1})$ (window positions
$[a,t)$) and $\text{eq\_suffix} = \text{MLE}$ of the $\xi$ values at positions
$[t+1, a+w)$ — the same `precompute_m_eval_round` kernel as today; the prefix
just crosses half-slot boundaries.

### Timeline

```
              slot 0        slot 1            slot 2             slot 3               slot 4
            ┌─────────────┬─────────────────┬──────────────────┬────────────────────┬──────────────────
 S_eval     │ eval [0,α)  │ eval [α,w)      │ eval [w,w+α)     │ eval [w+α,2w)      │ eval [2w,2w+α)
 (serial:   │ ← M[0,w)    │ ← M[0,w)        │ ← M[α,w+α)       │ ← M[w,2w)          │ ← M[w+α,2w+α)
 transcript)│             │                 │                  │                    │
            ├─────────────┼─────────────────┼──────────────────┼────────────────────┼──────────────────
 S_build    │ (idle;      │ fold ρ[0,α)     │ fold ρ[α,w)      │ fold ρ[w,w+α)      │ fold ρ[w+α,2w)
            │  M[0,w)     │ build M[α,w+α)  │ build M[w,2w)    │ build M[w+α,2w+α)  │ build M[2w,3w)
            │  built in   │                 │                  │                    │
            │  startup)   │                 │                  │                    │
            └─────────────┴─────────────────┴──────────────────┴────────────────────┴──────────────────

 deps:  fold+build in slot k  ←  challenges sampled in slot k-1, pq output of slot k-1's fold
        evals in slot k       ←  M built in slot k-1  (slots 0 and 1 both read the startup M[0,w))
```


### Notes

- **M double-buffering.** `M` and `M_next` are alive simultaneously — two
  $4^w$-sized M buffers (ping-pong), instead of one in the eager driver.
- **Work overhead.** Builds and folds occur at bases spaced
  $\alpha, w-\alpha, \alpha, \dots$ instead of $w$. Buffer sizes shrink
  geometrically, so total build work and fold read-traffic scale by
  $(1 + 2^{-\alpha})$ vs the non-pipelined windowed driver (+25% at
  $\alpha = 2$). This is the price of moving them off the transcript-serial
  critical path.
- **Win condition.** Per stage, pipelining hides
  $\min(\text{eval half-slot},\ \text{fold}+\text{build})$; it wins roughly iff
  $w \cdot t_{\text{eval+transcript}} \gtrsim 2^{-\alpha} (t_{\text{fold}} +
  t_{\text{build}})$ at that window's buffer size. For the first (largest)
  window of a large outer round, build ≫ eval time and the schedule can
  regress wall-clock; gate per-window (e.g. via the what-if/ATG harness)
  rather than enabling unconditionally. Larger $\alpha$ reduces the extra work
  but shortens the half-slot the build must hide behind; $\alpha = \lfloor w/2
  \rfloor$ balances the two.
- **Challenge visibility.** Builds on S_build consume $\rho$'s sampled on the
  host mid-stream; use the `DEV_CH` build variant / eq-MLE-table staging (as
  in the existing graph-IR driver) so S_build doesn't serialize on an extra
  host round trip.
- **Eq-table bookkeeping.** A build at base $a$ runs while eval bookkeeping is
  still at an earlier round, so tail tables $\text{eq}(b, \xi)$ must be
  selected with explicit per-stage drop counts (per-stage prefix/suffix
  chains), not the shared mutable `SqrtEqLayers::drop_layer` cursor.

