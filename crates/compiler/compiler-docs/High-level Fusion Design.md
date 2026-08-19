
# Kernel Fusion V2 — Architecture

A GPU kernel fusion optimizer that **enumerates fusion possibilities**, **scores them statically**, and **picks the best subset** with a solver.

---

## The three pieces

```
                ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  Input   ───▶  │  1 · BUILD   │▶  │  2 · SCORE   │▶  │  3 · PICK    │  ───▶  Output
  A DAG of      │  Alternative │   │  Static Cost │   │   CP-SAT     │      Chosen subgraph
  GPU kernels   │    Graph     │   │    Model     │   │   Solver     │      → compile → run
                └──────────────┘   └──────────────┘   └──────────────┘
```

### 1 · BUILD — Alternative Graph

*All fusion possibilities in one DAG.*

- Wraps the original kernels
- Fusion passes propose fused alternatives
- Multiple producers per value ⇒ alternatives
- Bounded "saturation" adds more each round

### 2 · SCORE — Static Cost Model

*Predicts runtime without compiling.*

- Lowers each candidate to KIR
- Estimates registers, occupancy, DRAM traffic, critical path
- Memoized by module hash
- No `nvcc`, no GPU profiling

### 3 · PICK — CP-SAT Solver

*Globally optimal closed subgraph.*

- Integer program over `select-alt`, `need-value`, `compile-module`
- Minimize runtime, then compile cost
- Original graph is always a feasible fallback

---

## What's in the Alternative Graph

Bipartite: **values** (circles) and **kernels** (boxes). Two boxes producing the same circle are **alternatives**. Picking one closed set = picking one execution plan.

```
  inputs                                                    outputs

   (a) ─────────▶ [ producer ] ──▶ (m) ──▶ [ consumer ] ──▶ (y)
    │  ╲                                        ▲            ▲
    │   ╲                                       │            ┊
   (b) ─────────────────────────────────────────┘            ┊
        ╲                                                    ┊
         ╲┈┈┈┈┈▶ ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                       ┊
                   fused P+C            ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┘
   (b) ┈┈┈┈┈┈┈▶  └(alternative for y) ┘

   ───  original          ┈┈┈  synthesized alternative
                          two producers of y = alternatives
```

---

## Fusion patterns the passes generate

Each pass proposes candidates against a shared graph; a candidate is one atomic fused kernel.

| Pattern | Shape | What it buys |
|---|---|---|
| **Producer → Consumer** | `[P] → [C]` collapses to `[P → C fused]` | Eliminate a store/load |
| **Fanout** | one `[P]` feeding many consumers becomes `1 launch, share P` | Compute P once, feed many |
| **Small kernel batch** | many tiny boxes collapse to `1 launch, all bodies` | Kill many launches |

---

## Static cost model

Each candidate is lowered through the ordinary compile path *up to the point right before `nvcc`*: HIR → KIR → layout inference → sync insertion → shared-memory planning. From that KIR we derive four numbers, take their max as the throughput cost, and add a launch overhead.

### Occupancy

How many blocks fit on an SM at once. Four independent budgets (threads, warps, registers, shared memory) each cap it; the min wins.

```
  ┌──────── one SM ────────┐
  │ ▭ ▭ ▭ ▭                │ ◀── threads · T
  │ ▭ ▭ ▭ ▭                │ ◀── warps  · ⌈T/32⌉
  │                        │ ◀── regs   · R·T
  │ blocks_per_sm resident │ ◀── smem   · S
  └────────────────────────┘
        each budget → cap → take min
```

$$
b_{\mathrm{SM}} = \min\left( \frac{T_{\mathrm{SM}}}{T},\ \frac{W_{\mathrm{SM}}}{\lceil T/32 \rceil},\ \frac{R_{\mathrm{SM}}}{R\,T},\ \frac{S_{\mathrm{SM}}}{S} \right)
$$

$`R`$ comes from KIR SSA liveness (peak live scalars × cost per word). $`S`$ is the static shared-memory plan. If min = 0, the candidate is rejected.

> $`T, R, S`$: per-block threads / registers-per-thread / shared bytes. $`T_{\mathrm{SM}}, W_{\mathrm{SM}}, R_{\mathrm{SM}}, S_{\mathrm{SM}}`$: the SM's hardware caps on threads / warps / registers / shared memory.

### Memory access: sample warps, count sectors

Coalescing depends on the actual index expression. At each load/store site we **evaluate the index for each of 32 warp lanes**, then count how many distinct 32-byte DRAM sectors those addresses fall into.

```
  32 warp lanes
  ┃│││││││││││││││││││││││││││││││┃        eval index(lane)  ↓ ↓ ↓ …

  coalesced load
  ┃         1 sector             ┃         32 lanes → 1×32B

  strided load
  ▭   ▭   ▭   ▭   ▭   ▭   ▭   ▭            8+ sectors → 8×32B

  Same 32 threads, very different bytes moved.
```

$$
\sigma = \Bigl| \bigl\lbrace \lfloor a(\ell)/32 \rfloor : \ell = 0, \dots, 31 \bigr\rbrace \Bigr|
$$

$$
B_{\mathrm{txn}} = \overline{\sigma} \cdot 32\mathrm{B} \cdot N_w
$$

> $`\sigma`$: distinct 32B sectors touched by one sampled warp. $`a(\ell)`$: byte address of lane $`\ell`$. $`\overline{\sigma}`$: mean across sampled warps. $`N_w`$: exact dynamic warp-site count from the enclosing loops and grid.

Warps are sampled deterministically (seeded by module hash + site) so the estimate is repeatable. Non-analyzable index expressions get a configured worst-case sector count.

### Compute: interpret the SSA graph

Two numbers come from one walk over the KIR SSA. **Critical path** — the longest dependency chain, in cycles — bounds latency. **Weighted op count** — sum of every dynamic op's issue weight — bounds throughput.

```
  load a   [load 400] ━━━━┓
                          ┣━━▶ [mul 6] ━━━▶ [add 4] ━━━▶ [store 4]
  load b   [load 400] ────┛                    ▲
                                    [const 1] ─┘

  ready @ 400        400 + 6 = 406    406 + 4 = 410    410 + 4 = 414

  ┌────────────────────────────┐  ┌────────────────────────────┐
  │ critical path (latency)    │  │ weighted op count (issue)  │
  │ longest chain of ━━ arrows │  │ Σ each op's cycle-weight   │
  │ = 414 cycles               │  │ × dynamic instance count   │
  └────────────────────────────┘  └────────────────────────────┘
```

$$
C_L = W \cdot \kappa
\qquad
C_I = \frac{\sum_i w_i\, n_i}{\rho}
$$

> $`\kappa`$: critical-path cycles for one block. $`W`$: number of block waves. $`w_i, n_i`$: per-op cycle-weight and dynamic instance count. $`\rho`$: GPU issue rate (ops/cycle). Loops multiply the body's chain by trip count; global loads divide their latency by the hiding factor $`h = \min(a_w, s_w)`$ (active vs. saturating warps); `sync` adds a sync latency.

### Aggregate: roofline max + launch

Three independent bottlenecks: latency, DRAM bandwidth, issue rate. The kernel runs at whichever is tightest — the roofline max. Then we add a launch-overhead tier based on how much of the GPU the grid fills.

```
  latency    ▭▭▭▭▭▭▭▭▭▭▭▭▭▭            = W · κ
  bandwidth  ▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭▭┊   = B_txn / β
  issue      ▭▭▭▭▭▭▭▭▭▭            ┊   = Σ wᵢnᵢ / ρ
                                   ┊
                       max = raw_cycles

  [launch][ raw_cycles ]  = total_cycles(a)
```

$$
C(a) = L(G) + \max\bigl( C_L,\ C_B,\ C_I \bigr)
$$

> $`C(a)`$: total estimated cycles for candidate $`a`$. $`G`$: grid block count. $`L(G)`$: one of three launch-overhead tiers — fits-one-SM, fits-one-wave, or multi-wave. Results are memoized by (module hash, param bindings hash); structurally identical candidates only cost estimator time once.

---

## CP-SAT extraction program

Pick a closed subgraph of alternatives that minimizes predicted runtime. The problem is small (a few thousand binary variables) and hands to CP-SAT as a lexicographic sequence of four linear objectives.

### Sets

$$
\begin{aligned}
A &: \text{candidate alternatives (nodes in the alt graph)} \\
V &: \text{value classes (circles in the alt graph)} \\
M &: \text{distinct compiled modules (ArtifactKeys)} \\
S &\subseteq V : \text{graph inputs} \\
D &\subseteq V : \text{demanded outputs} \\
P(v) &= \lbrace a \in A : v \in \mathrm{outputs}(a) \rbrace \\
R(a) &\subseteq V : \text{inputs required by } a \\
K(a) &\in M : \text{module } a \text{ compiles to}
\end{aligned}
$$

### Decision variables

$$
x_a,\ y_v,\ z_m \in \lbrace 0, 1 \rbrace
$$

- $x_a$ — candidate $a$ selected
- $y_v$ — value $v$ materialized
- $z_m$ — module $m$ compiled

### Constraints

$$
\begin{array}{rll}
(1) & y_v = 1 & \forall\, v \in S \cup D \\
(2) & \displaystyle\sum_{a \in P(v)} x_a = y_v & \forall\, v \in V \setminus S \\
(3) & x_a \le y_v & \forall\, a \in A,\ v \in R(a) \\
(4) & x_a \le z_{K(a)} & \forall\, a \in A \\
(5) & z_m \le \displaystyle\sum_{a\, :\, K(a) = m} x_a & \forall\, m \in M \\
(6) & \displaystyle\sum_{m \in M} z_m \le M_{\max} & \text{(optional)} \\
(7) & \displaystyle\sum_{m \in M \setminus M_{\mathrm{orig}}} z_m \le M_{\mathrm{new}} & \text{(optional)}
\end{array}
$$

1. Graph inputs and demanded outputs are pinned on.
2. Each materialized value has *exactly one* selected producer. This also encodes "pick a candidate and you get all its outputs": a multi-output $`x_a`$ appears in this sum for each of its outputs, so selecting it forces every $`y_v`$ in $`\mathrm{outputs}(a)`$ to 1.
3. A selected candidate needs all its inputs materialized.
4. Selecting a candidate forces its module to compile.
5. A module is compiled only if at least one candidate using it is selected — together with (4), $`z_m`$ is the OR of the $`x_a`$'s that compile to $`m`$.
6. Optional: cap total compiled modules.
7. Optional: cap new modules beyond what the original graph already needs.

> No acyclicity constraint is needed: the alternative graph is a DAG by construction and every subgraph of a DAG is a DAG.

### Objective — solved as a sequence of four linear minimizations

$$
r(a) = \max\left( 1,\ \left\lfloor \frac{C(a)}{q} \right\rceil \right)
$$

where $`C(a)`$ comes from the cost model and $`q`$ is the cycle quantum.

**Stage 1**

$$
R^{\star} = \min \sum_{a \in A} r(a)\, x_a
\qquad\text{then lock}\qquad
\sum_{a} r(a)\, x_a \le R^{\star} + \left\lceil \frac{R^{\star} p}{10^6} \right\rceil
$$

Primary target: predicted runtime. The lock's slack $`p`$ is `runtime_tolerance_ppm` — allows a tiny runtime concession for a big compile-time saving next stage.

**Stage 2**

$$
Z^{\star} = \min \sum_{m \in M} z_m
\qquad\text{then lock}\qquad
\sum_{m} z_m \le Z^{\star}
$$

Compile count: one `nvcc` invocation per selected module. Proxy for build time until per-module compile cost is calibrated.

**Stage 3**

$$
N^{\star} = \min \sum_{a \in A} x_a
\qquad\text{then lock}\qquad
\sum_{a} x_a \le N^{\star}
$$

Graph size: fewer selected candidates means less launcher and planner work at runtime.

**Stage 4**

$$
Y^{\star} = \min \sum_{v \in V} y_v
$$

Materialized-value count: a peak-memory proxy. Inputs and demanded outputs are pinned by (1), so this only prunes optional intermediates.

> Each stage is a separate CP-SAT solve; the previous stage's solution is passed to the next as a hint. If any stage returns `Unknown` or `Infeasible` within the wall-time limit, we fall back to the seed prefix (the unfused graph is always feasible) and report the reason.