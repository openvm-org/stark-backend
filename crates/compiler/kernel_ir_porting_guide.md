# Porting blackbox kernels to structured `ir::Module`s

This guide is the sequel to `gpu_ir_porting_guide.md`. Where that guide
covers wrapping an existing CUDA kernel as a `GraphNode::BlackboxKernel`,
this one covers moving from that blackbox to a **structured**
`GraphNode::Kernel` whose body is a DSL `ir::Module` — the higher-level
functional IR built through
[`crypto_compiler::ir::IRBuilder`](src/ir.rs) that the compiler
lowers, layouts, fuses and emits as CUDA C++.

The running example is the fractional-GKR kernel family in
`crates/cuda-backend/src/logup_zerocheck/`: eight kernels from
`gkr.cu` are ported into structured modules in
`fractional_ir_dsl.rs`, tested pairwise against the eager
`_frac_*` / `fold_ef_*` blackbox launches. The rest of this document
distills the concrete decisions and gotchas from that port.

## When to port

**Port** a kernel when its per-thread body is:

- a straight elementwise map — an `A[i] op B[i] op ...` pattern with
  either no reduction or a single reduction over one axis;
- a `compute [N] |i| { reduce [K] |k| { ... } }` shape (the canonical
  GKR compute-round shape);
- a fixed-arity chain of pure field-extension operations (add, sub,
  mul, inverse), with challenges either as bake-in host constants
  (`alpha`) or `[D_EF] BabyBear` device buffers (`lambda`, `r`).

Concrete anchors:

- `build_extract_root_module` (`fractional_ir.rs:299`) — a two-output
  scatter/gather with no reduction.
- `build_reduce_to_single_evaluation_module` (`fractional_ir.rs:1186`)
  — pure elementwise `FpExt` arithmetic with a lifted challenge scalar.
- `build_frac_compute_round_module` (`fractional_ir_dsl.rs`) — the
  canonical `compute [2] |i| reduce [num_x/2] |idx| { ... }` shape.

**Do NOT port** — keep as a blackbox — when the kernel uses any of:

- **Virtual/compact addressing.** The fractional-GKR kernels have a
  `virtual_mode` branch that reads through
  `virtual_node_value(layer, dense_idx, active_size, real_len,
  logical_len, alpha)`: a bit-reverse of the index plus a runtime
  `start >= real_len` guard, with a spill-slot fallback. The bit-reverse
  is not a quasi-affine expression, so the DSL's index-expression
  checker rejects it. All eight ports here handle **only the dense
  branch** (`real_len == logical_len`); the virtual branch stays a
  blackbox in `fractional_ir.rs` and is still what `fractional_sumcheck_gpu_ir`
  uses.
- **Warp-shuffle primitives** or explicit shared-memory tiling. The DSL
  emits register/shared layouts through its layout inference pass, and
  cannot express `__shfl_xor_sync` or hand-rolled `extern __shared__`
  striping directly. Kernels like `precompute_m_build_partial_kernel`
  (2-D thread blocks + tiled shared-memory loads) stay blackbox for this
  reason — see "Kernels that remain blackbox" below.
- **Dynamic shared memory** or **2-D thread blocks** (`dim3(m, m)` with
  `m = 1 << w`). The current DSL codegen assumes 1-D thread blocks with
  a compile-time-chosen size.

## The `ir::Module` shape

A structured kernel module is an
[`IRBuilder`](src/ir.rs) program terminated by
`b.finish(name, body)`. Its shape mirrors the design.md canonical form:

- **Inputs.** Declared with
  `b.input("name", ScalarType::{BabyBear|FpExt|U32|Bool}, shape)` where
  `shape` is a `Vec<usize>`. Each declaration becomes a module input at
  a fixed 0-based index and returns a `NodeId` referring to that tensor.

- **Elementwise arithmetic.** `b.add / sub / mul / select / lt / eq /
  rem / div / and / or / xor` on `NodeId`s. Operates on the current
  scalar type (BabyBear or FpExt propagates through arithmetic; `select`
  works on scalars of any type). Both BabyBear and FpExt participate in
  reduce (see below); `LiftFpExt(x)` (`b.lift_fpext(x)`) promotes a
  BabyBear scalar to `x + 0·t + 0·t² + 0·t³ ∈ FpExt`.

- **Indexing.** `b.index(tensor, &[i, j, k, ...])` fully indexes a
  tensor down to a scalar. Every index expression must be quasi-affine
  over the enclosing compute variables — the compiler rejects arbitrary
  `select`-of-index or bit-reverse.

- **Parallel map.** `b.compute(bound, |b, i| body(b, i))` — produces a
  tensor of size `bound` (or `[bound, ..inner]` if `body` returns a
  tensor). At the top level, one `compute` becomes one CUDA kernel;
  nested `compute` becomes a per-thread inner loop.

- **Reduction.** `b.reduce_add(bound, |b, i| body)` (or the more
  general `b.reduce(op, bound, ...)`) — produces a scalar. Native
  BabyBear and FpExt reductions were both added recently; the
  compiler emits a `ConstField(0)` or `ConstFpExt([0,0,0,0])` seed
  based on the accumulator type (see
  `crates/compiler/src/passes/lower_to_kir.rs:234`).

- **Scalar hoisting.** `b.let_bound(value)` binds a computation once
  and returns a variable node referring to it; used to lift work
  that would otherwise run per-thread. Every `let_bound` binding is
  materialized as a top-level `let` chain around the body passed to
  `b.finish`.

- **Multi-output modules.** Return `b.tuple(&[out0, out1, ...])` from
  the top-level compute (or as the module body). Each tuple entry maps
  to one output buffer at the same index. See
  `build_reconstruct_s_evals_module` (`fractional_ir.rs:1379`) for a
  six-output example.

- **Packing scalars into a tensor row.** `b.pack(&[x0, x1, ...])`
  builds a `T[k]` from `k` scalars of the same element type — used
  when a `compute [N]` returns a row of the output. Both fold-column
  ports pack `[p, q]` into a `[2] FpExt` inner row so the total output
  shape is `[N, 2] FpExt`.

## Frac<EF> and challenge buffers: layout conventions

A `Frac<EF>` element is `(p: EF, q: EF)` = two 16-byte FpExt scalars
= 32 bytes. It matters both for buffer allocation and for how a
module reads it.

**Frac<EF> allocation.** `add_frac_ef_buf(g, device, name, n)` in
`fractional_ir.rs:98` uses `elem_size = 32` and byte size `32 * n`.
The `elem_size` is only used for the memory planner's offset alignment;
the module binding sees the flat byte size (see
`crypto_compiler::graph_exe::validate_module_binding`,
`graph_exe.rs:1114` — it checks `sizes[b.0] == module.input_size(i)`).

**Frac<EF> viewed as `[n, 2] FpExt`.** Every module in this port that
reads Frac data binds it as
`b.input("...", ScalarType::FpExt, vec![n, 2])`. Row `i` is one
`Frac<EF>`; column 0 is `p`, column 1 is `q`. Byte size:
`n * 2 * 16 = 32 * n` — matches the buffer. See
`build_fold_ef_frac_columns_module` for a straightforward example
(reads `src[a_idx, 0/1]`, packs `[out_p, out_q]`, produces
`[out_len, 2] FpExt`).

**Frac<EF> viewed as `[n, 2, D_EF] BabyBear`.** Same 32 bytes per row
= 8 base-field u32s. Used by `build_frac_build_tree_layer_revert_module`
because the DSL's `ef_inverse_coeffs`
(`crates/compiler/src/field_ext.rs:100`) takes four *base-field*
coefficients, not an FpExt scalar. The BabyBear coefficients are
read directly, inverted, then reconstituted into an FpExt scalar
via `fpext_from_coeffs`.

**Transcript challenges: `[D_EF] BabyBear`.** `sample_ext(g)` returns a
`BufId` allocated by `add_ext_scalar_buf` (`fractional_ir.rs:135`):
byte size `D_EF * 4 = 16`, `elem_size = 16` (16-byte aligned so the
buffer can also be bound as `[1] FpExt` when a module wants
FpExt-scalar loads). Inside a module the standard idiom lifts these
four BabyBear coefficients into a single `FpExt` scalar and hoists it
once per launch:

```rust
fn bind_challenge_as_fpext(b: &mut IRBuilder, name: &str) -> NodeId {
    let x = b.input(name, ScalarType::BabyBear, vec![D_EF]);
    let coeffs = load_ext_coeffs(b, x);        // 4 BabyBear scalars
    let combined = fpext_from_coeffs(b, coeffs); // recombine {1, t, t², t³}
    b.let_bound(combined)                      // hoist above the compute
}
```

`load_ext_coeffs` / `fpext_from_coeffs` live in `fractional_ir.rs` at
lines 1283 and 1292 and are re-exported `pub(crate)` for the DSL
port module. The `let_bound` around `combined` is important: without
it, the four-slot lift and the recombination would fire once per
thread instead of once per launch.

## `insert_kernel` vs `insert_blackbox_kernel`

The choice is orthogonal to correctness but changes what the compiler
can do:

- `insert_blackbox_kernel` — opaque closure. The graph node has
  declared input/output arity and modifies-flags, but the closure's
  body is a function pointer the graph runtime calls. The compiler
  never inspects the kernel body, so no fusion / layout / dedup.

- `insert_kernel` — passes an `ir::Module` (via
  `Into<Arc<ir::Module>>`). The compiler content-hashes the module
  (`GraphBuilder::insert_kernel` in `graph_ir.rs:249` folds duplicates
  onto one `Arc`), JIT-compiles it once through the
  `GraphCompiler`, runs its layout / fusion / codegen passes, and can
  in principle reuse the same artifact across many kernel nodes.

Reach for `insert_kernel` when you're already writing structured
code (a compute + reduce) and want the compiler to layout it. Keep
`insert_blackbox_kernel` for the escape-hatch cases: virtual/compact
addressing, warp-shuffle-heavy code, or interop with an existing
handwritten `.cu` file. In this port, the `_ir_dsl` inserters all use
`insert_kernel`; the `_ir` / `_ir_bufid` inserters (still in
`fractional_ir.rs`, used by `fractional_sumcheck_gpu_ir`) all use
`insert_blackbox_kernel`.

## The `alpha` carve-out: host EF constants baked into modules

`alpha` is fixed for a whole prover invocation, does not depend on
any kernel output, and is safely captured by value at graph-build
time (see gpu_ir_porting_guide.md Principle 4's carve-out). Bake it
into a module as an `FpExt` const:

```rust
use p3_field::{BasedVectorSpace, PrimeField32};

fn ef_canonical_coeffs(v: EF) -> [u32; 4] {
    let cs = v.as_basis_coefficients_slice();
    [cs[0].as_canonical_u32(), cs[1].as_canonical_u32(),
     cs[2].as_canonical_u32(), cs[3].as_canonical_u32()]
}

let alpha_e = b.const_fpext(ef_canonical_coeffs(alpha));
```

Pass the **canonical** BabyBear u32 coefficients — the compiler's
codegen converts to Montgomery internally (see
`build_reconstruct_s_evals_module`, which uses `const_fpext([1,0,0,0])`
etc. for the constants 1, 2, 3, 5). Do **not** pre-Montgomery-encode
the coefficients yourself.

In the dense case none of the fold/multifold/revert modules here
actually consume `alpha` (it only enters via `virtual_node_value`,
which the dense branch skips). But other kernels in the family do —
`_frac_precompute_m_build`'s optional `apply_alpha` step, for
instance.

## Test recipe

Every module in this port has a paired test in
`fractional_ir_dsl.rs`'s `dsl_port_tests` module. All follow the same
"compare bytes against the eager kernel" shape:

1. **Deterministic random input.** `make_host_leaves(len, seed)`
   generates a `Vec<Frac<EF>>` from an `StdRng::seed_from_u64(seed)`.
   Use per-test seeds so failures reproduce.

2. **Eager reference.** Copy the input to the device, call the
   underlying `frac_*` safe wrapper directly on a stream, sync, read
   back to host.

3. **DSL side.** Build a `GraphBuilder`, stage every input as a
   `ConstBuf::HostBuf` (`frac_const_buf`, `ef_slice_const_buf`,
   `ef_const_ext_scalar_buf`), insert the module, memcpy the output
   into a fresh reader-less buffer so it becomes a graph output,
   compile with `GraphCompiler::new().scheduler(SchedulerMode::Heuristic)`
   (heuristic scheduler runs in ms; the CP-SAT one can time out on
   larger graphs — but for these single-module tests either works),
   read back.

4. **Byte-equality.** Compare using `frac_bytes(&want)` slices and
   `assert_eq!`. Every DSL emit is bit-identical to the CUDA kernel:
   the compiler uses the same Montgomery layout and the same reduction
   order.

Keep sizes small (`num_x = 32..256`, `w ∈ [2, 5]`, `tail_size ∈ [4,
16]`). At `n = 4..5` (num_x = 32..64), a single test compiles and
runs in ~1–3 s.

**Note on the `dev_challenge` reference path.** For the challenge-
carrying kernels (compute_round, compute_round_and_fold, ...) the
DSL module and the eager `_dev_challenge` variant read the same
`[D_EF]` BabyBear buffer. They should agree byte-for-byte with the
non-dev-challenge variant on the same input. Existing test
`dev_challenge_entry_points_match_host_value` in
`fractional_ir.rs:3611` asserts the eager side; the DSL tests here
compare against the by-value variant (they use the eager
`frac_compute_round(&eq_host, ..., lambda_by_value, ...)` path
since it's simpler to set up).

## Gotchas we actually hit

- **`elem_size` doesn't need to match the module scalar size.** The
  validate-binding check (`graph_exe.rs:1114`) is only on total byte
  size. A `Frac<EF>` buffer allocated with `elem_size = 32` can be
  bound as `[n, 2] FpExt` (per-element size 16, total `32n`) or as
  `[n, 2, D_EF] BabyBear` (per-element size 4, total `32n`). The
  compiler's memory planner uses `elem_size` only for offset
  alignment, and 32 is a valid alignment for both views.

- **Virtual mode is unexpressible.** The `virtual_node_value(layer,
  dense_idx, active_size, real_len, logical_len, alpha)` addressing
  in `gkr.cu` bit-reverses `dense_idx` under a runtime `active_size`
  before offsetting by `subtree_len`. Neither the bit-reverse nor
  the `if (start >= real_len)` guard is a quasi-affine index
  expression. This is a hard blocker: **skip the virtual branch, port
  only the dense case, note it in a module-level comment.** The
  eager blackbox path handles virtual mode.

- **In-place kernels become out-of-place.** DSL modules are pure; a
  kernel that mutates its input must be ported as a function of the
  input producing a fresh output. The `fold_..._inplace` /
  `multifold_inplace` / `compute_round_and_fold_inplace` variants all
  vanish — the same fold module handles both, and the memory planner
  can (in principle) alias the output back to the input. This
  matters at the caller side: `frac_compute_round_and_fold_ir_dsl`
  takes both a `src_pq_buffer` and a fresh `dst_pq_buffer`.

- **Two-borrow errors on `b.eq(x, b.const_u32(0))`.** The `IRBuilder`
  methods borrow `&mut self`; passing another builder call as an
  argument violates the borrow checker. Bind the constant first:
  ```rust
  let zero_u = b.const_u32(0);
  let cond = b.eq(x, zero_u);
  ```
  Same for any nested `b.foo(..., b.bar(...))`.

- **`with_rev_bits` is a compile-time bit pattern, not a runtime
  computation.** The CUDA `with_rev_bits(idx, size, hi, lo)` looks
  data-dependent but `hi` and `lo` are compile-time constants (either
  0 or 1 in the compute-round kernel). Encode it in the DSL by
  hoisting the compile-time offset out and using `b.or(idx, offset)`
  (works because `idx < pq_size/4` has its top two bits zero, so
  `or` is `+` on disjoint bits — a quasi-affine expression the DSL
  accepts). See `with_rev_bits_dsl` in `fractional_ir_dsl.rs`.

- **Multiple `reduce` bodies over the same domain share work.** The
  precompute-M-eval-round module writes to two output slots via a
  single `compute [2] |out_i| reduce [total] |idx| { ... }`; the
  entire per-`idx` prelude (weight computation, four `m_total` loads,
  the s1/s2 candidate values) is hash-consed once and only the final
  `select(is_s1, ...)` differs. This kept the emitted kernel a single
  reduction rather than two loops.

- **FpExt inversion needs base-field coefficients.**
  `crypto_compiler::field_ext::ef_inverse_coeffs`
  (`crates/compiler/src/field_ext.rs:100`) takes and returns four
  `NodeId`s of type BabyBear, not one FpExt scalar. To invert an
  FpExt value inside a module you need its coefficients — either
  bind the source buffer as `[N, 2, D_EF] BabyBear` (as the tree-
  revert module does) or otherwise recover coefficients from the
  transcript-shape input (as the eq-hypercube stage module does with
  `load_ext_coeffs`). There is no `ef_inverse(scalar_fpext)`
  primitive today.

- **Content-hash dedup collapses structurally identical modules.**
  `GraphBuilder::insert_kernel` folds modules by content hash
  (`graph_ir.rs:270`), so two `build_frac_multifold_module(4, 2)`
  calls produce the same JIT'd artifact. The size/`w`/`t` /
  `eq_low_cap` shape parameters *do* differentiate module names —
  the format string in `b.finish(format!("... _n{num_x}_c{eq_low_cap}"),
  ..)` disambiguates so distinct shapes compile as distinct kernels.

- **Reduce-of-select vs select-of-reduce.** Emitting the compute-round
  module as `compute [2] |i| { reduce [K] |k| { select(i, ...) } }`
  hash-cons-shares the reduction domain across the two outputs. The
  alternative `select(i, reduce ..., reduce ...)` — which would look
  more natural — is illegal in canonical form (a `reduce` cannot be
  the operand of a `select` inside a `compute`; canonicalize would
  reject it). Prefer the compute-outer-select-inner form.

## Kernels that remain blackbox

Even with all eight kernels in this port ported, these stay blackbox
in `fractional_ir.rs`:

- **`bit_rev_frac_ext_build_k2`** and **`bit_rev_frac_ext`.** The
  bit-reverse index isn't quasi-affine — the mapping from the linear
  loop index to the target buffer index is `idx -> reverse_bits(idx,
  n_bits)`, which no polynomial expression captures.

- **`frac_precompute_m_build`
  (`precompute_m_build_partial_kernel`).** 2-D thread blocks (`dim3(m,
  m)`) with `m = 1 << w`, dynamic shared memory (5 arrays of shape
  `[m+1, BATCH]` — the +1 is bank-conflict padding), and a
  three-phase pipeline (weight load / pq load-and-premul / matrix
  accumulate) all coordinated by `__syncthreads()`. This is
  structurally beyond the current DSL — the pass pipeline in
  `crates/compiler/src/passes/` assumes 1-D thread blocks and fixed
  shared-memory allocations.

- **`_frac_add_alpha`** — trivially portable in principle, but it's a
  tiny elementwise mutation over a *slice* of a larger buffer (the
  second half of the input layer). The DSL has no "read/write from
  offset X inside a buffer" primitive today; the buffer would need
  to be treated as a whole, materialize a fresh output, and rely on
  the planner aliasing it back. Blackbox is one line of code, and
  the kernel launches only twice per prover run.

- **The virtual-mode branches of every ported kernel.** Documented at
  the top of `fractional_ir_dsl.rs`: every DSL module assumes
  `real_len == logical_len`. The `_ir_bufid` blackbox wrappers in
  `fractional_ir.rs` are what `fractional_sumcheck_gpu_ir` calls in
  practice, so the DSL ports here are enablement-only until the
  driver switches over on a per-round basis (or the DSL gains a
  bit-reverse-plus-guard idiom).

## Where to look

- Each `build_<name>_module` in `fractional_ir_dsl.rs` starts with a
  block comment quoting the eager CUDA loop and the shape parameters
  it takes.
- Test cases in `dsl_port_tests` scan a few sizes each; the seeds are
  called out explicitly so failures reproduce byte-for-byte.
- The existing `build_reduce_to_single_evaluation_module` /
  `build_reconstruct_s_evals_module` / `build_eq_hypercube_stage_module`
  in `fractional_ir.rs` are the "already living in production" DSL
  modules to compare against — they use the same challenge-lift
  idiom and the same module-cache pattern.

