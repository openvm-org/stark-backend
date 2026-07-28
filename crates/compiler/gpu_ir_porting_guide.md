# Porting eager GPU code to the Graph IR

This guide is for porting existing eager CUDA code (the kind that opens a
`GpuDeviceCtx`, launches kernels on `ctx.stream`, and interleaves host
computation) to the Graph IR defined in `graph_ir.rs`. The IR records every
kernel launch, memcpy and memset as a graph node; the whole graph is then
compiled once via `GraphCompiler` and re-run cheaply via `GraphExe`, which
handles memory planning and single-stream ordering for you.

Four principles govern how you should port. Follow them and the resulting
port compiles into a planner-friendly graph that the memory planner
(`planner.rs`) can schedule and pack correctly.

- [Principle 1](#principle-1-one-cuda-kernel-per-insert_blackbox_kernel-no-host-work) —
  one CUDA kernel per `insert_blackbox_kernel`, no syncs or other host
  work inside the closure.
- [Principle 2](#principle-2-wrap-each-kernel-in-a-safe-ir-inserter) —
  wrap each kernel in a safe function that inserts it into the graph.
- [Principle 3](#principle-3-use-insert_memcpy--insert_memset-for-buffer-scale-copies-and-fills) —
  use `insert_memcpy` / `insert_memset` for buffer-scale copies and
  fills, not `cudaMemcpy*` / `cudaMemset*` inside a blackbox.
- [Principle 4](#principle-4-no-data-dependent-host-values) —
  no host values that depend on kernel outputs. Every value produced
  by a device computation must stay on-device as a `BufId`.

The running example throughout is the port of the fractional-GKR
segment-tree build from `crates/cuda-backend/src/logup_zerocheck/fractional.rs`
into `crates/cuda-backend/src/logup_zerocheck/fractional_ir.rs`.

---

## Principle 1: one CUDA kernel per `insert_blackbox_kernel`, no host work

A blackbox kernel node's closure has the signature

```rust
Fn(&[*mut ()], &[*mut ()], cudaStream_t)
```

and is contractually required to launch its work *asynchronously* on the
supplied `stream`. Concretely, that means the closure should:

- Launch **exactly one** underlying CUDA kernel (via its safe wrapper).
- Not synchronize the stream.
- Not read anything back to host memory.
- Not call into the host-side transcript, RNG, math, etc.
- Not decide anything based on runtime kernel output.

### Why

- Multiple kernels in one closure defeat the planner. The planner sees a
  single node with one lifetime and one modifier set, so it can't
  interleave other work between the fused launches, can't reason about
  intermediate buffers, and can't reorder any of it. It also makes buffer
  metadata (`elem_size`, size in `BufInfo`) less meaningful because the
  closure hides the intermediate accesses.
- Host work inside the closure serializes on the stream. `cudaMemcpyAsync`
  device-to-host implicitly syncs; reading a value and branching on it
  stalls every downstream node until the read completes. The graph
  runtime batches launches on one stream for latency; any host round-trip
  breaks that.
- If a step needs a host decision (sample a challenge, check whether
  `root.p == 0`), it belongs in a separate graph node — either
  `insert_kernel` with a structured `ir::Module` or a dedicated blackbox
  that writes its result to a `BufId` a later node can consume.

### Example

**Wrong.** Fuses several launches, embeds a host-side decision, and hides
the intermediate mutation of `layer` from the planner.

```rust
g.insert_blackbox_kernel(
    "segment_tree",
    std::iter::once(layer),
    [root_p, root_q].into_iter(),
    std::iter::once(true),
    move |inputs, outputs, stream| unsafe {
        let layer_ptr = inputs[0] as *mut Frac<EF>;
        run_segment_tree_build(layer_ptr, real_len, total_leaves, alpha, stream);
        extract_root_pq(layer_ptr, outputs[0] as *mut u8, outputs[1] as *mut u8, stream);
        // Host-side branch on a device value — synchronizes the stream.
        let root_p_host = read_ef_from_device(outputs[0], stream);
        if assert_zero && root_p_host != EF::ZERO {
            panic!("root.p != 0");
        }
        run_segment_tree_revert(layer_ptr, real_len, total_leaves, alpha, stream);
    },
);
```

**Right.** One node per kernel; the top-level porting function composes
them.

```rust
bit_rev_frac_ext_build_k2_ir(g, layer, layer_len, real_len, total_rounds as u32, alpha);
let mut i = 2;
while i + 1 < total_rounds {
    let half_i1 = total_leaves >> (i + 2);
    frac_build_tree_two_layers_ir(g, layer, layer_len, half_i1, total_leaves, alpha);
    i += 2;
}
if i < total_rounds {
    frac_build_tree_layer_ir(g, layer, layer_len, total_leaves >> i, total_leaves, false, alpha, false);
}
extract_root_pq_ir(g, layer, root_p, root_q);
frac_build_tree_layer_ir(g, layer, layer_len, 2, total_leaves, true, alpha, false);
```

### Corollary: challenges must flow through `BufId`s

The single biggest reason people are tempted to sync inside a closure is
"I need this sampled challenge to launch the next kernel". Don't. Emit
the sample as its own transcript node (which produces a `BufId`) and pass
the `BufId` into the next kernel wrapper. Kernels that currently take an
`EF` challenge by value need a graph-friendly variant that reads the
challenge from a `BufId`-resident slot; see the module docs on
`fractional_ir.rs` for the full rationale.

---

## Principle 2: wrap each kernel in a safe IR inserter

For every CUDA kernel you port, add a public function that takes a
`&mut GraphBuilder`, whatever `BufId`s the kernel needs, and any scalar
config the kernel needs, and inserts exactly one `BlackboxKernel` node.
Callers compose those inserters — they do not call `insert_blackbox_kernel`
directly.

### Why

- Inserter functions are typed. Callers hand over `BufId`s and typed
  scalars; the inserter is the only place that touches raw pointers and
  `unsafe`.
- Each inserter is one place to encode the `modifies` flags, the input
  and output arity, and the pointer-typing conventions. Callers can't
  get those wrong.
- The pattern makes it obvious when a "kernel wrapper" quietly grows into
  two kernels — it's now two `insert_blackbox_kernel`s in one function,
  which reads wrong. Split it.
- Test coverage naturally lands on the inserter: a test can call the
  inserter, compile the graph, and compare against a host reference that
  calls the underlying safe wrapper directly.

### Shape

An inserter has three responsibilities:

1. Declare the node's inputs / outputs / `modifies` to `insert_blackbox_kernel`.
2. Reconstruct borrowed `DeviceBuffer` views from the raw pointers the
   runtime hands the closure.
3. Call the underlying safe wrapper and `mem::forget` the borrowed views
   so the runtime keeps ownership of the memory.

### Example

The safe wrapper `frac_build_tree_layer(&mut layer, layer_size,
logical_len, revert, alpha, apply_alpha, stream)` becomes:

```rust
#[allow(clippy::too_many_arguments)]
pub fn frac_build_tree_layer_ir(
    g: &mut GraphBuilder,
    layer: BufId,
    layer_len: usize,
    layer_size: usize,
    logical_len: usize,
    revert: bool,
    alpha: EF,
    apply_alpha: bool,
) {
    g.insert_blackbox_kernel(
        "frac_build_tree_layer",
        std::iter::once(layer),        // one input
        std::iter::empty(),            // no dedicated outputs
        std::iter::once(true),         // modifies layer in place
        move |inputs, _outputs, stream| unsafe {
            let mut buf =
                DeviceBuffer::<Frac<EF>>::from_raw_parts(inputs[0] as *mut Frac<EF>, layer_len);
            frac_build_tree_layer(
                &mut buf, layer_size, logical_len, revert, alpha, apply_alpha, stream,
            )
            .expect("frac_build_tree_layer");
            forget(buf);
        },
    );
}
```

Notes:

- Buffer arguments become `BufId`. Scalar config is captured by value.
- `layer_len` (the buffer's element count) is a captured constant used
  only to reconstruct the borrowed `DeviceBuffer` view. It is not the
  same as `layer_size` (the segment tree's current layer's element
  count).
- `modifies=[true]` on the one input tells the planner this node is an
  in-place modifier of `layer`. The planner derives versioned RAW/WAR/WAW
  precedence from node insertion order (see `planner.rs`,
  `PlanCtx::edges`): writers of the same buffer are serialized in
  insertion order, and a pure reader is pinned between the write that
  produced the version it reads and the next overwrite. Composing
  multiple `*_ir` calls that modify the same `BufId` — with snapshot
  reads (e.g. claim extraction) interleaved between them — is safe.
- `mem::forget(buf)` is required. `DeviceBuffer::from_raw_parts` is a
  view; letting it drop would double-free the memory the graph runtime
  owns.

Once the wrapper exists, the porting function is a straight line of
calls:

```rust
frac_build_tree_layer_ir(g, layer, layer_len, total_leaves, total_leaves, false, alpha, true);
frac_add_alpha_ir(g, layer, half, half, alpha);
```

If a kernel's original safe wrapper takes many raw arguments the inserter
signature grows; that is fine. Prefer many named arguments over a struct
until you have three or more inserters sharing an obvious argument
bundle.

### Where to put the file

Colocate the inserters with the eager code they mirror. The convention
used so far is `<module>_ir.rs`: `logup_zerocheck/fractional_ir.rs`
mirrors `logup_zerocheck/fractional.rs`, `sponge_graph_ir.rs` mirrors
`sponge.rs`. Feature-gate the module behind `graph-ir` on the parent
crate so the pure-CUDA build doesn't pull in the compiler dependency.

---

## Principle 3: use `insert_memcpy` / `insert_memset` for buffer-scale copies and fills

`GraphBuilder` exposes first-class `Memcpy` and `Memset` nodes. When you
need to copy an entire buffer to another buffer of the same byte length,
or fill a buffer with a byte pattern, use those nodes. Do not smuggle a
`cudaMemcpyAsync` or `cudaMemsetAsync` inside a blackbox closure to
achieve the same effect.

### Why

- `Memcpy` / `Memset` nodes participate in the planner's dependency graph
  as first-class reads / writes. A `cudaMemcpyAsync` inside a blackbox
  looks like an in-place modification of an input from the planner's
  point of view — so lifetime tracking, output detection ("no readers →
  graph output"), and offset packing are all subtly wrong for the copied
  buffer.
- Graph inputs and outputs are detected by "no writer" / "no reader"
  respectively. A common pattern is to declare a graph input `leaves_in`
  and prepend `insert_memcpy(leaves_in, layer)` so `layer` is a
  scratch-pool buffer the planner is free to pack. Symmetrically, appending
  `insert_memcpy(layer, layer_out)` at the end makes `layer_out` a graph
  output the caller can read after `GraphExe::run`. Neither is possible
  if the copies live inside a blackbox.
- `Memset` node values are byte patterns, checked to be uniform. This
  matches the semantics of `cudaMemsetAsync` and rules out the class of
  bugs where "memset 1" is expected to write `1` to every element but
  writes `0x01010101` to every u32 slot.

### Example — using `insert_memcpy` to plumb graph inputs / outputs

```rust
let leaves_in = add_frac_ef_buf(&mut g, device, "leaves_in", real_len);
let layer     = add_frac_ef_buf(&mut g, device, "layer",     real_len);
g.insert_memcpy(leaves_in, layer);

build_segment_tree_ir(&mut g, &mut sponge, layer, sizes, alpha, false, device)?;

let layer_out = add_frac_ef_buf(&mut g, device, "layer_out", real_len);
g.insert_memcpy(layer, layer_out);
```

`leaves_in` has no writers → planner identifies it as a graph input.
`layer_out` has no readers → planner identifies it as a graph output.
`layer` is a scratch-pool intermediate the planner is free to pack.

### Example — using `insert_memset` to zero-initialize state

```rust
let state_buf = alloc_state_buf(g, device, "sponge_state_init");
g.insert_memset(state_buf, 0);
```

(from `sponge_graph_ir.rs`, initializing the Poseidon2 sponge state to
zero at the start of a transcript).

### When you legitimately can't use them

Whole-buffer semantics only. If you need to copy a *slice* of a buffer —
for example, extract the 16 bytes of `layer[0].p` into a fresh
`[D_EF]`-shaped buffer — `insert_memcpy` isn't applicable (there is no
"partial memcpy" node today) and you fall back to a targeted
blackbox that issues one `cudaMemcpyAsync` per slice. That is what
`extract_root_pq_ir` does. Treat this as an escape hatch, not the
default: the moment you find yourself writing two memcpys in one
closure, ask if you can restructure the source graph to use two separate
whole-buffer copies.

---

## Principle 4: no data-dependent host values

Any value that is a function of kernel outputs — a sampled Fiat-Shamir
challenge, a claim read off a `Frac<EF>` slot, a random `r` interpolated
from observed sumcheck evaluations — must stay on-device as a `BufId`
that flows into the next graph node. It is never read back to a host
`EF` (or `u32`, or `bool`, …) at graph-build time and captured by a
subsequent kernel's closure.

Practically: at graph-build time you only have access to *static* host
values — shape parameters, loop bounds, alpha (if it is a compile-time
constant of the prover), the round index, table sizes. Anything that
depends on the state of a buffer *produced by the graph* is
data-dependent and must be a `BufId`.

Contrast with the eager code: `fractional_sumcheck_gpu` is full of
patterns like

```rust
let root = copy_from_device(&layer, 0, &mut scratch, ctx)?;         // host-side read
transcript.observe_ext(root.p);                                     // host update
let r = transcript.sample_ext();                                    // host sample
frac_compute_round_and_fold(.., lambda, r, alpha, ..)?;             // r captured by value
```

That is fine on the CPU-driving path because every launch happens on
the same stream and `copy_from_device` implicitly syncs. It is
**wrong** in the graph-IR port because:

- Reading `root.p` to a host `EF` would require a D2H sync inside a
  closure ([Principle 1](#principle-1-one-cuda-kernel-per-insert_blackbox_kernel-no-host-work)),
  or a synchronous read at graph-build time (which pins the value at
  compile time, before the graph has even started running).
- `r = transcript.sample_ext()` at graph-build time is meaningless —
  the transcript state at graph-build time is whatever was observed
  *before this build call*, not what the graph will observe when it
  runs. In the graph, `sample_ext(g)` returns a `BufId`; the value it
  ultimately holds is only defined at `GraphExe::run` time.
- Capturing an `EF` challenge by value in an `insert_blackbox_kernel`
  closure freezes it at graph-build time. Any downstream kernel that
  needs the actual runtime value has to read it from a `BufId`.

### Kernels that took `EF` by value need graph-friendly variants

The eager `_frac_*` kernels take challenges as `EF` scalars because the
host has already resolved them. In the graph-IR port those challenges
are `BufId`s. The corresponding `*_ir` inserter must either:

- Wrap a **new** CUDA kernel entry point that takes `x_i` as a
  `*const EF` and loads it on-device (preferred: keeps the launch a
  blackbox and stays 1:1 with an underlying `_frac_*` variant).
- Or emit an `insert_kernel` with a structured `ir::Module` that
  loads the challenge from a `[D_EF]`-shaped `BabyBear` input and
  reconstructs an `FpExt` scalar in the DSL — see
  `build_eq_hypercube_stage_module` in `fractional_ir.rs` for the
  full pattern (`lift_fpext` per coefficient, then recombine against
  `{1, t, t², t³}` via `const_fpext`, `let_bound` so the reconstruction
  fires once per launch rather than once per compute thread).

Neither path is a "read the challenge to host inside the closure and
pass it as an `EF`". That would defeat the purpose of the graph.

### What still can be an `EF` at graph-build time

Constants the caller *has already resolved* before `GraphBuilder`
sees the graph. In the fractional-GKR case that is `alpha` (fixed for
the whole prover), the compile-time `total_leaves` shape, the round
index, the `w` window size, etc. These do not depend on kernel outputs
so capturing them by value in a closure is fine.

The test is: "does this host value depend on any buffer that appears
in the graph?" If yes, it must be a `BufId`. If no (it is a static
prover parameter or shape), captured by value is fine.

### Example — porting the root observe

**Wrong.**

```rust
let root_bytes = /* D2H sync of layer[0] */;
let root = unsafe { *(root_bytes.as_ptr() as *const Frac<EF>) };
transcript.observe_ext(root.p);
transcript.observe_ext(root.q);
```

`root` is a data-dependent host value here — a host `Frac<EF>` synthesized
from a kernel output — and every downstream launch that consumes it
would freeze it at build time.

**Right.**

```rust
let root_p = add_ext_scalar_buf(g, device, "root_p");
let root_q = add_ext_scalar_buf(g, device, "root_q");
extract_root_pq_ir(g, layer, real_len, root_p, root_q);
transcript.observe_ext(g, root_p);
transcript.observe_ext(g, root_q);
```

`root_p` and `root_q` are `BufId`s. `extract_root_pq_ir` is a graph node
whose closure runs at `GraphExe::run` time and populates them from the
layer buffer; `transcript.observe_ext(g, buf)` is a graph node that
consumes them. No host value is ever synthesized from a kernel output.

---

## Checklist for porting a function

1. Enumerate every CUDA kernel the eager code launches. Each one gets a
   `*_ir` inserter in the `_ir.rs` mirror module.
2. Enumerate every whole-buffer copy or zero-fill. Each one becomes an
   `insert_memcpy` or `insert_memset` at the porting call site.
3. Enumerate every host-side decision that consumes a device value
   (branches on `root.p == 0`, "sample a challenge and pass it to the
   next kernel", …). Each one is either a graph node that produces a
   `BufId`, or a follow-up kernel that reads the value from a `BufId`
   slot. It is never a sync-and-branch inside a closure.
4. Write the inserters. Each is one `insert_blackbox_kernel` call, no
   host math in the closure, `mem::forget` all borrowed views.
5. Write the top-level porting function. It should read like the eager
   code with `_ir` suffixes and `BufId`s in place of `DeviceBuffer`s.
6. Test against the eager reference: build the graph, compile it, run
   it, and compare the final buffer bytes (or observed transcript
   values) against the output of the eager function running on the same
   inputs.
