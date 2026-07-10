# Static Memory Planner (`static_alloc`)

## Motivation

The GPU prover allocates buffers with heavily skewed sizes — from single-element sums to
multi-gigabyte LDE matrices — through the dynamic memory manager ([VPMM](./vpmm_spec.md)). VPMM
fights fragmentation at run time by remapping pages, which costs work, makes peak usage an emergent
property of allocator state, and makes pointer values non-deterministic across runs. For workloads
like OpenVM where, once the trace heights are known, **every** downstream allocation size and
lifetime is a deterministic function of `(heights, widths, quotient degrees, FRI params)`, we can
instead plan GPU memory statically:

- declare all allocations of a proof (or of a phase) together with their lifetimes up front;
- build an interference graph from those lifetimes and run a packing pass that assigns every
  allocation a fixed byte offset inside one workspace, minimizing the total workspace size;
- at run time, hand out buffers at the planned offsets — no allocator behavior at all, no
  fragmentation, and stable pointers (a prerequisite for CUDA graph replay, where captured kernel
  launches bake in device addresses).

The design prefers **explicit over implicit**: the programmer declares each phase's buffers and
lifetimes in code, and the plan is inspectable (`StaticAllocator::describe()`).

## Two-stage usage

### Stage 1: setup — declare allocations with RAII handles

Each phase of the computation is represented by a struct that declares its allocations in a
`setup`-style constructor taking an `&AllocBuilder`:

```rust
struct QuotientPhase {
    accumulator: AllocIdx<F>, // temp, internal to the phase
    selectors: AllocIdx<F>,   // temp, internal to the phase
    chunks: AllocIdx<F>,      // output, used downstream
}

impl QuotientPhase {
    fn setup(builder: &AllocBuilder, qsize: usize) -> (Self, DeviceBufFake<F>) {
        let accumulator = builder.alloc_fake_labeled::<F>(4 * qsize, "q_acc");
        let selectors = builder.alloc_fake_labeled::<F>(4 * qsize, "q_sel");
        let chunks = builder.alloc_fake_labeled::<F>(4 * qsize, "q_chunks");
        let this = Self {
            accumulator: accumulator.idx(),
            selectors: selectors.idx(),
            chunks: chunks.idx(),
        };
        (this, chunks)
        // accumulator + selectors fakes drop here: their space becomes
        // reusable by later phases. chunks is returned; the caller decides
        // when the output's lifetime ends by dropping the fake.
    }
}
```

`AllocBuilder::alloc_fake::<T>(len)` returns a `DeviceBufFake<T>`: an RAII handle that owns **no
memory** — its Rust lifetime *is* the declared lifetime of the future allocation. The builder keeps
a logical clock; each `alloc_fake` records a birth tick, each fake drop records a death tick. Two
allocations *interfere* (may not share memory) iff their `[birth, death)` intervals overlap — plus
the stream rule below. Rust's ordinary ownership rules therefore build the interference graph:
return a fake from `setup` to extend a buffer's life into later phases, drop it (explicitly or by
scope end) to end it.

`DeviceBufFake::idx()` yields an `AllocIdx<T>` — a small `Copy` handle (think: a symbolic pointer)
stored in the phase struct and redeemed for the real buffer in stage 2.

**Footgun to know about**: `builder.alloc_fake::<F>(n).idx()` creates and immediately drops the
fake (Rust drops the temporary at the end of the statement), declaring a buffer live for a single
tick. Buffers used simultaneously in the run stage must be declared by fakes that are alive
simultaneously in the setup stage — bind them to locals. Misdeclarations are caught at run time
*while the guards involved are held simultaneously* (see below). What is not detectable: releasing
a guard, letting an overlapping allocation use the space, then reacquiring the same index expecting
its old contents — `get` succeeds at the same pointer and the prior contents are simply gone (the
allocator cannot distinguish "reacquire to rewrite", which is the normal replay pattern, from
"reacquire expecting persistence"). The `touchemall` feature closes this gap for testing: it
poisons every planned region with `0xff` on each acquisition, so stale-read bugs surface
immediately.

### Stage 2: run — build once, then acquire at fixed offsets

```rust
let ctx = GpuDeviceCtx::for_current_device()?;
let alloc: StaticAllocator = builder.build_on(&ctx)?; // packs + allocates workspace

// in each phase:
let mut acc = alloc.get(&self.accumulator)?;   // StaticDeviceBuffer<F>
let mut chunks = alloc.get(&self.chunks)?;
// ... launch kernels: both deref to &DeviceBuffer<F> ...
drop(acc); // releases the planned region (the memory is not freed)
```

`build_on(&GpuDeviceCtx)` runs the packing pass, allocates one workspace buffer through the normal
memory manager on the given stream, and returns a `StaticAllocator` bound to that stream. `get` returns a `StaticDeviceBuffer<T>` — an RAII
view at the planned offset that derefs to `DeviceBuffer<T>` (shared access), so kernels and
device-to-host copies work unchanged; host-to-device copies go through the inherent
`copy_from(&mut self, &[T], &GpuDeviceCtx)` method. (The guard deliberately does not implement `DerefMut`:
`&mut DeviceBuffer<T>` would let safe code `mem::replace` the non-owning view out of the guard and
free workspace-interior memory on its drop.) Dropping the view releases the planned region;
acquiring the same `AllocIdx` again returns the **same pointer** (stable across acquire/release
cycles and proof rounds — the property CUDA graphs need).

The whole schedule can be replayed every proof round with zero allocator work: `get` is pointer
arithmetic plus a liveness check under a mutex.

## Packing algorithm

Offline interval placement (offline dynamic storage allocation) is NP-hard; the planner uses the
classic *greedy-by-size* heuristic, which is near-optimal on skewed, phase-structured workloads:

1. Sort allocations by decreasing size (ties: birth tick, then declaration order — the plan is
   fully deterministic).
2. For each allocation, collect the `[offset, offset + size)` ranges of already-placed allocations
   that interfere with it, and place it at the lowest 256-byte-aligned offset that fits in a gap
   (first fit).

The workspace size is compared against the *peak-live* lower bound (max over time of the sum of
live sizes — the weight of the heaviest clique of the interval graph); `build_on` logs both. On the
prover-shaped benchmark schedule the greedy pass lands within a few percent of the lower bound.

Alignment: every offset is at least 256-byte aligned (the `cudaMalloc` guarantee), raised to
`align_of::<T>()` if larger; `build_on` verifies the workspace base pointer satisfies the largest
requested alignment.

## Streams

Reusing memory between buffers used on *different* streams would require event synchronization that
the planner does not insert, so allocations with different stream keys **always interfere** (never
share memory). Within one stream key, reuse is safe when lifetime intervals are disjoint, because
work on a single stream executes in issue order — the run stage must therefore issue work in the
same order as the declared lifetimes. `alloc_fake` uses stream key 0; `alloc_fake_on_stream` declares work issued elsewhere.

An allocator is bound to the `GpuDeviceCtx` it was built with: the workspace lives on that stream,
fallback allocations use it, and planned reuse relies on work being issued in declared-lifetime
order on it — the same discipline every explicitly-allocated `DeviceBuffer` already follows. Using
a planned buffer on a foreign stream requires the caller to synchronize, exactly as with any other
device buffer.

## Use-after-lifetime detection

The planner trusts the declared lifetimes for packing, but the allocator *checks* them at run time:
it tracks which planned regions are currently acquired, and `get(b)` fails if any live buffer's
planned region overlaps `b`'s. Since interfering allocations never overlap in the plan, any overlap
among live buffers means some buffer outlived its declared lifetime.

Two policies (chosen at `build_with_policy`):

- `ViolationPolicy::Error` (default): `get(b)` returns `StaticAllocError::LifetimeViolation` naming
  the holders. When the offending buffer is dropped, the space becomes acquirable again.
- `ViolationPolicy::WarnAndAllocate`: `get(b)` logs a warning and serves `b` from a fresh dynamic
  allocation (correct but off-plan: `b`'s pointer is not stable for that acquisition).

Double-acquiring an index and redeeming an index against the wrong allocator are errors in both
policies.

## Interaction with the dynamic memory manager

The workspace is a single ordinary `d_malloc` allocation, so a `StaticAllocator` coexists with
dynamic allocation: code outside the plan keeps using `DeviceBuffer::with_capacity` and the VPMM
pool. Because the workspace is page-backed and never freed until the allocator drops, VPMM will
never remap (defragment) pages under it mid-use — pointer stability holds for the allocator's
lifetime.

## Limitations and future work

- **Adoption in the prover.** The current GPU prover shares buffers via `Arc<DeviceBuffer<T>>`
  (`DeviceMatrix`), with lifetimes ended by the last Arc drop. Migrating a phase to the planner
  means expressing that phase's buffers as `AllocIdx` handles and threading a `StaticAllocator`
  through; heights are only known at prove time, so the plan is built (or memoized per
  height-shape) after trace generation.
- The planner covers allocations declared through the builder; small incidental allocations can
  stay dynamic (they bypass the VPMM pool below page size anyway).
- Growing a plan after `build_on` is not supported — build a new plan instead.
- The greedy packer does not currently exploit cross-stream event dependencies to allow safe
  cross-stream reuse.

## Record-and-replay integration (`VPMM_REPLAY`)

The planner is integrated into the memory manager through a record-and-replay mode that requires no
prover code changes. Record a workload once with `VPMM_TRACE=trace.csv`, then rerun it with
`VPMM_REPLAY=trace.csv`: at startup the manager plans all traced allocations of at least
`VPMM_REPLAY_MIN` bytes (default 16 MiB), lazily allocates one workspace through the pool, and
serves matching allocations at their planned offsets — pointer arithmetic instead of pool work,
with frees reduced to bookkeeping. Correctness does not depend on an exact replay: an allocation is
served only when its size matches the next planned event and its planned range overlaps no live
served allocation; anything else (including a second stream — replay is single-stream by design)
falls back to the dynamic allocator, and divergence counters are logged.

On the openvm reth workload this replays perfectly (all planned allocations served, zero
fallbacks), making it a faithful way to run real proving on statically planned memory.
