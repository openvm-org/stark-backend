//! Static GPU memory planning.
//!
//! For provers with very large, heavily skewed allocation sizes we want GPU
//! memory laid out statically: all allocations and their lifetimes are
//! declared up front, a packing pass assigns each one a fixed offset inside a
//! single workspace, and the compute code then acquires buffers at those
//! fixed offsets. This avoids fragmentation and non-deterministic allocator
//! behavior entirely (same plan → same pointers, which is also what CUDA
//! graph replay needs), and it makes peak memory a compile-time-style
//! artifact of the plan instead of an emergent property of allocator state.
//!
//! Usage happens in two stages (see `docs/static_alloc_spec.md`):
//!
//! 1. **Setup**: each phase of the computation declares its allocations through an
//!    [`AllocBuilder`], receiving RAII [`DeviceBufFake`] handles. A fake buffer owns no memory —
//!    its Rust lifetime *is* the declared lifetime of the future allocation, and the builder's
//!    logical clock records the birth/death interval of every handle. Keep the handle alive exactly
//!    as long as the real buffer must stay valid, then drop it. [`DeviceBufFake::idx`] yields a
//!    small, copyable [`AllocIdx`] used to retrieve the real buffer later.
//! 2. **Run**: [`AllocBuilder::build_on`] packs the declared allocations into a minimal workspace
//!    (one big device allocation) and returns a [`StaticAllocator`]. Compute code calls
//!    [`StaticAllocator::get`] with an [`AllocIdx`] to receive a [`StaticDeviceBuffer`] — a RAII
//!    view of the planned region that derefs to [`DeviceBuffer`], so kernels and device-to-host
//!    copies work unchanged; host-to-device copies go through [`StaticDeviceBuffer::copy_from`].
//!    Pointers are stable across acquire/release cycles.
//!
//! The allocator also *checks* the plan at run time: while a planned buffer
//! is acquired, acquiring another buffer whose planned region overlaps is a
//! lifetime-contract violation. Depending on [`ViolationPolicy`] this either
//! fails with [`StaticAllocError::LifetimeViolation`] or logs a warning and
//! serves the request from a fresh dynamic allocation.
//!
//! # Declaring lifetimes correctly
//!
//! The planner only knows what the fake handles tell it. In particular,
//! `builder.alloc_fake::<F>(n).idx()` creates *and immediately drops* the
//! fake handle, declaring a buffer that is live for a single tick and whose
//! space is instantly reusable. Buffers that are used simultaneously in the
//! run stage must be declared by handles that are alive simultaneously in the
//! setup stage:
//!
//! ```no_run
//! use openvm_cuda_common::static_alloc::{
//!     AllocBuilder, AllocIdx, DeviceBufFake, StaticAllocator,
//! };
//!
//! struct QuotientPhase {
//!     accumulator: AllocIdx<u32>, // temp, internal to this phase
//!     selectors: AllocIdx<u32>,   // temp, internal to this phase
//!     chunks: AllocIdx<u32>,      // output, used downstream
//! }
//!
//! impl QuotientPhase {
//!     fn setup(builder: &AllocBuilder, quotient_size: usize) -> (Self, DeviceBufFake<u32>) {
//!         // Bind every fake to a local so all three are alive together —
//!         // they are used together in `run`.
//!         let accumulator = builder.alloc_fake::<u32>(4 * quotient_size);
//!         let selectors = builder.alloc_fake::<u32>(4 * quotient_size);
//!         let chunks = builder.alloc_fake::<u32>(4 * quotient_size);
//!         let this = Self {
//!             accumulator: accumulator.idx(),
//!             selectors: selectors.idx(),
//!             chunks: chunks.idx(),
//!         };
//!         // `accumulator` and `selectors` fakes die here: their space is
//!         // free for later phases. `chunks` is used downstream, so its fake
//!         // is returned — the CALLER ends its lifetime by dropping it.
//!         (this, chunks)
//!     }
//!
//!     fn run(&self, alloc: &StaticAllocator) {
//!         let accumulator = alloc.get(&self.accumulator).unwrap();
//!         let selectors = alloc.get(&self.selectors).unwrap();
//!         let chunks = alloc.get(&self.chunks).unwrap();
//!         // ... launch kernels on the buffers; all three deref to
//!         // &DeviceBuffer<u32> ...
//!         let _ = (accumulator.as_ptr(), selectors.as_ptr(), chunks.as_ptr());
//!     }
//! }
//! ```
//!
//! # Streams
//!
//! A [`StaticAllocator`] is bound to the [`GpuDeviceCtx`] it was built with
//! ([`AllocBuilder::build_on`]): the workspace lives on that stream, fallback
//! allocations use it, and planned memory reuse is safe because work on a
//! single stream executes in issue order — the run stage must therefore issue
//! work against the buffers in declared-lifetime order on that stream, the
//! same discipline every explicitly-allocated [`DeviceBuffer`] already
//! follows. Memory is never shared between different declared stream keys
//! ([`AllocBuilder::alloc_fake_on_stream`]): cross-stream reuse would need
//! event synchronization the planner does not insert, and using a planned
//! buffer on a foreign stream requires the caller to synchronize, exactly as
//! with any other device buffer.

use std::{
    fmt,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::Deref,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, PoisonError,
    },
};

use bytesize::ByteSize;
use thiserror::Error;

use crate::{
    copy::MemCopyH2D,
    d_buffer::DeviceBuffer,
    error::{CudaError, MemCopyError, MemoryError},
    memory_manager::d_malloc_on,
    stream::GpuDeviceCtx,
};

mod planner;
#[cfg(test)]
mod tests;

pub use planner::MIN_ALIGN;
use planner::{plan_offsets, PlannedAlloc, Tick, IMMORTAL};

/// Stream key used by [`AllocBuilder::alloc_fake`]: the single logical stream
/// of a program that runs everything on the per-thread default stream.
pub const DEFAULT_STREAM_KEY: u64 = 0;

static NEXT_PLAN_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Error, Debug)]
pub enum StaticAllocError {
    #[error(transparent)]
    Cuda(#[from] CudaError),

    #[error(transparent)]
    Memory(#[from] MemoryError),

    #[error(
        "allocation index belongs to a different plan (expected plan {expected}, got {actual})"
    )]
    PlanMismatch { expected: u64, actual: u64 },

    #[error("allocation '{label}' is already acquired")]
    AlreadyAcquired { label: String },

    #[error(
        "planned space of '{requested}' is still held by {holders:?}: \
         a buffer is being used beyond its declared lifetime"
    )]
    LifetimeViolation {
        requested: String,
        holders: Vec<String>,
    },

    #[error("workspace base pointer is not aligned to {required} bytes")]
    MisalignedWorkspace { required: usize },

    #[error("failed to acquire static allocator lock")]
    LockError,
}

/// Result of a dry-run packing pass ([`AllocBuilder::plan_stats`]).
#[derive(Debug, Clone, Copy)]
pub struct PlanStats {
    /// Total workspace the plan would allocate, in bytes.
    pub workspace_size: usize,
    /// Peak of concurrently live declared bytes (lower bound for any layout).
    pub peak_live_size: usize,
    /// Sum of all declared allocation sizes, in bytes.
    pub sum_of_sizes: usize,
    pub num_allocs: usize,
}

/// What [`StaticAllocator::get`] does when the requested buffer's planned
/// space is still occupied by a buffer that outlived its declared lifetime.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ViolationPolicy {
    /// Fail with [`StaticAllocError::LifetimeViolation`].
    #[default]
    Error,
    /// Log a warning and serve the request from a fresh dynamic allocation.
    ///
    /// This is a diagnostic/recovery mode, not a steady state: the pointer is
    /// NOT the planned one (replay-stability is lost for that buffer) and
    /// data written through the fallback is freed with it — it never reaches
    /// the planned region, so it does NOT survive into a later acquisition of
    /// the same index (the next planned acquisition warns about this).
    WarnAndAllocate,
}

// ============================================================================
// Setup stage: AllocBuilder + DeviceBufFake
// ============================================================================

struct BuilderState {
    clock: Tick,
    allocs: Vec<PlannedAlloc>,
    /// Element counts parallel to `allocs` (sizes there are in bytes).
    elem_lens: Vec<usize>,
    /// Set by `build_on`: later fake drops must no longer mutate the plan.
    sealed: bool,
}

/// Records allocations and their lifetimes during the setup stage.
///
/// Create one, let every phase declare its allocations with
/// [`alloc_fake`](Self::alloc_fake), then call [`build_on`](Self::build_on) to pack
/// the plan and allocate the workspace.
pub struct AllocBuilder {
    plan_id: u64,
    state: Arc<Mutex<BuilderState>>,
}

impl AllocBuilder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            plan_id: NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed),
            state: Arc::new(Mutex::new(BuilderState {
                clock: 0,
                allocs: Vec::new(),
                elem_lens: Vec::new(),
                sealed: false,
            })),
        }
    }

    /// Declares an allocation of `len` elements of `T` on the default stream
    /// key. The returned RAII handle's lifetime declares the real buffer's
    /// lifetime: drop it at the point where the buffer may be reused.
    pub fn alloc_fake<T>(&self, len: usize) -> DeviceBufFake<T> {
        self.alloc_fake_impl(len, DEFAULT_STREAM_KEY, None)
    }

    /// Same as [`alloc_fake`](Self::alloc_fake) with a label used in
    /// diagnostics and lifetime-violation errors.
    pub fn alloc_fake_labeled<T>(&self, len: usize, label: &str) -> DeviceBufFake<T> {
        self.alloc_fake_impl(len, DEFAULT_STREAM_KEY, Some(label.to_string()))
    }

    /// Same as [`alloc_fake_labeled`](Self::alloc_fake_labeled) for work
    /// issued on a different stream. Allocations never share memory across
    /// stream keys.
    pub fn alloc_fake_on_stream<T>(
        &self,
        len: usize,
        label: &str,
        stream: u64,
    ) -> DeviceBufFake<T> {
        self.alloc_fake_impl(len, stream, Some(label.to_string()))
    }

    fn alloc_fake_impl<T>(
        &self,
        len: usize,
        stream: u64,
        label: Option<String>,
    ) -> DeviceBufFake<T> {
        assert_ne!(len, 0, "Zero capacity request is wrong");
        let size = std::mem::size_of::<T>()
            .checked_mul(len)
            .expect("allocation size overflows usize");
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(!state.sealed, "BUG: alloc_fake after build");
        let id = state.allocs.len() as u32;
        let label =
            label.unwrap_or_else(|| format!("alloc{}: {}x{}", id, std::any::type_name::<T>(), len));
        let birth = state.clock;
        state.clock += 1;
        state.allocs.push(PlannedAlloc {
            size,
            align: std::mem::align_of::<T>().max(MIN_ALIGN),
            birth,
            death: IMMORTAL,
            stream,
            label,
        });
        state.elem_lens.push(len);
        DeviceBufFake {
            id,
            plan_id: self.plan_id,
            len,
            state: Arc::clone(&self.state),
            _marker: PhantomData,
        }
    }

    /// Runs the packing pass WITHOUT allocating device memory and returns the
    /// resulting sizes. Non-consuming: fakes still alive are treated as live
    /// until the end of the plan, and more allocations can be declared (or
    /// [`build_on`](Self::build_on) called) afterwards.
    ///
    /// Useful for offline analysis, e.g. replaying a recorded allocation
    /// trace to see what workspace a static plan would need.
    pub fn plan_stats(&self) -> PlanStats {
        let allocs = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.allocs.clone()
        };
        let layout = plan_offsets(&allocs);
        PlanStats {
            workspace_size: layout.total_size,
            peak_live_size: layout.peak_live_size,
            sum_of_sizes: allocs.iter().map(|a| a.size).sum(),
            num_allocs: allocs.len(),
        }
    }

    /// Packs all declared allocations and allocates the workspace on the
    /// given stream.
    ///
    /// The returned allocator is bound to `ctx`: fallback allocations and
    /// diagnostics use it, and planned memory reuse relies on work being
    /// issued in declared-lifetime order on that stream. Fake handles still
    /// alive are treated as live until the end of the plan; dropping them
    /// afterwards has no effect. Uses the default [`ViolationPolicy::Error`].
    pub fn build_on(self, ctx: &GpuDeviceCtx) -> Result<StaticAllocator, StaticAllocError> {
        self.build_with_policy_on(ctx, ViolationPolicy::default())
    }

    pub fn build_with_policy_on(
        self,
        ctx: &GpuDeviceCtx,
        policy: ViolationPolicy,
    ) -> Result<StaticAllocator, StaticAllocError> {
        let (allocs, elem_lens) = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            state.sealed = true;
            (state.allocs.clone(), state.elem_lens.clone())
        };

        let layout = plan_offsets(&allocs);
        let workspace = if layout.total_size == 0 {
            DeviceBuffer::new()
        } else {
            let ptr = d_malloc_on(layout.total_size, &ctx.stream)?;
            // SAFETY: `ptr` was just returned by `d_malloc_on` for
            // `layout.total_size` bytes, so the buffer owns it and its drop
            // (`d_free`) is the matching deallocation.
            unsafe { DeviceBuffer::from_raw_parts(ptr as *mut u8, layout.total_size) }
        };
        #[cfg(feature = "touchemall")]
        if !workspace.is_empty() {
            // Poison so reads of never-written planned bytes are detectable.
            unsafe {
                crate::d_buffer::cudaMemsetAsync(
                    workspace.as_mut_raw_ptr(),
                    0xff,
                    layout.total_size,
                    ctx.stream.as_raw(),
                );
            }
        }

        let max_align = allocs
            .iter()
            .map(|a| a.align.max(MIN_ALIGN))
            .max()
            .unwrap_or(MIN_ALIGN);
        if !workspace.is_empty() && !(workspace.as_ptr() as usize).is_multiple_of(max_align) {
            return Err(StaticAllocError::MisalignedWorkspace {
                required: max_align,
            });
        }

        let sum_of_sizes: usize = allocs.iter().map(|a| a.size).sum();
        tracing::info!(
            "StaticAllocator: {} allocations packed into {} (peak live {}, sum of sizes {})",
            allocs.len(),
            ByteSize::b(layout.total_size as u64),
            ByteSize::b(layout.peak_live_size as u64),
            ByteSize::b(sum_of_sizes as u64),
        );

        let entries = allocs
            .into_iter()
            .zip(layout.offsets)
            .zip(elem_lens)
            .map(|((alloc, offset), elem_len)| PlanEntry {
                offset,
                elem_len,
                alloc,
            })
            .collect::<Vec<_>>();
        let live = (0..entries.len())
            .map(|_| EntryRuntime {
                state: LiveState::Free,
                contents_lost: false,
            })
            .collect();

        Ok(StaticAllocator {
            inner: Arc::new(AllocatorInner {
                ctx: ctx.clone(),
                workspace,
                entries,
                live: Mutex::new(live),
                plan_id: self.plan_id,
                policy,
                total_size: layout.total_size,
                peak_live_size: layout.peak_live_size,
            }),
        })
    }
}

/// RAII stand-in for a future allocation, created during the setup stage.
///
/// Owns no device memory. Its Rust lifetime declares the lifetime of the real
/// buffer: the allocation is considered live from the `alloc_fake` call until
/// the handle is dropped, and the packer only reuses its space for
/// allocations whose declared lifetimes do not overlap it.
pub struct DeviceBufFake<T> {
    id: u32,
    plan_id: u64,
    len: usize,
    state: Arc<Mutex<BuilderState>>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> DeviceBufFake<T> {
    /// Returns the persistent handle used to acquire the real buffer from the
    /// [`StaticAllocator`] in the run stage.
    pub fn idx(&self) -> AllocIdx<T> {
        AllocIdx {
            id: self.id,
            plan_id: self.plan_id,
            _marker: PhantomData,
        }
    }

    /// Alias for [`idx`](Self::idx).
    pub fn get_idx(&self) -> AllocIdx<T> {
        self.idx()
    }

    /// Number of `T` elements declared.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for DeviceBufFake<T> {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if !state.sealed {
            let death = state.clock;
            state.clock += 1;
            state.allocs[self.id as usize].death = death;
        }
    }
}

impl<T> fmt::Debug for DeviceBufFake<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeviceBufFake(id={}, len={})", self.id, self.len)
    }
}

/// Copyable handle to a planned allocation. Obtained from
/// [`DeviceBufFake::idx`] during setup; redeemed with
/// [`StaticAllocator::get`] during the run stage.
pub struct AllocIdx<T> {
    id: u32,
    plan_id: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for AllocIdx<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for AllocIdx<T> {}

impl<T> fmt::Debug for AllocIdx<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllocIdx(plan={}, id={})", self.plan_id, self.id)
    }
}

// ============================================================================
// Run stage: StaticAllocator + StaticDeviceBuffer
// ============================================================================

#[derive(Debug)]
struct PlanEntry {
    offset: usize,
    elem_len: usize,
    alloc: PlannedAlloc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveState {
    Free,
    /// Acquired and occupying its planned region of the workspace.
    Planned,
    /// Acquired but served from a fresh dynamic allocation
    /// ([`ViolationPolicy::WarnAndAllocate`] fallback).
    Fallback,
}

/// Per-allocation runtime state, guarded by the allocator mutex.
struct EntryRuntime {
    state: LiveState,
    /// The last acquisition was served off-plan
    /// ([`ViolationPolicy::WarnAndAllocate`]), so anything written through it
    /// never reached the planned region. The next planned acquisition warns
    /// and clears this.
    contents_lost: bool,
}

struct AllocatorInner {
    /// The stream this allocator is bound to: the workspace is allocated on
    /// it, fallback allocations use it, and all planned reuse relies on its
    /// issue order.
    ctx: GpuDeviceCtx,
    workspace: DeviceBuffer<u8>,
    entries: Vec<PlanEntry>,
    live: Mutex<Vec<EntryRuntime>>,
    plan_id: u64,
    policy: ViolationPolicy,
    total_size: usize,
    peak_live_size: usize,
}

/// Serves planned buffers at fixed offsets inside one workspace allocation.
///
/// Cheap to clone (shared handle). Buffers acquired through [`get`](Self::get)
/// release their planned region when dropped; the same [`AllocIdx`] can be
/// acquired again afterwards and returns the same device pointer.
#[derive(Clone)]
pub struct StaticAllocator {
    inner: Arc<AllocatorInner>,
}

impl StaticAllocator {
    /// Acquires the planned buffer for `idx`.
    ///
    /// Errors if `idx` belongs to another plan, is already acquired, or —
    /// under [`ViolationPolicy::Error`] — if its planned space is still held
    /// by a buffer that outlived its declared lifetime. Under
    /// [`ViolationPolicy::WarnAndAllocate`] the latter case logs a warning
    /// and serves a fresh dynamic allocation instead; the space conflict
    /// resolves once the offending buffer is dropped.
    pub fn get<T>(&self, idx: &AllocIdx<T>) -> Result<StaticDeviceBuffer<T>, StaticAllocError> {
        let inner = &self.inner;
        if idx.plan_id != inner.plan_id {
            return Err(StaticAllocError::PlanMismatch {
                expected: inner.plan_id,
                actual: idx.plan_id,
            });
        }
        let entry = &inner.entries[idx.id as usize];
        debug_assert_eq!(entry.alloc.size, entry.elem_len * std::mem::size_of::<T>());
        let range = entry.offset..entry.offset + entry.alloc.size;
        let overlaps = |other: &PlanEntry| {
            other.offset < range.end && range.start < other.offset + other.alloc.size
        };

        let mut live = inner.live.lock().map_err(|_| StaticAllocError::LockError)?;
        if live[idx.id as usize].state != LiveState::Free {
            return Err(StaticAllocError::AlreadyAcquired {
                label: entry.alloc.label.clone(),
            });
        }

        let holders: Vec<String> = inner
            .entries
            .iter()
            .enumerate()
            .filter(|&(j, other)| live[j].state == LiveState::Planned && overlaps(other))
            .map(|(_, other)| other.alloc.label.clone())
            .collect();

        if holders.is_empty() {
            // Diagnostics: a live fallback buffer whose planned range
            // overlaps is a declared-lifetime violation too (its holder was
            // served off-plan), even though it does not physically occupy
            // this space. Report it so violations don't disappear just
            // because an earlier one was absorbed by the fallback policy.
            for (j, other) in inner.entries.iter().enumerate() {
                if live[j].state == LiveState::Fallback && overlaps(other) {
                    tracing::warn!(
                        "StaticAllocator: granting '{}' while '{}' (overlapping plan) \
                         is still live off-plan — a buffer outlived its declared lifetime",
                        entry.alloc.label,
                        other.alloc.label
                    );
                }
            }
            if live[idx.id as usize].contents_lost {
                live[idx.id as usize].contents_lost = false;
                tracing::warn!(
                    "StaticAllocator: '{}' was previously served off-plan; data written \
                     during that acquisition was not preserved in its planned region",
                    entry.alloc.label
                );
            }
            live[idx.id as usize].state = LiveState::Planned;
            // SAFETY: the planned region lies inside the workspace allocation
            // (offset + size <= total_size by construction), is aligned for T
            // (offset is MIN_ALIGN-aligned, base checked at build), and the
            // returned buffer is wrapped in ManuallyDrop so it never frees.
            let buf = unsafe {
                DeviceBuffer::from_raw_parts(
                    inner.workspace.as_mut_ptr().add(entry.offset) as *mut T,
                    entry.elem_len,
                )
            };
            // Poison each acquisition so reads of stale bytes from a
            // previous occupant of this region are detectable, mirroring
            // `with_capacity_on` semantics under this feature.
            #[cfg(feature = "touchemall")]
            unsafe {
                crate::d_buffer::cudaMemsetAsync(
                    buf.as_mut_raw_ptr(),
                    0xff,
                    entry.alloc.size,
                    inner.ctx.stream.as_raw(),
                );
            }
            return Ok(StaticDeviceBuffer {
                buf: ManuallyDrop::new(buf),
                id: idx.id,
                kind: LiveState::Planned,
                alloc: Arc::clone(inner),
            });
        }

        match inner.policy {
            ViolationPolicy::Error => Err(StaticAllocError::LifetimeViolation {
                requested: entry.alloc.label.clone(),
                holders,
            }),
            ViolationPolicy::WarnAndAllocate => {
                tracing::warn!(
                    "StaticAllocator: planned space of '{}' still held by {:?}; \
                     serving a fresh dynamic allocation instead",
                    entry.alloc.label,
                    holders
                );
                // Allocate with the lock released: `with_capacity_on` panics
                // on OOM and must not poison the live set.
                drop(live);
                let buf = DeviceBuffer::<T>::with_capacity_on(entry.elem_len, &inner.ctx);
                let mut live = inner.live.lock().map_err(|_| StaticAllocError::LockError)?;
                if live[idx.id as usize].state != LiveState::Free {
                    // A racing `get` acquired this idx while the lock was
                    // released; `buf` is dropped and freed normally.
                    return Err(StaticAllocError::AlreadyAcquired {
                        label: entry.alloc.label.clone(),
                    });
                }
                live[idx.id as usize].state = LiveState::Fallback;
                // Data written through this guard lands in the fallback
                // allocation and is freed with it — the planned region never
                // sees it. Remember that so the next planned acquisition can
                // warn about the persistence break.
                live[idx.id as usize].contents_lost = true;
                Ok(StaticDeviceBuffer {
                    buf: ManuallyDrop::new(buf),
                    id: idx.id,
                    kind: LiveState::Fallback,
                    alloc: Arc::clone(inner),
                })
            }
        }
    }

    /// Total workspace size in bytes.
    pub fn workspace_size(&self) -> usize {
        self.inner.total_size
    }

    /// Peak of concurrently live declared bytes — the lower bound the packer
    /// was aiming for.
    pub fn peak_live_size(&self) -> usize {
        self.inner.peak_live_size
    }

    /// Number of planned allocations.
    pub fn num_allocs(&self) -> usize {
        self.inner.entries.len()
    }

    /// Human-readable table of the plan, ordered by offset.
    pub fn describe(&self) -> String {
        let mut order: Vec<usize> = (0..self.inner.entries.len()).collect();
        order.sort_by_key(|&i| {
            (
                self.inner.entries[i].offset,
                self.inner.entries[i].alloc.birth,
            )
        });
        let mut out = format!(
            "StaticAllocator plan {}: workspace {} (peak live {})\n",
            self.inner.plan_id,
            ByteSize::b(self.inner.total_size as u64),
            ByteSize::b(self.inner.peak_live_size as u64),
        );
        for i in order {
            let e = &self.inner.entries[i];
            let death = if e.alloc.death == IMMORTAL {
                "end".to_string()
            } else {
                e.alloc.death.to_string()
            };
            out.push_str(&format!(
                "  [{:#012x}..{:#012x}) {:>10} life=[{},{}) stream={} '{}'\n",
                e.offset,
                e.offset + e.alloc.size,
                ByteSize::b(e.alloc.size as u64).to_string(),
                e.alloc.birth,
                death,
                e.alloc.stream,
                e.alloc.label,
            ));
        }
        out
    }
}

impl fmt::Debug for StaticAllocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

/// RAII view of a planned region of the workspace (or of a fallback dynamic
/// allocation under [`ViolationPolicy::WarnAndAllocate`]).
///
/// Derefs to [`DeviceBuffer`] (shared access), so kernels and the
/// device-to-host copy traits work unchanged; host-to-device copies go
/// through [`copy_from`](Self::copy_from). Dropping it releases the planned
/// region for the allocations that reuse the same space; it does NOT free the
/// workspace memory.
pub struct StaticDeviceBuffer<T> {
    buf: ManuallyDrop<DeviceBuffer<T>>,
    id: u32,
    kind: LiveState,
    alloc: Arc<AllocatorInner>,
}

impl<T> StaticDeviceBuffer<T> {
    /// Copies `src` from the host into this buffer on the given stream.
    ///
    /// This is an inherent method rather than `DerefMut` +
    /// [`MemCopyH2D::copy_to_on`]: handing out `&mut DeviceBuffer<T>` would
    /// let safe code move the non-owning view out of the guard (e.g. with
    /// `mem::replace`) and `d_free` workspace-interior memory on its drop.
    pub fn copy_from(&mut self, src: &[T], ctx: &GpuDeviceCtx) -> Result<(), MemCopyError> {
        src.copy_to_on(&mut self.buf, ctx)
    }
}

impl<T> Deref for StaticDeviceBuffer<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl<T> Drop for StaticDeviceBuffer<T> {
    fn drop(&mut self) {
        let mut live = self
            .alloc
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let rt = &mut live[self.id as usize];
        if self.kind == LiveState::Fallback {
            // SAFETY: dropped exactly once, here; the buffer owns a real
            // dynamic allocation made in `get`.
            unsafe { ManuallyDrop::drop(&mut self.buf) };
        }
        rt.state = LiveState::Free;
    }
}

impl<T> fmt::Debug for StaticDeviceBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StaticDeviceBuffer(id={}, len={}, ptr={:p})",
            self.id,
            self.buf.len(),
            self.buf.as_ptr()
        )
    }
}
