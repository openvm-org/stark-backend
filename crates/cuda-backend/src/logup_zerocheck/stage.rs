//! Single-upload staging of small per-round kernel parameter tables.
//!
//! The batch MLE sumcheck rounds launch many kernels whose parameters (pointer
//! tables, block/AIR context arrays) are small host-built arrays. Uploading each
//! array with its own `to_device_on` costs a device allocation plus a pageable
//! `cudaMemcpyAsync` per array — tens of tiny H2D transfers per sumcheck round.
//! [`RoundStager`] instead packs all arrays of a round into one host staging
//! buffer and issues a single H2D copy into a persistent device arena.
//!
//! Usage is cycle-based, with strict phases enforced by debug assertions:
//! 1. [`RoundStager::push`] any number of host arrays (returns [`StagedSlice`] handles),
//! 2. [`RoundStager::commit`] once (single H2D copy),
//! 3. [`RoundStager::ptr`] to resolve handles to device pointers.
//!
//! Multiple commit cycles per round are supported; earlier cycles' device
//! pointers stay valid until [`RoundStager::begin_round`] is called (device
//! arenas are only recycled at round boundaries, and frees are stream-ordered,
//! so in-flight kernels from the previous round are unaffected).

use std::marker::PhantomData;

use openvm_cuda_common::{
    copy::cuda_memcpy_on, d_buffer::DeviceBuffer, error::MemCopyError, stream::GpuDeviceCtx,
};

/// All staged slices are aligned to this many bytes. Must be a multiple of the
/// alignment of every staged element type (the largest we stage is `EF`-bearing
/// context structs at 16 bytes).
const STAGE_ALIGN: usize = 16;

/// Handle to a slice staged via [`RoundStager::push`]. Resolves to a device
/// pointer with [`RoundStager::ptr`] after the cycle's [`RoundStager::commit`].
pub(crate) struct StagedSlice<T> {
    /// Index into `RoundStager::cycle_bases`.
    cycle: u32,
    /// Byte offset within the cycle's staged block.
    offset: u32,
    _marker: PhantomData<T>,
}

impl<T> Clone for StagedSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for StagedSlice<T> {}

/// (arena index, byte offset of the cycle's base within that arena)
type CycleBase = (u32, u32);

pub(crate) struct RoundStager {
    /// Host staging for the current (uncommitted) cycle.
    host: Vec<u8>,
    /// Device arenas. In steady state there is exactly one, sized to the
    /// high-water mark of a full round.
    arenas: Vec<DeviceBuffer<u8>>,
    /// Base (arena, offset) per committed cycle this round.
    cycle_bases: Vec<CycleBase>,
    /// Write cursor within the last arena.
    cur_offset: usize,
    /// Total bytes committed this round (for high-water tracking).
    round_bytes: usize,
    /// High-water mark of `round_bytes` across rounds.
    high_water: usize,
}

impl RoundStager {
    pub fn new() -> Self {
        Self {
            host: Vec::new(),
            arenas: Vec::new(),
            cycle_bases: Vec::new(),
            cur_offset: 0,
            round_bytes: 0,
            high_water: 0,
        }
    }

    /// Starts a new round: recycles device arenas (consolidating to a single
    /// arena sized to the high-water mark) and invalidates all previously
    /// issued [`StagedSlice`] handles.
    pub fn begin_round(&mut self, device_ctx: &GpuDeviceCtx) {
        debug_assert!(
            self.host.is_empty(),
            "begin_round called with uncommitted staged data"
        );
        self.high_water = self.high_water.max(self.round_bytes);
        if self.arenas.len() != 1 || self.arenas[0].len() < self.high_water {
            // Dropping old arenas is safe mid-stream: frees are stream-ordered.
            self.arenas.clear();
            if self.high_water > 0 {
                self.arenas.push(DeviceBuffer::with_capacity_on(
                    self.high_water.next_power_of_two(),
                    device_ctx,
                ));
            }
        }
        self.cycle_bases.clear();
        self.cur_offset = 0;
        self.round_bytes = 0;
    }

    /// Stages a host array for upload in the current cycle.
    pub fn push<T: Copy>(&mut self, items: &[T]) -> StagedSlice<T> {
        debug_assert!(std::mem::align_of::<T>() <= STAGE_ALIGN);
        let offset = self.host.len().next_multiple_of(STAGE_ALIGN);
        self.host.resize(offset, 0u8);
        // SAFETY: `T: Copy` (plain old data); we view its bytes for staging.
        let bytes = unsafe {
            std::slice::from_raw_parts(items.as_ptr() as *const u8, std::mem::size_of_val(items))
        };
        self.host.extend_from_slice(bytes);
        StagedSlice {
            cycle: self.cycle_bases.len() as u32,
            offset: offset as u32,
            _marker: PhantomData,
        }
    }

    /// Uploads the current cycle's staged bytes with a single H2D copy.
    ///
    /// Slices pushed since the previous commit resolve against this cycle.
    pub fn commit(&mut self, device_ctx: &GpuDeviceCtx) -> Result<(), MemCopyError> {
        let need = self.host.len();
        if need == 0 {
            // Preserve cycle indexing for slices (none were pushed, but keep
            // the invariant that each commit consumes one cycle id).
            self.cycle_bases.push((0, 0));
            return Ok(());
        }
        let need_aligned = need.next_multiple_of(STAGE_ALIGN);
        let fits = self
            .arenas
            .last()
            .is_some_and(|a| self.cur_offset + need <= a.len());
        if !fits {
            let cap = need_aligned.next_power_of_two().max(1 << 16);
            self.arenas
                .push(DeviceBuffer::with_capacity_on(cap, device_ctx));
            self.cur_offset = 0;
        }
        let arena_idx = self.arenas.len() - 1;
        let arena = &self.arenas[arena_idx];
        // SAFETY: `cur_offset + need <= arena.len()` by the check above; the
        // source is a live host slice. `cudaMemcpyAsync` from pageable memory
        // returns only after the source has been consumed, so clearing `host`
        // afterwards is safe.
        unsafe {
            cuda_memcpy_on::<false, true>(
                arena.as_mut_ptr().add(self.cur_offset) as *mut _,
                self.host.as_ptr() as *const _,
                need,
                device_ctx,
            )?;
        }
        self.cycle_bases
            .push((arena_idx as u32, self.cur_offset as u32));
        self.cur_offset += need_aligned;
        self.round_bytes += need_aligned;
        self.host.clear();
        Ok(())
    }

    /// Resolves a staged slice to its device pointer. The slice's cycle must
    /// have been committed.
    pub fn ptr<T>(&self, slice: StagedSlice<T>) -> *const T {
        let (arena, base) = self.cycle_bases[slice.cycle as usize];
        debug_assert!((arena as usize) < self.arenas.len());
        // SAFETY: offset stays within the arena by construction in `commit`.
        unsafe {
            self.arenas[arena as usize]
                .as_ptr()
                .add(base as usize + slice.offset as usize) as *const T
        }
    }
}
