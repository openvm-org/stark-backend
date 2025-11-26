# GPU Memory Manager with Virtual Pool (VPMM)

## Introduction

The **GPU Memory Manager** is a Rust module responsible for GPU memory allocation and reclamation via a simple API:

* `malloc(size) -> ptr`
* `free(ptr) -> ()`

The original **stream‑oriented memory manager (SOMM)** used CUDA Runtime APIs (`cudaMallocAsync` / `cudaFreeAsync`) and relied on CUDA’s built‑in memory pool. While simple, it can lead to:

* **Fragmentation & peak usage inflation:** extra GPU memory consumption that can cause unexpected OOMs.
* **Performance degradation:** allocations/frees tied to stream progress can introduce waits.

The **Virtual Memory Pool Memory Manager (VPMM)** replaces this with a design based on the CUDA **Virtual Memory Management (VMM) Driver API** to reduce fragmentation and improve utilization.

## Goals

* Eliminate or minimize fragmentation without copying by **remapping pages** in virtual address space.
* Maintain predictable GPU memory usage aligned with actual live allocations.
* Support **multi-stream workloads** with cross-stream memory reuse while minimizing synchronization overhead.
* Provide internal observability of memory usage.

## Concepts & Notation

* **Virtual address (VA) space** — Per‑process GPU-visible address range. Reserving VA does **not** allocate physical memory.
* **Page** — Fixed-size physical GPU chunk (≥ CUDA VMM granularity). All mappings and allocations are rounded up to page size.
* **Region** — Consecutive VA range tracked with symbolic states:
  * `[+X]` – current allocation returned by `malloc`
  * `[X]`  – previously allocated user region
  * `[-X]` – mapped free region with an associated stream/event
  * `[*X]` – **unmapped** region (hole created after remapping)
  * `[#X]` – brand new region created while satisfying the current allocation

## High‑Level Design

1. **Reserve** a large VA chunk once (size configurable via `VPMM_VA_SIZE`). Additional chunks are reserved on demand.
2. **Track** three disjoint maps: `malloc_regions`, `free_regions` (with CUDA events/stream ids), and `unmapped_regions` (holes).
3. **Allocate** by finding the best-fit free region in the current stream; otherwise reuse a completed region; otherwise wait on the oldest event.
4. **Defragment** only when necessary: grab a hole, detach just enough pages from the oldest free regions (split + reinsert leftovers), unmap their old VA, and remap into the hole.
5. **Grow** by allocating the shortfall in physical pages and mapping them into the same hole when free regions still aren’t enough.
6. **Observe** all activity via maps (no implicit “active end”). Debug output lists every region in address order.

Small allocations (< page size) bypass the pool and use `cudaMallocAsync` for simplicity and backward compatibility.

## Example Walkthrough

Scenario on a single stream with 5 sequential calls (`+` = malloc, `-` = free):

```
+10 GB  >  +1 GB  >  -10 GB  >  +4 GB  >  +11 GB
```

After the sequence, live allocations total **16 GB**. Early steps are identical across implementations when starting from an empty pool:

```
[+10][...]  >  [10][+1][...]  >  [-10][1][...]
```

**VPMM behavior** depends on page availability. For simplicity, assume 1 GB pages. First two allocations consumed 11 pages; define `X = PAGES - 11`, `X ≥ 0`.

### Case A: `X ≥ 11`

Enough free pages remain; VPMM maps without defragmentation ([BestFit policy](#?)):

```
4.  [-10][1][-X]   >  [+4][-6][1][-X]
5.  [4][-6][1][-X]  >  [4][-6][1][+11][-(X-11)]
```

### Case B: `5 ≤ X < 11`

Insufficient contiguous space for `+11`. VPMM [defragments](#?) by remapping the earliest free region to the end of active space and then fulfills the request.

```
4.  [-10][1][-X]   >  [-10][1][+4][-(X-4)]        
5.1 [-10][1][4][-(X-4)] > [*10][1][4][-((X-4)+10)]  (remap + merge = defrag)
5.2 [*10][1][4][-(X+6)] > [*10][1][4][+11][-(X-5)]  ( X + 6 ≥ 11)
```

### Case C: `X == 4`

Defragmentation occurs but still not enough pages for `+11`; VPMM allocates new pages and maps them after the active end, then merges.

```
4.   [-10][1][-X]   >  [-10][1][+4]
5.1  [-10][1][4]    >  [*10][1][4][-10]           (defrag)
5.2  [*10][1][4][-10][#1]  >  [*10][1][4][-11]  >  [*10][1][4][+11]
```

### Case D: `0 ≤ X < 4`

Similar to Case C, except `+4` in step 4 cannot fit the third region, so layout is different.

```
4.  [-10][1][-X]     >  [+4][-6][1][-X]
5.1 [4][-6][1][-X]   >  [4][*6][1][-(X+6)]        (defrag)
5.2 [4][*6][1][-(X+6)][#(11-X)]  >  [4][*6][1][-11]  >  [4][*6][1][+11]
```

## Data Structures

```rust
// cuda-common/src/memory_manager/vm_pool.rs
struct FreeRegion {
    size: usize,
    event: Arc<CudaEvent>,   // Event marking when this region was freed
    stream_id: CudaStreamId, // Stream that freed this region
    id: usize,               // Creation order for temporal tracking
}

pub(super) struct VirtualMemoryPool {
    roots: Vec<CUdeviceptr>,     // Every reserved VA chunk
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,
    free_regions: BTreeMap<CUdeviceptr, FreeRegion>,
    malloc_regions: HashMap<CUdeviceptr, usize>,
    unmapped_regions: BTreeMap<CUdeviceptr, usize>,
    free_num: usize,
    pub(super) page_size: usize,
    va_size: usize,
    device_id: i32,
}
```

**Invariants**

* Every `roots` entry corresponds to a reserved VA chunk of size `va_size`. We only map/unmap within these chunks.
* `active_pages` tracks the current virtual address for every mapped page; keys move when we remap.
* `free_regions`, `malloc_regions`, and `unmapped_regions` partition the reserved VA space with no overlap; `free_regions` are coalesced by stream/event when possible.
* Each `FreeRegion` retains the CUDA event recorded at free time plus the originating `CudaStreamId`; we block on this event before stealing the region from another stream.

## Initialization

* **VA Reservation:** Reserve a `VPMM_VA_SIZE` chunk (default 8 TB) at startup. When every hole is consumed, reserve another chunk and append it to `roots`.
* **Page Size:** Configurable via `VPMM_PAGE_SIZE` (≥ CUDA’s VMM granularity, typically 2 MB). All requests are rounded up to this size.
* **Initial Pages:** `VPMM_PAGES` controls how many pages are eagerly mapped. Defaults to 0 (purely on-demand).
* **Mapping Unit:** Always page-sized; the pool never subdivides a page.

## Allocation & Growth

Allocations occur during initialization (preallocation) and on‑demand when the pool runs short of free pages.

* **Synchronous:** The CUDA Driver API performs allocations synchronously.
* **Granularity:** Allocation is **by page**.
* **MinUse policy:** When the pool is out of pages, allocate only as many pages as required to satisfy the current request.
* **Lifetime:** Allocated pages are **retained** for the lifetime of the process (not returned to the OS).

## Defragmentation

Triggered when Phase 1 fails. The new defragmentation flow is deterministic and bounded:

1. **Take a hole** — `take_unmapped_region` returns the smallest unmapped interval ≥ `requested`. If none exists, we reserve another VA chunk and carve the hole out of it.
2. **Allocate shortfall** — compute the page shortfall (`requested - free_bytes`). Allocate that many physical pages and map them into the beginning of the hole.
3. **Harvest oldest frees** — iterate `free_regions` ordered by `free_id`. For each region:
   * Block on its CUDA event via `CudaEvent::synchronize()`.
   * Detach only the portion we need (`take = min(remaining, region.size)`), unmap it, and push `(addr, take)` to a work list.
   * Reinsert the leftover tail (if any) back into `free_regions` with its original stream/event.
   Stop once the total `take` covers the remainder of the request.
4. **Remap into the hole** — unmap each harvested chunk, add its VA back to `unmapped_regions`, and remap the chunk (page by page) contiguously into the hole immediately after the newly allocated portion. The combined span becomes the new free region for the requesting stream.


## malloc(size) — Allocation Policy

**Phase 1: Zero-cost attempts**
1. **Best fit on current stream** — smallest region from the caller’s stream that fits `requested`.
2. **Completed from other streams** — any region whose event has already completed (no blocking).
3. **Oldest in-flight** — take the lowest `free_id`, call `CudaEvent::synchronize()`, and hand that region to the caller. This avoids full defrag if one region can satisfy the request.

**Phase 2: Hole-based defragmentation**
4. **Reserve hole** — via `take_unmapped_region`.
5. **Allocate shortfall** — map the missing number of pages into the hole.
6. **Harvest oldest frees** — synchronously detach just enough pages from free regions (splitting leftovers).
7. **Remap into hole** — unmap the detached slices, add their old VA back to `unmapped_regions`, and remap them contiguously into the hole. The combined span (new pages + remapped slices) becomes the new free region for the caller.

Additional rules:

* **Alignment:** All allocations are **rounded up** to page‑size multiples. A page is either entirely free or entirely used.
* **Small Buffers:** If `size < page_size`, bypass the VM pool and call `cudaMallocAsync` instead (preserves compatibility; setting `VMM_PAGE_SIZE = usize::MAX` effectively disables the pool for typical sizes).

## free(ptr)

* Look up `ptr` in `malloc_regions` to obtain the aligned size.
* Record a CUDA event on the calling stream (always `cudaStreamPerThread`) and store the stream id/event pair in `free_regions`.
* Attempt to coalesce with adjacent regions from the same stream that have matching completion guarantees.
* Remove the entry from `malloc_regions`.

## Status & Observability

(All tracking is implemented in the outer **MemoryManager**, but the pool exposes enough state for debug dumps.)

* **Total GPU memory mapped:** `pool.active_pages.len() * pool.page_size`
* **Reserved VA:** `pool.roots.len() * pool.va_size`
* **Currently allocated (live) bytes:** `sum(pool.malloc_regions.values())`
* **Currently reusable bytes:** `sum(pool.free_regions.values().map(|r| r.size))`
* **Holes:** `sum(pool.unmapped_regions.values())`
* `Debug` output prints the metrics above plus every region in ascending VA order.

## Asynchrony & Streams

* VPMM supports **multi-stream workloads** using `cudaStreamPerThread`.
* A single shared `VirtualMemoryPool` serves all streams. Each free region carries the stream id plus a CUDA event. We reuse a region immediately if its event has completed; otherwise we call `event.synchronize()` right before detaching it. Events are dropped as soon as the synchronous wait completes.
* **Access permissions:** After remapping (or mapping newly allocated pages) we call `cuMemSetAccess` on the destination hole to ensure the caller’s device has read/write permission.