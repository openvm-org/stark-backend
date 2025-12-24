# GPU Memory Manager with Virtual Pool (VPMM)

## Introduction

The **GPU Memory Manager** is a Rust module responsible for GPU memory allocation and reclamation via a simple API:

* `malloc(size) -> ptr`
* `free(ptr) -> ()`

The original **stream-oriented memory manager (SOMM)** used CUDA Runtime APIs (`cudaMallocAsync` / `cudaFreeAsync`) and relied on CUDA's built-in memory pool. While simple, it can lead to:

* **Fragmentation & peak usage inflation:** extra GPU memory consumption that can cause unexpected OOMs.
* **Performance degradation:** allocations/frees tied to stream progress can introduce waits.

The **Virtual Memory Pool Memory Manager (VPMM)** replaces this with a design based on the CUDA **Virtual Memory Management (VMM) Driver API** to reduce fragmentation and improve utilization.

## Goals

* Eliminate or minimize fragmentation without copying by **remapping pages** in virtual address space.
* Maintain predictable GPU memory usage aligned with actual live allocations.
* Support **multi-stream workloads** with cross-stream memory reuse while minimizing synchronization overhead.
* Provide internal observability of memory usage.

## Concepts & Notation

* **Virtual address (VA) space** — Per-process GPU-visible address range. Reserving VA does **not** allocate physical memory.
* **Page** — Fixed-size physical GPU chunk (≥ CUDA VMM granularity). All mappings and allocations are rounded up to page size.
* **Region** — Consecutive VA range tracked with symbolic states:
  * `[+X]` – current allocation returned by `malloc`
  * `[X]`  – previously allocated user region
  * `[-X]` – mapped free region with an associated stream/event
  * `[*X]` – **unmapped** region (hole created after remapping)
  * `[#X]` – brand new region created while satisfying the current allocation

## High-Level Design

1. **Reserve** a large VA chunk once (size configurable via `VPMM_VA_SIZE`). Additional chunks are reserved on demand.
2. **Track** four maps: `malloc_regions`, `free_regions` (with CUDA events/stream ids), `unmapped_regions` (holes), and `zombie_regions` (old VAs awaiting async unmap).
3. **Allocate** by finding the best-fit free region in the current stream; otherwise reuse a region from another stream only if its event has already completed (no synchronization in this phase).
4. **Defragment** only when necessary: grab a hole, harvest pages from free regions (current stream first, then other streams), **double-map** them to the new VA, and queue the old VAs as zombies for async unmap.
5. **Grow** by allocating the shortfall in physical pages and mapping them into the same hole when free regions still aren't enough.
6. **Cleanup** zombie regions opportunistically at the start of each malloc — unmap old VAs whose events have completed and return them to `unmapped_regions`.
7. **Observe** all activity via maps (no implicit "active end"). Debug output lists every region in ascending VA order.

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

/// Metadata for a free region in the virtual address space.
struct FreeRegionMeta {
    size: usize,
    event: Arc<CudaEvent>,   // Event marking when this region was freed
    stream_id: CudaStreamId, // Stream that freed this region
    id: usize,               // Creation order for temporal tracking
}

/// Remapped region that will be unmapped when the event completes.
struct ZombieRegion {
    ptr: CUdeviceptr,
    size: usize,
    event: Arc<CudaEvent>,
}

pub(super) struct VirtualMemoryPool {
    roots: Vec<CUdeviceptr>,     // Every reserved VA chunk
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,
    free_regions: BTreeMap<CUdeviceptr, FreeRegionMeta>,
    malloc_regions: HashMap<CUdeviceptr, usize>,
    unmapped_regions: BTreeMap<CUdeviceptr, usize>,
    zombie_regions: Vec<ZombieRegion>,  // Old VAs awaiting async unmap
    free_num: usize,
    pub(super) page_size: usize,
    va_size: usize,
    device_id: i32,
}
```

**Invariants**

* Every `roots` entry corresponds to a reserved VA chunk of size `va_size`. We only map/unmap within these chunks.
* `active_pages` tracks the current virtual address for every mapped page; keys move when we remap.
* `free_regions`, `malloc_regions`, `unmapped_regions`, and `zombie_regions` partition the reserved VA space. Note that zombie regions are temporarily **double-mapped** (the same physical pages are accessible via both old and new VAs until the zombie is cleaned up).
* `free_regions` are coalesced by stream/event when possible.
* Each `FreeRegionMeta` retains the CUDA event recorded at free time plus the originating `CudaStreamId`.

## Initialization

* **VA Reservation:** Reserve a `VPMM_VA_SIZE` chunk (default 8 TB) at startup. When every hole is consumed, reserve another chunk and append it to `roots`.
* **Page Size:** Configurable via `VPMM_PAGE_SIZE` (≥ CUDA's VMM granularity, typically 2 MB). All requests are rounded up to this size.
* **Initial Pages:** `VPMM_PAGES` controls how many pages are eagerly mapped. Defaults to 0 (purely on-demand).
* **Mapping Unit:** Always page-sized; the pool never subdivides a page.

## Recommended Configuration

**For best performance**, preallocate ~80% of available GPU memory to avoid runtime allocations:

```bash
# Example: 40 GB GPU → preallocate 32 GB (80%)
export VPMM_PAGES=$(((32 << 30) / (2 << 20)))  # 16384 pages at 2 MB each
```

**To disable VPMM** and fall back to `cudaMallocAsync`, use a page size larger than any expected allocation:

```bash
export VPMM_PAGE_SIZE=$((32 << 30))  # 32 GB page size
export VPMM_PAGES=0
```

## Allocation & Growth

Allocations occur during initialization (preallocation) and on-demand when the pool runs short of free pages.

* **Synchronous:** The CUDA Driver API performs allocations synchronously.
* **Granularity:** Allocation is **by page**.
* **MinUse policy:** When the pool is out of pages, allocate only as many pages as required to satisfy the current request.
* **Lifetime:** Allocated pages are **retained** for the lifetime of the process (not returned to the OS).

## Defragmentation

Triggered when Phase 1 fails. The defragmentation flow is deterministic and minimizes synchronization:

1. **Take a hole** — `take_unmapped_region` returns the smallest unmapped interval ≥ `requested`. If none exists, we reserve another VA chunk and carve the hole out of it.
2. **Allocate shortfall** — compute the page shortfall (`requested - free_bytes`). Allocate that many physical pages and map them into the beginning of the hole.
3. **Harvest free regions** — iterate `free_regions` with priority:
   * **Current stream first** (no wait needed — stream ordering guarantees safety)
   * **Other streams ordered by `id`** (oldest first)
   
   For each region from another stream with a non-completed event, call `default_stream_wait(&event)` to insert a GPU-side dependency (no CPU blocking). Detach only the portion we need (`take = min(remaining, region.size)`). Reinsert the leftover tail (if any) back into `free_regions` with the original stream but a new event/id. Stop once the total `take` covers the remainder of the request.
4. **Double-map into the hole** — map the harvested pages (page by page) contiguously into the hole. The old VAs remain mapped temporarily (double-mapping). Add each old VA + event to `zombie_regions` for async cleanup later. The combined span becomes the new free region for the requesting stream.

**Key insight:** CUDA VMM allows the same physical allocation to be mapped to multiple VAs simultaneously. This enables async unmap — we map to the new VA immediately, and unmap the old VA later when its event completes.

## malloc(size) — Allocation Policy

**Phase 0: Cleanup zombies**
* Iterate `zombie_regions` and unmap any whose events have completed. Return their VAs to `unmapped_regions`.

**Phase 1: Zero-cost attempts (no synchronization)**
1. **Best fit on current stream** — smallest region from the caller's stream that fits `requested`. No wait needed.
2. **Completed from other streams** — smallest region from other streams where `event.completed() == true`. No wait needed.

**Phase 2: Hole-based defragmentation (async GPU wait)**
1. **Reserve hole** — via `take_unmapped_region`.
2. **Allocate shortfall** — map the missing number of pages into the hole.
3. **Harvest free regions** — current stream first, then other streams (oldest first). For other streams with non-completed events, call `default_stream_wait` (GPU waits, CPU continues).
4. **Double-map into hole** — map pages to new VA, queue old VAs as zombies for async unmap. The combined span becomes the new free region for the caller.

Additional rules:

* **Alignment:** All allocations are **rounded up** to page-size multiples. A page is either entirely free or entirely used.
* **Small Buffers:** If `size < page_size`, bypass the VM pool and call `cudaMallocAsync` instead (preserves compatibility; setting `VPMM_PAGE_SIZE = usize::MAX` effectively disables the pool for typical sizes).

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
* **Pending unmap (zombies):** `sum(pool.zombie_regions.iter().map(|z| z.size))`
* `Debug` output prints the metrics above plus every region in ascending VA order.

## Asynchrony & Streams

* VPMM supports **multi-stream workloads** using `cudaStreamPerThread`.
* A single shared `VirtualMemoryPool` serves all streams. Each free region carries the stream id plus a CUDA event (wrapped in `Arc`).
* **Cross-stream reuse:**
  * In Phase 1 (`find_best_fit`): only take from other streams if `event.completed()` — no synchronization at all.
  * In Phase 2 (defrag): call `default_stream_wait(&event)` which inserts a GPU-side stream dependency. The CPU does **not** block; the GPU stream waits for the event before accessing the memory.
* **Double-mapping & zombies:** When remapping, we map pages to the new VA while the old VA is still mapped. The old VA is added to `zombie_regions` with its event. At the start of each `malloc`, we check zombies and unmap any whose events have completed.
* **Access permissions:** After remapping (or mapping newly allocated pages) we call `cuMemSetAccess` on the destination hole to ensure the caller's device has read/write permission.
