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
* Provide internal observability of memory usage.
* Support multi-threaded workloads with per-stream memory managers sharing a global page pool.

## Concepts & Notation

* **Virtual address (VA) space** — Per‑process range of GPU‑visible addresses. Reserved VA has no storage until mapped.
* **Active space** — The prefixed portion of the reserved VA that currently has mappings; grows as needed.
* **Page** — Fixed‑size chunk of physical GPU memory, the basic mapping unit. A page may be mapped at multiple VAs.
* **Region** — Consecutive VA range with a single state:

  * `[+X]`  allocated region returned by the **current** `malloc` call
  * `[X]`   allocated region from a **previous** call
  * `[-X]`  region marked as **free**
  * `[*X]`  region marked as **dead** (free but already remapped)
  * `[#X]`  region marked as **new** (free but created by current allocation)

## Architecture Overview

The system consists of three layers:

1. **GlobalMemoryManager** - Singleton that owns a shared pool of unused physical pages and coordinates per-stream managers.
2. **MemoryManager** (per-stream) - Handles allocations for a specific CUDA stream, managing both small buffers and a virtual memory pool.
3. **VirtualMemoryPool** - Maps physical pages into virtual address space, handles defragmentation.

```
+--------------------------------------+
|     GlobalMemoryManager (static)     |
|  +--------------------------------+  |
|  |   Unused Pages (Arc<Mutex>)    |  |
|  |  [Global pool of free pages]   |  |
|  +--------------------------------+  |
|           ^              ^           |
|           |              |           |
|  +--------+---+   +------+-----+     |
|  |MemoryMgr   |   |MemoryMgr  |      |
|  |(Stream 1)  |   |(Stream 2) | ...  |
|  |  +------+  |   |  +------+ |      |
|  |  | Pool |  |   |  | Pool | |      |
|  |  +------+  |   |  +------+ |      |
|  +------------+   +-----------+      |
+--------------------------------------+
```

## High‑Level Design

1. **Global Initialization:**
   - Reserve a sufficiently large VA range once per stream manager at initialization.
   - Pre‑allocate a configurable number of physical pages into the global unused pool.

2. **Per-Stream Operation:**
   - Each stream gets its own `MemoryManager` with its own `VirtualMemoryPool`.
   - When a pool needs pages, it takes them from the global unused pool.
   - If the global pool is empty, new pages are allocated on-demand.

3. **Allocation within a pool:**
   - **Map** pages contiguously at the **end of active space** for new allocations.
   - **Find‑or‑defragment**: if no free region fits, **remap** pages from earlier freed regions to the end (no data copy, just page table updates).
   - **Request more pages**: if still insufficient, take pages from global pool or allocate new ones.

4. **Stream cleanup:**
   - When a stream's usage hits zero, synchronize the stream and return all pages to the global unused pool.
   - The per-stream manager is removed, freeing its resources.

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
// cuda-common/src/memory_manager/mod.rs
// Global singleton
struct GlobalMemoryManager {
    per_stream_memory_managers: Mutex<HashMap<CudaStreamId, MemoryManager>>,
    unused_pages: Arc<Mutex<Vec<CUmemGenericAllocationHandle>>>,
    page_size: usize,
    device_id: i32,
}

// Per-stream manager
struct MemoryManager {
    pool: VirtualMemoryPool,
    allocated_ptrs: HashMap<NonNull<c_void>, usize>,  // Small buffers
    unused_pages: Arc<Mutex<Vec<CUmemGenericAllocationHandle>>>,  // Shared reference
    current_size: usize,
    max_used_size: usize,
}

// cuda-common/src/memory_manager/vm_pool.rs
pub(super) struct VirtualMemoryPool {
    device_id: i32,
    root: CUdeviceptr,           // Virtual address space root
    curr_end: CUdeviceptr,       // Current end of active address space
    pub(super) page_size: usize, // Mapping granularity (multiple of 2MB)

    // Map for all active pages (last mapping only; keys and values are unique)
    active_pages: HashMap<CUdeviceptr, CUmemGenericAllocationHandle>,

    // Free regions in virtual space (sorted by address; non-adjacent)
    free_regions: BTreeMap<CUdeviceptr, usize>,

    // Active allocations
    used_regions: HashMap<CUdeviceptr, usize>,
}
```

**Invariants**

* `root` is returned by VA reservation and satisfies: `root ≤ key < curr_end` for all keys in the collections.
* `active_pages` tracks the **current** mapping address for each page.
* `free_regions` cover only the **active** VA range and are always coalesced (neighbors merged on free).
* Pages in `GlobalMemoryManager::unused_pages` are not mapped to any virtual address.

## Initialization

* **VA Reservation:** Each `VirtualMemoryPool` reserves a large VA range at creation. **Current value:** 1 TB (constant).
* **Page Size:** Configurable via `VPMM_PAGE_SIZE`. Must be a multiple of CUDA’s minimum allocation granularity (typically 2 MB). Larger pages reduce mapping overhead; all allocations are rounded up to page size.
* **Initial Pages:** Count is configurable via `VPMM_PAGES`. If set, that many pages are pre-allocated into the global unused pool at startup. More pages improve performance but increase baseline memory footprint.
* **Mapping Unit:** All mappings are performed **by page**.

## Allocation & Growth

Allocations occur during initialization (preallocation) and on‑demand when the pool runs short of free pages.

* **Synchronous:** The CUDA Driver API performs allocations synchronously.
* **Granularity:** Allocation is **by page**.
* **MinUse policy:** When the pool is out of pages, allocate only as many pages as required to satisfy the current request.
* **Page lifecycle:** 
  - Pages are created via `cuMemCreate` and added to the global unused pool.
  - When a stream needs pages, they're moved from the global pool to that stream's pool.
  - When a stream's usage hits zero, pages are returned to the global pool.
  - Pages are only released to the OS on program termination.

## Defragmentation

Triggered when no free region can satisfy a request.

* **Remapping:** Select a free region and **remap its pages** to the end of active space. The original mapping remains valid during in‑flight use on the stream. After remapping, the original region becomes a **dead zone** and is not reused for new allocations. Note: unmapping may return `cudaErrorInvalidAddress` if attempted prematurely.
* **Region Selection:** **BiggestFirst** — prefer defragmenting larger regions to minimize the number of defragmented segments.

## malloc(size) — Allocation Policy

1. **Route by size:**
   - If `size < page_size`: Use `cudaMallocAsync` (bypass pool).
   - Otherwise: Round up to page size multiple and use pool.

2. **Pool allocation:**
   - **Find:** Search `free_regions` for a region large enough (BestFit).
   - **Defragment:** If none found, remap free regions until a region fits.
   - **Request pages:** If still insufficient after defragmentation, calculate needed pages.
   - **Acquire pages:** Lock global unused pool, take needed pages (or allocate new if insufficient).
   - **Map pages:** Call `pool.add_pages()` to map them to virtual address space.
   - **Retry allocation:** Now the pool has enough space.

Additional rules:

* **BestFit:** Among multiple fitting free regions, choose the smallest that satisfies the request.
* **Alignment:** All pool allocations are **rounded up** to page‑size multiples.

## free(ptr)

1. **Determine allocation type:**
   - Check `allocated_ptrs` (small buffers). If found, call `cudaFreeAsync`.
   - Otherwise, call `pool.free_internal(ptr)`.

2. **Check for stream cleanup:**
   - If `is_empty()` (both small buffers and pool are empty):
     - Synchronize the stream (`cudaStreamSynchronize`).
     - Call `pool.reset()` to unmap all pages and get handles back.
     - Return handles to global unused pool.
     - Trim CUDA's async memory pool.
     - Remove this stream's `MemoryManager` from the global map.

## Status & Observability

(All tracking is implemented in the outer **MemoryManager**, not inside `VirtualMemoryPool`, since small buffers bypass the pool.)

* **Total GPU memory used by pages:** `pool.active_pages.len() * pool.page_size`
* **Active VA extent:** `pool.curr_end - pool.base`
* **Currently allocated (live) bytes:** `sum(pool.used_regions.values())`
* **Currently freed (reusable) bytes:** `sum(pool.free_regions.values())`

## Asynchrony & Streams

* **Multi-stream support:** Each CUDA stream gets its own `MemoryManager` and `VirtualMemoryPool`.
* **Page sharing:** Physical pages are shared across streams via the global unused pool.
* **Thread safety:** All access to `GlobalMemoryManager` and shared pools is protected by mutexes.
* **Stream isolation:** Allocations and frees on one stream don't block operations on other streams (except during page acquisition from the global pool).
* **Automatic cleanup:** When a stream's usage drops to zero, its pages are returned to the global pool for reuse by other streams.
* **Stream synchronization:** Required only when cleaning up an empty stream, ensuring all GPU work is complete before unmapping pages.