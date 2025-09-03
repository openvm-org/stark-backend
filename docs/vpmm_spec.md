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

## High‑Level Design

1. **Reserve** a sufficiently large VA range once at initialization.
2. **Pre‑allocate** a configurable number of physical pages.
3. **Map** pages contiguously at the **end of active space** for new allocations.
4. **Find‑or‑defragment**: if no free region fits, **remap** pages from earlier freed regions to the end (no data copy, just page table updates).
5. **Grow**: if still insufficient, **allocate more pages** and map them at the end.

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

## Initialization

* **VA Reservation:** Reserve a large VA range at startup. **Current value:** 1 TB (constant).
* **Page Size:** Configurable via `VPMM_PAGE_SIZE`. Must be a multiple of CUDA’s minimum allocation granularity (typically 2 MB). Larger pages reduce mapping overhead; all allocations are rounded up to page size.
* **Initial Pages:** Count is configurable via `VPMM_PAGES`. Defaults are derived from available device memory (e.g., fraction of `cudaMemGetInfo()` free memory), divided by expected parallelism. More pages improve performance but increase baseline memory footprint.
* **Mapping Unit:** All mappings are performed **by page**.

## Allocation & Growth

Allocations occur during initialization (preallocation) and on‑demand when the pool runs short of free pages.

* **Synchronous:** The CUDA Driver API performs allocations synchronously.
* **Granularity:** Allocation is **by page**.
* **MinUse policy:** When the pool is out of pages, allocate only as many pages as required to satisfy the current request.
* **Lifetime:** Allocated pages are **retained** for the lifetime of the process (not returned to the OS).

## Defragmentation

Triggered when no free region can satisfy a request.

* **Remapping:** Select a free region and **remap its pages** to the end of active space. The original mapping remains valid during in‑flight use on the stream. After remapping, the original region becomes a **dead zone** and is not reused for new allocations. Note: unmapping may return `cudaErrorInvalidAddress` if attempted prematurely.
* **Region Selection:** **BiggestFirst** — prefer defragmenting larger regions to minimize the number of defragmented segments.

## malloc(size) — Allocation Policy

1. **Find:** Search `free_regions` for a region large enough to satisfy the request.
2. **Defragment:** If none found, repeatedly remap free regions (as above) until a region fits.
3. **Grow:** If still insufficient, **allocate** the remaining number of pages and map them after `curr_end`.

Additional rules:

* **BestFit:** Among multiple fitting free regions, choose the smallest that satisfies the request.
* **Alignment:** All allocations are **rounded up** to page‑size multiples. A page is either entirely free or entirely used.
* **Small Buffers:** If `size < page_size`, bypass the VM pool and call `cudaMallocAsync` instead (preserves compatibility; setting `VMM_PAGE_SIZE = usize::MAX` effectively disables the pool for typical sizes).

## free(ptr)

* Look up `ptr` in `used_regions` to obtain the region size.
* Mark the region free in `free_regions` and **coalesce** with adjacent free neighbors.

## Status & Observability

(All tracking is implemented in the outer **MemoryManager**, not inside `VirtualMemoryPool`, since small buffers bypass the pool.)

* **Total GPU memory used by pages:** `pool.active_pages.len() * pool.page_size`
* **Active VA extent:** `pool.curr_end - pool.base`
* **Currently allocated (live) bytes:** `sum(pool.used_regions.values())`
* **Currently freed (reusable) bytes:** `sum(pool.free_regions.values())`

## Asynchrony & Streams

* VPMM assumes all operations occur on **the same stream** (`cudaStreamPerThread`).
* Multi‑stream scenarios are out of scope and would require per‑stream tracking of pending frees or separate pools.
