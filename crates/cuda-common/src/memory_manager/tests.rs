//! Tests for memory_manager - focused on edge cases and dangerous scenarios

use super::{
    d_free, d_malloc_on,
    vm_pool::{VirtualMemoryPool, VpmmConfig},
};
use crate::{
    d_buffer::DeviceBuffer,
    error::MemoryError,
    stream::{GpuDeviceCtx, StreamGuard},
};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemGetInfo(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;
}

fn get_gpu_free_memory() -> usize {
    let mut free = 0usize;
    let mut total = 0usize;
    let err = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    assert_eq!(err, 0, "cudaMemGetInfo failed: {}", err);
    free
}

fn test_ctx() -> GpuDeviceCtx {
    GpuDeviceCtx::for_current_device().unwrap()
}

fn test_stream() -> StreamGuard {
    test_ctx().stream
}

// ============================================================================
// Coalescing: free B first, then A, then C - should coalesce into one region
// ============================================================================
#[test]
fn test_coalescing_via_combined_alloc() {
    let ctx = test_ctx();
    let len = 2 << 30; // 2 GB per allocation
    let buf_a = DeviceBuffer::<u8>::with_capacity_on(len, &ctx);
    let buf_b = DeviceBuffer::<u8>::with_capacity_on(len, &ctx);
    let buf_c = DeviceBuffer::<u8>::with_capacity_on(len, &ctx);

    let addr_a = buf_a.as_raw_ptr();

    // Free in order: B, A, C - this tests both next and prev neighbor coalescing
    drop(buf_b);
    drop(buf_a);
    drop(buf_c);

    // Request combined size - if coalescing worked, this should reuse from A's start
    let combined_len = 3 * len;
    let buf_combined = DeviceBuffer::<u8>::with_capacity_on(combined_len, &ctx);
    assert_eq!(
        addr_a,
        buf_combined.as_raw_ptr(),
        "Should reuse coalesced region starting at A"
    );
}

// ============================================================================
// VA exhaustion: use tiny VA size to force multiple VA reservations
// ============================================================================
#[test]
fn test_va_exhaustion_reserves_more() {
    // Create pool with very small VA (4 MB) - will exhaust quickly
    let config = VpmmConfig {
        page_size: None,
        va_size: 4 << 20, // 4 MB VA per chunk
        initial_pages: 0,
    };
    let mut pool = VirtualMemoryPool::new(config);

    if pool.page_size == usize::MAX {
        println!("VPMM not supported, skipping test");
        return;
    }

    let page_size = pool.page_size;
    let stream = test_stream();

    // Initial state: 1 VA root
    assert_eq!(pool.roots.len(), 1);

    // Allocate enough pages to exhaust first VA chunk and trigger second reservation
    // 4MB VA / 2MB page = 2 pages max in first chunk
    let mut ptrs = Vec::new();
    for _ in 0..4 {
        // Allocate 4 pages total → needs 2 VA chunks
        match pool.malloc_internal(page_size, &stream) {
            Ok(ptr) => ptrs.push(ptr),
            Err(e) => panic!("Allocation failed: {:?}", e),
        }
    }

    assert!(
        pool.roots.len() >= 2,
        "Should have reserved additional VA chunks. Got {} roots",
        pool.roots.len()
    );

    // Cleanup
    for ptr in ptrs {
        pool.free_internal(ptr).unwrap();
    }
}

// ============================================================================
// Defragmentation scenario from vpmm_spec.md:
//   +10  >  +1  >  -10  >  +4  >  +11 (in units of PAGE_SIZE)
//
// X = PAGES - 11 determines behavior:
//   Case A: X ≥ 11 (PAGES ≥ 22) - enough free pages, no defrag
//   Case B: 5 ≤ X < 11 (16 ≤ PAGES < 22) - defrag for +11, no new pages
//   Case C: X == 4 (PAGES = 15) - defrag + allocate 1 new page
//   Case D: 0 ≤ X < 4 (11 ≤ PAGES < 15) - different layout, defrag + new pages
// ============================================================================

/// Helper to run the doc scenario and return final state
fn run_doc_scenario(
    initial_pages: usize,
) -> (
    VirtualMemoryPool,
    usize,                 // page_size
    *mut std::ffi::c_void, // ptr_1 (kept)
    *mut std::ffi::c_void, // ptr_4
    *mut std::ffi::c_void, // ptr_11
) {
    let config = VpmmConfig {
        page_size: None,  // Use device granularity
        va_size: 1 << 30, // 1 GB VA space
        initial_pages,
    };
    let mut pool = VirtualMemoryPool::new(config);

    if pool.page_size == usize::MAX {
        panic!("VPMM not supported");
    }

    let page_size = pool.page_size;
    let stream = test_stream();

    // Step 1: +10 pages
    let ptr_10 = pool.malloc_internal(10 * page_size, &stream).unwrap();
    assert!(!ptr_10.is_null());

    // Step 2: +1 page
    let ptr_1 = pool.malloc_internal(page_size, &stream).unwrap();
    assert!(!ptr_1.is_null());
    // Should be right after the 10-page allocation
    assert_eq!(ptr_1 as usize, ptr_10 as usize + 10 * page_size);

    // Step 3: -10 pages
    pool.free_internal(ptr_10).unwrap();

    // Step 4: +4 pages
    let ptr_4 = pool.malloc_internal(4 * page_size, &stream).unwrap();
    assert!(!ptr_4.is_null());

    // Step 5: +11 pages
    let ptr_11 = pool.malloc_internal(11 * page_size, &stream).unwrap();
    assert!(!ptr_11.is_null());

    (pool, page_size, ptr_1, ptr_4, ptr_11)
}

#[test]
fn test_defrag_case_a_enough_free_pages() {
    // Case A: X ≥ 11, so PAGES ≥ 22
    // After +10 +1 we use 11 pages, leaving X=11 free
    // +4 takes from the freed 10-page region (best fit)
    // +11 can fit in remaining preallocated space
    let initial_pages = 22; // X = 22 - 11 = 11

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);

    // Memory usage should be exactly 22 pages (no new allocation needed)
    assert_eq!(
        pool.memory_usage(),
        22 * page_size,
        "Case A: no new pages allocated"
    );

    // Step 4 layout: [+4][-6][1][-X] - 4 takes start of freed 10
    assert_eq!(ptr_4 as usize, pool.roots[0] as usize, "4 at VA start");

    // Step 5 layout: [4][-6][1][+11][...] - 11 goes after the 1
    assert!(
        ptr_11 as usize > ptr_1 as usize,
        "Case A: 11 should be after 1 (no defrag)"
    );

    // Cleanup
    let mut pool = pool;
    pool.free_internal(ptr_1).unwrap();
    pool.free_internal(ptr_4).unwrap();
    pool.free_internal(ptr_11).unwrap();
}

#[test]
fn test_defrag_case_b_defrag_no_new_pages() {
    // Case B: 5 ≤ X < 11, so 16 ≤ PAGES < 22
    // After +10 +1, we have X free pages (5 ≤ X < 11)
    // +4 goes after 1 (fits in X pages)
    // +11 needs defrag: remap the 10-page free region
    let initial_pages = 18; // X = 18 - 11 = 7

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);

    // Memory usage should still be 18 pages (defrag reuses existing)
    assert_eq!(
        pool.memory_usage(),
        18 * page_size,
        "Case B: no new pages allocated"
    );

    // In Case B, +4 goes after 1: [-10][1][+4][-(X-4)]
    assert_eq!(
        ptr_4 as usize,
        ptr_1 as usize + page_size,
        "Case B: 4 right after 1"
    );

    // Cleanup
    let mut pool = pool;
    pool.free_internal(ptr_1).unwrap();
    pool.free_internal(ptr_4).unwrap();
    pool.free_internal(ptr_11).unwrap();
}

#[test]
fn test_defrag_case_c_defrag_plus_new_page() {
    // Case C: X == 4, so PAGES = 15
    // After +10 +1, we have exactly 4 free pages
    // +4 takes all free pages: [-10][1][+4] (no leftover)
    // +11 needs defrag (remap 10) + allocate 1 new page
    let initial_pages = 15; // X = 15 - 11 = 4

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);

    // Memory usage: 15 original + 1 new = 16 pages
    assert_eq!(
        pool.memory_usage(),
        16 * page_size,
        "Case C: 1 new page allocated"
    );

    // +4 goes after 1 (uses all remaining X=4 pages)
    assert_eq!(
        ptr_4 as usize,
        ptr_1 as usize + page_size,
        "Case C: 4 right after 1"
    );

    // Cleanup
    let mut pool = pool;
    pool.free_internal(ptr_1).unwrap();
    pool.free_internal(ptr_4).unwrap();
    pool.free_internal(ptr_11).unwrap();
}

#[test]
fn test_defrag_case_d_not_enough_for_4() {
    // Case D: 0 ≤ X < 4, so 11 ≤ PAGES < 15
    // After +10 +1, we have X < 4 free pages
    // +4 cannot fit after 1, so it takes from freed 10: [+4][-6][1][-X]
    // +11 needs defrag of the 6 + allocate (11-X-6) new pages
    let initial_pages = 12; // X = 12 - 11 = 1

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);

    // Memory usage: need 11 more pages but only have 6+1=7 free
    // So allocate 11-7=4 new pages → 12 + 4 = 16 total
    assert_eq!(
        pool.memory_usage(),
        16 * page_size,
        "Case D: 4 new pages allocated"
    );

    // +4 at VA start (takes from freed 10 since X < 4)
    assert_eq!(
        ptr_4 as usize, pool.roots[0] as usize,
        "Case D: 4 at VA start"
    );

    // Cleanup
    let mut pool = pool;
    pool.free_internal(ptr_1).unwrap();
    pool.free_internal(ptr_4).unwrap();
    pool.free_internal(ptr_11).unwrap();
}

// ============================================================================
// Mixed allocations: small (cudaMallocAsync) and large (VPMM) across threads
// ============================================================================
#[test]
fn test_mixed_allocations() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let mut handles = Vec::new();

        for thread_idx in 0..4 {
            let handle = tokio::task::spawn_blocking(move || {
                let ctx = test_ctx();
                let mut buffers: Vec<DeviceBuffer<u8>> = Vec::new();

                for op in 0..15 {
                    let len = if op % 3 == 0 {
                        // Small: 1KB - 100KB (cudaMallocAsync path)
                        ((thread_idx + 1) * (op + 1) * 1024) % (100 << 10) + 1024
                    } else {
                        // Large: 100MB - 400MB (VPMM path)
                        ((thread_idx + 1) * (op + 1) % 4 + 1) * (100 << 20)
                    };

                    let buf = DeviceBuffer::<u8>::with_capacity_on(len, &ctx);
                    buffers.push(buf);

                    if op % 2 == 0 && !buffers.is_empty() {
                        buffers.remove(0);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("thread failed");
        }
    });

    // Verify pool is functional after mixed operations
    let ctx = test_ctx();
    ctx.stream.synchronize().expect("stream sync failed");
    let large = DeviceBuffer::<u8>::with_capacity_on(1 << 30, &ctx);
    assert!(
        !large.as_ptr().is_null(),
        "Large allocation should work after mixed operations"
    );
}

// ============================================================================
// OOM recovery: allocator should still work after an OOM event
// ============================================================================
#[test]
#[ignore] // Heavy: intentionally exhausts GPU memory
fn test_oom_recovery_after_error() {
    let ctx = test_ctx();

    // Allocate large chunks to drive the device near OOM. Keep them alive to
    // simulate a server that retains existing allocations.
    let chunk_size = 2 << 30; // 2 GB
    let mut buffers: Vec<*mut std::ffi::c_void> = Vec::new();

    // Fill GPU memory until we hit OOM on a 2 GB request
    loop {
        match d_malloc_on(chunk_size, &ctx.stream) {
            Ok(ptr) => buffers.push(ptr),
            Err(MemoryError::OutOfMemory { .. }) => break,
            Err(e) => panic!("Expected OOM, got {:?}", e),
        }
    }

    // After OOM, allocator should still succeed for smaller requests without
    // freeing the large buffers we already hold.
    let small = d_malloc_on(1 << 20, &ctx.stream).expect("Small allocation after OOM failed");

    // Request just under the currently free memory to exercise the pool path.
    let free_after_oom = get_gpu_free_memory();
    let safety = 64 << 20; // leave 64 MB headroom to avoid rounding issues
    assert!(
        free_after_oom > safety + (2 << 20),
        "Not enough free memory to attempt medium alloc after OOM"
    );
    let medium_req = free_after_oom - safety;
    let medium = d_malloc_on(medium_req, &ctx.stream).expect("Pool allocation after OOM failed");

    // Cleanup
    unsafe { d_free(small).unwrap() };
    unsafe { d_free(medium).unwrap() };
    for ptr in buffers {
        unsafe { d_free(ptr).unwrap() };
    }
    ctx.stream.synchronize().expect("stream sync after cleanup");
}

// ============================================================================
// Edge Case Tests for Defragmentation
// ============================================================================

// Helper to create a small test pool
fn create_test_pool(initial_pages: usize) -> VirtualMemoryPool {
    let config = VpmmConfig {
        page_size: None,  // Use device granularity
        va_size: 1 << 30, // 1 GB VA space
        initial_pages,
    };
    let pool = VirtualMemoryPool::new(config);
    if pool.page_size == usize::MAX {
        panic!("VPMM not supported, cannot run this test");
    }
    pool
}

// ============================================================================
// Test: Multiple consecutive defragmentation operations
// ============================================================================
#[test]
fn test_multiple_defrag_cycles() {
    let mut pool = create_test_pool(8);
    let page_size = pool.page_size;
    let stream = test_stream();

    for cycle in 0..3 {
        // Create fragmentation
        let ptrs: Vec<_> = (0..4)
            .map(|_| pool.malloc_internal(2 * page_size, &stream).unwrap())
            .collect();

        // Free alternating
        pool.free_internal(ptrs[0]).unwrap();
        pool.free_internal(ptrs[2]).unwrap();

        // Request contiguous that requires defrag
        let ptr_big = pool.malloc_internal(4 * page_size, &stream).unwrap();
        assert!(!ptr_big.is_null(), "Cycle {} failed to allocate", cycle);

        // Cleanup for next cycle
        pool.free_internal(ptrs[1]).unwrap();
        pool.free_internal(ptrs[3]).unwrap();
        pool.free_internal(ptr_big).unwrap();
    }
}

// ============================================================================
// Additional Edge Case Tests for Complete Coverage
// ============================================================================

// ============================================================================
// Test: Comprehensive coalescing verification for prev, next, and both neighbors
// Tests that merged regions have correct sizes and addresses
// ============================================================================
#[test]
fn test_coalesce_all_neighbor_cases() {
    let mut pool = create_test_pool(6);
    let page_size = pool.page_size;
    let stream = test_stream();

    // --- Case 1: Coalesce with PREV neighbor ---
    // Allocate: [A(2)][B(2)][C(2)]
    let ptr_a = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, &stream).unwrap();

    // Free B, then A -> A merges with B (prev neighbor)
    pool.free_internal(ptr_b).unwrap();
    pool.free_internal(ptr_a).unwrap();

    // Verify 4 pages available from coalesced A+B
    let ptr_4 = pool.malloc_internal(4 * page_size, &stream).unwrap();
    assert_eq!(ptr_4 as usize, ptr_a as usize, "Prev: should start at A");
    assert_eq!(pool.memory_usage(), 6 * page_size, "Prev: no new alloc");

    // Reset for next case
    pool.free_internal(ptr_c).unwrap();
    pool.free_internal(ptr_4).unwrap();

    // --- Case 2: Coalesce with NEXT neighbor ---
    let ptr_a = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, &stream).unwrap();

    // Free A, then B -> B merges with A (next neighbor from A's POV)
    pool.free_internal(ptr_a).unwrap();
    pool.free_internal(ptr_b).unwrap();

    let ptr_4 = pool.malloc_internal(4 * page_size, &stream).unwrap();
    assert_eq!(ptr_4 as usize, ptr_a as usize, "Next: should start at A");

    // Reset for next case
    pool.free_internal(ptr_c).unwrap();
    pool.free_internal(ptr_4).unwrap();

    // --- Case 3: Coalesce with BOTH neighbors ---
    let ptr_a = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, &stream).unwrap();

    // Free A, C, then B (middle) -> B merges with both
    pool.free_internal(ptr_a).unwrap();
    pool.free_internal(ptr_c).unwrap();
    pool.free_internal(ptr_b).unwrap();

    let ptr_6 = pool.malloc_internal(6 * page_size, &stream).unwrap();
    assert_eq!(ptr_6 as usize, ptr_a as usize, "Both: should start at A");
    assert_eq!(pool.memory_usage(), 6 * page_size, "Both: no new alloc");

    pool.free_internal(ptr_6).unwrap();
}

// ============================================================================
// Test: NO coalescing across different streams; defrag still works
// ============================================================================
#[test]
fn test_no_coalesce_across_streams() {
    let mut pool = create_test_pool(6);
    let page_size = pool.page_size;
    let stream_1 = test_stream();
    let stream_2 = test_stream();

    // Allocate: [A(2)][B(2)][C(2)]
    let ptr_a = pool.malloc_internal(2 * page_size, &stream_1).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, &stream_1).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, &stream_1).unwrap();

    // B keeps its own recorded stream, so it should NOT coalesce with A or C.
    pool.free_internal(ptr_a).unwrap();
    pool.free_internal(ptr_b).unwrap();
    pool.free_internal(ptr_c).unwrap();

    // Requesting 4 pages needs defrag to combine A + C (skipping B)
    let memory_before = pool.memory_usage();
    let ptr_4 = pool.malloc_internal(4 * page_size, &stream_2).unwrap();
    assert!(!ptr_4.is_null());
    assert_eq!(pool.memory_usage(), memory_before, "Defrag reused existing");

    pool.free_internal(ptr_4).unwrap();
}

// ============================================================================
// Test: Tail free regions are properly returned and usable after defrag
// Covers: oldest-first ordering, partial consumption, tail verification
// ============================================================================
#[test]
fn test_defrag_tail_regions_returned() {
    let mut pool = create_test_pool(10);
    let page_size = pool.page_size;
    let stream = test_stream();

    // Allocate 5 x 2-page regions
    let ptrs: Vec<_> = (0..5)
        .map(|_| pool.malloc_internal(2 * page_size, &stream).unwrap())
        .collect();

    // Free alternating to create fragmentation with 6 free pages in 3 regions
    // Layout: [2 free][2 alloc][2 free][2 alloc][2 free]
    pool.free_internal(ptrs[0]).unwrap();
    pool.free_internal(ptrs[2]).unwrap();
    pool.free_internal(ptrs[4]).unwrap();

    // Request 5 pages -> defrag takes 2+2+1, leaving 1 page as tail
    let memory_before = pool.memory_usage();
    let ptr_5 = pool.malloc_internal(5 * page_size, &stream).unwrap();
    assert_eq!(pool.memory_usage(), memory_before, "No new pages needed");

    // Verify tail exists: can allocate 1 more without new allocation
    let ptr_1 = pool.malloc_internal(page_size, &stream).unwrap();
    assert_eq!(pool.memory_usage(), memory_before, "Tail page reused");

    // Now requesting more requires new allocation
    let ptr_extra = pool.malloc_internal(page_size, &stream).unwrap();
    assert_eq!(
        pool.memory_usage(),
        memory_before + page_size,
        "New page allocated"
    );

    // Cleanup
    pool.free_internal(ptrs[1]).unwrap();
    pool.free_internal(ptrs[3]).unwrap();
    pool.free_internal(ptr_5).unwrap();
    pool.free_internal(ptr_1).unwrap();
    pool.free_internal(ptr_extra).unwrap();
}

// ============================================================================
// Test: Unmapped region coalescing during remap operations
// ============================================================================
#[test]
fn test_unmapped_region_coalescing_comprehensive() {
    let mut pool = create_test_pool(6);
    let page_size = pool.page_size;
    let stream = test_stream();

    // Allocate: [A(2)][B(2)][C(2)]
    let ptr_a = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, &stream).unwrap();

    // Free A and B -> should coalesce to 4 free pages
    pool.free_internal(ptr_a).unwrap();
    pool.free_internal(ptr_b).unwrap();

    // Request 4 pages from coalesced region
    let ptr_4 = pool.malloc_internal(4 * page_size, &stream).unwrap();
    assert_eq!(ptr_4 as usize, ptr_a as usize);

    // Free everything
    pool.free_internal(ptr_4).unwrap();
    pool.free_internal(ptr_c).unwrap();

    // Request 7 pages -> needs remap + 1 new page
    // Unmapped regions from remap should coalesce properly
    let memory_before = pool.memory_usage();
    let ptr_7 = pool.malloc_internal(7 * page_size, &stream).unwrap();
    assert_eq!(pool.memory_usage(), memory_before + page_size);

    pool.free_internal(ptr_7).unwrap();
}

// ============================================================================
// Test: Defrag with new pages merging with existing free regions
// ============================================================================
#[test]
fn test_defrag_new_pages_merge_with_existing() {
    let config = VpmmConfig {
        page_size: None,
        va_size: 1 << 30,
        initial_pages: 0,
    };
    let mut pool = VirtualMemoryPool::new(config);

    if pool.page_size == usize::MAX {
        println!("VPMM not supported, skipping test");
        return;
    }

    let page_size = pool.page_size;
    let stream = test_stream();

    // Allocate 2 pages, then free them
    let ptr_2 = pool.malloc_internal(2 * page_size, &stream).unwrap();
    assert_eq!(pool.memory_usage(), 2 * page_size);
    pool.free_internal(ptr_2).unwrap();

    // Request 4 pages -> allocates 2 new + uses 2 free (merge)
    let ptr_4 = pool.malloc_internal(4 * page_size, &stream).unwrap();
    assert_eq!(pool.memory_usage(), 4 * page_size);
    assert!(!ptr_4.is_null());

    // Test with preallocated pool: alloc merges with prev free
    pool.free_internal(ptr_4).unwrap();

    let ptr_a = pool.malloc_internal(2 * page_size, &stream).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, &stream).unwrap();
    pool.free_internal(ptr_a).unwrap();

    // Request 4 pages with 2 free -> need 2 more
    let ptr_req = pool.malloc_internal(4 * page_size, &stream).unwrap();
    assert!(!ptr_req.is_null());
    assert_eq!(pool.memory_usage(), 6 * page_size);

    pool.free_internal(ptr_b).unwrap();
    pool.free_internal(ptr_req).unwrap();
}

// ============================================================================
// Test: Multiple defrag scenarios in sequence
// Covers: exact fit (no tail), remap with new page allocation, combining regions
// ============================================================================
#[test]
fn test_defrag_various_scenarios() {
    let mut pool = create_test_pool(10);
    let page_size = pool.page_size;
    let stream = test_stream();

    // --- Scenario 1: Exact fit, no tail ---
    let ptrs: Vec<_> = (0..5)
        .map(|_| pool.malloc_internal(2 * page_size, &stream).unwrap())
        .collect();

    // Free all in scattered order -> should coalesce completely
    pool.free_internal(ptrs[1]).unwrap();
    pool.free_internal(ptrs[3]).unwrap();
    pool.free_internal(ptrs[0]).unwrap();
    pool.free_internal(ptrs[4]).unwrap();
    pool.free_internal(ptrs[2]).unwrap();

    // Request all 10 pages (exact fit)
    let ptr_10 = pool.malloc_internal(10 * page_size, &stream).unwrap();
    assert!(!ptr_10.is_null());
    assert_eq!(pool.memory_usage(), 10 * page_size);

    // --- Scenario 2: Remap with new page allocation (reuse same pool state) ---
    // Layout now: [10 malloc]
    // Free the 10-page region, then create fragmentation
    pool.free_internal(ptr_10).unwrap();

    // Allocate 6 + 4 pages
    let ptr_a = pool.malloc_internal(6 * page_size, &stream).unwrap();
    let ptr_b = pool.malloc_internal(4 * page_size, &stream).unwrap();

    // Free A, keep B -> [6 free][4 malloc]
    pool.free_internal(ptr_a).unwrap();

    // Request 7 pages: 6 free + need 1 more
    let memory_before = pool.memory_usage();
    let ptr_7 = pool.malloc_internal(7 * page_size, &stream).unwrap();
    assert_eq!(pool.memory_usage(), memory_before + page_size);

    pool.free_internal(ptr_b).unwrap();
    pool.free_internal(ptr_7).unwrap();
}

// ============================================================================
// Release-stream handoff
//
// A buffer produced on one stream and consumed on another is released according
// to the stream recorded against it. While that record names the *producer*, the
// release is enqueued on a stream that has already drained, so the allocation
// becomes reusable by any sibling stream immediately — including while the
// consumer is still reading it. These tests pin down both the metadata change
// and the reuse it prevents.
// ============================================================================

use std::{
    ffi::c_void,
    sync::{Arc, Condvar, Mutex},
};

use super::small_alloc_release_stream;
use crate::{copy::cudaMemcpyKind, stream::cudaStream_t};

#[link(name = "cudart")]
extern "C" {
    fn cudaMemsetAsync(dst: *mut c_void, value: i32, count: usize, stream: cudaStream_t) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> i32;
    fn cudaLaunchHostFunc(
        stream: cudaStream_t,
        func: extern "C" fn(*mut c_void),
        user_data: *mut c_void,
    ) -> i32;
}

/// What A writes into the trace, and what B must still see when it reads.
const PRODUCED_BYTE: u8 = 0xAB;
/// What the sibling stream writes if it is allowed to reuse the allocation.
const OVERWRITE_BYTE: u8 = 0x5C;

fn cuda_ok(code: i32, what: &str) {
    assert_eq!(code, 0, "{what} failed with CUDA error {code}");
}

/// Holds a CUDA stream at a chosen point until the host lets it go.
///
/// `cudaLaunchHostFunc` runs in stream order, so everything queued behind it
/// waits for it to return. Blocking inside it is what makes the schedules below
/// exact: no sleeps, no polling, and no dependence on how long any copy happens
/// to take. Nothing this callback waits on is itself CUDA work, so it cannot
/// deadlock against the device.
struct StreamGate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl StreamGate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            open: Mutex::new(false),
            changed: Condvar::new(),
        })
    }

    /// Queues the gate on `stream`. The returned `Arc` must outlive the stream's
    /// work; every caller here synchronizes the stream before dropping it.
    fn hold(self: &Arc<Self>, stream: &StreamGuard) {
        extern "C" fn wait_for_gate(data: *mut c_void) {
            // SAFETY: `hold`'s caller keeps the `Arc` alive until the stream has
            // been synchronized, so this pointer is valid whenever CUDA runs it.
            let gate = unsafe { &*(data as *const StreamGate) };
            let mut open = gate.open.lock().unwrap();
            while !*open {
                open = gate.changed.wait(open).unwrap();
            }
        }
        cuda_ok(
            unsafe {
                cudaLaunchHostFunc(
                    stream.as_raw(),
                    wait_for_gate,
                    Arc::as_ptr(self) as *mut c_void,
                )
            },
            "cudaLaunchHostFunc",
        );
    }

    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

/// What one run of the cross-stream schedule observed.
#[derive(Debug)]
struct CrossStreamOutcome {
    /// The sibling was handed the very address the consumer is still reading.
    reused_the_live_address: bool,
    /// The freed region's reuse event had already completed while the consumer's
    /// read was still queued behind the gate — i.e. reuse was unguarded.
    reusable_while_consumer_reads: bool,
    /// The freed region's release is ordered on the consumer's stream.
    released_on_consumer_stream: bool,
    /// The consumer read back exactly what the producer wrote.
    consumer_read_intact: bool,
    /// The sibling was served out of the same physical pages, with none added.
    physical_pages_reused: bool,
}

/// Runs the scheduler's shape end to end on a pool sized so that the trace's own
/// pages are the *only* thing that can satisfy the sibling's request.
///
/// `adopt` is a test-local switch for skipping the handoff, which is exactly the
/// pre-fix behaviour. It exists only here — there is no production feature,
/// environment variable or configuration field that turns the handoff off.
fn run_cross_stream_reuse(adopt: bool) -> CrossStreamOutcome {
    const PAGES: usize = 2;

    // A produces, B consumes, C is the sibling slot proving alongside B.
    let producer_ctx = test_ctx();
    let consumer_ctx = test_ctx();
    let sibling_ctx = test_ctx();
    let producer = producer_ctx.stream.clone();
    let consumer = consumer_ctx.stream.clone();
    let sibling = sibling_ctx.stream.clone();

    let mut pool = create_test_pool(PAGES);
    let size = PAGES * pool.page_size;

    // Produce the trace on A, then synchronize A. This is the tracegen fence:
    // by the time the context is handed on, nothing is outstanding on A.
    let trace = pool.malloc_internal(size, &producer).unwrap();
    cuda_ok(
        unsafe { cudaMemsetAsync(trace, PRODUCED_BYTE as i32, size, producer.as_raw()) },
        "cudaMemsetAsync(produce)",
    );
    producer.synchronize().unwrap();

    // B's landing buffer comes from the global manager, so it never competes for
    // this pool's pages and cannot perturb the reuse decision under test.
    let landing = DeviceBuffer::<u8>::with_capacity_on(size, &consumer_ctx);

    if adopt {
        let previous = pool.rebind_release_stream(trace, &consumer).unwrap();
        assert_eq!(
            previous, producer,
            "the handoff must report the producer as the previous release stream"
        );
        assert_eq!(
            pool.release_stream_of(trace).unwrap(),
            consumer,
            "the live allocation's record must now name the consumer's stream"
        );
    }

    // B reads the trace, held at the gate so the read is provably still pending
    // for the rest of the schedule.
    let gate = StreamGate::new();
    gate.hold(&consumer);
    cuda_ok(
        unsafe {
            cudaMemcpyAsync(
                landing.as_mut_ptr() as *mut c_void,
                trace,
                size,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                consumer.as_raw(),
            )
        },
        "cudaMemcpyAsync(consume)",
    );

    // The sole owner drops. The release is enqueued on the recorded stream.
    pool.free_internal(trace).unwrap();
    // A is idle in both arms and never depends on B, so this is free here. It
    // makes the freed region's event state a settled fact rather than a query
    // racing the driver's own processing of the record.
    producer.synchronize().unwrap();

    let (region_stream, region_event) = pool
        .free_region_release(trace)
        .expect("the freed trace must leave exactly one free region at its address");
    let released_on_consumer_stream = region_stream == consumer;
    let reusable_while_consumer_reads = region_event.completed();

    // The sibling asks for memory. This pool holds exactly the trace's pages, so
    // the region B is reading is the only thing that can satisfy the request.
    let mapped_before = pool.memory_usage();
    let sibling_ptr = pool.malloc_internal(size, &sibling).unwrap();
    let physical_pages_reused = pool.memory_usage() == mapped_before;
    let reused_the_live_address = sibling_ptr == trace;
    cuda_ok(
        unsafe { cudaMemsetAsync(sibling_ptr, OVERWRITE_BYTE as i32, size, sibling.as_raw()) },
        "cudaMemsetAsync(overwrite)",
    );

    if reused_the_live_address {
        // The sibling holds the address B is still reading. Drain it before
        // opening the gate so the overwrite is a fact on the memory rather than
        // a race this run happened to win.
        sibling.synchronize().unwrap();
    }
    gate.release();
    consumer.synchronize().unwrap();
    sibling.synchronize().unwrap();

    let mut host = vec![0u8; size];
    cuda_ok(
        unsafe {
            cudaMemcpyAsync(
                host.as_mut_ptr() as *mut c_void,
                landing.as_ptr() as *const c_void,
                size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                consumer.as_raw(),
            )
        },
        "cudaMemcpyAsync(readback)",
    );
    consumer.synchronize().unwrap();
    let consumer_read_intact = host.iter().all(|&byte| byte == PRODUCED_BYTE);

    pool.free_internal(sibling_ptr).unwrap();

    CrossStreamOutcome {
        reused_the_live_address,
        reusable_while_consumer_reads,
        released_on_consumer_stream,
        consumer_read_intact,
        physical_pages_reused,
    }
}

// ============================================================================
// Test: adopting the consumer's stream defers reuse until its reads are done
// ============================================================================
#[test]
fn handoff_defers_sibling_reuse_until_the_consumer_has_read() {
    let outcome = run_cross_stream_reuse(true);

    assert!(
        outcome.physical_pages_reused,
        "the fixture is vacuous unless the sibling is served from the trace's own \
         pages: {outcome:?}"
    );
    assert!(
        outcome.released_on_consumer_stream,
        "the freed region's release must be ordered on the consumer's stream: {outcome:?}"
    );
    assert!(
        !outcome.reusable_while_consumer_reads,
        "the freed region must not look reusable while the consumer's read is \
         still queued: {outcome:?}"
    );
    assert!(
        !outcome.reused_the_live_address,
        "the sibling must not be handed the address the consumer is reading: {outcome:?}"
    );
    assert!(
        outcome.consumer_read_intact,
        "the consumer must read what the producer wrote: {outcome:?}"
    );
}

// ============================================================================
// Test (negative control): the same schedule without the handoff corrupts the
// consumer's read. Skipping the handoff is what the code did before the fix, so
// this is the shape of the defect itself, not a synthetic hazard.
// ============================================================================
#[test]
fn without_the_handoff_a_sibling_overwrites_a_live_trace() {
    let outcome = run_cross_stream_reuse(false);

    assert!(
        outcome.physical_pages_reused,
        "the fixture is vacuous unless the sibling is served from the trace's own \
         pages: {outcome:?}"
    );
    assert!(
        !outcome.released_on_consumer_stream,
        "without the handoff the release stays on the producer's stream: {outcome:?}"
    );
    assert!(
        outcome.reusable_while_consumer_reads,
        "without the handoff the region looks reusable the moment the producer \
         drains, which is the defect: {outcome:?}"
    );
    assert!(
        outcome.reused_the_live_address,
        "without the handoff the sibling is handed the consumer's live address: {outcome:?}"
    );
    assert!(
        !outcome.consumer_read_intact,
        "NEGATIVE CONTROL FAILED: the sibling overwrote the trace the consumer was \
         still reading, so the consumer must not read back the produced bytes. If \
         this passes, the schedule above is not reproducing the race and the \
         positive test proves nothing: {outcome:?}"
    );
}

// ============================================================================
// Test: the VPMM record follows the handoff, and the free lands on it
// ============================================================================
#[test]
fn vpmm_record_and_free_region_follow_the_handoff() {
    let producer = test_stream();
    let consumer = test_stream();

    let mut pool = create_test_pool(1);
    let size = pool.page_size;

    let ptr = pool.malloc_internal(size, &producer).unwrap();
    assert_eq!(pool.release_stream_of(ptr).unwrap(), producer);

    let previous = pool.rebind_release_stream(ptr, &consumer).unwrap();
    assert_eq!(previous, producer);
    assert_eq!(pool.release_stream_of(ptr).unwrap(), consumer);

    // The allocation is still live, so nothing describes it in `free_regions`
    // yet; the handoff must not have invented an entry there.
    assert!(pool.free_region_release(ptr).is_none());

    let (_, guard) = pool.free_internal(ptr).unwrap();
    assert_eq!(guard, consumer, "the free must be ordered on the consumer");
    let (region_stream, _) = pool.free_region_release(ptr).unwrap();
    assert_eq!(
        region_stream, consumer,
        "the freed region's reuse event must be recorded on the consumer"
    );

    // Rebinding a pointer the pool does not hold is rejected rather than silently
    // creating a record.
    assert!(matches!(
        pool.rebind_release_stream(ptr, &producer),
        Err(MemoryError::InvalidPointer)
    ));
}

// ============================================================================
// Test: the small (`cudaMallocAsync`) record follows the handoff too
// ============================================================================
#[test]
fn small_allocation_record_follows_the_handoff() {
    let producer = test_ctx();
    let consumer = test_ctx();

    // Far below any plausible pool page size, so this takes the small path.
    let mut buffer = DeviceBuffer::<u8>::with_capacity_on(256, &producer);
    let ptr = buffer.as_raw_ptr() as *mut c_void;
    assert_eq!(
        small_alloc_release_stream(ptr).expect("256 bytes must take the small path"),
        producer.stream,
        "a fresh allocation is released on the stream that made it"
    );

    unsafe { buffer.adopt_release_stream_after_sync(&consumer) }.unwrap();
    assert_eq!(
        small_alloc_release_stream(ptr).unwrap(),
        consumer.stream,
        "the small-path record must name the adopted stream"
    );

    drop(buffer);
    assert!(
        small_alloc_release_stream(ptr).is_none(),
        "the record is removed by the free"
    );
}

// ============================================================================
// Test: a null buffer has no allocation to hand off
// ============================================================================
#[test]
fn adopting_a_release_stream_on_an_empty_buffer_is_a_no_op() {
    let consumer = test_ctx();
    let mut buffer = DeviceBuffer::<u8>::new();
    unsafe { buffer.adopt_release_stream_after_sync(&consumer) }.unwrap();
}
