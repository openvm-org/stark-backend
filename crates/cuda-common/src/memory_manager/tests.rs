//! Tests for memory_manager - focused on edge cases and dangerous scenarios

use super::vm_pool::{VirtualMemoryPool, VpmmConfig};
use crate::{
    d_buffer::DeviceBuffer,
    stream::{current_stream_id, current_stream_sync},
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

// ============================================================================
// Coalescing: free B first, then A, then C - should coalesce into one region
// ============================================================================
#[test]
fn test_coalescing_via_combined_alloc() {
    let len = 2 << 30; // 2 GB per allocation
    let buf_a = DeviceBuffer::<u8>::with_capacity(len);
    let buf_b = DeviceBuffer::<u8>::with_capacity(len);
    let buf_c = DeviceBuffer::<u8>::with_capacity(len);

    let addr_a = buf_a.as_raw_ptr();

    // Free in order: B, A, C - this tests both next and prev neighbor coalescing
    drop(buf_b);
    drop(buf_a);
    drop(buf_c);

    // Request combined size - if coalescing worked, this should reuse from A's start
    let combined_len = 3 * len;
    let buf_combined = DeviceBuffer::<u8>::with_capacity(combined_len);
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
    let stream_id = current_stream_id().unwrap();

    // Initial state: 1 VA root
    assert_eq!(pool.roots.len(), 1);

    // Allocate enough pages to exhaust first VA chunk and trigger second reservation
    // 4MB VA / 2MB page = 2 pages max in first chunk
    let mut ptrs = Vec::new();
    for _ in 0..4 {
        // Allocate 4 pages total → needs 2 VA chunks
        match pool.malloc_internal(page_size, stream_id) {
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
        pool.free_internal(ptr, stream_id).unwrap();
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
    let stream_id = current_stream_id().unwrap();

    // Step 1: +10 pages
    let ptr_10 = pool.malloc_internal(10 * page_size, stream_id).unwrap();
    assert!(!ptr_10.is_null());

    // Step 2: +1 page
    let ptr_1 = pool.malloc_internal(page_size, stream_id).unwrap();
    assert!(!ptr_1.is_null());
    // Should be right after the 10-page allocation
    assert_eq!(ptr_1 as usize, ptr_10 as usize + 10 * page_size);

    // Step 3: -10 pages
    pool.free_internal(ptr_10, stream_id).unwrap();

    // Step 4: +4 pages
    let ptr_4 = pool.malloc_internal(4 * page_size, stream_id).unwrap();
    assert!(!ptr_4.is_null());

    // Step 5: +11 pages
    let ptr_11 = pool.malloc_internal(11 * page_size, stream_id).unwrap();
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
    let stream_id = current_stream_id().unwrap();

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
    pool.free_internal(ptr_1, stream_id).unwrap();
    pool.free_internal(ptr_4, stream_id).unwrap();
    pool.free_internal(ptr_11, stream_id).unwrap();
}

#[test]
fn test_defrag_case_b_defrag_no_new_pages() {
    // Case B: 5 ≤ X < 11, so 16 ≤ PAGES < 22
    // After +10 +1, we have X free pages (5 ≤ X < 11)
    // +4 goes after 1 (fits in X pages)
    // +11 needs defrag: remap the 10-page free region
    let initial_pages = 18; // X = 18 - 11 = 7

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);
    let stream_id = current_stream_id().unwrap();

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
    pool.free_internal(ptr_1, stream_id).unwrap();
    pool.free_internal(ptr_4, stream_id).unwrap();
    pool.free_internal(ptr_11, stream_id).unwrap();
}

#[test]
fn test_defrag_case_c_defrag_plus_new_page() {
    // Case C: X == 4, so PAGES = 15
    // After +10 +1, we have exactly 4 free pages
    // +4 takes all free pages: [-10][1][+4] (no leftover)
    // +11 needs defrag (remap 10) + allocate 1 new page
    let initial_pages = 15; // X = 15 - 11 = 4

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);
    let stream_id = current_stream_id().unwrap();

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
    pool.free_internal(ptr_1, stream_id).unwrap();
    pool.free_internal(ptr_4, stream_id).unwrap();
    pool.free_internal(ptr_11, stream_id).unwrap();
}

#[test]
fn test_defrag_case_d_not_enough_for_4() {
    // Case D: 0 ≤ X < 4, so 11 ≤ PAGES < 15
    // After +10 +1, we have X < 4 free pages
    // +4 cannot fit after 1, so it takes from freed 10: [+4][-6][1][-X]
    // +11 needs defrag of the 6 + allocate (11-X-6) new pages
    let initial_pages = 12; // X = 12 - 11 = 1

    let (pool, page_size, ptr_1, ptr_4, ptr_11) = run_doc_scenario(initial_pages);
    let stream_id = current_stream_id().unwrap();

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
    pool.free_internal(ptr_1, stream_id).unwrap();
    pool.free_internal(ptr_4, stream_id).unwrap();
    pool.free_internal(ptr_11, stream_id).unwrap();
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
                let mut buffers: Vec<DeviceBuffer<u8>> = Vec::new();

                for op in 0..15 {
                    let len = if op % 3 == 0 {
                        // Small: 1KB - 100KB (cudaMallocAsync path)
                        ((thread_idx + 1) * (op + 1) * 1024) % (100 << 10) + 1024
                    } else {
                        // Large: 100MB - 400MB (VPMM path)
                        ((thread_idx + 1) * (op + 1) % 4 + 1) * (100 << 20)
                    };

                    let buf = DeviceBuffer::<u8>::with_capacity(len);
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
    current_stream_sync().expect("stream sync failed");
    let large = DeviceBuffer::<u8>::with_capacity(1 << 30);
    assert!(
        !large.as_ptr().is_null(),
        "Large allocation should work after mixed operations"
    );
}

// ============================================================================
// OOM test: exhaust GPU memory and verify we get OutOfMemory error
// ============================================================================
#[test]
#[ignore] // Run explicitly: exhausts GPU memory
fn test_oom_error() {
    use super::d_malloc;
    use crate::error::MemoryError;

    let initial_free = get_gpu_free_memory();
    println!(
        "GPU memory: {:.2} GB free",
        initial_free as f64 / (1 << 30) as f64
    );

    // Fill GPU memory in 1GB chunks until we can't anymore
    let chunk_size = 1 << 30; // 1 GB
    let mut buffers: Vec<*mut std::ffi::c_void> = Vec::new();

    loop {
        match d_malloc(chunk_size) {
            Ok(ptr) => {
                buffers.push(ptr);
                println!(
                    "Allocated chunk {}: {:.2} GB total",
                    buffers.len(),
                    (buffers.len() * chunk_size) as f64 / (1 << 30) as f64
                );
            }
            Err(MemoryError::OutOfMemory { requested, .. }) => {
                println!(
                    "Got expected OutOfMemory error: requested {} bytes ({:.2} GB)",
                    requested,
                    requested as f64 / (1 << 30) as f64
                );
                // This is what we wanted - OOM error was properly returned
                // Clean up and exit successfully
                for ptr in buffers {
                    unsafe { super::d_free(ptr).unwrap() };
                }
                return; // Test passed!
            }
            Err(e) => {
                // Clean up before panicking
                for ptr in buffers {
                    unsafe { super::d_free(ptr).unwrap() };
                }
                panic!("Expected OutOfMemory error, got: {:?}", e);
            }
        }
    }
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
    let stream_id = current_stream_id().unwrap();

    for cycle in 0..3 {
        // Create fragmentation
        let ptrs: Vec<_> = (0..4)
            .map(|_| pool.malloc_internal(2 * page_size, stream_id).unwrap())
            .collect();

        // Free alternating
        pool.free_internal(ptrs[0], stream_id).unwrap();
        pool.free_internal(ptrs[2], stream_id).unwrap();

        // Request contiguous that requires defrag
        let ptr_big = pool.malloc_internal(4 * page_size, stream_id).unwrap();
        assert!(!ptr_big.is_null(), "Cycle {} failed to allocate", cycle);

        // Cleanup for next cycle
        pool.free_internal(ptrs[1], stream_id).unwrap();
        pool.free_internal(ptrs[3], stream_id).unwrap();
        pool.free_internal(ptr_big, stream_id).unwrap();
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
    let stream_id = current_stream_id().unwrap();

    // --- Case 1: Coalesce with PREV neighbor ---
    // Allocate: [A(2)][B(2)][C(2)]
    let ptr_a = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, stream_id).unwrap();

    // Free B, then A -> A merges with B (prev neighbor)
    pool.free_internal(ptr_b, stream_id).unwrap();
    pool.free_internal(ptr_a, stream_id).unwrap();

    // Verify 4 pages available from coalesced A+B
    let ptr_4 = pool.malloc_internal(4 * page_size, stream_id).unwrap();
    assert_eq!(ptr_4 as usize, ptr_a as usize, "Prev: should start at A");
    assert_eq!(pool.memory_usage(), 6 * page_size, "Prev: no new alloc");

    // Reset for next case
    pool.free_internal(ptr_c, stream_id).unwrap();
    pool.free_internal(ptr_4, stream_id).unwrap();

    // --- Case 2: Coalesce with NEXT neighbor ---
    let ptr_a = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, stream_id).unwrap();

    // Free A, then B -> B merges with A (next neighbor from A's POV)
    pool.free_internal(ptr_a, stream_id).unwrap();
    pool.free_internal(ptr_b, stream_id).unwrap();

    let ptr_4 = pool.malloc_internal(4 * page_size, stream_id).unwrap();
    assert_eq!(ptr_4 as usize, ptr_a as usize, "Next: should start at A");

    // Reset for next case
    pool.free_internal(ptr_c, stream_id).unwrap();
    pool.free_internal(ptr_4, stream_id).unwrap();

    // --- Case 3: Coalesce with BOTH neighbors ---
    let ptr_a = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, stream_id).unwrap();

    // Free A, C, then B (middle) -> B merges with both
    pool.free_internal(ptr_a, stream_id).unwrap();
    pool.free_internal(ptr_c, stream_id).unwrap();
    pool.free_internal(ptr_b, stream_id).unwrap();

    let ptr_6 = pool.malloc_internal(6 * page_size, stream_id).unwrap();
    assert_eq!(ptr_6 as usize, ptr_a as usize, "Both: should start at A");
    assert_eq!(pool.memory_usage(), 6 * page_size, "Both: no new alloc");

    pool.free_internal(ptr_6, stream_id).unwrap();
}

// ============================================================================
// Test: NO coalescing across different streams; defrag still works
// ============================================================================
#[test]
fn test_no_coalesce_across_streams() {
    let mut pool = create_test_pool(6);
    let page_size = pool.page_size;
    let stream_id_1 = current_stream_id().unwrap();
    let stream_id_2 = stream_id_1 + 1000; // Simulate different stream

    // Allocate: [A(2)][B(2)][C(2)]
    let ptr_a = pool.malloc_internal(2 * page_size, stream_id_1).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, stream_id_1).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, stream_id_1).unwrap();

    // Free A on stream_1, B on stream_2, C on stream_1
    // B should NOT coalesce with A or C (different stream)
    pool.free_internal(ptr_a, stream_id_1).unwrap();
    pool.free_internal(ptr_b, stream_id_2).unwrap();
    pool.free_internal(ptr_c, stream_id_1).unwrap();

    // Requesting 4 pages needs defrag to combine A + C (skipping B)
    let memory_before = pool.memory_usage();
    let ptr_4 = pool.malloc_internal(4 * page_size, stream_id_1).unwrap();
    assert!(!ptr_4.is_null());
    assert_eq!(pool.memory_usage(), memory_before, "Defrag reused existing");

    pool.free_internal(ptr_4, stream_id_1).unwrap();
}

// ============================================================================
// Test: Tail free regions are properly returned and usable after defrag
// Covers: oldest-first ordering, partial consumption, tail verification
// ============================================================================
#[test]
fn test_defrag_tail_regions_returned() {
    let mut pool = create_test_pool(10);
    let page_size = pool.page_size;
    let stream_id = current_stream_id().unwrap();

    // Allocate 5 x 2-page regions
    let ptrs: Vec<_> = (0..5)
        .map(|_| pool.malloc_internal(2 * page_size, stream_id).unwrap())
        .collect();

    // Free alternating to create fragmentation with 6 free pages in 3 regions
    // Layout: [2 free][2 alloc][2 free][2 alloc][2 free]
    pool.free_internal(ptrs[0], stream_id).unwrap();
    pool.free_internal(ptrs[2], stream_id).unwrap();
    pool.free_internal(ptrs[4], stream_id).unwrap();

    // Request 5 pages -> defrag takes 2+2+1, leaving 1 page as tail
    let memory_before = pool.memory_usage();
    let ptr_5 = pool.malloc_internal(5 * page_size, stream_id).unwrap();
    assert_eq!(pool.memory_usage(), memory_before, "No new pages needed");

    // Verify tail exists: can allocate 1 more without new allocation
    let ptr_1 = pool.malloc_internal(page_size, stream_id).unwrap();
    assert_eq!(pool.memory_usage(), memory_before, "Tail page reused");

    // Now requesting more requires new allocation
    let ptr_extra = pool.malloc_internal(page_size, stream_id).unwrap();
    assert_eq!(
        pool.memory_usage(),
        memory_before + page_size,
        "New page allocated"
    );

    // Cleanup
    pool.free_internal(ptrs[1], stream_id).unwrap();
    pool.free_internal(ptrs[3], stream_id).unwrap();
    pool.free_internal(ptr_5, stream_id).unwrap();
    pool.free_internal(ptr_1, stream_id).unwrap();
    pool.free_internal(ptr_extra, stream_id).unwrap();
}

// ============================================================================
// Test: Unmapped region coalescing during remap operations
// ============================================================================
#[test]
fn test_unmapped_region_coalescing_comprehensive() {
    let mut pool = create_test_pool(6);
    let page_size = pool.page_size;
    let stream_id = current_stream_id().unwrap();

    // Allocate: [A(2)][B(2)][C(2)]
    let ptr_a = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_c = pool.malloc_internal(2 * page_size, stream_id).unwrap();

    // Free A and B -> should coalesce to 4 free pages
    pool.free_internal(ptr_a, stream_id).unwrap();
    pool.free_internal(ptr_b, stream_id).unwrap();

    // Request 4 pages from coalesced region
    let ptr_4 = pool.malloc_internal(4 * page_size, stream_id).unwrap();
    assert_eq!(ptr_4 as usize, ptr_a as usize);

    // Free everything
    pool.free_internal(ptr_4, stream_id).unwrap();
    pool.free_internal(ptr_c, stream_id).unwrap();

    // Request 7 pages -> needs remap + 1 new page
    // Unmapped regions from remap should coalesce properly
    let memory_before = pool.memory_usage();
    let ptr_7 = pool.malloc_internal(7 * page_size, stream_id).unwrap();
    assert_eq!(pool.memory_usage(), memory_before + page_size);

    pool.free_internal(ptr_7, stream_id).unwrap();
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
    let stream_id = current_stream_id().unwrap();

    // Allocate 2 pages, then free them
    let ptr_2 = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    assert_eq!(pool.memory_usage(), 2 * page_size);
    pool.free_internal(ptr_2, stream_id).unwrap();

    // Request 4 pages -> allocates 2 new + uses 2 free (merge)
    let ptr_4 = pool.malloc_internal(4 * page_size, stream_id).unwrap();
    assert_eq!(pool.memory_usage(), 4 * page_size);
    assert!(!ptr_4.is_null());

    // Test with preallocated pool: alloc merges with prev free
    pool.free_internal(ptr_4, stream_id).unwrap();

    let ptr_a = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    let ptr_b = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    pool.free_internal(ptr_a, stream_id).unwrap();

    // Request 4 pages with 2 free -> need 2 more
    let ptr_req = pool.malloc_internal(4 * page_size, stream_id).unwrap();
    assert!(!ptr_req.is_null());
    assert_eq!(pool.memory_usage(), 6 * page_size);

    pool.free_internal(ptr_b, stream_id).unwrap();
    pool.free_internal(ptr_req, stream_id).unwrap();
}

// ============================================================================
// Test: Multiple defrag scenarios in sequence
// Covers: exact fit (no tail), remap with new page allocation, combining regions
// ============================================================================
#[test]
fn test_defrag_various_scenarios() {
    let mut pool = create_test_pool(10);
    let page_size = pool.page_size;
    let stream_id = current_stream_id().unwrap();

    // --- Scenario 1: Exact fit, no tail ---
    let ptrs: Vec<_> = (0..5)
        .map(|_| pool.malloc_internal(2 * page_size, stream_id).unwrap())
        .collect();

    // Free all in scattered order -> should coalesce completely
    pool.free_internal(ptrs[1], stream_id).unwrap();
    pool.free_internal(ptrs[3], stream_id).unwrap();
    pool.free_internal(ptrs[0], stream_id).unwrap();
    pool.free_internal(ptrs[4], stream_id).unwrap();
    pool.free_internal(ptrs[2], stream_id).unwrap();

    // Request all 10 pages (exact fit)
    let ptr_10 = pool.malloc_internal(10 * page_size, stream_id).unwrap();
    assert!(!ptr_10.is_null());
    assert_eq!(pool.memory_usage(), 10 * page_size);

    // --- Scenario 2: Remap with new page allocation (reuse same pool state) ---
    // Layout now: [10 malloc]
    // Free the 10-page region, then create fragmentation
    pool.free_internal(ptr_10, stream_id).unwrap();

    // Allocate 6 + 4 pages
    let ptr_a = pool.malloc_internal(6 * page_size, stream_id).unwrap();
    let ptr_b = pool.malloc_internal(4 * page_size, stream_id).unwrap();

    // Free A, keep B -> [6 free][4 malloc]
    pool.free_internal(ptr_a, stream_id).unwrap();

    // Request 7 pages: 6 free + need 1 more
    let memory_before = pool.memory_usage();
    let ptr_7 = pool.malloc_internal(7 * page_size, stream_id).unwrap();
    assert_eq!(pool.memory_usage(), memory_before + page_size);

    pool.free_internal(ptr_b, stream_id).unwrap();
    pool.free_internal(ptr_7, stream_id).unwrap();
}
