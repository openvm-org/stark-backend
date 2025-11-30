//! Tests for memory_manager - focused on edge cases and dangerous scenarios

use std::sync::Arc;

use tokio::sync::Barrier;

use super::vm_pool::{VirtualMemoryPool, VpmmConfig};
use crate::{d_buffer::DeviceBuffer, stream::current_stream_id};

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
// Cross-thread: Thread A frees, Thread B should be able to reuse after sync
// ============================================================================
#[test]
fn test_cross_thread_handoff() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(2)
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let barrier = Arc::new(Barrier::new(2));
        // Store address as usize (which is Send + Sync)
        let addr_holder = Arc::new(std::sync::Mutex::new(0usize));

        let barrier1 = barrier.clone();
        let addr_holder1 = addr_holder.clone();

        // Thread A: allocate and free
        let handle_a = tokio::task::spawn(async move {
            tokio::task::spawn_blocking(move || {
                let len = (200 << 20) / size_of::<u8>(); // 200 MB
                let buf = DeviceBuffer::<u8>::with_capacity(len);
                *addr_holder1.lock().unwrap() = buf.as_raw_ptr() as usize;
                // buf drops here, freeing memory
            })
            .await
            .expect("task A panicked");

            barrier1.wait().await;
        });

        let barrier2 = barrier.clone();
        let addr_holder2 = addr_holder.clone();

        // Thread B: wait for A, then allocate same size
        let handle_b = tokio::task::spawn(async move {
            barrier2.wait().await;

            tokio::task::spawn_blocking(move || {
                let len = (200 << 20) / size_of::<u8>();
                let buf = DeviceBuffer::<u8>::with_capacity(len);

                // Check if we got the same address (depends on event sync timing)
                let original_addr = *addr_holder2.lock().unwrap();
                let new_addr = buf.as_raw_ptr() as usize;
                if new_addr == original_addr {
                    println!("Cross-thread: reused same address (event synced)");
                } else {
                    println!("Cross-thread: different address (event pending)");
                }
                // buf drops here
            })
            .await
            .expect("task B panicked");
        });

        handle_a.await.expect("thread A failed");
        handle_b.await.expect("thread B failed");
    });
}

// ============================================================================
// Stress: many threads doing random alloc/free patterns
// ============================================================================
#[test]
fn test_stress_multithread() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .max_blocking_threads(8)
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let mut handles = Vec::new();

        for thread_idx in 0..8 {
            let handle = tokio::task::spawn_blocking(move || {
                let mut buffers: Vec<DeviceBuffer<u8>> = Vec::new();

                for op in 0..20 {
                    // Random-ish size: 100-500 MB (VPMM path)
                    let len = ((thread_idx * 7 + op * 3) % 5 + 1) * (100 << 20);
                    let buf = DeviceBuffer::<u8>::with_capacity(len);
                    buffers.push(buf);

                    // Free every 3rd allocation
                    if op % 3 == 0 && !buffers.is_empty() {
                        buffers.remove(0); // drops and frees
                    }
                }
                // remaining buffers drop here
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("stress thread failed");
        }
    });

    println!("Stress test: 8 threads × 20 ops completed");
}

// ============================================================================
// Mixed: interleave small (cudaMallocAsync) and large (VPMM) allocations
// ============================================================================
#[test]
fn test_mixed_small_large_multithread() {
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
                        // Small: 1KB - 100KB (cudaMallocAsync)
                        ((thread_idx + 1) * (op + 1) * 1024) % (100 << 10) + 1024
                    } else {
                        // Large: 100MB - 400MB (VPMM)
                        ((thread_idx + 1) * (op + 1) % 4 + 1) * (100 << 20)
                    };

                    let buf = DeviceBuffer::<u8>::with_capacity(len);
                    buffers.push(buf);

                    if op % 2 == 0 && !buffers.is_empty() {
                        buffers.remove(0); // drops and frees
                    }
                }
                // remaining buffers drop here
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.expect("thread failed");
        }
    });
}

// ============================================================================
// Preallocation: test pool with initial pages already mapped
// ============================================================================
#[test]
fn test_preallocation() {
    let config = VpmmConfig {
        page_size: None,  // Use device granularity
        va_size: 1 << 30, // 1 GB VA
        initial_pages: 4, // Preallocate 4 pages
    };
    let mut pool = VirtualMemoryPool::new(config);

    if pool.page_size == usize::MAX {
        println!("VPMM not supported, skipping test");
        return;
    }

    let page_size = pool.page_size;
    let stream_id = current_stream_id().unwrap();

    // Should already have 4 pages allocated
    assert_eq!(
        pool.memory_usage(),
        4 * page_size,
        "Should have 4 preallocated pages"
    );

    // Allocate 2 pages from preallocated pool - no new pages needed
    let ptr = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    assert_eq!(
        pool.memory_usage(),
        4 * page_size,
        "No new pages should be allocated"
    );

    // Allocate 3 more pages - more than remaining 2 → triggers growth
    let ptr2 = pool.malloc_internal(3 * page_size, stream_id).unwrap();
    assert!(
        pool.memory_usage() > 4 * page_size,
        "Pool should have grown past initial 4 pages"
    );

    // Cleanup
    pool.free_internal(ptr, stream_id).unwrap();
    pool.free_internal(ptr2, stream_id).unwrap();
}

// ============================================================================
// Reuse after free: same stream should immediately reuse freed region
// ============================================================================
#[test]
fn test_same_stream_immediate_reuse() {
    let config = VpmmConfig {
        page_size: None,  // Use device granularity
        va_size: 1 << 30, // 1 GB VA
        initial_pages: 0,
    };
    let mut pool = VirtualMemoryPool::new(config);

    if pool.page_size == usize::MAX {
        println!("VPMM not supported, skipping test");
        return;
    }

    let page_size = pool.page_size;
    let stream_id = current_stream_id().unwrap();

    // Allocate 2 pages and free
    let ptr1 = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    pool.free_internal(ptr1, stream_id).unwrap();

    // Same stream should immediately get the same address back
    let ptr2 = pool.malloc_internal(2 * page_size, stream_id).unwrap();
    assert_eq!(
        ptr1, ptr2,
        "Same stream should immediately reuse freed region"
    );

    pool.free_internal(ptr2, stream_id).unwrap();
}
