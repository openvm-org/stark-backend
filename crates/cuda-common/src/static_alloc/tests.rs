//! GPU tests for the static allocator: end-to-end acquire/copy/release,
//! memory reuse, pointer stability, and lifetime-violation detection.

use super::*;
use crate::{copy::MemCopyD2H, stream::GpuDeviceCtx};

fn test_ctx() -> GpuDeviceCtx {
    GpuDeviceCtx::for_current_device().unwrap()
}

/// Two-phase plan where phase temps share space:
///   phase 1: input (temp) -> output (long-lived)
///   phase 2: scratch (temp, expected to reuse input's space)
struct TwoPhasePlan {
    input: AllocIdx<u32>,
    output: AllocIdx<u32>,
    scratch: AllocIdx<u32>,
    alloc: StaticAllocator,
    ctx: GpuDeviceCtx,
}

fn two_phase_plan(len: usize, policy: ViolationPolicy) -> TwoPhasePlan {
    let builder = AllocBuilder::new();
    let input_fake = builder.alloc_fake_labeled::<u32>(len, "input");
    let output_fake = builder.alloc_fake_labeled::<u32>(len, "output");
    let (input, output) = (input_fake.idx(), output_fake.idx());
    drop(input_fake); // phase 1 ends: input's space becomes reusable
    let scratch = builder.alloc_fake_labeled::<u32>(len, "scratch").idx();
    // `output_fake` is still alive at build: output lives until the end.
    let ctx = test_ctx();
    let alloc = builder.build_with_policy_on(&ctx, policy).unwrap();
    TwoPhasePlan {
        input,
        output,
        scratch,
        alloc,
        ctx,
    }
}

// ============================================================================
// Roundtrip: planned buffers work with the MemCopy traits and kernels
// ============================================================================
#[test]
fn test_roundtrip_through_planned_buffers() {
    let len = 1 << 12;
    let plan = two_phase_plan(len, ViolationPolicy::Error);

    let host: Vec<u32> = (0..len as u32).collect();
    let mut input = plan.alloc.get(&plan.input).unwrap();
    input.copy_from(&host, &plan.ctx).unwrap();
    assert_eq!(input.to_host_on(&plan.ctx).unwrap(), host);

    let mut output = plan.alloc.get(&plan.output).unwrap();
    output.copy_from(&host, &plan.ctx).unwrap();
    drop(input);

    // Phase 2: scratch may reuse input's space; output data must survive.
    let scratch = plan.alloc.get(&plan.scratch).unwrap();
    scratch.fill_zero_on(&plan.ctx).unwrap();
    assert_eq!(scratch.to_host_on(&plan.ctx).unwrap(), vec![0u32; len]);
    assert_eq!(output.to_host_on(&plan.ctx).unwrap(), host);
}

// ============================================================================
// Packing: disjoint lifetimes share memory, workspace stays at peak-live
// ============================================================================
#[test]
fn test_scratch_reuses_input_space() {
    let len = 1 << 20;
    let plan = two_phase_plan(len, ViolationPolicy::Error);

    // input dies before scratch is born -> same planned offset; the
    // workspace holds 2 buffers' worth, not 3.
    assert_eq!(plan.alloc.workspace_size(), 2 * 4 * len);
    assert_eq!(plan.alloc.workspace_size(), plan.alloc.peak_live_size());

    let input_ptr = {
        let input = plan.alloc.get(&plan.input).unwrap();
        input.as_ptr() as usize
    };
    let scratch = plan.alloc.get(&plan.scratch).unwrap();
    assert_eq!(
        scratch.as_ptr() as usize,
        input_ptr,
        "scratch should reuse input's planned space"
    );
}

// ============================================================================
// Pointer stability across acquire/release rounds (cudaGraphs prerequisite)
// ============================================================================
#[test]
fn test_stable_pointers_across_rounds() {
    let len = 1 << 10;
    let plan = two_phase_plan(len, ViolationPolicy::Error);

    let mut ptrs = Vec::new();
    for _round in 0..3 {
        let input = plan.alloc.get(&plan.input).unwrap();
        let output = plan.alloc.get(&plan.output).unwrap();
        ptrs.push((input.as_ptr() as usize, output.as_ptr() as usize));
    }
    assert_eq!(ptrs[0], ptrs[1]);
    assert_eq!(ptrs[1], ptrs[2]);
}

// ============================================================================
// Lifetime violations: error policy
// ============================================================================
#[test]
fn test_use_after_lifetime_errors_then_recovers() {
    let len = 1 << 10;
    let plan = two_phase_plan(len, ViolationPolicy::Error);

    // Hold `input` beyond its declared lifetime.
    let input = plan.alloc.get(&plan.input).unwrap();
    let err = plan.alloc.get(&plan.scratch).unwrap_err();
    match err {
        StaticAllocError::LifetimeViolation { requested, holders } => {
            assert_eq!(requested, "scratch");
            assert_eq!(holders, vec!["input".to_string()]);
        }
        other => panic!("expected LifetimeViolation, got {other:?}"),
    }

    // Dropping the offender makes the space safe to use.
    drop(input);
    plan.alloc.get(&plan.scratch).unwrap();
}

#[test]
fn test_double_acquire_errors() {
    let len = 1 << 10;
    let plan = two_phase_plan(len, ViolationPolicy::Error);

    let _output = plan.alloc.get(&plan.output).unwrap();
    assert!(matches!(
        plan.alloc.get(&plan.output),
        Err(StaticAllocError::AlreadyAcquired { .. })
    ));
}

// ============================================================================
// Lifetime violations: warn-and-allocate fallback policy
// ============================================================================
#[test]
fn test_warn_and_allocate_fallback() {
    let len = 1 << 10;
    let plan = two_phase_plan(len, ViolationPolicy::WarnAndAllocate);

    let input = plan.alloc.get(&plan.input).unwrap();
    let planned_ptr = input.as_ptr() as usize;

    // Conflict -> fresh dynamic allocation with a different pointer.
    let scratch = plan.alloc.get(&plan.scratch).unwrap();
    assert_ne!(scratch.as_ptr() as usize, planned_ptr);

    // Fallback buffers still work end-to-end.
    let host: Vec<u32> = (0..len as u32).collect();
    let mut scratch = scratch;
    scratch.copy_from(&host, &plan.ctx).unwrap();
    assert_eq!(scratch.to_host_on(&plan.ctx).unwrap(), host);

    // Once the offender and the fallback are gone, the planned space serves
    // the same idx at its planned pointer again.
    drop(input);
    drop(scratch);
    let scratch = plan.alloc.get(&plan.scratch).unwrap();
    assert_eq!(scratch.as_ptr() as usize, planned_ptr);
}

// ============================================================================
// Plan identity: indices are not interchangeable between allocators
// ============================================================================
#[test]
fn test_plan_mismatch_detected() {
    let plan_a = two_phase_plan(1 << 10, ViolationPolicy::Error);
    let plan_b = two_phase_plan(1 << 10, ViolationPolicy::Error);

    assert!(matches!(
        plan_b.alloc.get(&plan_a.input),
        Err(StaticAllocError::PlanMismatch { .. })
    ));
}

// ============================================================================
// Interfering buffers acquired together hold disjoint memory
// ============================================================================
#[test]
fn test_concurrent_buffers_do_not_alias() {
    let len = 1 << 12;
    let builder = AllocBuilder::new();
    let a = builder.alloc_fake_labeled::<u64>(len, "a");
    let b = builder.alloc_fake_labeled::<u64>(len, "b");
    let (a, b) = (a.idx(), b.idx());
    let ctx = test_ctx();
    let alloc = builder.build_on(&ctx).unwrap();

    let mut buf_a = alloc.get(&a).unwrap();
    let mut buf_b = alloc.get(&b).unwrap();
    let host_a: Vec<u64> = (0..len as u64).collect();
    let host_b: Vec<u64> = (0..len as u64).map(|x| x.wrapping_mul(31)).collect();
    buf_a.copy_from(&host_a, &ctx).unwrap();
    buf_b.copy_from(&host_b, &ctx).unwrap();
    assert_eq!(buf_a.to_host_on(&ctx).unwrap(), host_a);
    assert_eq!(buf_b.to_host_on(&ctx).unwrap(), host_b);
}

// ============================================================================
// Data written to a planned buffer survives release + reacquire as long as
// no overlapping allocation ran in between
// ============================================================================
// Under `touchemall` every planned acquisition is poisoned with 0xff, so
// cross-acquisition persistence intentionally does not hold there.
#[cfg(not(feature = "touchemall"))]
#[test]
fn test_data_survives_release_without_reuse() {
    let len = 1 << 10;
    let plan = two_phase_plan(len, ViolationPolicy::Error);

    let host: Vec<u32> = (0..len as u32).rev().collect();
    {
        let mut output = plan.alloc.get(&plan.output).unwrap();
        output.copy_from(&host, &plan.ctx).unwrap();
    }
    let output = plan.alloc.get(&plan.output).unwrap();
    assert_eq!(output.to_host_on(&plan.ctx).unwrap(), host);
}

// ============================================================================
// Cross-stream declarations never share memory
// ============================================================================
#[test]
fn test_cross_stream_no_reuse() {
    let len = 1 << 16;
    let builder = AllocBuilder::new();
    let a = builder
        .alloc_fake_on_stream::<u32>(len, "stream1-temp", 1)
        .idx();
    // `a`'s fake is already dropped; a same-stream alloc would reuse its
    // space, but a different-stream alloc must not.
    let b = builder
        .alloc_fake_on_stream::<u32>(len, "stream2-temp", 2)
        .idx();
    let alloc = builder.build_on(&test_ctx()).unwrap();

    let buf_a = alloc.get(&a).unwrap();
    let buf_b = alloc.get(&b).unwrap();
    assert_ne!(buf_a.as_ptr() as usize, buf_b.as_ptr() as usize);
    assert_eq!(alloc.workspace_size(), 2 * 4 * len);
}

// ============================================================================
// Sanity: describe() renders and empty plans build
// ============================================================================
#[test]
fn test_describe_and_empty_plan() {
    let builder = AllocBuilder::new();
    let alloc = builder.build_on(&test_ctx()).unwrap();
    assert_eq!(alloc.workspace_size(), 0);
    assert_eq!(alloc.num_allocs(), 0);

    let plan = two_phase_plan(1 << 10, ViolationPolicy::Error);
    let desc = plan.alloc.describe();
    assert!(desc.contains("input"));
    assert!(desc.contains("output"));
    assert!(desc.contains("scratch"));
}
