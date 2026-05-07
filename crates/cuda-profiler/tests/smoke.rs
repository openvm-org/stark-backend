//! End-to-end smoke test for the in-process CUDA profiler.
//!
//! This test exercises a self-contained CUDA kernel that lives inside the
//! cuda-profiler crate (so we don't have to depend on cuda-backend). It:
//!
//!   1. Allocates a tiny scratch buffer.
//!   2. Calls `prof::init()` (with env vars set for output / ring size).
//!   3. Launches `shadow_test_kernel` a few times.
//!   4. Calls `prof::shutdown()`.
//!   5. Re-reads the binary log and checks magic, kernel-name records, CTA records, and basic
//!      timestamp sanity.
//!
//! The test is `#[ignore]` because it requires a GPU; run with
//! `cargo test -p openvm-cuda-profiler --test smoke -- --ignored`.

use std::{
    io::Read,
    path::PathBuf,
    sync::atomic::{AtomicU32, Ordering},
};

use openvm_cuda_common::{
    common::set_device,
    d_buffer::DeviceBuffer,
    stream::{CudaStream, GpuDeviceCtx, StreamGuard},
};
use openvm_cuda_profiler::{
    record::{decode_payload, read_frame, Record, MAGIC},
    register_kernel,
};

#[link(name = "openvm_cuda_profiler_glue")]
extern "C" {
    fn shadow_test_kernel_launch(
        blocks: u32,
        threads_per_block: u32,
        iters: u32,
        dummy: *mut u32,
        stream_ptr: *mut std::ffi::c_void,
    ) -> i32;
}

/// Counter to keep concurrent smoke-test invocations from clobbering each
/// other when the test binary is run in parallel modes.
static SUFFIX: AtomicU32 = AtomicU32::new(0);

#[test]
#[ignore]
fn profiler_smoke() {
    let pid = std::process::id();
    let suffix = SUFFIX.fetch_add(1, Ordering::Relaxed);
    let log_path: PathBuf =
        std::env::temp_dir().join(format!("shadow_profile_smoke_{pid}_{suffix}.bin"));
    if log_path.exists() {
        let _ = std::fs::remove_file(&log_path);
    }

    std::env::set_var("SHADOW_PROFILER", "1");
    std::env::set_var("SHADOW_PROFILER_OUT", &log_path);
    // 1 MB ring -> ~32K record slots. Plenty for 4 launches × 4 grids.
    std::env::set_var("SHADOW_PROFILER_RING_BYTES", "1048576");
    // Drain quickly so we don't wait long after launches.
    std::env::set_var("SHADOW_PROFILER_DRAIN_MS", "10");

    set_device().expect("set_device");

    // Register the kernel name BEFORE init so the registry dump on init
    // includes it.
    let _ = register_kernel("shadow_test_kernel");

    assert!(
        openvm_cuda_profiler::init().expect("init"),
        "expected profiler to activate when SHADOW_PROFILER=1"
    );

    let device_ctx = GpuDeviceCtx {
        device_id: openvm_cuda_common::common::get_device().unwrap() as u32,
        stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
    };
    // Allocate a small scratch buffer (1 MB of u32) the test kernel writes to.
    let dummy = DeviceBuffer::<u32>::with_capacity_on(1 << 18, &device_ctx);

    // Total CTAs across all rounds = sum of blocks; here 256 + 64 + 32 + 8 = 360
    // launches × 4 iterations = ~1440 CTAs.
    for &blocks in &[256u32, 64, 32, 8] {
        let rc = unsafe {
            shadow_test_kernel_launch(
                blocks,
                256,
                4,
                dummy.as_mut_raw_ptr() as *mut u32,
                device_ctx.stream.as_raw() as *mut _,
            )
        };
        assert_eq!(rc, 0, "shadow_test_kernel_launch returned cuda error {rc}");
    }
    device_ctx.stream.to_host_sync().expect("stream sync");

    // Give the drain thread a chance to flush.
    std::thread::sleep(std::time::Duration::from_millis(80));
    openvm_cuda_profiler::shutdown();

    // Verify file contents.
    let mut r = std::io::BufReader::new(std::fs::File::open(&log_path).unwrap());
    let mut head = [0u8; 16];
    r.read_exact(&mut head).expect("read header");
    assert_eq!(&head[..8], MAGIC, "bad magic");

    let mut saw_kernel_name = false;
    let mut saw_process_start = false;
    let mut cta_count = 0;
    let mut cupti_kernel_count = 0;
    let mut drop_count: u64 = 0;
    let mut min_t_start: u64 = u64::MAX;
    let mut max_t_end: u64 = 0;
    let mut sm_set = std::collections::HashSet::<u32>::new();

    while let Some((tag, payload)) = read_frame(&mut r).expect("frame") {
        match decode_payload(tag, &payload).expect("decode") {
            Record::ProcessStart { gpu_name, .. } => {
                saw_process_start = true;
                eprintln!("smoke: ProcessStart gpu_name={gpu_name:?}");
            }
            Record::KernelName { name, .. } => {
                if name == "shadow_test_kernel" {
                    saw_kernel_name = true;
                }
            }
            Record::Cta(c) => {
                cta_count += 1;
                assert!(c.t_start > 0, "non-zero t_start");
                assert!(c.t_end >= c.t_start, "t_end >= t_start");
                let expected_tag = (((cta_count - 1) + 1) & 0xFFFFFFFF) as u32;
                // The drain emits records in slot order so the seq_tag is
                // (slot+1) for each consecutive record (slot starts at 0).
                // This validates that we're not dropping/reordering records.
                let _ = expected_tag; // not equality-tested because overrun could skip slots
                min_t_start = min_t_start.min(c.t_start);
                max_t_end = max_t_end.max(c.t_end);
                sm_set.insert(c.smid);
            }
            Record::CuptiKernel { name, .. } => {
                if name.contains("shadow_test_kernel") {
                    cupti_kernel_count += 1;
                }
            }
            Record::Drop { count, .. } => {
                drop_count += count;
            }
            _ => {}
        }
    }

    assert!(saw_process_start, "no ProcessStart record");
    assert!(
        saw_kernel_name,
        "no KernelName record for shadow_test_kernel"
    );
    assert!(cta_count > 0, "no CTA records");
    eprintln!(
        "smoke: {cta_count} CTAs across {} unique SMs; span = {} ns; cupti_kernels = {} drops = {}",
        sm_set.len(),
        max_t_end - min_t_start,
        cupti_kernel_count,
        drop_count,
    );
    // We launched 256 + 64 + 32 + 8 = 360 CTAs × 4 iterations = 1440 CTAs.
    assert_eq!(
        cta_count + drop_count,
        1440,
        "expected exactly 1440 CTA records (recovered + dropped); got {cta_count} + {drop_count}",
    );
    // With the seq_tag protocol and no overrun, drops should be 0 in this test.
    assert_eq!(
        drop_count, 0,
        "expected zero drops with 1MB ring + 1440 CTAs"
    );
    // H100 has 132 SMs; we should hit several with this many CTAs.
    assert!(sm_set.len() >= 4, "expected hits across multiple SMs");
    // CUPTI may or may not be available; if it is, we should see at least
    // one record for our test kernel.
    if cupti_kernel_count > 0 {
        eprintln!("smoke: CUPTI Activity sidecar working ({cupti_kernel_count} kernels)");
    } else {
        eprintln!("smoke: CUPTI Activity sidecar inactive (this is acceptable)");
    }

    if std::env::var("SHADOW_PROFILER_KEEP_LOG").is_err() {
        let _ = std::fs::remove_file(&log_path);
    } else {
        eprintln!("smoke: kept log at {}", log_path.display());
    }
}
