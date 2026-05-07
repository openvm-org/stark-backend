//! CUPTI Activity API streaming.
//!
//! The Activity API is the preferred way to get per-kernel/memcpy/NVTX timing
//! without replaying kernels (in contrast to the Range Profiler / Events API).
//! It's batch-buffer based:
//!
//! 1. We register two callbacks: `buffer_requested` (CUPTI asks us for a fresh buffer) and
//!    `buffer_completed` (CUPTI hands us back a filled buffer).
//! 2. We enable the activity kinds we want (KERNEL, MEMCPY, MARKER).
//! 3. Records flow through `buffer_completed`. We parse them, write to the binary log, and free the
//!    buffer.
//!
//! There is exactly one global instance of this backend per process — the
//! callbacks are static `extern "C"` and route to it via a `OnceCell`.

use std::{
    ffi::{c_void, CStr},
    sync::Arc,
};

use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use super::{
    sys::{
        check, CUpti_Activity, CUpti_ActivityKernel, CUpti_ActivityKind, CUpti_ActivityMarker,
        CUpti_ActivityMemcpy, Cupti, CUPTI_SUCCESS,
    },
    CuptiError,
};
use crate::{log_writer::LogWriter, record::Record};

/// CUPTI hands us 16 KB-aligned buffers; we use 256 KB so a busy proof phase
/// can fit many records between drains.
const BUFFER_SIZE: usize = 256 * 1024;
const BUFFER_ALIGN: usize = 16;

/// Process-singleton holding the writer + the loaded library so the static
/// callbacks can find them. There is no clean way around this: CUPTI's
/// callback-registration API is C-style and takes raw function pointers.
static GLOBAL: OnceCell<GlobalState> = OnceCell::new();

struct GlobalState {
    cupti: Arc<Cupti>,
    writer: Arc<Mutex<LogWriter>>,
    /// Total bytes of records dropped (CUPTI couldn't allocate, host couldn't
    /// keep up, etc). Reported in shutdown traces.
    dropped: parking_lot::Mutex<u64>,
}

/// Owner of the activity subscription. Calling `stop()` flushes outstanding
/// buffers and disables the activities; the global state lives forever
/// (CUPTI doesn't allow re-registering callbacks safely after they've fired,
/// so we make this a strict one-shot per process).
pub struct ActivityBackend {
    cupti: Arc<Cupti>,
    enabled_kinds: Vec<CUpti_ActivityKind>,
}

impl ActivityBackend {
    pub fn start(cupti: Arc<Cupti>, writer: Arc<Mutex<LogWriter>>) -> Result<Self, CuptiError> {
        // Refuse early if `GLOBAL` is already set: CUPTI's callback API does
        // not let us cleanly tear down and re-register, so we are strictly
        // one-shot per process. This is documented at the lib.rs `init`
        // doc-comment.
        if GLOBAL.get().is_some() {
            return Err(CuptiError::Call {
                func: "ActivityBackend::start",
                code: -1,
            });
        }

        // Register callbacks first; the static buffer_completed routes through
        // GLOBAL, but until we publish GLOBAL it will short-circuit because
        // OnceCell::get returns None. We rely on CUPTI not delivering buffers
        // before any kind is enabled.
        unsafe {
            check(
                "cuptiActivityRegisterCallbacks",
                (cupti.activity_register_callbacks)(buffer_requested, buffer_completed),
            )?;
        }

        // Subscribe. The plan asks for kernels, memcpys, NVTX. We add OVERHEAD
        // so we can warn about CUPTI's own self-overhead in the trace.
        //
        // Track which kinds we successfully enabled; on failure we roll back
        // by disabling those kinds before bubbling the error. This avoids
        // leaving CUPTI in a half-enabled state that produces records with
        // no consumer.
        let kinds = [
            CUpti_ActivityKind::CONCURRENT_KERNEL,
            CUpti_ActivityKind::MEMCPY,
            CUpti_ActivityKind::MARKER,
            CUpti_ActivityKind::OVERHEAD,
        ];
        let mut enabled: Vec<CUpti_ActivityKind> = Vec::with_capacity(kinds.len());
        for k in &kinds {
            let rc = unsafe { (cupti.activity_enable)(*k as u32) };
            if rc != CUPTI_SUCCESS {
                for done in &enabled {
                    unsafe {
                        let _ = (cupti.activity_disable)(*done as u32);
                    }
                }
                return Err(CuptiError::Call {
                    func: "cuptiActivityEnable",
                    code: rc,
                });
            }
            enabled.push(*k);
        }

        // Only now publish GLOBAL — at this point all subscriptions are live
        // and any incoming buffer will land at our callback.
        let _ = GLOBAL.set(GlobalState {
            cupti: cupti.clone(),
            writer,
            dropped: parking_lot::Mutex::new(0),
        });

        Ok(Self {
            cupti,
            enabled_kinds: enabled,
        })
    }

    pub fn stop(self) {
        // Flush all outstanding records, then disable each kind. We set
        // `flag=1` (CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) so partially-filled
        // buffers also get returned through buffer_completed.
        unsafe {
            let _ = (self.cupti.activity_flush_all)(1);
            for k in &self.enabled_kinds {
                let _ = (self.cupti.activity_disable)(*k as u32);
            }
        }
        if let Some(g) = GLOBAL.get() {
            let dropped = *g.dropped.lock();
            if dropped > 0 {
                tracing::warn!(
                    dropped,
                    "cuda-profiler: CUPTI dropped {dropped} record bytes"
                );
            }
        }
    }
}

// ---- Static callbacks ---------------------------------------------------

unsafe extern "C" fn buffer_requested(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_records: *mut usize,
) {
    // Allocate a fresh aligned buffer. CUPTI takes ownership and returns it
    // to us via `buffer_completed`.
    let layout = match std::alloc::Layout::from_size_align(BUFFER_SIZE, BUFFER_ALIGN) {
        Ok(l) => l,
        Err(_) => {
            *buffer = std::ptr::null_mut();
            *size = 0;
            *max_records = 0;
            return;
        }
    };
    let p = std::alloc::alloc(layout);
    if p.is_null() {
        *buffer = std::ptr::null_mut();
        *size = 0;
        *max_records = 0;
        return;
    }
    *buffer = p;
    *size = BUFFER_SIZE;
    *max_records = 0; // 0 means "fit as many as possible".
}

unsafe extern "C" fn buffer_completed(
    ctx: *mut c_void,
    stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
) {
    let _ = ctx;
    let _ = stream_id;

    let Some(g) = GLOBAL.get() else {
        // Late callback after stop; just free.
        free_buffer(buffer);
        return;
    };

    if buffer.is_null() {
        return;
    }

    if valid_size > 0 {
        // Walk the records.
        let mut record_ptr: *mut CUpti_Activity = std::ptr::null_mut();
        loop {
            let code = (g.cupti.activity_get_next_record)(buffer, valid_size, &mut record_ptr);
            if code != CUPTI_SUCCESS {
                break;
            }
            if record_ptr.is_null() {
                break;
            }
            handle_record(record_ptr, &g.writer);
        }

        // Tally dropped (CUPTI's internal drops, not ours).
        let mut dropped: usize = 0;
        let _ = (g.cupti.activity_get_num_dropped_records)(std::ptr::null_mut(), 0, &mut dropped);
        if dropped > 0 {
            *g.dropped.lock() += dropped as u64;
        }
    }

    let _ = size;
    free_buffer(buffer);
}

fn free_buffer(buffer: *mut u8) {
    if buffer.is_null() {
        return;
    }
    let layout = match std::alloc::Layout::from_size_align(BUFFER_SIZE, BUFFER_ALIGN) {
        Ok(l) => l,
        Err(_) => return,
    };
    unsafe { std::alloc::dealloc(buffer, layout) };
}

unsafe fn handle_record(record: *mut CUpti_Activity, writer: &Arc<Mutex<LogWriter>>) {
    let kind = (*record).kind;
    if kind == CUpti_ActivityKind::CONCURRENT_KERNEL as u32
        || kind == CUpti_ActivityKind::KERNEL as u32
    {
        let k = &*(record as *const CUpti_ActivityKernel);
        let name = if k.name.is_null() {
            String::new()
        } else {
            CStr::from_ptr(k.name).to_string_lossy().into_owned()
        };
        let rec = Record::CuptiKernel {
            correlation_id: k.correlation_id,
            stream_id: k.stream_id,
            device_id: k.device_id,
            t_start_ns: k.start,
            t_end_ns: k.end,
            name,
        };
        let _ = writer.lock().write_record(&rec);
    } else if kind == CUpti_ActivityKind::MEMCPY as u32
        || kind == CUpti_ActivityKind::MEMCPY2 as u32
    {
        let m = &*(record as *const CUpti_ActivityMemcpy);
        let rec = Record::CuptiMemcpy {
            copy_kind: m.copy_kind as u32,
            src_kind: m.src_kind as u32,
            dst_kind: m.dst_kind as u32,
            bytes: m.bytes,
            t_start_ns: m.start,
            t_end_ns: m.end,
            stream_id: m.stream_id,
        };
        let _ = writer.lock().write_record(&rec);
    } else if kind == CUpti_ActivityKind::MARKER as u32 {
        let m = &*(record as *const CUpti_ActivityMarker);
        // CUPTI emits one record per marker event. `flags` carries
        // CUPTI_ACTIVITY_FLAG_MARKER_*: 1=instantaneous, 2=start, 4=end.
        // END records have a NULL name (cupti_activity.h:4322); the
        // assembler reuses the START's name on pair-up.
        let name = if m.name.is_null() {
            String::new()
        } else {
            CStr::from_ptr(m.name).to_string_lossy().into_owned()
        };
        let domain = if m.domain.is_null() {
            String::new()
        } else {
            CStr::from_ptr(m.domain).to_string_lossy().into_owned()
        };
        // Mask out the marker-kind bits we care about so future flags
        // (sync acquire, etc.) don't collide.
        const MARKER_KIND_MASK: u32 = 0b111; // INSTANTANEOUS | START | END
        let rec = Record::CuptiNvtxRange {
            domain,
            name,
            timestamp_ns: m.timestamp,
            id: m.id,
            marker_kind: m.flags & MARKER_KIND_MASK,
        };
        let _ = writer.lock().write_record(&rec);
    } else if kind == CUpti_ActivityKind::OVERHEAD as u32 {
        // Layout of `CUpti_ActivityOverhead3` (cupti_activity.h:4403):
        //   off 0  : u32 kind
        //   off 4  : u32 overheadKind
        //   off 8  : u32 objectKind
        //   off 12 : CUpti_ActivityObjectKindId objectId — a *12-byte* union
        //            (NOT 16 — see cupti_activity.h:711-733: the largest variant
        //             is the dcs struct of three u32s).
        //   off 24 : u64 start
        //   off 32 : u64 end
        //   ...    : correlationId, reserved0, overheadData
        // The struct is `__attribute__((packed)) __attribute__((aligned(8)))`,
        // so all fields lie at their packed offsets.
        let raw = record as *const u8;
        let oh_kind = std::ptr::read_unaligned(raw.add(4) as *const u32);
        let start = std::ptr::read_unaligned(raw.add(24) as *const u64);
        let end = std::ptr::read_unaligned(raw.add(32) as *const u64);
        let rec = Record::CuptiOverhead {
            kind: oh_kind,
            t_start_ns: start,
            t_end_ns: end,
        };
        let _ = writer.lock().write_record(&rec);
    }
    // Unknown kinds are silently ignored — we already filtered subscriptions.
}
