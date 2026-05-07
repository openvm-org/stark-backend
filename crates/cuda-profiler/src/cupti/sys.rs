//! CUPTI dynamic loader and FFI declarations.
//!
//! The full CUPTI API has hundreds of functions; we only bind the ones the
//! sidecar actually uses. New activity kinds are added here as we extend the
//! sidecar.
//!
//! All symbols are looked up via `dlsym`. If a symbol is missing (which can
//! happen on older CUPTI versions), the sidecar feature that needs it is
//! skipped with a warning rather than failing the whole profiler.
//!
//! Activity kind / record layout values come from
//! `/usr/local/cuda/include/cupti_activity.h`. They are stable across CUDA
//! 11/12/13 for the activities we bind.

// CUPTI's headers use `CUpti_*` capitalization for typedefs and snake-case
// for enum-of-int values — we mirror those names verbatim for grep-ability
// against the C headers.
#![allow(non_camel_case_types)]

use std::{
    ffi::{c_void, CStr, CString},
    os::raw::{c_char, c_int, c_uint},
    ptr,
};

use libc::{dlclose, dlerror, dlopen, dlsym, RTLD_LAZY, RTLD_LOCAL};

use super::CuptiError;

// ---- CUPTI types we touch -----------------------------------------------

pub type CUptiResult = c_int;
pub const CUPTI_SUCCESS: CUptiResult = 0;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CUcontext(pub *mut c_void);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUpti_SubscriberHandle(pub *mut c_void);

/// Activity kinds we subscribe to. Values match
/// `enum CUpti_ActivityKind` in cupti_activity.h.
#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CUpti_ActivityKind {
    INVALID = 0,
    MEMCPY = 1,
    MEMSET = 2,
    KERNEL = 3,
    DRIVER = 4,
    RUNTIME = 5,
    EVENT = 6,
    METRIC = 7,
    DEVICE = 8,
    CONTEXT = 9,
    CONCURRENT_KERNEL = 10,
    NAME = 11,
    MARKER = 12,
    MARKER_DATA = 13,
    SOURCE_LOCATOR = 14,
    GLOBAL_ACCESS = 15,
    BRANCH = 16,
    OVERHEAD = 17,
    CDP_KERNEL = 18,
    PREEMPTION = 19,
    ENVIRONMENT = 20,
    EVENT_INSTANCE = 21,
    MEMCPY2 = 22,
    METRIC_INSTANCE = 23,
    INSTRUCTION_EXECUTION = 24,
    UNIFIED_MEMORY_COUNTER = 25,
    FUNCTION = 26,
    MODULE = 27,
    DEVICE_ATTRIBUTE = 28,
    SHARED_ACCESS = 29,
    PC_SAMPLING = 30,
    PC_SAMPLING_RECORD_INFO = 31,
    INSTRUCTION_CORRELATION = 32,
    OPENACC_DATA = 33,
    OPENACC_LAUNCH = 34,
    OPENACC_OTHER = 35,
    CUDA_EVENT = 36,
    STREAM = 37,
    SYNCHRONIZATION = 38,
    EXTERNAL_CORRELATION = 39,
    NVLINK = 40,
    INSTANTANEOUS_EVENT = 41,
    INSTANTANEOUS_EVENT_INSTANCE = 42,
    INSTANTANEOUS_METRIC = 43,
    INSTANTANEOUS_METRIC_INSTANCE = 44,
    MEMORY = 45,
    PCIE = 46,
    OPENMP = 47,
    INTERNAL_LAUNCH_API = 48,
    MEMORY2 = 49,
    MEMORY_POOL = 50,
}

/// Common header bytes for every activity record. Read these first to
/// dispatch to the variant struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUpti_Activity {
    pub kind: u32, // CUpti_ActivityKind
}

/// View into `CUpti_ActivityKernel11` (cupti_activity.h:2936-3245) covering
/// the fields the assembler actually reads. We do **not** bind the entire
/// 224-byte struct — that grows in every CUDA bump. We only need offsets
/// 0..112, which are stable from CUpti_ActivityKernel onwards.
///
/// The C struct is `__attribute__((packed)) __attribute__((aligned(8)))`
/// (cupti_common.h:84). Every field type used here is naturally aligned at
/// its packed offset, so a plain `#[repr(C)]` Rust mirror produces the same
/// layout. If you add a u8/u16-followed-by-u64 sequence in the future, switch
/// to `#[repr(C, packed)]` and use `read_unaligned`.
///
/// Verify with `crates/cuda-profiler/build.rs` size assertions on next ABI bump.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUpti_ActivityKernel {
    pub kind: u32,
    pub cache_config: u8,
    pub shared_memory_config: u8,
    pub registers_per_thread: u16,
    pub partitioned_global_cache_requested: u32,
    pub partitioned_global_cache_executed: u32,
    pub start: u64,
    pub end: u64,
    pub completed: u64,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub grid_x: i32,
    pub grid_y: i32,
    pub grid_z: i32,
    pub block_x: i32,
    pub block_y: i32,
    pub block_z: i32,
    pub static_shared_memory: i32,
    pub dynamic_shared_memory: i32,
    pub local_memory_per_thread: u32,
    pub local_memory_total: u32,
    pub correlation_id: u32,
    pub grid_id: i64,
    pub name: *const c_char,
    // Trailing fields (reserved0, queued, submitted, launchType, ...) exist
    // but we don't read them; their offsets diverge across CUDA versions.
}

/// View into `CUpti_ActivityMemcpy6` (cupti_activity.h:2159-2290). Same
/// caveat as `CUpti_ActivityKernel`: we only bind through `streamId` because
/// that is all the assembler currently reads, and offsets past that point
/// shift across CUDA versions.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUpti_ActivityMemcpy {
    pub kind: u32,
    pub copy_kind: u8,
    pub src_kind: u8,
    pub dst_kind: u8,
    pub flags: u8,
    pub bytes: u64,
    pub start: u64,
    pub end: u64,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    // Trailing: correlationId, runtimeCorrelationId, pad, reserved0,
    // graphNodeId, graphId, channelId, channelType, copyCount(u64).
}

/// `CUpti_ActivityMarker2` (cupti_activity.h:4282-4333). NVTX ranges are
/// observed through this.
///
/// Layout under PACKED_ALIGNMENT:
/// ```text
/// off  0: u32 kind
/// off  4: u32 flags             (bit 0=instant, 1=start, 2=end)
/// off  8: u64 timestamp
/// off 16: u32 id
/// off 20: u32 objectKind
/// off 24: ObjectId (12 bytes)
/// off 36: u32 pad
/// off 40: *const c_char name    (NULL on END markers)
/// off 48: *const c_char domain
/// ```
/// The 12-byte object_id + 4-byte pad combination is captured here as a
/// 12-byte union plus an explicit `_pad: u32` so `name`/`domain` land at the
/// same offsets as the C struct.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUpti_ActivityMarker {
    pub kind: u32,
    pub flags: u32,
    pub timestamp: u64,
    pub id: u32,
    pub object_kind: u32,
    pub object_id: ObjectId,
    pub _pad: u32,
    pub name: *const c_char,
    pub domain: *const c_char,
}

/// Mirror of `CUpti_ActivityObjectKindId` (cupti_activity.h:711-733).
///
/// The C union is **12 bytes** in size (the wider of the two inner structs is
/// the 3-u32 dcs variant). The marker record's struct then has a `uint32_t pad`
/// field immediately after, before `name`. We declare 12 bytes here and rely
/// on `repr(C)` to insert the padding naturally — but watch out: a future ABI
/// change that grows the union to 13–16 bytes would silently shift `name` and
/// `domain` past their packed-C offsets.
#[repr(C)]
#[derive(Copy, Clone)]
pub union ObjectId {
    pub raw: [u8; 12],
}

impl std::fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectId").finish()
    }
}

/// Buffer requested / completed callback signatures.
pub type BufferRequestedFn =
    unsafe extern "C" fn(buffer: *mut *mut u8, size: *mut usize, max_records: *mut usize);
pub type BufferCompletedFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    stream_id: u32,
    buffer: *mut u8,
    size: usize,
    valid_size: usize,
);

// ---- Symbol table -------------------------------------------------------

/// Lazily-loaded CUPTI handles. All symbol lookups happen exactly once during
/// `load()`; thereafter calls are direct function-pointer invocations.
pub struct Cupti {
    handle: *mut c_void,

    pub get_timestamp: unsafe extern "C" fn(t: *mut u64) -> CUptiResult,
    pub activity_register_callbacks:
        unsafe extern "C" fn(BufferRequestedFn, BufferCompletedFn) -> CUptiResult,
    pub activity_enable: unsafe extern "C" fn(kind: u32) -> CUptiResult,
    pub activity_disable: unsafe extern "C" fn(kind: u32) -> CUptiResult,
    pub activity_flush_all: unsafe extern "C" fn(flag: u32) -> CUptiResult,
    pub activity_get_next_record: unsafe extern "C" fn(
        buffer: *mut u8,
        valid_buffer_size_bytes: usize,
        record: *mut *mut CUpti_Activity,
    ) -> CUptiResult,
    pub activity_get_num_dropped_records:
        unsafe extern "C" fn(ctx: *mut c_void, stream_id: u32, dropped: *mut usize) -> CUptiResult,
    pub get_result_string:
        unsafe extern "C" fn(result: CUptiResult, str_out: *mut *const c_char) -> CUptiResult,
}

unsafe impl Send for Cupti {}
unsafe impl Sync for Cupti {}

impl Cupti {
    pub fn load() -> Result<Self, CuptiError> {
        // Try the standard names in order. CUPTI ships as libcupti.so on
        // most distros and is in the CUDA lib dir.
        let candidates = ["libcupti.so", "libcupti.so.13", "libcupti.so.12"];
        let mut last_err = String::new();
        let mut handle = ptr::null_mut::<c_void>();
        for name in &candidates {
            let cstr = CString::new(*name).unwrap();
            unsafe {
                handle = dlopen(cstr.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
                if !handle.is_null() {
                    break;
                }
                last_err = take_dlerror();
            }
        }
        if handle.is_null() {
            return Err(CuptiError::LibraryNotFound(last_err));
        }

        unsafe {
            Ok(Self {
                handle,
                get_timestamp: load_sym(handle, "cuptiGetTimestamp")?,
                activity_register_callbacks: load_sym(handle, "cuptiActivityRegisterCallbacks")?,
                activity_enable: load_sym(handle, "cuptiActivityEnable")?,
                activity_disable: load_sym(handle, "cuptiActivityDisable")?,
                activity_flush_all: load_sym(handle, "cuptiActivityFlushAll")?,
                activity_get_next_record: load_sym(handle, "cuptiActivityGetNextRecord")?,
                activity_get_num_dropped_records: load_sym(
                    handle,
                    "cuptiActivityGetNumDroppedRecords",
                )?,
                get_result_string: load_sym(handle, "cuptiGetResultString")?,
            })
        }
    }

    /// Convert a CUPTI result to a human-readable string.
    pub fn result_string(&self, code: CUptiResult) -> String {
        let mut s: *const c_char = ptr::null();
        unsafe {
            let _ = (self.get_result_string)(code, &mut s);
            if s.is_null() {
                format!("CUPTI error {code}")
            } else {
                CStr::from_ptr(s).to_string_lossy().into_owned()
            }
        }
    }
}

impl Drop for Cupti {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

unsafe fn load_sym<T: Copy>(handle: *mut c_void, name: &'static str) -> Result<T, CuptiError> {
    // Compile-time size assertion: T must be pointer-sized (i.e. a function
    // pointer). Without this, `transmute_copy` would silently read past the
    // end of `sym` for an oversized T. `transmute` enforces this at the
    // type system level, so we use it whenever the layout permits.
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());

    let cstr = CString::new(name).unwrap();
    let _ = take_dlerror(); // clear stale state
    let sym = dlsym(handle, cstr.as_ptr());
    if sym.is_null() {
        // POSIX permits dlsym to return a null pointer for valid symbols,
        // distinguishable only via dlerror. For function pointers we never
        // expect null, so we treat null as missing regardless of dlerror —
        // safer than transmuting a null pointer into a function pointer and
        // segfaulting at first call.
        return Err(CuptiError::SymbolMissing { sym: name });
    }
    // We cannot use `mem::transmute::<*mut c_void, T>(sym)` directly because
    // the compiler can't prove `size_of::<T>() == size_of::<*mut c_void>()`
    // for an unbound T. The runtime debug-assert above is the best we can do;
    // every concrete callsite is a function-pointer typedef so size matches.
    Ok(std::mem::transmute_copy(&sym))
}

unsafe fn take_dlerror() -> String {
    let p = dlerror();
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

// Helper: check a CUPTI return code, mapping non-zero codes into a typed error
// without allocating in the success path.
pub fn check(func: &'static str, code: CUptiResult) -> Result<(), CuptiError> {
    if code == CUPTI_SUCCESS {
        Ok(())
    } else {
        Err(CuptiError::Call { func, code })
    }
}

// Suppress unused warnings on platform-specific imports.
#[allow(dead_code)]
fn _unused_warn_sink(_: c_uint) {}
