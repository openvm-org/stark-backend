use std::ffi::c_void;

use openvm_benchmarks_fields::*;
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};

const TEST_ELEMENTS: usize = 1_000_000;

// ============================================================================
// FFI Bindings for Verification Kernels
// ============================================================================

#[link(name = "ext_field_bench")]
extern "C" {
    // Baby Bear Extensions
    fn verify_inv_fp4(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp4(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_fp5(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp5(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_fp6(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp6(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_fp2x3(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp2x3(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_fp3x2(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_fp3x2(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    // KoalaBear Extensions
    fn verify_inv_kb(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_kb5(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb5(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_kb6(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb6(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_kb2x3(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb2x3(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;

    fn verify_inv_kb3x2(failures: *mut u32, a: *const c_void, n: usize) -> i32;
    fn verify_distrib_kb3x2(
        failures: *mut u32,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        n: usize,
    ) -> i32;
}

// ============================================================================
// Verification Helper
// ============================================================================

/// Helper function to verify a field
fn verify_field(
    n: usize,
    elems_per_field: usize,
    init_fn: unsafe extern "C" fn(*mut c_void, *const u32, usize) -> i32,
    inv_fn: unsafe extern "C" fn(*mut u32, *const c_void, usize) -> i32,
    distrib_fn: unsafe extern "C" fn(
        *mut u32,
        *const c_void,
        *const c_void,
        *const c_void,
        usize,
    ) -> i32,
) -> Result<(), String> {
    // Create test data
    let d_a = random_u32s(n * elems_per_field, 11111).to_device().unwrap();
    let d_b = random_u32s(n * elems_per_field, 22222).to_device().unwrap();
    let d_c = random_u32s(n * elems_per_field, 33333).to_device().unwrap();

    // Initialize
    cuda_check(unsafe { init_fn(d_a.as_mut_raw_ptr(), d_a.as_ptr(), n) });
    cuda_check(unsafe { init_fn(d_b.as_mut_raw_ptr(), d_b.as_ptr(), n) });
    cuda_check(unsafe { init_fn(d_c.as_mut_raw_ptr(), d_c.as_ptr(), n) });

    // Test 1: a * inv(a) = 1
    let d_failures = DeviceBuffer::<u32>::with_capacity(1);
    d_failures.fill_zero().unwrap();
    cuda_check(unsafe { inv_fn(d_failures.as_mut_raw_ptr() as *mut u32, d_a.as_raw_ptr(), n) });
    let inv_failures = d_failures.to_host().unwrap();

    // Test 2: Distributivity
    let d_failures2 = DeviceBuffer::<u32>::with_capacity(1);
    d_failures2.fill_zero().unwrap();
    cuda_check(unsafe {
        distrib_fn(
            d_failures2.as_mut_raw_ptr() as *mut u32,
            d_a.as_raw_ptr(),
            d_b.as_raw_ptr(),
            d_c.as_raw_ptr(),
            n,
        )
    });
    let distrib_failures = d_failures2.to_host().unwrap();

    if inv_failures[0] != 0 {
        return Err(format!(
            "Inversion test failed with {} failures",
            inv_failures[0]
        ));
    }
    if distrib_failures[0] != 0 {
        return Err(format!(
            "Distributivity test failed with {} failures",
            distrib_failures[0]
        ));
    }

    Ok(())
}

// Baby Bear Extensions
#[test]
fn test_fp4_verification() {
    verify_field(
        TEST_ELEMENTS,
        4,
        init_fp4,
        verify_inv_fp4,
        verify_distrib_fp4,
    )
    .expect("Fp4 verification failed");
}

#[test]
fn test_fp5_verification() {
    verify_field(
        TEST_ELEMENTS,
        5,
        init_fp5,
        verify_inv_fp5,
        verify_distrib_fp5,
    )
    .expect("Fp5 verification failed");
}

#[test]
fn test_fp6_verification() {
    verify_field(
        TEST_ELEMENTS,
        6,
        init_fp6,
        verify_inv_fp6,
        verify_distrib_fp6,
    )
    .expect("Fp6 verification failed");
}

#[test]
fn test_fp2x3_verification() {
    verify_field(
        TEST_ELEMENTS,
        6,
        init_fp2x3,
        verify_inv_fp2x3,
        verify_distrib_fp2x3,
    )
    .expect("Fp2x3 verification failed");
}

#[test]
fn test_fp3x2_verification() {
    verify_field(
        TEST_ELEMENTS,
        6,
        init_fp3x2,
        verify_inv_fp3x2,
        verify_distrib_fp3x2,
    )
    .expect("Fp3x2 verification failed");
}

// KoalaBear Extensions
#[test]
fn test_kb_verification() {
    verify_field(TEST_ELEMENTS, 1, init_kb, verify_inv_kb, verify_distrib_kb)
        .expect("Kb verification failed");
}

#[test]
fn test_kb5_verification() {
    verify_field(
        TEST_ELEMENTS,
        5,
        init_kb5,
        verify_inv_kb5,
        verify_distrib_kb5,
    )
    .expect("Kb5 verification failed");
}

#[test]
fn test_kb6_verification() {
    verify_field(
        TEST_ELEMENTS,
        6,
        init_kb6,
        verify_inv_kb6,
        verify_distrib_kb6,
    )
    .expect("Kb6 verification failed");
}

#[test]
fn test_kb2x3_verification() {
    verify_field(
        TEST_ELEMENTS,
        6,
        init_kb2x3,
        verify_inv_kb2x3,
        verify_distrib_kb2x3,
    )
    .expect("Kb2x3 verification failed");
}

#[test]
fn test_kb3x2_verification() {
    verify_field(
        TEST_ELEMENTS,
        6,
        init_kb3x2,
        verify_inv_kb3x2,
        verify_distrib_kb3x2,
    )
    .expect("Kb3x2 verification failed");
}
