use openvm_benchmarks_fields::verify_all_fields;

#[test]
fn test_extension_field_verification() {
    assert!(verify_all_fields(1_000_000), "Extension field verification failed!");
}
