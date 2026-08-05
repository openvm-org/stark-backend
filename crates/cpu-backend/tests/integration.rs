use openvm_stark_sdk::config::baby_bear_poseidon2::*;

type Engine =
    openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine<DuplexSponge>;
openvm_backend_tests::backend_test_suite!(Engine);

// ===========================================================================
// Non-power-of-two trace heights (chunked stacking) — CPU backends only
// ===========================================================================

#[test_case::test_case(2, 6, 12, 20 ; "l2 basic chunked pair")]
#[test_case::test_case(2, 6, 24, 28 ; "l2 tallest non pow2")]
#[test_case::test_case(2, 6, 12, 16 ; "l2 mixed with pow2")]
#[test_case::test_case(0, 8, 5, 7 ; "l0 odd heights")]
#[test_case::test_case(3, 5, 24, 40 ; "l3 chunked pair")]
#[test_case::test_case(2, 6, 2, 12 ; "l2 sub l_skip sender")]
fn test_non_pow2_heights_roundtrip(
    l_skip: usize,
    n_stack: usize,
    sender_height: usize,
    receiver_height: usize,
) -> eyre::Result<()> {
    openvm_backend_tests::non_pow2_heights_roundtrip::<Engine>(
        l_skip,
        n_stack,
        sender_height,
        receiver_height,
    )
}
