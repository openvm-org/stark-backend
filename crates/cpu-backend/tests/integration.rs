use openvm_stark_sdk::config::baby_bear_poseidon2::*;

type Engine =
    openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine<DuplexSponge>;
openvm_backend_tests::backend_test_suite!(Engine);
