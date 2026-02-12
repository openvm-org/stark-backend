use std::sync::Arc;

use p3_field::PrimeField32;
use p3_matrix::Matrix;

use super::trace::generate_trace_rows;
use crate::{
    test_utils::dummy_airs::fib_air::air::FibonacciAir,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
    AirRef, ChipV2, StarkProtocolConfig,
};

#[derive(Clone, Debug)]
pub struct FibonacciChip {
    /// The 0th number in the fibonacci sequence.
    pub a: u32,
    /// The 1st number in the fibonacci sequence.
    pub b: u32,
    /// Target n-th number in the fibonacci sequence.
    pub n: usize,
}

impl FibonacciChip {
    pub fn new(a: u32, b: u32, n: usize) -> Self {
        assert!(n.is_power_of_two());
        Self { a, b, n }
    }

    pub fn air<SC: StarkProtocolConfig>(&self) -> AirRef<SC> {
        Arc::new(FibonacciAir)
    }
}

impl<SC: StarkProtocolConfig> ChipV2<(), CpuBackendV2<SC>> for FibonacciChip
where
    SC::F: PrimeField32,
{
    fn generate_proving_ctx(&self, _: ()) -> AirProvingContextV2<CpuBackendV2<SC>> {
        let common_main = generate_trace_rows::<SC::F>(self.a, self.b, self.n);
        let a = common_main.get(0, 0).expect("matrix index out of bounds");
        let b = common_main.get(0, 1).expect("matrix index out of bounds");
        let last_val = common_main
            .get(self.n - 1, 1)
            .expect("matrix index out of bounds");
        AirProvingContextV2::simple(
            ColMajorMatrix::from_row_major(&common_main),
            vec![a, b, last_val],
        )
    }
}
