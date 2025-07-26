use std::sync::Arc;

use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    p3_matrix::Matrix,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    AirRef, Chip, ChipUsageGetter,
};

use super::trace::generate_trace_rows;
use crate::dummy_airs::fib_air::{air::FibonacciAir, columns::NUM_FIBONACCI_COLS};

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

    pub fn air<SC: StarkGenericConfig>(&self) -> AirRef<SC> {
        Arc::new(FibonacciAir)
    }
}

impl<SC: StarkGenericConfig> Chip<(), CpuBackend<SC>> for FibonacciChip
where
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, _: ()) -> AirProvingContext<CpuBackend<SC>> {
        let common_main = generate_trace_rows::<Val<SC>>(self.a, self.b, self.n);
        let a = common_main.get(0, 0);
        let b = common_main.get(0, 1);
        let last_val = common_main.get(self.n - 1, 1);
        AirProvingContext::simple(Arc::new(common_main), vec![a, b, last_val])
    }
}

impl ChipUsageGetter for FibonacciChip {
    fn air_name(&self) -> String {
        "FibonacciAir".to_string()
    }
    fn current_trace_height(&self) -> usize {
        self.n
    }
    fn trace_width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}
