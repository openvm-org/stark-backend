//! Row-major [ProverBackend] implementation.

use std::marker::PhantomData;

use openvm_stark_backend::{prover::ProverBackend, StarkProtocolConfig};
use p3_matrix::dense::RowMajorMatrix;

use crate::pcs_data::CpuStackedPcsData;

/// Row-major CPU prover backend.
///
/// Uses `RowMajorMatrix<SC::F>` as the matrix type for better cache locality
/// during constraint evaluation.
#[derive(Clone, Copy)]
pub struct CpuBackend<SC: StarkProtocolConfig>(PhantomData<SC>);

impl<SC: StarkProtocolConfig> CpuBackend<SC> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<SC: StarkProtocolConfig> Default for CpuBackend<SC> {
    fn default() -> Self {
        Self::new()
    }
}

impl<SC: StarkProtocolConfig> ProverBackend for CpuBackend<SC> {
    const CHALLENGE_EXT_DEGREE: u8 = SC::D_EF as u8;

    type Val = SC::F;
    type Challenge = SC::EF;
    type Commitment = SC::Digest;
    type Matrix = RowMajorMatrix<SC::F>;
    type OtherAirData = ();
    type PcsData = CpuStackedPcsData<SC::F, SC::Digest>;
}
