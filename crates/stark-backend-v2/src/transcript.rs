use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::StarkProtocolConfig;

pub trait FiatShamirTranscript<SC: StarkProtocolConfig>: Clone + Send + Sync {
    fn observe(&mut self, value: SC::F);
    fn sample(&mut self) -> SC::F;

    fn observe_commit(&mut self, digest: SC::Digest);

    fn observe_ext(&mut self, value: SC::EF) {
        // for i in 0..D
        for &base_val in value.as_basis_coefficients_slice() {
            self.observe(base_val);
        }
    }

    fn sample_ext(&mut self) -> SC::EF {
        SC::EF::from_basis_coefficients_fn(|_| self.sample())
    }

    fn sample_bits(&mut self, bits: usize) -> u64 {
        assert!(bits < (u32::BITS as usize));
        assert!((1 << bits) < SC::F::ORDER_U64);
        let rand_f: SC::F = self.sample();
        let rand_u64 = rand_f.as_canonical_u64();
        rand_u64 & ((1 << bits) - 1)
    }

    #[must_use]
    fn check_witness(&mut self, bits: usize, witness: SC::F) -> bool {
        self.observe(witness);
        self.sample_bits(bits) == 0
    }

    #[instrument(name = "grind_pow", skip_all)]
    fn grind(&mut self, bits: usize) -> SC::F {
        assert!(bits < (u32::BITS as usize));
        assert!((1 << bits) < SC::F::ORDER_U64);

        let witness = (0..SC::F::ORDER_U64)
            .into_par_iter()
            .map(SC::F::from_u64)
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find PoW witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}
