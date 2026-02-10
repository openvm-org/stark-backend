use p3_challenger::{CanObserve, CanSample};
use p3_field::{BasedVectorSpace, PrimeField64};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

/// Unified trait describing the interface for the Fiat-Shamir transcript needed by the SWIRL
/// protocol.
pub trait FiatShamirTranscript<F, EF, Digest>: Clone + Send + Sync
where
    F: PrimeField64,
    EF: BasedVectorSpace<F>,
{
    fn observe(&mut self, value: F);
    fn sample(&mut self) -> F;

    /// Implementations should pass through to [Self::observe], but no default implementation is
    /// provided since an explicit conversion from `Digest` to array of `F` is required.
    fn observe_commit(&mut self, digest: Digest);

    fn observe_ext(&mut self, value: EF) {
        // for i in 0..D
        for &base_val in value.as_basis_coefficients_slice() {
            self.observe(base_val);
        }
    }

    fn sample_ext(&mut self) -> EF {
        EF::from_basis_coefficients_fn(|_| self.sample())
    }

    fn sample_bits(&mut self, bits: usize) -> u64 {
        assert!(bits < (u32::BITS as usize));
        assert!((1 << bits) < F::ORDER_U64);
        let rand_f: F = self.sample();
        let rand_u64 = rand_f.as_canonical_u64();
        rand_u64 & ((1 << bits) - 1)
    }

    #[must_use]
    fn check_witness(&mut self, bits: usize, witness: F) -> bool {
        self.observe(witness);
        self.sample_bits(bits) == 0
    }

    #[instrument(name = "grind_pow", skip_all)]
    fn grind(&mut self, bits: usize) -> F {
        assert!(bits < (u32::BITS as usize));
        assert!((1 << bits) < F::ORDER_U64);

        let witness = (0..F::ORDER_U64)
            .into_par_iter()
            .map(F::from_u64)
            .find_any(|witness| self.clone().check_witness(bits, *witness))
            .expect("failed to find PoW witness");
        assert!(self.check_witness(bits, witness));
        witness
    }
}

impl<F, EF, Digest, C> FiatShamirTranscript<F, EF, Digest> for C
where
    F: PrimeField64,
    EF: BasedVectorSpace<F>,
    C: CanObserve<F> + CanObserve<Digest> + CanSample<F> + Clone + Send + Sync,
{
    fn observe(&mut self, value: F) {
        CanObserve::observe(self, value);
    }

    fn sample(&mut self) -> F {
        CanSample::sample(self)
    }

    fn observe_commit(&mut self, digest: Digest) {
        CanObserve::observe(self, digest);
    }
}
