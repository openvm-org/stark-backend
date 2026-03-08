//! SIMD-optimized transcript for the CPU backend.
//!
//! Wraps Plonky3's [`DuplexChallenger`] to get architecture-portable SIMD proof-of-work
//! grinding (4x throughput on NEON, 8x on AVX2). The standard [`DuplexSponge`] transcript
//! does scalar grinding with one permutation per candidate; this uses `F::Packing::WIDTH`
//! candidates per permutation call.
//!
//! [`DuplexChallenger`]: openvm_stark_backend::p3_challenger::DuplexChallenger
//! [`DuplexSponge`]: openvm_stark_backend::duplex_sponge::DuplexSponge

use openvm_stark_backend::{
    p3_challenger::{CanObserve, CanSample, GrindingChallenger},
    FiatShamirTranscript,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};

const WIDTH: usize = 16;
const RATE: usize = 8;
type Perm = Poseidon2BabyBear<WIDTH>;
type Inner = openvm_stark_backend::p3_challenger::DuplexChallenger<BabyBear, Perm, WIDTH, RATE>;

/// SIMD-optimized transcript backed by Plonky3's `DuplexChallenger`.
///
/// Produces identical Fiat-Shamir challenges as the standard `DuplexSponge` transcript,
/// but with ~4x faster proof-of-work grinding on aarch64 NEON (8x on x86 AVX2).
#[derive(Clone, Debug)]
pub struct CpuTranscript {
    inner: Inner,
}

impl From<Perm> for CpuTranscript {
    fn from(perm: Perm) -> Self {
        Self {
            inner: Inner::new(perm),
        }
    }
}

impl FiatShamirTranscript<BabyBearPoseidon2Config> for CpuTranscript {
    #[inline]
    fn observe(&mut self, value: BabyBear) {
        CanObserve::observe(&mut self.inner, value);
    }

    #[inline]
    fn sample(&mut self) -> BabyBear {
        CanSample::sample(&mut self.inner)
    }

    fn observe_commit(&mut self, digest: [BabyBear; RATE]) {
        for x in digest {
            CanObserve::observe(&mut self.inner, x);
        }
    }

    fn grind(&mut self, bits: usize) -> BabyBear {
        GrindingChallenger::grind(&mut self.inner, bits)
    }
}
