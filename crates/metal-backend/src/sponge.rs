use openvm_stark_backend::{
    p3_challenger::CanObserve,
    FiatShamirTranscript,
};
use p3_baby_bear::default_babybear_poseidon2_16;

use crate::types::{Challenger, Digest, F, SC, WIDTH};

/// Device-side sponge state, matching the Metal `DeviceSpongeState` struct.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DeviceSpongeState {
    pub state: [F; WIDTH],
    pub absorb_idx: u32,
    pub sample_idx: u32,
}

impl Default for DeviceSpongeState {
    fn default() -> Self {
        Self {
            state: [F::default(); WIDTH],
            absorb_idx: 0,
            sample_idx: 0,
        }
    }
}

/// GPU-accelerated duplex sponge. Currently runs on host only.
#[derive(Debug)]
pub struct DuplexSpongeMetal {
    host: Challenger,
}

impl Default for DuplexSpongeMetal {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for DuplexSpongeMetal {
    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
        }
    }
}

impl DuplexSpongeMetal {
    pub fn new() -> Self {
        Self {
            host: Challenger::new(default_babybear_poseidon2_16()),
        }
    }
}

impl FiatShamirTranscript<SC> for DuplexSpongeMetal {
    #[inline]
    fn observe(&mut self, value: F) {
        self.host.observe(value);
    }

    #[inline]
    fn sample(&mut self) -> F {
        use openvm_stark_backend::p3_challenger::CanSample;
        self.host.sample()
    }

    #[inline]
    fn observe_commit(&mut self, digest: Digest) {
        for x in digest {
            self.observe(x);
        }
    }
}
