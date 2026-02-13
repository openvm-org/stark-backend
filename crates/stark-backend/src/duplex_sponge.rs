use p3_challenger::CanObserve;
use p3_field::PrimeField;
use p3_symmetric::CryptographicPermutation;

use crate::{FiatShamirTranscript, StarkProtocolConfig, TranscriptHistory, TranscriptLog};

/// Permutation-based duplex sponge in overwrite mode.
///
/// "Duplex" refers to being able to alternately absorb (observe) and squeeze
/// (sample), rather than a single absorb phase followed by a single squeeze
/// phase.
///
/// This variant operates in *overwrite mode*, meaning new inputs overwrite
/// state elements directly (instead of, e.g., being added in).
#[derive(Clone, Debug)]
pub struct DuplexSponge<F, P, const WIDTH: usize, const RATE: usize> {
    perm: P,
    /// Poseidon2 state
    state: [F; WIDTH],
    /// Invariant to be preserved: 0 <= absorb_idx < CHUNK
    absorb_idx: usize,
    /// Invariant to be preserved: 0 <= sample_idx <= CHUNK
    sample_idx: usize,
    /// True iff last sample/observe triggered a permutation
    last_op_perm: bool,
}

impl<F: Default + Copy, P, const WIDTH: usize, const RATE: usize> From<P>
    for DuplexSponge<F, P, WIDTH, RATE>
{
    fn from(perm: P) -> Self {
        Self {
            perm,
            state: [F::default(); WIDTH],
            absorb_idx: 0,
            sample_idx: 0,
            last_op_perm: false,
        }
    }
}

// This implementation **must** be equivalent to Plonky3's `DuplexChallenger`.
impl<SC, F, P, const WIDTH: usize, const RATE: usize> FiatShamirTranscript<SC>
    for DuplexSponge<F, P, WIDTH, RATE>
where
    F: PrimeField,
    SC: StarkProtocolConfig<Digest = [F; RATE], F = F>,
    P: CryptographicPermutation<[F; WIDTH]> + Send + Sync,
{
    fn observe(&mut self, value: F) {
        // See below
        CanObserve::observe(self, value);
    }

    fn sample(&mut self) -> F {
        self.last_op_perm = self.absorb_idx != 0 || self.sample_idx == 0;
        if self.last_op_perm {
            self.perm.permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = RATE;
        }
        self.sample_idx -= 1;
        self.state[self.sample_idx]
    }

    fn observe_commit(&mut self, digest: [F; RATE]) {
        // See below
        CanObserve::observe(self, digest);
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> CanObserve<F> for DuplexSponge<F, P, WIDTH, RATE>
where
    F: Clone,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        self.state[self.absorb_idx] = value;
        self.absorb_idx += 1;
        self.last_op_perm = self.absorb_idx == RATE;
        if self.last_op_perm {
            self.perm.permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = RATE;
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize, const N: usize> CanObserve<[F; N]>
    for DuplexSponge<F, P, WIDTH, RATE>
where
    F: Clone,
    P: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        for x in values {
            CanObserve::observe(self, x);
        }
    }
}

#[derive(Clone)]
pub struct DuplexSpongeRecorder<F, P, const WIDTH: usize, const RATE: usize> {
    pub inner: DuplexSponge<F, P, WIDTH, RATE>,
    pub log: TranscriptLog<F, [F; WIDTH]>,
}

impl<F: Default + Copy, P, const WIDTH: usize, const RATE: usize> From<P>
    for DuplexSpongeRecorder<F, P, WIDTH, RATE>
{
    fn from(perm: P) -> Self {
        let inner = DuplexSponge::from(perm);
        let mut log = TranscriptLog::default();
        log.push_perm_result([F::default(); WIDTH]);
        Self { inner, log }
    }
}

impl<SC, F, P, const WIDTH: usize, const RATE: usize> FiatShamirTranscript<SC>
    for DuplexSpongeRecorder<F, P, WIDTH, RATE>
where
    F: PrimeField,
    SC: StarkProtocolConfig<Digest = [F; RATE], F = F>,
    P: CryptographicPermutation<[F; WIDTH]> + Send + Sync,
{
    fn observe(&mut self, x: F) {
        CanObserve::observe(&mut self.inner, x);
        self.log.push_observe(x);
        if self.inner.last_op_perm {
            self.log.push_perm_result(self.inner.state);
        }
    }

    fn sample(&mut self) -> F {
        let x = FiatShamirTranscript::<SC>::sample(&mut self.inner);
        self.log.push_sample(x);
        if self.inner.last_op_perm {
            self.log.push_perm_result(self.inner.state);
        }
        x
    }

    fn observe_commit(&mut self, digest: [F; RATE]) {
        for x in digest {
            FiatShamirTranscript::<SC>::observe(self, x);
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> TranscriptHistory
    for DuplexSpongeRecorder<F, P, WIDTH, RATE>
{
    type F = F;
    type State = [F; WIDTH];

    fn len(&self) -> usize {
        self.log.len()
    }

    fn into_log(self) -> TranscriptLog<F, [F; WIDTH]> {
        self.log
    }
}

/// [DuplexSpongeRecorder] that checks the live transcript logs against a provided transcript log.
/// For testing usage.
#[derive(Clone)]
pub struct DuplexSpongeValidator<F, P, const WIDTH: usize, const RATE: usize> {
    pub inner: DuplexSpongeRecorder<F, P, WIDTH, RATE>,
    pub idx: usize,
    log: TranscriptLog<F, [F; WIDTH]>,
}

impl<F: Default + Copy, P, const WIDTH: usize, const RATE: usize>
    DuplexSpongeValidator<F, P, WIDTH, RATE>
{
    pub fn new(perm: P, log: TranscriptLog<F, [F; WIDTH]>) -> Self {
        debug_assert_eq!(log.len(), log.samples().len());
        Self {
            inner: perm.into(),
            idx: 0,
            log,
        }
    }
}

impl<SC, F, P, const WIDTH: usize, const RATE: usize> FiatShamirTranscript<SC>
    for DuplexSpongeValidator<F, P, WIDTH, RATE>
where
    F: PrimeField,
    SC: StarkProtocolConfig<Digest = [F; RATE], F = F>,
    P: CryptographicPermutation<[F; WIDTH]> + Send + Sync,
{
    fn observe(&mut self, x: F) {
        debug_assert!(self.idx < self.log.len(), "transcript replay overflow");
        assert!(!self.log.samples()[self.idx]);
        let exp_x = self.log[self.idx];
        assert_eq!(x, exp_x);
        self.idx += 1;
        FiatShamirTranscript::<SC>::observe(&mut self.inner, x);
    }

    fn sample(&mut self) -> F {
        debug_assert!(self.idx < self.log.len(), "transcript replay overflow");
        assert!(self.log.samples()[self.idx]);
        let x = FiatShamirTranscript::<SC>::sample(&mut self.inner);
        let exp_x = self.log[self.idx];
        self.idx += 1;
        assert_eq!(x, exp_x);
        x
    }

    fn observe_commit(&mut self, digest: [F; RATE]) {
        for x in digest {
            FiatShamirTranscript::<SC>::observe(self, x);
        }
    }
}

impl<F, P, const WIDTH: usize, const RATE: usize> TranscriptHistory
    for DuplexSpongeValidator<F, P, WIDTH, RATE>
{
    type F = F;
    type State = [F; WIDTH];

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn into_log(self) -> TranscriptLog<F, [F; WIDTH]> {
        debug_assert_eq!(self.inner.len(), self.log.len());
        debug_assert_eq!(
            self.inner.len(),
            self.idx,
            "transcript replay ended with {} of {} entries consumed",
            self.idx,
            self.inner.len()
        );
        debug_assert_eq!(
            self.log.len(),
            self.idx,
            "transcript replay ended with {} of {} entries consumed",
            self.idx,
            self.log.len()
        );
        self.inner.into_log()
    }
}
