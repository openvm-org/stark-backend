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
            FiatShamirTranscript::<SC>::observe(&mut self.inner, x);
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

/// Read-only transcript that replays a recorded log.
#[derive(Clone, Debug)]
pub struct ReadOnlyTranscript<'a, F, State> {
    log: &'a TranscriptLog<F, State>,
    position: usize,
}

impl<'a, F, State> ReadOnlyTranscript<'a, F, State> {
    pub fn new(log: &'a TranscriptLog<F, State>, start_idx: usize) -> Self {
        debug_assert!(start_idx <= log.len(), "start index out of bounds");
        Self {
            log,
            position: start_idx,
        }
    }
}

impl<SC, F, const WIDTH: usize, const RATE: usize> FiatShamirTranscript<SC>
    for ReadOnlyTranscript<'_, F, [F; WIDTH]>
where
    F: PrimeField,
    SC: StarkProtocolConfig<F = F, Digest = [F; RATE]>,
{
    #[inline]
    fn observe(&mut self, value: F) {
        debug_assert!(
            !self.log.samples()[self.position],
            "expected observe at {}",
            self.position
        );
        debug_assert_eq!(
            self.log.values()[self.position],
            value,
            "value mismatch at {}",
            self.position
        );
        self.position += 1;
    }

    #[inline]
    fn sample(&mut self) -> F {
        debug_assert!(
            self.log.samples()[self.position],
            "expected sample at {}",
            self.position
        );
        let value = self.log.values()[self.position];
        self.position += 1;
        value
    }

    fn observe_commit(&mut self, digest: [F; RATE]) {
        for x in digest {
            FiatShamirTranscript::<SC>::observe(self, x);
        }
    }
}

impl<F, State> TranscriptHistory for ReadOnlyTranscript<'_, F, State>
where
    F: Clone,
    State: Clone,
{
    type F = F;
    type State = State;

    fn len(&self) -> usize {
        self.position
    }

    fn into_log(self) -> TranscriptLog<F, State> {
        self.log.clone()
    }
}

#[cfg(test)]
mod test {
    use p3_baby_bear::BabyBear;
    use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;

    use crate::{
        test_utils::{
            baby_bear_poseidon2::{self, BabyBearPoseidon2ConfigV2},
            default_duplex_sponge, default_duplex_sponge_recorder,
        },
        FiatShamirTranscript, TranscriptHistory,
    };

    type SCV2 = BabyBearPoseidon2ConfigV2;

    const WIDTH: usize = 16;
    const CHUNK: usize = 8;

    type Challenger =
        DuplexChallenger<BabyBear, p3_baby_bear::Poseidon2BabyBear<WIDTH>, WIDTH, CHUNK>;

    type ReadOnlyTranscript<'a> =
        super::ReadOnlyTranscript<'a, BabyBear, [BabyBear; WIDTH]>;

    #[test]
    fn test_sponge() {
        let perm = baby_bear_poseidon2::poseidon2_perm();

        let mut challenger = Challenger::new(perm.clone());
        let mut sponge = default_duplex_sponge();

        for i in 0..5 {
            for _ in 0..(i + 1) * i {
                let a: BabyBear = CanSample::sample(&mut challenger);
                let b =
                    FiatShamirTranscript::<BabyBearPoseidon2ConfigV2>::sample(&mut sponge);
                assert_eq!(a, b);
            }

            for j in 0..i * i {
                CanObserve::observe(&mut challenger, BabyBear::from_usize(j));
                FiatShamirTranscript::<BabyBearPoseidon2ConfigV2>::observe(
                    &mut sponge,
                    BabyBear::from_usize(j),
                );
            }
        }
    }

    #[test]
    fn test_read_only_transcript() {
        // Record a sequence of operations
        let mut recorder = default_duplex_sponge_recorder();
        FiatShamirTranscript::<SCV2>::observe(&mut recorder, BabyBear::from_u32(42));
        FiatShamirTranscript::<SCV2>::observe(&mut recorder, BabyBear::from_u32(100));
        let s1 = FiatShamirTranscript::<SCV2>::sample(&mut recorder);
        FiatShamirTranscript::<SCV2>::observe(&mut recorder, BabyBear::from_u32(200));
        let s2 = FiatShamirTranscript::<SCV2>::sample(&mut recorder);
        let s3 = FiatShamirTranscript::<SCV2>::sample(&mut recorder);

        let log = recorder.into_log();

        // Replay from start
        let mut replay = ReadOnlyTranscript::new(&log, 0);
        FiatShamirTranscript::<SCV2>::observe(&mut replay, BabyBear::from_u32(42));
        FiatShamirTranscript::<SCV2>::observe(&mut replay, BabyBear::from_u32(100));
        assert_eq!(FiatShamirTranscript::<SCV2>::sample(&mut replay), s1);
        FiatShamirTranscript::<SCV2>::observe(&mut replay, BabyBear::from_u32(200));
        assert_eq!(FiatShamirTranscript::<SCV2>::sample(&mut replay), s2);
        assert_eq!(FiatShamirTranscript::<SCV2>::sample(&mut replay), s3);
        assert_eq!(replay.len(), 6);

        // Replay from middle
        let mut replay2 = ReadOnlyTranscript::new(&log, 2);
        assert_eq!(FiatShamirTranscript::<SCV2>::sample(&mut replay2), s1);
        FiatShamirTranscript::<SCV2>::observe(&mut replay2, BabyBear::from_u32(200));
        assert_eq!(FiatShamirTranscript::<SCV2>::sample(&mut replay2), s2);
        assert_eq!(replay2.len(), 5);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "expected observe at 0")]
    fn test_read_only_transcript_wrong_operation() {
        let mut recorder = default_duplex_sponge_recorder();
        let _ = FiatShamirTranscript::<SCV2>::sample(&mut recorder);
        let log = recorder.into_log();

        let mut replay = ReadOnlyTranscript::new(&log, 0);
        FiatShamirTranscript::<SCV2>::observe(&mut replay, BabyBear::from_u32(42)); // Should panic
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "value mismatch at 0")]
    fn test_read_only_transcript_wrong_value() {
        let mut recorder = default_duplex_sponge_recorder();
        FiatShamirTranscript::<SCV2>::observe(&mut recorder, BabyBear::from_u32(42));
        let log = recorder.into_log();

        let mut replay = ReadOnlyTranscript::new(&log, 0);
        FiatShamirTranscript::<SCV2>::observe(&mut replay, BabyBear::from_u32(99)); // Should panic
    }
}
