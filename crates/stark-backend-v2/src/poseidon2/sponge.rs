use core::ops::Deref;

use p3_baby_bear::Poseidon2BabyBear;
use p3_challenger::CanObserve;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::Permutation;

use super::{poseidon2_perm, CHUNK, WIDTH};
use crate::{baby_bear_poseidon2::{Digest, F, BabyBearPoseidon2ConfigV2}, FiatShamirTranscript};

pub trait TranscriptHistory {
    fn len(&self) -> usize;
    fn into_log(self) -> TranscriptLog;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct TranscriptLog {
    /// Every sampled or observed value F
    values: Vec<F>,
    /// True iff values[tidx] was a sampled value
    is_sample: Vec<bool>,
    /// Sponge state after every permutation; note that not all implementations of
    /// TranscriptHistory will define this
    perm_results: Vec<[F; WIDTH]>,
}

impl TranscriptLog {
    pub fn new(values: Vec<F>, is_sample: Vec<bool>) -> Self {
        debug_assert_eq!(values.len(), is_sample.len());
        Self {
            values,
            is_sample,
            perm_results: vec![],
        }
    }

    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }

    pub fn samples(&self) -> &[bool] {
        &self.is_sample
    }

    pub fn samples_mut(&mut self) -> &mut [bool] {
        &mut self.is_sample
    }

    pub fn push_observe(&mut self, value: F) {
        self.values.push(value);
        self.is_sample.push(false);
    }

    pub fn push_sample(&mut self, value: F) {
        self.values.push(value);
        self.is_sample.push(true);
    }

    pub fn push_perm_result(&mut self, state: [F; WIDTH]) {
        self.perm_results.push(state);
    }

    pub fn extend_observe(&mut self, values: &[F]) {
        self.values.extend_from_slice(values);
        self.is_sample
            .extend(core::iter::repeat_n(false, values.len()));
    }

    pub fn extend_sample(&mut self, values: &[F]) {
        self.values.extend_from_slice(values);
        self.is_sample
            .extend(core::iter::repeat_n(true, values.len()));
    }

    pub fn extend_with_flags(&mut self, values: &[F], sample_flags: &[bool]) {
        debug_assert_eq!(values.len(), sample_flags.len());
        self.values.extend_from_slice(values);
        self.is_sample.extend(sample_flags.iter().copied());
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn into_parts(self) -> (Vec<F>, Vec<bool>) {
        (self.values, self.is_sample)
    }

    pub fn perm_results(&self) -> &Vec<[F; WIDTH]> {
        &self.perm_results
    }
}

impl Deref for TranscriptLog {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// Poseidon2-based duplex sponge in overwrite mode.
///
/// "Duplex" refers to being able to alternately absorb (observe) and squeeze
/// (sample), rather than a single absorb phase followed by a single squeeze
/// phase.
///
/// This variant operates in *overwrite mode*, meaning new inputs overwrite
/// state elements directly (instead of, e.g., being added in).
#[derive(Clone, Debug)]
pub struct DuplexSponge {
    perm: Poseidon2BabyBear<WIDTH>,
    /// Poseidon2 state
    state: [F; WIDTH],
    /// Invariant to be preserved: 0 <= absorb_idx < CHUNK
    absorb_idx: usize,
    /// Invariant to be preserved: 0 <= sample_idx <= CHUNK
    sample_idx: usize,
    /// True iff last sample/observe triggered a permutation
    last_op_perm: bool,
}

impl Default for DuplexSponge {
    fn default() -> Self {
        Self {
            perm: poseidon2_perm().clone(),
            state: [F::ZERO; WIDTH],
            absorb_idx: 0,
            sample_idx: 0,
            last_op_perm: false,
        }
    }
}

impl FiatShamirTranscript<BabyBearPoseidon2ConfigV2> for DuplexSponge {
    fn observe(&mut self, value: F) {
        self.state[self.absorb_idx] = value;
        self.absorb_idx += 1;
        self.last_op_perm = self.absorb_idx == CHUNK;
        if self.last_op_perm {
            self.perm.permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK;
        }
    }

    fn sample(&mut self) -> F {
        self.last_op_perm = self.absorb_idx != 0 || self.sample_idx == 0;
        if self.last_op_perm {
            self.perm.permute_mut(&mut self.state);
            self.absorb_idx = 0;
            self.sample_idx = CHUNK;
        }
        self.sample_idx -= 1;
        self.state[self.sample_idx]
    }

    fn observe_commit(&mut self, digest: [F; CHUNK]) {
        for x in digest {
            FiatShamirTranscript::observe(self, x);
        }
    }
}

impl CanObserve<F> for DuplexSponge {
    fn observe(&mut self, value: F) {
        FiatShamirTranscript::observe(self, value);
    }
}

impl CanObserve<Digest> for DuplexSponge {
    fn observe(&mut self, digest: Digest) {
        FiatShamirTranscript::observe_commit(self, digest);
    }
}

pub fn poseidon2_hash_slice(vals: &[F]) -> [F; CHUNK] {
    let perm = poseidon2_perm();
    let mut state = [F::ZERO; WIDTH];
    let mut i = 0;
    for &val in vals {
        state[i] = val;
        i += 1;
        if i == CHUNK {
            perm.permute_mut(&mut state);
            i = 0;
        }
    }
    if i != 0 {
        perm.permute_mut(&mut state);
    }
    state[..CHUNK].try_into().unwrap()
}

pub fn poseidon2_compress_with_capacity(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> ([F; CHUNK], [F; CHUNK]) {
    let mut state = [F::ZERO; WIDTH];
    state[..CHUNK].copy_from_slice(&left);
    state[CHUNK..].copy_from_slice(&right);
    poseidon2_perm().permute_mut(&mut state);
    (
        state[..CHUNK].try_into().unwrap(),
        state[CHUNK..].try_into().unwrap(),
    )
}

pub fn poseidon2_compress(left: [F; CHUNK], right: [F; CHUNK]) -> [F; CHUNK] {
    poseidon2_compress_with_capacity(left, right).0
}

pub fn poseidon2_tree_compress(mut hashes: Vec<Digest>) -> Digest {
    debug_assert!(hashes.len().is_power_of_two());
    while hashes.len() > 1 {
        let mut next = Vec::with_capacity(hashes.len() / 2);
        for pair in hashes.chunks_exact(2) {
            next.push(poseidon2_compress(pair[0], pair[1]));
        }
        hashes = next;
    }
    hashes.pop().unwrap()
}

#[derive(Clone)]
pub struct DuplexSpongeRecorder {
    pub inner: DuplexSponge,
    pub log: TranscriptLog,
}

impl Default for DuplexSpongeRecorder {
    fn default() -> Self {
        let mut log = TranscriptLog::default();
        log.push_perm_result([F::ZERO; WIDTH]);
        Self {
            inner: Default::default(),
            log,
        }
    }
}

impl FiatShamirTranscript<BabyBearPoseidon2ConfigV2> for DuplexSpongeRecorder {
    fn observe(&mut self, x: F) {
        <DuplexSponge as FiatShamirTranscript<BabyBearPoseidon2ConfigV2>>::observe(&mut self.inner, x);
        self.log.push_observe(x);
        if self.inner.last_op_perm {
            self.log.push_perm_result(self.inner.state);
        }
    }

    fn sample(&mut self) -> F {
        let x = self.inner.sample();
        self.log.push_sample(x);
        if self.inner.last_op_perm {
            self.log.push_perm_result(self.inner.state);
        }
        x
    }

    fn observe_commit(&mut self, digest: [F; CHUNK]) {
        for x in digest {
            self.observe(x);
        }
    }
}

impl TranscriptHistory for DuplexSpongeRecorder {
    fn len(&self) -> usize {
        self.log.len()
    }

    fn into_log(self) -> TranscriptLog {
        self.log
    }
}

/// Read-only transcript that replays a recorded log.
#[derive(Clone, Debug)]
pub struct ReadOnlyTranscript<'a> {
    log: &'a TranscriptLog,
    position: usize,
}

impl<'a> ReadOnlyTranscript<'a> {
    pub fn new(log: &'a TranscriptLog, start_idx: usize) -> Self {
        debug_assert!(start_idx <= log.len(), "start index out of bounds");
        Self {
            log,
            position: start_idx,
        }
    }
}

impl FiatShamirTranscript<BabyBearPoseidon2ConfigV2> for ReadOnlyTranscript<'_> {
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

    fn observe_commit(&mut self, digest: [F; CHUNK]) {
        for x in digest {
            self.observe(x);
        }
    }
}

impl TranscriptHistory for ReadOnlyTranscript<'_> {
    fn len(&self) -> usize {
        self.position
    }

    fn into_log(self) -> TranscriptLog {
        self.log.clone()
    }
}

#[cfg(test)]
mod test {
    use openvm_stark_sdk::config::baby_bear_poseidon2::Challenger;
    use p3_baby_bear::BabyBear;
    use p3_challenger::{CanObserve, CanSample};
    use p3_field::PrimeCharacteristicRing;

    use crate::{
        poseidon2::{
            poseidon2_perm,
            sponge::{DuplexSponge, DuplexSpongeRecorder, ReadOnlyTranscript, TranscriptHistory},
        },
        FiatShamirTranscript,
    };

    #[test]
    fn test_sponge() {
        let perm = poseidon2_perm();

        let mut challenger = Challenger::new(perm.clone());
        let mut sponge = DuplexSponge::default();

        for i in 0..5 {
            for _ in 0..(i + 1) * i {
                let a: BabyBear = CanSample::sample(&mut challenger);
                let b = sponge.sample();
                assert_eq!(a, b);
            }

            for j in 0..i * i {
                CanObserve::observe(&mut challenger, BabyBear::from_usize(j));
                FiatShamirTranscript::observe(&mut sponge, BabyBear::from_usize(j));
            }
        }
    }

    #[test]
    fn test_read_only_transcript() {
        // Record a sequence of operations
        let mut recorder = DuplexSpongeRecorder::default();
        recorder.observe(BabyBear::from_u32(42));
        recorder.observe(BabyBear::from_u32(100));
        let s1 = recorder.sample();
        recorder.observe(BabyBear::from_u32(200));
        let s2 = recorder.sample();
        let s3 = recorder.sample();

        let log = recorder.into_log();

        // Replay from start
        let mut replay = ReadOnlyTranscript::new(&log, 0);
        replay.observe(BabyBear::from_u32(42));
        replay.observe(BabyBear::from_u32(100));
        assert_eq!(replay.sample(), s1);
        replay.observe(BabyBear::from_u32(200));
        assert_eq!(replay.sample(), s2);
        assert_eq!(replay.sample(), s3);
        assert_eq!(replay.len(), 6);

        // Replay from middle
        let mut replay2 = ReadOnlyTranscript::new(&log, 2);
        assert_eq!(replay2.sample(), s1);
        replay2.observe(BabyBear::from_u32(200));
        assert_eq!(replay2.sample(), s2);
        assert_eq!(replay2.len(), 5);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "expected observe at 0")]
    fn test_read_only_transcript_wrong_operation() {
        let mut recorder = DuplexSpongeRecorder::default();
        let _ = recorder.sample();
        let log = recorder.into_log();

        let mut replay = ReadOnlyTranscript::new(&log, 0);
        replay.observe(BabyBear::from_u32(42)); // Should panic
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "value mismatch at 0")]
    fn test_read_only_transcript_wrong_value() {
        let mut recorder = DuplexSpongeRecorder::default();
        recorder.observe(BabyBear::from_u32(42));
        let log = recorder.into_log();

        let mut replay = ReadOnlyTranscript::new(&log, 0);
        replay.observe(BabyBear::from_u32(99)); // Should panic
    }
}
