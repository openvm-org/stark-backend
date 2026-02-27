//! Unit tests for the transcript primitives (`DuplexSponge`, `DuplexSpongeRecorder`,
//! `ReadOnlyTranscript`).
//!
//! These verify observe/sample sequencing, value matching, and panic behavior on
//! misuse. No engine or prover backend is involved — they test the transcript
//! layer in isolation, so there is nothing to parameterize across backends.

use openvm_stark_backend::{FiatShamirTranscript, ReadOnlyTranscript, TranscriptHistory};
use openvm_stark_sdk::{config::baby_bear_poseidon2::*, p3_baby_bear::BabyBear};
use p3_challenger::{CanObserve, CanSample, DuplexChallenger};
use p3_field::PrimeCharacteristicRing;

type SC = BabyBearPoseidon2Config;

const WIDTH: usize = 16;
const CHUNK: usize = 8;

type Challenger = DuplexChallenger<BabyBear, p3_baby_bear::Poseidon2BabyBear<WIDTH>, WIDTH, CHUNK>;

#[test]
fn test_sponge() {
    let perm = poseidon2_perm();

    let mut challenger = Challenger::new(perm.clone());
    let mut sponge = default_duplex_sponge();

    for i in 0..5 {
        for _ in 0..(i + 1) * i {
            let a: BabyBear = CanSample::sample(&mut challenger);
            let b = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample(&mut sponge);
            assert_eq!(a, b);
        }

        for j in 0..i * i {
            CanObserve::observe(&mut challenger, BabyBear::from_usize(j));
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
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
    FiatShamirTranscript::<SC>::observe(&mut recorder, BabyBear::from_u32(42));
    FiatShamirTranscript::<SC>::observe(&mut recorder, BabyBear::from_u32(100));
    let s1 = FiatShamirTranscript::<SC>::sample(&mut recorder);
    FiatShamirTranscript::<SC>::observe(&mut recorder, BabyBear::from_u32(200));
    let s2 = FiatShamirTranscript::<SC>::sample(&mut recorder);
    let s3 = FiatShamirTranscript::<SC>::sample(&mut recorder);

    let log = recorder.into_log();

    // Replay from start
    let mut replay = ReadOnlyTranscript::new(&log, 0);
    FiatShamirTranscript::<SC>::observe(&mut replay, BabyBear::from_u32(42));
    FiatShamirTranscript::<SC>::observe(&mut replay, BabyBear::from_u32(100));
    assert_eq!(FiatShamirTranscript::<SC>::sample(&mut replay), s1);
    FiatShamirTranscript::<SC>::observe(&mut replay, BabyBear::from_u32(200));
    assert_eq!(FiatShamirTranscript::<SC>::sample(&mut replay), s2);
    assert_eq!(FiatShamirTranscript::<SC>::sample(&mut replay), s3);
    assert_eq!(replay.len(), 6);

    // Replay from middle
    let mut replay2 = ReadOnlyTranscript::new(&log, 2);
    assert_eq!(FiatShamirTranscript::<SC>::sample(&mut replay2), s1);
    FiatShamirTranscript::<SC>::observe(&mut replay2, BabyBear::from_u32(200));
    assert_eq!(FiatShamirTranscript::<SC>::sample(&mut replay2), s2);
    assert_eq!(replay2.len(), 5);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "expected observe at 0")]
fn test_read_only_transcript_wrong_operation() {
    let mut recorder = default_duplex_sponge_recorder();
    let _ = FiatShamirTranscript::<SC>::sample(&mut recorder);
    let log = recorder.into_log();

    let mut replay = ReadOnlyTranscript::new(&log, 0);
    FiatShamirTranscript::<SC>::observe(&mut replay, BabyBear::from_u32(42)); // Should panic
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "value mismatch at 0")]
fn test_read_only_transcript_wrong_value() {
    let mut recorder = default_duplex_sponge_recorder();
    FiatShamirTranscript::<SC>::observe(&mut recorder, BabyBear::from_u32(42));
    let log = recorder.into_log();

    let mut replay = ReadOnlyTranscript::new(&log, 0);
    FiatShamirTranscript::<SC>::observe(&mut replay, BabyBear::from_u32(99)); // Should panic
}
