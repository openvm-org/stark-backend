//! Sampled synthetic benchmark runner — idea 0008 phase 3 (cuda-backend / GPU).
//!
//! Reads a captured profile JSONL (segment-by-segment AIR shapes from
//! `SHADOW_BENCH_PROFILE_PATH`), samples a fraction of segments, and
//! replays each as a synthetic `Coordinator::prove` call on the GPU.
//! Outputs a scorecard JSON with per-segment timings.
//!
//! Usage:
//!     cargo run --release -p openvm-cuda-backend \
//!         --example synthetic_runner -- \
//!         --profile benchmarks/synthetic/reth-block-23992138-profile.jsonl \
//!         --sample-frac 0.1 --seed 42 \
//!         --max-log-height 22 \
//!         --out benchmarks/synthetic/scorecard.json
//!
//! Set `VPMM_PAGE_SIZE`/`VPMM_PAGES` to size the GPU memory pool
//! (4 MiB × 4096 = 16 GiB matches the upstream driver default; same is
//! fine here).

use std::{
    collections::HashSet,
    env,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

use eyre::eyre;
use openvm_cuda_backend::{prelude::SC, BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_stark_backend::{
    prover::{
        segment_profile::{AirShapeRecord, SegmentProfile},
        AirProvingContext, ColMajorMatrix, DeviceDataTransporter, ProvingContext,
    },
    AirRef, StarkEngine, SystemParams,
};
use openvm_stark_sdk::{
    bench::synthetic_air::{SyntheticAir, SyntheticShape},
    config::app_params_with_100_bits_security,
};
use p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::Serialize;

#[derive(Serialize)]
struct SegmentResult {
    segment_idx: usize,
    num_airs: usize,
    /// Sum of (height * common_main_width) across all AIRs in the segment.
    total_main_cells: u64,
    keygen_ms: u128,
    prove_ms: u128,
    /// True if any AIR's captured log_height exceeded `max_log_height`
    /// and was clamped down for this run.
    clamped: bool,
}

#[derive(Serialize)]
struct Scorecard<'a> {
    profile_path: &'a str,
    sample_frac: f64,
    seed: u64,
    max_log_height: usize,
    total_segments: usize,
    sampled_segments: usize,
    skipped_segments: usize,
    total_keygen_ms: u128,
    total_prove_ms: u128,
    results: Vec<SegmentResult>,
}

fn parse_args() -> eyre::Result<(PathBuf, f64, u64, usize, Option<PathBuf>)> {
    let mut profile: Option<PathBuf> = None;
    let mut sample_frac: f64 = 0.1;
    let mut seed: u64 = 42;
    let mut max_log_height: usize = 22;
    let mut out: Option<PathBuf> = None;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--profile" => profile = args.next().map(PathBuf::from),
            "--sample-frac" => sample_frac = args.next().unwrap().parse()?,
            "--seed" => seed = args.next().unwrap().parse()?,
            "--max-log-height" => max_log_height = args.next().unwrap().parse()?,
            "--out" => out = args.next().map(PathBuf::from),
            _ => eyre::bail!("unknown arg: {}", a),
        }
    }
    let profile = profile.ok_or_else(|| eyre!("--profile <path> is required"))?;
    Ok((profile, sample_frac, seed, max_log_height, out))
}

fn read_profile(path: &PathBuf) -> eyre::Result<Vec<SegmentProfile>> {
    let f = std::fs::File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let segment: SegmentProfile = serde_json::from_str(&line)?;
        out.push(segment);
    }
    Ok(out)
}

fn shape_from_record(rec: &AirShapeRecord, max_log_height: usize) -> SyntheticShape {
    let buses_set: HashSet<_> = rec.buses.iter().copied().collect();
    let log_height = rec.log_height.min(max_log_height);
    SyntheticShape {
        air_name: rec.air_name.clone(),
        log_height,
        preprocessed_width: rec.width.preprocessed.unwrap_or(0),
        cached_main_widths: rec.width.cached_mains.clone(),
        common_main_width: rec.width.common_main,
        after_challenge_widths: rec.width.after_challenge.clone(),
        num_constraints: rec.num_constraints,
        num_interactions: rec.num_interactions,
        num_distinct_buses: buses_set.len(),
        max_constraint_degree: rec.max_constraint_degree,
        interaction_message_lens: rec.interaction_message_lens.clone(),
        interaction_count_weights: rec.interaction_count_weights.clone(),
        occurrences: 1,
    }
}

/// Use the production `app_params_with_100_bits_security` config —
/// the standard 100-bit-security params. The captured distribution
/// lives within these bounds (heights ≤ 2^22, ≤ 100 AIRs, ≤ 5,000
/// constraints/AIR, ≤ 1,000 interactions/AIR). The widest captured AIR
/// is `KeccakfPermAir` at width 2,634 — w_stack=2,048 is the *stacked*
/// width bound on common_main partitions, NOT a per-AIR-width cap, so
/// this is fine.
fn build_params(max_log_height: usize) -> SystemParams {
    app_params_with_100_bits_security(max_log_height)
}

fn main() -> eyre::Result<()> {
    let (profile_path, sample_frac, seed, max_log_height, out_path) = parse_args()?;
    let segments = read_profile(&profile_path)?;
    println!(
        "loaded {} segments from {}",
        segments.len(),
        profile_path.display()
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let total = segments.len();
    let n = ((total as f64) * sample_frac).ceil() as usize;
    let n = n.min(total).max(1);
    let mut idxs: Vec<usize> = (0..total).collect();
    idxs.shuffle(&mut rng);
    let sample_idxs = &idxs[..n];
    let mut sample_idxs_sorted = sample_idxs.to_vec();
    sample_idxs_sorted.sort_unstable();
    println!(
        "sampled {}/{} segments (frac={:.3}, seed={}, max_log_h={})",
        n, total, sample_frac, seed, max_log_height
    );

    // One GpuEngine reused across all sampled segments. The CUDA stream
    // and VPMM are shared, so per-segment overhead is the host-side
    // keygen and the H2D/D2H transports — not GPU init.
    let params = build_params(max_log_height);
    let mut engine = BabyBearPoseidon2GpuEngine::new(params);
    // The captured runs disable this knob for AIRs without
    // interactions; safer default for our heterogeneous workload.
    engine
        .device_mut()
        .prover_config_mut()
        .zerocheck_save_memory = false;

    let mut results = Vec::new();
    let mut skipped = 0usize;
    for (i, &seg_idx) in sample_idxs_sorted.iter().enumerate() {
        let segment = &segments[seg_idx];
        let shapes: Vec<SyntheticShape> = segment
            .airs
            .iter()
            .map(|r| {
                let mut s = shape_from_record(r, max_log_height);
                // SyntheticAir reads `local` and `next` rows in eval, so
                // the trace must have height >= 2.
                s.log_height = s.log_height.max(1);
                s
            })
            .collect();
        let clamped = segment.airs.iter().any(|r| r.log_height > max_log_height);

        let airs: Vec<AirRef<_>> = shapes
            .iter()
            .map(|s| {
                let air: AirRef<_> = Arc::new(SyntheticAir::from_shape(s));
                air
            })
            .collect();

        // Keygen (host-side; engine type is GPU but keygen runs on host).
        let kg_start = Instant::now();
        let (pk, _vk) = engine.keygen(&airs);
        let device = engine.device();
        let d_pk =
            <_ as DeviceDataTransporter<SC, GpuBackend>>::transport_pk_to_device(device, &pk);
        let keygen_ms = kg_start.elapsed().as_millis();

        // Build per-AIR proving contexts with all-zero traces, transport
        // each to the device.
        let mut total_main_cells: u64 = 0;
        let mut per_trace = Vec::new();
        for (air_id, s) in shapes.iter().enumerate() {
            let synth = SyntheticAir::from_shape(s);
            let row_trace = synth.generate_trace::<BabyBear>(s.log_height);
            total_main_cells += ((1u64) << s.log_height as u32) * (synth.width() as u64);
            let d_trace = <_ as DeviceDataTransporter<SC, GpuBackend>>::transport_matrix_to_device(
                device,
                &ColMajorMatrix::from_row_major(&row_trace),
            );
            per_trace.push((air_id, AirProvingContext::simple_no_pis(d_trace)));
        }
        let ctx = ProvingContext::new(per_trace);

        let pv_start = Instant::now();
        let prove_res = engine.prove(&d_pk, ctx);
        let prove_ms = pv_start.elapsed().as_millis();
        if let Err(e) = prove_res {
            eprintln!(
                "[{}/{}] segment {} prove FAILED: {:?}",
                i + 1,
                n,
                seg_idx,
                e
            );
            skipped += 1;
            continue;
        }

        results.push(SegmentResult {
            segment_idx: seg_idx,
            num_airs: segment.airs.len(),
            total_main_cells,
            keygen_ms,
            prove_ms,
            clamped,
        });

        println!(
            "[{:>3}/{:>3}] seg {:>3} | {:>2} airs | {:>11} cells | keygen {:>5} ms | prove {:>6} ms{}",
            i + 1,
            n,
            seg_idx,
            segment.airs.len(),
            total_main_cells,
            keygen_ms,
            prove_ms,
            if clamped { " (clamped)" } else { "" },
        );
    }

    let total_keygen: u128 = results.iter().map(|r| r.keygen_ms).sum();
    let total_prove: u128 = results.iter().map(|r| r.prove_ms).sum();
    let scorecard = Scorecard {
        profile_path: profile_path.to_str().unwrap_or(""),
        sample_frac,
        seed,
        max_log_height,
        total_segments: total,
        sampled_segments: results.len(),
        skipped_segments: skipped,
        total_keygen_ms: total_keygen,
        total_prove_ms: total_prove,
        results,
    };
    let json = serde_json::to_string_pretty(&scorecard)?;
    println!(
        "\nsummary: {} segments, keygen {} ms total, prove {} ms total, {} skipped",
        scorecard.sampled_segments, total_keygen, total_prove, skipped
    );
    if let Some(out) = out_path {
        std::fs::write(&out, json)?;
        println!("wrote scorecard to {}", out.display());
    } else {
        println!("\n{}", json);
    }
    Ok(())
}
