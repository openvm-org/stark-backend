//! Memory-metering validation runner (cuda-backend / GPU).
//!
//! Replays captured segment profiles (same input as `synthetic_runner`) through
//! `Coordinator::prove` on the GPU and compares the analytic proving-memory
//! estimate from `openvm_stark_backend::memory_metering` against the actual
//! peak GPU memory tracked by the cuda-common memory manager.
//!
//! Per segment it reports:
//! - the model estimate breakdown (main / rs_code_matrix / main_secondary / interaction /
//!   secondary_peak / total),
//! - the measured baseline before proving (device pk + traces), the measured peak during proving,
//!   per-phase peaks from the `gpu_mem.*` gauges emitted by `MemTracker`, and
//! - the estimate-vs-measured gap.
//!
//! Usage:
//!     cargo run --release -p openvm-benchmark-synthetic \
//!         --features cuda --bin mem_meter_runner -- \
//!         --profile benchmarks/synthetic/reth-block-23992138-profile.jsonl \
//!         --sample-frac 0.1 --seed 42 \
//!         --max-log-height 22 \
//!         --out benchmarks/synthetic/mem-report.json

use std::{
    collections::{BTreeMap, HashSet},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::Parser;
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use openvm_benchmark_synthetic::{
    segment_profile::{AirShapeRecord, SegmentProfile},
    synthetic_air::{SyntheticAir, SyntheticShape},
};
use openvm_cuda_backend::{DefaultHashScheme, GpuEngine};
use openvm_cuda_common::memory_manager::{
    device_memory_used, reset_session_peak, tracked_memory_stats,
};
use openvm_stark_backend::{
    memory_metering::ProvingMemoryCounts,
    prover::{
        stacked_pcs::StackedLayout, AirProvingContext, ColMajorMatrix, DeviceDataTransporter,
        ProvingContext,
    },
    AirRef, StarkEngine,
};
use openvm_stark_sdk::config::app_params_with_100_bits_security;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::Serialize;

#[derive(Serialize)]
struct EstimateReport {
    total: usize,
    main: usize,
    rs_code_matrix: usize,
    main_secondary: usize,
    interaction: usize,
    secondary_peak: usize,
    main_cells_with_rot: usize,
    main_cells_without_rot: usize,
    interaction_cells: usize,
}

#[derive(Serialize)]
struct MeasuredReport {
    /// Tracked bytes resident before `prove` (device pk + traces + transcript).
    baseline_bytes: usize,
    /// Peak tracked bytes during `prove` (absolute, includes baseline).
    peak_bytes: usize,
    /// Tracked bytes resident after `prove` returned.
    after_bytes: usize,
    /// `cudaMemGetInfo` used bytes before/after prove (whole device; catches
    /// allocations that bypass the memory manager).
    device_used_before: usize,
    device_used_after: usize,
    /// Absolute `gpu_mem.local_peak_bytes` gauge per MemTracker module label.
    phase_peaks: BTreeMap<String, u64>,
}

#[derive(Serialize)]
struct SegmentReport {
    segment_idx: usize,
    num_airs: usize,
    clamped: bool,
    /// Sum of `height * width` over the traces actually transported (bytes = cells * 4).
    trace_bytes: usize,
    /// Stacked-matrix layout of the common-main commit: `2^(l_skip+n_stack)` height and the
    /// number of stacked columns actually used.
    stacked_height: usize,
    stacked_width: usize,
    prove_ms: u128,
    estimate: EstimateReport,
    measured: MeasuredReport,
    /// `measured.peak_bytes - estimate.total` (positive = model underestimates).
    gap_bytes: i64,
    /// `measured.peak_bytes / estimate.total`.
    peak_over_estimate: f64,
}

#[derive(Serialize)]
struct RunHeader {
    profile_path: String,
    max_log_height: usize,
    zerocheck_save_memory: bool,
    cache_rs_code_matrix: bool,
    vpmm_page_size: Option<String>,
    log_blowup: usize,
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    max_constraint_degree: usize,
}

#[derive(Parser)]
#[command(about = "Memory-metering validation runner (cuda-backend / GPU)")]
struct Args {
    /// Path to the profile JSONL.
    #[arg(long)]
    profile: PathBuf,

    /// Comma-separated segment indices to run. Overrides sampling if set.
    #[arg(long)]
    segments: Option<String>,

    /// Fraction of segments to sample (0 < frac <= 1).
    #[arg(long, default_value_t = 0.1)]
    sample_frac: f64,

    /// RNG seed for the segment sampler.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Per-AIR `log_height` clamp. Anything above is clipped.
    #[arg(long, default_value_t = 22)]
    max_log_height: usize,

    /// Value for `GpuProverConfig::zerocheck_save_memory`. Defaults to the
    /// production setting for the params (`log_blowup == 1`).
    #[arg(long)]
    zerocheck_save_memory: Option<bool>,

    /// Enable `GpuProverConfig::cache_rs_code_matrix` (production default: off).
    #[arg(long, default_value_t = false)]
    cache_rs_code_matrix: bool,

    /// If set, write the report here as JSONL (one header line, then one line
    /// per segment, flushed incrementally). Otherwise print to stdout.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn read_profile(path: &Path) -> eyre::Result<Vec<SegmentProfile>> {
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

/// Extract the absolute `gpu_mem.local_peak_bytes` gauge per module label from a
/// metrics snapshot.
fn phase_peaks_from_snapshot(snapshotter: &Snapshotter) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for (ckey, _, _, value) in snapshotter.snapshot().into_vec() {
        let (_kind, key) = ckey.into_parts();
        if key.name() != "gpu_mem.local_peak_bytes" {
            continue;
        }
        let module = key
            .labels()
            .find(|l| l.key() == "module")
            .map(|l| l.value().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        if let DebugValue::Gauge(v) = value {
            out.insert(module, v.into_inner() as u64);
        }
    }
    out
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    eyre::ensure!(
        args.sample_frac.is_finite() && args.sample_frac > 0.0 && args.sample_frac <= 1.0,
        "--sample-frac must be in (0, 1], got {}",
        args.sample_frac
    );

    // Install a debugging recorder to capture the `gpu_mem.*` gauges emitted by
    // `MemTracker::emit_metrics` inside the prover phases.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder).expect("failed to install metrics recorder");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let segments = read_profile(&args.profile)?;
    println!(
        "loaded {} segments from {}",
        segments.len(),
        args.profile.display()
    );

    let sample_idxs_sorted: Vec<usize> = if let Some(list) = &args.segments {
        let mut idxs = list
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        idxs.sort_unstable();
        idxs.dedup();
        for &i in &idxs {
            eyre::ensure!(i < segments.len(), "segment index {i} out of range");
        }
        idxs
    } else {
        let mut rng = StdRng::seed_from_u64(args.seed);
        let total = segments.len();
        let n = ((total as f64) * args.sample_frac).ceil() as usize;
        let n = n.min(total).max(1);
        let mut idxs: Vec<usize> = (0..total).collect();
        idxs.shuffle(&mut rng);
        let mut sample = idxs[..n].to_vec();
        sample.sort_unstable();
        sample
    };
    println!(
        "running {}/{} segments (max_log_h={})",
        sample_idxs_sorted.len(),
        segments.len(),
        args.max_log_height
    );

    let params = app_params_with_100_bits_security(args.max_log_height);
    let mut engine = GpuEngine::<DefaultHashScheme>::new(params.clone());
    let zerocheck_save_memory = args.zerocheck_save_memory.unwrap_or(params.log_blowup == 1);
    engine
        .device_mut()
        .prover_config_mut()
        .zerocheck_save_memory = zerocheck_save_memory;
    engine.device_mut().prover_config_mut().cache_rs_code_matrix = args.cache_rs_code_matrix;

    let header = RunHeader {
        profile_path: args.profile.display().to_string(),
        max_log_height: args.max_log_height,
        zerocheck_save_memory,
        cache_rs_code_matrix: args.cache_rs_code_matrix,
        vpmm_page_size: std::env::var("VPMM_PAGE_SIZE").ok(),
        log_blowup: params.log_blowup,
        l_skip: params.l_skip,
        n_stack: params.n_stack,
        k_whir: params.k_whir(),
        max_constraint_degree: params.max_constraint_degree,
    };
    let mut out_file = args
        .out
        .as_ref()
        .map(|p| std::fs::File::create(p).expect("failed to create --out file"));
    let emit = |line: &str, out_file: &mut Option<std::fs::File>| {
        if let Some(f) = out_file {
            use std::io::Write;
            writeln!(f, "{line}").expect("failed to write report line");
            f.flush().expect("failed to flush report");
        } else {
            println!("{line}");
        }
    };
    emit(&serde_json::to_string(&header)?, &mut out_file);

    let mut reports = Vec::new();
    for (i, &seg_idx) in sample_idxs_sorted.iter().enumerate() {
        let segment = &segments[seg_idx];
        // Keygen rejects AIRs whose constraint degree exceeds the params limit.
        let max_degree = segment
            .airs
            .iter()
            .map(|r| r.max_constraint_degree)
            .max()
            .unwrap_or(0);
        if max_degree > params.max_constraint_degree {
            println!(
                "[{:>3}/{:>3}] seg {:>3} SKIPPED (constraint degree {} > {})",
                i + 1,
                sample_idxs_sorted.len(),
                seg_idx,
                max_degree,
                params.max_constraint_degree
            );
            continue;
        }
        let shapes: Vec<SyntheticShape> = segment
            .airs
            .iter()
            .map(|r| {
                let mut s = shape_from_record(r, args.max_log_height);
                // SyntheticAir reads `local` and `next` rows in eval, so
                // the trace must have height >= 2.
                s.log_height = s.log_height.max(1);
                s
            })
            .collect();
        let clamped = segment
            .airs
            .iter()
            .any(|r| r.log_height > args.max_log_height);

        let synths: Vec<Arc<SyntheticAir>> = shapes
            .iter()
            .map(|s| Arc::new(SyntheticAir::from_shape(s)))
            .collect();
        let airs: Vec<AirRef<_>> = synths.iter().map(|a| a.clone() as AirRef<_>).collect();

        let (pk, _vk) = engine.keygen(&airs);

        // Replicate OpenVM's metered-execution cell counting
        // (`SegmentationCtx::calculate_cell_counts`): per AIR,
        // `padded_height * total_width` split by `need_rot`, plus
        // `padded_height * num_interactions`.
        let mut main_cells_with_rot = 0usize;
        let mut main_cells_without_rot = 0usize;
        let mut interaction_cells = 0usize;
        for (shape, air_pk) in shapes.iter().zip(pk.per_air.iter()) {
            let padded_height = 1usize << shape.log_height;
            let width = air_pk.vk.params.width.total_width();
            if air_pk.vk.params.need_rot {
                main_cells_with_rot += padded_height * width;
            } else {
                main_cells_without_rot += padded_height * width;
            }
            interaction_cells += padded_height * air_pk.vk.symbolic_constraints.interactions.len();
        }
        let counts = ProvingMemoryCounts::new(
            main_cells_with_rot,
            main_cells_without_rot,
            interaction_cells,
        );
        let memory_config = engine.proving_memory_config();
        let estimate = memory_config.estimate(counts);

        // Stacked layout of the common-main commit (all synthetic traces are
        // common-main only), for offline model fitting.
        let stacked_layout = {
            let mut sorted_meta = shapes
                .iter()
                .zip(synths.iter())
                .map(|(s, synth)| (synth.width(), s.log_height))
                .collect::<Vec<_>>();
            sorted_meta.sort_by(|a, b| b.1.cmp(&a.1));
            StackedLayout::new(params.l_skip, params.l_skip + params.n_stack, sorted_meta)
                .expect("stacked layout")
        };

        let device = engine.device();
        let d_pk = <_ as DeviceDataTransporter<
            <GpuEngine<DefaultHashScheme> as StarkEngine>::SC,
            <GpuEngine<DefaultHashScheme> as StarkEngine>::PB,
        >>::transport_pk_to_device(device, &pk);

        let mut trace_bytes = 0usize;
        let mut per_trace = Vec::new();
        for (air_id, (s, synth)) in shapes.iter().zip(synths.iter()).enumerate() {
            let row_trace = synth.generate_trace::<_>(s.log_height);
            trace_bytes += (1usize << s.log_height) * synth.width() * size_of::<u32>();
            let d_trace = <_ as DeviceDataTransporter<
                <GpuEngine<DefaultHashScheme> as StarkEngine>::SC,
                <GpuEngine<DefaultHashScheme> as StarkEngine>::PB,
            >>::transport_matrix_to_device(
                device, &ColMajorMatrix::from_row_major(&row_trace)
            );
            per_trace.push((air_id, AirProvingContext::simple_no_pis(d_trace)));
        }
        let ctx = ProvingContext::new(per_trace);

        let device_used_before = device_memory_used();
        let baseline = tracked_memory_stats().current;
        reset_session_peak();

        let pv_start = std::time::Instant::now();
        let prove_res = engine.prove(&d_pk, ctx);
        let prove_ms = pv_start.elapsed().as_millis();

        let stats = tracked_memory_stats();
        let device_used_after = device_memory_used();
        let phase_peaks = phase_peaks_from_snapshot(&snapshotter);

        if let Err(e) = prove_res {
            eprintln!(
                "[{}/{}] segment {} prove FAILED: {:?}",
                i + 1,
                sample_idxs_sorted.len(),
                seg_idx,
                e
            );
            continue;
        }

        let gap_bytes = stats.session_peak as i64 - estimate.total as i64;
        let peak_over_estimate = stats.session_peak as f64 / estimate.total as f64;

        let peak_phase = phase_peaks
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.as_str())
            .unwrap_or("?");
        println!(
            "[{:>3}/{:>3}] seg {:>3} | {:>2} airs | est {:>6} MiB | peak {:>6} MiB | ratio {:.3} | base {:>5} MiB | peak@{} | {:>6} ms{}",
            i + 1,
            sample_idxs_sorted.len(),
            seg_idx,
            segment.airs.len(),
            estimate.total >> 20,
            stats.session_peak >> 20,
            peak_over_estimate,
            baseline >> 20,
            peak_phase,
            prove_ms,
            if clamped { " (clamped)" } else { "" },
        );

        let report = SegmentReport {
            segment_idx: seg_idx,
            num_airs: segment.airs.len(),
            clamped,
            trace_bytes,
            stacked_height: stacked_layout.height(),
            stacked_width: stacked_layout.width(),
            prove_ms,
            estimate: EstimateReport {
                total: estimate.total,
                main: estimate.main,
                rs_code_matrix: estimate.rs_code_matrix,
                main_secondary: estimate.main_secondary,
                interaction: estimate.interaction,
                secondary_peak: estimate.secondary_peak,
                main_cells_with_rot,
                main_cells_without_rot,
                interaction_cells,
            },
            measured: MeasuredReport {
                baseline_bytes: baseline,
                peak_bytes: stats.session_peak,
                after_bytes: stats.current,
                device_used_before,
                device_used_after,
                phase_peaks,
            },
            gap_bytes,
            peak_over_estimate,
        };
        emit(&serde_json::to_string(&report)?, &mut out_file);
        reports.push(report);
    }

    let underestimates = reports.iter().filter(|r| r.gap_bytes > 0).count();
    let max_ratio = reports
        .iter()
        .map(|r| r.peak_over_estimate)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\nsummary: {} segments, {} underestimated by the model, worst peak/estimate = {:.3}",
        reports.len(),
        underestimates,
        max_ratio
    );
    if let Some(out) = &args.out {
        println!("wrote report to {}", out.display());
    }
    Ok(())
}
