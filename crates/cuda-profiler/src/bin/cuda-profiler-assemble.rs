//! `cuda-profiler-assemble`: read a binary log produced by the in-process
//! profiler and emit a Perfetto-compatible Chrome trace JSON.
//!
//! The Chrome trace event format is the simpler of the two Perfetto inputs
//! (the other being protobuf). It loads cleanly in https://ui.perfetto.dev
//! and supports the SQL surface via `trace_processor`. We emit:
//!
//!   - One process for the GPU.
//!   - One thread per CUDA stream we observed (CUPTI activity).
//!   - One thread per SM for the per-CTA Gantt — this is the central view that nsys/ncu cannot
//!     give.
//!   - NVTX ranges as async events at the top-level GPU process.
//!   - Plan-selection events from runtime_choices.jsonl as instant events.
//!
//! Run:  `cuda-profiler-assemble shadow_profile.bin > trace.json`

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
};

use clap::Parser;
use openvm_cuda_profiler::record::{decode_payload, read_frame, Record, MAGIC, VERSION_MAJOR};

#[derive(Parser, Debug)]
#[command(
    name = "cuda-profiler-assemble",
    about = "Convert a SHDWPROF binary log to a Perfetto Chrome trace JSON",
    long_about = None,
)]
struct Cli {
    /// Input .bin file.
    input: PathBuf,

    /// Output Chrome trace JSON. Defaults to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optional `runtime_choices.jsonl` — adds plan-selection markers.
    #[arg(long)]
    runtime_choices: Option<PathBuf>,
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let mut r = BufReader::new(File::open(&cli.input)?);
    read_header(&mut r)?;

    let mut kernels: HashMap<u32, String> = HashMap::new();
    let mut events: Vec<TraceEvent> = Vec::new();
    let mut t0_ns: Option<u64> = None;
    let sm_pid: i64 = 1; // process id for the SM-lane group
    let stream_pid: i64 = 2;
    let nvtx_pid: i64 = 3;
    let plan_pid: i64 = 4;

    // Pair NVTX START/END markers by `(domain, id)`.
    // Value is `(name_from_start_if_seen, timestamp_ns_of_first_seen, kind_of_first_seen)`.
    let mut nvtx_open: HashMap<(String, u32), (String, u64, u32)> = HashMap::new();

    while let Some((tag, payload)) = read_frame(&mut r)? {
        let rec = match decode_payload(tag, &payload) {
            Ok(r) => r,
            Err(_) => continue, // Skip unknown/forward-compat tags.
        };
        match rec {
            Record::ProcessStart {
                start_walltime_ns, ..
            } => {
                t0_ns = Some(start_walltime_ns);
            }
            Record::KernelName { id, name } => {
                kernels.insert(id, name);
            }
            Record::Cta(c) => {
                let name = kernels
                    .get(&c.kernel_id)
                    .cloned()
                    .unwrap_or_else(|| format!("kid:{:#x}", c.kernel_id));
                events.push(TraceEvent::complete(
                    name,
                    "gpu_cta",
                    sm_pid,
                    c.smid as i64,
                    c.t_start,
                    c.t_end.saturating_sub(c.t_start),
                    serde_json::json!({
                        "block_linear": c.block_linear,
                        "kernel_id": format!("{:#010x}", c.kernel_id),
                    }),
                ));
            }
            Record::Drop {
                count,
                head_at_detection,
                approx_t_ns,
            } => {
                events.push(TraceEvent::instant(
                    format!("ring overrun: lost {count} CTAs"),
                    "gpu_meta",
                    sm_pid,
                    0,
                    approx_t_ns,
                    serde_json::json!({"head_at_detection": head_at_detection, "lost": count}),
                ));
            }
            Record::CuptiKernel {
                stream_id,
                t_start_ns,
                t_end_ns,
                name,
                ..
            } => {
                events.push(TraceEvent::complete(
                    name,
                    "gpu_kernel",
                    stream_pid,
                    stream_id as i64,
                    t_start_ns,
                    t_end_ns.saturating_sub(t_start_ns),
                    serde_json::json!({}),
                ));
            }
            Record::CuptiMemcpy {
                copy_kind,
                bytes,
                t_start_ns,
                t_end_ns,
                stream_id,
                ..
            } => {
                let label = format!("memcpy {} bytes (kind={})", bytes, copy_kind);
                events.push(TraceEvent::complete(
                    label,
                    "gpu_memcpy",
                    stream_pid,
                    stream_id as i64,
                    t_start_ns,
                    t_end_ns.saturating_sub(t_start_ns),
                    serde_json::json!({"bytes": bytes, "copy_kind": copy_kind}),
                ));
            }
            Record::CuptiNvtxRange {
                domain,
                name,
                timestamp_ns,
                id,
                marker_kind,
            } => {
                // Marker kinds (from CUPTI_ACTIVITY_FLAG_MARKER_*):
                //   1 = INSTANTANEOUS, 2 = START, 4 = END.
                const MK_INSTANT: u32 = 1;
                const MK_START: u32 = 2;
                const MK_END: u32 = 4;
                let key = (domain.clone(), id);
                if marker_kind == MK_INSTANT {
                    events.push(TraceEvent::instant(
                        format!("{}:{}", domain, name),
                        "nvtx",
                        nvtx_pid,
                        0,
                        timestamp_ns,
                        serde_json::json!({"id": id}),
                    ));
                } else if let Some((seen_name, seen_ts, seen_kind)) = nvtx_open.remove(&key) {
                    // Pair: figure out which is start/end and emit a complete event.
                    let (start_ts, end_ts, range_name) = match (seen_kind, marker_kind) {
                        (MK_START, MK_END) => (seen_ts, timestamp_ns, seen_name),
                        (MK_END, MK_START) => (timestamp_ns, seen_ts, name.clone()),
                        // Same-kind duplicate (rare): treat as instant, re-open the new one.
                        _ => {
                            nvtx_open
                                .insert(key.clone(), (name.clone(), timestamp_ns, marker_kind));
                            (seen_ts, seen_ts, seen_name)
                        }
                    };
                    let dur = end_ts.saturating_sub(start_ts);
                    events.push(TraceEvent::complete(
                        format!("{}:{}", domain, range_name),
                        "nvtx",
                        nvtx_pid,
                        0,
                        start_ts,
                        dur,
                        serde_json::json!({"id": id}),
                    ));
                } else {
                    nvtx_open.insert(key, (name, timestamp_ns, marker_kind));
                }
            }
            Record::CuptiOverhead {
                t_start_ns,
                t_end_ns,
                ..
            } => {
                events.push(TraceEvent::complete(
                    "cupti overhead".to_string(),
                    "gpu_overhead",
                    sm_pid,
                    -1,
                    t_start_ns,
                    t_end_ns.saturating_sub(t_start_ns),
                    serde_json::json!({}),
                ));
            }
            Record::RuntimeChoice {
                t_walltime_ns,
                json,
            } => {
                events.push(TraceEvent::instant(
                    "runtime_choice".to_string(),
                    "plan",
                    plan_pid,
                    0,
                    t_walltime_ns,
                    serde_json::from_str(&json).unwrap_or(serde_json::json!({"raw": json})),
                ));
            }
            Record::PcSample { .. } | Record::RangeMetric { .. } => {
                // Not emitted into the timeline yet (would clutter the view);
                // the assembler can be extended to roll these up into counter
                // tracks later. Present in the binary log for offline SQL.
            }
        }
    }

    // Drop unmatched NVTX opens to avoid losing them. Use a fixed lane (-2)
    // so they don't scatter across thousands of distinct tids by id.
    for ((dom, id), (n, start, _kind)) in nvtx_open {
        events.push(TraceEvent::instant(
            format!("{}:{} (unmatched)", dom, n),
            "nvtx_unmatched",
            nvtx_pid,
            -2,
            start,
            serde_json::json!({"id": id}),
        ));
    }

    // Optional runtime_choices.jsonl join (P5).
    //
    // NOTE on time domains: CTA records and CUPTI activity records use the
    // GPU's `%globaltimer` (ns since boot). Whatever produces
    // runtime_choices.jsonl currently emits wall-clock timestamps, which are
    // NOT in the same domain. Until both sides agree on the same clock,
    // these markers will appear at the right *order* relative to other
    // wall-clock events but at an arbitrary offset relative to GPU events.
    //
    // The plan's intent is to associate plan choices with NVTX phases by
    // *containment* (which NVTX range was open when the choice was made),
    // not by direct timestamp join. Implementing that correctly requires
    // either (a) emitting a `ClockSync` record at profiler init that pairs a
    // wall-clock and a GPU timestamp, or (b) recording the active NVTX
    // phase name alongside each runtime choice on the producer side.
    //
    // For now we emit the markers as instant events on a dedicated lane;
    // they're useful even mis-aligned because the assembler can identify
    // *which* family was selected for each AIR.
    if let Some(path) = cli.runtime_choices {
        if let Ok(content) = std::fs::read_to_string(&path) {
            for line in content.lines() {
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let t_ns = v
                    .get("t_ns")
                    .and_then(|x| x.as_u64())
                    .or_else(|| v.get("ts").and_then(|x| x.as_u64()))
                    .unwrap_or(0);
                events.push(TraceEvent::instant(
                    "plan-choice".to_string(),
                    "plan",
                    plan_pid,
                    0,
                    t_ns,
                    v,
                ));
            }
        }
    }

    // Stable order so file diffs are reproducible. f64 isn't Ord; convert to
    // bits — events with NaN ts wouldn't sort meaningfully anyway.
    events.sort_by(|a, b| {
        a.pid
            .cmp(&b.pid)
            .then(a.tid.cmp(&b.tid))
            .then(a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal))
    });

    let metadata = serde_json::json!({
        "gpu_t0_ns": t0_ns.unwrap_or(0),
        "kernels": kernels.iter().map(|(k, v)| (format!("{:#010x}", k), v)).collect::<HashMap<_, _>>(),
    });

    let output = serde_json::json!({
        "traceEvents": events,
        "displayTimeUnit": "ns",
        "metadata": metadata,
    });

    let mut w: Box<dyn Write> = match cli.output {
        Some(p) => Box::new(BufWriter::new(File::create(p)?)),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    serde_json::to_writer_pretty(&mut w, &output).map_err(std::io::Error::other)?;
    writeln!(w)?;
    Ok(())
}

fn read_header<R: Read>(r: &mut R) -> std::io::Result<()> {
    let mut buf = [0u8; 16];
    r.read_exact(&mut buf)?;
    if &buf[..8] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    let version = u16::from_le_bytes([buf[8], buf[9]]);
    if version != VERSION_MAJOR {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported version {version}"),
        ));
    }
    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct TraceEvent {
    name: String,
    cat: String,
    ph: String,
    pid: i64,
    tid: i64,
    ts: f64, // microseconds since epoch (Chrome trace convention)
    #[serde(skip_serializing_if = "Option::is_none")]
    dur: Option<f64>,
    args: serde_json::Value,
}

impl TraceEvent {
    fn complete(
        name: String,
        cat: &str,
        pid: i64,
        tid: i64,
        t_start_ns: u64,
        dur_ns: u64,
        args: serde_json::Value,
    ) -> Self {
        Self {
            name,
            cat: cat.to_string(),
            ph: "X".into(),
            pid,
            tid,
            ts: (t_start_ns as f64) / 1000.0,
            dur: Some((dur_ns as f64) / 1000.0),
            args,
        }
    }
    fn instant(
        name: String,
        cat: &str,
        pid: i64,
        tid: i64,
        t_ns: u64,
        args: serde_json::Value,
    ) -> Self {
        Self {
            name,
            cat: cat.to_string(),
            ph: "i".into(),
            pid,
            tid,
            ts: (t_ns as f64) / 1000.0,
            dur: None,
            args,
        }
    }
}
