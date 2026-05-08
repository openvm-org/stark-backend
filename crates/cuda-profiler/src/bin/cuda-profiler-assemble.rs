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
    collections::{HashMap, HashSet},
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

    // Sub-lane spread per primary lane. Multiple CTAs are routinely resident
    // on the same SM concurrently (H100 allows up to 32), and Perfetto's
    // slice tracker drops complete events that overlap on a single
    // (pid, tid) — surfacing as `slice_drop_overlapping_complete_event` in
    // the trace's import errors. The same applies to CUPTI kernel records
    // that genuinely run concurrently on a single CUDA stream (rare, but it
    // does happen with default-stream + per-thread-default-stream mixes).
    //
    // We post-process the buffered CTA / kernel events with a greedy
    // sub-lane packer so each (smid, sub) and (stream, sub) pair stays
    // strictly serial on its tid. Tid encoding is `primary * SUBLANES_STRIDE
    // + sub`; the stride is large enough to keep distinct primaries from
    // colliding even with worst-case concurrency.
    const SUBLANES_STRIDE: i64 = 64;

    // Defer per-SM CTA emission so we can pack overlapping records onto
    // sub-lanes after the full scan. (smid, t_start, t_end, name, kid, blkl).
    let mut cta_buf: Vec<(u32, u64, u64, String, u32, u32)> = Vec::new();
    // Same idea for CUPTI kernel records on stream lanes.
    let mut stream_kernel_buf: Vec<(u32, u64, u64, String)> = Vec::new();
    // CUPTI memcpy goes onto its own sub-lane group within each stream so it
    // doesn't fight kernels for sub-lane 0.
    let mut stream_memcpy_buf: Vec<(u32, u64, u64, String, serde_json::Value)> = Vec::new();

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
                cta_buf.push((c.smid, c.t_start, c.t_end, name, c.kernel_id, c.block_linear));
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
                stream_kernel_buf.push((stream_id, t_start_ns, t_end_ns, name));
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
                stream_memcpy_buf.push((
                    stream_id,
                    t_start_ns,
                    t_end_ns,
                    label,
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

    // ---- Sub-lane packing for SM CTAs ----------------------------------
    // Sort by primary (smid) then by start time so the greedy lane assigner
    // sees events in temporal order within each SM.
    cta_buf.sort_by_key(|x| (x.0, x.1));
    {
        // Per-SM lane end-time stacks. Index = primary smid (small dense
        // integer space; max ~131 on H100 SXM).
        let mut lanes: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut used_tids: HashSet<(u32, u32)> = HashSet::new();
        for (smid, t_start, t_end, name, kid, blkl) in cta_buf {
            let v = lanes.entry(smid).or_default();
            // Greedy: assign to the first lane that's free at t_start.
            // Strict-less-than (not `<=`) so that exactly-abutting events
            // (`prev.t_end == this.t_start` in u64 ns) land on *different*
            // sub-lanes. Same-lane abutters trigger a Perfetto false-positive
            // overlap because Perfetto computes `ts + dur` for the previous
            // X event in f64 and the resulting microsecond value can differ
            // from the next event's `ts` by one ulp.
            let lane = match v.iter().position(|&end| end < t_start) {
                Some(i) => {
                    v[i] = t_end;
                    i as u32
                }
                None => {
                    v.push(t_end);
                    (v.len() - 1) as u32
                }
            };
            // Each unique (smid, lane) becomes its own tid. Track for the
            // thread_name metadata pass below.
            used_tids.insert((smid, lane));
            let tid = (smid as i64) * SUBLANES_STRIDE + (lane as i64);
            events.push(TraceEvent::complete(
                name,
                "gpu_cta",
                sm_pid,
                tid,
                t_start,
                t_end.saturating_sub(t_start),
                serde_json::json!({
                    "block_linear": blkl,
                    "kernel_id": format!("{:#010x}", kid),
                }),
            ));
        }
        // thread_name M events so Perfetto labels each (smid, lane) clearly.
        for (smid, lane) in used_tids {
            let tid = (smid as i64) * SUBLANES_STRIDE + (lane as i64);
            events.push(TraceEvent::thread_name(sm_pid, tid, format!("SM {smid} / {lane}")));
        }
        events.push(TraceEvent::process_name(sm_pid, "GPU SMs (per-CTA)".into()));
    }

    // ---- Sub-lane packing for CUPTI kernel records ----------------------
    // Per-stream kernels are normally serial, but CUDA does allow concurrent
    // kernels (e.g., per-thread default streams, NCCL background work), and
    // CUPTI wobble can produce 1-cycle apparent overlaps on adjacent
    // launches. Pack defensively.
    stream_kernel_buf.sort_by_key(|x| (x.0, x.1));
    let mut stream_lanes: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut stream_kernel_used: HashSet<(u32, u32)> = HashSet::new();
    for (stream_id, t_start, t_end, name) in stream_kernel_buf {
        let v = stream_lanes.entry(stream_id).or_default();
        let lane = match v.iter().position(|&end| end <= t_start) {
            Some(i) => {
                v[i] = t_end;
                i as u32
            }
            None => {
                v.push(t_end);
                (v.len() - 1) as u32
            }
        };
        stream_kernel_used.insert((stream_id, lane));
        let tid = (stream_id as i64) * SUBLANES_STRIDE + (lane as i64);
        events.push(TraceEvent::complete(
            name,
            "gpu_kernel",
            stream_pid,
            tid,
            t_start,
            t_end.saturating_sub(t_start),
            serde_json::json!({}),
        ));
    }
    for (stream_id, lane) in stream_kernel_used {
        let tid = (stream_id as i64) * SUBLANES_STRIDE + (lane as i64);
        events.push(TraceEvent::thread_name(
            stream_pid,
            tid,
            if lane == 0 {
                format!("stream {stream_id}")
            } else {
                format!("stream {stream_id} (concurrent #{lane})")
            },
        ));
    }

    // Memcpy lanes live in a separate tid range from kernels within the same
    // stream (so the kernel timeline isn't visually shoved aside by long copies).
    // We offset by SUBLANES_STRIDE/2 to keep them unambiguously distinct.
    stream_memcpy_buf.sort_by_key(|x| (x.0, x.1));
    let mut memcpy_lanes: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut memcpy_used: HashSet<(u32, u32)> = HashSet::new();
    for (stream_id, t_start, t_end, label, args) in stream_memcpy_buf {
        let v = memcpy_lanes.entry(stream_id).or_default();
        let lane = match v.iter().position(|&end| end <= t_start) {
            Some(i) => {
                v[i] = t_end;
                i as u32
            }
            None => {
                v.push(t_end);
                (v.len() - 1) as u32
            }
        };
        memcpy_used.insert((stream_id, lane));
        let tid =
            (stream_id as i64) * SUBLANES_STRIDE + (SUBLANES_STRIDE / 2) + (lane as i64);
        events.push(TraceEvent::complete(
            label,
            "gpu_memcpy",
            stream_pid,
            tid,
            t_start,
            t_end.saturating_sub(t_start),
            args,
        ));
    }
    for (stream_id, lane) in memcpy_used {
        let tid =
            (stream_id as i64) * SUBLANES_STRIDE + (SUBLANES_STRIDE / 2) + (lane as i64);
        events.push(TraceEvent::thread_name(
            stream_pid,
            tid,
            if lane == 0 {
                format!("stream {stream_id} memcpy")
            } else {
                format!("stream {stream_id} memcpy (concurrent #{lane})")
            },
        ));
    }
    events.push(TraceEvent::process_name(stream_pid, "GPU streams".into()));
    events.push(TraceEvent::process_name(nvtx_pid, "NVTX".into()));
    events.push(TraceEvent::process_name(plan_pid, "Plan choices".into()));

    // Stable order so file diffs are reproducible.
    events.sort_by(|a, b| {
        a.pid
            .cmp(&b.pid)
            .then(a.tid.cmp(&b.tid))
            .then(a.t_ns.cmp(&b.t_ns))
    });

    // Compute t0 from the first non-metadata timestamp (events are now sorted
    // by (pid, tid, t_ns), so the min over t_ns lives in here). Subtracting
    // `t0_ns` in u64 space *before* dividing by 1000 is what keeps the f64
    // ts microsecond values precise enough not to introduce phantom overlaps.
    let t0_ns_for_rebase = events
        .iter()
        .filter(|e| e.ph != "M" && e.t_ns != 0)
        .map(|e| e.t_ns)
        .min()
        .unwrap_or(0);

    let metadata = serde_json::json!({
        "wallclock_t0_ns": t0_ns.unwrap_or(0),
        "gpu_t0_ns": t0_ns_for_rebase,
        "kernels": kernels.iter().map(|(k, v)| (format!("{:#010x}", k), v)).collect::<HashMap<_, _>>(),
    });

    let trace_events: Vec<serde_json::Value> =
        events.iter().map(|e| e.to_json(t0_ns_for_rebase)).collect();

    let output = serde_json::json!({
        "traceEvents": trace_events,
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

/// In-memory trace event. We keep the raw nanosecond timestamps as `u64` and
/// only convert to the Chrome-trace `ts` / `dur` floats (in microseconds) at
/// serialization time, after a single `rebase` pass subtracts a common
/// `t0_ns`. That order matters: `t_ns` is up to ~2^60-ish (nanoseconds since
/// boot, ~1.8e18 in this run), and dividing it by 1000 directly produces a
/// `f64` ~1.78e15 µs which has worse-than-microsecond precision — adjacent
/// events that are *non-overlapping* in `u64` ns can become apparently
/// overlapping after the lossy divide, and Perfetto then drops them with
/// `slice_drop_overlapping_complete_event`. Subtracting a per-trace `t0_ns`
/// in integer space first keeps the post-divide `f64` under ~3e8 µs (a few
/// hundred seconds), where f64 has ~1e-8 µs precision — comfortably below the
/// ns we care about.
#[derive(Debug)]
struct TraceEvent {
    name: String,
    cat: &'static str,
    ph: &'static str,
    pid: i64,
    tid: i64,
    /// Raw start-time ns. For M (metadata) events this is 0 and is unused.
    t_ns: u64,
    /// Raw duration ns. Zero for I/M events.
    dur_ns: u64,
    args: serde_json::Value,
}

impl TraceEvent {
    fn complete(
        name: String,
        cat: &'static str,
        pid: i64,
        tid: i64,
        t_start_ns: u64,
        dur_ns: u64,
        args: serde_json::Value,
    ) -> Self {
        Self {
            name,
            cat,
            ph: "X",
            pid,
            tid,
            t_ns: t_start_ns,
            dur_ns,
            args,
        }
    }
    fn instant(
        name: String,
        cat: &'static str,
        pid: i64,
        tid: i64,
        t_ns: u64,
        args: serde_json::Value,
    ) -> Self {
        Self {
            name,
            cat,
            ph: "i",
            pid,
            tid,
            t_ns,
            dur_ns: 0,
            args,
        }
    }
    /// Chrome-trace metadata event that names a thread (`(pid, tid)`).
    /// `ph: "M"` events are not timed; `ts` and `dur` are ignored by Perfetto.
    fn thread_name(pid: i64, tid: i64, name: String) -> Self {
        Self {
            name: "thread_name".into(),
            cat: "__metadata",
            ph: "M",
            pid,
            tid,
            t_ns: 0,
            dur_ns: 0,
            args: serde_json::json!({ "name": name }),
        }
    }
    /// Chrome-trace metadata event that names a process (`pid`).
    fn process_name(pid: i64, name: String) -> Self {
        Self {
            name: "process_name".into(),
            cat: "__metadata",
            ph: "M",
            pid,
            tid: 0,
            t_ns: 0,
            dur_ns: 0,
            args: serde_json::json!({ "name": name }),
        }
    }

    /// Convert to the JSON shape Perfetto expects, with timestamps rebased
    /// by `t0_ns` and divided by 1000 (ns -> µs).
    fn to_json(&self, t0_ns: u64) -> serde_json::Value {
        let ts_us = if self.ph == "M" {
            0.0
        } else {
            (self.t_ns.saturating_sub(t0_ns) as f64) / 1000.0
        };
        let mut obj = serde_json::Map::new();
        obj.insert("name".into(), serde_json::Value::String(self.name.clone()));
        obj.insert("cat".into(), serde_json::Value::String(self.cat.to_string()));
        obj.insert("ph".into(), serde_json::Value::String(self.ph.to_string()));
        obj.insert("pid".into(), self.pid.into());
        obj.insert("tid".into(), self.tid.into());
        obj.insert(
            "ts".into(),
            serde_json::Number::from_f64(ts_us)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
        );
        if self.ph == "X" {
            let dur_us = (self.dur_ns as f64) / 1000.0;
            obj.insert(
                "dur".into(),
                serde_json::Number::from_f64(dur_us)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        obj.insert("args".into(), self.args.clone());
        serde_json::Value::Object(obj)
    }
}
