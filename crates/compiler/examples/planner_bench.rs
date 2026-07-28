//! Benchmark: compare CP-SAT vs. heuristic scheduler on random connected
//! DAGs of increasing size.
//!
//! Builds an N-node graph where each non-root node picks 1..=fanin random
//! predecessors from the already-generated nodes (yielding a connected
//! DAG), assigns each output buffer a random byte size, then plans it
//! twice — once with `SchedulerMode::CpSat { max_secs: 30.0 }` and once
//! with `SchedulerMode::Heuristic` — reporting wall time and peak bytes.
//!
//! Run with:
//!     cargo run -p crypto-compiler --features planner \
//!         --release --example planner_bench

use std::{collections::BTreeMap, time::Instant};

use crypto_compiler::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder},
    planner::{access_from_node, plan_raw, SchedulerMode},
    quast::Quast,
};

/// xorshift64* — cheap deterministic PRNG so the whole bench is
/// reproducible without pulling `rand` into the compiler's deps.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo)
    }
}

/// Builds a random connected DAG:
/// - Node 0 has no inputs (source producer).
/// - Node i > 0 picks 1..=fanin distinct inputs from nodes 0..i.
///
/// Every output buffer has a random size in `size_range`.
fn random_graph(n: usize, fanin: usize, size_range: (i64, i64), seed: u64) -> GraphBuilder {
    let mut g = GraphBuilder::new();
    let mut rng = Rng::new(seed);
    let mut outputs: Vec<BufId> = Vec::with_capacity(n);
    for i in 0..n {
        let sz = rng.range(size_range.0 as usize, size_range.1 as usize) as i64;
        let out = g.add_buf(BufInfo {
            name: Some(format!("out{i}")),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(sz),
            elem_size: 4,
        });
        // Inputs: for i > 0 pick between 1 and min(fanin, i) distinct
        // predecessors uniformly at random.
        let inputs: Vec<BufId> = if i == 0 {
            Vec::new()
        } else {
            let want = rng.range(1, fanin.min(i) + 1);
            let mut picks = Vec::with_capacity(want);
            let mut tries = 0;
            while picks.len() < want && tries < want * 4 {
                let j = rng.range(0, i);
                let bid = outputs[j];
                if !picks.contains(&bid) {
                    picks.push(bid);
                }
                tries += 1;
            }
            picks
        };
        let modifies: Vec<bool> = inputs.iter().map(|_| false).collect();
        g.insert_blackbox_kernel(
            format!("k{i}"),
            inputs.into_iter(),
            [out].into_iter(),
            modifies.into_iter(),
            |_, _, _| {},
        );
        outputs.push(out);
    }
    g
}

fn bench_one(name: &str, n: usize, fanin: usize, seed: u64) {
    let size_range = (128, 8192);
    let g = random_graph(n, fanin, size_range, seed);
    let nodes: Vec<_> = g.nodes.iter().map(access_from_node).collect();
    let env = BTreeMap::new();
    let device = DeviceType::Cuda(0);

    println!(
        "\n=== {name}: N={n} nodes, fanin<={fanin}, size range {}..{} bytes ===",
        size_range.0, size_range.1
    );

    // Heuristic first (cheap, always finishes).
    let t0 = Instant::now();
    let heur = plan_raw(
        &g.bufs,
        &nodes,
        &env,
        device,
        &[],
        &SchedulerMode::Heuristic,
    )
    .expect("heuristic plan");
    let heur_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "Heuristic : {heur_ms:>10.2} ms   peak = {:>10} bytes",
        heur.peak_bytes
    );

    // CP-SAT with the default 30 s cap.
    let t0 = Instant::now();
    let cpsat = plan_raw(
        &g.bufs,
        &nodes,
        &env,
        device,
        &[],
        &SchedulerMode::CpSat { max_secs: 30.0 },
    );
    let cpsat_ms = t0.elapsed().as_secs_f64() * 1e3;
    match cpsat {
        Ok(p) => {
            println!(
                "CP-SAT    : {cpsat_ms:>10.2} ms   peak = {:>10} bytes",
                p.peak_bytes
            );
            let ratio = p.peak_bytes as f64 / heur.peak_bytes.max(1) as f64;
            println!("Heuristic / CP-SAT peak ratio: {:.3}", 1.0 / ratio);
        }
        Err(e) => println!("CP-SAT    : {cpsat_ms:>10.2} ms   FAILED: {e}"),
    }
}

fn main() {
    // Same seed across sizes so the graph families share PRNG state
    // patterns; each `bench_one` call still gets a distinct sub-seed.
    for (n, seed) in [
        (10usize, 0xC0FFEE_u64),
        (100, 0xDEADBEEF),
        (500, 0x1234_5678),
    ] {
        bench_one(&format!("random-dag-{n}"), n, 3, seed);
    }
}
