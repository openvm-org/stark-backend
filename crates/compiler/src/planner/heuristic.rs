//! Solver-free heuristic backend: memory-aware greedy topological order,
//! offline best-fit-decreasing offset assignment, hill-climbing local
//! search over legal adjacent swaps with a repack after each accepted
//! move.

use crate::{
    graph_ir::BufInfo,
    planner::{
        ctx::{align_up, PlanCtx},
        plan::StreamMemoryPlan,
        PlanError,
    },
};

pub fn plan_heuristic(bufs: &[BufInfo], ctx: &PlanCtx) -> Result<StreamMemoryPlan, PlanError> {
    let (succ, indeg) = ctx.edges();

    let mut adj = vec![vec![false; ctx.n_nodes]; ctx.n_nodes];
    for (u, sv) in succ.iter().enumerate() {
        for &v in sv {
            adj[u][v] = true;
        }
    }

    let initial_order = greedy_topo(ctx, &succ, &indeg);
    let (mut best_offsets, mut best_peak) = pack_order(ctx, bufs, &initial_order);
    let mut best_order = initial_order;

    const MAX_PASSES: usize = 50;
    for _ in 0..MAX_PASSES {
        let mut improved = false;
        let mut order = best_order.clone();
        let n = order.len();
        let mut i = 0;
        while i + 1 < n {
            let a = order[i];
            let b = order[i + 1];
            if !adj[a][b] {
                order.swap(i, i + 1);
                let (offs, peak) = pack_order(ctx, bufs, &order);
                if peak < best_peak {
                    best_peak = peak;
                    best_offsets = offs;
                    best_order = order.clone();
                    improved = true;
                } else {
                    order.swap(i, i + 1);
                }
            }
            i += 1;
        }
        if !improved {
            break;
        }
    }

    Ok(StreamMemoryPlan::single_stream(
        best_order,
        best_offsets,
        best_peak,
        ctx.n_nodes,
    ))
}

fn greedy_topo(ctx: &PlanCtx, succ: &[Vec<usize>], initial_indeg: &[usize]) -> Vec<usize> {
    let n = ctx.n_nodes;
    let mut indeg = initial_indeg.to_vec();
    let mut order = Vec::with_capacity(n);

    let mut remaining_readers: Vec<usize> = ctx.readers.iter().map(|r| r.len()).collect();
    let mut alive: Vec<bool> = vec![false; ctx.n_bufs];
    let (node_writes, node_reads) = ctx.per_node_access();

    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();

    while !ready.is_empty() {
        let mut best_pos = 0;
        let mut best_key: (i64, i64) = (i64::MAX, i64::MIN);
        for (idx, &node) in ready.iter().enumerate() {
            let (nb, fr) = score_node(
                ctx,
                &node_writes,
                &node_reads,
                &alive,
                &remaining_readers,
                node,
            );
            let key = (nb, -fr);
            if key < best_key {
                best_key = key;
                best_pos = idx;
            }
        }
        let picked = ready.swap_remove(best_pos);
        order.push(picked);

        for &b in &node_writes[picked] {
            if ctx.packable(b) && !alive[b] {
                alive[b] = true;
            }
        }
        for &b in &node_reads[picked] {
            if remaining_readers[b] > 0 {
                remaining_readers[b] -= 1;
            }
            if remaining_readers[b] == 0 && ctx.packable(b) && alive[b] && !ctx.pinned[b] {
                alive[b] = false;
            }
        }
        for &s in &succ[picked] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }

    if order.len() != n {
        for i in 0..n {
            if !order.contains(&i) {
                order.push(i);
            }
        }
    }
    order
}

fn score_node(
    ctx: &PlanCtx,
    node_writes: &[Vec<usize>],
    node_reads: &[Vec<usize>],
    alive: &[bool],
    remaining_readers: &[usize],
    node: usize,
) -> (i64, i64) {
    let mut new_births = 0i64;
    for &b in &node_writes[node] {
        if ctx.packable(b) && !alive[b] {
            new_births += ctx.sizes[b];
        }
    }
    let mut freed = 0i64;
    for &b in &node_reads[node] {
        if remaining_readers[b] == 1 && ctx.packable(b) && !ctx.pinned[b] {
            if alive[b] || node_writes[node].contains(&b) {
                freed += ctx.sizes[b];
            }
        }
    }
    (new_births, freed)
}

/// Offline best-fit-decreasing packing: with `order` fixed, derive each
/// packable buffer's `[birth, death]` lifetime, sort buffers by size
/// (largest first) then place each at the smallest aligned offset that
/// avoids every already-placed lifetime-overlapping buffer.
pub fn pack_order(ctx: &PlanCtx, bufs: &[BufInfo], order: &[usize]) -> (Vec<Option<u64>>, u64) {
    let n_bufs = ctx.n_bufs;
    let n = order.len();

    let mut pos = vec![usize::MAX; ctx.n_nodes];
    for (i, &nid) in order.iter().enumerate() {
        pos[nid] = i;
    }

    let mut birth = vec![i64::MAX; n_bufs];
    let mut death = vec![i64::MIN; n_bufs];
    for b in 0..n_bufs {
        if !ctx.packable(b) {
            continue;
        }
        for &w in &ctx.writers[b] {
            birth[b] = birth[b].min(pos[w] as i64);
            death[b] = death[b].max(pos[w] as i64);
        }
        for &r in &ctx.readers[b] {
            death[b] = death[b].max(pos[r] as i64);
        }
        if ctx.writers[b].is_empty() {
            birth[b] = -1;
        }
        if ctx.pinned[b] || ctx.readers[b].is_empty() {
            death[b] = n as i64;
        }
    }

    let overlaps =
        |b1: usize, b2: usize| -> bool { !(death[b1] < birth[b2] || death[b2] < birth[b1]) };

    let mut to_place: Vec<usize> = (0..n_bufs).filter(|&b| ctx.packable(b)).collect();
    to_place.sort_by_key(|&b| (std::cmp::Reverse(ctx.sizes[b]), birth[b], b));

    let mut offsets: Vec<Option<u64>> = vec![None; n_bufs];
    let mut peak: i64 = 0;

    for &b in &to_place {
        let size = ctx.sizes[b];
        let align = (bufs[b].elem_size as i64).max(1);

        let mut blocked: Vec<(i64, i64)> = to_place
            .iter()
            .filter_map(|&b2| {
                if b2 == b {
                    return None;
                }
                let off = offsets[b2]? as i64;
                overlaps(b, b2).then_some((off, off + ctx.sizes[b2]))
            })
            .collect();
        blocked.sort_unstable();

        let mut o: i64 = 0;
        let mut placed = false;
        for &(a_off, a_end) in &blocked {
            let o_aligned = align_up(o, align);
            if o_aligned + size <= a_off {
                o = o_aligned;
                placed = true;
                break;
            }
            o = o.max(a_end);
        }
        if !placed {
            o = align_up(o, align);
        }
        offsets[b] = Some(o as u64);
        peak = peak.max(o + size);
    }

    (offsets, peak as u64)
}
