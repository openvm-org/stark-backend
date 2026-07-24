//! CP-SAT-based memory planner.
//!
//! Given a graph built via [`GraphBuilder`] and an assignment for the
//! symbolic size variables it was built with, [`plan`] jointly picks:
//!
//! 1. A topological execution order for the nodes.
//! 2. A byte offset per buffer on the target device.
//!
//! such that the peak bytes needed on that device are minimized. Two
//! buffers must either be alive at disjoint times or occupy disjoint memory
//! regions; the CP-SAT model expresses this directly and lets the solver
//! trade off schedule and packing jointly.
//!
//! Feature-gated behind `planner`; requires an OR-Tools install accessible
//! to `cp_sat` (see the crate's `Cargo.toml`).

use std::collections::BTreeMap;

use cp_sat::{
    builder::{CpModelBuilder, IntVar, LinearExpr},
    proto::CpSolverStatus,
};

use crate::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder, GraphNode},
    ir::VarId,
};

/// Read/write set for one graph node, as seen by the planner.
///
/// This is the internal shape the CP-SAT model consumes: a node's `reads`
/// contribute to `death[b]` of every read buffer, its `writes` contribute
/// to `birth[b]` of every written buffer, and (writer, reader) pairs across
/// different nodes become precedence edges. A buffer that is both read
/// and written by the same node (e.g. an in-place modify, or a per-node
/// scratch region) has `birth[b] = death[b] = t[n]` and thus a single
/// time-step lifetime.
#[derive(Debug, Default, Clone)]
pub struct NodeAccess {
    pub reads: Vec<BufId>,
    pub writes: Vec<BufId>,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("size expression for buffer {buf:?} references unbound symbol {sym:?}")]
    UnboundSizeSymbol { buf: BufId, sym: VarId },
    #[error("size expression for buffer {buf:?} evaluates to a negative value {value}")]
    NegativeSize { buf: BufId, value: i64 },
    #[error("CP-SAT returned no solution (status: {0:?})")]
    NoSolution(CpSolverStatus),
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Permutation of node indices in execution order (`order[0]` runs
    /// first).
    pub order: Vec<usize>,
    /// Byte offset per [`BufId`]. `Some(off)` for buffers whose device
    /// matches the planner's target device, `None` otherwise.
    pub offsets: Vec<Option<u64>>,
    /// Peak bytes required on the target device.
    pub peak_bytes: u64,
}

/// Jointly plans execution order and buffer offsets on `device`.
///
/// Buffers whose device does not match `device` are ignored (they still
/// affect scheduling through their reads/writes but do not contribute to
/// the packed memory pool).
pub fn plan(
    graph: &GraphBuilder,
    env: &BTreeMap<VarId, i64>,
    device: DeviceType,
) -> Result<MemoryPlan, PlanError> {
    let nodes: Vec<NodeAccess> = graph.nodes.iter().map(access_from_node).collect();
    plan_raw(&graph.bufs, &nodes, env, device, &[])
}

/// Plans over an explicit `(bufs, nodes)` view. Prefer [`plan`] for the
/// common case where the accesses come straight from a [`GraphBuilder`];
/// this entry point exists so [`crate::graph_exe::GraphCompiler`] can inject
/// synthetic per-kernel scratch buffers into the model.
///
/// `exclude` lists buffers that participate in scheduling (so their reads
/// and writes still induce precedence edges between nodes) but that are
/// **not** packed into the returned pool: they get no offset and their
/// sizes don't count toward `peak_bytes`. This is what
/// [`crate::graph_exe::GraphCompiler`] uses for graph inputs/outputs — the
/// caller supplies their storage at run time so no pool slot is needed.
pub fn plan_raw(
    bufs: &[BufInfo],
    nodes: &[NodeAccess],
    env: &BTreeMap<VarId, i64>,
    device: DeviceType,
    exclude: &[BufId],
) -> Result<MemoryPlan, PlanError> {
    let n_nodes = nodes.len();
    let n_bufs = bufs.len();

    let mut excluded = vec![false; n_bufs];
    for &b in exclude {
        excluded[b.0] = true;
    }

    // Concrete sizes for buffers on the target device.
    let mut sizes = vec![0i64; n_bufs];
    let mut on_device = vec![false; n_bufs];
    for (idx, info) in bufs.iter().enumerate() {
        if info.device_type == device {
            on_device[idx] = true;
            sizes[idx] = eval_size(BufId(idx), &info.size, env)?;
        }
    }

    // Per-node read/write sets, already provided by the caller.
    let writes: Vec<Vec<usize>> = nodes
        .iter()
        .map(|a| a.writes.iter().map(|b| b.0).collect())
        .collect();
    let reads: Vec<Vec<usize>> = nodes
        .iter()
        .map(|a| a.reads.iter().map(|b| b.0).collect())
        .collect();

    // Per-buffer writer/reader node lists.
    let mut writers: Vec<Vec<usize>> = vec![vec![]; n_bufs];
    let mut readers: Vec<Vec<usize>> = vec![vec![]; n_bufs];
    for n in 0..n_nodes {
        for &b in &writes[n] {
            writers[b].push(n);
        }
        for &b in &reads[n] {
            readers[b].push(n);
        }
    }

    // Trivial early-out: nothing to schedule and nothing on-device.
    if n_nodes == 0 {
        return Ok(MemoryPlan {
            order: Vec::new(),
            offsets: vec![None; n_bufs],
            peak_bytes: 0,
        });
    }

    let mut m = CpModelBuilder::default();
    let last_time = (n_nodes as i64).saturating_sub(1);

    // Execution time slot per node — an all-different permutation over
    // [0, n_nodes).
    let t: Vec<IntVar> = (0..n_nodes)
        .map(|n| m.new_int_var_with_name([(0, last_time)], format!("t_{n}")))
        .collect();
    m.add_all_different(t.iter().copied());

    // Precedence: every writer of a buffer runs before every reader of it.
    for b in 0..n_bufs {
        for &w in &writers[b] {
            for &r in &readers[b] {
                if w != r {
                    m.add_lt(t[w], t[r]);
                }
            }
        }
    }

    // Device buffers with non-zero size and not in `exclude` are what we're
    // packing. Excluded buffers still participate in precedence (their
    // writers/readers appear in `writers`/`readers` above) but aren't
    // assigned offsets or interfered with in memory.
    let device_bufs: Vec<usize> = (0..n_bufs)
        .filter(|&b| on_device[b] && sizes[b] > 0 && !excluded[b])
        .collect();

    // Live interval [birth, death] per device buffer. Using the sentinel
    // range [-1, n_nodes] lets buffers that are never written (graph
    // inputs) be born at -1 and buffers that are never read (graph
    // outputs) die at n_nodes, so they interfere with everything.
    let mut birth: BTreeMap<usize, IntVar> = BTreeMap::new();
    let mut death: BTreeMap<usize, IntVar> = BTreeMap::new();
    for &b in &device_bufs {
        let birth_v = m.new_int_var_with_name([(-1, n_nodes as i64)], format!("birth_{b}"));
        let death_v = m.new_int_var_with_name([(-1, n_nodes as i64)], format!("death_{b}"));
        if writers[b].is_empty() {
            m.add_eq(birth_v, -1i64);
        } else if writers[b].len() == 1 {
            m.add_eq(birth_v, t[writers[b][0]]);
        } else {
            m.add_min_eq(birth_v, writers[b].iter().map(|&w| t[w]));
        }
        if readers[b].is_empty() {
            m.add_eq(death_v, n_nodes as i64);
        } else if readers[b].len() == 1 {
            m.add_eq(death_v, t[readers[b][0]]);
        } else {
            m.add_max_eq(death_v, readers[b].iter().map(|&r| t[r]));
        }
        birth.insert(b, birth_v);
        death.insert(b, death_v);
    }

    // Upper bound on peak memory = sum of all device buffer sizes.
    let sum_sizes: i64 = device_bufs.iter().map(|&b| sizes[b]).sum();

    let offsets: BTreeMap<usize, IntVar> = device_bufs
        .iter()
        .map(|&b| {
            (
                b,
                m.new_int_var_with_name([(0, sum_sizes)], format!("off_{b}")),
            )
        })
        .collect();

    // Pairwise no-overlap: either the buffers' lifetimes are disjoint, or
    // their memory regions are disjoint. Modeled as a disjunction over four
    // reified linear constraints.
    for (i, &b1) in device_bufs.iter().enumerate() {
        for &b2 in &device_bufs[i + 1..] {
            let lit_t12 = m.new_bool_var(); // death[b1] < birth[b2]
            let lit_t21 = m.new_bool_var(); // death[b2] < birth[b1]
            let lit_m12 = m.new_bool_var(); // off[b1] + size[b1] <= off[b2]
            let lit_m21 = m.new_bool_var(); // off[b2] + size[b2] <= off[b1]
            m.add_or([lit_t12, lit_t21, lit_m12, lit_m21]);

            let c = m.add_lt(death[&b1], birth[&b2]);
            m.only_enforce_if(c, [lit_t12]);
            let c = m.add_lt(death[&b2], birth[&b1]);
            m.only_enforce_if(c, [lit_t21]);
            let c = m.add_le(LinearExpr::from(offsets[&b1]) + sizes[b1], offsets[&b2]);
            m.only_enforce_if(c, [lit_m12]);
            let c = m.add_le(LinearExpr::from(offsets[&b2]) + sizes[b2], offsets[&b1]);
            m.only_enforce_if(c, [lit_m21]);
        }
    }

    // Objective: minimize the peak byte usage.
    let peak = m.new_int_var_with_name([(0, sum_sizes)], "peak");
    for &b in &device_bufs {
        m.add_ge(peak, LinearExpr::from(offsets[&b]) + sizes[b]);
    }
    m.minimize(peak);

    let response = m.solve();
    match response.status() {
        CpSolverStatus::Optimal | CpSolverStatus::Feasible => {}
        status => return Err(PlanError::NoSolution(status)),
    }

    let times: Vec<i64> = t.iter().map(|v| v.solution_value(&response)).collect();
    let mut order: Vec<usize> = (0..n_nodes).collect();
    order.sort_by_key(|&n| times[n]);

    let mut out_offsets = vec![None; n_bufs];
    for &b in &device_bufs {
        out_offsets[b] = Some(offsets[&b].solution_value(&response) as u64);
    }
    let peak_bytes = peak.solution_value(&response) as u64;

    Ok(MemoryPlan {
        order,
        offsets: out_offsets,
        peak_bytes,
    })
}

/// Builds a [`NodeAccess`] from a [`GraphNode`], matching the semantics
/// [`plan_raw`] expects (`modifies` on a `BlackboxKernel` input adds it to
/// both reads and writes; a `Const` writes its target; `Memcpy` reads src
/// and writes dst; etc.).
pub fn access_from_node(node: &GraphNode) -> NodeAccess {
    let mut a = NodeAccess::default();
    match node {
        GraphNode::BlackboxKernel(k) => {
            for (i, b) in k.inputs.iter().enumerate() {
                a.reads.push(*b);
                if k.modifies[i] {
                    a.writes.push(*b);
                }
            }
            a.writes.extend(k.outputs.iter().copied());
        }
        GraphNode::Kernel(k) => {
            a.reads.extend(k.inputs.iter().copied());
            a.writes.extend(k.outputs.iter().copied());
        }
        GraphNode::Const(c) => a.writes.push(c.buf),
        GraphNode::Memcpy(m) => {
            a.reads.push(m.src);
            a.writes.push(m.dst);
        }
        GraphNode::Memset(m) => a.writes.push(m.node),
    }
    a
}

fn eval_size(
    buf: BufId,
    size: &crate::quast::Quast,
    env: &BTreeMap<VarId, i64>,
) -> Result<i64, PlanError> {
    let mut syms = std::collections::BTreeSet::new();
    size.syms(&mut syms);
    for s in &syms {
        if !env.contains_key(s) {
            return Err(PlanError::UnboundSizeSymbol { buf, sym: *s });
        }
    }
    let v = size.eval(env);
    if v < 0 {
        return Err(PlanError::NegativeSize { buf, value: v });
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph_ir::{BufInfo, DeviceType},
        quast::Quast,
    };

    fn buf(g: &mut GraphBuilder, name: &str, size: i64) -> BufId {
        g.add_buf(BufInfo {
            name: Some(name.to_string()),
            device_type: DeviceType::Cuda(0),
            size: Quast::cst(size),
            elem_size: 4,
        })
    }

    /// Two blackbox kernels in a chain producing three buffers; peak memory
    /// with optimal reuse should be `max(a) + max(b, c)` after `a` dies.
    #[test]
    fn packs_disjoint_lifetimes() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 100);
        let b = buf(&mut g, "b", 200);
        let c = buf(&mut g, "c", 300);
        // k0: produces a
        g.insert_blackbox_kernel(
            "k0",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _| {},
        );
        // k1: reads a, produces b
        g.insert_blackbox_kernel(
            "k1",
            [a].into_iter(),
            [b].into_iter(),
            [false].into_iter(),
            |_, _| {},
        );
        // k2: reads b, produces c (a is dead by now)
        g.insert_blackbox_kernel(
            "k2",
            [b].into_iter(),
            [c].into_iter(),
            [false].into_iter(),
            |_, _| {},
        );

        let plan = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap();

        // Topological order: k0, k1, k2.
        assert_eq!(plan.order, vec![0, 1, 2]);

        // a is only live during k0..k1, then dies; b and c overlap
        // (b is read by k2 which produces c) so they must be disjoint.
        // Optimal peak = 500 (b at [0,200) + c at [200,500)) or reuse a's
        // space when it dies. Since a and c never coexist, they can share
        // the same offset. Optimal peak is max(a's slot, b + c) = 500.
        assert_eq!(plan.peak_bytes, 500);

        // Every device buffer got an offset.
        assert!(plan.offsets.iter().all(Option::is_some));
        // Peak equals max(offset + size).
        let observed = plan
            .offsets
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.map(|off| off + sizes(&g)[i]))
            .max()
            .unwrap_or(0);
        assert_eq!(observed, plan.peak_bytes);
    }

    fn sizes(g: &GraphBuilder) -> Vec<u64> {
        g.bufs
            .iter()
            .map(|b| match &b.size {
                Quast::Const(c) => *c as u64,
                _ => panic!("test buffer must have constant size"),
            })
            .collect()
    }

    /// Buffers with overlapping live ranges must not share memory.
    #[test]
    fn overlapping_lifetimes_do_not_share_memory() {
        let mut g = GraphBuilder::new();
        let a = buf(&mut g, "a", 128);
        let b = buf(&mut g, "b", 256);
        let out = buf(&mut g, "out", 128);
        // k0 produces both a and b in parallel.
        g.insert_blackbox_kernel(
            "produce_both",
            std::iter::empty(),
            [a, b].into_iter(),
            std::iter::empty(),
            |_, _| {},
        );
        // k1 consumes both and produces out.
        g.insert_blackbox_kernel(
            "consume_both",
            [a, b].into_iter(),
            [out].into_iter(),
            [false, false].into_iter(),
            |_, _| {},
        );

        let plan = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap();
        // a and b are simultaneously alive so their regions must be disjoint.
        let oa = plan.offsets[a.0].unwrap();
        let ob = plan.offsets[b.0].unwrap();
        let sa = sizes(&g)[a.0];
        let sb = sizes(&g)[b.0];
        assert!(oa + sa <= ob || ob + sb <= oa, "a and b overlap in memory");
    }

    /// Symbolic sizes are resolved via the passed assignment.
    #[test]
    fn respects_symbol_assignment() {
        let mut g = GraphBuilder::new();
        let n = g.register_symbol("n");
        let a = g.add_buf(BufInfo {
            name: Some("a".into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::sym(n).mul_c(4),
            elem_size: 4,
        });
        g.insert_blackbox_kernel(
            "k",
            std::iter::empty(),
            [a].into_iter(),
            std::iter::empty(),
            |_, _| {},
        );

        let env = BTreeMap::from([(n, 64)]);
        let plan = plan(&g, &env, DeviceType::Cuda(0)).unwrap();
        assert_eq!(plan.peak_bytes, 256);
        assert_eq!(plan.offsets[a.0], Some(0));
    }

    #[test]
    fn unbound_symbol_is_reported() {
        let mut g = GraphBuilder::new();
        let n = g.register_symbol("n");
        let _a = g.add_buf(BufInfo {
            name: Some("a".into()),
            device_type: DeviceType::Cuda(0),
            size: Quast::sym(n),
            elem_size: 4,
        });
        let err = plan(&g, &BTreeMap::new(), DeviceType::Cuda(0)).unwrap_err();
        assert!(matches!(err, PlanError::UnboundSizeSymbol { sym, .. } if sym == n));
    }
}
