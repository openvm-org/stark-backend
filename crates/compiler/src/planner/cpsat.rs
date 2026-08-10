//! CP-SAT backend: jointly solves execution order and buffer offsets as
//! one integer program via OR-Tools. Wall-time capped; accepts any
//! `Optimal`/`Feasible` solution the solver returns before the deadline.
//!
//! Requires the `planner-ortools` feature's OR-Tools install.

use cp_sat::{
    builder::{CpModelBuilder, IntVar, LinearExpr},
    proto::{CpSolverStatus, SatParameters},
};

use crate::{
    graph_ir::BufInfo,
    planner::{ctx::PlanCtx, plan::StreamMemoryPlan, PlanError},
};

pub fn plan_cpsat(
    bufs: &[BufInfo],
    ctx: &PlanCtx,
    max_secs: f64,
) -> Result<StreamMemoryPlan, PlanError> {
    let PlanCtx {
        n_nodes,
        n_bufs,
        sizes,
        writers,
        readers,
        ..
    } = ctx;
    let n_nodes = *n_nodes;
    let n_bufs = *n_bufs;

    let mut m = CpModelBuilder::default();
    let last_time = (n_nodes as i64).saturating_sub(1);

    let t: Vec<IntVar> = (0..n_nodes)
        .map(|n| m.new_int_var_with_name([(0, last_time)], format!("t_{n}")))
        .collect();
    m.add_all_different(t.iter().copied());

    let (succ, _) = ctx.edges();
    for (u, sv) in succ.iter().enumerate() {
        for &v in sv {
            m.add_lt(t[u], t[v]);
        }
    }

    let device_bufs: Vec<usize> = (0..n_bufs).filter(|&b| ctx.packable(b)).collect();

    let mut birth: std::collections::BTreeMap<usize, IntVar> = std::collections::BTreeMap::new();
    let mut death: std::collections::BTreeMap<usize, IntVar> = std::collections::BTreeMap::new();
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
        if ctx.pinned[b] || readers[b].is_empty() {
            m.add_eq(death_v, n_nodes as i64);
        } else {
            let accesses: std::collections::BTreeSet<usize> = readers[b]
                .iter()
                .chain(writers[b].iter())
                .copied()
                .collect();
            if accesses.len() == 1 {
                m.add_eq(death_v, t[*accesses.first().unwrap()]);
            } else {
                m.add_max_eq(death_v, accesses.iter().map(|&n| t[n]));
            }
        }
        birth.insert(b, birth_v);
        death.insert(b, death_v);
    }

    let sum_sizes: i64 = device_bufs.iter().map(|&b| sizes[b]).sum();

    let offsets: std::collections::BTreeMap<usize, IntVar> = device_bufs
        .iter()
        .map(|&b| {
            (
                b,
                m.new_int_var_with_name([(0, sum_sizes)], format!("off_{b}")),
            )
        })
        .collect();

    for &b in &device_bufs {
        let align = bufs[b].elem_size as i64;
        if align > 1 {
            let ub = if sum_sizes == 0 { 0 } else { sum_sizes / align };
            let k = m.new_int_var_with_name([(0, ub)], format!("align_k_{b}"));
            m.add_eq(offsets[&b], LinearExpr::from((align, k)));
        }
    }

    for (i, &b1) in device_bufs.iter().enumerate() {
        for &b2 in &device_bufs[i + 1..] {
            let lit_t12 = m.new_bool_var();
            let lit_t21 = m.new_bool_var();
            let lit_m12 = m.new_bool_var();
            let lit_m21 = m.new_bool_var();
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

    let peak = m.new_int_var_with_name([(0, sum_sizes)], "peak");
    for &b in &device_bufs {
        m.add_ge(peak, LinearExpr::from(offsets[&b]) + sizes[b]);
    }
    m.minimize(peak);

    let params = SatParameters {
        max_time_in_seconds: Some(max_secs),
        ..Default::default()
    };
    let response = m.solve_with_parameters(&params);
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

    Ok(StreamMemoryPlan::single_stream(
        order,
        out_offsets,
        peak_bytes,
        n_nodes,
    ))
}
