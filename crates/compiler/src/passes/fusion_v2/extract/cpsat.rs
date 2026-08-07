//! CP-SAT extractor implementing `detailed-fusion-plan-v2.md` §13.
//!
//! Model:
//! - `x_a in {0,1}` — alternative `a` selected;
//! - `y_v in {0,1}` — value class `v` materialized;
//! - `z_m in {0,1}` — compiled artifact `m` required.
//!
//! Constraints (§13.3):
//! - `y_v = 1` for `v in S union D` (sources and demanded outputs);
//! - `sum_{a in P(v)} x_a = y_v` for every non-source value;
//! - `x_a <= y_v` for every alternative input;
//! - `x_a <= z_m` for every artifact required by an alternative;
//! - `z_m <= sum_{a: m in K(a)} x_a` for exact OR semantics on `z_m`.
//!
//! Objective is a strict four-stage lexicographic minimization (§13.5)
//! solved sequentially: after each stage, an equality constraint pins
//! the achieved value before moving to the next.

use std::collections::HashMap;

use cp_sat::{
    builder::{BoolVar, CpModelBuilder, LinearExpr},
    proto::{CpSolverStatus, SatParameters},
};

use crate::passes::fusion_v2::{
    cost::ArtifactKey,
    extract::{ExtractOptions, ExtractionData, ExtractionSolution, FallbackReason, SolverStatus},
    model::{GraphFuser, NodeId, ValueClassId},
};

/// Runs the CP-SAT extractor. Returns [`FallbackReason::SolverStatusUnknown`]
/// (etc.) via the returned solution's `fallback` field when the solver
/// gives up; the solution's `nodes` in that case is the original seed
/// prefix so `apply_solution` still succeeds.
pub fn extract(
    gf: &GraphFuser,
    data: &ExtractionData,
    options: &ExtractOptions,
) -> ExtractionSolution {
    let n_nodes = gf.nodes.len();
    assert_eq!(data.costs.len(), n_nodes);
    assert_eq!(data.artifact_keys.len(), n_nodes);

    // Enumerate distinct artifact keys and assign each an index. Keys
    // ordering is deterministic: BTreeMap by (module_hash, target_arch,
    // compiler_flags_hash).
    let mut artifact_index: std::collections::BTreeMap<ArtifactKey, usize> =
        std::collections::BTreeMap::new();
    for k in data.artifact_keys.iter().flatten() {
        let next = artifact_index.len();
        artifact_index.entry(k.clone()).or_insert(next);
    }
    let n_artifacts = artifact_index.len();
    // For the `max_new_modules` budget: which artifacts are used by the
    // original-seed subset.
    let mut original_artifact_bit: Vec<bool> = vec![false; n_artifacts];
    for k in data.artifact_keys[..gf.seed_node_count].iter().flatten() {
        original_artifact_bit[artifact_index[k]] = true;
    }

    let mut m = CpModelBuilder::default();

    // Variables.
    let x: Vec<BoolVar> = (0..n_nodes)
        .map(|i| m.new_bool_var_with_name(format!("x_{i}")))
        .collect();
    let y: Vec<BoolVar> = (0..gf.num_values())
        .map(|i| m.new_bool_var_with_name(format!("y_{i}")))
        .collect();
    let z: Vec<BoolVar> = (0..n_artifacts)
        .map(|i| m.new_bool_var_with_name(format!("z_{i}")))
        .collect();

    // Source and demanded outputs are pinned.
    for &v in &gf.inputs {
        m.add_eq(y[v.0], 1i64);
    }
    for &v in &gf.outputs {
        m.add_eq(y[v.0], 1i64);
    }

    // Single-producer constraint: for every non-source value class,
    // `sum_{a in P(v)} x_a == y_v`.
    let inputs_set: HashMap<ValueClassId, ()> = gf.inputs.iter().map(|&v| (v, ())).collect();
    for (v, yv) in y.iter().enumerate() {
        if inputs_set.contains_key(&ValueClassId(v)) {
            continue;
        }
        // Producers of v = every alt node whose outputs contain v.
        let mut producer_expr = LinearExpr::from(0);
        for u in &gf.producers[v] {
            producer_expr += LinearExpr::from(x[u.node.0]);
        }
        m.add_eq(producer_expr, LinearExpr::from(*yv));
    }

    // Boundary-input constraint: `x_a <= y_v` for each input v of a.
    for (a, alt) in gf.nodes.iter().enumerate() {
        for &v in &alt.inputs {
            m.add_le(LinearExpr::from(x[a]), LinearExpr::from(y[v.0]));
        }
    }

    // Artifact activation: `x_a <= z_m` upward, and
    // `z_m <= sum_{a: m in K(a)} x_a` downward.
    let mut artifact_users: Vec<Vec<usize>> = vec![Vec::new(); n_artifacts];
    for (a, k) in data.artifact_keys.iter().enumerate() {
        if let Some(k) = k {
            let m_idx = artifact_index[k];
            artifact_users[m_idx].push(a);
            m.add_le(LinearExpr::from(x[a]), LinearExpr::from(z[m_idx]));
        }
    }
    for (m_idx, users) in artifact_users.iter().enumerate() {
        let mut sum = LinearExpr::from(0);
        for &a in users {
            sum += LinearExpr::from(x[a]);
        }
        m.add_le(LinearExpr::from(z[m_idx]), sum);
    }

    // Optional module budgets.
    if let Some(cap) = options.max_modules {
        let mut sum = LinearExpr::from(0);
        for zv in &z {
            sum += LinearExpr::from(*zv);
        }
        m.add_le(sum, LinearExpr::from(cap as i64));
    }
    if let Some(cap) = options.max_new_modules {
        let mut sum = LinearExpr::from(0);
        for (m_idx, zv) in z.iter().enumerate() {
            if !original_artifact_bit[m_idx] {
                sum += LinearExpr::from(*zv);
            }
        }
        m.add_le(sum, LinearExpr::from(cap as i64));
    }

    // Objective terms (each recomputed as an expression each stage).
    let runtime_expr = |m: &CpModelBuilder| -> LinearExpr {
        let _ = m;
        let mut e = LinearExpr::from(0);
        for (a, cost) in data.costs.iter().enumerate() {
            if cost.runtime_units != 0 {
                e += LinearExpr::from((cost.runtime_units, x[a]));
            }
        }
        e
    };
    let artifact_expr = |_m: &CpModelBuilder| -> LinearExpr {
        let mut e = LinearExpr::from(0);
        for zv in &z {
            e += LinearExpr::from(*zv);
        }
        e
    };
    let node_count_expr = |_m: &CpModelBuilder| -> LinearExpr {
        let mut e = LinearExpr::from(0);
        for xv in &x {
            e += LinearExpr::from(*xv);
        }
        e
    };
    let value_count_expr = |_m: &CpModelBuilder| -> LinearExpr {
        let mut e = LinearExpr::from(0);
        for yv in &y {
            e += LinearExpr::from(*yv);
        }
        e
    };

    let params = SatParameters {
        max_time_in_seconds: Some(options.solver_time_limit_secs),
        num_search_workers: Some(1),
        random_seed: Some(1),
        ..Default::default()
    };

    // Stage 1: runtime.
    m.minimize(runtime_expr(&m));
    let r1 = m.solve_with_parameters(&params);
    let status1 = solver_status(r1.status());
    if !status_is_solution(&status1) {
        return fallback_original(gf, status1);
    }
    let runtime_opt: i64 = data
        .costs
        .iter()
        .enumerate()
        .map(|(a, c)| c.runtime_units * x[a].solution_value(&r1) as i64)
        .sum();
    // Stage 1 -> Stage 2 lock. Honor runtime_tolerance_ppm as an upper
    // slack: `runtime_expr <= runtime_opt + slack`.
    let slack: i64 = if options.runtime_tolerance_ppm == 0 {
        0
    } else {
        // ceil(runtime_opt * ppm / 1_000_000)
        let n = (runtime_opt as i128).abs() * options.runtime_tolerance_ppm as i128;
        ((n + 999_999) / 1_000_000) as i64
    };
    m.add_le(runtime_expr(&m), LinearExpr::from(runtime_opt + slack));

    // Stage 2: artifact count.
    m.minimize(artifact_expr(&m));
    let r2 = m.solve_with_parameters(&params);
    let status2 = solver_status(r2.status());
    if !status_is_solution(&status2) {
        return fallback_original(gf, status2);
    }
    let artifact_opt: i64 = z.iter().map(|zv| zv.solution_value(&r2) as i64).sum();
    m.add_le(artifact_expr(&m), LinearExpr::from(artifact_opt));

    // Stage 3: node count.
    m.minimize(node_count_expr(&m));
    let r3 = m.solve_with_parameters(&params);
    let status3 = solver_status(r3.status());
    if !status_is_solution(&status3) {
        return fallback_original(gf, status3);
    }
    let node_opt: i64 = x.iter().map(|xv| xv.solution_value(&r3) as i64).sum();
    m.add_le(node_count_expr(&m), LinearExpr::from(node_opt));

    // Stage 4: value count.
    m.minimize(value_count_expr(&m));
    let r4 = m.solve_with_parameters(&params);
    let status4 = solver_status(r4.status());
    if !status_is_solution(&status4) {
        return fallback_original(gf, status4);
    }

    let selected: Vec<NodeId> = (0..n_nodes)
        .filter(|&i| x[i].solution_value(&r4))
        .map(NodeId)
        .collect();
    ExtractionSolution {
        nodes: selected,
        fallback: None,
        status: Some(status4),
    }
}

fn solver_status(s: CpSolverStatus) -> SolverStatus {
    match s {
        CpSolverStatus::Optimal => SolverStatus::Optimal,
        CpSolverStatus::Feasible => SolverStatus::Feasible,
        CpSolverStatus::Infeasible => SolverStatus::Infeasible,
        _ => SolverStatus::Unknown,
    }
}

fn status_is_solution(s: &SolverStatus) -> bool {
    matches!(s, SolverStatus::Optimal | SolverStatus::Feasible)
}

fn fallback_original(gf: &GraphFuser, status: SolverStatus) -> ExtractionSolution {
    let reason = match status {
        SolverStatus::Unknown => FallbackReason::SolverStatusUnknown,
        SolverStatus::Infeasible => FallbackReason::SolverStatusInfeasible,
        _ => FallbackReason::InternalError {
            message: format!("unexpected solver status: {status:?}"),
        },
    };
    ExtractionSolution {
        nodes: (0..gf.seed_node_count).map(NodeId).collect(),
        fallback: Some(reason),
        status: Some(status),
    }
}
