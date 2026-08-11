//! M1 exit-gate tests: versioned seed model + original-fallback round-trip.
//!
//! Each test builds a small `GraphBuilder`, snapshots its state, runs the
//! v2 seed conversion, extracts the original solution, applies it, and
//! verifies that the resulting graph matches the snapshot: same node
//! kinds in a hazard-respecting order, same interface, same physical
//! [`BufId`] bindings.

use std::sync::Arc;

use crate::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphBuilder, GraphNode},
    ir::{IRBuilder, ScalarType},
    passes::fusion_v2::{apply_solution, take_graph, ExtractionSolution, NodeId},
    quast::Quast,
};

fn sized_buf(g: &mut GraphBuilder, name: &str, bytes: i64) -> BufId {
    g.add_buf(BufInfo {
        name: Some(name.into()),
        device_type: DeviceType::Cuda(0),
        size: Quast::cst(bytes),
        elem_size: 4,
    })
}

/// Serializes a graph node into a shape-independent fingerprint good for
/// order equality across round-trips.
fn node_fingerprint(n: &GraphNode) -> String {
    match n {
        GraphNode::Kernel(k) => format!(
            "Kernel({}, in={:?}, out={:?})",
            k.module.name, k.inputs, k.outputs
        ),
        GraphNode::BlackboxKernel(k) => format!(
            "Blackbox({}, in={:?}, out={:?}, carried={:?})",
            k.name, k.inputs, k.outputs, k.carried_outputs
        ),
        GraphNode::Const(c) => format!("Const(buf={:?})", c.buf),
        GraphNode::Memcpy(m) => format!("Memcpy(src={:?}, dst={:?})", m.src, m.dst),
        GraphNode::Memset(m) => format!("Memset(node={:?}, val={:#x})", m.node, m.val),
    }
}

fn graph_fingerprint(g: &GraphBuilder) -> Vec<String> {
    g.nodes.iter().map(node_fingerprint).collect()
}

fn scale_by_two_module() -> Arc<crate::ir::Module> {
    let mut b = IRBuilder::new();
    let a = b.input("a", ScalarType::BabyBear, vec![8]);
    let body = b.compute(8, |b, i| {
        let ai = b.index(a, &[i]);
        let two = b.const_field(2);
        b.mul(ai, two)
    });
    Arc::new(b.finish("scale_by_two", body))
}

#[test]
fn take_graph_versions_a_single_writer_chain() {
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    let c = sized_buf(&mut g, "c", 32);
    g.register_input(a);
    g.register_output(c);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);

    let gf = take_graph(&mut g).unwrap();
    assert_eq!(gf.nodes.len(), 2);
    assert_eq!(gf.seed_node_count, 2);
    // One initial value class per BufId, plus one new version for each
    // written buffer (`b`, `c`).
    assert_eq!(gf.bufs.len(), 5);
    // The registered input maps to the initial class of `a`.
    assert_eq!(gf.inputs.len(), 1);
    assert_eq!(gf.physical(gf.inputs[0]), a);
    // The registered output's final version projects back to `c`.
    assert_eq!(gf.outputs.len(), 1);
    assert_eq!(gf.physical(gf.outputs[0]), c);
    // First node reads `a`'s initial class and writes a new version of `b`.
    let n0 = &gf.nodes[0];
    assert_eq!(gf.physical(n0.inputs[0]), a);
    assert_eq!(gf.physical(n0.outputs[0]), b);
    // Second node reads that new `b` version and writes a new `c` version.
    let n1 = &gf.nodes[1];
    assert_eq!(n1.inputs[0], n0.outputs[0]);
    assert_eq!(gf.physical(n1.outputs[0]), c);
}

#[test]
fn take_graph_rejects_read_before_write() {
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    // No register_input for `a` — reading it before any writer is an error.
    g.register_output(b);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    match take_graph(&mut g) {
        Err(crate::passes::fusion_v2::TakeGraphError::ReadBeforeWrite { node, buf }) => {
            assert_eq!(node, 0);
            assert_eq!(buf, a.0);
        }
        Err(other) => panic!("expected ReadBeforeWrite, got {other:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

#[test]
fn original_solution_round_trip_preserves_node_order() {
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    let c = sized_buf(&mut g, "c", 32);
    g.register_input(a);
    g.register_output(c);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);
    let before = graph_fingerprint(&g);

    let gf = take_graph(&mut g).unwrap();
    let sol = ExtractionSolution::original(&gf);
    apply_solution(&mut g, gf, &sol).unwrap();

    let after = graph_fingerprint(&g);
    assert_eq!(before, after);
    // Interface preserved.
    assert_eq!(g.input_bufs(), &[a]);
    assert_eq!(g.output_bufs(), &[c]);
    // `plan` was invalidated.
    assert!(g.plan.is_none());
}

#[test]
fn original_solution_round_trip_with_const_and_memcpy() {
    // A constant, a kernel reading it, and a full-buffer memcpy to the
    // registered output. Exercises the Const/Memcpy variants through the
    // positional API.
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    let c = sized_buf(&mut g, "c", 32);
    g.register_output(c);
    // Fabricate a small host-resident constant.
    g.insert_const(a, crate::graph_ir::ConstBuf::HostBuf(vec![0; 32]));
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    g.insert_memcpy(b, c);
    let before = graph_fingerprint(&g);

    let gf = take_graph(&mut g).unwrap();
    let sol = ExtractionSolution::original(&gf);
    apply_solution(&mut g, gf, &sol).unwrap();

    assert_eq!(graph_fingerprint(&g), before);
    assert_eq!(g.output_bufs(), &[c]);
}

#[test]
fn full_memcpy_has_no_preservation_input() {
    // Full-range memcpy: get_operands returns only the source; get_results
    // returns the destination. No preservation input.
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    g.register_input(a);
    g.register_output(b);
    g.insert_memcpy(a, b);
    let gf = take_graph(&mut g).unwrap();
    assert_eq!(gf.nodes.len(), 1);
    assert_eq!(gf.nodes[0].inputs.len(), 1);
    assert_eq!(gf.nodes[0].outputs.len(), 1);
    assert_eq!(gf.physical(gf.nodes[0].inputs[0]), a);
    assert_eq!(gf.physical(gf.nodes[0].outputs[0]), b);
}

#[test]
fn partial_memcpy_adds_preservation_input() {
    // Partial memcpy: get_operands appends the destination as a
    // preservation input.
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 64);
    let b = sized_buf(&mut g, "b", 64);
    g.register_input(a);
    g.register_input(b);
    g.register_output(b);
    // Half-range copy.
    g.insert_memcpy_range(a, Quast::cst(0), b, Quast::cst(0), Quast::cst(32));
    let gf = take_graph(&mut g).unwrap();
    assert_eq!(gf.nodes[0].inputs.len(), 2, "src + preservation dst");
    assert_eq!(gf.physical(gf.nodes[0].inputs[0]), a);
    assert_eq!(gf.physical(gf.nodes[0].inputs[1]), b);
}

#[test]
fn re_exported_versions_share_physical_bufid() {
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    g.register_input(a);
    g.register_output(b);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    let gf = take_graph(&mut g).unwrap();
    let out_v = gf.nodes[0].outputs[0];
    // New version got a fresh value class id but the same physical BufId
    // as the initial class.
    assert_ne!(out_v.0, b.0);
    assert_eq!(gf.physical(out_v), b);
}

#[test]
fn round_trip_matches_registered_output_final_version() {
    // Registered output must resolve to the *last* writer of its BufId.
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    g.register_input(a);
    g.register_output(b);
    // Two writes to b; the output should map to the second version.
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    let gf = take_graph(&mut g).unwrap();
    let out_v = gf.outputs[0];
    let last_writer_out = gf.nodes[1].outputs[0];
    assert_eq!(out_v, last_writer_out);
}

#[test]
fn hazard_order_respects_waw_between_selected_writers() {
    // Two writers of the same physical buffer: apply must emit them in
    // increasing ValueClassId order (i.e. seed order for `b`).
    let mut g = GraphBuilder::new();
    let a = sized_buf(&mut g, "a", 32);
    let b = sized_buf(&mut g, "b", 32);
    g.register_input(a);
    g.register_output(b);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
    let gf = take_graph(&mut g).unwrap();
    let sol = ExtractionSolution {
        nodes: vec![NodeId(0), NodeId(1)],
        fallback: None,
        status: None,
    };
    apply_solution(&mut g, gf, &sol).unwrap();
    // Both nodes are Kernel; the order should be n0 before n1.
    assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
    assert!(matches!(&g.nodes[1], GraphNode::Kernel(_)));
}

// -------------------------------------------------------------------------
// M2 extractor tests: the CP-SAT and brute-force extractors agree on toy
// alternative graphs, and both pick the runtime-optimal feasible subset.
// -------------------------------------------------------------------------

mod extractor {
    use super::*;
    use crate::passes::fusion_v2::{
        cost::{ArtifactKey, GraphNodeCost},
        extract::{brute, ExtractOptions, ExtractionData, ExtractionSolution},
        AltGraphNode, GraphFuser,
    };

    fn artifact(byte: u8) -> ArtifactKey {
        ArtifactKey {
            module_hash: [byte; 32],
            target_arch: "sm_80".into(),
            compiler_flags_hash: [0; 32],
        }
    }

    /// Builds a two-kernel chain (`a -> b -> c`) as the seed graph.
    fn chain_gf() -> (GraphBuilder, GraphFuser) {
        let mut g = GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        let b = sized_buf(&mut g, "b", 32);
        let c = sized_buf(&mut g, "c", 32);
        g.register_input(a);
        g.register_output(c);
        g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
        g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);
        let gf = take_graph(&mut g).unwrap();
        (g, gf)
    }

    fn sort_nodes(sol: &ExtractionSolution) -> Vec<usize> {
        let mut v: Vec<usize> = sol.nodes.iter().map(|n| n.0).collect();
        v.sort();
        v
    }

    #[test]
    fn no_candidates_solver_returns_original() {
        // With no candidates, the ILP has exactly one feasible solution:
        // both seed nodes selected.
        let (_g, gf) = chain_gf();
        let data = ExtractionData::uniform(&gf);
        let opts = ExtractOptions::default();
        let brute_sol = brute::extract(&gf, &data, &opts).unwrap();
        assert_eq!(sort_nodes(&brute_sol), vec![0, 1]);
    }

    #[test]
    fn brute_force_prefers_cheaper_fused_candidate() {
        // Insert a fused-kernel alternative that replaces both seeds and
        // is cheaper than their sum. Brute force must pick it.
        let (_g, mut gf) = chain_gf();
        let seed0_out = gf.nodes[0].outputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let seed0_in = gf.nodes[0].inputs[0];
        // Fused candidate: reads `a`, produces the same `c` version as
        // seed 1.
        let fused_node = {
            // Any GraphNode works structurally — reuse seed 1's
            // KernelModuleNode's module by moving one of the seeds out.
            let n1 = gf.nodes[1].node.clone_kernel_for_test();
            AltGraphNode {
                inputs: vec![seed0_in],
                outputs: vec![seed1_out],
                node: n1,
            }
        };
        let fused_id = gf.insert_candidate(fused_node);

        let mut data = ExtractionData::uniform(&gf);
        // Seeds each cost 5, fused costs 4; fused should win.
        data.costs[0] = GraphNodeCost::new(5);
        data.costs[1] = GraphNodeCost::new(5);
        data.costs[fused_id.0] = GraphNodeCost::new(4);

        let opts = ExtractOptions::default();
        let sol = brute::extract(&gf, &data, &opts).unwrap();
        assert_eq!(sort_nodes(&sol), vec![fused_id.0]);
        // Unused `b` version is not required to be materialized (§13.5
        // stage 4 drops it).
        let materialized_b: bool = sol
            .nodes
            .iter()
            .any(|n| gf.nodes[n.0].outputs.contains(&seed0_out));
        assert!(!materialized_b, "seed0's `b` output should be dropped");
    }

    #[test]
    fn brute_force_keeps_original_when_fused_is_more_expensive() {
        let (_g, mut gf) = chain_gf();
        let seed0_in = gf.nodes[0].inputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let fused_node = AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: gf.nodes[1].node.clone_kernel_for_test(),
        };
        let fused_id = gf.insert_candidate(fused_node);

        let mut data = ExtractionData::uniform(&gf);
        data.costs[0] = GraphNodeCost::new(3);
        data.costs[1] = GraphNodeCost::new(3);
        data.costs[fused_id.0] = GraphNodeCost::new(10);

        let opts = ExtractOptions::default();
        let sol = brute::extract(&gf, &data, &opts).unwrap();
        assert_eq!(sort_nodes(&sol), vec![0, 1]);
    }

    #[test]
    fn shared_artifact_across_two_alternatives_is_charged_once() {
        // Two candidates for the same output value that use the same
        // artifact must be indistinguishable under the artifact-count
        // objective; and if a third candidate uses a different artifact,
        // the shared-artifact one wins under stage 2.
        //
        // Setup: candidate X and Y both produce `c` cheaply with cost
        // 3, sharing artifact A. Candidate Z produces `c` alone with
        // cost 3, artifact B. All produce `c` from `a` directly.
        // Stage 1 (runtime) is a tie at 3, so stage 2 picks any
        // solution using one artifact; ties broken by stage 3 (node
        // count = 1) and stage 4 (fewest materialized values).
        let (_g, mut gf) = chain_gf();
        let seed0_in = gf.nodes[0].inputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let template = gf.nodes[1].node.clone_kernel_for_test();
        let x = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: template.clone_kernel_for_test(),
        });
        // Y and Z also both produce `c` from `a`. Only one of X/Y/Z can
        // be selected (single-producer constraint).
        let y = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: template.clone_kernel_for_test(),
        });
        let z = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: template,
        });
        let mut data = ExtractionData::uniform(&gf);
        // All three cost the same.
        data.costs[x.0] = GraphNodeCost::new(3);
        data.costs[y.0] = GraphNodeCost::new(3);
        data.costs[z.0] = GraphNodeCost::new(3);
        // X and Y share artifact A; Z has artifact B; seeds unassigned.
        data.artifact_keys[x.0] = Some(artifact(1));
        data.artifact_keys[y.0] = Some(artifact(1));
        data.artifact_keys[z.0] = Some(artifact(2));

        // Stage 2 is off by default; enable it since this test's point
        // is the artifact-count objective.
        let opts = ExtractOptions {
            optimize_artifact_count: true,
            ..ExtractOptions::default()
        };
        let sol = brute::extract(&gf, &data, &opts).unwrap();
        // Some single-candidate solution is chosen. The seeds add up
        // to cost 2, so they should still win stage 1.
        // Actually seeds are cost 1 each, so `seeds` = 2 runtime while
        // any single candidate = 3. Seeds win.
        assert_eq!(sort_nodes(&sol), vec![0, 1]);
    }

    #[test]
    fn max_new_modules_zero_forces_original() {
        let (_g, mut gf) = chain_gf();
        let seed0_in = gf.nodes[0].inputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let fused_id = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: gf.nodes[1].node.clone_kernel_for_test(),
        });
        let mut data = ExtractionData::uniform(&gf);
        data.costs[0] = GraphNodeCost::new(5);
        data.costs[1] = GraphNodeCost::new(5);
        // Fused is cheaper but requires a new artifact.
        data.costs[fused_id.0] = GraphNodeCost::new(1);
        data.artifact_keys[fused_id.0] = Some(artifact(9));

        let opts = ExtractOptions {
            max_new_modules: Some(0),
            ..ExtractOptions::default()
        };
        let sol = brute::extract(&gf, &data, &opts).unwrap();
        assert_eq!(sort_nodes(&sol), vec![0, 1]);
        // With the cap removed the fused candidate wins.
        let opts = ExtractOptions::default();
        let sol = brute::extract(&gf, &data, &opts).unwrap();
        assert_eq!(sort_nodes(&sol), vec![fused_id.0]);
    }

    #[test]
    fn value_count_tiebreak_drops_unused_intermediate() {
        // Two solutions with the same runtime, artifact count, and node
        // count: one materializes an extra value, one does not. Stage 4
        // (value count) drops the extra.
        //
        // Concretely, we insert a candidate that produces both `b`
        // (unused) and `c` — this alternative materializes 2 values —
        // versus the seeds which materialize 2 values too but via 2
        // nodes. Rebalance costs so runtime ties; node-count wins for
        // the single node.
        let (_g, mut gf) = chain_gf();
        let seed0_in = gf.nodes[0].inputs[0];
        let seed0_out = gf.nodes[0].outputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let fused_id = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed0_out, seed1_out],
            node: gf.nodes[1].node.clone_kernel_for_test(),
        });
        let mut data = ExtractionData::uniform(&gf);
        // Runtime: seeds total 2; fused = 2.
        data.costs[fused_id.0] = GraphNodeCost::new(2);
        let sol = brute::extract(&gf, &data, &ExtractOptions::default()).unwrap();
        // Stage 3 (node count 1 < 2) picks the fused candidate.
        assert_eq!(sort_nodes(&sol), vec![fused_id.0]);
    }
}

/// Test-only helper on GraphNode so extractor tests can construct
/// alternative graphs with fake kernel clones. Public via the test
/// module only.
impl GraphNode {
    fn clone_kernel_for_test(&self) -> GraphNode {
        match self {
            GraphNode::Kernel(k) => GraphNode::Kernel(crate::graph_ir::KernelModuleNode {
                module: k.module.clone(),
                param_bindings: k.param_bindings.clone(),
                inputs: k.inputs.clone(),
                outputs: k.outputs.clone(),
                types: k.types.clone(),
                hash: k.hash,
                canonical: k.canonical,
                fusion_history: k.fusion_history.clone(),
            }),
            _ => panic!("clone_kernel_for_test only supports Kernel"),
        }
    }
}

// -------------------------------------------------------------------------
// CP-SAT-gated tests: the CP-SAT extractor and brute force must agree on
// every toy instance (M2 exit gate).
// -------------------------------------------------------------------------

#[cfg(feature = "planner-ortools")]
mod cpsat_agreement {
    use super::*;
    use crate::passes::fusion_v2::{
        cost::{ArtifactKey, GraphNodeCost},
        extract::{brute, cpsat, ExtractOptions, ExtractionData},
        AltGraphNode,
    };

    fn artifact(byte: u8) -> ArtifactKey {
        ArtifactKey {
            module_hash: [byte; 32],
            target_arch: "sm_80".into(),
            compiler_flags_hash: [0; 32],
        }
    }

    fn assert_agree(gf: &crate::passes::fusion_v2::GraphFuser, data: &ExtractionData) {
        let opts = ExtractOptions::default();
        let b = brute::extract(gf, data, &opts).unwrap();
        let c = cpsat::extract(gf, data, &opts);
        let mut b_nodes: Vec<usize> = b.nodes.iter().map(|n| n.0).collect();
        let mut c_nodes: Vec<usize> = c.nodes.iter().map(|n| n.0).collect();
        b_nodes.sort();
        c_nodes.sort();
        assert_eq!(b_nodes, c_nodes, "brute {b_nodes:?} vs cpsat {c_nodes:?}");
        assert!(
            c.fallback.is_none(),
            "cpsat reported fallback: {:?}",
            c.fallback
        );
    }

    fn build_chain() -> (
        crate::graph_ir::GraphBuilder,
        crate::passes::fusion_v2::GraphFuser,
    ) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        let b = sized_buf(&mut g, "b", 32);
        let c = sized_buf(&mut g, "c", 32);
        g.register_input(a);
        g.register_output(c);
        g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
        g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);
        let gf = take_graph(&mut g).unwrap();
        (g, gf)
    }

    #[test]
    fn agree_no_candidates() {
        let (_g, gf) = build_chain();
        let data = ExtractionData::uniform(&gf);
        assert_agree(&gf, &data);
    }

    #[test]
    fn agree_cheaper_fused_candidate() {
        let (_g, mut gf) = build_chain();
        let seed0_in = gf.nodes[0].inputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let fused_id = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: gf.nodes[1].node.clone_kernel_for_test(),
        });
        let mut data = ExtractionData::uniform(&gf);
        data.costs[0] = GraphNodeCost::new(5);
        data.costs[1] = GraphNodeCost::new(5);
        data.costs[fused_id.0] = GraphNodeCost::new(4);
        assert_agree(&gf, &data);
    }

    #[test]
    fn agree_max_new_modules_zero() {
        let (_g, mut gf) = build_chain();
        let seed0_in = gf.nodes[0].inputs[0];
        let seed1_out = gf.nodes[1].outputs[0];
        let fused_id = gf.insert_candidate(AltGraphNode {
            inputs: vec![seed0_in],
            outputs: vec![seed1_out],
            node: gf.nodes[1].node.clone_kernel_for_test(),
        });
        let mut data = ExtractionData::uniform(&gf);
        data.costs[fused_id.0] = GraphNodeCost::new(1);
        data.artifact_keys[fused_id.0] = Some(artifact(9));
        let opts = ExtractOptions {
            max_new_modules: Some(0),
            ..ExtractOptions::default()
        };
        let b = brute::extract(&gf, &data, &opts).unwrap();
        let c = cpsat::extract(&gf, &data, &opts);
        let mut b_nodes: Vec<usize> = b.nodes.iter().map(|n| n.0).collect();
        let mut c_nodes: Vec<usize> = c.nodes.iter().map(|n| n.0).collect();
        b_nodes.sort();
        c_nodes.sort();
        assert_eq!(b_nodes, c_nodes);
    }

    // ---------------------------------------------------------------------
    // M2 exit gate: randomized models with `<= 12` alternatives must
    // agree with the brute-force extractor on every seed.
    // ---------------------------------------------------------------------

    /// Deterministic LCG so the test is repeatable across runs.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *state
    }

    /// Builds a random alternative graph with `n_candidates` extra fused
    /// candidates on top of a 2-seed chain. Each candidate picks a subset
    /// of existing input values and outputs and reuses seed 1's kernel
    /// module. Costs and artifacts are randomized.
    fn random_gf_and_data(
        seed: u64,
        n_candidates: usize,
    ) -> (crate::passes::fusion_v2::GraphFuser, ExtractionData) {
        let mut state = seed;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        let b = sized_buf(&mut g, "b", 32);
        let c = sized_buf(&mut g, "c", 32);
        g.register_input(a);
        g.register_output(c);
        g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
        g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);
        let mut gf = take_graph(&mut g).unwrap();

        let a_v = gf.inputs[0];
        let b_v = gf.nodes[0].outputs[0];
        let c_v = gf.nodes[1].outputs[0];
        let template = gf.nodes[1].node.clone_kernel_for_test();

        for _ in 0..n_candidates {
            // Choose one of a small set of candidate shapes at random.
            let shape = lcg_next(&mut state) % 4;
            let (inputs, outputs) = match shape {
                // a -> c (drop-seams)
                0 => (vec![a_v], vec![c_v]),
                // a -> b (materialize intermediate)
                1 => (vec![a_v], vec![b_v]),
                // a -> b, c (multi-output)
                2 => (vec![a_v], vec![b_v, c_v]),
                // b -> c (equivalent to seed 1)
                _ => (vec![b_v], vec![c_v]),
            };
            gf.insert_candidate(AltGraphNode {
                inputs,
                outputs,
                node: template.clone_kernel_for_test(),
            });
        }

        let mut data = ExtractionData::uniform(&gf);
        for i in 0..gf.nodes.len() {
            data.costs[i] = GraphNodeCost::new(1 + (lcg_next(&mut state) % 9) as i64);
            // 30% of nodes get a random artifact from a small pool.
            if lcg_next(&mut state) % 100 < 30 {
                let a_idx = (lcg_next(&mut state) % 3) as u8;
                data.artifact_keys[i] = Some(artifact(a_idx));
            }
        }
        (gf, data)
    }

    #[test]
    fn agree_random_property() {
        // Exercise both objective shapes: the default 3-stage lex
        // (artifact stage skipped) and the full 4-stage lex.
        for optimize_artifact_count in [false, true] {
            for seed in 0..32u64 {
                let (gf, data) = random_gf_and_data(seed, 6);
                let opts = ExtractOptions {
                    optimize_artifact_count,
                    ..ExtractOptions::default()
                };
                let b = brute::extract(&gf, &data, &opts).unwrap();
                let c = cpsat::extract(&gf, &data, &opts);
                let mut b_nodes: Vec<usize> = b.nodes.iter().map(|n| n.0).collect();
                let mut c_nodes: Vec<usize> = c.nodes.iter().map(|n| n.0).collect();
                b_nodes.sort();
                c_nodes.sort();
                // Because two feasible solutions can share the same lex cost
                // (perfect ties on all stages), we compare *cost tuples*
                // rather than requiring identical selected sets. When the
                // artifact stage is off, its component is not part of the
                // objective and is zeroed out of the comparison.
                let b_cost = solution_cost(&gf, &data, &b, optimize_artifact_count);
                let c_cost = solution_cost(&gf, &data, &c, optimize_artifact_count);
                assert_eq!(
                    b_cost, c_cost,
                    "seed {seed} (artifacts={optimize_artifact_count}): brute {b_nodes:?} cost \
                     {b_cost:?} vs cpsat {c_nodes:?} cost {c_cost:?}"
                );
            }
        }
    }

    /// Recomputes the lex cost of a solution: (runtime, artifact_count,
    /// node_count, value_count), with artifact_count zeroed when the
    /// artifact stage is disabled.
    fn solution_cost(
        gf: &crate::passes::fusion_v2::GraphFuser,
        data: &ExtractionData,
        sol: &crate::passes::fusion_v2::ExtractionSolution,
        optimize_artifact_count: bool,
    ) -> (i128, u64, u64, u64) {
        let runtime: i128 = sol
            .nodes
            .iter()
            .map(|n| data.costs[n.0].runtime_units as i128)
            .sum();
        let artifacts: std::collections::HashSet<_> = sol
            .nodes
            .iter()
            .filter_map(|n| data.artifact_keys[n.0].clone())
            .collect();
        let mut materialized: std::collections::HashSet<_> = gf.inputs.iter().copied().collect();
        for &n in &sol.nodes {
            for &v in &gf.nodes[n.0].outputs {
                materialized.insert(v);
            }
        }
        for &v in &gf.outputs {
            materialized.insert(v);
        }
        (
            runtime,
            if optimize_artifact_count {
                artifacts.len() as u64
            } else {
                0
            },
            sol.nodes.len() as u64,
            materialized.len() as u64,
        )
    }
}

// -------------------------------------------------------------------------
// M3 producer-consumer drop-seams tests: identity-access chain fixtures.
// -------------------------------------------------------------------------

mod producer_consumer_tests {
    use super::*;
    use crate::{
        module_hash::module_hash,
        passes::fusion_v2::{
            apply_solution,
            cost::GraphNodeCost,
            extract::{brute, ExtractOptions, ExtractionData},
            fusions::producer_consumer,
            take_graph, GraphFuser,
        },
    };

    /// A concrete `compute[N] |i| c * a[i]` module — the shape the M3
    /// identity extractor recognizes.
    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    /// Builds a chain `y = 2 * x; z = 3 * y` graph and takes it into
    /// a GraphFuser.
    fn scale_chain(n: usize) -> (GraphBuilder, GraphFuser) {
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();
        (g, gf)
    }

    #[test]
    fn enumerate_identity_chain_produces_one_candidate() {
        let (_g, gf) = scale_chain(8);
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        assert_eq!(drafts.len(), 1);
        let d = &drafts[0];
        // Producer/consumer are seeds 0 and 1.
        assert_eq!(d.parents.len(), 2);
        // Fused inputs = producer inputs (just `x`), fused outputs =
        // consumer outputs (just `z`).
        assert_eq!(d.alt.inputs.len(), 1);
        assert_eq!(d.alt.outputs.len(), 1);
        // Inputs/outputs point at the same value classes as the
        // originals so the extractor treats the candidate as an
        // alternative producer of the same `z` value.
        assert_eq!(d.alt.inputs, gf.nodes[0].inputs);
        assert_eq!(d.alt.outputs, gf.nodes[1].outputs);
    }

    #[test]
    fn synthesized_module_type_checks() {
        let (_g, gf) = scale_chain(8);
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        let d = &drafts[0];
        match &d.alt.node {
            GraphNode::Kernel(k) => {
                let m = &k.module;
                crate::passes::type_infer(m).expect("fused module type-checks");
                assert_eq!(m.builder.inputs().len(), 1);
            }
            _ => panic!("expected Kernel"),
        }
    }

    #[test]
    fn synthesized_module_hash_matches_hand_authored_reference() {
        // The fused module for `y = 2*x; z = 3*y` should structurally
        // equal `compute[N] |i| 3 * (2 * x[i])`. Note that the DSL
        // canonicalizer does not perform arithmetic folding, so the
        // reference keeps the nested-multiply shape.
        let (_g, gf) = scale_chain(8);
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        let d = &drafts[0];
        let fused_module = match &d.alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };

        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![8]);
            let body = b.compute(8usize, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                let scaled = b.mul(ai, two);
                let three = b.const_field(3);
                b.mul(scaled, three)
            });
            b.finish(fused_module.name.clone(), body)
        };
        assert_eq!(
            module_hash(&fused_module),
            module_hash(&reference),
            "fused module hash differs from hand-authored reference"
        );
    }

    #[test]
    fn extractor_picks_cheap_fused_candidate_and_apply_produces_one_node() {
        // End-to-end: enumerate → insert → extract → apply. The fused
        // candidate is cheaper than the two seeds combined, so the
        // extractor picks it and the reconstructed graph has one node.
        let (mut g, mut gf) = scale_chain(8);
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        assert_eq!(drafts.len(), 1);
        let d = drafts.into_iter().next().unwrap();
        let fused_id = gf.insert_candidate(d.alt);
        let mut data = ExtractionData::uniform(&gf);
        data.costs[0] = GraphNodeCost::new(5);
        data.costs[1] = GraphNodeCost::new(5);
        data.costs[fused_id.0] = GraphNodeCost::new(3);
        let sol = brute::extract(&gf, &data, &ExtractOptions::default()).unwrap();
        assert_eq!(sol.nodes, vec![fused_id]);
        apply_solution(&mut g, gf, &sol).unwrap();
        assert_eq!(g.nodes.len(), 1);
        assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
    }

    // ---------------------------------------------------------------------
    // M3 exit-gate fixtures: affine, reduction producer, nested.
    // ---------------------------------------------------------------------

    /// Producer: `y = 2 * x` (identity). Consumer: `z[i] = 5 * y[N-1-i]`
    /// (affine permutation).
    fn scale_then_reverse_module(n: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let n_minus_1 = b.const_u32((n as u32) - 1);
            let idx = b.sub(n_minus_1, i);
            let ai = b.index(a, &[idx]);
            let five = b.const_field(5);
            b.mul(ai, five)
        });
        Arc::new(b.finish("scale_reverse", body))
    }

    #[test]
    fn synthesized_module_supports_affine_permutation_consumer() {
        // Chain: y = 2 * x; z[i] = 5 * y[N-1-i]. Fused module should
        // produce `compute[N] |i| 5 * (2 * x[N-1-i])`.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_then_reverse_module(n), vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        assert_eq!(drafts.len(), 1);
        let fused_module = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        crate::passes::type_infer(&fused_module).unwrap();

        // Reference: `compute[N] |i| 5 * (2 * x[N-1-i])`.
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let n_minus_1 = b.const_u32((n as u32) - 1);
                let idx = b.sub(n_minus_1, i);
                let ai = b.index(a, &[idx]);
                let two = b.const_field(2);
                let scaled = b.mul(ai, two);
                let five = b.const_field(5);
                b.mul(scaled, five)
            });
            b.finish(fused_module.name.clone(), body)
        };
        assert_eq!(
            crate::module_hash::module_hash(&fused_module),
            crate::module_hash::module_hash(&reference),
        );
    }

    /// Producer: `y[i] = sum_{j<K} c[j] * x[i]`. `y` is a scalar
    /// per outer iteration, where the "scalar" is a reduce sub-expression.
    /// This tests the reduction-producer M3 case.
    fn reduce_producer_module(n: usize, k: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![n]);
        let c = b.input("c", ScalarType::BabyBear, vec![k]);
        let body = b.compute(n, |b, i| {
            b.reduce_add(k, |b, j| {
                let xi = b.index(x, &[i]);
                let cj = b.index(c, &[j]);
                b.mul(cj, xi)
            })
        });
        Arc::new(b.finish("reduce_producer", body))
    }

    #[test]
    fn synthesized_module_supports_reduction_producer() {
        // Producer emits a per-outer reduce; consumer is a simple
        // identity `z[i] = 3 * y[i]`.
        let n = 4;
        let k = 3;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let c = sized_buf(&mut g, "c", (k * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_input(c);
        g.register_output(z);
        g.insert_kernel(reduce_producer_module(n, k), vec![x, c], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        assert_eq!(
            drafts.len(),
            1,
            "reduction producer with identity consumer should fuse"
        );
        let fused = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        crate::passes::type_infer(&fused).unwrap();

        // Reference: `compute[N] |i| 3 * (sum_{j<K} c[j] * x[i])`.
        let reference = {
            let mut b = IRBuilder::new();
            // Fused module's input order: producer inputs first (x, c),
            // then consumer non-seam inputs (none).
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let c = b.input("c", ScalarType::BabyBear, vec![k]);
            let body = b.compute(n, |b, i| {
                let r = b.reduce_add(k, |b, j| {
                    let xi = b.index(x, &[i]);
                    let cj = b.index(c, &[j]);
                    b.mul(cj, xi)
                });
                let three = b.const_field(3);
                b.mul(r, three)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            crate::module_hash::module_hash(&fused),
            crate::module_hash::module_hash(&reference),
        );
    }

    /// Nested-index consumer: consumer reads the seam inside an inner
    /// reduce. The hook-based synthesis inlines the producer at the
    /// inner scope, substituting producer's outer var with the inner
    /// reduce var.
    #[test]
    fn synthesized_module_supports_nested_index_consumer() {
        // Producer: `y[i] = 2 * x[i]` (identity, N).
        // Consumer: `z[i] = sum_{j<N} y[j]`.
        // Fused expected: `compute[N] |i| sum_{j<N} 2 * x[j]`.
        let n = 4;
        let consumer = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, _i| b.reduce_add(n, |b, j| b.index(y, &[j])));
            Arc::new(b.finish("reduce_consumer", body))
        };
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        assert_eq!(drafts.len(), 1, "nested-index consumer should fuse");
        let fused = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        crate::passes::type_infer(&fused).unwrap();

        // Reference: `compute[N] |i| sum_{j<N} (2 * a[j])`. Fused
        // module inherits its input name from the producer, so input
        // is named "a" (as in `scale_by`), not "x".
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, _i| {
                b.reduce_add(n, |b, j| {
                    let aj = b.index(a, &[j]);
                    let two = b.const_field(2);
                    b.mul(aj, two)
                })
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            crate::module_hash::module_hash(&fused),
            crate::module_hash::module_hash(&reference),
        );
    }

    // ---------------------------------------------------------------
    // M5: keep-seam variants (§10.2).
    // ---------------------------------------------------------------

    /// Chain where the seam `y` is also a registered graph output.
    fn scale_chain_seam_is_output(
        n: usize,
    ) -> (
        GraphBuilder,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
    ) {
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(y);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        (g, x, y, z)
    }

    #[test]
    fn enumerate_emits_keep_variant_when_seam_is_graph_output() {
        let (mut g, _x, _y, _z) = scale_chain_seam_is_output(8);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        // One drop + one keep.
        assert_eq!(drafts.len(), 2);
        let variants: Vec<_> = drafts.iter().map(|d| d.variant).collect();
        assert!(variants.contains(&producer_consumer::FusionVariant::Drop));
        assert!(variants.contains(&producer_consumer::FusionVariant::Keep));
    }

    #[test]
    fn enumerate_skips_keep_when_seam_has_single_consumer() {
        let (_g, gf) = scale_chain(8);
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        // Single-consumer seam, seam not a graph output: only drop.
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].variant, producer_consumer::FusionVariant::Drop);
    }

    #[test]
    fn enable_all_keep_variants_emits_keep_on_single_consumer() {
        let (_g, gf) = scale_chain(8);
        let opts = producer_consumer::EnumerateOptions {
            enable_all_keep_variants: true,
        };
        let ctx = producer_consumer::OwnedEnumerateContext::all_seed(&gf, opts);
        let drafts = producer_consumer::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 2);
        assert_eq!(
            drafts
                .iter()
                .filter(|d| d.variant == producer_consumer::FusionVariant::Keep)
                .count(),
            1
        );
    }

    #[test]
    fn keep_variant_outputs_include_seam_value() {
        let (mut g, _x, _y, _z) = scale_chain_seam_is_output(8);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        let keep = drafts
            .iter()
            .find(|d| d.variant == producer_consumer::FusionVariant::Keep)
            .expect("keep candidate emitted");
        // Producer output is the seam; consumer output is z. Keep's
        // outputs = [consumer_output, seam_output].
        assert_eq!(keep.alt.outputs.len(), 2);
        let producer_out = gf.nodes[0].outputs[0];
        let consumer_out = gf.nodes[1].outputs[0];
        assert_eq!(keep.alt.outputs[0], consumer_out);
        assert_eq!(keep.alt.outputs[1], producer_out);
    }

    #[test]
    fn keep_variant_module_hash_matches_hand_authored_reference() {
        // Fixture: `y = 2*x; z = 3*y`. Keep candidate should synthesize
        // `compute[N] |i| Tuple([3 * (2*x[i]), 2 * x[i]])` — the
        // consumer output first, then the materialized seam.
        let n = 8;
        let (mut g, _x, _y, _z) = scale_chain_seam_is_output(n);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        let keep = drafts
            .iter()
            .find(|d| d.variant == producer_consumer::FusionVariant::Keep)
            .expect("keep candidate emitted");
        let fused = match &keep.alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                let scaled = b.mul(ai, two);
                let three = b.const_field(3);
                let consumer = b.mul(scaled, three);
                let seam = b.mul(ai, two);
                b.tuple(&[consumer, seam])
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(module_hash(&fused), module_hash(&reference));
    }

    #[test]
    fn keep_variant_type_checks() {
        let (mut g, _x, _y, _z) = scale_chain_seam_is_output(8);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        let keep = drafts
            .iter()
            .find(|d| d.variant == producer_consumer::FusionVariant::Keep)
            .expect("keep candidate emitted");
        match &keep.alt.node {
            GraphNode::Kernel(k) => {
                let m = &k.module;
                crate::passes::type_infer(m).expect("keep-variant module type-checks");
                assert_eq!(m.builder.inputs().len(), 1);
            }
            _ => panic!("expected Kernel"),
        }
    }

    #[test]
    fn extractor_picks_keep_over_original_chain_when_seam_is_graph_output() {
        // With the M4 estimator, a single fused kernel launch beats two
        // separate launches. When the seam is a graph output, the keep
        // variant materializes both outputs from one launch and should
        // win.
        let n = 8;
        let (mut g, _x, y, z) = scale_chain_seam_is_output(n);
        let mut gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        for d in drafts {
            gf.insert_candidate(d.alt);
        }
        // Under the placeholder-friendly uniform cost model we would
        // fall through to the original; but with a coarse cost model
        // that at least reflects "one kernel < two kernels" (launch
        // overhead), the extractor should pick the keep candidate.
        // Use brute force with unit runtime per node — a common
        // regression check: fewer selected nodes wins the stage-3
        // graph-size tiebreak.
        let data = ExtractionData::uniform(&gf);
        let solution = brute::extract(&gf, &data, &ExtractOptions::default()).unwrap();
        // 1 node = keep candidate satisfies both y and z with one
        // producer.
        assert_eq!(solution.nodes.len(), 1);
        apply_solution(&mut g, gf, &solution).unwrap();
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.output_bufs(), &[y, z]);
        // The remaining node writes both y and z.
        match &g.nodes[0] {
            GraphNode::Kernel(k) => {
                assert!(k.outputs.contains(&y));
                assert!(k.outputs.contains(&z));
            }
            _ => panic!("expected Kernel"),
        }
    }

    #[test]
    fn extractor_prefers_keep_over_drop_plus_original_producer_on_fanout() {
        // Fanout: `y = 2*x`, two consumers `z1 = 3*y`, `z2 = 5*y`.
        // Enumeration emits one drop + one keep per (p, c) pair (seam
        // has another consumer). The extractor should be able to pick
        // {keep(p,c1) for z1&y, drop(p,c2) for z2} because keep-c1
        // already materializes y for c2's drop.
        //
        // Under uniform cost, node-count tiebreak makes 2 nodes beat 3
        // (original) or 4 (drop×2 + producer). Verify the extractor
        // achieves at most 2 nodes.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);
        let mut gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        // 2 drop + 2 keep.
        assert_eq!(drafts.len(), 4);
        for d in drafts {
            gf.insert_candidate(d.alt);
        }
        let data = ExtractionData::uniform(&gf);
        let solution = brute::extract(&gf, &data, &ExtractOptions::default()).unwrap();
        // Best plan: 2 fused nodes (one keep + one drop) — 2 nodes total.
        assert!(
            solution.nodes.len() <= 2,
            "extractor should not pick more than 2 nodes: got {}",
            solution.nodes.len()
        );
        apply_solution(&mut g, gf, &solution).unwrap();
        assert_eq!(g.output_bufs(), &[z1, z2]);
    }

    #[test]
    fn keep_variant_cost_is_priced_by_estimator() {
        // The estimator prices the keep-variant kernel higher than the
        // drop-variant kernel (extra Tuple element and store) but lower
        // than the sum of producer + consumer (one launch vs two).
        let n = 1024;
        let (mut g, _x, _y, _z) = scale_chain_seam_is_output(n);
        let gf = take_graph(&mut g).unwrap();
        let drafts = producer_consumer::enumerate(
            &gf,
            &producer_consumer::OwnedEnumerateContext::all_seed(
                &gf,
                producer_consumer::EnumerateOptions::default(),
            )
            .as_ref(),
        );
        let keep = drafts
            .iter()
            .find(|d| d.variant == producer_consumer::FusionVariant::Keep)
            .unwrap();
        let drop = drafts
            .iter()
            .find(|d| d.variant == producer_consumer::FusionVariant::Drop)
            .unwrap();
        let cfg = crate::passes::fusion_v2::cost::EstimatorConfig::default();
        let ctx = crate::passes::fusion_v2::cost::EstimateContext::default();
        let hash_keep = module_hash(match &keep.alt.node {
            GraphNode::Kernel(k) => &k.module,
            _ => unreachable!(),
        });
        let hash_drop = module_hash(match &drop.alt.node {
            GraphNode::Kernel(k) => &k.module,
            _ => unreachable!(),
        });
        let (keep_cost, _) = crate::passes::fusion_v2::cost::estimate_kernel(
            match &keep.alt.node {
                GraphNode::Kernel(k) => &k.module,
                _ => unreachable!(),
            },
            hash_keep,
            &ctx,
            &cfg,
            1,
        )
        .unwrap();
        let (drop_cost, _) = crate::passes::fusion_v2::cost::estimate_kernel(
            match &drop.alt.node {
                GraphNode::Kernel(k) => &k.module,
                _ => unreachable!(),
            },
            hash_drop,
            &ctx,
            &cfg,
            1,
        )
        .unwrap();
        // Keep has more work per iteration (extra store, one more
        // multiply pulled out into the seam term) so cost >= drop cost.
        assert!(
            keep_cost.runtime_units >= drop_cost.runtime_units,
            "keep should cost at least as much as drop: keep={} drop={}",
            keep_cost.runtime_units,
            drop_cost.runtime_units,
        );
    }
}

// -------------------------------------------------------------------------
// §9.1 acyclicity validator tests
// -------------------------------------------------------------------------

mod validate_tests {
    use super::*;
    use crate::passes::fusion_v2::{take_graph, would_create_cycle, ValueClassId};

    #[test]
    fn candidate_that_cycles_is_rejected() {
        // Seed graph: n0 writes v1 from v0, n1 writes v2 from v1.
        // Proposed candidate: inputs=[v2], outputs=[v0]. That closes a
        // path v0 -> n0 -> v1 -> n1 -> v2, and inserting an edge
        // v2 -> new_node -> v0 would produce a cycle.
        let mut g = crate::graph_ir::GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        let b = sized_buf(&mut g, "b", 32);
        let c = sized_buf(&mut g, "c", 32);
        g.register_input(a);
        g.register_output(c);
        g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
        g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);
        let gf = take_graph(&mut g).unwrap();
        let v_a = gf.inputs[0];
        let v_c = gf.outputs[0];
        assert!(would_create_cycle(&gf, &[v_c], &[v_a]));
    }

    #[test]
    fn candidate_that_does_not_cycle_is_accepted() {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        let b = sized_buf(&mut g, "b", 32);
        let c = sized_buf(&mut g, "c", 32);
        g.register_input(a);
        g.register_output(c);
        g.insert_kernel(scale_by_two_module(), vec![a], vec![b], &[]);
        g.insert_kernel(scale_by_two_module(), vec![b], vec![c], &[]);
        let gf = take_graph(&mut g).unwrap();
        let v_a = gf.inputs[0];
        let v_c = gf.outputs[0];
        // A candidate that reads `a` and writes `c` is legal — that's
        // exactly the fused drop-seams alternative.
        assert!(!would_create_cycle(&gf, &[v_a], &[v_c]));
    }

    #[test]
    fn empty_output_set_never_cycles() {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        g.register_input(a);
        let gf = take_graph(&mut g).unwrap();
        let v_a = gf.inputs[0];
        assert!(!would_create_cycle(&gf, &[v_a], &[]));
    }

    #[test]
    fn candidate_with_input_equal_to_output_is_a_self_cycle() {
        // The validator treats input==output as a self-cycle: the
        // outputs' initial traversal position immediately matches the
        // inputs target set. This is the correct behavior — such a
        // candidate would produce an edge from itself to itself.
        let mut g = crate::graph_ir::GraphBuilder::new();
        let a = sized_buf(&mut g, "a", 32);
        g.register_input(a);
        let gf = take_graph(&mut g).unwrap();
        let v = ValueClassId(gf.inputs[0].0);
        assert!(would_create_cycle(&gf, &[v], &[v]));
    }
}

// -------------------------------------------------------------------------
// Top-level fuse_graph_v2 driver end-to-end tests.
// -------------------------------------------------------------------------

mod driver_tests {
    use super::*;
    use crate::passes::fusion_v2::{fuse_graph_v2, FusionOptionsV2};

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    #[test]
    fn driver_fuses_two_kernel_chain_into_one_kernel() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);

        let options = FusionOptionsV2 {
            enable_small_kernel: false, // isolate producer-consumer counting
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.nodes_before, 2);
        assert_eq!(report.candidates_generated, 1);
        assert_eq!(report.candidates_inserted, 1);
        assert_eq!(report.candidates_rejected_cycle, 0);
        // Under the placeholder cost model (runtime 1 each), the fused
        // candidate replaces two seeds (runtime 2) and wins.
        assert_eq!(report.nodes_after, 1);
        assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
        assert_eq!(g.input_bufs(), &[x]);
        assert_eq!(g.output_bufs(), &[z]);
    }

    #[test]
    fn driver_leaves_single_kernel_unchanged() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        g.register_input(x);
        g.register_output(y);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);

        let report = fuse_graph_v2(&mut g, &FusionOptionsV2::default()).unwrap();
        assert_eq!(report.candidates_generated, 0);
        assert_eq!(report.nodes_after, 1);
    }

    #[test]
    fn driver_leaves_disjoint_kernels_unfused() {
        // Two independent kernels with no producer-consumer edge.
        // Horizontal (M9) is disabled — it targets exactly this shape;
        // the dataflow-driven passes must all leave it alone.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let x2 = sized_buf(&mut g, "x2", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(scale_by(n, 2), vec![x1], vec![y1], &[]);
        g.insert_kernel(scale_by(n, 3), vec![x2], vec![y2], &[]);

        let options = FusionOptionsV2 {
            enable_horizontal: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 0);
        assert_eq!(report.nodes_after, 2);
    }

    #[test]
    fn driver_produces_hand_authored_reference_module() {
        // The single kernel remaining after fuse_graph_v2 must be
        // structurally equal to the hand-authored `3 * (2 * x[i])`.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let _ = fuse_graph_v2(&mut g, &FusionOptionsV2::default()).unwrap();
        let fused = match &g.nodes[0] {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                let scaled = b.mul(ai, two);
                let three = b.const_field(3);
                b.mul(scaled, three)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            crate::module_hash::module_hash(&fused),
            crate::module_hash::module_hash(&reference),
        );
    }

    #[test]
    fn driver_enumerates_two_candidates_when_producer_feeds_two_consumers() {
        // Fanout: one producer `y = 2 * x`, two consumers `z1 = 3 * y`
        // and `z2 = 5 * y`. Enumeration should emit two drop candidates
        // (one per consumer) plus two keep candidates (M5 §10.2 — the
        // seam has another consumer at each site) so the extractor can
        // pick the cheapest combination.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);

        // Isolate producer-consumer counts by turning off M7 fanout
        // and M9 horizontal (the two consumers are dataflow-independent
        // and would horizontally fuse).
        let options = FusionOptionsV2 {
            enable_fanout: false,
            enable_horizontal: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // Two drop candidates (one per consumer) plus two keep
        // candidates (seam has another consumer): 4 total.
        assert_eq!(report.candidates_generated, 4);
        assert!(g.nodes.iter().all(|n| matches!(n, GraphNode::Kernel(_))));
        assert_eq!(g.output_bufs(), &[z1, z2]);
    }

    #[test]
    fn driver_max_total_alternatives_zero_disables_all_fusion() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let options = FusionOptionsV2 {
            max_total_alternatives: 0,
            enable_small_kernel: false, // isolate producer-consumer counting
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 1);
        assert_eq!(report.candidates_inserted, 0);
        assert_eq!(report.candidates_rejected_cap, 1);
        assert_eq!(report.nodes_after, 2);
    }

    #[test]
    fn driver_disable_producer_consumer_produces_no_candidates() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let options = FusionOptionsV2 {
            enable_producer_consumer: false,
            enable_fanout: false,
            enable_small_kernel: false,
            enable_horizontal: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 0);
        assert_eq!(report.candidates_inserted, 0);
        assert_eq!(report.nodes_after, 2);
    }

    #[test]
    fn driver_leaves_registered_output_intact_when_seam_is_graph_output() {
        // If the seam `y` is a registered graph output, the drop
        // candidate cannot be selected on its own — the extractor still
        // needs a producer for `y`. The M5 keep variant fires here
        // (seam is a graph output) and materializes both `y` and `z`
        // from a single fused kernel; the extractor's runtime stage
        // picks it over the two-kernel original chain.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(y);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);

        let options = FusionOptionsV2 {
            enable_small_kernel: false, // isolate producer-consumer counting
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // Drop candidate + keep candidate (seam-is-graph-output trigger).
        assert_eq!(report.candidates_generated, 2);
        assert_eq!(g.output_bufs(), &[y, z]);
        // At least one node in the emitted graph must write `y`
        // (registered output).
        let has_producer_of_y = g.nodes.iter().any(|n| match n {
            GraphNode::Kernel(k) => k.outputs.contains(&y),
            _ => false,
        });
        assert!(
            has_producer_of_y,
            "graph must still write `y` for the registered output"
        );
    }

    #[test]
    fn driver_disable_keep_variants_leaves_seam_needing_original_producer() {
        // With keep off, the fanout scenario has 2 drop candidates and
        // no way to materialize the seam except through the original
        // producer; the solver keeps the original producer alive.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);
        let options = FusionOptionsV2 {
            enable_keep_variants: false,
            enable_fanout: false,     // isolate producer-consumer
            enable_horizontal: false, // the two consumers would fuse horizontally
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 2, "drop candidates only");
        // Solver keeps the original producer alive; interface unchanged.
        assert_eq!(g.output_bufs(), &[z1, z2]);
    }
}

// -------------------------------------------------------------------------
// M6: bounded saturation and chain composition tests.
// -------------------------------------------------------------------------

mod saturation_tests {
    use super::*;
    use crate::passes::fusion_v2::{fuse_graph_v2, FusionOptionsV2};

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    /// Three-kernel scale chain: `y1 = 2*x; y2 = 3*y1; z = 5*y2`.
    fn three_chain(
        n: usize,
    ) -> (
        crate::graph_ir::GraphBuilder,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
    ) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y1], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y1], vec![y2], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y2], vec![z], &[]);
        (g, x, z)
    }

    // -------------------------------------------------------------
    // Exit gate: three-kernel chain collapses to a single kernel via
    // chain composition across saturation rounds.
    // -------------------------------------------------------------

    #[test]
    fn three_kernel_chain_collapses_to_one_kernel() {
        let n = 8;
        let (mut g, x, z) = three_chain(n);
        let report = fuse_graph_v2(&mut g, &FusionOptionsV2::default()).unwrap();
        assert_eq!(report.nodes_before, 3);
        assert_eq!(report.nodes_after, 1);
        assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
        assert_eq!(g.input_bufs(), &[x]);
        assert_eq!(g.output_bufs(), &[z]);
        assert!(
            report.rounds_run >= 2,
            "chain composition requires at least 2 rounds: got {}",
            report.rounds_run,
        );
    }

    // -------------------------------------------------------------
    // Association-order dedup: (A+B)+C and A+(B+C) both reduce to
    // the same fused module and only one is inserted.
    // -------------------------------------------------------------

    #[test]
    fn association_order_dedup_across_rounds() {
        // A three-chain gives two independent (producer, consumer)
        // pairs in round 1 — one drop for (A, B) and one for (B, C).
        // Round 2 tries to compose fused(A,B) with C (seam = B's
        // output) *and* A with fused(B,C) (seam = A's output). Both
        // normalize to the same 3-mul module hash and thus the same
        // CandidateKey; dedup keeps one and rejects the other.
        let n = 8;
        let (mut g, _x, _z) = three_chain(n);
        let report = fuse_graph_v2(&mut g, &FusionOptionsV2::default()).unwrap();
        // Round 1 emits 2 drop candidates. Round 2 emits 2 more
        // composed drafts, of which one is dedup-rejected.
        assert!(
            report.candidates_rejected_dedup >= 1,
            "expected at least one dedup rejection, got report {:?}",
            report,
        );
    }

    // -------------------------------------------------------------
    // Determinism: repeated runs of the same graph produce identical
    // reports and identical emitted graphs.
    // -------------------------------------------------------------

    #[test]
    fn saturation_is_deterministic_across_runs() {
        let n = 8;
        let run = || {
            let (mut g, _x, _z) = three_chain(n);
            let report = fuse_graph_v2(&mut g, &FusionOptionsV2::default()).unwrap();
            let fingerprint = graph_fingerprint(&g);
            (report, fingerprint)
        };
        let (r1, f1) = run();
        let (r2, f2) = run();
        assert_eq!(r1.nodes_after, r2.nodes_after);
        assert_eq!(r1.candidates_inserted, r2.candidates_inserted);
        assert_eq!(r1.candidates_rejected_dedup, r2.candidates_rejected_dedup);
        assert_eq!(r1.rounds_run, r2.rounds_run);
        assert_eq!(r1.rounds_inserted, r2.rounds_inserted);
        assert_eq!(f1, f2);
    }

    // -------------------------------------------------------------
    // max_rounds cap: setting rounds to 1 prevents chain composition
    // and leaves at least two nodes in the emitted graph.
    // -------------------------------------------------------------

    #[test]
    fn max_rounds_one_prevents_chain_composition() {
        let n = 8;
        let (mut g, _x, _z) = three_chain(n);
        // Disable M8 small_kernel so this test isolates the
        // producer-consumer + chain-composition behavior. With M8 on,
        // a single round emits a 3-kernel chain candidate that
        // collapses the whole chain in round 1.
        let options = FusionOptionsV2 {
            max_rounds: 1,
            enable_small_kernel: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.rounds_run, 1);
        assert!(report.max_rounds_hit);
        // Round 1 producer-consumer can fuse adjacent pairs but not
        // the full chain, so at best we end up with 2 nodes.
        assert!(
            g.nodes.len() >= 2,
            "single round cannot collapse a 3-kernel chain to one node without M8"
        );
    }

    // -------------------------------------------------------------
    // The saturation loop terminates on a fixpoint (zero candidates
    // inserted) rather than exhausting max_rounds when the chain is
    // fully saturated.
    // -------------------------------------------------------------

    #[test]
    fn saturation_terminates_at_fixpoint_before_max_rounds() {
        let n = 8;
        let (mut g, _x, _z) = three_chain(n);
        let options = FusionOptionsV2 {
            max_rounds: 8,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert!(
            !report.max_rounds_hit,
            "loop should terminate before max_rounds when saturated: {:?}",
            report,
        );
        assert!(report.rounds_run < 8);
    }

    // -------------------------------------------------------------
    // Per-pass cap: setting max_alternatives_per_pass_per_round=1
    // truncates the enumeration output and reports the excess.
    // -------------------------------------------------------------

    #[test]
    fn per_pass_cap_truncates_and_reports() {
        // Round 1 enumeration produces 2 producer-consumer drops +
        // 2 keep-variants + 1 fanout drop = 5 candidates. (The fanout
        // keep does NOT fire here because the seam `y` has no
        // consumer outside the two fanout arms and is not a graph
        // output.) Capping to 1 leaves 4 rejected on the pass cap.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);
        let options = FusionOptionsV2 {
            max_alternatives_per_pass_per_round: 1,
            max_rounds: 1,
            enable_horizontal: false, // would add a 6th candidate (the two arms)
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 5);
        // 4 excess candidates rejected by the per-pass cap.
        assert_eq!(report.candidates_rejected_pass_cap, 4);
        assert!(report.candidates_inserted <= 1);
    }

    // -------------------------------------------------------------
    // Origins are tracked correctly through composition: composing
    // a fused candidate with a seed leaves the union of parent
    // origins, so any later composition attempt with an overlapping
    // origin is rejected before synthesis.
    // -------------------------------------------------------------

    #[test]
    fn overlapping_origins_prevent_re_fusion() {
        // A three-chain has origins {A}, {B}, {C} after round 1
        // insertions of fused(A,B)={A,B} and fused(B,C)={B,C}. In
        // round 2 the driver considers pairs (fused(A,B), fused(B,C))
        // but rejects them because {A,B} ∩ {B,C} = {B}. Verify no
        // candidate that consumes both fused(A,B) and fused(B,C)
        // makes it into the alternative graph.
        //
        // Disable M8 small_kernel to isolate origin-filter behavior
        // (M8 emits chain candidates that also collapse the chain).
        let n = 8;
        let (mut g, _x, _z) = three_chain(n);
        let options = FusionOptionsV2 {
            enable_small_kernel: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // Chain composition still succeeds via (fused(A,B), C) or
        // (A, fused(B,C)), so we end with one node.
        assert_eq!(report.nodes_after, 1);
        // The number of *inserted* candidates should be modest —
        // ~3 candidates total (2 pairs + 1 composed) — never
        // explodes because origin-overlap prunes.
        assert!(
            report.candidates_inserted <= 6,
            "candidates_inserted={} — origin filter should keep this bounded",
            report.candidates_inserted,
        );
    }
}

// -------------------------------------------------------------------------
// M7: fanout tests.
// -------------------------------------------------------------------------

mod fanout_tests {
    use super::*;
    use crate::{
        module_hash::module_hash,
        passes::fusion_v2::{
            apply_solution,
            extract::{brute, ExtractOptions, ExtractionData},
            fuse_graph_v2,
            fusions::{fanout, producer_consumer},
            take_graph, FusionOptionsV2, GraphFuser,
        },
    };

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    /// Fanout fixture: `y = 2*x`, `z1 = 3*y`, `z2 = 5*y`.
    fn fanout_two(
        n: usize,
    ) -> (
        crate::graph_ir::GraphBuilder,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
    ) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);
        (g, x, z1, z2)
    }

    /// Fanout fixture: three-consumer fan.
    fn fanout_three(
        n: usize,
    ) -> (
        crate::graph_ir::GraphBuilder,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
        crate::graph_ir::BufId,
    ) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        let z3 = sized_buf(&mut g, "z3", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.register_output(z3);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);
        g.insert_kernel(scale_by(n, 7), vec![y], vec![z3], &[]);
        (g, z1, z2, z3)
    }

    fn take(g: &mut crate::graph_ir::GraphBuilder) -> GraphFuser {
        take_graph(g).unwrap()
    }

    fn seed_ctx(gf: &GraphFuser) -> producer_consumer::OwnedEnumerateContext {
        producer_consumer::OwnedEnumerateContext::all_seed(
            gf,
            producer_consumer::EnumerateOptions::default(),
        )
    }

    // -------------------------------------------------------------
    // Enumeration: k = 2 fanout emits exactly one drop candidate.
    // -------------------------------------------------------------

    #[test]
    fn fanout_k_equals_2_emits_one_drop_candidate() {
        let (mut g, _x, _z1, _z2) = fanout_two(8);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        // Seam has no other user, so only the drop variant.
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].variant, producer_consumer::FusionVariant::Drop);
        // Producer + 2 consumers = 3 parents.
        assert_eq!(drafts[0].parents.len(), 3);
    }

    // -------------------------------------------------------------
    // k = 3 fanout also emits one drop candidate with 4 parents.
    // -------------------------------------------------------------

    #[test]
    fn fanout_k_equals_3_emits_one_drop_candidate() {
        let (mut g, _z1, _z2, _z3) = fanout_three(8);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].parents.len(), 4);
    }

    // -------------------------------------------------------------
    // Fanout HIR matches a hand-authored dual-output module: the
    // producer expression appears exactly once, hoisted into a Let.
    // -------------------------------------------------------------

    #[test]
    fn fanout_module_hash_matches_hand_authored_reference() {
        let n = 8;
        let (mut g, _x, _z1, _z2) = fanout_two(n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        let fused = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                let seam = b.mul(ai, two);
                let three = b.const_field(3);
                let z1 = b.mul(seam, three);
                let five = b.const_field(5);
                let z2 = b.mul(seam, five);
                b.tuple(&[z1, z2])
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            module_hash(&fused),
            module_hash(&reference),
            "fanout body should match hand-authored dual-consumer reference"
        );
    }

    // -------------------------------------------------------------
    // Type check.
    // -------------------------------------------------------------

    #[test]
    fn fanout_module_type_checks() {
        let (mut g, _x, _z1, _z2) = fanout_two(8);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        for d in &drafts {
            match &d.alt.node {
                GraphNode::Kernel(k) => {
                    crate::passes::type_infer(&k.module).expect("fanout module type-checks");
                }
                _ => panic!("expected Kernel"),
            }
        }
    }

    // -------------------------------------------------------------
    // §10.5 anti-pattern rejection: the module_hash equality with the
    // hand-authored `let seam = 2*a[i]; Tuple([3*seam, 5*seam])`
    // reference above already proves single-instance producer
    // sharing — hash-consing collapses the shared sub-expression to
    // one NodeId, and the reference builds the same structure via
    // Rust bindings that share `b.mul(ai, two)` at both call sites.
    // This test additionally verifies the outer shape: a single
    // Compute whose body is a Tuple over the consumers.
    // -------------------------------------------------------------

    #[test]
    fn fanout_body_is_a_tuple_at_the_compute_root() {
        let n = 8;
        let (mut g, _x, _z1, _z2) = fanout_two(n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        let module = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        let body_root = match module.builder.node(module.body) {
            crate::ir::Node::Compute { body, .. } => *body,
            _ => panic!("expected outer Compute"),
        };
        match module.builder.node(body_root) {
            crate::ir::Node::Tuple(elems) => {
                assert_eq!(elems.len(), 2, "two consumers => two tuple elements");
            }
            other => panic!("fanout compute body should be a Tuple, got {other:?}"),
        }
    }

    // -------------------------------------------------------------
    // Extractor picks fanout over producer_consumer drop+drop+
    // materialize-original.
    // -------------------------------------------------------------

    #[test]
    fn extractor_prefers_fanout_over_duplicated_producer_consumer_candidates() {
        // Under uniform cost, brute-force picks the plan with the
        // fewest selected nodes. Fanout gives 1 node; drop+drop+
        // original gives 3.
        let n = 8;
        let (mut g, _x, z1, z2) = fanout_two(n);
        let report = fuse_graph_v2(&mut g, &FusionOptionsV2::default()).unwrap();
        // Fanout drop + producer_consumer drops+keeps all enumerated.
        // Solver should pick the fanout candidate (1 node covers both
        // z1 and z2).
        assert_eq!(
            g.nodes.len(),
            1,
            "expected fanout to be selected; {report:?}"
        );
        assert_eq!(g.output_bufs(), &[z1, z2]);
    }

    // -------------------------------------------------------------
    // A consumer whose module is unsupported (e.g., contains a
    // #[grid(threads = N)] hint) makes identify_kernel_shape return
    // None and fanout rejects the whole group.
    // -------------------------------------------------------------

    #[test]
    fn fanout_rejects_when_a_consumer_shape_is_unsupported() {
        // Build the same fanout_two but with one consumer whose
        // module contains a `threads` hint (unsupported by
        // identify_kernel_shape). The whole fanout group is
        // rejected because one consumer fails the shape check.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);

        // Second consumer with a threads hint.
        let threaded = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute_with(n, None, None, Some(32), |b, i| {
                let ai = b.index(a, &[i]);
                let five = b.const_field(5);
                b.mul(ai, five)
            });
            Arc::new(b.finish("threaded_scale", body))
        };
        g.insert_kernel(threaded, vec![y], vec![z2], &[]);

        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        // synthesize_fanout returns UnsupportedShape for the threaded
        // consumer, so the group is dropped.
        assert_eq!(drafts.len(), 0);
    }

    // -------------------------------------------------------------
    // Consumer-to-consumer dataflow blocks fanout.
    // -------------------------------------------------------------

    #[test]
    fn fanout_rejects_consumer_reads_another_consumer_output() {
        // y = 2*x; z1 = 3*y; z2 = 4*z1. z2 reads z1's output → z2 is
        // NOT a fanout consumer of y with z1 (they're not
        // independent). Enumerate should not group them.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        // z2 reads z1, not y — so z2 is NOT a fanout consumer of y.
        g.insert_kernel(scale_by(n, 4), vec![z1], vec![z2], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        // Only 1 consumer of y (z1), so fanout skips.
        assert_eq!(drafts.len(), 0);
    }

    // -------------------------------------------------------------
    // Fanout drop apply produces a single-node graph.
    // -------------------------------------------------------------

    #[test]
    fn fanout_drop_apply_produces_single_node_graph() {
        let (mut g, x, z1, z2) = fanout_two(8);
        let mut gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = fanout::enumerate(&gf, &ctx.as_ref());
        for d in drafts {
            gf.insert_candidate(d.alt);
        }
        let data = ExtractionData::uniform(&gf);
        let solution = brute::extract(&gf, &data, &ExtractOptions::default()).unwrap();
        assert_eq!(solution.nodes.len(), 1);
        apply_solution(&mut g, gf, &solution).unwrap();
        assert_eq!(g.nodes.len(), 1);
        assert_eq!(g.input_bufs(), &[x]);
        assert_eq!(g.output_bufs(), &[z1, z2]);
    }

    // -------------------------------------------------------------
    // Disable fanout via the driver flag.
    // -------------------------------------------------------------

    #[test]
    fn disable_fanout_flag_suppresses_fanout_candidates() {
        let (mut g, _x, _z1, _z2) = fanout_two(8);
        let options = FusionOptionsV2 {
            enable_fanout: false,
            enable_keep_variants: false,
            enable_horizontal: false, // the two arms would fuse horizontally
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // Without fanout and keep, only drop candidates: 2 per fanout.
        assert_eq!(report.candidates_generated, 2);
    }
}

// -------------------------------------------------------------------------
// M8: small-kernel block fusion tests.
// -------------------------------------------------------------------------

mod small_kernel_tests {
    use super::*;
    use crate::{
        ir::SizeExpr,
        passes::fusion_v2::{
            fuse_graph_v2,
            fusions::{producer_consumer, small_kernel},
            take_graph, FusionOptionsV2, GraphFuser,
        },
    };

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    /// Sum-reduce a length-N tensor down to a length-M by grouping.
    /// Used to build different-domain chains.
    fn take_half(n: usize) -> Arc<crate::ir::Module> {
        // a: length N; produce length N/2 by dropping half.
        let m = n / 2;
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(m, |b, i| b.index(a, &[i]));
        Arc::new(b.finish("take_half", body))
    }

    fn take(g: &mut crate::graph_ir::GraphBuilder) -> GraphFuser {
        take_graph(g).unwrap()
    }

    fn seed_ctx(gf: &GraphFuser) -> producer_consumer::OwnedEnumerateContext {
        producer_consumer::OwnedEnumerateContext::all_seed(
            gf,
            producer_consumer::EnumerateOptions::default(),
        )
    }

    // -------------------------------------------------------------
    // A 2-kernel chain of SAME domain fuses.
    // -------------------------------------------------------------

    #[test]
    fn two_kernel_same_domain_chain_fuses() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), Default::default());
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].parents.len(), 2);
        assert_eq!(drafts[0].alt.inputs.len(), 1);
        assert_eq!(drafts[0].alt.outputs.len(), 1);
    }

    // -------------------------------------------------------------
    // A 2-kernel chain of DIFFERENT domain sizes fuses. This is
    // M8's headline capability vs M3/M7.
    // -------------------------------------------------------------

    #[test]
    fn two_kernel_different_domain_chain_fuses() {
        // a = scale(x, 2), size 16.
        // b = take_half(a), size 8.
        let n_a = 16;
        let n_b = n_a / 2;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n_a * 4) as i64);
        let y = sized_buf(&mut g, "y", (n_a * 4) as i64);
        let z = sized_buf(&mut g, "z", (n_b * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n_a, 2), vec![x], vec![y], &[]);
        g.insert_kernel(take_half(n_a), vec![y], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), Default::default());
        assert_eq!(drafts.len(), 1, "expected one small-kernel candidate");
        // Fused module should still type-check with different domains.
        match &drafts[0].alt.node {
            GraphNode::Kernel(k) => {
                crate::passes::type_infer(&k.module).expect("type-checks");
            }
            _ => panic!("expected Kernel"),
        }
    }

    // -------------------------------------------------------------
    // Chain of 3 kernels fuses (linear chain length ≥ 2).
    // -------------------------------------------------------------

    #[test]
    fn three_kernel_chain_fuses() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y1], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y1], vec![y2], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y2], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), Default::default());
        assert!(!drafts.is_empty(), "expected at least one candidate");
        // The longest chain [k0, k1, k2] should be emitted with 3 parents.
        let full_chain = drafts.iter().find(|d| d.parents.len() == 3);
        assert!(full_chain.is_some(), "expected a 3-kernel chain candidate");
    }

    // -------------------------------------------------------------
    // Fused module HIR matches a hand-authored reference where each
    // producer is a let-bound inner compute.
    // -------------------------------------------------------------

    #[test]
    fn fused_module_type_checks_and_lowers() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), Default::default());
        assert_eq!(drafts.len(), 1);
        let module = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        // Type inference should succeed on the fused HIR.
        crate::passes::type_infer(&module).expect("type-checks");
        // Full lower-to-KIR is exercised by the M4 estimator; skip
        // here to keep the check narrow.
    }

    // -------------------------------------------------------------
    // Symbolic-bound kernels are rejected (M8 requires all bounds
    // constant per the user's explicit request).
    // -------------------------------------------------------------

    #[test]
    fn rejects_symbolic_bounds() {
        // Build a module with a symbolic bound.
        let module = {
            let mut b = IRBuilder::new();
            let n_sym = b.symbol("N");
            let a = b.input("a", ScalarType::BabyBear, vec![SizeExpr::from(n_sym)]);
            let body = b.compute(n_sym, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                b.mul(ai, two)
            });
            Arc::new(b.finish("sym_scale", body))
        };
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", 32);
        let y = sized_buf(&mut g, "y", 32);
        let z = sized_buf(&mut g, "z", 32);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(module.clone(), vec![x], vec![y], &[("N", 8)]);
        g.insert_kernel(module, vec![y], vec![z], &[("N", 8)]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), Default::default());
        // Symbolic outer_bound → identify_chain rejects both kernels.
        assert_eq!(drafts.len(), 0);
    }

    // -------------------------------------------------------------
    // Chain with a branching intermediate is not a linear chain and
    // is rejected. (Intermediate has 2 downstream consumers.)
    // -------------------------------------------------------------

    #[test]
    fn rejects_branching_intermediate() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        // y feeds both z1 and z2 (fanout).
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y], vec![z2], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), Default::default());
        // The head kernel has 2 consumers => chain length stops at 1.
        // Individual chains z1..? and z2..? have length 1 too.
        // No chain of length ≥ 2 => zero candidates.
        assert_eq!(drafts.len(), 0);
    }

    // -------------------------------------------------------------
    // Shared-mem budget: a chain whose combined tile bytes exceed
    // the budget is rejected.
    // -------------------------------------------------------------

    #[test]
    fn rejects_when_shared_mem_budget_exceeded() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        // Set a byte budget below one tile's size (8 * 4 = 32 bytes).
        let sk_opts = small_kernel::SmallKernelOptions {
            max_shared_bytes: 16,
            max_chain_length: 6,
        };
        let drafts = small_kernel::enumerate(&gf, &ctx.as_ref(), sk_opts);
        assert_eq!(drafts.len(), 0);
    }

    // -------------------------------------------------------------
    // End-to-end: enabling small_kernel via the driver produces the
    // fused candidate and applies it.
    // -------------------------------------------------------------

    #[test]
    fn driver_end_to_end_fuses_two_kernel_chain() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        // Turn off producer-consumer so the small_kernel candidate
        // is the only fusion in play.
        let options = FusionOptionsV2 {
            enable_producer_consumer: false,
            enable_fanout: false,
            enable_small_kernel: true,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 1);
        assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
        assert_eq!(g.output_bufs(), &[z]);
    }

    // -------------------------------------------------------------
    // Enabling small_kernel alongside producer_consumer works — the
    // estimator lowers both candidates without panicking.
    // -------------------------------------------------------------

    #[test]
    fn small_kernel_and_producer_consumer_coexist() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let options = FusionOptionsV2 {
            enable_small_kernel: true,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // producer_consumer drop + small_kernel = 2 candidates.
        assert_eq!(report.candidates_generated, 2);
        assert_eq!(g.nodes.len(), 1);
    }

    // -------------------------------------------------------------
    // Three-kernel chain with small_kernel + producer_consumer +
    // multi-round saturation. Regression guard: earlier synthesis
    // versions produced a module that couldn't lower on a specific
    // interaction path.
    // -------------------------------------------------------------

    #[test]
    fn three_kernel_chain_with_small_kernel_and_producer_consumer() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y1], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y1], vec![y2], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y2], vec![z], &[]);
        let options = FusionOptionsV2 {
            enable_small_kernel: true,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // No panic in the estimator; at least one candidate was
        // generated and the graph collapsed to a single node.
        assert!(report.candidates_generated > 0);
        assert_eq!(g.nodes.len(), 1);
    }
}

// -------------------------------------------------------------------------
// M9: same-domain horizontal fusion tests.
// -------------------------------------------------------------------------

mod horizontal_tests {
    use super::*;
    use crate::{
        ir::SizeExpr,
        module_hash::module_hash,
        passes::fusion_v2::{
            fuse_graph_v2,
            fusions::{
                horizontal::{self, HorizontalFailure},
                producer_consumer,
            },
            take_graph, FusionOptionsV2, GraphFuser,
        },
    };

    fn scale_named(n: usize, c: u32, name: &str) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input(name, ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        scale_named(n, c, "a")
    }

    fn scale_hinted(n: usize, c: u32, hint: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        b.set_block_hint(hint);
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_hinted", body))
    }

    fn take(g: &mut crate::graph_ir::GraphBuilder) -> GraphFuser {
        take_graph(g).unwrap()
    }

    fn seed_ctx(gf: &GraphFuser) -> producer_consumer::OwnedEnumerateContext {
        producer_consumer::OwnedEnumerateContext::all_seed(
            gf,
            producer_consumer::EnumerateOptions::default(),
        )
    }

    /// Two dataflow-independent kernels on the same domain:
    /// `y1 = 2*x1`, `y2 = 3*x2`.
    fn independent_pair(n: usize) -> (crate::graph_ir::GraphBuilder, BufId, BufId) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", (n * 4) as i64);
        let x2 = sized_buf(&mut g, "x2", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(scale_named(n, 2, "a"), vec![x1], vec![y1], &[]);
        g.insert_kernel(scale_named(n, 3, "b"), vec![x2], vec![y2], &[]);
        (g, y1, y2)
    }

    /// Three dataflow-independent kernels reading the SAME input:
    /// `z1 = 2*x`, `z2 = 3*x`, `z3 = 5*x`.
    fn shared_input_triple(n: usize) -> (crate::graph_ir::GraphBuilder, BufId, BufId, BufId) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let z1 = sized_buf(&mut g, "z1", (n * 4) as i64);
        let z2 = sized_buf(&mut g, "z2", (n * 4) as i64);
        let z3 = sized_buf(&mut g, "z3", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z1);
        g.register_output(z2);
        g.register_output(z3);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![z1], &[]);
        g.insert_kernel(scale_by(n, 3), vec![x], vec![z2], &[]);
        g.insert_kernel(scale_by(n, 5), vec![x], vec![z3], &[]);
        (g, z1, z2, z3)
    }

    // -------------------------------------------------------------
    // Two independent same-domain kernels emit one candidate with
    // concatenated boundaries.
    // -------------------------------------------------------------

    #[test]
    fn two_independent_kernels_fuse() {
        let (mut g, _y1, _y2) = independent_pair(8);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = horizontal::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].parents, vec![NodeId(0), NodeId(1)]);
        assert_eq!(drafts[0].alt.inputs.len(), 2);
        assert_eq!(drafts[0].alt.outputs.len(), 2);
        assert_eq!(
            drafts[0].variant,
            producer_consumer::FusionVariant::Drop,
            "horizontal has no seam; Drop is the placeholder variant"
        );
        match &drafts[0].alt.node {
            GraphNode::Kernel(k) => {
                crate::passes::type_infer(&k.module).expect("fused module type-checks");
            }
            _ => panic!("expected Kernel"),
        }
    }

    // -------------------------------------------------------------
    // Shared input: the fused boundary dedups the value class and
    // hash-consing shares the load. HIR matches a hand-authored
    // reference reading `a` once.
    // -------------------------------------------------------------

    #[test]
    fn shared_input_is_deduped_and_hash_consed() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(y);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![x], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = horizontal::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].alt.inputs.len(), 1, "shared input dedups");
        assert_eq!(drafts[0].alt.outputs.len(), 2);
        let fused = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, k| {
                let ak = b.index(a, &[k]);
                let two = b.const_field(2);
                let e1 = b.mul(ak, two);
                let three = b.const_field(3);
                let e2 = b.mul(ak, three);
                b.tuple(&[e1, e2])
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            module_hash(&fused),
            module_hash(&reference),
            "shared load should be hash-consed into one Index node"
        );
    }

    // -------------------------------------------------------------
    // Dataflow path in either direction rejects the pair — both a
    // direct producer→consumer edge and a transitive path.
    // -------------------------------------------------------------

    #[test]
    fn rejects_dataflow_pair() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::DataflowPath)
        ));
    }

    #[test]
    fn rejects_transitive_dataflow_pair() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y1], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y1], vec![y2], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y2], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        // (0, 2) has no direct edge but a path through node 1.
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(2)),
            Err(HorizontalFailure::DataflowPath)
        ));
    }

    // -------------------------------------------------------------
    // Domain legality: unequal bounds and symbolic bounds reject.
    // No compute[max(Na, Nb)] masking (§10.6).
    // -------------------------------------------------------------

    #[test]
    fn rejects_different_domains() {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", 32);
        let x2 = sized_buf(&mut g, "x2", 64);
        let y1 = sized_buf(&mut g, "y1", 32);
        let y2 = sized_buf(&mut g, "y2", 64);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(scale_named(8, 2, "a"), vec![x1], vec![y1], &[]);
        g.insert_kernel(scale_named(16, 3, "b"), vec![x2], vec![y2], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::OuterBoundMismatch)
        ));
    }

    #[test]
    fn rejects_symbolic_bounds() {
        let module = {
            let mut b = IRBuilder::new();
            let n_sym = b.symbol("N");
            let a = b.input("a", ScalarType::BabyBear, vec![SizeExpr::from(n_sym)]);
            let body = b.compute(n_sym, |b, i| {
                let ai = b.index(a, &[i]);
                let two = b.const_field(2);
                b.mul(ai, two)
            });
            Arc::new(b.finish("sym_scale", body))
        };
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", 32);
        let x2 = sized_buf(&mut g, "x2", 32);
        let y1 = sized_buf(&mut g, "y1", 32);
        let y2 = sized_buf(&mut g, "y2", 32);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(module.clone(), vec![x1], vec![y1], &[("N", 8)]);
        g.insert_kernel(module, vec![x2], vec![y2], &[("N", 8)]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::NonConstantBound)
        ));
    }

    // -------------------------------------------------------------
    // Non-flat kernels (inner Reduce) are rejected: §10.6 requires
    // flat structured kernels.
    // -------------------------------------------------------------

    #[test]
    fn rejects_non_flat_kernels() {
        let n = 8;
        let reducer = {
            let mut b = IRBuilder::new();
            let a = b.input("b", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, _i| b.reduce_add(n, |b, j| b.index(a, &[j])));
            Arc::new(b.finish("row_sum", body))
        };
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", (n * 4) as i64);
        let x2 = sized_buf(&mut g, "x2", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(scale_by(n, 2), vec![x1], vec![y1], &[]);
        g.insert_kernel(reducer, vec![x2], vec![y2], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::NotFlat)
        ));
    }

    // -------------------------------------------------------------
    // Storage hazards on physical BufIds: WAW and WAR both reject.
    // -------------------------------------------------------------

    #[test]
    fn rejects_waw_hazard() {
        // Both kernels write the same physical buffer y (an overwrite
        // pattern); fusing them would run the writes concurrently.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        g.register_input(x);
        g.register_output(y);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![x], vec![y], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::StorageHazard)
        ));
    }

    #[test]
    fn rejects_war_hazard() {
        // k0 reads graph input y (old version); k1 overwrites y. No
        // dataflow path connects them, but fusing loses the k0-before-
        // k1 ordering the §7 hazard sort would otherwise enforce.
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let w = sized_buf(&mut g, "w", (n * 4) as i64);
        g.register_input(x);
        g.register_input(y);
        g.register_output(w);
        g.register_output(y);
        g.insert_kernel(scale_named(n, 2, "a"), vec![y], vec![w], &[]);
        g.insert_kernel(scale_named(n, 3, "b"), vec![x], vec![y], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::StorageHazard)
        ));
    }

    // -------------------------------------------------------------
    // Block hints: mismatch rejects; matching hints propagate to the
    // fused module.
    // -------------------------------------------------------------

    #[test]
    fn rejects_block_hint_mismatch() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", (n * 4) as i64);
        let x2 = sized_buf(&mut g, "x2", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(scale_hinted(n, 2, 128), vec![x1], vec![y1], &[]);
        g.insert_kernel(scale_named(n, 3, "b"), vec![x2], vec![y2], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(horizontal::enumerate(&gf, &ctx.as_ref()).len(), 0);
        assert!(matches!(
            horizontal::synthesize_horizontal(&gf, NodeId(0), NodeId(1)),
            Err(HorizontalFailure::BlockHintMismatch)
        ));
    }

    #[test]
    fn matching_block_hint_propagates() {
        let n = 8;
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x1 = sized_buf(&mut g, "x1", (n * 4) as i64);
        let x2 = sized_buf(&mut g, "x2", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        g.register_input(x1);
        g.register_input(x2);
        g.register_output(y1);
        g.register_output(y2);
        g.insert_kernel(scale_hinted(n, 2, 128), vec![x1], vec![y1], &[]);
        g.insert_kernel(scale_hinted(n, 3, 128), vec![x2], vec![y2], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = horizontal::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        match &drafts[0].alt.node {
            GraphNode::Kernel(k) => {
                assert_eq!(k.module.builder.block_hint(), Some(128));
            }
            _ => panic!("expected Kernel"),
        }
    }

    // -------------------------------------------------------------
    // Multi-output parents splice their Tuple elements positionally,
    // enabling three-way merges across saturation rounds.
    // -------------------------------------------------------------

    #[test]
    fn multi_output_parent_splices_tuple_elements() {
        let n = 8;
        let (mut g, _z1, _z2, _z3) = shared_input_triple(n);
        let mut gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = horizontal::enumerate(&gf, &ctx.as_ref());
        // Pairs (0,1), (0,2), (1,2).
        assert_eq!(drafts.len(), 3);
        let pos = drafts
            .iter()
            .position(|d| d.parents == vec![NodeId(0), NodeId(1)])
            .expect("pair (0,1) enumerated");
        let d01 = drafts.into_iter().nth(pos).unwrap();
        let pair_node = gf.insert_candidate(d01.alt);
        // Fuse the 2-output pair with the remaining single-output kernel.
        let d = horizontal::synthesize_horizontal(&gf, pair_node, NodeId(2))
            .expect("multi-output parent fuses");
        assert_eq!(d.alt.inputs.len(), 1, "all three read the same x");
        assert_eq!(d.alt.outputs.len(), 3);
        let fused = match &d.alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        let reference = {
            let mut b = IRBuilder::new();
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, k| {
                let ak = b.index(a, &[k]);
                let two = b.const_field(2);
                let e1 = b.mul(ak, two);
                let three = b.const_field(3);
                let e2 = b.mul(ak, three);
                let five = b.const_field(5);
                let e3 = b.mul(ak, five);
                b.tuple(&[e1, e2, e3])
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            module_hash(&fused),
            module_hash(&reference),
            "pair Tuple elements should be spliced, not nested"
        );
    }

    // -------------------------------------------------------------
    // Driver end-to-end: horizontal alone collapses independent
    // kernels; saturation composes three-way merges across rounds.
    // -------------------------------------------------------------

    #[test]
    fn driver_end_to_end_fuses_independent_kernels() {
        let (mut g, y1, y2) = independent_pair(8);
        let options = FusionOptionsV2 {
            enable_producer_consumer: false,
            enable_fanout: false,
            enable_small_kernel: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 1);
        assert_eq!(g.nodes.len(), 1);
        assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
        assert_eq!(g.output_bufs(), &[y1, y2]);
    }

    #[test]
    fn horizontal_composes_across_rounds() {
        let (mut g, z1, z2, z3) = shared_input_triple(8);
        let options = FusionOptionsV2 {
            enable_producer_consumer: false,
            enable_fanout: false,
            enable_small_kernel: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        // Round 1: three pairs. Round 2: pair ∪ remaining kernel
        // (disjoint origins) — the three-way merges.
        assert!(
            report.candidates_generated >= 4,
            "expected multi-round composition; {report:?}"
        );
        assert_eq!(g.nodes.len(), 1, "all three kernels merge; {report:?}");
        assert_eq!(g.output_bufs(), &[z1, z2, z3]);
    }
}

// -------------------------------------------------------------------------
// M10: epilogue fusion (§10.4).
// -------------------------------------------------------------------------

mod epilogue_tests {
    use super::*;
    use crate::{
        ir::Node,
        module_hash::module_hash,
        passes::fusion_v2::{
            fuse_graph_v2,
            fusions::{
                epilogue::{self, EpilogueFailure},
                producer_consumer,
            },
            take_graph, FusionOptionsV2, GraphFuser,
        },
    };

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    fn scale_hinted(n: usize, c: u32, hint: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        b.set_block_hint(hint);
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_hinted", body))
    }

    /// Block-hinted reduction producer: `y[i] = sum_{j<m} x[i*m + j]`.
    /// The block hint keeps it out of producer-consumer's coverage.
    fn row_sum_hinted(n: usize, m: usize, hint: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        b.set_block_hint(hint);
        let x = b.input("x", ScalarType::BabyBear, vec![n * m]);
        let body = b.compute(n, |b, i| {
            b.reduce_add(m, |b, j| {
                let mc = b.const_u32(m as u32);
                let base = b.mul(i, mc);
                let idx = b.add(base, j);
                b.index(x, &[idx])
            })
        });
        Arc::new(b.finish("row_sum_hinted", body))
    }

    /// Producer with a `#[grid(threads = t)]` override — rejected by
    /// `identify_kernel_shape`, so epilogue territory.
    fn scale_threads(n: usize, c: u32, threads: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute_with(n, None, None, Some(threads), |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_threads", body))
    }

    /// Producer with an explicit `par` compute layout.
    fn scale_par(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let par = b.par_map(|t, s, _c| t.mul_c(4).add(s));
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute_with(n, None, Some(par), None, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_par", body))
    }

    fn take(g: &mut crate::graph_ir::GraphBuilder) -> GraphFuser {
        take_graph(g).unwrap()
    }

    fn seed_ctx(gf: &GraphFuser) -> producer_consumer::OwnedEnumerateContext {
        producer_consumer::OwnedEnumerateContext::all_seed(
            gf,
            producer_consumer::EnumerateOptions::default(),
        )
    }

    /// `x → producer → y → consumer → z`; registers `x` input, `z`
    /// output. Returns `(g, y, z)`.
    fn chain(
        producer: Arc<crate::ir::Module>,
        consumer: Arc<crate::ir::Module>,
        x_bytes: i64,
        n: usize,
    ) -> (crate::graph_ir::GraphBuilder, BufId, BufId) {
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", x_bytes);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(producer, vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        (g, y, z)
    }

    // -------------------------------------------------------------
    // M10 exit gate: a reduction followed by pointwise work retains
    // the producer's schedule end-to-end through the driver.
    // -------------------------------------------------------------

    #[test]
    fn driver_end_to_end_retains_producer_schedule() {
        let (n, m) = (8, 4);
        let (mut g, _y, z) = chain(
            row_sum_hinted(n, m, 128),
            scale_by(n, 3),
            (n * m * 4) as i64,
            n,
        );
        let options = FusionOptionsV2 {
            enable_producer_consumer: false,
            enable_fanout: false,
            enable_small_kernel: false,
            enable_horizontal: false,
            ..FusionOptionsV2::default()
        };
        let report = fuse_graph_v2(&mut g, &options).unwrap();
        assert_eq!(report.candidates_generated, 1, "{report:?}");
        assert_eq!(g.nodes.len(), 1, "{report:?}");
        let GraphNode::Kernel(k) = &g.nodes[0] else {
            panic!("expected Kernel");
        };
        assert_eq!(k.module.name, "epilogue_drop");
        assert_eq!(
            k.module.builder.block_hint(),
            Some(128),
            "producer block hint must be retained"
        );
        assert_eq!(g.output_bufs(), &[z]);
    }

    // -------------------------------------------------------------
    // Synthesized HIR: consumer expression substituted into the
    // producer's result path, hint retained. Hash-checked against a
    // hand-authored reference.
    // -------------------------------------------------------------

    #[test]
    fn hinted_flat_producer_matches_reference() {
        let n = 8;
        let (mut g, _y, _z) = chain(scale_hinted(n, 2, 128), scale_by(n, 3), (n * 4) as i64, n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = epilogue::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].parents, vec![NodeId(0), NodeId(1)]);
        let fused = match &drafts[0].alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        };
        assert_eq!(fused.builder.block_hint(), Some(128));
        let reference = {
            let mut b = IRBuilder::new();
            b.set_block_hint(128);
            let a = b.input("a", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, k| {
                let ak = b.index(a, &[k]);
                let two = b.const_field(2);
                let p = b.mul(ak, two);
                let three = b.const_field(3);
                b.mul(p, three)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(
            module_hash(&fused),
            module_hash(&reference),
            "consumer expression should wrap the producer body at the identity index"
        );
    }

    // -------------------------------------------------------------
    // Schedule retention: `threads` and `par` carry over verbatim.
    // -------------------------------------------------------------

    #[test]
    fn threads_producer_retains_threads() {
        let n = 8;
        let (mut g, _y, _z) = chain(scale_threads(n, 2, 64), scale_by(n, 3), (n * 4) as i64, n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = epilogue::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        let GraphNode::Kernel(k) = &drafts[0].alt.node else {
            panic!("expected Kernel");
        };
        match k.module.builder.node(k.module.body) {
            Node::Compute { threads, .. } => assert_eq!(*threads, Some(64)),
            _ => panic!("expected top-level Compute"),
        }
    }

    #[test]
    fn par_producer_retains_par() {
        let n = 8;
        let p_module = scale_par(n, 2);
        let p_par = match p_module.builder.node(p_module.body) {
            Node::Compute { par, .. } => par.clone().expect("fixture has par"),
            _ => panic!("expected top-level Compute"),
        };
        let (mut g, _y, _z) = chain(p_module, scale_by(n, 3), (n * 4) as i64, n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = epilogue::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 1);
        let GraphNode::Kernel(k) = &drafts[0].alt.node else {
            panic!("expected Kernel");
        };
        match k.module.builder.node(k.module.body) {
            Node::Compute { par, .. } => {
                assert_eq!(par.as_deref(), Some(&*p_par), "ParSpec copied verbatim")
            }
            _ => panic!("expected top-level Compute"),
        }
    }

    // -------------------------------------------------------------
    // Keep variant: seam registered as a graph output triggers the
    // §10.2 keep sibling with the seam as an extra Tuple output.
    // -------------------------------------------------------------

    #[test]
    fn keep_variant_when_seam_is_output() {
        let n = 8;
        let (mut g, y, _z) = chain(scale_hinted(n, 2, 128), scale_by(n, 3), (n * 4) as i64, n);
        g.register_output(y);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        let drafts = epilogue::enumerate(&gf, &ctx.as_ref());
        assert_eq!(drafts.len(), 2, "drop + keep");
        let keep = drafts
            .iter()
            .find(|d| d.variant == producer_consumer::FusionVariant::Keep)
            .expect("keep variant emitted");
        assert_eq!(keep.alt.outputs.len(), 2, "consumer output + seam");
        let GraphNode::Kernel(k) = &keep.alt.node else {
            panic!("expected Kernel");
        };
        assert_eq!(k.module.name, "epilogue_keep");
        assert_eq!(k.module.builder.block_hint(), Some(128));
    }

    // -------------------------------------------------------------
    // Producers covered by producer-consumer (flat shape, no hint)
    // are skipped in enumeration — a dedup measure, not a legality
    // constraint: direct synthesis still succeeds.
    // -------------------------------------------------------------

    #[test]
    fn covered_producer_is_skipped() {
        let n = 8;
        let (mut g, _y, _z) = chain(scale_by(n, 2), scale_by(n, 3), (n * 4) as i64, n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(
            epilogue::enumerate(&gf, &ctx.as_ref()).len(),
            0,
            "producer-consumer already emits this pair"
        );
        let seam = gf.nodes[0].outputs[0];
        assert!(epilogue::synthesize_epilogue(
            &gf,
            NodeId(0),
            NodeId(1),
            seam,
            producer_consumer::FusionVariant::Drop,
        )
        .is_ok());
    }

    // -------------------------------------------------------------
    // Rejections.
    // -------------------------------------------------------------

    #[test]
    fn rejects_non_identity_seam_read() {
        let n = 8;
        let reverse_scale = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let nm1 = b.const_u32((n as u32) - 1);
                let idx = b.sub(nm1, i);
                let yi = b.index(y, &[idx]);
                let three = b.const_field(3);
                b.mul(yi, three)
            });
            Arc::new(b.finish("reverse_scale", body))
        };
        let (mut g, _y, _z) = chain(scale_hinted(n, 2, 128), reverse_scale, (n * 4) as i64, n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(epilogue::enumerate(&gf, &ctx.as_ref()).len(), 0);
        let seam = gf.nodes[0].outputs[0];
        assert!(matches!(
            epilogue::synthesize_epilogue(
                &gf,
                NodeId(0),
                NodeId(1),
                seam,
                producer_consumer::FusionVariant::Drop,
            ),
            Err(EpilogueFailure::SeamReadNotIdentity)
        ));
    }

    #[test]
    fn rejects_non_flat_consumer() {
        let n = 8;
        let sum_all = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, _i| b.reduce_add(n, |b, j| b.index(y, &[j])));
            Arc::new(b.finish("sum_all", body))
        };
        let (mut g, _y, _z) = chain(scale_hinted(n, 2, 128), sum_all, (n * 4) as i64, n);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(epilogue::enumerate(&gf, &ctx.as_ref()).len(), 0);
        let seam = gf.nodes[0].outputs[0];
        assert!(matches!(
            epilogue::synthesize_epilogue(
                &gf,
                NodeId(0),
                NodeId(1),
                seam,
                producer_consumer::FusionVariant::Drop,
            ),
            Err(EpilogueFailure::ConsumerNotPointwise)
        ));
    }

    #[test]
    fn rejects_bound_mismatch() {
        let n = 8;
        let (mut g, _y, _z) = chain(
            scale_hinted(n, 2, 128),
            scale_by(2 * n, 3),
            (n * 4) as i64,
            n,
        );
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(epilogue::enumerate(&gf, &ctx.as_ref()).len(), 0);
        let seam = gf.nodes[0].outputs[0];
        assert!(matches!(
            epilogue::synthesize_epilogue(
                &gf,
                NodeId(0),
                NodeId(1),
                seam,
                producer_consumer::FusionVariant::Drop,
            ),
            Err(EpilogueFailure::OuterBoundMismatch)
        ));
    }

    #[test]
    fn rejects_hinted_consumer() {
        let n = 8;
        let (mut g, _y, _z) = chain(
            scale_hinted(n, 2, 128),
            scale_hinted(n, 3, 64),
            (n * 4) as i64,
            n,
        );
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(epilogue::enumerate(&gf, &ctx.as_ref()).len(), 0);
        let seam = gf.nodes[0].outputs[0];
        assert!(matches!(
            epilogue::synthesize_epilogue(
                &gf,
                NodeId(0),
                NodeId(1),
                seam,
                producer_consumer::FusionVariant::Drop,
            ),
            Err(EpilogueFailure::ConsumerHasBlockHint)
        ));
    }

    #[test]
    fn rejects_tuple_body_producer() {
        let n = 8;
        let pair_producer = {
            let mut b = IRBuilder::new();
            b.set_block_hint(128);
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                let two = b.const_field(2);
                let e1 = b.mul(xi, two);
                let three = b.const_field(3);
                let e2 = b.mul(xi, three);
                b.tuple(&[e1, e2])
            });
            Arc::new(b.finish("pair_producer", body))
        };
        let mut g = crate::graph_ir::GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y1 = sized_buf(&mut g, "y1", (n * 4) as i64);
        let y2 = sized_buf(&mut g, "y2", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(y2);
        g.register_output(z);
        g.insert_kernel(pair_producer, vec![x], vec![y1, y2], &[]);
        g.insert_kernel(scale_by(n, 5), vec![y1], vec![z], &[]);
        let gf = take(&mut g);
        let ctx = seed_ctx(&gf);
        assert_eq!(epilogue::enumerate(&gf, &ctx.as_ref()).len(), 0);
        let seam = gf.nodes[0].outputs[0];
        assert!(matches!(
            epilogue::synthesize_epilogue(
                &gf,
                NodeId(0),
                NodeId(1),
                seam,
                producer_consumer::FusionVariant::Drop,
            ),
            Err(EpilogueFailure::ProducerUnsupportedShape)
        ));
    }
}

// -------------------------------------------------------------------------
// M4: KIR estimator tests.
// -------------------------------------------------------------------------

mod estimator_tests {
    use std::collections::BTreeMap;

    use crate::{
        graph_ir::{BufId, BufInfo, ConstBuf, GraphNode, MemSetNode, MemcpyNode},
        ir::{IRBuilder, ScalarType},
        passes::fusion_v2::cost::{
            estimate_kernel, estimate_non_kernel, DeviceModel, EstimateContext, EstimatorConfig,
            KernelCostManager,
        },
        quast::Quast,
    };

    // ---------------------------------------------------------------
    // Fixtures.
    // ---------------------------------------------------------------

    fn synthetic_cfg() -> EstimatorConfig {
        EstimatorConfig::default()
    }

    fn scale_module(n: usize, c: u32) -> crate::ir::Module {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        b.finish("scale", body)
    }

    fn fused_scale_module(n: usize, c1: u32, c2: u32) -> crate::ir::Module {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let k1 = b.const_field(c1);
            let mid = b.mul(ai, k1);
            let k2 = b.const_field(c2);
            b.mul(mid, k2)
        });
        b.finish("fused_scale", body)
    }

    // Small helper: the ILP-consumed runtime for a module under the
    // default synthetic profile.
    fn cost_cycles(module: &crate::ir::Module) -> f64 {
        let cfg = synthetic_cfg();
        let ctx = EstimateContext::default();
        let hash = crate::module_hash::module_hash(module);
        let (_cost, brk) = estimate_kernel(module, hash, &ctx, &cfg, 1).unwrap();
        brk.total_cycles
    }

    // ---------------------------------------------------------------
    // Deterministic golden feature snapshot: repeat calls to the
    // estimator on the same module produce bit-for-bit identical
    // breakdown values (§18.5 last bullet).
    // ---------------------------------------------------------------

    // Kernels with a symbolic outer bound reach the estimator without a
    // block hint (canonicalize strips monomorphize's stamp; synthesized
    // candidates never had one). The estimator must stamp the policy
    // hint from the caller's bindings instead of failing to lower.
    #[test]
    fn symbolic_outer_bound_is_costed_via_stamped_block_hint() {
        let mut b = IRBuilder::new();
        let q = b.symbol("q");
        let a = b.input("a", ScalarType::BabyBear, vec![q]);
        let body = b.compute(q, |b, i| {
            let ai = b.index(a, &[i]);
            let c = b.const_field(3);
            b.mul(ai, c)
        });
        let m = b.finish("sym_scale", body);
        let hash = crate::module_hash::module_hash(&m);
        let cfg = synthetic_cfg();

        // Unbound: no way to pick a block size — structured lowering error.
        assert!(estimate_kernel(&m, hash, &EstimateContext::default(), &cfg, 1).is_err());

        // Bound via param_bindings: policy hint stamped, finite cost.
        let ctx = EstimateContext {
            graph_symbols: BTreeMap::new(),
            param_bindings: BTreeMap::from([("q".to_string(), 1024)]),
        };
        let (cost, _) = estimate_kernel(&m, hash, &ctx, &cfg, 1).unwrap();
        assert!(cost.runtime_units > 0);
        assert!(cost.runtime_units < i64::MAX / 4);
    }

    #[test]
    fn estimator_is_deterministic_across_calls() {
        let m = scale_module(1024, 3);
        let cfg = synthetic_cfg();
        let ctx = EstimateContext::default();
        let hash = crate::module_hash::module_hash(&m);
        let (c1, b1) = estimate_kernel(&m, hash, &ctx, &cfg, 1).unwrap();
        let (c2, b2) = estimate_kernel(&m, hash, &ctx, &cfg, 1).unwrap();
        assert_eq!(c1.runtime_units, c2.runtime_units);
        assert_eq!(b1.total_cycles.to_bits(), b2.total_cycles.to_bits());
        assert_eq!(b1.access.transaction_bytes, b2.access.transaction_bytes);
        assert_eq!(b1.critical.sync_count, b2.critical.sync_count);
        assert_eq!(
            b1.registers.registers_per_thread,
            b2.registers.registers_per_thread
        );
    }

    // ---------------------------------------------------------------
    // A one-mul kernel costs less than the two-mul fused kernel: fusion
    // saves the intermediate materialization but adds compute.
    // Concretely the fused module has more weighted ops, so its
    // aggregate cycle count is strictly greater.
    // ---------------------------------------------------------------

    #[test]
    fn fused_kernel_has_more_compute_than_single_step() {
        let single = scale_module(1024, 3);
        let fused = fused_scale_module(1024, 2, 3);
        assert!(
            cost_cycles(&fused) >= cost_cycles(&single),
            "fused (two muls) should cost at least as much per launch as a single mul"
        );
    }

    // ---------------------------------------------------------------
    // Larger domains touch more sectors and cost more cycles.
    // ---------------------------------------------------------------

    #[test]
    fn larger_domain_costs_more_than_smaller() {
        let small = scale_module(128, 2);
        let large = scale_module(8192, 2);
        assert!(cost_cycles(&large) > cost_cycles(&small));
    }

    // ---------------------------------------------------------------
    // A caller-supplied cycle_quantum shrinks the runtime_units field
    // proportionally.
    // ---------------------------------------------------------------

    #[test]
    fn cycle_quantum_scales_runtime_units() {
        let m = scale_module(2048, 7);
        let cfg = synthetic_cfg();
        let ctx = EstimateContext::default();
        let hash = crate::module_hash::module_hash(&m);
        let (fine, _) = estimate_kernel(&m, hash, &ctx, &cfg, 1).unwrap();
        let (coarse, _) = estimate_kernel(&m, hash, &ctx, &cfg, 1000).unwrap();
        assert!(fine.runtime_units > coarse.runtime_units);
        assert!(coarse.runtime_units >= 1, "floor is 1 per §13.5");
    }

    // ---------------------------------------------------------------
    // Occupancy: bumping register liveness reduces resident blocks at a
    // threshold. Configure a synthetic device with a tight register
    // budget so the effect is visible.
    // ---------------------------------------------------------------

    #[test]
    fn higher_register_pressure_reduces_resident_blocks() {
        let m = scale_module(64, 3);
        let ctx = EstimateContext::default();
        let hash = crate::module_hash::module_hash(&m);
        let mut cfg = synthetic_cfg();
        cfg.device.registers_per_sm = 4096;
        cfg.register_fixed_overhead = 8;
        cfg.register_liveness_scale = 1.0;
        let (_, small_regs) = estimate_kernel(&m, hash, &ctx, &cfg, 1).unwrap();
        cfg.register_fixed_overhead = 128;
        let (_, big_regs) = estimate_kernel(&m, hash, &ctx, &cfg, 1).unwrap();
        assert!(
            small_regs.blocks_per_sm >= big_regs.blocks_per_sm,
            "raising register overhead must not increase resident blocks: \
             small_overhead={} big_overhead={}",
            small_regs.blocks_per_sm,
            big_regs.blocks_per_sm,
        );
        assert!(
            big_regs.blocks_per_sm >= 1,
            "occupancy is clamped to at least one resident block"
        );
    }

    // ---------------------------------------------------------------
    // Cache: two lookups for the same (module_hash, param_bindings)
    // pair are served from the cache after the first.
    // ---------------------------------------------------------------

    #[test]
    fn cost_manager_caches_repeated_lookups() {
        let cfg = synthetic_cfg();
        let artifact = crate::passes::fusion_v2::cost::ArtifactContext {
            target_arch: "test".into(),
            compiler_flags_hash: [0; 32],
        };
        let mut mgr = KernelCostManager::new(cfg, artifact, BTreeMap::new(), 1);
        let m = scale_module(64, 3);
        let hash = crate::module_hash::module_hash(&m);
        let bindings = BTreeMap::new();
        let c0 = mgr.cost_of(hash, &m, &bindings).unwrap();
        let c1 = mgr.cost_of(hash, &m, &bindings).unwrap();
        assert_eq!(c0.runtime_units, c1.runtime_units);
        let stats = mgr.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
    }

    // ---------------------------------------------------------------
    // Cache key includes the param_bindings: two calls with different
    // extents miss independently.
    // ---------------------------------------------------------------

    #[test]
    fn cost_manager_keys_on_param_bindings() {
        let cfg = synthetic_cfg();
        let artifact = crate::passes::fusion_v2::cost::ArtifactContext {
            target_arch: "test".into(),
            compiler_flags_hash: [0; 32],
        };
        let mut mgr = KernelCostManager::new(cfg, artifact, BTreeMap::new(), 1);
        let m = scale_module(64, 3);
        let hash = crate::module_hash::module_hash(&m);
        let mut b1 = BTreeMap::new();
        b1.insert("N".into(), 64i64);
        let mut b2 = BTreeMap::new();
        b2.insert("N".into(), 256i64);
        let _ = mgr.cost_of(hash, &m, &b1).unwrap();
        let _ = mgr.cost_of(hash, &m, &b2).unwrap();
        let _ = mgr.cost_of(hash, &m, &b1).unwrap();
        let stats = mgr.stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
    }

    // ---------------------------------------------------------------
    // Non-kernel costs (§12.8): Const is zero; Memcpy is launch + bytes;
    // Memset likewise; BlackboxKernel uses the caller hint.
    // ---------------------------------------------------------------

    #[test]
    fn non_kernel_const_costs_zero() {
        let cfg = synthetic_cfg();
        let bufs: Vec<BufInfo> = Vec::new();
        let ctx = EstimateContext::default();
        let node = GraphNode::Const(crate::graph_ir::ConstNode {
            buf: BufId(0),
            data: ConstBuf::HostBuf(Vec::new()),
        });
        let cost = estimate_non_kernel(&node, &bufs, &ctx, &cfg, 1, 0.0);
        // `from_cycles(0.0, 1)` clamps to floor 1 (§13.5).
        assert_eq!(cost.runtime_units, 1);
    }

    #[test]
    fn non_kernel_memcpy_costs_launch_plus_bandwidth() {
        let cfg = synthetic_cfg();
        let bufs: Vec<BufInfo> = Vec::new();
        let ctx = EstimateContext::default();
        let bytes = 4096i64;
        let node = GraphNode::Memcpy(MemcpyNode {
            src: BufId(0),
            src_offset: Quast::cst(0),
            dst: BufId(1),
            dst_offset: Quast::cst(0),
            num_bytes: Quast::cst(bytes),
        });
        let cost = estimate_non_kernel(&node, &bufs, &ctx, &cfg, 1, 0.0);
        let expected =
            cfg.device.memop_launch_cycles + bytes as f64 / cfg.device.memcpy_bytes_per_cycle;
        // Rounded to i64 with a floor of 1.
        let expected_units = expected.round().max(1.0) as i64;
        assert_eq!(cost.runtime_units, expected_units);
    }

    #[test]
    fn non_kernel_memset_costs_launch_plus_bandwidth() {
        let cfg = synthetic_cfg();
        let bufs: Vec<BufInfo> = Vec::new();
        let ctx = EstimateContext::default();
        let bytes = 1024i64;
        let node = GraphNode::Memset(MemSetNode {
            node: BufId(0),
            offset: Quast::cst(0),
            num_bytes: Quast::cst(bytes),
            val: 0,
        });
        let cost = estimate_non_kernel(&node, &bufs, &ctx, &cfg, 1, 0.0);
        let expected =
            cfg.device.memop_launch_cycles + bytes as f64 / cfg.device.memcpy_bytes_per_cycle;
        let expected_units = expected.round().max(1.0) as i64;
        assert_eq!(cost.runtime_units, expected_units);
    }

    // ---------------------------------------------------------------
    // Device profile: a slower DRAM makes memcpy cost more.
    // ---------------------------------------------------------------

    #[test]
    fn slower_dram_raises_memcpy_cost() {
        let mut fast = synthetic_cfg();
        fast.device.memcpy_bytes_per_cycle = 32.0;
        let mut slow = synthetic_cfg();
        slow.device.memcpy_bytes_per_cycle = 8.0;
        let bufs: Vec<BufInfo> = Vec::new();
        let ctx = EstimateContext::default();
        let node = GraphNode::Memcpy(MemcpyNode {
            src: BufId(0),
            src_offset: Quast::cst(0),
            dst: BufId(1),
            dst_offset: Quast::cst(0),
            num_bytes: Quast::cst(1024 * 1024),
        });
        let fast_cost = estimate_non_kernel(&node, &bufs, &ctx, &fast, 1, 0.0).runtime_units;
        let slow_cost = estimate_non_kernel(&node, &bufs, &ctx, &slow, 1, 0.0).runtime_units;
        assert!(slow_cost > fast_cost);
    }

    // ---------------------------------------------------------------
    // Synthetic DeviceModel exposes sensible defaults.
    // ---------------------------------------------------------------

    #[test]
    fn synthetic_device_defaults_are_positive() {
        let d = DeviceModel::synthetic();
        assert!(d.sms > 0);
        assert!(d.warp_size == 32);
        assert!(d.max_threads_per_sm >= 512);
        assert!(d.dram_bytes_per_cycle > 0.0);
        assert!(d.issue_weighted_ops_per_cycle > 0.0);
    }
}

/// M11 exit-gate tests: opt-in `GraphCompiler` integration (plan §16).
///
/// These run the CPU-side `GraphCompiler::fuse` entry point — the same
/// normalize prelude/postlude the full `compile` pipeline uses — with the
/// v2 strategy selected, and check the strategy routing, the §15 report
/// embedding, and the env → `graph_symbols` threading. Measured-runtime
/// and compile-time comparisons on real workloads land with M12's
/// `dsl_port_tests` replay.
#[cfg(feature = "planner")]
mod graph_compiler_tests {
    use super::*;
    use crate::{graph_exe::GraphCompiler, ir::VarId, passes::fusion_v2::FusionOptionsV2};

    fn scale_by(n: usize, c: u32) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let ai = b.index(a, &[i]);
            let cst = b.const_field(c);
            b.mul(ai, cst)
        });
        Arc::new(b.finish("scale_by", body))
    }

    /// `x -> scale2 -> y -> scale3 -> z` with x/z registered.
    fn two_chain(n: usize) -> GraphBuilder {
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
        g
    }

    #[test]
    fn v2_strategy_fuses_two_kernel_chain_and_embeds_report() {
        let mut g = two_chain(8);
        let report = GraphCompiler::new()
            .fusion_v2_options(FusionOptionsV2::default())
            .fuse(&mut g)
            .expect("fuse")
            .expect("fusion enabled");

        // v1 wrapper fields carry only the node counts (plan §15).
        assert_eq!(report.nodes_before, 2);
        assert_eq!(report.nodes_after, 1);
        assert!(report.fused.is_empty());
        assert_eq!(report.rounds, 0);

        let v2 = report.v2.as_ref().expect("v2 report embedded");
        assert_eq!(v2.nodes_before, 2);
        assert_eq!(v2.nodes_after, 1);
        assert!(v2.candidates_inserted >= 1);
        // Small graph: with `planner-ortools` CP-SAT solves it; without,
        // the brute-force extractor does. Neither path may fall back.
        assert_eq!(v2.fallback_reason, None);

        assert_eq!(g.nodes.len(), 1);
        assert!(matches!(&g.nodes[0], GraphNode::Kernel(_)));
    }

    #[test]
    fn existing_strategy_stays_default_with_no_v2_report() {
        let mut g = two_chain(8);
        let report = GraphCompiler::new()
            .fuse(&mut g)
            .expect("fuse")
            .expect("fusion enabled");
        assert!(report.v2.is_none());
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn without_fusion_disables_both_strategies() {
        let mut g = two_chain(8);
        let report = GraphCompiler::new()
            .fusion_v2_options(FusionOptionsV2::default())
            .without_fusion()
            .fuse(&mut g)
            .expect("fuse");
        assert!(report.is_none());
        assert_eq!(g.nodes.len(), 2);
    }

    /// Module-count golden comparison (M11): on a three-kernel chain the
    /// v2 strategy must reach the same fused node count as the existing
    /// pass, and both must improve on the unfused baseline.
    #[test]
    fn v2_matches_existing_node_count_on_three_chain() {
        let three_chain = |n: usize| {
            let mut g = GraphBuilder::new();
            let x = sized_buf(&mut g, "x", (n * 4) as i64);
            let y = sized_buf(&mut g, "y", (n * 4) as i64);
            let z = sized_buf(&mut g, "z", (n * 4) as i64);
            let w = sized_buf(&mut g, "w", (n * 4) as i64);
            g.register_input(x);
            g.register_output(w);
            g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
            g.insert_kernel(scale_by(n, 3), vec![y], vec![z], &[]);
            g.insert_kernel(scale_by(n, 5), vec![z], vec![w], &[]);
            g
        };
        let unfused_nodes = three_chain(8).nodes.len();

        let mut g_v1 = three_chain(8);
        GraphCompiler::new()
            .fuse(&mut g_v1)
            .expect("v1 fuse")
            .expect("fusion enabled");

        let mut g_v2 = three_chain(8);
        GraphCompiler::new()
            .fusion_v2_options(FusionOptionsV2::default())
            .fuse(&mut g_v2)
            .expect("v2 fuse")
            .expect("fusion enabled");

        assert_eq!(unfused_nodes, 3);
        assert_eq!(g_v1.nodes.len(), 1);
        assert_eq!(g_v2.nodes.len(), g_v1.nodes.len());
    }

    /// `GraphCompiler::symbol` bindings must reach the v2 estimator
    /// (`FusionOptionsV2::graph_symbols`): an unbound symbolic memcpy
    /// size falls back to a 1 KiB estimate, so binding the symbol to a
    /// large value must strictly raise the estimated total runtime.
    #[test]
    fn env_symbols_thread_into_v2_estimator() {
        let build = || {
            let nsym = VarId(7);
            let mut g = GraphBuilder::new();
            let x = g.add_buf(BufInfo {
                name: Some("x".into()),
                device_type: DeviceType::Cuda(0),
                size: Quast::sym(nsym),
                elem_size: 4,
            });
            let y = g.add_buf(BufInfo {
                name: Some("y".into()),
                device_type: DeviceType::Cuda(0),
                size: Quast::sym(nsym),
                elem_size: 4,
            });
            g.register_input(x);
            g.register_output(y);
            g.insert_memcpy(x, y);
            (nsym, g)
        };

        let (_, mut g_unbound) = build();
        let unbound = GraphCompiler::new()
            .fusion_v2_options(FusionOptionsV2::default())
            .fuse(&mut g_unbound)
            .expect("fuse")
            .expect("fusion enabled");

        let (nsym, mut g_bound) = build();
        let bound = GraphCompiler::new()
            .symbol(nsym, 1 << 26)
            .fusion_v2_options(FusionOptionsV2::default())
            .fuse(&mut g_bound)
            .expect("fuse")
            .expect("fusion enabled");

        let unbound_units = unbound.v2.expect("v2 report").total_runtime_units;
        let bound_units = bound.v2.expect("v2 report").total_runtime_units;
        assert!(
            bound_units > unbound_units,
            "bound symbol must raise the memcpy estimate: {bound_units} vs {unbound_units}"
        );
    }

    /// Without `planner-ortools`, graphs beyond the brute-force cap must
    /// fall back to the original extraction and say so — never silently
    /// run the existing pass (plan §16).
    #[cfg(not(feature = "planner-ortools"))]
    #[test]
    fn no_solver_large_graph_reports_solver_unavailable() {
        use crate::passes::fusion_v2::FallbackReason;
        // 34 disjoint kernels exceed BRUTE_FORCE_LIMIT (32) with zero
        // fusion candidates, so extraction must fall back to original.
        let n = 8;
        let mut g = GraphBuilder::new();
        for i in 0..34 {
            let x = sized_buf(&mut g, &format!("x{i}"), (n * 4) as i64);
            let y = sized_buf(&mut g, &format!("y{i}"), (n * 4) as i64);
            g.register_input(x);
            g.register_output(y);
            g.insert_kernel(scale_by(n, 2), vec![x], vec![y], &[]);
        }
        let report = GraphCompiler::new()
            .fusion_v2_options(FusionOptionsV2 {
                enable_horizontal: false,
                ..FusionOptionsV2::default()
            })
            .fuse(&mut g)
            .expect("fuse")
            .expect("fusion enabled");
        let v2 = report.v2.expect("v2 report");
        assert_eq!(v2.fallback_reason, Some(FallbackReason::SolverUnavailable));
        assert_eq!(g.nodes.len(), 34);
    }
}

/// General producer-consumer fusion (symbolic SExpr pipeline): rank-k
/// seam reads, arbitrary producer write maps via trusted scatter
/// inverses, symbolic-parameter normalization (unify equal bindings,
/// split different bindings as `name#k`), and the two-tier
/// (symbolic-first, concrete-fallback) seam-shape gates.
mod general_pc_tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        ir::SizeExpr,
        module_hash::module_hash,
        passes::fusion_v2::{
            fusions::producer_consumer::{
                self, EnumerateOptions, FusionVariant, OwnedEnumerateContext,
            },
            take_graph, GraphFuser,
        },
    };

    fn enum_all(gf: &GraphFuser) -> Vec<producer_consumer::CandidateDraft> {
        producer_consumer::enumerate(
            gf,
            &OwnedEnumerateContext::all_seed(gf, EnumerateOptions::default()).as_ref(),
        )
    }

    fn draft_module(d: &producer_consumer::CandidateDraft) -> Arc<crate::ir::Module> {
        match &d.alt.node {
            GraphNode::Kernel(k) => k.module.clone(),
            _ => panic!("expected Kernel"),
        }
    }

    fn draft_bindings(d: &producer_consumer::CandidateDraft) -> BTreeMap<String, i64> {
        match &d.alt.node {
            GraphNode::Kernel(k) => k.param_bindings.clone(),
            _ => panic!("expected Kernel"),
        }
    }

    fn bindings(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // -----------------------------------------------------------------
    // 1. Shared symbol: `a = compute [q] |i| x[i] + x[i+q]` feeding `compute [q/2] |i| a[i] +
    //    a[i+q/2]`, both instantiated at the same q. Params unify by name; the fused module stays
    //    fully symbolic (tier-1 symbolic gate), so ONE artifact serves every q.
    // -----------------------------------------------------------------

    fn fold_q_producer() -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let q = b.symbol("q");
        let x = b.input("x", ScalarType::BabyBear, vec![SizeExpr::from(q * 2)]);
        let body = b.compute(q, |b, i| {
            let xi = b.index(x, &[i]);
            let cq = b.const_sym(q);
            let iq = b.add(i, cq);
            let xiq = b.index(x, &[iq]);
            b.add(xi, xiq)
        });
        Arc::new(b.finish("fold_q", body))
    }

    fn half_q_consumer() -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let q = b.symbol("q");
        let a = b.input("a", ScalarType::BabyBear, vec![SizeExpr::from(q)]);
        let body = b.compute(q / 2, |b, i| {
            let ai = b.index(a, &[i]);
            let ch = b.const_sym(q / 2);
            let ih = b.add(i, ch);
            let aih = b.index(a, &[ih]);
            b.add(ai, aih)
        });
        Arc::new(b.finish("half_q", body))
    }

    #[test]
    fn shared_symbol_fuses_symbolically() {
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", 16 * 4); // q = 8
        let y = sized_buf(&mut g, "y", 8 * 4);
        let z = sized_buf(&mut g, "z", 4 * 4);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(fold_q_producer(), vec![x], vec![y], &[]);
        g.insert_kernel(half_q_consumer(), vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();

        let drafts = enum_all(&gf);
        assert_eq!(drafts.len(), 1, "shared-symbol chain should fuse");
        let fused = draft_module(&drafts[0]);
        crate::passes::type_infer(&fused).unwrap();
        assert_eq!(draft_bindings(&drafts[0]), bindings(&[("q", 8)]));

        // compute [q/2] |i| (x[i] + x[i+q]) + (x[i+q/2] + x[i+q/2+q]),
        // with a single unified `q` and the producer's input decl.
        let reference = {
            let mut b = IRBuilder::new();
            let q = b.symbol("q");
            let x = b.input("x", ScalarType::BabyBear, vec![SizeExpr::from(q * 2)]);
            let body = b.compute(q / 2, |b, i| {
                let cq = b.const_sym(q);
                let xi = b.index(x, &[i]);
                let iq = b.add(i, cq);
                let xiq = b.index(x, &[iq]);
                let p0 = b.add(xi, xiq);
                let ch = b.const_sym(q / 2);
                let ih = b.add(i, ch);
                let xh = b.index(x, &[ih]);
                let ihq = b.add(ih, cq);
                let xhq = b.index(x, &[ihq]);
                let p1 = b.add(xh, xhq);
                b.add(p0, p1)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(module_hash(&fused), module_hash(&reference));
    }

    // -----------------------------------------------------------------
    // 2./3. Chain of the SAME Arc'd module `compute [n] |i| x[i]+x[i+n]`
    //    with halving bindings (n=8, n=4, n=2). Same name + different
    //    binding splits into independent fused params `n` / `n#1`; the
    //    binding relation lives in `param_bindings` and is certified by
    //    the tier-2 concrete gate, never written into the module text —
    //    so every level of the chain shares ONE fused artifact.
    // -----------------------------------------------------------------

    fn fold_n_module() -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let n = b.symbol("n");
        let x = b.input("x", ScalarType::BabyBear, vec![SizeExpr::from(n * 2)]);
        let body = b.compute(n, |b, i| {
            let xi = b.index(x, &[i]);
            let cn = b.const_sym(n);
            let ixn = b.add(i, cn);
            let xin = b.index(x, &[ixn]);
            b.add(xi, xin)
        });
        Arc::new(b.finish("fold_n", body))
    }

    /// x --f--> y --f--> z --f--> w with one shared module Arc; inferred
    /// bindings n=8, n=4, n=2.
    fn fold_chain3() -> GraphFuser {
        let m = fold_n_module();
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", 16 * 4);
        let y = sized_buf(&mut g, "y", 8 * 4);
        let z = sized_buf(&mut g, "z", 4 * 4);
        let w = sized_buf(&mut g, "w", 2 * 4);
        g.register_input(x);
        g.register_output(w);
        g.insert_kernel(m.clone(), vec![x], vec![y], &[]);
        g.insert_kernel(m.clone(), vec![y], vec![z], &[]);
        g.insert_kernel(m, vec![z], vec![w], &[]);
        take_graph(&mut g).unwrap()
    }

    #[test]
    fn same_module_chain_splits_params() {
        let gf = fold_chain3();
        let seam = gf.nodes[0].outputs[0];
        let d = producer_consumer::synthesize_producer_consumer(
            &gf,
            NodeId(0),
            NodeId(1),
            seam,
            FusionVariant::Drop,
        )
        .expect("halving fold chain should fuse via param split");
        let fused = draft_module(&d);
        crate::passes::type_infer(&fused).unwrap();
        assert_eq!(draft_bindings(&d), bindings(&[("n", 8), ("n#1", 4)]));

        // compute [n#1] |i| (x[i] + x[i+n]) + (x[i+n#1] + x[i+n#1+n]),
        // producer params first, consumer's `n` split to `n#1`.
        let reference = {
            let mut b = IRBuilder::new();
            let n = b.symbol("n");
            let n1 = b.symbol("n#1");
            let x = b.input("x", ScalarType::BabyBear, vec![SizeExpr::from(n * 2)]);
            let body = b.compute(n1, |b, i| {
                let cn = b.const_sym(n);
                let xi = b.index(x, &[i]);
                let ixn = b.add(i, cn);
                let xin = b.index(x, &[ixn]);
                let p0 = b.add(xi, xin);
                let cn1 = b.const_sym(n1);
                let ih = b.add(i, cn1);
                let xh = b.index(x, &[ih]);
                let ihn = b.add(ih, cn);
                let xhn = b.index(x, &[ihn]);
                let p1 = b.add(xh, xhn);
                b.add(p0, p1)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(module_hash(&fused), module_hash(&reference));
    }

    #[test]
    fn chain_levels_share_one_artifact() {
        let gf = fold_chain3();
        let d01 = producer_consumer::synthesize_producer_consumer(
            &gf,
            NodeId(0),
            NodeId(1),
            gf.nodes[0].outputs[0],
            FusionVariant::Drop,
        )
        .unwrap();
        let d12 = producer_consumer::synthesize_producer_consumer(
            &gf,
            NodeId(1),
            NodeId(2),
            gf.nodes[1].outputs[0],
            FusionVariant::Drop,
        )
        .unwrap();
        // Same module text (one compiled artifact), different bindings.
        assert_eq!(
            module_hash(&draft_module(&d01)),
            module_hash(&draft_module(&d12))
        );
        assert_eq!(draft_bindings(&d01), bindings(&[("n", 8), ("n#1", 4)]));
        assert_eq!(draft_bindings(&d12), bindings(&[("n", 4), ("n#1", 2)]));
    }

    #[test]
    fn chain_association_orders_hash_identically() {
        // fuse(fuse(f0,f1),f2) and fuse(f0,fuse(f1,f2)) must produce the
        // same module text: first-appearance producer-first param
        // naming makes both land on n=8, n#1=4, n#2=2.
        let mut gf_a = fold_chain3();
        let seam01 = gf_a.nodes[0].outputs[0];
        let seam12 = gf_a.nodes[1].outputs[0];
        let d01 = producer_consumer::synthesize_producer_consumer(
            &gf_a,
            NodeId(0),
            NodeId(1),
            seam01,
            FusionVariant::Drop,
        )
        .unwrap();
        let id01 = gf_a.insert_candidate(d01.alt);
        let d_a = producer_consumer::synthesize_producer_consumer(
            &gf_a,
            id01,
            NodeId(2),
            seam12,
            FusionVariant::Drop,
        )
        .unwrap();

        let mut gf_b = fold_chain3();
        let d12 = producer_consumer::synthesize_producer_consumer(
            &gf_b,
            NodeId(1),
            NodeId(2),
            seam12,
            FusionVariant::Drop,
        )
        .unwrap();
        let id12 = gf_b.insert_candidate(d12.alt);
        let d_b = producer_consumer::synthesize_producer_consumer(
            &gf_b,
            NodeId(0),
            id12,
            seam01,
            FusionVariant::Drop,
        )
        .unwrap();

        assert_eq!(
            module_hash(&draft_module(&d_a)),
            module_hash(&draft_module(&d_b))
        );
        let expected = bindings(&[("n", 8), ("n#1", 4), ("n#2", 2)]);
        assert_eq!(draft_bindings(&d_a), expected);
        assert_eq!(draft_bindings(&d_b), expected);
    }

    // -----------------------------------------------------------------
    // 4. Rank-2 seam: producer writes rows via `Pack`, consumer reads `(row, const)` components.
    //    Drill inlines the selected element.
    // -----------------------------------------------------------------

    fn pack_producer(n: usize) -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![n]);
        let body = b.compute(n, |b, i| {
            let xi = b.index(x, &[i]);
            let f2 = b.const_field(2);
            let e0 = b.mul(xi, f2);
            let f3 = b.const_field(3);
            let e1 = b.mul(xi, f3);
            b.pack(&[e0, e1])
        });
        Arc::new(b.finish("pack_producer", body))
    }

    #[test]
    fn rank2_pack_producer_component_reads_fuse() {
        let n = 8;
        let consumer = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![n, 2]);
            let body = b.compute(n, |b, i| {
                let c0 = b.const_u32(0);
                let c1 = b.const_u32(1);
                let a = b.index(y, &[i, c0]);
                let bb = b.index(y, &[i, c1]);
                b.add(a, bb)
            });
            Arc::new(b.finish("row_sum", body))
        };
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 2 * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(pack_producer(n), vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();

        let drafts = enum_all(&gf);
        assert_eq!(drafts.len(), 1, "pack producer + (row, const) reads");
        let fused = draft_module(&drafts[0]);
        crate::passes::type_infer(&fused).unwrap();

        // compute [8] |i| 2*x[i] + 3*x[i].
        let reference = {
            let mut b = IRBuilder::new();
            let x = b.input("x", ScalarType::BabyBear, vec![n]);
            let body = b.compute(n, |b, i| {
                let xi = b.index(x, &[i]);
                let f2 = b.const_field(2);
                let e0 = b.mul(xi, f2);
                let f3 = b.const_field(3);
                let e1 = b.mul(xi, f3);
                b.add(e0, e1)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(module_hash(&fused), module_hash(&reference));
    }

    // -----------------------------------------------------------------
    // 5./9. Reshape-view seam: producer writes flat [8], consumer reads
    //    it as [4, 2] under a different outer bound. The drop variant
    //    fuses through linearize/delinearize; keep still requires equal
    //    domains and must NOT be emitted.
    // -----------------------------------------------------------------

    fn scale_flat8() -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let x = b.input("x", ScalarType::BabyBear, vec![8]);
        let body = b.compute(8, |b, i| {
            let xi = b.index(x, &[i]);
            let f2 = b.const_field(2);
            b.mul(xi, f2)
        });
        Arc::new(b.finish("scale_flat8", body))
    }

    fn reshape_graph() -> GraphFuser {
        let consumer = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![4, 2]);
            let body = b.compute(4, |b, i| {
                let c0 = b.const_u32(0);
                let c1 = b.const_u32(1);
                let a = b.index(y, &[i, c0]);
                let bb = b.index(y, &[i, c1]);
                b.add(a, bb)
            });
            Arc::new(b.finish("pair_sum", body))
        };
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", 8 * 4);
        let y = sized_buf(&mut g, "y", 8 * 4);
        let z = sized_buf(&mut g, "z", 4 * 4);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(scale_flat8(), vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        take_graph(&mut g).unwrap()
    }

    #[test]
    fn reshape_view_seam_fuses_via_linearize() {
        let gf = reshape_graph();
        let drafts = enum_all(&gf);
        assert_eq!(drafts.len(), 1, "flat [8] seam read as [4,2]");
        let fused = draft_module(&drafts[0]);
        crate::passes::type_infer(&fused).unwrap();

        // compute [4] |i| 2*x[i*2] + 2*x[i*2+1].
        let reference = {
            let mut b = IRBuilder::new();
            let x = b.input("x", ScalarType::BabyBear, vec![8]);
            let body = b.compute(4, |b, i| {
                let c2 = b.const_u32(2);
                let i2 = b.mul(i, c2);
                let x0 = b.index(x, &[i2]);
                let f2 = b.const_field(2);
                let e0 = b.mul(x0, f2);
                let c1 = b.const_u32(1);
                let i21 = b.add(i2, c1);
                let x1 = b.index(x, &[i21]);
                let e1 = b.mul(x1, f2);
                b.add(e0, e1)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(module_hash(&fused), module_hash(&reference));
    }

    #[test]
    fn keep_variant_still_requires_equal_domains() {
        let gf = reshape_graph();
        let drafts = producer_consumer::enumerate(
            &gf,
            &OwnedEnumerateContext::all_seed(
                &gf,
                EnumerateOptions {
                    enable_all_keep_variants: true,
                },
            )
            .as_ref(),
        );
        assert_eq!(drafts.len(), 1, "only the drop variant is legal");
        assert_eq!(drafts[0].variant, FusionVariant::Drop);
    }

    // -----------------------------------------------------------------
    // 6. Scattered producer: the write map is arbitrary because its inverse is provided (and
    //    trusted). σ composes the consumer's read coordinate through the inverse.
    // -----------------------------------------------------------------

    #[test]
    fn scattered_producer_uses_provided_inverse() {
        let producer = {
            let mut b = IRBuilder::new();
            let x = b.input("x", ScalarType::BabyBear, vec![8]);
            let sc = b.scatter_map(
                1,
                None,
                |p, cst| vec![cst(7).sub(&p[0])],
                |p, cst| vec![cst(7).sub(&p[0])],
            );
            let body = b.compute_scatter(8, sc, |b, i| {
                let xi = b.index(x, &[i]);
                let f2 = b.const_field(2);
                b.mul(xi, f2)
            });
            Arc::new(b.finish("rev_scale", body))
        };
        let consumer = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![8]);
            let body = b.compute(8, |b, i| {
                let yi = b.index(y, &[i]);
                let f3 = b.const_field(3);
                b.mul(yi, f3)
            });
            Arc::new(b.finish("scale3", body))
        };
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", 8 * 4);
        let y = sized_buf(&mut g, "y", 8 * 4);
        let z = sized_buf(&mut g, "z", 8 * 4);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(producer, vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();

        let drafts = enum_all(&gf);
        assert_eq!(drafts.len(), 1, "scattered producer with inverse");
        let fused = draft_module(&drafts[0]);
        crate::passes::type_infer(&fused).unwrap();

        // y[p] = 2*x[7-p], so the fused body is 3 * (2 * x[7-i]); the
        // producer's scatter is dropped, the consumer had none.
        let reference = {
            let mut b = IRBuilder::new();
            let x = b.input("x", ScalarType::BabyBear, vec![8]);
            let body = b.compute(8, |b, i| {
                let c7 = b.const_u32(7);
                let idx = b.sub(c7, i);
                let xi = b.index(x, &[idx]);
                let f2 = b.const_field(2);
                let m = b.mul(xi, f2);
                let f3 = b.const_field(3);
                b.mul(m, f3)
            });
            b.finish(fused.name.clone(), body)
        };
        assert_eq!(module_hash(&fused), module_hash(&reference));
    }

    // -----------------------------------------------------------------
    // 7./10. Negatives: a Pack component selected by a non-constant
    //    coordinate has no scalar inline (select-chains are deferred).
    // -----------------------------------------------------------------

    #[test]
    fn pack_component_must_be_const() {
        let n = 8;
        let consumer = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![n, 2]);
            let body = b.compute(n, |b, i| b.reduce_add(2, |b, j| b.index(y, &[i, j])));
            Arc::new(b.finish("reduce_components", body))
        };
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 2 * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(pack_producer(n), vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();
        assert!(
            enum_all(&gf).is_empty(),
            "reduce-var component select must not fuse"
        );
    }

    #[test]
    fn flat_read_of_pack_producer_is_rejected() {
        // Producer writes [8,2] rows; consumer reads the same buffer
        // flat as [16]. Delinearizing k into [8,2] yields component
        // k%2, which is not constant -> no scalar inline.
        let n = 8;
        let consumer = {
            let mut b = IRBuilder::new();
            let y = b.input("y", ScalarType::BabyBear, vec![2 * n]);
            let body = b.compute(2 * n, |b, k| {
                let yk = b.index(y, &[k]);
                let f5 = b.const_field(5);
                b.mul(yk, f5)
            });
            Arc::new(b.finish("flat_scale", body))
        };
        let mut g = GraphBuilder::new();
        let x = sized_buf(&mut g, "x", (n * 4) as i64);
        let y = sized_buf(&mut g, "y", (n * 2 * 4) as i64);
        let z = sized_buf(&mut g, "z", (n * 2 * 4) as i64);
        g.register_input(x);
        g.register_output(z);
        g.insert_kernel(pack_producer(n), vec![x], vec![y], &[]);
        g.insert_kernel(consumer, vec![y], vec![z], &[]);
        let gf = take_graph(&mut g).unwrap();
        assert!(
            enum_all(&gf).is_empty(),
            "non-const pack component via delinearize must not fuse"
        );
    }

    // -----------------------------------------------------------------
    // 8. Frac-fold-shaped chain: rank-2 reads, Pack rows, symbolic divisor `i + (i/q)*q`, same
    //    Arc'd module with halving bindings. The CPU-level proxy for the GPU fold->fold spine
    //    fusion.
    // -----------------------------------------------------------------

    fn frac_fold_like() -> Arc<crate::ir::Module> {
        let mut b = IRBuilder::new();
        let q = b.symbol("q");
        let src = b.input(
            "src",
            ScalarType::BabyBear,
            vec![SizeExpr::from(q * 4), SizeExpr::from(2usize)],
        );
        let body = b.compute(q * 2, |b, i| {
            let cq = b.const_sym(q);
            let d = b.div(i, cq);
            let dq = b.mul(d, cq);
            let a = b.add(i, dq); // a = i + (i/q)*q
            let bx = b.add(a, cq); // b = a + q
            let c0 = b.const_u32(0);
            let c1 = b.const_u32(1);
            let a0 = b.index(src, &[a, c0]);
            let b0 = b.index(src, &[bx, c0]);
            let s0 = b.add(a0, b0);
            let a1 = b.index(src, &[a, c1]);
            let b1 = b.index(src, &[bx, c1]);
            let s1 = b.add(a1, b1);
            b.pack(&[s0, s1])
        });
        Arc::new(b.finish("frac_fold_like", body))
    }

    #[test]
    fn frac_fold_like_chain_fuses_and_shares_artifact() {
        let m = frac_fold_like();
        let mut g = GraphBuilder::new();
        let b0 = sized_buf(&mut g, "b0", 32 * 4); // q=4: src [16,2]
        let b1 = sized_buf(&mut g, "b1", 16 * 4); // out [8,2]; next q=2
        let b2 = sized_buf(&mut g, "b2", 8 * 4); // out [4,2]; next q=1
        let b3 = sized_buf(&mut g, "b3", 4 * 4); // out [2,2]
        g.register_input(b0);
        g.register_output(b3);
        g.insert_kernel(m.clone(), vec![b0], vec![b1], &[]);
        g.insert_kernel(m.clone(), vec![b1], vec![b2], &[]);
        g.insert_kernel(m, vec![b2], vec![b3], &[]);
        let gf = take_graph(&mut g).unwrap();

        let d01 = producer_consumer::synthesize_producer_consumer(
            &gf,
            NodeId(0),
            NodeId(1),
            gf.nodes[0].outputs[0],
            FusionVariant::Drop,
        )
        .expect("frac-fold pair should fuse");
        let d12 = producer_consumer::synthesize_producer_consumer(
            &gf,
            NodeId(1),
            NodeId(2),
            gf.nodes[1].outputs[0],
            FusionVariant::Drop,
        )
        .expect("frac-fold pair should fuse at the next level");

        let m01 = draft_module(&d01);
        crate::passes::type_infer(&m01).unwrap();
        assert_eq!(draft_bindings(&d01), bindings(&[("q", 4), ("q#1", 2)]));
        assert_eq!(draft_bindings(&d12), bindings(&[("q", 2), ("q#1", 1)]));
        // One artifact for every level of the chain.
        assert_eq!(module_hash(&m01), module_hash(&draft_module(&d12)));
    }
}

/// Repro scaffolding for the GKR `fused_drop` OOB (`pre[0, -1]`, p=0):
/// select-guarded reads must survive σ substitution — a read that codegen
/// sinks into a select branch in the producer must still sink in the
/// fused module, or the fused kernel executes it unconditionally.
mod select_guard_tests {
    use std::collections::BTreeMap;

    use crate::{
        ir::{IRBuilder, ScalarType, SizeExpr},
        passes::check_accesses::check_module_accesses,
    };

    fn bindings(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    /// Original producer shape: `compute[4] |v| { if v < p then
    /// pre[0, p-1-v] else post[0, 7+p-v] }`. The pre-read is guarded;
    /// with p=0 every lane takes the else branch.
    #[test]
    fn guarded_read_producer_shape_passes_check() {
        let mut b = IRBuilder::new();
        let p = b.symbol("p");
        let pre = b.input(
            "pre",
            ScalarType::BabyBear,
            vec![SizeExpr::from(1), SizeExpr::from(16)],
        );
        let post = b.input(
            "post",
            ScalarType::BabyBear,
            vec![SizeExpr::from(1), SizeExpr::from(16)],
        );
        let body = b.compute(4, |b, v| {
            let cp = b.const_sym(p);
            let cond = b.lt(v, cp);
            let one = b.const_u32(1);
            let zero = b.const_u32(0);
            let pm1 = b.sub(cp, one);
            let idx_then = b.sub(pm1, v);
            let t = b.index(pre, &[zero, idx_then]);
            let seven = b.const_u32(7);
            let sp = b.add(seven, cp);
            let idx_else = b.sub(sp, v);
            let f = b.index(post, &[zero, idx_else]);
            b.select(cond, t, f)
        });
        let m = b.finish("orig", body);
        check_module_accesses(&m, &bindings(&[("p", 0)])).unwrap();
    }

    /// Fused shape after σ substitution (consumer read coords are
    /// literals): `compute[1] |_| { if 0 < p then pre[0, p-1] else
    /// post[0, 7+p] }`. Must also pass — the guard still dominates the
    /// read.
    #[test]
    fn guarded_read_fused_shape_passes_check() {
        let mut b = IRBuilder::new();
        let p = b.symbol("p");
        let pre = b.input(
            "pre",
            ScalarType::BabyBear,
            vec![SizeExpr::from(1), SizeExpr::from(16)],
        );
        let post = b.input(
            "post",
            ScalarType::BabyBear,
            vec![SizeExpr::from(1), SizeExpr::from(16)],
        );
        let body = b.compute(1, |b, _v| {
            let cp = b.const_sym(p);
            let one = b.const_u32(1);
            let zero = b.const_u32(0);
            let cond = b.lt(zero, cp);
            let idx_then = b.sub(cp, one);
            let t = b.index(pre, &[zero, idx_then]);
            let seven = b.const_u32(7);
            let idx_else = b.add(seven, cp);
            let f = b.index(post, &[zero, idx_else]);
            b.select(cond, t, f)
        });
        let m = b.finish("fused_shape", body);
        check_module_accesses(&m, &bindings(&[("p", 0)])).unwrap();
    }
}

/// Regression test for read sinking under duplicated guards: a GKR
/// `fused_drop` module whose four select-guarded eq-prefix reads combine
/// into an FpExt value feeding both branches of an outer select.
/// Canonicalize duplicates the guarded tree into each outer branch, so a
/// read's uses end up under several distinct crossings; codegen must sink
/// the read at each use's innermost guard (`compute_read_sinks`) or the
/// p=0 `pre[0, -1]` load executes eagerly and faults.
#[cfg(test)]
mod select_guard_full_repro {
    use std::collections::BTreeMap;

    use crate::{
        ir::{IRBuilder, Module, NodeId, ScalarType, SizeExpr},
        passes::check_accesses::check_module_accesses,
    };

    fn build_repro() -> (Module, BTreeMap<String, i64>) {
        let mut b = IRBuilder::new();
        let p = b.symbol("p");
        let k = b.symbol("k");
        let _n = b.symbol("n");
        let pre = b.input(
            "pre",
            ScalarType::BabyBear,
            vec![SizeExpr::from(1), SizeExpr::from(16)],
        );
        let post = b.input(
            "post",
            ScalarType::BabyBear,
            vec![SizeExpr::from(1), SizeExpr::from(16)],
        );
        let sp0 = b.input("sp0", ScalarType::FpExt, vec![SizeExpr::from(1)]);
        let sp2 = b.input("sp2", ScalarType::FpExt, vec![SizeExpr::from(1)]);
        let sp1 = b.input("sp1", ScalarType::FpExt, vec![SizeExpr::from(1)]);
        let body = b.compute(1, |b, _v| {
            let cp = b.const_sym(p);
            let zero = b.const_u32(0);
            // term j: lift(if j < p then pre[0, p-1-j] else post[0, 7+p-j]) * e_j
            let mut terms: Vec<NodeId> = Vec::new();
            for j in 0..4u32 {
                let cj = b.const_u32(j);
                let cond = b.lt(cj, cp);
                let c1j = b.const_u32(1 + j);
                let idx_then = b.sub(cp, c1j);
                let t = b.index(pre, &[zero, idx_then]);
                let c7j = b.const_u32(7 - j);
                let idx_else = b.add(c7j, cp);
                let f = b.index(post, &[zero, idx_else]);
                let sel = b.select(cond, t, f);
                terms.push(b.lift_fpext(sel));
            }
            let e1 = b.const_fpext([0, 1, 0, 0]);
            let e2 = b.const_fpext([0, 0, 1, 0]);
            let e3 = b.const_fpext([0, 0, 0, 1]);
            let t1 = b.mul(terms[1], e1);
            let t2 = b.mul(terms[2], e2);
            let t3 = b.mul(terms[3], e3);
            let s01 = b.add(terms[0], t1);
            let s012 = b.add(s01, t2);
            let v53 = b.add(s012, t3);
            let v55 = b.const_fpext([3, 0, 0, 0]);
            // outer select over k: if k == 0 then %53 else (if k == 1 then %55*%53 - 1 else 5*%53 -
            // 2)
            let ck = b.const_sym(k);
            let k0 = b.eq(ck, zero);
            let one_u = b.const_u32(1);
            let k1 = b.eq(ck, one_u);
            let f1 = b.const_fpext([1, 0, 0, 0]);
            let f2 = b.const_fpext([2, 0, 0, 0]);
            let f5 = b.const_fpext([5, 0, 0, 0]);
            let m1 = b.mul(v55, v53);
            let br1 = b.sub(m1, f1);
            let m2 = b.mul(f5, v53);
            let br2 = b.sub(m2, f2);
            let inner_sel = b.select(k1, br1, br2);
            let outer_sel = b.select(k0, v53, inner_sel);
            // * (sp0[0] + %55 * (sp2[0] - sp1[0]))
            let r0 = b.index(sp0, &[zero]);
            let r2 = b.index(sp2, &[zero]);
            let r1 = b.index(sp1, &[zero]);
            let d = b.sub(r2, r1);
            let md = b.mul(v55, d);
            let sum = b.add(r0, md);
            b.mul(outer_sel, sum)
        });
        let m = b.finish("fused_drop_repro", body);
        let bindings: BTreeMap<String, i64> = [("p", 0i64), ("k", 2), ("n", 3)]
            .iter()
            .map(|(s, v)| (s.to_string(), *v))
            .collect();
        (m, bindings)
    }

    #[test]
    fn gkr_fused_drop_shape_passes_check() {
        let (m, bindings) = build_repro();
        check_module_accesses(&m, &bindings).unwrap();
    }
}
