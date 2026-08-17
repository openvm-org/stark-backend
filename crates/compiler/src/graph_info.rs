//! Per-node timing snapshot of a compiled [`crate::graph_exe::GraphExe`].
//!
//! Produced by [`crate::graph_exe::GraphExe::collect_graph_info`] and
//! keyed by the source graph's `graph_hash` so a serialized snapshot can
//! be paired with the exe it came from. All times are in **milliseconds**.
//! Per-node cost is measured with CUDA events (`cudaEventElapsedTime`);
//! total per-iteration cost is measured with host wall clock. Both are
//! reported as sample mean plus sample standard deviation with the
//! `(n - 1)` (Bessel-corrected) denominator.

use serde::{Deserialize, Serialize};

/// One-byte tag mapping to an [`crate::graph_exe::ExeNode`] variant.
/// The tag is the only per-node structural information kept in a
/// [`GraphInfo`] snapshot; the rest is timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Kernel,
    Blackbox,
    Const,
    Memcpy,
    Memset,
}

/// Per-node aggregated cost across `GraphInfo::num_iters` iterations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeTiming {
    /// Node variant. Enough to attribute the cost to a kind of work
    /// without carrying the full compiled kernel or closure back to a
    /// reader-side consumer.
    pub kind: NodeKind,
    /// Human-readable label. Kernel and blackbox nodes carry their
    /// registered `name`; memcpy / memset / const use a stable short
    /// description like `"memcpy"`.
    pub name: String,
    /// Sample mean cost, in milliseconds.
    pub mean_ms: f64,
    /// Sample standard deviation, in milliseconds. Divides sum of
    /// squared deviations by `(n - 1)` (`0.0` when `n < 2`).
    pub std_ms: f64,
}

/// Timing snapshot for a whole graph. Serializable so callers can dump
/// it alongside the compiled `GraphExe` (see
/// [`crate::graph_serializer`]) for offline comparison across compiler
/// changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphInfo {
    /// Fingerprint of the source [`crate::graph_ir::GraphBuilder`]; see
    /// [`crate::graph_ir::GraphBuilder::content_hash`]. Matches
    /// [`crate::graph_exe::GraphExe::graph_hash`] on the exe this
    /// snapshot was collected from.
    pub graph_hash: [u8; 32],
    pub num_warmup: usize,
    pub num_iters: usize,
    /// One entry per exe node, indexed by the exe-node position — which
    /// is also the position in the source `GraphBuilder.nodes` at
    /// compile-consume time. So a cytoscape dump of that same graph
    /// (post-fusion, pre-DCE) can attach each `NodeTiming` directly by
    /// index.
    pub nodes: Vec<NodeTiming>,
    /// Host-side wall-clock cost per iteration, aggregated. Includes
    /// per-node event record overhead and the final stream sync; useful
    /// as an upper bound on the sum of per-node means.
    pub total_ms_mean: f64,
    pub total_ms_std: f64,
}

/// Sample mean and sample standard deviation. Uses Bessel's correction
/// (denominator `n - 1`); returns `0.0` std when `n < 2`.
pub(crate) fn mean_and_sample_std(samples: &[f64]) -> (f64, f64) {
    let n = samples.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let var = samples
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64 - 1.0);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_zeros() {
        assert_eq!(mean_and_sample_std(&[]), (0.0, 0.0));
    }

    #[test]
    fn single_sample_has_zero_std() {
        let (mean, std) = mean_and_sample_std(&[3.5]);
        assert_eq!(mean, 3.5);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn bessel_corrected_std_matches_manual() {
        // [2, 4, 4, 4, 5, 5, 7, 9]: sample std = 2.0 (well-known).
        let s = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = mean_and_sample_std(&s);
        assert_eq!(mean, 5.0);
        assert!((std - 2.138_089_935_299_39).abs() < 1e-12);
    }
}
