//! Shared read/write access model and preprocessing consumed by every
//! planner backend.
//!
//! The `PlanCtx` collapses buffer sizes, alignments, device residency,
//! pinning and per-buffer writer/reader lists into a single view. Backends
//! then build precedence edges and lifetime intervals off of it (see
//! [`PlanCtx::edges`], [`PlanCtx::per_node_access`]).

use std::collections::BTreeMap;

use crate::{
    graph_ir::{BufId, BufInfo, DeviceType, GraphNode},
    ir::VarId,
    quast::Quast,
};

/// Read/write set for one graph node, as seen by the planner.
///
/// A node's `reads` and `writes` contribute to `death[b]` of every accessed
/// buffer, its `writes` contribute to `birth[b]` of every written buffer,
/// and versioned RAW/WAR/WAW pairs become precedence edges (see
/// [`PlanCtx::edges`]). A buffer that is both read and written by the same
/// node has `birth[b] = death[b] = t[n]` and thus a single time-step
/// lifetime.
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
    #[cfg(feature = "planner-ortools")]
    #[error("CP-SAT returned no solution (status: {0:?})")]
    NoSolution(cp_sat::proto::CpSolverStatus),
    #[error("no legal schedule found: {0}")]
    Infeasible(String),
}

/// Preprocessed planner input: concrete sizes/alignments on the target
/// device and per-buffer writer / reader lists (ascending in node insertion
/// order). Every backend consumes this same view.
pub struct PlanCtx {
    pub n_nodes: usize,
    pub n_bufs: usize,
    /// Concrete byte sizes, `0` for off-device buffers.
    pub sizes: Vec<i64>,
    /// Alignment (in bytes) per buffer; `1` for off-device or `elem_size`
    /// otherwise.
    pub aligns: Vec<u64>,
    pub on_device: Vec<bool>,
    /// Lifetime pinned to program end (`death = n_nodes`).
    pub pinned: Vec<bool>,
    /// Per-buffer node indices that write to it.
    pub writers: Vec<Vec<usize>>,
    /// Per-buffer node indices that read it.
    pub readers: Vec<Vec<usize>>,
}

impl PlanCtx {
    pub fn build(
        bufs: &[BufInfo],
        nodes: &[NodeAccess],
        env: &BTreeMap<VarId, i64>,
        device: DeviceType,
        pin: &[BufId],
    ) -> Result<Self, PlanError> {
        let n_nodes = nodes.len();
        let n_bufs = bufs.len();

        let mut pinned = vec![false; n_bufs];
        for &b in pin {
            pinned[b.0] = true;
        }

        let mut sizes = vec![0i64; n_bufs];
        let mut aligns = vec![1u64; n_bufs];
        let mut on_device = vec![false; n_bufs];
        for (idx, info) in bufs.iter().enumerate() {
            if info.device_type == device {
                on_device[idx] = true;
                sizes[idx] = eval_size(BufId(idx), &info.size, env)?;
                aligns[idx] = (info.elem_size as u64).max(1);
            }
        }

        let writes: Vec<Vec<usize>> = nodes
            .iter()
            .map(|a| a.writes.iter().map(|b| b.0).collect())
            .collect();
        let reads: Vec<Vec<usize>> = nodes
            .iter()
            .map(|a| a.reads.iter().map(|b| b.0).collect())
            .collect();

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

        Ok(Self {
            n_nodes,
            n_bufs,
            sizes,
            aligns,
            on_device,
            pinned,
            writers,
            readers,
        })
    }

    /// A packable buffer occupies a slot in the returned plan (on the
    /// target device with non-zero size).
    pub fn packable(&self, b: usize) -> bool {
        // Buffers no node reads or writes need no pool slot; pinned buffers
        // are exempt so registered interface buffers keep a stable slot
        // even if optimization removed every access.
        self.on_device[b]
            && self.sizes[b] > 0
            && (self.pinned[b] || !(self.writers[b].is_empty() && self.readers[b].is_empty()))
    }

    /// Direct precedence edges implied by the read/write sets, using node
    /// insertion order to version each buffer:
    ///
    /// - **WAW**: consecutive writers of the same buffer.
    /// - **RAW**: reader after the last writer inserted before it.
    /// - **WAR**: reader before the next writer inserted after it.
    ///
    /// Returns `succ` (adjacency lists, deduped and sorted) and the
    /// corresponding in-degrees.
    pub fn edges(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut succ: Vec<Vec<usize>> = vec![vec![]; self.n_nodes];
        for b in 0..self.n_bufs {
            let writers = &self.writers[b];
            for pair in writers.windows(2) {
                succ[pair[0]].push(pair[1]);
            }
            for &r in &self.readers[b] {
                let i = writers.partition_point(|&w| w < r);
                if i > 0 {
                    succ[writers[i - 1]].push(r);
                }
                let j = writers.partition_point(|&w| w <= r);
                if j < writers.len() {
                    succ[r].push(writers[j]);
                }
            }
        }
        for s in &mut succ {
            s.sort_unstable();
            s.dedup();
        }
        let mut indeg = vec![0usize; self.n_nodes];
        for sv in &succ {
            for &v in sv {
                indeg[v] += 1;
            }
        }
        (succ, indeg)
    }

    /// Per-node read/write buffer id lists.
    pub fn per_node_access(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n = self.n_nodes;
        let mut writes = vec![vec![]; n];
        let mut reads = vec![vec![]; n];
        for b in 0..self.n_bufs {
            for &w in &self.writers[b] {
                writes[w].push(b);
            }
            for &r in &self.readers[b] {
                reads[r].push(b);
            }
        }
        (writes, reads)
    }
}

/// Builds a [`NodeAccess`] from a [`GraphNode`], matching the semantics
/// [`crate::planner::plan_raw`] expects.
pub fn access_from_node(node: &GraphNode) -> NodeAccess {
    let mut a = NodeAccess::default();
    match node {
        GraphNode::BlackboxKernel(k) => {
            a.reads.extend(k.inputs.iter().copied());
            a.writes.extend(k.carried_outputs.iter().copied());
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

pub fn eval_size(buf: BufId, size: &Quast, env: &BTreeMap<VarId, i64>) -> Result<i64, PlanError> {
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

#[inline]
pub fn align_up(off: i64, align: i64) -> i64 {
    if align > 1 {
        (off + align - 1) / align * align
    } else {
        off
    }
}
