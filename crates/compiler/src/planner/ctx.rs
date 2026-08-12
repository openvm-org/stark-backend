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
///
/// # Alias-aware indexing
///
/// When the graph carries an alias table (produced by
/// `passes::restore_ssa`), the ctx canonicalizes every buffer reference
/// at build time: `writers[canon(b)]` / `readers[canon(b)]` accumulate
/// on the canonical, non-canonical members have empty lists, and
/// `packable` returns `false` for them. After the backend picks
/// offsets, `propagate_alias_offsets` copies `offsets[canon]` to every
/// member.
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
    /// Per-buffer node indices that write to it. Empty for non-canonical
    /// alias members.
    pub writers: Vec<Vec<usize>>,
    /// Per-buffer node indices that read it. Same canonicalization.
    pub readers: Vec<Vec<usize>>,
    /// Canonical `BufId` for each buffer under the graph's alias table.
    /// `canon[b] == b` when `b` is its own canonical (default, no
    /// aliases). Path-compressed at build time.
    pub canon: Vec<usize>,
}

impl PlanCtx {
    /// Alias-free variant. Every buffer is its own canonical.
    pub fn build(
        bufs: &[BufInfo],
        nodes: &[NodeAccess],
        env: &BTreeMap<VarId, i64>,
        device: DeviceType,
        pin: &[BufId],
    ) -> Result<Self, PlanError> {
        Self::build_with_aliases(bufs, nodes, env, device, pin, &[])
    }

    /// Alias-aware entry point. `aliases[b] = Some(parent)` means `b` is
    /// a fresh SSA version of `parent` and must share `parent`'s pool
    /// slot. `aliases.len()` must equal `bufs.len()` (or be empty for
    /// "no aliases").
    pub fn build_with_aliases(
        bufs: &[BufInfo],
        nodes: &[NodeAccess],
        env: &BTreeMap<VarId, i64>,
        device: DeviceType,
        pin: &[BufId],
        aliases: &[Option<BufId>],
    ) -> Result<Self, PlanError> {
        let n_nodes = nodes.len();
        let n_bufs = bufs.len();

        // Canonicalize with path compression. For an alias-free graph
        // this loop degenerates to identity.
        let mut canon: Vec<usize> = (0..n_bufs).collect();
        if !aliases.is_empty() {
            assert_eq!(
                aliases.len(),
                n_bufs,
                "aliases length {} must equal bufs {}",
                aliases.len(),
                n_bufs,
            );
            for start in 0..n_bufs {
                let mut cur = start;
                while let Some(parent) = aliases.get(cur).copied().flatten() {
                    cur = parent.0;
                    debug_assert!(cur < n_bufs);
                }
                canon[start] = cur;
                let mut step = start;
                while let Some(parent) = aliases.get(step).copied().flatten() {
                    canon[step] = cur;
                    step = parent.0;
                    if step == cur {
                        break;
                    }
                }
            }
        }

        let mut pinned = vec![false; n_bufs];
        for &b in pin {
            // Pin the canonical — the runtime resolves an aliased member
            // to the canonical's slot.
            pinned[canon[b.0]] = true;
        }

        let mut sizes = vec![0i64; n_bufs];
        let mut aligns = vec![1u64; n_bufs];
        let mut on_device = vec![false; n_bufs];
        for (idx, info) in bufs.iter().enumerate() {
            if info.device_type == device {
                on_device[idx] = true;
                let s = eval_size(BufId(idx), &info.size, env)?;
                let align = (info.elem_size as u64).max(1);
                let c = canon[idx];
                // Accumulate on the canonical. Members within a class
                // carry identical BufInfos in practice (restore_ssa
                // clones) but be defensive and take the max.
                sizes[c] = sizes[c].max(s);
                aligns[c] = aligns[c].max(align);
                on_device[c] = true;
            }
        }

        let writes: Vec<Vec<usize>> = nodes
            .iter()
            .map(|a| a.writes.iter().map(|b| canon[b.0]).collect())
            .collect();
        let reads: Vec<Vec<usize>> = nodes
            .iter()
            .map(|a| a.reads.iter().map(|b| canon[b.0]).collect())
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
        // Dedup consecutive duplicates: if a single node writes to two
        // SSA-renames aliasing the same canonical, we push the node
        // index twice into `writers[canon]`. That's a spurious repeat
        // for lifetime derivation.
        for ws in writers.iter_mut() {
            ws.dedup();
        }
        for rs in readers.iter_mut() {
            rs.dedup();
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
            canon,
        })
    }

    /// A packable buffer occupies a slot in the returned plan (on the
    /// target device with non-zero size). Only the canonical of an
    /// alias class is packable — members share the canonical's slot.
    pub fn packable(&self, b: usize) -> bool {
        if self.canon[b] != b {
            return false;
        }
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

/// Copy `offsets[canon] → offsets[member]` for every alias member.
///
/// Backends fill offsets only for canonical entries (the non-canonicals
/// return `packable == false`). Callers must call this once on the
/// backend's output — the runtime resolves a member `BufId` directly
/// via `offsets[b]`, so every member of an alias class must carry the
/// same pool offset the canonical was assigned.
pub fn propagate_alias_offsets(offsets: &mut [Option<u64>], canon: &[usize]) {
    for b in 0..offsets.len() {
        let c = canon[b];
        if c != b {
            offsets[b] = offsets[c];
        }
    }
}
