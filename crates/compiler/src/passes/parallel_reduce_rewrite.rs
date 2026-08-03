//! Block-reduce lowering for `reduce` nodes with insufficient outer
//! parallelism.
//!
//! The default lowering of `reduce [K] |i| body` inside a `compute [M]`
//! parent hoists the accumulator to a per-thread register and runs a
//! sequential SSA loop of length K on ONE thread per outer index. That's
//! what we want when `M` (the outer parallelism) is large enough to
//! saturate the GPU. For small `M` (top-level scalar reduce → `M = 1`
//! after canonicalization, or the `frac_compute_round` shape with
//! `M = 2`), only `M` threads do useful work.
//!
//! This pass rewrites `compute [M] |t| reduce [K] |i| body(t, i)` (and
//! bare top-level `Reduce`, which canonicalize otherwise wraps as
//! `compute [1] |_| reduce [K]`) into the standard CUDA block-reduce
//! shape: each of the `M * G` blocks does a per-thread sequential
//! accumulation (K / (G * BLOCK_SIZE) iterations per thread) followed by
//! a `BLOCK_SIZE`-wide halving tree.
//!
//! ```text
//! let stage0 = compute [M * G] |bid| {
//!   let per_thread = compute [BLOCK_SIZE] |tid| {
//!     reduce_add [K / (G * BLOCK_SIZE)] |c| body_at(bid, tid, c)
//!   };
//!   // Halving tree on per_thread (bounded by BLOCK_SIZE = 256):
//!   let l0 = compute [BLOCK_SIZE/2] |j| per_thread[j] `op` per_thread[j + BLOCK_SIZE/2];
//!   let l1 = compute [BLOCK_SIZE/4] |j| l0[j]         `op` l0[j + BLOCK_SIZE/4];
//!   ...
//!   let ln = compute [1] |_| ...;
//!   ln[0]
//! }
//! // If G > 1: recurse on stage0 with elems_per_row := G.
//! ```
//!
//! Compared with a straight `[K]`-sized halving chain, this has two
//! wins:
//!
//! - The tile stays small (`BLOCK_SIZE = 256` elements, not K), so downstream layout inference and
//!   shared-memory planning are dealing with a fixed 32 KB per block regardless of K.
//! - The bulk of the work — the K/(G·BLOCK_SIZE) sequential accumulation — lives in a per-thread
//!   register accumulator via the existing "hoisted reduce" lowering; each of the BLOCK_SIZE
//!   threads in each block is fully utilized.
//!
//! # Kernel count
//!
//! We prefer a single stage whenever `G = 1` suffices, i.e. the outer
//! `M * BLOCK_SIZE` threads meet a lower bound
//! [`MIN_TOTAL_THREADS`]. When `M` is too small even for that, we split
//! into two stages, picking `G` so both stages have at least
//! `MIN_STAGE_WORK` items so we don't launch a tiny follow-up kernel to
//! reduce a handful of partials — see [`pick_g`]. `MIN_STAGE_WORK = 256`
//! matches the user's rule of thumb: don't launch a kernel just to
//! reduce fewer than one block's worth of items.
//!
//! # When it fires
//!
//! [`should_tree_lower`] gates the rewrite on:
//! - `K` is a power of two and `K >= REDUCE_TREE_MIN` (below that the default sequential lowering
//!   wins on launch overhead);
//! - outer parallelism `M < OUTER_SATURATION_THRESHOLD` (above that we already have enough
//!   threads).

use rustc_hash::FxHashMap;

use crate::{
    ir::{BinOp, IRBuilder, Module, Node, NodeId, ReduceOp, VarId},
    passes::type_infer::type_infer,
    CompileError,
};

/// Outer parallelism above which the default sequential lowering wins.
const OUTER_SATURATION_THRESHOLD: usize = 256;

/// Below this `K` a tree launches more kernels than it saves work.
///
/// Set low (`4`) so small-K reductions still get block-shaped parallelism
/// when `M` alone (the default sequential-per-outer-thread lowering) can't
/// keep the block busy. At `K < 4` the sequential path wins on any
/// reasonable metric.
const REDUCE_TREE_MIN: usize = 4;

/// Kernel-level block size; must match `lower_to_kir::BLOCK_SIZE`.
const BLOCK_SIZE: usize = 256;

/// If `M * BLOCK_SIZE` reaches this many threads, a single-stage
/// block-reduce keeps the GPU busy. Below that we prefer multi-stage.
const MIN_TOTAL_THREADS: usize = 4 * BLOCK_SIZE;

/// Absorb rule: never launch a follow-up kernel that would have fewer
/// than this many items to consume. `256` = one block; below that the
/// launch overhead dominates.
const MIN_STAGE_WORK: usize = BLOCK_SIZE;

/// Runs the parallel-reduce rewrite. Idempotent: a second run does
/// nothing because rewritten reduces are no longer `Node::Reduce`.
pub fn rewrite_parallel_reduce(module: Module) -> Result<Module, CompileError> {
    let _ = type_infer(&module)?;
    debug_assert!(BLOCK_SIZE.is_power_of_two());
    let Module {
        name,
        mut builder,
        body,
    } = module;

    let mut cx = RewriteCx {
        b: &mut builder,
        subst: FxHashMap::default(),
        subst_memo: FxHashMap::default(),
        changed: false,
    };

    let new_body = cx.walk_top(body)?;
    let changed = cx.changed;

    Ok(Module {
        name,
        builder,
        body: if changed { new_body } else { body },
    })
}

struct RewriteCx<'a> {
    b: &'a mut IRBuilder,
    subst: FxHashMap<VarId, NodeId>,
    subst_memo: FxHashMap<NodeId, NodeId>,
    changed: bool,
}

impl RewriteCx<'_> {
    fn walk_top(&mut self, id: NodeId) -> Result<NodeId, CompileError> {
        let mut bindings: Vec<(VarId, NodeId)> = Vec::new();
        let mut cur = id;
        let terminal = loop {
            match self.b.node(cur).clone() {
                Node::Let { var, value, body } => {
                    bindings.push((var, value));
                    cur = body;
                }
                _ => break cur,
            }
        };

        let mut new_bindings: Vec<(VarId, NodeId)> = Vec::new();
        for (var, value) in bindings {
            let (extra, new_value) = self.rewrite_top_value(value)?;
            new_bindings.extend(extra);
            new_bindings.push((var, new_value));
        }
        let (extra, new_terminal) = self.rewrite_top_value(terminal)?;
        new_bindings.extend(extra);

        let mut body = new_terminal;
        for (var, value) in new_bindings.into_iter().rev() {
            body = self.b.intern(Node::Let { var, value, body });
        }
        Ok(body)
    }

    fn rewrite_top_value(
        &mut self,
        id: NodeId,
    ) -> Result<(Vec<(VarId, NodeId)>, NodeId), CompileError> {
        match self.b.node(id).clone() {
            Node::Reduce {
                op,
                bound: k,
                var: i,
                body,
            } => {
                if should_tree_lower(k, 1) {
                    let (extras, final_expr) = self.build_block_reduce(op, 1, None, k, i, body);
                    self.changed = true;
                    Ok((extras, final_expr))
                } else {
                    Ok((Vec::new(), id))
                }
            }
            Node::Compute {
                bound: m,
                var: t,
                body: compute_body,
                scatter: None,
                par: None,
                threads: None,
            } => {
                if let Node::Reduce {
                    op,
                    bound: k,
                    var: i,
                    body,
                } = self.b.node(compute_body).clone()
                {
                    if should_tree_lower(k, m) {
                        let (extras, final_expr) =
                            self.build_block_reduce(op, m, Some(t), k, i, body);
                        self.changed = true;
                        return Ok((extras, final_expr));
                    }
                }
                Ok((Vec::new(), id))
            }
            Node::Tuple(elems) => {
                let mut extras_all: Vec<(VarId, NodeId)> = Vec::new();
                let mut new_elems = Vec::with_capacity(elems.len());
                let mut any = false;
                for &e in &elems {
                    let (extras, ne) = self.rewrite_top_value(e)?;
                    if !extras.is_empty() || ne != e {
                        any = true;
                    }
                    extras_all.extend(extras);
                    new_elems.push(ne);
                }
                if any {
                    Ok((extras_all, self.b.intern(Node::Tuple(new_elems))))
                } else {
                    Ok((Vec::new(), id))
                }
            }
            _ => Ok((Vec::new(), id)),
        }
    }

    /// Build the block-reduce chain: one or more top-level stages, each
    /// launching `M * G` blocks that do a per-thread sequential
    /// accumulation followed by a BLOCK_SIZE halving tree. Returns
    /// `(intermediate_bindings, final_expr)` — same shape as the halving
    /// variant used to.
    fn build_block_reduce(
        &mut self,
        op: ReduceOp,
        m: usize,
        outer_var: Option<VarId>,
        k: usize,
        reduce_var: VarId,
        body: NodeId,
    ) -> (Vec<(VarId, NodeId)>, NodeId) {
        debug_assert!(k.is_power_of_two() && k >= 2);
        let bin_op = match op {
            ReduceOp::Add => BinOp::Add,
            ReduceOp::Mul => BinOp::Mul,
        };

        let mut extras: Vec<(VarId, NodeId)> = Vec::new();
        let mut prev_tensor: Option<VarId> = None;
        let mut elems_per_row = k;

        loop {
            let g = pick_g(m, elems_per_row);
            let per_block = elems_per_row / g;
            let stage_expr = self.build_stage(
                op,
                bin_op,
                m,
                g,
                per_block,
                elems_per_row,
                outer_var,
                k,
                reduce_var,
                body,
                prev_tensor,
            );

            if g == 1 {
                // Final stage — [M]-shaped output replaces the original
                // expression.
                return (extras, stage_expr);
            }

            let stage_var = self.b.fresh_var();
            extras.push((stage_var, stage_expr));
            prev_tensor = Some(stage_var);
            elems_per_row = g;
        }
    }

    /// Emit one stage as an outer `compute [M * G]` whose body is a
    /// let-bound per-thread `compute [BLOCK_SIZE]` (with a nested
    /// sequential `reduce_add` of length `per_block / BLOCK_SIZE`),
    /// followed by a chain of let-bound halving `compute`s down to `[1]`,
    /// yielding a scalar via `Index(_, 0)`.
    #[allow(clippy::too_many_arguments)]
    fn build_stage(
        &mut self,
        op: ReduceOp,
        bin_op: BinOp,
        m: usize,
        g: usize,
        per_block: usize,
        elems_per_row: usize,
        outer_var: Option<VarId>,
        k: usize,
        reduce_var: VarId,
        original_body: NodeId,
        prev_tensor: Option<VarId>,
    ) -> NodeId {
        let total_blocks = m * g;
        let bid_var = self.b.fresh_var();
        let bid = self.b.intern(Node::Var(bid_var));

        let _ = (m, k);
        // Build the per-thread tile inline (avoids borrowing self twice).
        let tile_iter_var = self.b.fresh_var();
        let tid = self.b.intern(Node::Var(tile_iter_var));
        let (tile_bound, tile_body) = if per_block <= BLOCK_SIZE {
            // Small per-block: use `per_block` threads, one element each.
            let load = self.emit_load(
                bid,
                tid,
                None,
                per_block,
                elems_per_row,
                g,
                outer_var,
                reduce_var,
                original_body,
                prev_tensor,
            );
            (per_block, load)
        } else {
            debug_assert!(per_block % BLOCK_SIZE == 0);
            let seq = per_block / BLOCK_SIZE;
            let c_var = self.b.fresh_var();
            let c_node = self.b.intern(Node::Var(c_var));
            let load = self.emit_load(
                bid,
                tid,
                Some(c_node),
                per_block,
                elems_per_row,
                g,
                outer_var,
                reduce_var,
                original_body,
                prev_tensor,
            );
            // `reduce_add [seq] |c| load(bid, tid, c)` lowers via the
            // default hoisted-register-accumulator path to a per-thread
            // SSA loop.
            let reduce = self.b.intern(Node::Reduce {
                op,
                bound: seq,
                var: c_var,
                body: load,
            });
            (BLOCK_SIZE, reduce)
        };
        let tile_compute = self.b.intern(Node::Compute {
            bound: tile_bound,
            var: tile_iter_var,
            body: tile_body,
            scatter: None,
            par: None,
            threads: None,
        });

        // Bind the per-thread tile as a let, then a chain of halving
        // levels each bound to its own let, ending in Index(last, 0).
        let tile_var = self.b.fresh_var();
        let mut let_chain: Vec<(VarId, NodeId)> = vec![(tile_var, tile_compute)];
        let mut cur_var_id = tile_var;
        let mut cur_size = tile_bound;

        while cur_size > 1 {
            let s = cur_size / 2;
            let cur_node = self.b.intern(Node::Var(cur_var_id));
            let j_var = self.b.fresh_var();
            let j = self.b.intern(Node::Var(j_var));
            let s_c = self.b.const_u32(s as u32);
            let j_plus_s = self.b.add(j, s_c);
            let lo = self.b.intern(Node::Index {
                tensor: cur_node,
                indices: vec![j],
            });
            let hi = self.b.intern(Node::Index {
                tensor: cur_node,
                indices: vec![j_plus_s],
            });
            let combined = self.b.bin(bin_op, lo, hi);
            let level_compute = self.b.intern(Node::Compute {
                bound: s,
                var: j_var,
                body: combined,
                scatter: None,
                par: None,
                threads: None,
            });
            let level_var = self.b.fresh_var();
            let_chain.push((level_var, level_compute));
            cur_var_id = level_var;
            cur_size = s;
        }

        let last_node = self.b.intern(Node::Var(cur_var_id));
        let zero = self.b.const_u32(0);
        let final_scalar = self.b.intern(Node::Index {
            tensor: last_node,
            indices: vec![zero],
        });

        let mut body_expr = final_scalar;
        for (var, value) in let_chain.into_iter().rev() {
            body_expr = self.b.intern(Node::Let {
                var,
                value,
                body: body_expr,
            });
        }

        self.b.intern(Node::Compute {
            bound: total_blocks,
            var: bid_var,
            body: body_expr,
            scatter: None,
            par: None,
            threads: None,
        })
    }

    /// Emit the load expression for a per-thread tile iteration:
    /// `bid` addresses the block within the flat `M * G` grid, `tid`
    /// is the thread within the block, `seq_iter` is `None` when the
    /// tile has one element per thread and `Some(c_node)` when there's
    /// a sequential per-thread accumulator over `c ∈ [0, seq)`.
    ///
    /// Coordinate math (all quasi-affine, all constants power of two):
    ///   t     = bid / G                                (outer row)
    ///   chunk = bid % G                                (which G-chunk of row t)
    ///   local = chunk * per_block + c * BLOCK_SIZE + tid
    ///     (with `c * BLOCK_SIZE` dropped when `seq_iter` is None)
    ///
    /// The absolute logical index within row t is `local`, so the
    /// original body sees `(outer_var → t, reduce_var → local)`. When
    /// there's a previous stage tensor, we skip the substitution and
    /// index `prev[t * elems_per_row + local]` directly.
    #[allow(clippy::too_many_arguments)]
    fn emit_load(
        &mut self,
        bid: NodeId,
        tid: NodeId,
        seq_iter: Option<NodeId>,
        per_block: usize,
        elems_per_row: usize,
        g: usize,
        outer_var: Option<VarId>,
        reduce_var: VarId,
        original_body: NodeId,
        prev_tensor: Option<VarId>,
    ) -> NodeId {
        let (t_expr, chunk_expr) = if g == 1 {
            // bid == t, chunk == 0 — save the div/sub.
            let zero = self.b.const_u32(0);
            (bid, zero)
        } else {
            let g_c = self.b.const_u32(g as u32);
            let t_expr = self.b.div(bid, g_c);
            let t_mul_g = self.b.mul(t_expr, g_c);
            let chunk_expr = self.b.sub(bid, t_mul_g);
            (t_expr, chunk_expr)
        };
        let per_block_c = self.b.const_u32(per_block as u32);
        let chunk_mul_pb = self.b.mul(chunk_expr, per_block_c);
        let mut local = self.b.add(chunk_mul_pb, tid);
        if let Some(c) = seq_iter {
            let bs_c = self.b.const_u32(BLOCK_SIZE as u32);
            let c_mul_bs = self.b.mul(c, bs_c);
            local = self.b.add(local, c_mul_bs);
        }

        match prev_tensor {
            None => {
                // Substitute the original body's outer / reduce vars with
                // `(t_expr, local)`.
                self.subst.clear();
                self.subst_memo.clear();
                if let Some(t) = outer_var {
                    self.subst.insert(t, t_expr);
                }
                self.subst.insert(reduce_var, local);
                let out = self.substitute(original_body);
                self.subst.clear();
                self.subst_memo.clear();
                out
            }
            Some(prev_var) => {
                // `prev` has shape `[M * elems_per_row]`. Access
                // `prev[t * elems_per_row + local]`.
                let prev_node = self.b.intern(Node::Var(prev_var));
                let epr_c = self.b.const_u32(elems_per_row as u32);
                let t_mul_epr = self.b.mul(t_expr, epr_c);
                let idx = self.b.add(t_mul_epr, local);
                self.b.intern(Node::Index {
                    tensor: prev_node,
                    indices: vec![idx],
                })
            }
        }
    }

    /// Post-order substitute per `self.subst` inside `expr`.
    fn substitute(&mut self, expr: NodeId) -> NodeId {
        if let Some(&r) = self.subst_memo.get(&expr) {
            return r;
        }
        let node = self.b.node(expr).clone();
        let out = match node {
            Node::Var(v) => match self.subst.get(&v).copied() {
                Some(n) => n,
                None => expr,
            },
            Node::Input(_) | Node::ConstU32(_) | Node::ConstField(_) | Node::ConstFpExt(_) => expr,
            Node::LiftFpExt(x) => {
                let nx = self.substitute(x);
                if nx == x {
                    expr
                } else {
                    self.b.intern(Node::LiftFpExt(nx))
                }
            }
            Node::Bin(op, a, b) => {
                let na = self.substitute(a);
                let nb = self.substitute(b);
                if na == a && nb == b {
                    expr
                } else {
                    self.b.intern(Node::Bin(op, na, nb))
                }
            }
            Node::Select {
                cond,
                then_val,
                else_val,
            } => {
                let nc = self.substitute(cond);
                let nt = self.substitute(then_val);
                let ne = self.substitute(else_val);
                if nc == cond && nt == then_val && ne == else_val {
                    expr
                } else {
                    self.b.intern(Node::Select {
                        cond: nc,
                        then_val: nt,
                        else_val: ne,
                    })
                }
            }
            Node::Index { tensor, indices } => {
                let nt = self.substitute(tensor);
                let ni: Vec<NodeId> = indices.iter().map(|&i| self.substitute(i)).collect();
                if nt == tensor && ni == indices {
                    expr
                } else {
                    self.b.intern(Node::Index {
                        tensor: nt,
                        indices: ni,
                    })
                }
            }
            Node::Tuple(elems) => {
                let ne: Vec<NodeId> = elems.iter().map(|&e| self.substitute(e)).collect();
                if ne == elems {
                    expr
                } else {
                    self.b.intern(Node::Tuple(ne))
                }
            }
            Node::Proj(t, k) => {
                let nt = self.substitute(t);
                if nt == t {
                    expr
                } else {
                    self.b.intern(Node::Proj(nt, k))
                }
            }
            Node::Pack(elems) => {
                let ne: Vec<NodeId> = elems.iter().map(|&e| self.substitute(e)).collect();
                if ne == elems {
                    expr
                } else {
                    self.b.intern(Node::Pack(ne))
                }
            }
            Node::Let { var, value, body } => {
                let nv = self.substitute(value);
                let nb = self.substitute(body);
                if nv == value && nb == body {
                    expr
                } else {
                    self.b.intern(Node::Let {
                        var,
                        value: nv,
                        body: nb,
                    })
                }
            }
            Node::Compute {
                bound,
                var,
                body,
                scatter,
                par,
                threads,
            } => {
                let nb = self.substitute(body);
                if nb == body {
                    expr
                } else {
                    self.b.intern(Node::Compute {
                        bound,
                        var,
                        body: nb,
                        scatter,
                        par,
                        threads,
                    })
                }
            }
            Node::Reduce {
                op,
                bound,
                var,
                body,
            } => {
                let nb = self.substitute(body);
                if nb == body {
                    expr
                } else {
                    self.b.intern(Node::Reduce {
                        op,
                        bound,
                        var,
                        body: nb,
                    })
                }
            }
        };
        self.subst_memo.insert(expr, out);
        out
    }
}

pub(crate) fn should_tree_lower(k: usize, outer_par: usize) -> bool {
    k.is_power_of_two() && k >= REDUCE_TREE_MIN && outer_par < OUTER_SATURATION_THRESHOLD
}

/// Number of blocks per row for a stage with `elems_per_row` items,
/// given `M` outer rows. Returns a power of two dividing `elems_per_row`.
///
/// Rules:
/// - Prefer `G = 1` (single-stage) whenever the resulting launch has at least `MIN_TOTAL_THREADS` —
///   that gives us all the GPU utilization we can squeeze from `M` rows alone.
/// - Otherwise, pick the smallest `G` such that `M * G * BLOCK_SIZE >= MIN_TOTAL_THREADS`. This
///   keeps stage 0 busy while keeping stage 1's work bounded.
/// - Also require `G <= elems_per_row` and that stage 1's work `M * G >= MIN_STAGE_WORK`, so we
///   never emit a follow-up kernel that would have less than one block's worth of items to reduce
///   (per the user's rule of thumb). If nothing satisfies both, fall back to `G = 1` and accept the
///   underutilization.
fn pick_g(m: usize, elems_per_row: usize) -> usize {
    debug_assert!(elems_per_row.is_power_of_two());
    // G = 1 already saturates? Take it.
    if m * BLOCK_SIZE >= MIN_TOTAL_THREADS {
        return 1;
    }
    // Multi-stage requires stage 0's per-block work to hold at least
    // BLOCK_SIZE elements (one element per thread, or a per-thread seq
    // reduce over more). Equivalently, `g <= elems_per_row / BLOCK_SIZE`.
    // This also guarantees progress: `elems_per_row / g >= BLOCK_SIZE > 1`.
    let max_g = elems_per_row / BLOCK_SIZE;
    if max_g < 2 {
        return 1;
    }
    // Preferred G: smallest that hits both saturation targets — enough
    // stage 0 threads AND enough stage 1 work to fill a block. When we
    // can hit both, this keeps stage 1's launch worthwhile.
    let mut g = 2usize;
    while g <= max_g {
        let stage0_threads = m * g * BLOCK_SIZE;
        let stage1_work = m * g;
        if stage0_threads >= MIN_TOTAL_THREADS && stage1_work >= MIN_STAGE_WORK {
            return g;
        }
        g <<= 1;
    }
    // Neither target hit — pick the largest legal G. It gives us the
    // most parallelism in stage 0 at the cost of a tiny stage 1
    // (a single warp or two). Two-kernel launch is still faster than
    // running the reduction on `m` blocks (each on its own SM).
    max_g
}
