# Kernel Fusion Framework V2

In the first version of the kernel fusion pass pairs of kernels are eagerly fused together, leading to sub-optimal kernel fusions and excessive compilation times.

In this version we are going for something more clever, by using a combination of equality saturation and combinatorial optimization with CP-SAT or an ILP.

We are going to separate the task of fusing kernels from the task of figuring out which set of fusions is optimal.

Therefore we need a `GraphFuser` that can represent alternative computations:


```rust 

pub struct ValueClassId(usize); 

pub struct NodeId(usize); 

pub struct AltGraphNode {
  inputs: Vec<ValueClassId>,
  outputs: Vec<ValueClassId>,
  node: GraphNode,
}


pub struct GraphFuser {
  /// ValueClassId indexes into this
  bufs: Vec<BufInfo>,
  /// NodeId indexes into this
  nodes: Vec<AltGraphNode>,

  /// hypergraph edges from BufClassId to BufId
  producers: HashMap<ValueClassId, Vec<UseInfo>>,
  /// hypergraph edges from BufClassId to their users
  consumers: HashMap<ValueClassId, Vec<UseInfo>>,

  access_relations: HashMap<NodeId, AccessRelation>,
  re_exported: HashMap<ValueClassId, ValueClassId>, // maps to the first instance

  inputs: Vec<ValueClassId>,
  outputs: Vec<ValueClassId>,
}

pub struct UseInfo {
  node: NodeId,
  pos: usize
}

```

This design is so the fusion can happen in parallel, `GraphFuser` gets borrowed immutably, and fusion logic produces `AltGraphNode`, which can be collected into a vec of updates. Then you'd insert `AltGraphNode` into `GraphFuser` mutably and apply the right updates to the metadata fields.

`AltGraphNode` is intended to wrap the existing graph nodes, and make the graph conform to the SSA property. Specifically, for nodes that mutate their inputs, say `n` inputs, `k` of them are mutated, and has `m` outputs, the `AltGraphNode` therefore has `n` inputs and `m+k` outputs. Each mutated input is re-exported as an output with new `ValueClassId`, and subsequent consumers are routed to use the new output.


Since the graph has the SSA property: each `ValueClassId` is the output of exactly one node, therefore we can deduce which nodes are alternatives simply based on the fact that if multiple nodes produced the same `ValueClassId`, then they are alternatives of each other.

We may assume that each node supports the following positional api:

```rust 
impl GraphNode {
  fn get_operands(&self) -> Vec<BufId> {...}
  fn get_results(&self) -> Vec<BufId> {...}
}
```

the access relations needs to be simplified to the following:

```rust

struct AccessRelation {
  reads: Vec<ReadRelation>,
  writes: Vec<WriteRelation>,
  index_bounds: HashMap<VarId, i64>,
  grid_index: VarId,
  inner_indices: Vec<VarId>, // for flat computes this is empty
}

struct ReadRelation {
  read: Quast,
  val: ValueClassId, // the corresponding buffer read
  node: NodeId // the corresponding index expression node in the kernel ir
}

struct WriteRelation {
  write: Quast,
  inv: Quast,
  val: ValueClassId,
  node: NodeId
}
```
since most writes are identities, the only writes that are not are those with scatter maps, on which the user supplies the inverse.

## Extraction

After a series of alternatives are inserted into the `GraphFuser`, they need to be extracted. We first need a cost function:

```rust
pub type Cycles = f64;

pub struct AccessEst {
  bytes: u32,
  avg_sectors: f64,
  est_cycles: Cycles,
}

pub struct GraphNodeCost {
  smem_use_bytes: u32,
  est_reg_use: u32,
  est_occupancy: u32, // number of blocks per SM

  grid_dim: u32,
  block_dim: u32,

  read_accesses: Vec<AccessEst>,
  write_accesses: Vec<AccessEst>,

  ops_per_output: u64,
  syncs_per_output: u32,
  cycles_per_output: Cycles,
  
  raw_cycles: Cycles, // not counting kernel launch
  kernel_launch: Cycles,

  total_cost: f64
}

pub struct EstimatorConfig {
  num_samples: u32,
}

pub fn estimate_cost(config: &EstimatorConfig, module: &GraphNode) -> GraphNodeCost {
  // we may assume module is already canonicalized and contains exactly one kernel
  // lower to kir, estimate registers and get shared memory requirement
  
  // for each read/write we know the access expression and grid size/block_size, and all symbolic parameters. If the access is not data-dependent, then we have a Quast that depends on a combo of the block index and thread index. We can randomly sample a subset of warps across the block/grid and compute their offsets. This way we can estimate the average number of coalesced reads, or the sectors per access.

  // We can perform an interpretation of the kir. At the leaves with reads, we can add memory access latency with the above probabilistic analysis. Then for each inner computation, add a number of cycles for compute. i.e. 3 cycles for babybear add, 4 for mul, 6 for div. And correspondingly for BabybearEst.
  // at the end we do the same for writes as reads

  // a more subtle detail about the cycles per read/write is necessary. In the non-bandwidth limited context, each memory access is the sectors * raw memory access latency. But in the bandwidth limited context the raw memory access latency is different, usually much higher. This corresponds to the single queue model from queuing theory.

  // At the end, if there are N elements we need to compute and the estimated cycle count is T for each element. Then the total raw_cycles is the 
  // wave count multiplied by T. Where wave count is the number of waves that the kernel takes.

}
```

To be clear, estimate_cost is only a rough approximation. 

### Extraction as an ILP

We'll need to encode the extraction problem as an ILP. 

- Let `A_i in {0,1}` denote that the `i`th `AltGraphNode` is chosen or not. 
- Let `V_j in {0,1}` denote that the `j`th `ValueClassId` is chosen or not. 
- Let `M_t in {0,1}` denote that the `t`th module is compiled or not

- Let `inputs(i)` be the set of `ValueClassId`s that are inputs to the `i`th `AltGraphNode`
- Let `outputs(i)` be the set of `ValueClassId`s that are outputs to the `i`th `AltGraphNode`
- Let `producers(j)` be the set of `AltGraphNode`s that produce the `j`th `ValueClassId`
- Let `K(i)` be the compiled artifact `t` that requires it

Then we need the following constraints:

Every output `ValueClassId` must be chosen:
- for every `j in outputs`, `V_j = 1`: 

Every input `ValueClassId` must be chosen:
- for every `j in inputs`, `V_j = 1`: 

Exactly one producer for every selected value:
- `sum_{i in producers(j)} A_i = v_j` for every `j`

Selected `AltGraphNode`s require their inputs:
- `A_i <= V_j` for every `j in inputs(i)`

Selected `AltGraphNode`s require compiling its module 
- `A_i <= M_{K(i)}`

The module is compiled only if at least 1 node is selected 
- `M_t <= sum_{i : K(i) = t} A_i`

Upper bound the number of compiled modules
- `sum_t M_t <= O`

Then the cost is `C == sum_i cost(i) * A_i`. 


