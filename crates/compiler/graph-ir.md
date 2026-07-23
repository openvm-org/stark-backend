# Graph IR 

The graph IR represents a raw computation graph over a GPU device. It has the following node types:

```
struct KernelNode {
  inputs: Vec<BufId>, outputs: Vec<BufId>, modifies: Vec<bool>, fn: Box<dyn Fn(&[*mut ()], &[*mut ()])>,
  name: String
}
```

The kernel `fn` takes the raw pointers of the input buffers as its first argument and those of the output buffers as its second argument.

```

struct MemcpyNode {
  src: BufId, 
  dst: BufId
}

struct MemSetNode {
  node: BufId, 
  val: u32
}

struct KernelModuleNode {
  module: ir::Module,
  inputs: Vec<BufId>,   // aligned with module.builder.inputs()
  outputs: Vec<BufId>,  // aligned with the module's top-level outputs
}

enum ConstBuf {
  DeviceBuf(DeviceBuffer<u8>),  // openvm-cuda-common
  HostBuf(Vec<u8>),
}

struct ConstNode {
  buf: BufId,
  data: ConstBuf,
}

enum GraphNode {
  /// Structured kernel: a high-level ir::Module plus explicit BufId
  /// bindings for its module inputs and outputs.
  Kernel(KernelModuleNode),
  /// Opaque host closure with explicit input/output pointer bindings.
  BlackboxKernel(KernelNode),
  /// Static data attached to a graph buffer (device- or host-resident).
  Const(ConstNode),
  Memcpy(MemcpyNode),
  Memset(MemSetNode)
}

```

`GraphNode::Kernel` pairs an `ir::Module` (see `ir.rs`) — the same top-level
unit that `compile_and_load` compiles end-to-end — with explicit `BufId`
bindings for its declared inputs and its top-level outputs, so downstream
passes know which graph buffers feed which module inputs without modifying
`ir::Module` itself. This lets a graph mix opaque host-launched kernels
(`BlackboxKernel`, useful for pre-existing hand-written CUDA) with
structured kernels expressed in the functional DSL.

`GraphNode::Const` makes `buf` refer to statically-provided bytes: either an
already-resident device allocation (`DeviceBuf(DeviceBuffer<u8>)`, from
`openvm-cuda-common`) or a host buffer (`HostBuf(Vec<u8>)`). Semantically
the constant node writes the buffer, so downstream nodes that read it get
an ordinary write-before-read dependency.

and each node is represented via the following info:

```
enum DeviceType {
  CUDA(usize),
  CpuPinned,
  CpuPaged
}
struct BufInfo {
  name: Option<String>,
  device_type: DeviceType, 
  size: Quast,  // symbolic size in bytes
  elem_size: usize
}
```

`size` is a `Quast` (see `quast.rs`) — a quasi-affine expression over `VarId`
symbols registered on the builder. This lets buffer sizes depend on
parameters that are only bound at execution time; the concrete byte count is
recovered via `Quast::eval` against a `BTreeMap<VarId, i64>` binding of each
symbol. `Quast`'s internal sharing uses `Rc` so that structurally identical
sub-expressions (e.g. the same `4 * n` reused across many buffers) can be
cheaply cloned.

A `GraphBuilder` builds a graph, it exports the following interfaces:

```
fn register_symbol(&mut self, name: impl Into<String>) -> VarId { ... }
fn add_buf(&mut self, info: BufInfo) -> BufId {...}
fn insert_kernel(&mut self, module: ir::Module,
                 inputs: impl IntoIterator<Item = BufId>,
                 outputs: impl IntoIterator<Item = BufId>) { ... }
fn insert_blackbox_kernel(&mut self, name: ..., inputs: impl Iterator<Item = BufId>, outputs: ..., modifies: ..., f: ...) { ... }
fn insert_const(&mut self, buf: BufId, data: ConstBuf) { ... }
fn insert_memcpy(&mut self, ...) { ... }
fn insert_memset(&mut self, ...)
```

`register_symbol` allocates a fresh `VarId` and stores its printable name in
the builder's `symbols: BTreeMap<VarId, String>` map, which is preserved on
the built graph so downstream passes and dumps can render symbolic sizes
with their user-facing names.



