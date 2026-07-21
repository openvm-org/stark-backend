# Graph IR 

The graph IR represents a raw computation graph over a GPU device. It has the following node types:

```
struct KernelNode {
  inputs: Vec<BufId>, outputs: Vec<BufId>, modifies: Vec<bool>, fn: Box<dyn Fn(&[Any]) -> Vec<Any>>,
  name: String
}

struct MemcpyNode {
  src: BufId, 
  dst: BufId
}

struct MemSetNode {
  node: BufId, 
  val: u32
}

enum GraphNode {
  Kernel(KernelNode),
  Memcpy(MemcpyNode),
  Memset(MemSetNode)
}

```

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
  size: usize,  // size in bytes
  elem_size: usize
}
```


A `GraphBuilder` builds a graph, it exports the following interfaces:

```
fn add_buf(&mut self, info: BufInfo) -> BufId {...}
fn insert_kernel(&mut self, inputs: impl Iterator<Item = BufId>, outputs: ..., modifies: ..., f: ...) { ... }
fn insert_memcpy(&mut self, ...) { ... }
fn insert_memset(&mut self, ...)
```



