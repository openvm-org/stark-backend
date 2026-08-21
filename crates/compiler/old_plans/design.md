# Cryptography GPU compiler

The goal of this compiler is to lower a light functional DSL that can express the vast majority of compute kernels in zkVMs with high performance on GPUs and low verbosity, with special optimizations for large fields like Bn254, and generic optimizations like fusion, tiling and layout management.

To this end, the compiler supports two key primitives:

```
compute [N] |i| {
  e(i)
}
```
and 
```
reduce [N] |i| {
  e(i)
}
```

that is the functional parallel for and associative reduction (more ops can be added in the future like scan). The language is pure and functional, everything is an expression operating on either scalar values or tensor values. 

`comppute` is semantically like a parallel map, if `e(i)` has type `T`, then `compute [N] |i| { e(i) }` has type `T[N]`, if `e(i)` has type `T[M]`, then `compute [N] |i| { e(i) }` has type `T[N, M]`.

`reduce` is semantically a parallel associative reduction, it preserves the type of `e(i)`


Aside from these primitives, there's the `if cond then e else e'` if guard, the indexing expression `A[i, j, k, ...]`, and standard elementwise operations like `add/sub/div/mul/type_cast`, etc., and the tuple expression `(a, ...)`. A compute with iteration bound `N` whose inner expression has tuple type `(T1, T2[N], T3[K, M], ...)` will have type `(T1[N], T2[N, N], T3[N, K, M], ...)`

For example, the standard matrix multiplication can be expressed as 
```
compute [N] |i| {
  compute [M] |j| {
    reduce [K] |k| {
      A[i, k] * B[k, j]
    }
  }
}
```

or equivalently:
```
compute [N * M] |i| {
  reduce [K] |k| {
    A[i / M, k] * B[k, i % M]
  }
}
```

except the first version has type `T[N, M]` and the second has type `T[N * M]`. All the shapes and iteration bounds here are constants.

To support multiple kernels or more complex programs, there's the let primitive that lets you bind expressions to variables.

```
let v = compute [N] |i| { ... } in 
let b = reduce [N] |i| { v[i] }
```

Finally, a complete program is a kernel module that declares the list of inputs, their types, and the expression that represents the entire sequence of computations.

```
kernel_mod(A: BB[512], B: BB[512]) {
  let a = ... in 
  ...
}
```

the job of the compiler is to lower this kernel module to a C++ file that exports the following interface:

```
extern "C" Prog* make_module();
extern "C" uint64_t scratch_size(Prog*);

extern "C" uint64_t num_outputs(Prog*);
extern "C" uint64_t output_size(Prog*, uint64_t);

extern "C" uint64_t num_inputs(Prog*);
extern "C" uint64_t input_size(Prog*, uint64_t);

extern "C" void set_input(Prog*, uint64_t, void*);
extern "C" void set_output(Prog*, uint64_t, void*);

extern "C" void set_scratch_buf(Prog*, void*);

extern "C" cudaError_t run(Prog*, cudaStream_t);
```

The job of the compiler is as follows:
1. canonicalize the expressions to the following canonical form: on the top-level is a sequence of let expressions, each let binding on the rhs is either a scalar expression or a compute. Each compute has a maximum of 1 compute nested within, or a reduce. For example:

the following top-level expressions are legal (as the rhs of a let bind):
```
compute [N] |i| { A[i] }

compute [N] |i| { 
  let a = compute [M] |j| { A[i, j] * 2 } in 
  let b = reduce [M] |j| { a[j] } in 
  b
}


```

the following are not legal:
```
compute [N] |i| {
  compute [N] |j| {
    compute [M] |k| {
      ...
    }
  }
}


compute [N] |i| {
  reduce [N] |j| {
    compute [M] |k| {
      ...
    }
  }
}
```

but it is always possible to canonicalize any arbitrary nesting into canonical form. 

Then after canonicalization in the outer scope we have an ordered sequence of let binds:
```
let (x1, x2, ...) = compute [N1] { ... } in 
let (y1, ...) = compute [N2] { ... } in 
...
```
and each `compute` is interpreted as a kernel, the outputs are arrays in global memory. Meanwhile the inner computes in the outer compute is interpreted as a thread block. Ie. the outer compute is parallel over the grid and the inner is parallel over the threads. 

At this point the compiler is free perform tiling and fusion and then to lower to a lower level representation, the KernelIR: each tensor gets annotated with a data layout, each compute gets annotated with a compute layout and each reduce is lowered to a sequence of computes and `for` loops. 

There are 3 kinds of data layouts: the register layout, the shared memory layout and the global memory layout. Shared and register layouts map logical indices into physical resource ids, for the register layout it maps logical indices into the local id, the lane id and the warp id. Global layouts are affine expressions that map logical indices to physical indices. 
Shared and register buffers are always powers of 2 and use triton's linear layout system: https://arxiv.org/html/2505.23819v1.

Compute layouts map physical compute ids into logical indices. This describes a factorization of the logical iteration domain into parts iterated over sequentially exploiting ILP and parts exploiting thread level parallelism. For example, `compute [N] |i| { e(i) }` can be factored into 
```
for T |i| {
  e(perm(i, threadIdx.x)) 
}
```
where `blockIdx.x == N / T` and `perm` is a bijection between `(Z_T, Z_{N/T})` and `Z_N`. For all inner computes over threads we can assume the bounds are powers of 2 and the bijection is anything describable via a linear layout, ie. a linear map `T: Z_2^k -> Z_2^k` between the indices represented as vectors in `Z_2`. 
For outer computes we cannot assume the buffers are powers of 2 sized, so we must allow a more general layout where expressions are polyhedral quasi-affine expressions mapping logical to physical indices.

The lower level IR must support a layout conversion op, and optimization passes to eliminate layout conversions. It must also deal with memory more explicitly: buffers are annotated with what address space they belong to. Buffers nested within a compute can belong to register, shared or global address space, while outer buffers can only belong to global. The lower level IR must also expression mutation semantics: in contrast to the higher level purely functional semantics, the lower level IR allows mutations to happen. 

After the compiler has layout information on all data buffers, all computes are annotated with compute layout information, and all `reduce`s are lowered to a combination of `for` and `compute`, the compiler can begin planning shared memory usage within each kernel, and global memory usage across kernels, insertion of synchronization primitives, and finally code generation. 

The compiler generates code in CUDA C++, which exports a C interface that can be DL-opened in Rust. In Rust for now the compiler exports the public IRBuilder through which users can build the higher level functional IR, as well as APIs for controlling codegen and exporting a `KernelModule` that wraps around the DL-opened dynamic library.

## Proc macro 

To make the programs easier to write in rust, write a proc macro 
```
kernel!(ctx, 
  let p = compute [N] |i| { ... };
  ...
```

that takes as first argument the ir builder and after that the kernel expression.

The proc macro should also support function calls, with the convention that a function `foo` called in the macro with n arguments has n+1 arguments, with the first argument being the ir builder.

The abstract syntax is as follows:

```
Expr e = let v = e;
       | compute [N] |i| { e }
       | reduce [N] |i| { e }
       | A[e]
       | e 'BIN_OP' e 
       | 'UNARY_OP' e 
       | 'FN_NAME' (e)
```

where `let v = e; ...` is equivalent to `let v = e in ...`. And to interpolate constants, use `#c`. 


## Higher level IR, lower level IR, codegen and passes

The compiler operates on two IRs, the higher level IR that has pure functional semantics, call it `TExpr`, and a lower level IR that has mutation semantics, call it `KernelIR`. The higher level IR is responsible for higher-level optimizations such as kernel fusion, algebraic rewrites and future optimizations like equality saturation. While the lower level IR is responsible for performing device specific optimizations like layout assignment, mapping computation onto the parallel hierarchy of the GPU, and finally codegen. 

`TExpr` is conceptually organized as follows:

```
TExpr e = let v = e in e 
       | compute [N] |i| { e }
       | reduce [N] |i| { e }
       | Elementwise

Elementwise e = A[e]  // index
              | Add e e 
              | ... // and other unary/binary ops
              | Var v // variables
              | (e)

```

Meanwhile the lower-level IR is organized as follows:

```
KernelIR = grid [N] |i| Block 
         | par [T] (par_attrs?) |i| Block 
         | A[Elementwise] = Elementwise 
         | for [N] |i| Block 
         | if Elementwise Block
         | Alloc A (alloc_attr?)
         | Sync
         | ConvertLayout A B // copy B to A, where these two may have different layouts

Block = (KernelIR+)
```

### Compilation Flow 

Starting with a `TExpr`:
1. type inference
2. perform algebraic rewrites, optimizations 
2.5 plan scratch buffers
3. canonicalize to standard form (at most one nested compute)
4. lower to `KernelIR` (initially attrs are empty)
  - lowers reduce to combination of for and compute

Starting with a `KernelIR`:
1. Layout inference, inferring `par_attr` and `alloc_attr`
2. various optimization passes, such as remove layout 
3. insert sync 
4. plan shared memory
4. codegen 


A `par_attr` is a linear layout mapping physical and sequential indices to the logical index. 

For example suppose we only have `64` threads but an inner `par` has `N = 128`. Then there's need for `sequential_size = 2`, and one such mapping from `(s, t) -> i`, ie. the sequential and thread map to logical index is `(s, t) -> s + t * 2`, another possibility is `(s, t) -> s * 64 + t`. Keep in mind that these are linear layouts, so all shapes are a power of 2. 

A linear layout is represented like:
```
struct LinearLayout {
  bases: Vec<u64>,
}
```
for a linear map that maps `T: Z_2^N -> Z_2^N`, we only need to record the bases `T(1), T(2), T(4), ...`, and we'll assume that `N < 2^64`.
We also need to keep track how indices are flattened and their representation in the input/output space. In the above example, `s \in {0, 1}` and `t \in {0, ..., 127}`, this induces a linear map from `Z_2^{256} -> Z_2^{256}`. We'll flatten `(s, t)` lexicographically where `s` is stored in the most significant bit, and `t` in the next 7 bits.

For the sake of simplicity, in the initial implementation infer all `par [N]` to use layout `(s, t) -> s * (N / T) + t`, where `T` is the number of threads. If `N < T` or `T` does not divide `N` insert an `if` guard for out of bounds indices.


A `alloc_attr` is a kind (shared/register/global) and a linear layout mapping logical index to physical index. For shared/global the physical index is the address, and for registers it's a tuple `(s, l, w)` where `s` is the spatial index (multiple elements stored on the same thread), `l` is the lane, and `w` is the warp id. Assume that all internal buffers have powers of 2 and their layouts are linear layouts, mapping the logical index `i` to their respective physical index. 

To simplify the initial implementation assume all internal buffers are shared buffers with layout `i -> i`. 

### Implementation

Since the expression/statements are separate enums, there should be multiple node id types representing each enum, there should also be two structs representing the IR for `TExpr` and `KernelIR`. 

For the initial implementation there should be distinct passes as functions on their respective IR managers
1. type inference 
2. canonicalize
3. lower_to_kernel_ir
4. layout_infer
5. insert_sync
6. codegen
7. plan shared mem


#### Layout infer pass 

After the lowering to kernel ir, the program consists of a sequence of primitive par ops, buffer declarations, loops and access relations. Each buffer is an abstract buffer written to by exactly one par op (aside from the global inputs and outputs). par blocks read and write to buffers using a Quast expression of the form `f(i, s)` where `i` ranges over the threads and `s` the local index, this Quast may be converted to a LinearLayout. 

The Layout infer pass aims to promote as many buffers and possible to use registers. It implements the following algorithm:

1. convert as much accesses to LinearLayouts as possible.
2. if the write access expression of a buffer is not convertible to a LinearLayout, it's address space is shared (and mapping from logical to physical is the identity)
3. if the write access expression is a LinearLayout, then its address space is Register. Since the write access expression is `Buf[f(i, s)] = ...`, then the layout, which is the map from physical, and spatial indices to logical index, is just `f`. 
4. if the read of a buffer is shared, with access expression `f(i, s)`, then insert a convert layout op that produces the register buffer from the shared buffer, where the register buffer has layout `f(i, s)`. 
5. if the read of a buffer has register layout `f(i, s)`, and the read expression is `g(i, s)`, then there are 2 cases:
  - `f == g`, then don't do anything
  - since `g` and `f` are both linear layouts, use the optimal convert layout algorithm from the triton paper:
    - determine if there's a convert layout between `f` to `g` that could be done with only local movements or warp shuffles, if so, then do so 
    - if not, then convert layout from `f` to `g` using shared memory


#### Smart memory planning 



