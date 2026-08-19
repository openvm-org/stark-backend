
A layout is something that describes how a particular buffer is represented on device. For a buffer of type `T` and shape `[N]`, we need to know if it's represented as shared memory, as the collection of a bunch of registers, or global memory.

If we decide it's represented as a bunch of registers, the descriptor need to map physical hardware indices `(spatial_id, lane_id, warp_id) -> logical_id` to the logical index (a lane is `threadIdx.x % WARP_SIZE` and a warp is `threadIdx.x / WARP_SIZE`, on Nvidia hardware `WARP_SIZE=32` and on AMD, `WARP_SIZE=64`; `spatial_id` represents multiple elements on the same thread).

If we decide it's represented as shared memory, then we need to decide the physical to logical mapping `(vectorized, bank, outer) -> logical_offset`. GPU shared memory is organized in banks of 32 bits. During execution, each bank can only service one lane per access, so if multiple lanes within the same warp access the same bank, the access becomes serialized.

The choice of the family of access functions is entirely arbitrary, I use [Linear Layouts](https://arxiv.org/pdf/2505.23819) from the paper, because it's simple enough to optimize and expressive enough for most GPU stuff.

The access function is the family of linear functions over $\mathbb Z_2$, where indices are represented as bit vectors. This means that all shapes has to be powers of 2.

I extend the paper slightly to allow for affine functions over $\mathbb Z_2$. And all results from the paper should generalize (affine functions are useful when we want to slice buffers).

For example, say the warp size is 2 and we have 2 warps. A particular encoding of a `u32[8]` buffer on registers is `(spatial_id, lane_id, warp_id) -> spatial_id + 2 * lane_id + 4 * warp_id`. This represents the buffer represented as `[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]`, where the first logical element is put on register 0 of lane 0 and warp 0, and the second is put on register 1 of lane 0 and warp 0, and so on...

The linear map of this corresponds to $$\begin{bmatrix}
& \text{spatial} & \text{lane} & \text{warp}\\
\text{logical least significant bit} & 1 & 0 & 0 \\ &0 & 1 & 0 \\ \text{logical most significant bit} &0 & 0 & 1
\end{bmatrix}$$
here I'm assuming that the linear map is $\mathbb Z_2^{\log_2(N_\text{spatial})} \times \mathbb Z_2^{\log_2(N_\text{lane})} \times \mathbb Z_2^{\log_2(N_\text{warp})} \to \mathbb Z_2^{\log_2(N_\text{logical})}$.

## Mapping to compilation 

The input to the layout inference pass is given an SSA-form IR that essentially consists of a sequence of parallel loops (denoted `par`) and abstract buffers. Each `par` has a sequence of reads and write accesses as well as a concrete iteration bound and loop inductive variable. Each `par` can be thought as `par(i, N, reads=[A_1[f_1(i,...),...], writes=[B_1[g_1(i,...)]])`. Where `N` is the concrete iteration bound, `i` is the induction variable. 

This sequence of `par` ops has the property that every buffer is written to exactly once by 1 `par` op, and every buf has concrete bounds that's a power of 2. 

The job of the layout infer pass is to assign each buffer a concrete layout that's either `Reg LinearLayout` or `Shared LinearLayout`, inserting `ConvertLayout` op as necessary that converts one layout to another. As well as determine the each `par` op's `par-attr`, which is the map from `(spatial_id, lane_id, warp_id) -> logical_id`, that determines how a generic `par` parallel loop is scheduled on the device. The logical id corresponds to the logical loop induction var `i`. This is similar to the register layout except this time `spatial_id` represents the loop iterations performed sequentially. For example, a  `par [8] |i| e(i)` with `par-attr = (spatial_id, lane_id, warp_id) -> spatial_id + 2 * lane_id + 4 * warp_id` would be scheduled like the following:
```
for spatial_id in 0..2 {
	e(spatial_id + 2 * lane_id + 4 * warp_id)
}
```


The algorithm for layout inference is as follows:

```
for each source par_i (par ops that reads from global buffers not produced by any other par), with concrete bound N_i, the `par-attr` is (s, l, w) -> S * s + w * WARP_SIZE + l. Where S = WARP_SIZE * NUM_WARPS, and the spatial repetition is N_i / S. 

deduce the layouts of the written to buffers once the par-attr is known (find_output_layout).

walk the list of pars in topological order. For each par, assume that the read buffers have known layouts. 

Then, call find_opt_par, to find the optimal par-attr given the read expressions. 
Once the par-attr for the current par op is known, call insert_convert_layout for each read from buffer A to convert the layout of the buffer to the right one:
Let P = (spatial, lane, warp) -> logical_idx be the par-attr.
let R be the layout of the buffer A.
let f be the read function. Then here are the cases:
- f is representable as an affine function.
  then required layout is Reg f ∘ P
- f is not an affine function
  then the required layout is Shared id
Then if the required layout is not equal to R, then we insert ConvertLayout(A, R, <required_layout>)

call find_output_layout to infer written buffers.

```

*note: if f is not an affine function then determining the required layout on registers may not be possible, because the analysis is either impossible if f is data-dependent or intractable for general f, since we need to optimize the convert layout*

There are three core pieces to this algorithm:
- `optimal_convert_layout_codegen`: the optimal codegen that uses a combination of warp shuffles and shared shuffles to convert one layout to another
- `find_output_layout`: deduce the output layout given `par-attr` and write expressions
- `find_opt_par`: deduce the optimal `par-attr` that minimizes convert layout cost

For `find_output_layout`, the reasoning is similar to the cases for reads. Let `P` be the determined `par-attr` and let `g` be the write expression for `B`. If `g` is affine then `g ∘ P` is a linear layout and the resulting layout is `Reg g ∘ P`, otherwise it's `Shared id`.

For `find_opt_par`, the algorithm is:
```
TODO
```

You can read more in the paper but I'll explain the optimal layout conversion because I had trouble understanding it. There are three cases:
1. `ConvertLayout(Reg f, Reg g)`
2. `ConvertLayout(Shared f, Reg g)`
3. `ConvertLayout(Reg f, Shared g)`

#### Case 1
For 1, the permutation is `C = g^{-1} ∘ f : (s, l, w) -> (s', l', w')` i.e. we want to move all elements on `(s, l, w)` to `(s', l', w')`. `g^{-1}` is the right inverse, meaning that `g ∘ g^{-1} = id`, this always exists since we assume all layouts are surjective. 

Then:
```

               →slot  →lane  →warp
       slot→ [ C_ss   C_sl   C_sw ]
C =    lane→ [ C_ls   C_ll   C_lw ]
       warp→ [ C_ws   C_wl   C_ww ]

```

There are 3 sub cases:

1. `C_sl, C_sw, C_ls, C_ws, C_wl = 0`, `C_ll, C_ww = id`:  this is a pure register permutation (within each thread), the cost is 0
2. `C_sw, C_lw, C_ws, C_wl = 0`, `C_ww = id`:  this requires register permutations and warp shuffles
3. otherwise, this requires shared memory.

For 3, it's then expressed as `ConvertLayout(Reg f, Shared o) -> ConvertLayout(Shared o, Reg g)`

1 is trivial since we can just rename registers. 

For 2, using the paper's notation. Let `A_thr = cols of f's lanes`.  I.e. 
```
     slot   lane   warp
f = [A_reg, A_thr, A_wrp] logical
```
Similarly for `B_thr = cols of g's lanes`. 

The number of elements that be be exchanged simultaneously is `n = |span(A_reg) ∩ span(B_reg)|`. Let `V` be the largest basis such that`span(V) ⊆ span(A_reg) ∩ span(B_reg)`,  where $2^{|V|} \times \text{bits\_per\_elem} = \text{WARP\_SIZE} \times \text{SHUFFLE\_BITS}$. 

Let $I = \text{span}(A_\text{thr}) \cap \text{span}(B_\text{thr})$ and $E, F$ be bases such that $\text{span}(A_\text{thr}) = \text{span}(E) \oplus I$,  $\text{span}(B_\text{thr}) = \text{span}(F) \oplus I$. 

Let $G = \{e_i \oplus f_i : e_i \in E, f_i \in F\}$. 

$I$ is the subspace of elements that belong to the same thread, and $G$ is the basis of the subspace such that every element belongs to a different thread of A and B.

#### Cases 2 and 3

See optimal swizzling algorithm of the paper
