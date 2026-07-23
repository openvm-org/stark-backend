# The layout inference problem

We have an abstract representation of a compute DAG as a bipartite graph where nodes in set A are resource buffers, nodes in set B are compute nodes, and edges between them are access functions.

For example:

```
T = N1 // number of threads (constant)
G = ... // grid size, constant

in: [A, B]
out: [F]

res A: [M1]; // shapes are constant
res B: [M2];
res C: [M3];
res D: [M4];
res E: [M5];
res F: [M6];

compute(S=N2, in={A[f_1(i, s)], B[f_2(i, s)]}, out={C[g_1(i, s)]})
compute(S=N3, in={C[f_3(i, s)], B[f_4(i, s)]}, out={D[g_2(i, s)], E[g_3(i, s)]})
compute(S=N4, in={E[f_5(i, s)], D[f_6(i, s)]}, out={F[g_4(i, s)]})

```




