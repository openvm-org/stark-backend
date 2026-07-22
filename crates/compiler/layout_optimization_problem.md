# The layout inference problem

We have an abstract representation of a compute DAG as a bipartite graph where nodes in set A are resource buffers, nodes in set B are compute nodes, and edges between them are access functions.

For example:

```
res A;
res B;
res C;
res D;
res E;

compute(I=N1, S=N2, in={A[f_1(i, s)], B[f_2(i, s)]}, out={C[g_1(i, s)]})

```

