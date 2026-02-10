# Protocol Changes: Data-Parallel LogUp-GKR

This document details the modifications made to the LogUp-GKR protocol in `main.tex` to support efficient data-parallel execution.

## Summary of Changes

The primary update is the introduction of a **batched sumcheck** mechanism within the GKR rounds. This allows the verifier to simultaneously check recursive relations for multiple parallel instances (grid points) using a single sumcheck protocol per round, significantly improving verification efficiency.

The system parameters introduces a new $n_grid$ for logup.

### 1. Initial Claims and Verification

Since n_logup is determined at runtime, let n'_grid = min(n_logup, n_grid) to clamp for edge cases. 

**Old Protocol**: Implicitly assumed a single GKR execution or separate executions.
**New Protocol**:
- The prover also provides specific claimed values $v_{p, \vect g}, v_{q, \vect g}$ for the numerator and denominator terms for each grid point $\vect g in H_{n'_grid}$.
- The verifier directly computes the sum $\sum_{\vect g} v_{p,\vect g} / v_{q, \vect g}$ and checks that it equals 0.

### 2. Data-Parallel GKR Rounds with Batched Sumcheck

**Old Protocol**: Described a standard GKR reduction for a single instance.
**New Protocol**:
- **Objective**: Verify the recursive relations for all grid points $\vect g$ simultaneously.
- **Mechanism**: In each round $j$, a **single batched sumcheck instance** is used.
- **Batching**:
    - The verifier samples a single random challenge $\gamma \in \Fext$ (once, before round 1).
    - Weight vectors are initialized as interleaved powers: $\omega_p[g] = \gamma^{2g}$, $\omega_q[g] = \gamma^{2g+1}$.
    - The initial batched claim is $\sum_g (\omega_p[g] \cdot v_{p,g} + \omega_q[g] \cdot v_{q,g})$.
    - The weights are updated each round via the chain rule of the recursive gate relation, maintaining the invariant that the batched claim tracks the weighted sum of all per-grid claims.
    - Using interleaved powers of a single $\gamma$ ensures all $2 \cdot 2^{n'_{\text{grid}}}$ weights are distinct (in particular $\omega_p[0] = 1 \neq \gamma = \omega_q[0]$), which is required for soundness.
- **Reduction**: The sumcheck reduces the set of claims at layer $j-1$ to a new set of evaluation claims for $\hat p_{j, \vect g}, \hat q_{j, \vect g}$ at a single random point $\vect \rho^{(j-1)}$ (shared across all $\vect g$).
- **Next Round Challenge**: A random $\mu_j$ is used to reduce the two evaluation claims per grid point (at $0, \vect \rho^{(j-1)}$ and $1, \vect \rho^{(j-1)}$) to a single claim at $\vect \xi^{(j)} = (\mu_j, \vect \rho^{(j-1)})$.

### 3. Final Grid Check

**Old Protocol**: Not explicitly defined.
**New Protocol**:
- After the GKR rounds, the verifier holds a collection of claims $\{v_{p, \vect g}, v_{q, \vect g}\}_{\vect g}$ for the final layer polynomials at $\vect \xi_{\text{block}}$.
- To avoid checking each claim individually, a **grid check** is performed.
- The transcript samples a random $\vect \xi_{\text{grid}} \in \Fext^{n'_{\logup,\text{grid}}}$.
- The verifier checks the random linear combination:
  $\sum_{\vect g} v_{p, \vect g} \cdot \eq(\vect \xi_{\text{grid}}, \vect g) = \hat p(\vect \xi_{\text{grid}}, \vect \xi_{\text{block}})$
  (and similarly for $\hat q$).
- This reduces the collection of claims to single point evaluations of $\hat p$ and $\hat q$ at the concatenated point $\vect \xi = (\vect \xi_{\text{grid}}, \vect \xi_{\text{block}})$.

### 4. Random Challenge Consistency

**New Protocol**: Clarifies that the random vector $\vect \xi$ used in the subsequent batch constraint sumcheck is composed of $(\vect \xi_{\text{grid}}, \vect \xi_{\text{block}})$, ensuring consistency between the GKR execution and the constraint checking.

Full diff with more details:
```
diff --git a/main.pdf b/main.pdf
index 1f6f9bc..010c35e 100644
Binary files a/main.pdf and b/main.pdf differ
diff --git a/main.tex b/main.tex
index aee3874..673d273 100644
--- a/main.tex
+++ b/main.tex
@@ -184,6 +184,7 @@ A set $\Scr L \subseteq \bbF$ is \emph{smooth} if it is a multiplicative coset o
 \item The base field $\bbF$ and extension field $\Fext$.
 \item The univariate skip parameter $\ell$ and the generator of the domain $D$.
 \item The stacked dimension $n_\stack \ge 0$ which determines the stacked domain $\bbD_{n_\stack}$.
+\item The LogUp grid dimension $n_{\logup,\text{grid}} \ge 0$ which determines the number of parallel GKR instances.
 \item The hash function $\mf H$ used.
 \item The rate $\rho$ of the constrained Reed-Solomon code.  
 \item A smooth domain $\mathcal L \subset \bbF^\times$ of order $2^{\ell + n_\stack}/\rho$. All evaluation domains for Reed--Solomon codes are subsets of $\mathcal L$.
@@ -784,47 +785,64 @@ Let $\hat p, \hat q \in \Fext[Y_1,\dotsc, Y_{\ell + n_\logup}]$ be the multiline
 \label{eq:logup-input-q}
 \end{align}
 
-\begin{protocol}[LogUp, fractional sumcheck via GKR, multiple AIRs] \label{protocol:logup-frac-sumcheck}
+\begin{protocol}[LogUp, fractional sumcheck with data parallel GKR] \label{protocol:logup-frac-sumcheck}
 The vanishing of \eqref{eq:logup-vanishing} is equivalent to the vanishing of 
 \begin{equation} \label{eq:logup-frac-sumcheck}
 \sum_{\vect y \in\bbH_{\ell + n_\logup}} \frac{\hat p(\vect y)}{\hat q(\vect y)} = 0.
 \end{equation}
-There exists a layered circuit such that application of the GKR protocol to the layered circuit reduces the computation of the sum \eqref{eq:logup-frac-sumcheck} to the evaluation 
-of $\hat p(\vect \xi)$ and $\hat q(\vect \xi)$ at a randomly sampled $\vect \xi \in \Fext^{\ell + n_\logup}$.
+Let $n'_{\logup,\text{grid}} = \min(n_\logup, n_{\logup,\text{grid}})$ and $n_{\logup,\text{block}} = n_\logup - n'_{\logup,\text{grid}}$. 
+We view $\bbH_{\ell + n_\logup} \simeq \bbH_{n'_{\logup,\text{grid}}} \times \bbH_{\ell + n_{\logup,\text{block}}}$. 
+For $\vect g \in \bbH_{n'_{\logup,\text{grid}}}$, let $\hat p_{\vect g}(\vect Y') = \hat p(\vect g, \vect Y')$ and $\hat q_{\vect g}(\vect Y') = \hat q(\vect g, \vect Y')$ be the restrictions to the block indexed by $\vect g$.
+Then \eqref{eq:logup-frac-sumcheck} is equivalent to
+\begin{equation} \label{eq:logup-grid-sum}
+\sum_{\vect g \in \bbH_{n'_{\logup,\text{grid}}}} \left( \sum_{\vect y' \in \bbH_{\ell + n_{\logup,\text{block}}}} \frac{\hat p_{\vect g}(\vect y')}{\hat q_{\vect g}(\vect y')} \right) = 0.
+\end{equation}
+The term inside the parenthesis is a fractional sum over a hypercube of dimension $\ell + n_{\logup,\text{block}}$.
+There exists a layered circuit of depth $O(\ell + n_{\logup,\text{block}})$ that computes this fractional sum.
+The data-parallel GKR protocol provides the verifier with a stream of claimed values $v_{\vect g}$ corresponding to the inner sums for all $\vect g \in \bbH_{n'_{\logup,\text{grid}}}$.
+The verifier has direct access to these values and checks that $\sum_{\vect g} v_{\vect g} = 0$.
+The GKR protocol reduces the inner sum claims to streams of claimed evaluation values $v_{p, \vect g}$ and $v_{q, \vect g}$ for $\hat p_{\vect g}(\vect \xi_{\text{block}})$ and $\hat q_{\vect g}(\vect \xi_{\text{block}})$ respectively, for all $\vect g$, at a randomly sampled $\vect \xi_{\text{block}} \in \Fext^{\ell + n_{\logup,\text{block}}}$.
+Note that if $n_{\logup,\text{block}} = 0$, then the GKR protocol is skipped and we proceed directly to the evaluations with empty $\vect\xi_{\text{block}}$.
+
+To check the collection of evaluation claims (e.g., that $v_{p, \vect g} = \hat p_{\vect g}(\vect \xi_{\text{block}})$ for all $\vect g$), the transcript samples a random $\vect \xi_{\text{grid}} \in \Fext^{n'_{\logup,\text{grid}}}$.
+The verifier checks that
+\[ \sum_{\vect g \in \bbH_{n'_{\logup,\text{grid}}}} v_{p, \vect g} \cdot \eq(\vect \xi_{\text{grid}}, \vect g) = \hat p(\vect \xi_{\text{grid}}, \vect \xi_{\text{block}}) \]
+and similarly for $\hat q$. This reduces the claims to a single point evaluation of $\hat p$ and $\hat q$ at $\vect \xi = (\vect \xi_{\text{grid}}, \vect \xi_{\text{block}})$.
 \end{protocol}
 
-To complete the LogUp protocol, we explain how to reduce the evaluation claims on the ``input layer'' $\hat p(\vect \xi)$ and $\hat q(\vect \xi)$ to polynomial opening claims on the trace polynomials and their rotational convolutions. The idea is simply to massage \eqref{eq:logup-input-p} and \eqref{eq:logup-input-q} for $\vect Y = \vect \xi$ until we can apply (non-fractional) sumcheck.
-Let $\vect \xi = \vect\xi_1 \cat \vect\xi_2 \cat \vect\xi_3$ for $\vect\xi_1 \in \Fext^\ell, \vect\xi_2 \in \Fext^{\tilde n_\bT}, \vect\xi_3 \in \Fext^{n_\logup - \tilde n_\bT}$.
+To complete the LogUp protocol, we explain how to reduce the evaluation claim of $\hat p(\vect \xi)$ to polynomial opening claims. The same logic applies to $\hat q(\vect \xi)$.
+Recall that $\vect \xi = (\vect \xi_{\text{grid}}, \vect \xi_{\text{block}}) \in \Fext^{\ell + n_\logup}$. 
+Let $\vect \xi = \vect \xi_1 \cat \vect \xi_2 \cat \vect \xi_3$ for $\vect \xi_1 \in \Fext^\ell, \vect \xi_2 \in \Fext^{\tilde n_\bT}, \vect \xi_3 \in \Fext^{n_\logup - \tilde n_\bT}$.
 We use formula \eqref{eq:logup-piecewise} and the property of $\jmath'$ from Lemma~\ref{lem:piecewise-multilinear} to see that
 \[ 
-\eq(\vect\xi, \jmath_\bT(\omega_D^i, \vect x, (\hat\vsigma, \hat m, b))) = \eq(\vect\xi_1, \on{bin}_\ell(i)) \eq(\vect\xi_2, \vect x) \eq(\vect\xi_3, \vect b_{\bT, \hat\vsigma})  
+\eq(\vect \xi, \jmath_\bT(\omega_D^i, \vect x, (\hat\vsigma, \hat m, b))) = \eq(\vect \xi_1, \on{bin}_\ell(i)) \eq(\vect \xi_2, \vect x) \eq(\vect \xi_3, \vect b_{\bT, \hat\vsigma})  
 \]
 for some $\vect b_{\bT, \hat\vsigma} \in \bbH_{n_\logup - \tilde n_\bT}$. This element depends on $(\bT, A, I)$ and $(\hat\vsigma, \hat m, b) \in I$ but we omitted some notation for brevity.
-We interpolate $(\omega_D^i, \vect x) \mapsto \eq(\xi_1, \on{bin}_\ell(i))\eq(\xi_2,\vect x)$ into a prismalinear polynomial by interpolating over $D$ to get
+We interpolate $(\omega_D^i, \vect x) \mapsto \eq(\vect \xi_1, \on{bin}_\ell(i))\eq(\vect \xi_2,\vect x)$ into a prismalinear polynomial by interpolating over $D$ to get
 \begin{equation} \label{eq:lagrange-univariate-sharp}
-\eq_{\vect\xi_1,\vect\xi_2}^\sharp (Z, \vect X):= \left(\sum_{i \in [2^\ell]} \eq_D(Z,\omega_D^i)\eq_{\bbH_\ell}(\vect\xi_1, \on{bin}_\ell(i))\right) \eq_{\bbH_{\tilde n_\bT}}(\vect \xi_2, \vect X)
+\eq_{\vect \xi_1,\vect \xi_2}^\sharp (Z, \vect X):= \left(\sum_{i \in [2^\ell]} \eq_D(Z,\omega_D^i)\eq_{\bbH_\ell}(\vect \xi_1, \on{bin}_\ell(i))\right) \eq_{\bbH_{\tilde n_\bT}}(\vect \xi_2, \vect X)
 \end{equation} 
 which is a polynomial in $\bbF[Z, \vect X]$.
 
-We conclude that $\hat p(\vect\xi)$ and $\hat q(\vect\xi) - \alpha$ can both be written in the form 
+We conclude that $\hat p(\vect \xi)$ and $\hat q(\vect \xi) - \alpha$ can both be written in the form 
 \[ 
 \sum_{(\bT,A,I)\in \Scr T}  
-\left(\sum_{\vect z \in \bbD_{\tilde n_\bT}} \eq_{\vect\xi_1,\vect\xi_2}^\sharp(\vect z)
-\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect\xi_3, \vect b_{\bT, \hat\vsigma}) \cdot\hat C(\wt\bT(\vect z), \wt\bT_\rot(\vect z))
+\left(\sum_{\vect z \in \bbD_{\tilde n_\bT}} \eq_{\vect \xi_1,\vect \xi_2}^\sharp(\vect z)
+\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect \xi_3, \vect b_{\bT, \hat\vsigma}) \cdot\hat C(\wt\bT(\vect z), \wt\bT_\rot(\vect z))
 \right)
 \]
 Recall that we constructed $\jmath$ so that $\vect b_{\bT, \hat\vsigma}$ is efficiently computable by the verifier.
 We can evaluate sums of the above form using batch sumcheck:
 
 \begin{protocol}[LogUp, input layer evaluation via batch sumcheck] \label{protocol:logup-input-batch-sumcheck}
-Fix $\vect \xi = \vect\xi_1 \cat \vect\xi_2 \cat \vect\xi_3$ for $\vect\xi_1 \in \Fext^\ell, \vect\xi_2 \in \Fext^{\tilde n_\bT}, \vect\xi_3 \in \Fext^{n_\logup - \tilde n_\bT}$.
-The evaluations of $\hat p(\vect\xi)$ and $\hat q(\vect\xi) - \alpha$ are equivalent to the computations of
+Fix $\vect \xi = \vect \xi_1 \cat \vect \xi_2 \cat \vect \xi_3$ for $\vect \xi_1 \in \Fext^\ell, \vect \xi_2 \in \Fext^{\tilde n_\bT}, \vect \xi_3 \in \Fext^{n_\logup - \tilde n_\bT}$.
+The evaluations of $\hat p(\vect \xi)$ and $\hat q(\vect \xi) - \alpha$ are equivalent to the computations of
 \begin{align*}
-\on{sum}_{\hat p,\bT,I} &= \sum_{\vect z \in \bbD_{\tilde n_\bT}} \eq_{\vect\xi_1,\vect\xi_2}^\sharp(\vect z) 
-\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect\xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
+\on{sum}_{\hat p,\bT,I} &= \sum_{\vect z \in \bbD_{\tilde n_\bT}} \eq_{\vect \xi_1,\vect \xi_2}^\sharp(\vect z) 
+\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect \xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
 \hat m(\wt\bT(\vect z), \wt\bT_\rot(\vect z)) \\
-\on{sum}_{\hat q,\bT,I} &= \sum_{\vect z \in \bbD_{\tilde n_\bT}} \eq_{\vect\xi_1,\vect\xi_2}^\sharp(\vect z) 
-\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect\xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
+\on{sum}_{\hat q,\bT,I} &= \sum_{\vect z \in \bbD_{\tilde n_\bT}} \eq_{\vect \xi_1,\vect \xi_2}^\sharp(\vect z) 
+\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect \xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
 h_\beta(\hat\vsigma \cat b)(\wt\bT(\vect z), \wt\bT_\rot(\vect z))
 \end{align*}
 for each $(\bT,A,I)\in \Scr T$ together with the computations of 
@@ -836,26 +854,25 @@ for each $(\bT,A,I)\in \Scr T$ together with the computations of
 The computations of \eqref{eq:logup-input-finalsum} are done directly by the verifier.
 The batched sumcheck protocol can be applied to reduce the computations of $\on{sum}_{\hat p,\bT,I}$ and $\on{sum}_{\hat q,\bT,I}$ for all $(\bT,A,I)\in \Scr T$ to the evaluations of 
 \begin{align}
-& \eq_{\vect\xi_1,\vect\xi_2}^\sharp(\vect r_{\tilde n_\bT}) 
-\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect\xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
-\hat m(\wt\bT(\vect r_{\tilde n_\bT}), \wt\bT_\rot(\vect r_{\tilde n_\bT})) 
+& \eq_{\vect \xi_1,\vect \xi_2}^\sharp(\vect r'_{\tilde n_\bT}) 
+\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect \xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
+\hat m(\wt\bT(\vect r'_{\tilde n_\bT}), \wt\bT_\rot(\vect r'_{\tilde n_\bT})) 
 \label{eq:interaction-constraint-eval-num}
 \\
-& \eq_{\vect\xi_1,\vect\xi_2}^\sharp(\vect r_{\tilde n_\bT}) 
-\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect\xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
-h_\beta(\hat\vsigma \cat b)(\wt\bT(\vect r_{\tilde n_\bT}), \wt\bT_\rot(\vect r_{\tilde n_\bT}))
+& \eq_{\vect \xi_1,\vect \xi_2}^\sharp(\vect r'_{\tilde n_\bT}) 
+\sum_{(\hat\vsigma, \hat m, b) \in I} \eq(\vect \xi_3, \vect b_{\bT, \hat\vsigma}) \cdot
+h_\beta(\hat\vsigma \cat b)(\wt\bT(\vect r'_{\tilde n_\bT}), \wt\bT_\rot(\vect r'_{\tilde n_\bT}))
 \label{eq:interaction-constraint-eval-denom}
 \end{align}
-for each $(\bT,I)$ with respect to a shared random $\vect r \in \Fext^{n_{\Scr T} + 1}$. 
-The random $\vect r \in \Fext^{n_{\Scr T}+1}$ is sampled and used across the parallel sumchecks, where $\vect r_{\tilde n_\bT}$ denotes its truncation to $\Fext^{\tilde n_\bT +1}$.
-
+for each $(\bT,I)$ with respect to a shared random $\vect r' \in \Fext^{n_{\Scr T} + 1}$. 
+The random $\vect r' \in \Fext^{n_{\Scr T}+1}$ is sampled and used across the parallel sumchecks, where $\vect r'_{\tilde n_\bT}$ denotes its truncation to $\Fext^{\tilde n_\bT +1}$.
 The evaluations of \eqref{eq:interaction-constraint-eval-num} and \eqref{eq:interaction-constraint-eval-denom} reduce to the evaluation of 
-\[ \hat\bT(\vect r_{n_\bT}) \text{ and }
-(\hat\bT \star \hat\kappa_\rot)(\vect r_{n_\bT})
+\[ \hat\bT(\vect r'_{n_\bT}) \text{ and }
+(\hat\bT \star \hat\kappa_\rot)(\vect r'_{n_\bT})
 \] 
 by requiring the verifier to directly evaluate 
 \eqref{eq:interaction-constraint-eval-num} and \eqref{eq:interaction-constraint-eval-denom}
- in terms of $\vect r_{\tilde n_\bT}, \hat\bT(\vect r_{n_\bT}), \hat\bT_\rot(\vect r_{n_\bT})$, where we let $\vect r_{n_\bT} = r_0^{2^{-n_\bT}}$ for $n_\bT < 0$ so that $\wt\bT(\vect r_{\tilde n_\bT}) = \hat\bT(\vect r_{n_\bT})$.
+ in terms of $\vect r'_{\tilde n_\bT}, \hat\bT(\vect r'_{n_\bT}), \hat\bT_\rot(\vect r'_{n_\bT})$, where we let $\vect r'_{n_\bT} = (r'_0)^{2^{-n_\bT}}$ for $n_\bT < 0$ so that $\wt\bT(\vect r'_{\tilde n_\bT}) = \hat\bT(\vect r'_{n_\bT})$.
 \end{protocol}
 We do not apply this protocol directly in the proof system. Instead, we combine it with the ZeroCheck protocol and run Protocol~\ref{protocol:batch-constraint-sumcheck} below.
 
@@ -1004,31 +1021,47 @@ We outline the full non-interactive STARK protocol here. We recall (cf.~\S\ref{s
 \end{enumerate}
 \item The fractional sumcheck protocol from Protocol~\ref{protocol:logup-frac-sumcheck} is applied. 
 The transcript samples random $\alpha,\beta \in \Fext$ to be used in the denominator terms of the fractional sum.
-The GKR layered circuit is described in \cite[Section 3.1]{LogUp-GKR}. The witness for the layered circuit consists of functions $p_j,q_j : \bbH_j \to \Fext$ for layers $j=0,\dotsc,\ell+n_\logup$. The witness functions are recursively defined starting from layer $\ell+n_\logup$ with $(p_{\ell+n_\logup}, q_{\ell+n_\logup})$ defined as the evaluations of $(\hat p, \hat q)$ and proceeding down to layer $0$ using the recursive definition\footnote{The choice to evaluate $0,1$ from the left makes no theoretical difference, but leads to a better memory layout in practical implementations where hypercube coordinates are represented as little endian integers.} 
+
+The prover provides claimed values $v_{p, \vect g}, v_{q, \vect g} \in \Fext$ for the inner sum numerator and denominator terms for all grid points $\vect g \in \bbH_{n'_{\logup,\text{grid}}}$. The verifier checks that $v_{q, \vect g} \ne 0$ for all $\vect g$ and that $\sum_{\vect g} v_{p, \vect g}/v_{q, \vect g} = 0$.
+The GKR layered circuit is described in \cite[Section 3.1]{LogUp-GKR}. The witness for the layered circuit consists of functions $p_{j,\vect g},q_{j,\vect g} : \bbH_j \to \Fext$ for layers $j=0,\dotsc,\ell+n_{\logup,\text{block}}$ and each grid point $\vect g$. The witness functions are recursively defined starting from layer $\ell+n_{\logup,\text{block}}$ with $(p_{\ell+n_{\logup,\text{block}},\vect g}, q_{\ell+n_{\logup,\text{block}},\vect g})$ defined as the evaluations of $(\hat p_{\vect g}, \hat q_{\vect g})$ and proceeding down to layer $0$ using the recursive definition\footnote{The choice to evaluate $0,1$ from the left makes no theoretical difference, but leads to a better memory layout in practical implementations where hypercube coordinates are represented as little endian integers.}
 \begin{align*}
-  p_{j-1}(\vect y) &= p_j(0, \vect y) q_j(1, \vect y) + p_j(1, \vect y) q_j(0, \vect y) \\
-  q_{j-1}(\vect y) &= q_j(0, \vect y) q_j(1, \vect y)
+  p_{j-1,\vect g}(\vect y) &= p_{j,\vect g}(0, \vect y) q_{j,\vect g}(1, \vect y) + p_{j,\vect g}(1, \vect y) q_{j,\vect g}(0, \vect y) \\
+  q_{j-1,\vect g}(\vect y) &= q_{j,\vect g}(0, \vect y) q_{j,\vect g}(1, \vect y)
 \end{align*}
 
-The GKR protocol proceeds in rounds $j=1,\dotsc,\ell+n_\logup$. In round $j$, the prover starts with the MLEs $\hat p_{j-1}, \hat q_{j-1}$ of $p_{j-1}, q_{j-1}$. 
-The verifier has evaluation claims of $\hat p_{j-1}(\vect \xi^{(j-1)}), \hat q_{j-1}(\vect \xi^{(j-1)})$ for a randomly sampled $\vect \xi^{(j-1)} \in \Fext^{j-1}$ from the last round. Note $\vect \xi^{(0)}$ is the empty vector and $\hat p_0,\hat q_0$ are constants. The value $\frac{p_0}{q_0}$ is the claimed fractional sum.
+The GKR protocol proceeds in rounds $j=1,\dotsc,\ell+n_{\logup,\text{block}}$. In round $j$, the prover starts with the MLEs $\hat p_{j-1,\vect g}, \hat q_{j-1,\vect g}$ of $p_{j-1,\vect g}, q_{j-1,\vect g}$.
+The verifier has evaluation claims of $v_{p, j-1, \vect g}, v_{q, j-1, \vect g}$ for $\hat p_{j-1, \vect g}(\vect \xi^{(j-1)}), \hat q_{j-1, \vect g}(\vect \xi^{(j-1)})$ for a randomly sampled $\vect \xi^{(j-1)} \in \Fext^{j-1}$ from the last round. Note $\vect \xi^{(0)}$ is the empty vector.
+For the base case $j=1$, the claims for $p_{0,\vect g}, q_{0,\vect g}$ are derived from $v_{p, \vect g}, v_{q, \vect g}$. specifically the claim is that $p_{0,\vect g} = v_{p, \vect g}$ and $q_{0,\vect g} = v_{q, \vect g}$.
 \begin{enumerate}
   \item
-In round $j$, the prover and verifier apply the batch sumcheck protocol to the MLEs $\hat p_{j-1}, \hat q_{j-1}$ using the equalities 
-\begin{align*}
-  \hat p_{j-1}(\vect Y) &= \sum_{\vect y \in \bbH_{j-1}} \eq_{j-1}(\vect Y, \vect y) \cdot \bigl(\hat p_j(0, \vect y) \hat q_j(1, \vect y) + \hat p_j(1, \vect y) \hat q_j(0, \vect y)\bigr) \\
-  \hat q_{j-1}(\vect Y) &= \sum_{\vect y \in \bbH_{j-1}} \eq_{j-1}(\vect Y, \vect y) \cdot \bigl(\hat q_j(0, \vect y) \hat q_j(1, \vect y)\bigr)
-\end{align*}
-In the batch sumcheck, the transcript samples randomness $\lambda_j\in \Fext$ for batching and the protocol reduces the evaluation claims of $\hat p_{j-1}(\vect \xi^{(j-1)}), \hat q_{j-1}(\vect \xi^{(j-1)})$ to the evaluation claims of 
+In round $j$, the prover and verifier apply a {\bf single batched sumcheck instance} to verify the recursive relations for all $\vect g$ simultaneously.
+The verifier has evaluation claims $\{ \hat p_{j-1,\vect g}(\vect \xi^{(j-1)}), \hat q_{j-1,\vect g}(\vect \xi^{(j-1)}) \}_{\vect g}$.
+The batched sumcheck reduces the verification of these claims to the verification of a random linear combination of the next layer's values.
+Specifically, the verifier samples a random challenge $\gamma_j \in \Fext$ (or a vector of challenges if needed) and considers the combination polynomial
+\[
+  P_{j-1}(\vect Y) = \sum_{\vect g \in \bbH_{n'_{\logup,\text{grid}}}} \left( \gamma_j^{2\vect g} \cdot \hat p_{j-1,\vect g}(\vect Y) + \gamma_j^{2\vect g+1} \cdot \hat q_{j-1,\vect g}(\vect Y) \right)
+\]
+where we identify the grid points $\vect g$ with integers $0, \dots, 2^{n'_{\logup,\text{grid}}}-1$.
+Using the recursive relations for $\hat p_{j-1,\vect g}(\vect Y)$ and $\hat q_{j-1,\vect g}(\vect Y)$, $P_{j-1}(\vect Y)$ is a sum over $\bbH_{j-1}$ of a polynomial that depends on values at layer $j$.
+At the end of the sumcheck protocol, the verifier checks the value of $P_{j-1}(\vect \xi^{(j-1)})$ against the claimed linear combination of evaluations.
+The sumcheck reduces this check to a claim about $\hat p_{j, \vect g}, \hat q_{j, \vect g}$ at a new random point $\vect \rho^{(j-1)}$.
+The protocol reduces the set of evaluation claims $\{ \hat p_{j-1,\vect g}(\vect \xi^{(j-1)}), \hat q_{j-1,\vect g}(\vect \xi^{(j-1)}) \}_{\vect g}$
+to a new set of evaluation claims 
 \begin{equation}\label{eq:gkr-eval-claims} 
-  \hat p_j(0, \vect \rho^{(j-1)}), \hat p_j(1, \vect \rho^{(j-1)}), \hat q_j(0, \vect \rho^{(j-1)}), \hat q_j(1, \vect \rho^{(j-1)}) 
+  \{ \hat p_{j,\vect g}(0, \vect \rho^{(j-1)}), \hat p_{j,\vect g}(1, \vect \rho^{(j-1)}), \hat q_{j,\vect g}(0, \vect \rho^{(j-1)}), \hat q_{j,\vect g}(1, \vect \rho^{(j-1)}) \}_{\vect g}
 \end{equation}
-for a randomly sampled $\vect \rho^{(j-1)} \in \Fext^j$.
-\item Observe that $\hat p_j(\bullet, \vect \rho^{(j-1)}), \hat q_j(\bullet, \vect \rho^{(j-1)})$ are linear polynomials. The transcript observes the claimed linear polynomials in terms of their evaluation claims \eqref{eq:gkr-eval-claims}.
-\item The transcript samples another random $\mu_j \in \Fext$. The protocol uses $\mu_j$ to reduce the evaluation claims of $\hat p_j(0, \vect \rho^{(j-1)}), \hat p_j(1, \vect \rho^{(j-1)})$ to the evaluation claim of $\hat p_j(\vect \xi^{(j)})$ with $\vect \xi^{(j)} = (\mu_j,\vect \rho^{(j-1)})$. Similarly, it reduces the denominator evaluation claim to the evaluation claim of $\hat q_j(\vect \xi^{(j)})$. Now proceed to the next round of GKR.
+for a single randomly sampled $\vect \rho^{(j-1)} \in \Fext^j$ (shared across all $\vect g$).
+\item Observe that $\hat p_{j,\vect g}(\bullet, \vect \rho^{(j-1)}), \hat q_{j,\vect g}(\bullet, \vect \rho^{(j-1)})$ are linear polynomials. The transcript observes the set of claimed linear polynomials $\{\hat p_{j,\vect g}(\bullet, \vect \rho^{(j-1)}), \hat q_{j,\vect g}(\bullet, \vect \rho^{(j-1)}) \}_{\vect g}$ in terms of their evaluation claims \eqref{eq:gkr-eval-claims}.
+\item The transcript samples another random $\mu_j \in \Fext$. The protocol uses $\mu_j$ to reduce the evaluation claims of $\hat p_{j,\vect g}(0, \vect \rho^{(j-1)}), \hat p_{j,\vect g}(1, \vect \rho^{(j-1)})$ to the evaluation claim of $\hat p_{j,\vect g}(\vect \xi^{(j)})$ with $\vect \xi^{(j)} = (\mu_j,\vect \rho^{(j-1)})$. Similarly, it reduces the denominator evaluation claim to the evaluation claim of $\hat q_{j,\vect g}(\vect \xi^{(j)})$. Now proceed to the next round of GKR.
 \end{enumerate} % end GKR
 
-\item Prover and verifier apply a \emph{batch constraint sumcheck} for ZeroCheck and the evaluation claims of the input layer of the LogUp GKR circuit. Prover and verifier sample two random elements $\lambda,\mu \in \Fext$ for batching purposes. The $\lambda$ is used for algebraic batching of constraint polynomials per AIR. The $\mu$ is the batching factor in the batch sumcheck. The protocol can also use a single random element $\lambda$ for both batching purposes, but we distinguish them to improve the soundness of the protocol. The batch sumcheck is applied as described in Protocol \ref{protocol:batch-constraint-sumcheck}, where the random vector $\vect \xi \in \Fext^{\ell + n_{\textnormal{global}}}$ is set to equal the randomly sampled $\vect \xi^{(\ell + n_\logup)}$ from the last round of GKR together with additional sampled elements if $n_\logup < n_{\textnormal{global}}$. 
+\item After the GKR rounds, the verifier has claims $v_{p, \vect g}, v_{q, \vect g}$ for $\hat p_{\vect g}(\vect \xi_{\text{block}})$ and $\hat q_{\vect g}(\vect \xi_{\text{block}})$, where $\vect \xi_{\text{block}} \in \Fext^{\ell + n_{\logup,\text{block}}}$ is the final random challenge from the GKR.
+The transcript samples a random $\vect \xi_{\text{grid}} \in \Fext^{n'_{\logup,\text{grid}}}$.
+The verifier checks that
+\[ \sum_{\vect g \in \bbH_{n'_{\logup,\text{grid}}}} v_{p, \vect g} \cdot \eq(\vect \xi_{\text{grid}}, \vect g) = \hat p(\vect \xi_{\text{grid}}, \vect \xi_{\text{block}}) \]
+and similarly for $\hat q$. This reduces the collection of claims to single point evaluations of $\hat p$ and $\hat q$ at $\vect \xi = (\vect \xi_{\text{grid}}, \vect \xi_{\text{block}})$.
+
+\item Prover and verifier apply a \emph{batch constraint sumcheck} for ZeroCheck and the evaluation claims of the input layer of the LogUp GKR circuit. Prover and verifier sample two random elements $\lambda,\mu \in \Fext$ for batching purposes. The $\lambda$ is used for algebraic batching of constraint polynomials per AIR. The $\mu$ is the batching factor in the batch sumcheck. The protocol can also use a single random element $\lambda$ for both batching purposes, but we distinguish them to improve the soundness of the protocol. The batch sumcheck is applied as described in Protocol \ref{protocol:batch-constraint-sumcheck}, where the random vector $\vect \xi \in \Fext^{\ell + n_{\textnormal{global}}}$ is set to equal $(\vect \xi_{\text{grid}}, \vect \xi_{\text{block}})$ (padded with additional sampled random elements if needed), where $\vect \xi_{\text{block}}$ is the challenge from the last round of GKR and $\vect \xi_{\text{grid}}$ is the challenge from the grid check. 
 
 In total the batch sumcheck batches $3 \abs{\Scr T}$ polynomials. If we let $s_{p, \bT}, s_{q, \bT}, s_{\text{zc}, \bT}$ denote the sumcheck claims associated with a single trace $\bT \in \Scr T$ for the LogUp numerator claim, LogUp denominator claim, and ZeroCheck claim, then the total ordering of the batch sumcheck is $s_{p, \bT_1}, s_{q, \bT_1}, \dotsc, s_{p, \bT_{\abs{\Scr T}}}, s_{q, \bT_{\abs{\Scr T}}}, s_{\text{zc}, \bT_1}, \dotsc, s_{\text{zc}, \bT_{\abs{\Scr T}}}$ where the total ordering of $\Scr T$ is in descending order of $n_\bT$ (with tie breaks determined by ordering of the AIRs in the verifying key).
 
```
