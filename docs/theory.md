# Theory

## The problem

`smopt` solves

$$
\begin{aligned}
	\min_{X \in \mathbb{R}^{n\times p}} ~ & f(X) + r(X)\
	\text{subject to}~& X^\top X = I_p,
\end{aligned}
$$

where $f$ is smooth and $r$ is a convex, possibly nonsmooth regularizer
reachable through its proximal operator. The feasible set

$$
\mathcal{S}_{n,p} := \left\{X \in \mathbb{R}^{n\times p}: X^\top X = I_p \right\}
$$

is the **Stiefel manifold**, a smooth embedded submanifold of
$\mathbb{R}^{n \times p}$ of dimension $np - p(p+1)/2$.

## Constraint dissolving

Classical Riemannian methods keep every iterate exactly on
$\mathcal{S}_{n,p}$, which costs a matrix decomposition per step. The
solvers here instead *dissolve* the constraint: they work in the ambient
space $\mathbb{R}^{n \times p}$ and rely on a cheap map that pulls a
drifting iterate back towards the manifold.

That map is

$$
\mathcal{A}(X) =
\begin{cases}
\tfrac{3}{2} X - \tfrac{1}{2} X X^\top X, & \|X^\top X - I_p\|_F < \tfrac{1}{2},\[4pt]
X \left( \tfrac{1}{2}\left(X^\top X + I_p\right) \right)^{-1}, & \text{otherwise.}
\end{cases}
$$

The first branch is the second-order expansion of $X(X^\top X)^{-1/2}$
about a feasible point and costs only matrix products. The second branch
is exact and needs a Cholesky factorization of order $p$, which is cheap
because $p \ll n$ in the problems of interest. Both branches fix the
manifold: $\mathcal{A}(X) = X$ whenever $X^\top X = I_p$.

## Ingredients

Writing $\Phi(M) = (M + M^\top)/2$ for the symmetrizing operator, the
solvers are built from

| Map | Definition | Role |
| --- | --- | --- |
| $C(X)$ | $X^\top X - I_p$ | constraint violation |
| $\mathcal{J}_C(X)[\Lambda]$ | $X\,\Phi(\Lambda)$ | constraint Jacobian |
| $\mathcal{J}_C^\ast(X)[D]$ | $\Phi(X^\top D)$ | its adjoint |
| $\mathcal{J}_A(X)[G]$ | $G - X\,\Phi(X^\top G)$ | projected gradient |
| $\mathcal{A}(X)$ | above | feasibility restoration |

Feasibility is measured by $\|C(X)\|_F$ throughout.

## The solvers

### SLPG

The SLPG family takes a Barzilai-Borwein step along
$\mathcal{J}_A(X)[\nabla f(X)]$, applies the proximal operator of $r$,
and restores feasibility with $\mathcal{A}$. The step size uses one of
the two BB formulas

$$
\eta_k = \left| \frac{\langle S_k, Y_k\rangle}{\langle Y_k, Y_k \rangle} \right|
\qquad\text{or}\qquad
\eta_k = \left| \frac{\langle S_k, S_k\rangle}{\langle S_k, Y_k \rangle} \right|,
$$

with $S_k = X_k - X_{k-1}$ and $Y_k$ the corresponding change in the
search direction. The first few iterations use a conservative $c/L$
instead, where $L$ is estimated from the gradient at the starting point.

Three drivers are provided:

- `slpg_smooth` for $r \equiv 0$.
- `slpg` for a general $r$, whose constraint multiplier $\Lambda$ is
  tracked by an inner **Arrow-Hurwicz** iteration so that no penalty
  parameter has to be tuned.
- `slpg_l21` for $r(X) = \gamma\|X\|_{2,1}$, where both the prox and the
  multiplier

    $$
    \Lambda(X) = -\gamma\, X^\top \operatorname{diag}\!\left(\frac{1}{\|X_{i,:}\|_2}\right) X
    $$

    are available in closed form, so no inner iteration is needed. The
    $\ell_{2,1}$ norm sums the Euclidean norms of the rows of $X$ and
    therefore drives whole rows to zero, which is how sparse principal
    component analysis and related models select variables.

### pencf

`pencf` adds an explicit penalty to the search direction,

$$
\mathcal{G}(X) = \mathcal{J}_A(X)[\nabla f(X)] + \beta\, \mathcal{J}_C(X)[C(X)],
$$

and restores feasibility only once $\|C(X)\|_F$ exceeds $10^{-1}$,
capping $\|X\|_F$ at $1.001\sqrt{p}$ to keep the iteration bounded. The
default $\beta$ is $0.1\|\nabla f(X_0)\|_F$.

## Post-processing

Because the iterates are only approximately feasible, every solver
optionally rounds the final point onto the manifold with the orthogonal
polar factor

$$
\mathcal{P}(X) = U V^\top = X \left(X^\top X\right)^{-1/2},
\qquad X = U \Sigma V^\top,
$$

which is the nearest point of $\mathcal{S}_{n,p}$ in the Frobenius norm.
`smopt` computes it from a Jacobi eigendecomposition of the $p \times p$
matrix $X^\top X$ rather than from a singular value decomposition of
$X$, which keeps the cost at $O(np^2 + p^3)$.

## Implementation

Every formula on this page is evaluated in Fortran 77, including the
solver loops themselves. Python supplies the objective through a
callback, and the linear algebra — matrix products, Cholesky, the Jacobi
eigensolver, modified Gram-Schmidt — is hand-written in the same
sources, so the extension links against nothing but the Fortran runtime.
