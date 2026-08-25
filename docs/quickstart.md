# Quickstart

## A nonlinear eigenvalue problem

The following problem is the standard smoke test for Stiefel solvers:

$$
\min_{X \in \mathcal{S}_{n, p}} ~ \frac{1}{2}\mathrm{tr}(X^\top L X) + \frac{\alpha}{4} \rho^\top L^{\dagger} \rho,
$$

where $\rho = \mathrm{Diag}(XX^\top)$ and $L^{\dagger}$ is the
pseudo-inverse of the positive definite $L$. The cost function and its
**Euclidean gradient** are

$$
\begin{aligned}
	 f(X) ={}& \frac{1}{2}\mathrm{tr}(X^\top L X) + \frac{\alpha}{4} \rho^\top L^{\dagger} \rho,\
	\nabla f(X) ={}& LX + \alpha \, \mathrm{diag}(L^{\dagger}\rho)X.
\end{aligned}
$$

Taking $L$ tridiagonal, so that $L^{\dagger} = L^{-1}$:

```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from smopt import SLPG_smooth, Stiefel

n, p, alpha = 1000, 10, 1.0
M = Stiefel(n, p)

L = diags([-1, 2, -1], [1, 0, -1], shape=(n, n)).tocsc()


def obj_fun(X):
    LX = L @ X
    rho = np.sum(X * X, 1)
    Lrho = spsolve(L, rho)
    fval = 0.5 * np.sum(X * LX) + (alpha / 4) * np.sum(rho * Lrho)
    grad = LX + alpha * Lrho[:, np.newaxis] * X
    return fval, grad


X, out = SLPG_smooth(obj_fun, M)
```

!!! note
    SciPy is only used to build this example's data; `smopt` itself
    depends on NumPy alone.

## Step by step

### 1. Fix the manifold

The dimensions are carried by a [`Stiefel`](api.md) instance, which also
exposes the geometric maps the solvers use:

```python
from smopt import Stiefel

M = Stiefel(1000, 10)
```

### 2. Define the objective

A solver expects **one** callable returning the value and the Euclidean
gradient together:

```python
def obj_fun(X):
    ...
    return fval, grad
```

Returning both at once is usually far cheaper than computing them
separately, even with caching, so `smopt` asks for them in a single
call. `X` arrives as an `(n, p)` array and `grad` must have the same
shape.

### 3. Run a solver

```python
X, out = SLPG_smooth(obj_fun, M)
```

`X` is the solution and `out` is a dictionary of log information:

| Key | Meaning |
| --- | --- |
| `fvals`, `kkts`, `feas` | per-iteration histories |
| `fval`, `kkt`, `fea` | the final objective, stationarity and feasibility |
| `beta` | the penalty `PenCF` actually used |

## Nonsmooth problems

### A regularizer of your own

Pass any `prox(X, eta)` that minimizes
$\|Y - X\|_F^2 / (2\eta) + r(Y)$. Here is $\ell_1$ regularization built
from the operator shipped with the package:

```python
from smopt import SLPG, prox_l1

gamma = 0.05
X, out = SLPG(obj_fun, M, prox=lambda X, eta: prox_l1(X, eta, gamma=gamma))
```

### Row sparsity

For $r(X) = \gamma\|X\|_{2,1}$ use the dedicated driver, which knows the
prox and the constraint multiplier in closed form:

```python
from smopt import SLPG_l21

X, out = SLPG_l21(obj_fun, M, gamma=1.0)
```

Whole rows of `X` are driven to zero, which selects variables:

```python
import numpy as np

live = np.linalg.norm(X, axis=1) > 1e-6
print(f"{live.sum()} of {len(live)} rows survive")
```

## Choosing a solver

| Solver | Use when |
| --- | --- |
| `SLPG_smooth` | the objective is smooth |
| `SLPG` | there is a nonsmooth term with a known prox |
| `SLPG_l21` | the nonsmooth term is $\gamma\|X\|_{2,1}$ |
| `PenCF` | a constraint dissolving penalty method is preferred |

## Common options

Every solver accepts:

```python
X, out = SLPG_smooth(
    obj_fun,
    M,
    Xinit=None,  # starting point; random feasible point if omitted
    maxit=100,  # iteration budget
    gtol=1e-5,  # stationarity tolerance
    post_process=True,  # round the answer onto the manifold
    verbosity=2,  # 0 silent, 1 final lines, 2 periodic
)
```

`PenCF` takes the starting point first and adds `beta`:

```python
from smopt import PenCF

X, out = PenCF(Xinit, obj_fun, M, beta=None)
```
