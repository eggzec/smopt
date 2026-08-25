# API Reference

Everything below is re-exported at the top level, so
`from smopt import slpg_smooth` and
`from smopt.solver import slpg_smooth` are equivalent.

## Manifold

### `smopt.Stiefel`

```python
Stiefel(n: int, p: int)
```

The manifold $\{X \in \mathbb{R}^{n\times p} : X^\top X = I_p\}$. Carries
the dimensions and exposes the geometric maps; each one is evaluated by
the Fortran 77 core. Raises `ValueError` if the dimensions are not
positive or if `p > n`.

| Attribute | Meaning |
| --- | --- |
| `dim` | `n * p`, the dimension of the ambient space |

| Method | Returns |
| --- | --- |
| `phi(m)` | $(M + M^\top)/2$ for a `(p, p)` matrix |
| `c(x)` | $X^\top X - I_p$ |
| `feas_eval(x)` | $\|X^\top X - I_p\|_F$, as a `float` |
| `ja(x, g)` | $G - X\,\Phi(X^\top G)$ |
| `jc(x, lam)` | $X\,\Phi(\Lambda)$ |
| `jc_transpose(x, d)` | $\Phi(X^\top D)$ |
| `a(x)` | the feasibility restoring map |
| `post_process(x)` | the orthogonal polar factor $UV^\top$ |
| `init_point(xinit=None)` | a feasible starting point |

`init_point` draws a standard normal matrix when `xinit` is omitted, and
orthonormalizes whatever it ends up with unless it is already feasible.

## Solvers

Every solver returns `(X, out)`, where `X` is the solution and `out` is a
dictionary:

| Key | Meaning |
| --- | --- |
| `fvals` | objective value per iteration |
| `kkts` | stationarity measure per iteration |
| `feas` | feasibility measure per iteration |
| `fval`, `kkt`, `fea` | the final values |
| `beta` | the penalty used, `pencf` only |

The objective is a single callable returning the value and the Euclidean
gradient together:

```python
def obj_fun(X):  # X has shape (n, p)
    return fval, grad  # grad has shape (n, p)
```

`verbosity` is `0` for silence, `1` for the convergence and
post-processing lines, and `2` to also print periodically. `maxit` must
be at least `1`. A `xinit` you supply is used exactly as given; only the
default one is drawn and orthonormalized.

### `smopt.slpg_smooth`

```python
slpg_smooth(
    obj_fun,
    manifold,
    xinit=None,
    maxit=100,
    gtol=1e-5,
    post_process=True,
    verbosity=2,
)
```

For a smooth objective. Prints every 20th iteration at `verbosity=2`.

### `smopt.slpg`

```python
slpg(
    obj_fun,
    manifold,
    xinit=None,
    maxit=100,
    prox=None,
    gtol=1e-5,
    post_process=True,
    verbosity=2,
)
```

For `f(X) + r(X)` with `r` reached through `prox(X, eta)`, which must
minimize $\|Y - X\|_F^2/(2\eta) + r(Y)$. `prox` defaults to the identity,
recovering the smooth case. Prints every 50th iteration at `verbosity=2`.

### `smopt.slpg_l21`

```python
slpg_l21(
    obj_fun,
    manifold,
    xinit=None,
    maxit=100,
    gamma=0,
    gtol=1e-5,
    post_process=True,
    verbosity=2,
)
```

For $f(X) + \gamma\|X\|_{2,1}$, which induces row sparsity. Prints every
50th iteration at `verbosity=2`.

### `smopt.pencf`

```python
pencf(
    xinit,
    obj_fun,
    manifold,
    beta=None,
    maxit=100,
    gtol=1e-5,
    post_process=True,
    verbosity=2,
)
```

A constraint dissolving penalty method. Note that the starting point
comes **first**. `beta` defaults to $0.1\|\nabla f(X_0)\|_F$; the value
actually used is reported as `out["beta"]`. Prints every 20th iteration
at `verbosity=2`.

## Proximal operators

### `smopt.prox_l1`

```python
prox_l1(x, eta, gamma=0)
```

Proximal operator of $\gamma\|X\|_1$: entrywise soft thresholding.

### `smopt.prox_l21`

```python
prox_l21(x, eta, gamma=0)
```

Proximal operator of $\gamma\|X\|_{2,1}$: shrinks whole rows towards the
origin.

Both are shaped so they can be handed straight to `slpg`:

```python
X, out = slpg(obj_fun, M, prox=lambda X, eta: prox_l1(X, eta, gamma=0.05))
```
