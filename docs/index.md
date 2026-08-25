# smopt

![smopt](https://raw.githubusercontent.com/eggzec/smopt/master/docs/assets/images/smopt-banner.png)

**Stiefel manifold optimization, with all numerics in Fortran 77**

`smopt` minimizes a smooth function, optionally plus a nonsmooth
regularizer, over the set of matrices with orthonormal columns:

$$
\min_{X \in \mathbb{R}^{n\times p}} f(X) + r(X)
\quad \text{subject to} \quad X^\top X = I_p.
$$

## Overview

The solvers are **penalty-free first-order methods**. Rather than
retracting along geodesics, they work in the ambient space and dissolve
the orthogonality constraint with a cheap feasibility restoring map, so
an iteration costs little more than a gradient evaluation and a couple
of small matrix products.

Everything numerical — the manifold geometry, the proximal operators,
the Barzilai-Borwein step sizes and the solver loops themselves — is
implemented in Fortran 77 and reached through f2py. Python supplies the
objective through a callback and handles reporting; NumPy is the only
runtime dependency.

```python
import numpy as np
from smopt import slpg_smooth, Stiefel

M = Stiefel(1000, 10)
A = np.diag(np.arange(1000, dtype=float))


def obj_fun(X):
    AX = A @ X
    return float(np.sum(X * AX)), 2.0 * AX


X, out = slpg_smooth(obj_fun, M)
```

## Solvers

| Name | Use when |
| --- | --- |
| `slpg_smooth` | the objective is smooth |
| `slpg` | there is a nonsmooth term with a known proximal operator |
| `slpg_l21` | the nonsmooth term is $\gamma\|X\|_{2,1}$ |
| `pencf` | a constraint dissolving penalty method is preferred |

## Documentation

- [Theory](theory.md) - the manifold, the constraint dissolving map, the algorithms
- [Installation](installation.md) - installation guide
- [Quickstart](quickstart.md) - runnable examples
- [API Reference](api.md) - class and function signatures and arguments
- [References](references.md) - literature citations
