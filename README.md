![smopt](https://raw.githubusercontent.com/eggzec/smopt/master/docs/assets/images/smopt-banner.png)

# smopt

**Stiefel manifold optimization, with all numerics in Fortran 77**

[![Tests](https://github.com/eggzec/smopt/actions/workflows/test.yml/badge.svg)](https://github.com/eggzec/smopt/actions/workflows/test.yml)
[![Documentation](https://github.com/eggzec/smopt/actions/workflows/docs.yml/badge.svg)](https://github.com/eggzec/smopt/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/github/eggzec/smopt/graph/badge.svg)](https://codecov.io/github/eggzec/smopt)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=eggzec_smopt&metric=alert_status)](https://sonarcloud.io/project/overview?id=eggzec_smopt)
[![License](https://img.shields.io/badge/license-GPL%203.0-blue.svg)](./LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/smopt.svg?label=PyPI%20downloads)](https://pypi.org/project/smopt/)
[![Python versions](https://img.shields.io/pypi/pyversions/smopt.svg)](https://pypi.org/project/smopt/)

`smopt` minimizes a smooth function, optionally plus a nonsmooth
regularizer, over the matrices with orthonormal columns:

$$
\min_{X \in \mathbb{R}^{n\times p}} f(X) + r(X)
\quad \text{subject to} \quad X^\top X = I_p.
$$

The solvers are penalty-free first-order methods. Rather than retracting
along geodesics, they work in the ambient space and dissolve the
orthogonality constraint with a cheap feasibility restoring map, so an
iteration costs little more than a gradient evaluation and a couple of
small matrix products.

Everything numerical — the manifold geometry, the proximal operators,
the Barzilai-Borwein step sizes and the solver loops themselves — is
written in Fortran 77 and reached through f2py. Python supplies the
objective through a callback and handles reporting. NumPy is the only
runtime dependency; the extension links against nothing but the Fortran
runtime.

## Quick example

```python
import numpy as np
from smopt import slpg_smooth, Stiefel

n, p = 1000, 10
M = Stiefel(n, p)
A = np.diag(np.arange(n, dtype=float))


def obj_fun(X):
    """Return the objective and its Euclidean gradient together."""
    AX = A @ X
    return float(np.sum(X * AX)), 2.0 * AX


X, out = slpg_smooth(obj_fun, M)
print(out["fval"], out["fea"])
```

## Solvers

| Name | Comment | Call |
| --- | --- | --- |
| `slpg_smooth` | penalty-free first-order method for smooth problems | `slpg_smooth(obj_fun, M)` |
| `slpg` | penalty-free first-order method for nonsmooth problems | `slpg(obj_fun, M, prox=...)` |
| `slpg_l21` | penalty-free first-order method for $\ell_{2,1}$ regularized problems | `slpg_l21(obj_fun, M, gamma=...)` |
| `pencf` | constraint dissolving penalty method | `pencf(xinit, obj_fun, M)` |

Every solver returns `(X, out)`, where `out` carries the `fvals`, `kkts`
and `feas` histories together with the final `fval`, `kkt` and `fea`.

## Installation

```bash
pip install smopt
```

Requires Python 3.10+ and NumPy. See the
[full installation guide](https://eggzec.github.io/smopt/installation/) for
uv, poetry, and source builds.

Building from source additionally needs a Fortran compiler; the wheels
carry the Fortran runtime, so installing one does not.

## Layout

```
src/
  smblas.f      dense kernels: matmul, Cholesky, Jacobi eigensolver, Gram-Schmidt
  smman.f       Stiefel geometry: C, JA, JC, the A map, the polar retraction
  smprox.f      proximal operators and the l_{2,1} multiplier
  smslpg.f      the SLPG solver drivers and the Arrow-Hurwicz inner iteration
  smpencf.f     the pencf driver
  _smopt.pyf    f2py signatures binding the above to Python
  smopt/        the thin Python layer: argument marshalling and reporting
tests/
  reference.py  a NumPy transcription of the algorithm, used as a test oracle
```

## Documentation

- [Theory](https://eggzec.github.io/smopt/theory/) — the manifold, the constraint dissolving map, the algorithms
- [Quickstart](https://eggzec.github.io/smopt/quickstart/) — runnable examples
- [API Reference](https://eggzec.github.io/smopt/api/) — class and function signatures and arguments
- [References](https://eggzec.github.io/smopt/references/) — literature citations

## Acknowledgement

The algorithm ported here originates in the STOP toolbox by Nachuan
Xiao, Lei Wang, Bin Gao, Xin Liu and Ya-xiang Yuan
(<https://stmopt.gitee.io/>). `smopt` re-implements its numerics in
Fortran 77 behind the same solver interface.

## License

GNU General Public License v3 (GPLv3) — see [LICENSE](LICENSE).
