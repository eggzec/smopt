r"""SMOPT, a toolbox for optimization over the Stiefel manifold.

The problems addressed here read

.. math::

    \min_{X \in \mathbb{R}^{n \times p}} f(X) + r(X)
    \quad \text{subject to} \quad X^\top X = I_p,

whose feasible set is the Stiefel manifold :math:`\mathcal{S}_{n,p}`.
The smooth part ``f`` is supplied as a Python callable returning the
function value and its Euclidean gradient together; the optional
nonsmooth part ``r`` is reached through its proximal operator.

All of the numerics, including the solver iterations themselves, are
implemented in Fortran 77 and reached through f2py. Python is
responsible only for marshalling arguments, calling back into the user
objective, and reporting progress.

Examples:
    >>> import numpy as np
    >>> from smopt import Stiefel, slpg_smooth
    >>> manifold = Stiefel(6, 2)
    >>> a = np.diag([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    >>> def obj(x):
    ...     return float(np.sum(x * (a @ x))), 2.0 * (a @ x)
    >>> x, out = slpg_smooth(
    ...     obj, manifold, xinit=np.arange(12.0).reshape(6, 2), verbosity=0
    ... )
    >>> bool(manifold.feas_eval(x) < 1e-8)
    True
"""

from importlib.metadata import PackageNotFoundError, version

from .manifold import Stiefel
from .solver import pencf, slpg, slpg_l21, slpg_smooth
from .utility import prox_l1, prox_l21


try:
    __version__ = version("smopt")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0"

__all__: list[str] = [
    "Stiefel",
    "__version__",
    "pencf",
    "prox_l1",
    "prox_l21",
    "slpg",
    "slpg_l21",
    "slpg_smooth",
]
