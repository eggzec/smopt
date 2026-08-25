"""Proximal operators of the regularizers SMOPT supports.

The proximal operator of a function :math:`r` at :math:`X` with step
:math:`\\eta` is the minimizer of

.. math:: \\frac{1}{2\\eta} \\|Y - X\\|_F^2 + r(Y).

Both operators below are evaluated by the Fortran 77 core, and both are
shaped so they can be handed straight to :func:`~smopt.solver.SLPG`.
"""

from .. import _smopt
from .._bridge import Matrix


#: Guard against dividing by a vanishing row norm.
_EPS = 1e-14


def prox_l1(X_input: Matrix, eta: float, gamma: float = 0) -> Matrix:  # noqa: N803
    r"""Proximal operator of :math:`\gamma \|X\|_1`.

    Args:
        X_input: The point at which to evaluate the operator.
        eta: The proximal step size.
        gamma: Weight of the regularization term.

    Returns:
        The soft-thresholded matrix, entry by entry.

    Examples:
        >>> import numpy as np
        >>> from smopt import prox_l1
        >>> Y = prox_l1(np.array([[-3.0, 0.5, 2.0]]), 1.0, gamma=1.0)
        >>> bool(np.allclose(Y, [[-2.0, 0.0, 1.0]]))
        True
    """
    return _smopt.smpl1(X_input, eta, gamma)


def prox_l21(X_input: Matrix, eta: float, gamma: float = 0) -> Matrix:  # noqa: N803
    r"""Proximal operator of :math:`\gamma \|X\|_{2,1}`.

    The :math:`\ell_{2,1}` norm sums the Euclidean norms of the rows of
    ``X``, so the operator shrinks whole rows towards the origin and
    induces row sparsity.

    Args:
        X_input: The point at which to evaluate the operator.
        eta: The proximal step size.
        gamma: Weight of the regularization term.

    Returns:
        The row-wise shrunk matrix.

    Examples:
        >>> import numpy as np
        >>> from smopt import prox_l21
        >>> X = np.array([[3.0, 4.0], [0.3, 0.4]])
        >>> bool(
        ...     np.allclose(
        ...         prox_l21(X, 1.0, gamma=1.0), [[2.4, 3.2], [0.0, 0.0]]
        ...     )
        ... )
        True
    """
    return _smopt.smpl21(X_input, eta, gamma, _EPS)


__all__: list[str] = ["prox_l1", "prox_l21"]
