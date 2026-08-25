"""SLPG, the penalty-free first-order solver family.

Each entry point marshals its arguments, hands control to the Fortran 77
driver, and turns the histories the driver fills in into the log
dictionary returned alongside the solution.
"""

from typing import Any

from .. import _smopt
from .._bridge import (
    Matrix,
    ObjFun,
    Prox,
    as_matrix,
    check_maxit,
    log_callback,
    obj_callback,
    output_dict,
    prox_callback,
)
from ..manifold import Stiefel


#: Iteration stride between the periodic progress lines.
_SMOOTH_PERIOD = 20
_PROX_PERIOD = 50


def SLPG_smooth(  # noqa: N802
    obj_fun: ObjFun,
    manifold: Stiefel,
    Xinit: Matrix | None = None,  # noqa: N803
    maxit: int = 100,
    gtol: float = 1e-5,
    post_process: bool = True,  # noqa: FBT001, FBT002
    verbosity: int = 2,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[Matrix, dict[str, object]]:
    r"""Minimize a smooth objective over the Stiefel manifold.

    Args:
        obj_fun: Callable mapping ``X`` to ``(fval, grad)``, where
            ``grad`` is the Euclidean gradient. Returning both at once
            is usually much cheaper than computing them separately.
        manifold: The :class:`~smopt.manifold.Stiefel` instance fixing
            the dimensions.
        Xinit: Starting point. A random feasible point is drawn when it
            is omitted.
        maxit: Maximum number of iterations.
        gtol: Stationarity tolerance that stops the iteration.
        post_process: Whether to round the final iterate onto the
            manifold.
        verbosity: ``0`` silences output, ``1`` prints the final lines,
            ``2`` also prints every twentieth iteration.
        **kwargs: Ignored, accepted so solvers stay interchangeable.

    Returns:
        The solution and a dictionary holding the ``fvals``, ``kkts``
        and ``feas`` histories together with the final ``fval``, ``kkt``
        and ``fea`` values.

    Examples:
        >>> import numpy as np
        >>> from smopt import SLPG_smooth, Stiefel
        >>> M = Stiefel(6, 2)
        >>> A = np.diag([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        >>> def obj(X):
        ...     return float(np.sum(X * (A @ X))), 2.0 * (A @ X)
        >>> X0 = np.arange(12.0).reshape(6, 2)
        >>> X, out = SLPG_smooth(obj, M, Xinit=X0, verbosity=0)
        >>> bool(M.Feas_eval(X) < 1e-8)
        True
    """
    maxit = check_maxit(maxit)
    n, p = manifold._n, manifold._p
    # A caller supplied starting point is used as given; only the
    # default one is drawn and orthonormalized.
    x0 = (
        manifold.Init_point()
        if Xinit is None
        else as_matrix(Xinit, n, p, "Xinit")
    )

    x, nit, fvals, kkts, feasv, fval, kkt, fea = _smopt.smslps(
        x0,
        maxit,
        gtol,
        int(post_process),
        obj_callback(obj_fun, n, p),
        log_callback(verbosity, _SMOOTH_PERIOD),
    )
    return x, output_dict(nit, fvals, kkts, feasv, fval, kkt, fea)


def SLPG(  # noqa: N802
    obj_fun: ObjFun,
    manifold: Stiefel,
    Xinit: Matrix | None = None,  # noqa: N803
    maxit: int = 100,
    prox: Prox | None = None,
    gtol: float = 1e-5,
    post_process: bool = True,  # noqa: FBT001, FBT002
    verbosity: int = 2,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[Matrix, dict[str, object]]:
    r"""Minimize ``f(X) + r(X)`` over the Stiefel manifold.

    The regularizer ``r`` is reached only through its proximal
    operator. The multiplier of the orthogonality constraint is tracked
    by an inner Arrow-Hurwicz iteration, so no penalty parameter has to
    be tuned.

    Args:
        obj_fun: Callable mapping ``X`` to ``(fval, grad)`` for the
            smooth part ``f``.
        manifold: The :class:`~smopt.manifold.Stiefel` instance fixing
            the dimensions.
        Xinit: Starting point. A random feasible point is drawn when it
            is omitted.
        maxit: Maximum number of iterations.
        prox: Callable mapping ``(X, eta)`` to the minimizer of
            ``||Y - X||_F^2 / (2 eta) + r(Y)``. Defaults to the identity,
            which recovers the smooth case.
        gtol: Stationarity tolerance that stops the iteration.
        post_process: Whether to round the final iterate onto the
            manifold.
        verbosity: ``0`` silences output, ``1`` prints the final lines,
            ``2`` also prints every fiftieth iteration.
        **kwargs: Ignored, accepted so solvers stay interchangeable.

    Returns:
        The solution and a dictionary holding the ``fvals``, ``kkts``
        and ``feas`` histories together with the final ``fval``, ``kkt``
        and ``fea`` values.
    """
    maxit = check_maxit(maxit)
    n, p = manifold._n, manifold._p
    # A caller supplied starting point is used as given; only the
    # default one is drawn and orthonormalized.
    x0 = (
        manifold.Init_point()
        if Xinit is None
        else as_matrix(Xinit, n, p, "Xinit")
    )

    if prox is None:

        def prox(x: Matrix, eta: float) -> Matrix:
            return x

    x, nit, fvals, kkts, feasv, fval, kkt, fea = _smopt.smslpg(
        x0,
        maxit,
        gtol,
        int(post_process),
        obj_callback(obj_fun, n, p),
        prox_callback(prox, n, p),
        log_callback(verbosity, _PROX_PERIOD),
    )
    return x, output_dict(nit, fvals, kkts, feasv, fval, kkt, fea)


def SLPG_l21(  # noqa: N802
    obj_fun: ObjFun,
    manifold: Stiefel,
    Xinit: Matrix | None = None,  # noqa: N803
    maxit: int = 100,
    gamma: float = 0,
    gtol: float = 1e-5,
    post_process: bool = True,  # noqa: FBT001, FBT002
    verbosity: int = 2,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[Matrix, dict[str, object]]:
    r"""Minimize ``f(X) + gamma ||X||_{2,1}`` over the Stiefel manifold.

    The row-sparsity inducing :math:`\ell_{2,1}` norm has a proximal
    operator and a constraint multiplier available in closed form, so
    this driver needs no inner iteration.

    Args:
        obj_fun: Callable mapping ``X`` to ``(fval, grad)`` for the
            smooth part ``f``.
        manifold: The :class:`~smopt.manifold.Stiefel` instance fixing
            the dimensions.
        Xinit: Starting point. A random feasible point is drawn when it
            is omitted.
        maxit: Maximum number of iterations.
        gamma: Weight of the regularization term.
        gtol: Stationarity tolerance that stops the iteration.
        post_process: Whether to round the final iterate onto the
            manifold.
        verbosity: ``0`` silences output, ``1`` prints the final lines,
            ``2`` also prints every fiftieth iteration.
        **kwargs: Ignored, accepted so solvers stay interchangeable.

    Returns:
        The solution and a dictionary holding the ``fvals``, ``kkts``
        and ``feas`` histories together with the final ``fval``, ``kkt``
        and ``fea`` values.
    """
    maxit = check_maxit(maxit)
    n, p = manifold._n, manifold._p
    # A caller supplied starting point is used as given; only the
    # default one is drawn and orthonormalized.
    x0 = (
        manifold.Init_point()
        if Xinit is None
        else as_matrix(Xinit, n, p, "Xinit")
    )

    x, nit, fvals, kkts, feasv, fval, kkt, fea = _smopt.smsl21(
        x0,
        maxit,
        gamma,
        gtol,
        int(post_process),
        obj_callback(obj_fun, n, p),
        log_callback(verbosity, _PROX_PERIOD),
    )
    return x, output_dict(nit, fvals, kkts, feasv, fval, kkt, fea)


__all__: list[str] = ["SLPG", "SLPG_l21", "SLPG_smooth"]
