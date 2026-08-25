"""PenCF, the constraint dissolving penalty solver."""

from typing import Any

from .. import _smopt
from .._bridge import (
    Matrix,
    ObjFun,
    as_matrix,
    check_maxit,
    log_callback,
    obj_callback,
    output_dict,
)
from ..manifold import Stiefel


#: Iteration stride between the periodic progress lines.
_PERIOD = 20

#: Sentinel handed to the Fortran driver to request the default penalty.
_AUTO_BETA = -1.0


def pencf(
    xinit: Matrix,
    obj_fun: ObjFun,
    manifold: Stiefel,
    beta: float | None = None,
    maxit: int = 100,
    gtol: float = 1e-5,
    post_process: bool = True,  # noqa: FBT001, FBT002
    verbosity: int = 2,
    **kwargs: Any,  # noqa: ANN401
) -> tuple[Matrix, dict[str, object]]:
    r"""Minimize a smooth objective with a constraint dissolving penalty.

    The search direction adds ``beta jc(x, c(x))`` to the projected
    gradient, and feasibility is restored only once the iterate has
    drifted appreciably off the manifold.

    Args:
        xinit: Starting point. A random feasible point is drawn when it
            is ``None``.
        obj_fun: Callable mapping ``x`` to ``(fval, grad)``, where
            ``grad`` is the Euclidean gradient.
        manifold: The :class:`~smopt.manifold.Stiefel` instance fixing
            the dimensions.
        beta: Penalty weight. Defaults to ``0.1`` times the Frobenius
            norm of the gradient at the starting point.
        maxit: Maximum number of iterations.
        gtol: Stationarity tolerance that stops the iteration.
        post_process: Whether to round the final iterate onto the
            manifold.
        verbosity: ``0`` silences output, ``1`` prints the final lines,
            ``2`` also prints every twentieth iteration.
        **kwargs: Ignored, accepted so solvers stay interchangeable.

    Returns:
        The solution and a dictionary holding the ``fvals``, ``kkts``
        and ``feas`` histories, the final ``fval``, ``kkt`` and ``fea``
        values, and the ``beta`` actually used.

    Examples:
        >>> import numpy as np
        >>> from smopt import Stiefel, pencf
        >>> manifold = Stiefel(6, 2)
        >>> a = np.diag([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        >>> def obj(x):
        ...     return float(np.sum(x * (a @ x))), 2.0 * (a @ x)
        >>> x0 = np.arange(12.0).reshape(6, 2)
        >>> x, out = pencf(x0, obj, manifold, verbosity=0)
        >>> bool(manifold.feas_eval(x) < 1e-8)
        True
    """
    maxit = check_maxit(maxit)
    n, p = manifold._n, manifold._p
    # A caller supplied starting point is used exactly as given; only
    # the default one is drawn and orthonormalized.
    start = (
        manifold.init_point()
        if xinit is None
        else as_matrix(xinit, n, p, "xinit")
    )

    x, nit, fvals, kkts, feasv, fval, kkt, fea, betout = _smopt.smpcf(
        start,
        _AUTO_BETA if beta is None else float(beta),
        maxit,
        gtol,
        int(post_process),
        obj_callback(obj_fun, n, p),
        log_callback(verbosity, _PERIOD),
    )
    out = output_dict(nit, fvals, kkts, feasv, fval, kkt, fea)
    out["beta"] = betout
    return x, out


__all__: list[str] = ["pencf"]
