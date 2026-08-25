"""Plumbing between the Python API and the Fortran 77 core.

The Fortran drivers reach back into Python for three things: the
objective, the proximal operator of a nonsmooth regularizer, and
progress reporting. Matrices cross that boundary flattened in column
major order, which is how Fortran stores them, so the adapters here
reshape in both directions and keep every user facing array in the
familiar ``(n, p)`` form.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


Matrix = NDArray[np.float64]
ObjFun = Callable[[Matrix], tuple[float, Matrix]]
Prox = Callable[[Matrix, float], Matrix]

_LINE = "Iter:{}    fval:{:.3e}   kkts:{:.3e}    feas:{:3e}"

#: ``stage`` values passed to the logging callback by the Fortran side.
_STAGE_ITERATION = 0
_STAGE_CONVERGED = 1

#: Verbosity at which the periodic progress lines are printed.
_VERBOSE_PERIODIC = 2


def as_matrix(x: Matrix, n: int, p: int, name: str = "X") -> Matrix:
    """Return ``x`` as a contiguous ``(n, p)`` array of doubles.

    Args:
        x: Array to validate and convert.
        n: Expected row count.
        p: Expected column count.
        name: Name used in the error message.

    Returns:
        A float64 array of shape ``(n, p)``.

    Raises:
        ValueError: If ``x`` does not have shape ``(n, p)``.
    """
    out = np.asarray(x, dtype=np.float64)
    if out.shape != (n, p):
        msg = f"{name} must have shape {(n, p)}, got {out.shape}"
        raise ValueError(msg)
    return out


def check_maxit(maxit: int) -> int:
    """Validate the iteration budget.

    Args:
        maxit: Requested maximum number of iterations.

    Returns:
        The validated budget.

    Raises:
        ValueError: If ``maxit`` is not at least one.
    """
    maxit = int(maxit)
    if maxit < 1:
        msg = f"maxit must be at least 1, got {maxit}"
        raise ValueError(msg)
    return maxit


def obj_callback(obj_fun: ObjFun, n: int, p: int) -> Callable[..., object]:
    """Adapt a user objective for the Fortran ``OBJFUN`` callback.

    Args:
        obj_fun: Callable mapping ``X`` to ``(fval, grad)``.
        n: Row count of the iterate.
        p: Column count of the iterate.

    Returns:
        A callable taking the flattened iterate and returning the
        function value together with the flattened gradient.
    """

    def objfun(x: Matrix) -> tuple[float, Matrix]:
        fval, grad = obj_fun(x.reshape((n, p), order="F"))
        flat = np.asarray(grad, dtype=np.float64).reshape(-1, order="F")
        return float(fval), flat

    return objfun


def prox_callback(prox: Prox, n: int, p: int) -> Callable[..., object]:
    """Adapt a user proximal operator for the Fortran ``PROXFN`` callback.

    Args:
        prox: Callable mapping ``(X, eta)`` to the proximal point.
        n: Row count of the iterate.
        p: Column count of the iterate.

    Returns:
        A callable taking the flattened point and the step size, and
        returning the flattened proximal point.
    """

    def proxfn(x: Matrix, eta: float) -> Matrix:
        y = prox(x.reshape((n, p), order="F"), float(eta))
        return np.asarray(y, dtype=np.float64).reshape(-1, order="F")

    return proxfn


def log_callback(verbosity: int, period: int) -> Callable[..., object]:
    """Build the progress reporter handed to the Fortran drivers.

    The Fortran side reports every iteration and leaves the decision of
    what to print here, so the printing policy stays in Python.

    Args:
        verbosity: ``0`` silences output, ``1`` prints only the final
            and post-processing lines, ``2`` also prints periodically.
        period: Iteration stride between periodic lines.

    Returns:
        A callable accepting ``(it, fval, kkt, fea, stage)``.
    """

    def logfun(
        it: int, fval: float, kkt: float, fea: float, stage: int
    ) -> None:
        if stage == _STAGE_ITERATION:
            if verbosity == _VERBOSE_PERIODIC and it % period == 0:
                print(_LINE.format(it, fval, kkt, fea))
        elif stage == _STAGE_CONVERGED:
            if verbosity >= 1:
                print(_LINE.format(it, fval, kkt, fea))
        elif verbosity >= 1:
            print("Post-processing")
            print(_LINE.format(it, fval, kkt, fea))

    return logfun


def output_dict(
    nit: int,
    fvals: Matrix,
    kkts: Matrix,
    feasv: Matrix,
    fval: float,
    kkt: float,
    fea: float,
) -> dict[str, object]:
    """Assemble the log dictionary returned alongside the solution.

    Args:
        nit: Number of iterations actually performed.
        fvals: Objective value history, padded to the iteration budget.
        kkts: Stationarity history, padded to the iteration budget.
        feasv: Feasibility history, padded to the iteration budget.
        fval: Final objective value.
        kkt: Final stationarity measure.
        fea: Final feasibility measure.

    Returns:
        A dictionary with the per-iteration histories truncated to the
        iterations that ran, plus the final scalars.
    """
    return {
        "kkts": kkts[:nit].tolist(),
        "fvals": fvals[:nit].tolist(),
        "fea": fea,
        "kkt": kkt,
        "fval": fval,
        "feas": feasv[:nit].tolist(),
    }
