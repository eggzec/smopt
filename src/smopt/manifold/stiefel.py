"""The Stiefel manifold and the maps the solvers need on it."""

import numpy as np

from .. import _smopt
from .._bridge import Matrix, as_matrix


class Stiefel:
    r"""The Stiefel manifold :math:`\{X \in R^{n \times p} : X^T X = I_p\}`.

    The object carries the dimensions and exposes the maps the solvers
    need. Every one of them is evaluated by the Fortran 77 core, and each
    is named after the symbol it carries in the theory documentation.

    Args:
        n: Number of rows of the iterate.
        p: Number of columns of the iterate.

    Raises:
        ValueError: If the dimensions are not positive or ``p`` exceeds
            ``n``.

    Examples:
        >>> import numpy as np
        >>> from smopt import Stiefel
        >>> manifold = Stiefel(4, 2)
        >>> x = manifold.init_point(np.eye(4, 2))
        >>> bool(manifold.feas_eval(x) < 1e-12)
        True
    """

    def __init__(self, n: int, p: int) -> None:
        n, p = int(n), int(p)
        if n < 1 or p < 1:
            msg = f"n and p must be positive, got n={n}, p={p}"
            raise ValueError(msg)
        if p > n:
            msg = f"p must not exceed n, got n={n}, p={p}"
            raise ValueError(msg)
        self._n = n
        self._p = p
        self.dim = n * p

    def _mat(self, x: Matrix, name: str = "x") -> Matrix:
        return as_matrix(x, self._n, self._p, name)

    def _sq(self, m: Matrix, name: str) -> Matrix:
        return as_matrix(m, self._p, self._p, name)

    def phi(self, m: Matrix) -> Matrix:
        """Symmetrize a square matrix.

        Args:
            m: A ``(p, p)`` matrix.

        Returns:
            ``(m + m.T) / 2``.
        """
        return _smopt.smsymm(self._sq(m, "m"))

    def a(self, x: Matrix) -> Matrix:
        """Pull a point back towards the manifold.

        Close to the manifold a second order expansion is used, and the
        exact map ``x ((x^T x + I) / 2)^-1`` otherwise.

        Args:
            x: An ``(n, p)`` matrix.

        Returns:
            The restored point.
        """
        return _smopt.smamap(self._mat(x))

    def ja(self, x: Matrix, g: Matrix) -> Matrix:
        """Project a Euclidean gradient onto the search direction.

        Args:
            x: An ``(n, p)`` matrix.
            g: The Euclidean gradient at ``x``.

        Returns:
            ``g - x phi(x^T g)``.
        """
        return _smopt.smja(self._mat(x), self._mat(g, "g"))

    def jc(self, x: Matrix, lam: Matrix) -> Matrix:
        """Apply the constraint Jacobian to a multiplier.

        Args:
            x: An ``(n, p)`` matrix.
            lam: A ``(p, p)`` multiplier.

        Returns:
            ``x phi(lam)``.
        """
        return _smopt.smjc(self._mat(x), self._sq(lam, "lam"))

    def jc_transpose(self, x: Matrix, d: Matrix) -> Matrix:
        """Apply the adjoint of the constraint Jacobian to a direction.

        Args:
            x: An ``(n, p)`` matrix.
            d: An ``(n, p)`` direction.

        Returns:
            ``phi(x^T d)``.
        """
        return _smopt.smjct(self._mat(x), self._mat(d, "d"))

    def c(self, x: Matrix) -> Matrix:
        """Evaluate the constraint violation.

        Args:
            x: An ``(n, p)`` matrix.

        Returns:
            ``x^T x - I``.
        """
        return _smopt.smcmap(self._mat(x))

    def feas_eval(self, x: Matrix) -> float:
        """Measure how far a point sits from the manifold.

        Args:
            x: An ``(n, p)`` matrix.

        Returns:
            The Frobenius norm of ``x^T x - I``.
        """
        return float(_smopt.smfeas(self._mat(x)))

    def init_point(self, xinit: Matrix | None = None) -> Matrix:
        """Produce a feasible starting point.

        Args:
            xinit: Optional starting matrix. A standard normal matrix is
                drawn when it is omitted. Either way the result is
                orthonormalized unless it is already feasible.

        Returns:
            An ``(n, p)`` matrix on the manifold.
        """
        start = (
            np.random.randn(self._n, self._p)
            if xinit is None
            else self._mat(xinit, "xinit")
        )
        return _smopt.sminit(start)

    def post_process(self, x: Matrix) -> Matrix:
        """Round a point onto the manifold.

        Args:
            x: An ``(n, p)`` matrix.

        Returns:
            The orthogonal polar factor ``u v^T`` of ``x``, where
            ``x = u s v^T`` is a thin singular value decomposition.
        """
        return _smopt.smpost(self._mat(x))


__all__: list[str] = ["Stiefel"]
