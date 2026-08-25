"""The Stiefel manifold and the maps the solvers need on it."""

import numpy as np

from .. import _smopt
from .._bridge import Matrix, as_matrix


class Stiefel:
    r"""The Stiefel manifold :math:`\{X \in R^{n \times p} : X^T X = I_p\}`.

    The object carries the dimensions and exposes the maps the solvers
    need. Every one of them is evaluated by the Fortran 77 core.

    Args:
        n: Number of rows of the iterate.
        p: Number of columns of the iterate.

    Raises:
        ValueError: If the dimensions are not positive or ``p`` exceeds
            ``n``.

    Examples:
        >>> import numpy as np
        >>> from smopt import Stiefel
        >>> M = Stiefel(4, 2)
        >>> X = M.Init_point(np.eye(4, 2))
        >>> bool(M.Feas_eval(X) < 1e-12)
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

    def _mat(self, x: Matrix, name: str = "X") -> Matrix:
        return as_matrix(x, self._n, self._p, name)

    def _sq(self, m: Matrix, name: str) -> Matrix:
        return as_matrix(m, self._p, self._p, name)

    def Phi(self, M: Matrix) -> Matrix:  # noqa: N802, N803
        """Symmetrize a square matrix.

        Args:
            M: A ``(p, p)`` matrix.

        Returns:
            ``(M + M.T) / 2``.
        """
        return _smopt.smsymm(self._sq(M, "M"))

    def A(self, X: Matrix) -> Matrix:  # noqa: N802, N803
        """Pull a point back towards the manifold.

        Close to the manifold a second order expansion is used, and the
        exact map ``X (X^T X + I)^{-1} 2`` otherwise.

        Args:
            X: An ``(n, p)`` matrix.

        Returns:
            The restored point.
        """
        return _smopt.smamap(self._mat(X))

    def JA(self, X: Matrix, G: Matrix) -> Matrix:  # noqa: N802, N803
        """Project a Euclidean gradient onto the search direction.

        Args:
            X: An ``(n, p)`` matrix.
            G: The Euclidean gradient at ``X``.

        Returns:
            ``G - X Phi(X^T G)``.
        """
        return _smopt.smja(self._mat(X), self._mat(G, "G"))

    def JC(self, X: Matrix, Lambda: Matrix) -> Matrix:  # noqa: N802, N803
        """Apply the constraint Jacobian to a multiplier.

        Args:
            X: An ``(n, p)`` matrix.
            Lambda: A ``(p, p)`` multiplier.

        Returns:
            ``X Phi(Lambda)``.
        """
        return _smopt.smjc(self._mat(X), self._sq(Lambda, "Lambda"))

    def JC_transpose(self, X: Matrix, D: Matrix) -> Matrix:  # noqa: N802, N803
        """Apply the adjoint of the constraint Jacobian to a direction.

        Args:
            X: An ``(n, p)`` matrix.
            D: An ``(n, p)`` direction.

        Returns:
            ``Phi(X^T D)``.
        """
        return _smopt.smjct(self._mat(X), self._mat(D, "D"))

    def C(self, X: Matrix) -> Matrix:  # noqa: N802, N803
        """Evaluate the constraint violation.

        Args:
            X: An ``(n, p)`` matrix.

        Returns:
            ``X^T X - I``.
        """
        return _smopt.smcmap(self._mat(X))

    def Feas_eval(self, X: Matrix) -> float:  # noqa: N802, N803
        """Measure how far a point sits from the manifold.

        Args:
            X: An ``(n, p)`` matrix.

        Returns:
            The Frobenius norm of ``X^T X - I``.
        """
        return float(_smopt.smfeas(self._mat(X)))

    def Init_point(self, Xinit: Matrix | None = None) -> Matrix:  # noqa: N802, N803
        """Produce a feasible starting point.

        Args:
            Xinit: Optional starting matrix. A standard normal matrix is
                drawn when it is omitted. Either way the result is
                orthonormalized unless it is already feasible.

        Returns:
            An ``(n, p)`` matrix on the manifold.
        """
        start = (
            np.random.randn(self._n, self._p)
            if Xinit is None
            else self._mat(Xinit, "Xinit")
        )
        return _smopt.sminit(start)

    def Post_process(self, X: Matrix) -> Matrix:  # noqa: N802, N803
        """Round a point onto the manifold.

        Args:
            X: An ``(n, p)`` matrix.

        Returns:
            The orthogonal polar factor ``U V^T`` of ``X``, where
            ``X = U S V^T`` is a thin singular value decomposition.
        """
        return _smopt.smpost(self._mat(X))


__all__: list[str] = ["Stiefel"]
