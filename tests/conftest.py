"""Shared fixtures and problem builders for the SMOPT test suite."""

import numpy as np
import pytest
import reference

import smopt


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator so every test is reproducible."""
    return np.random.default_rng(20260825)


def orthonormal(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Draw a random point on the Stiefel manifold."""
    q, _ = np.linalg.qr(rng.standard_normal((n, p)))
    return np.ascontiguousarray(q[:, :p])


def quadratic(a_diag: np.ndarray):
    """Build ``tr(X^T A X)`` and its gradient for a diagonal ``A``."""
    a = np.asarray(a_diag, dtype=np.float64)

    def obj(x: np.ndarray) -> tuple[float, np.ndarray]:
        ax = a[:, None] * x
        return float(np.sum(x * ax)), 2.0 * ax

    return obj


def manifolds(n: int, p: int) -> tuple[smopt.Stiefel, reference.Stiefel]:
    """Return the shipped manifold and its reference twin."""
    return smopt.Stiefel(n, p), reference.Stiefel(n, p)
