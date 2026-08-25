"""The Fortran proximal operators must agree with the NumPy reference."""

import numpy as np
import pytest
import reference

import smopt


SHAPES = [(1, 1), (5, 1), (12, 3), (40, 7)]
GAMMAS = [0.0, 0.05, 0.5, 4.0]


@pytest.mark.parametrize(("n", "p"), SHAPES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_prox_l1_matches_reference(
    rng: np.random.Generator, n: int, p: int, gamma: float
) -> None:
    x = rng.standard_normal((n, p))
    got = smopt.prox_l1(x, 0.7, gamma=gamma)
    assert np.allclose(got, reference.prox_l1(x, 0.7, gamma=gamma))


@pytest.mark.parametrize(("n", "p"), SHAPES)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_prox_l21_matches_reference(
    rng: np.random.Generator, n: int, p: int, gamma: float
) -> None:
    x = rng.standard_normal((n, p))
    got = smopt.prox_l21(x, 0.7, gamma=gamma)
    assert np.allclose(got, reference.prox_l21(x, 0.7, gamma=gamma))


def test_prox_with_zero_weight_is_the_identity(
    rng: np.random.Generator,
) -> None:
    x = rng.standard_normal((10, 3))
    assert np.allclose(smopt.prox_l1(x, 1.0), x)
    assert np.allclose(smopt.prox_l21(x, 1.0), x)


def test_prox_l1_soft_thresholds() -> None:
    x = np.array([[-3.0, -0.5, 0.0, 0.5, 3.0]])
    got = smopt.prox_l1(x, 1.0, gamma=1.0)
    assert np.allclose(got, [[-2.0, 0.0, 0.0, 0.0, 2.0]])


def test_prox_l21_shrinks_whole_rows() -> None:
    x = np.array([[3.0, 4.0], [0.3, 0.4]])
    got = smopt.prox_l21(x, 1.0, gamma=1.0)
    # The first row has norm 5 and keeps its direction scaled by 4/5;
    # the second has norm 0.5 and is annihilated.
    assert np.allclose(got, [[2.4, 3.2], [0.0, 0.0]])


def test_prox_l21_survives_a_zero_row() -> None:
    x = np.zeros((3, 2))
    assert np.allclose(smopt.prox_l21(x, 1.0, gamma=1.0), 0.0)


@pytest.mark.parametrize("gamma", [0.1, 1.0])
def test_prox_l1_solves_its_own_subproblem(
    rng: np.random.Generator, gamma: float
) -> None:
    """The prox must beat nearby points on its defining objective."""
    x = rng.standard_normal((8, 3))
    eta = 0.3
    y = smopt.prox_l1(x, eta, gamma=gamma)

    def value(z: np.ndarray) -> float:
        return float(
            np.sum((z - x) ** 2) / (2 * eta) + gamma * np.sum(np.abs(z))
        )

    best = value(y)
    for _ in range(20):
        assert best <= value(y + 1e-3 * rng.standard_normal(y.shape)) + 1e-12
