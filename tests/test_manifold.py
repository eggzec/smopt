"""The Fortran manifold maps must agree with the NumPy reference."""

import numpy as np
import pytest
import reference
from conftest import orthonormal

import smopt


SHAPES = [(1, 1), (5, 1), (12, 3), (40, 7), (9, 9)]


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_phi_symmetrizes(rng: np.random.Generator, n: int, p: int) -> None:
    m = smopt.Stiefel(n, p)
    a = rng.standard_normal((p, p))
    got = m.Phi(a)
    assert np.allclose(got, reference.Stiefel(n, p).Phi(a))
    assert np.allclose(got, got.T)


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_phi_does_not_mutate_input(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m = smopt.Stiefel(n, p)
    a = rng.standard_normal((p, p))
    before = a.copy()
    m.Phi(a)
    assert np.array_equal(a, before)


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_constraint_and_feasibility(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m, ref = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    x = rng.standard_normal((n, p))
    assert np.allclose(m.C(x), ref.C(x))
    assert m.Feas_eval(x) == pytest.approx(ref.Feas_eval(x))


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_feasibility_vanishes_on_the_manifold(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m = smopt.Stiefel(n, p)
    assert m.Feas_eval(orthonormal(rng, n, p)) < 1e-12


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_jacobians(rng: np.random.Generator, n: int, p: int) -> None:
    m, ref = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    x = orthonormal(rng, n, p)
    g = rng.standard_normal((n, p))
    d = rng.standard_normal((n, p))
    lam = rng.standard_normal((p, p))

    assert np.allclose(m.JA(x, g), ref.JA(x, g))
    assert np.allclose(m.JC(x, lam), ref.JC(x, lam))
    assert np.allclose(m.JC_transpose(x, d), ref.JC_transpose(x, d))


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_ja_output_is_tangent(rng: np.random.Generator, n: int, p: int) -> None:
    """``JA`` removes the symmetric part of ``X^T G``."""
    m = smopt.Stiefel(n, p)
    x = orthonormal(rng, n, p)
    g = rng.standard_normal((n, p))
    r = m.JA(x, g)
    xtr = x.T @ r
    assert np.allclose(xtr, -xtr.T, atol=1e-10)


@pytest.mark.parametrize(("n", "p"), SHAPES)
@pytest.mark.parametrize("scale", [1e-3, 0.05, 0.4, 3.0])
def test_a_map_matches_reference(
    rng: np.random.Generator, n: int, p: int, scale: float
) -> None:
    """Both the near-manifold expansion and the exact solve branch."""
    m, ref = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    x = orthonormal(rng, n, p) + scale * rng.standard_normal((n, p))
    assert np.allclose(m.A(x), ref.A(x), atol=1e-10)


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_a_map_improves_feasibility(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m = smopt.Stiefel(n, p)
    x = orthonormal(rng, n, p) + 0.05 * rng.standard_normal((n, p))
    assert m.Feas_eval(m.A(x)) < m.Feas_eval(x)


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_post_process_is_the_polar_factor(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m, ref = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    x = rng.standard_normal((n, p))
    got = m.Post_process(x)
    assert np.allclose(got, ref.Post_process(x), atol=1e-9)
    assert m.Feas_eval(got) < 1e-12


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_init_point_lands_on_the_manifold(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m = smopt.Stiefel(n, p)
    assert m.Feas_eval(m.Init_point(rng.standard_normal((n, p)))) < 1e-12
    assert m.Feas_eval(m.Init_point()) < 1e-12


@pytest.mark.parametrize(("n", "p"), SHAPES)
def test_init_point_keeps_a_feasible_argument(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m = smopt.Stiefel(n, p)
    x = orthonormal(rng, n, p)
    assert np.allclose(m.Init_point(x), x)


def test_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        smopt.Stiefel(0, 1)
    with pytest.raises(ValueError, match="must not exceed"):
        smopt.Stiefel(2, 3)


def test_rejects_mismatched_shapes(rng: np.random.Generator) -> None:
    m = smopt.Stiefel(6, 2)
    with pytest.raises(ValueError, match="must have shape"):
        m.C(rng.standard_normal((6, 3)))
    with pytest.raises(ValueError, match="must have shape"):
        m.JC(orthonormal(rng, 6, 2), rng.standard_normal((3, 3)))


def test_dim_attribute() -> None:
    assert smopt.Stiefel(7, 3).dim == 21


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_post_process_completes_a_rank_deficient_point(
    rng: np.random.Generator, rank: int
) -> None:
    """A vanishing singular value leaves its direction unconstrained.

    There is no polar factor then, so the missing directions get an
    arbitrary orthonormal completion rather than dividing by zero.
    """
    n, p = 9, 3
    m = smopt.Stiefel(n, p)
    x = np.zeros((n, p))
    if rank:
        x[:, :rank] = orthonormal(rng, n, rank)

    got = m.Post_process(x)

    assert np.all(np.isfinite(got))
    assert m.Feas_eval(got) < 1e-12
    # Whatever the completion picks, it must not disturb the directions
    # the input did pin down.
    assert np.allclose(got @ got.T @ x, x, atol=1e-10)


def test_post_process_of_a_zero_matrix_is_still_feasible() -> None:
    m = smopt.Stiefel(5, 2)
    got = m.Post_process(np.zeros((5, 2)))
    assert np.all(np.isfinite(got))
    assert m.Feas_eval(got) < 1e-12
