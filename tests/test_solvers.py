"""The Fortran solver drivers must reproduce the reference iteration.

Each solver is run side by side with the NumPy transcription in
``reference.py`` from the same starting point, and the whole trajectory
is compared, not just the final answer.
"""

import numpy as np
import pytest
import reference
from conftest import orthonormal, quadratic

import smopt


# The two implementations are transcriptions of one another, not
# bit-identical: NumPy sums through BLAS while the Fortran kernels sum in
# their own order. That seed then amplifies, because the
# Barzilai-Borwein step divides differences of nearly equal quantities
# and the l_{2,1} prox thresholds whole rows on or off. Iterates agree to
# ~1e-16 at the start, ~6e-10 by iteration 10 and only ~3e-7 by iteration
# 15. These tests therefore compare a short horizon strictly; that the
# solvers actually converge is checked separately below.
TRACE_MAXIT = 10
TRACE_RTOL = 1e-7
TRACE_ATOL = 1e-10
TRACE_X_ATOL = 1e-7


def eig_problem(rng: np.random.Generator, n: int, p: int):
    """A trace minimization problem with a known optimal value."""
    a = np.sort(rng.uniform(0.5, 10.0, size=n))
    return quadratic(a), float(np.sum(a[:p]))


def assert_same_trace(got: dict, want: dict) -> None:
    """Compare a Fortran trajectory with the reference one."""
    kw = {"rtol": TRACE_RTOL, "atol": TRACE_ATOL}
    assert len(got["fvals"]) == len(want["fvals"])
    assert np.allclose(got["fvals"], want["fvals"], **kw)
    assert np.allclose(got["kkts"], want["kkts"], **kw)
    assert np.allclose(got["feas"], want["feas"], **kw)
    scalar = {"rel": TRACE_RTOL, "abs": TRACE_ATOL}
    assert got["fval"] == pytest.approx(want["fval"], **scalar)
    assert got["kkt"] == pytest.approx(want["kkt"], **scalar)
    assert got["fea"] == pytest.approx(want["fea"], **scalar)


@pytest.mark.parametrize(("n", "p"), [(20, 3), (60, 5), (8, 8)])
@pytest.mark.parametrize("post_process", [True, False])
def test_slpg_smooth_matches_reference(
    rng: np.random.Generator, n: int, p: int, post_process: bool
) -> None:
    m, ref_m = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    got_x, got = smopt.SLPG_smooth(
        obj,
        m,
        Xinit=x0.copy(),
        maxit=TRACE_MAXIT,
        post_process=post_process,
        verbosity=0,
    )
    want_x, want = reference.SLPG_smooth(
        obj,
        ref_m,
        Xinit=x0.copy(),
        maxit=TRACE_MAXIT,
        post_process=post_process,
    )

    assert_same_trace(got, want)
    assert np.allclose(got_x, want_x, atol=TRACE_X_ATOL)


@pytest.mark.parametrize(("n", "p"), [(20, 3), (60, 5)])
def test_slpg_matches_reference(
    rng: np.random.Generator, n: int, p: int
) -> None:
    m, ref_m = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)
    gamma = 0.05

    def prox(x: np.ndarray, eta: float) -> np.ndarray:
        return smopt.prox_l1(x, eta, gamma=gamma)

    def ref_prox(x: np.ndarray, eta: float) -> np.ndarray:
        return reference.prox_l1(x, eta, gamma=gamma)

    got_x, got = smopt.SLPG(
        obj, m, Xinit=x0.copy(), maxit=TRACE_MAXIT, prox=prox, verbosity=0
    )
    want_x, want = reference.SLPG(
        obj, ref_m, Xinit=x0.copy(), maxit=TRACE_MAXIT, prox=ref_prox
    )

    assert_same_trace(got, want)
    assert np.allclose(got_x, want_x, atol=TRACE_X_ATOL)


@pytest.mark.parametrize(("n", "p"), [(20, 3), (60, 5)])
def test_slpg_without_a_prox_matches_reference(
    rng: np.random.Generator, n: int, p: int
) -> None:
    """The default prox is the identity, recovering the smooth case."""
    m, ref_m = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    got_x, got = smopt.SLPG(
        obj, m, Xinit=x0.copy(), maxit=TRACE_MAXIT, verbosity=0
    )
    want_x, want = reference.SLPG(
        obj, ref_m, Xinit=x0.copy(), maxit=TRACE_MAXIT
    )

    assert_same_trace(got, want)
    assert np.allclose(got_x, want_x, atol=TRACE_X_ATOL)


@pytest.mark.parametrize(("n", "p"), [(20, 3), (60, 5)])
@pytest.mark.parametrize("gamma", [0.0, 0.02])
def test_slpg_l21_matches_reference(
    rng: np.random.Generator, n: int, p: int, gamma: float
) -> None:
    m, ref_m = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    got_x, got = smopt.SLPG_l21(
        obj, m, Xinit=x0.copy(), maxit=TRACE_MAXIT, gamma=gamma, verbosity=0
    )
    want_x, want = reference.SLPG_l21(
        obj, ref_m, Xinit=x0.copy(), maxit=TRACE_MAXIT, gamma=gamma
    )

    assert_same_trace(got, want)
    assert np.allclose(got_x, want_x, atol=TRACE_X_ATOL)


@pytest.mark.parametrize(("n", "p"), [(20, 3), (60, 5)])
@pytest.mark.parametrize("beta", [None, 0.5])
def test_pencf_matches_reference(
    rng: np.random.Generator, n: int, p: int, beta: float | None
) -> None:
    m, ref_m = smopt.Stiefel(n, p), reference.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    got_x, got = smopt.PenCF(
        x0.copy(), obj, m, beta=beta, maxit=TRACE_MAXIT, verbosity=0
    )
    want_x, want = reference.PenCF(
        x0.copy(), obj, ref_m, beta=beta, maxit=TRACE_MAXIT
    )

    assert_same_trace(got, want)
    assert np.allclose(got_x, want_x, atol=TRACE_X_ATOL)


def test_pencf_reports_the_beta_it_used(rng: np.random.Generator) -> None:
    n, p = 20, 3
    m = smopt.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    _, out = smopt.PenCF(x0.copy(), obj, m, maxit=5, verbosity=0)
    _, grad = obj(x0)
    assert out["beta"] == pytest.approx(0.1 * np.linalg.norm(grad, "fro"))

    _, out = smopt.PenCF(x0.copy(), obj, m, beta=0.0, maxit=5, verbosity=0)
    assert out["beta"] == pytest.approx(0.0)


SOLVERS = ["SLPG_smooth", "SLPG", "SLPG_l21"]


@pytest.mark.parametrize("name", SOLVERS)
def test_solvers_reach_the_known_minimum(
    rng: np.random.Generator, name: str
) -> None:
    """Trace minimization converges to the sum of the smallest values."""
    n, p = 50, 4
    m = smopt.Stiefel(n, p)
    obj, best = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    solver = getattr(smopt, name)
    x, out = solver(obj, m, Xinit=x0, maxit=3000, gtol=1e-9, verbosity=0)

    assert m.Feas_eval(x) < 1e-10
    assert out["fval"] == pytest.approx(best, rel=1e-5)


#: PenCF lands near the optimum rather than on it. Holding the problem
#: below fixed and varying only the starting point over 60 draws, the
#: relative error has median 7.6e-07 and worst case 1.2e-04; a 150-seed
#: sweep over problems too tops out at 1.0e-04, and the NumPy reference
#: in tests/reference.py is worse still at 1.7e-04. That is the penalty
#: method, not this port. The SLPG solvers above reach machine precision
#: on the same instance, hence their much tighter tolerance.
PENALTY_RTOL = 1e-3


def test_pencf_reaches_the_known_minimum(rng: np.random.Generator) -> None:
    """PenCF converges to the optimum to within a penalty method's reach."""
    n, p = 50, 4
    m = smopt.Stiefel(n, p)
    obj, best = eig_problem(rng, n, p)

    x, out = smopt.PenCF(
        orthonormal(rng, n, p), obj, m, maxit=3000, gtol=1e-9, verbosity=0
    )

    # Feasibility is the part the method does guarantee: it holds to
    # ~3e-15 across every instance measured, so it is asserted tightly.
    assert m.Feas_eval(x) < 1e-10
    assert out["fval"] == pytest.approx(best, rel=PENALTY_RTOL)
    # A feasible point can never beat the true minimum; catching that
    # would mean the objective and the constraint had come apart.
    assert out["fval"] > best - 1e-8


def test_l21_regularization_induces_row_sparsity(
    rng: np.random.Generator,
) -> None:
    n, p = 40, 3
    m = smopt.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    dense, _ = smopt.SLPG_l21(
        obj, m, Xinit=x0.copy(), maxit=500, gamma=0.0, verbosity=0
    )
    sparse, _ = smopt.SLPG_l21(
        obj, m, Xinit=x0.copy(), maxit=500, gamma=1.0, verbosity=0
    )

    def live_rows(x: np.ndarray) -> int:
        return int(np.sum(np.linalg.norm(x, axis=1) > 1e-6))

    assert live_rows(sparse) < live_rows(dense)


@pytest.mark.parametrize("name", SOLVERS)
def test_histories_have_one_entry_per_iteration(
    rng: np.random.Generator, name: str
) -> None:
    n, p = 15, 2
    m = smopt.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)

    solver = getattr(smopt, name)
    _, out = solver(
        obj, m, Xinit=orthonormal(rng, n, p), maxit=7, gtol=0.0, verbosity=0
    )

    assert len(out["fvals"]) == 7
    assert len(out["kkts"]) == 7
    assert len(out["feas"]) == 7


@pytest.mark.parametrize("name", SOLVERS)
def test_a_random_start_is_drawn_when_none_is_given(
    rng: np.random.Generator, name: str
) -> None:
    n, p = 15, 2
    m = smopt.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)

    x, _ = getattr(smopt, name)(obj, m, maxit=20, verbosity=0)
    assert x.shape == (n, p)
    assert m.Feas_eval(x) < 1e-10


@pytest.mark.parametrize("name", SOLVERS)
def test_rejects_an_empty_iteration_budget(
    rng: np.random.Generator, name: str
) -> None:
    m = smopt.Stiefel(6, 2)
    obj, _ = eig_problem(rng, 6, 2)
    with pytest.raises(ValueError, match="maxit must be at least 1"):
        getattr(smopt, name)(obj, m, maxit=0, verbosity=0)


def test_verbosity_controls_printing(
    rng: np.random.Generator, capsys: pytest.CaptureFixture
) -> None:
    n, p = 15, 2
    m = smopt.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)
    x0 = orthonormal(rng, n, p)

    smopt.SLPG_smooth(obj, m, Xinit=x0.copy(), maxit=25, verbosity=0)
    assert capsys.readouterr().out == ""

    smopt.SLPG_smooth(obj, m, Xinit=x0.copy(), maxit=25, verbosity=2)
    out = capsys.readouterr().out
    assert "Iter:0" in out
    assert "Iter:20" in out
    assert "Post-processing" in out


def test_the_objective_sees_the_iterate_as_a_matrix(
    rng: np.random.Generator,
) -> None:
    """The callback must receive an ``(n, p)`` array, not a flat one."""
    n, p = 12, 3
    m = smopt.Stiefel(n, p)
    seen = []

    def obj(x: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append(x.shape)
        return float(np.sum(x * x)), 2.0 * x

    smopt.SLPG_smooth(
        obj, m, Xinit=orthonormal(rng, n, p), maxit=3, verbosity=0
    )
    assert seen
    assert set(seen) == {(n, p)}


def test_l21_stays_feasible_when_it_collapses_the_rank(
    rng: np.random.Generator,
) -> None:
    """A heavy penalty can zero enough rows to make the iterate singular.

    Post-processing must still return a point on the manifold instead of
    dividing by a vanishing singular value.
    """
    n, p = 20, 3
    m = smopt.Stiefel(n, p)
    obj, _ = eig_problem(rng, n, p)

    x, out = smopt.SLPG_l21(
        obj, m, Xinit=orthonormal(rng, n, p), maxit=40, gamma=0.1, verbosity=0
    )

    assert np.all(np.isfinite(x))
    assert m.Feas_eval(x) < 1e-10
    assert np.isfinite(out["fval"])
