"""A NumPy transcription of the algorithm SMOPT ports to Fortran 77.

This module exists purely as a test oracle. It mirrors the original
implementation statement by statement so that the Fortran drivers can be
checked against it, and it is deliberately kept free of the validation,
packaging and reporting concerns that the shipped package handles.

Names follow the conventions of this project rather than those of the
original, so the oracle and the package under test can be driven through
the same attribute lookups. The one behavioural departure is
:meth:`Stiefel.init_point`, whose ``xinit == None`` test raises on array
input; it is spelled ``is None`` here so the reference can be driven from
the tests.
"""

import numpy as np
from numpy.linalg import norm, svd


class Stiefel:
    """Reference geometry of the Stiefel manifold."""

    def __init__(self, n, p):
        self._n = n
        self._p = p
        self.dim = n * p

    def phi(self, m):
        return (m + m.T) / 2

    def a(self, x):
        xx = x.T @ x
        feas_tmp = norm(xx - np.eye(self._p), "fro")
        if feas_tmp < 0.5:
            return 1.5 * x - x @ (xx / 2)
        return np.linalg.solve((xx + np.eye(self._p)) / 2, x.T).T

    def ja(self, x, g):
        return g - x @ self.phi(x.T @ g)

    def jc(self, x, lam):
        return x @ self.phi(lam)

    def jc_transpose(self, x, d):
        return self.phi(x.T @ d)

    def c(self, x):
        return x.T @ x - np.eye(self._p)

    def feas_eval(self, x):
        return norm(self.c(x), "fro")

    def init_point(self, xinit=None):
        if xinit is None:
            xinit = np.random.randn(self._n, self._p)
        if norm(xinit.T @ xinit - np.eye(self._p), "fro") > 1e-6:
            xinit, _ = np.linalg.qr(xinit)
        return xinit

    def post_process(self, x):
        ux, _, vx = svd(x, full_matrices=False)
        return ux @ vx


def slpg_smooth(
    obj_fun,
    manifold,
    xinit=None,
    maxit=100,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference SLPG for a smooth objective."""
    kkts, feas, fvals = [], [], []

    if xinit is None:
        xinit = manifold.init_point()

    x = xinit
    fval, gradf = obj_fun(x)
    gradr = manifold.ja(x, gradf)
    lip = norm(gradf, "fro") + norm(gradr, "fro")

    s = y = None
    for jj in range(maxit):
        if jj < 3:
            stepsize = 0.01 / lip
        else:
            stepsize = np.abs(np.sum(s * y) / np.sum(y * y))
            stepsize = np.min((stepsize, 1e10))

        x_p = x
        x = x - stepsize * gradr
        x = manifold.a(x)
        s = x - x_p

        fval, gradf = obj_fun(x)
        gradr_p = gradr
        gradr = manifold.ja(x, gradf)
        y = gradr - gradr_p

        substationarity = norm(gradr, "fro")
        feasibility = manifold.feas_eval(x)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        x = manifold.post_process(x)
        fval, gradf = obj_fun(x)
        gradr = manifold.ja(x, gradf)
        substationarity = norm(gradr, "fro")
        feasibility = manifold.feas_eval(x)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return x, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def arrow_hurwicz_slpg(x, g, eta, prox, lam, manifold, tol=0):
    """Reference Arrow-Hurwicz multiplier update."""
    lam_temp = lam
    try_stepsize = eta
    z_tmp = x - try_stepsize * g
    for _ in range(5):
        x_try = prox(
            z_tmp - try_stepsize * manifold.jc(x, lam_temp), try_stepsize
        )
        d_x = 1 / try_stepsize * (x_try - x)
        lam_inc = manifold.jc_transpose(x, d_x)
        lam_temp = lam_temp + lam_inc
        if norm(lam_inc, "fro") < tol:
            break
    return lam_temp


def slpg(
    obj_fun,
    manifold,
    xinit=None,
    maxit=100,
    prox=lambda x, eta: x,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference SLPG for a proximable regularizer."""
    kkts, feas, fvals, steps = [], [], [], []

    if xinit is None:
        xinit = manifold.init_point()

    p = manifold._p
    x = xinit
    fval, gradf = obj_fun(x)
    gradr = manifold.ja(x, gradf)
    lip = norm(gradf, "fro") + norm(gradr, "fro")

    lam = np.zeros([p, p])
    lam = arrow_hurwicz_slpg(x, gradr, 0.01 / lip, prox, lam, manifold)
    grad = gradr + manifold.jc(x, lam)

    s = y = None
    for jj in range(maxit):
        if jj < 5:
            stepsize = 0.01 / lip
        else:
            stepsize = np.abs(np.sum(s * s) / np.sum(s * y))
            stepsize = np.min((stepsize, 1e10))

        x_p = x
        steps.append(stepsize)

        x = prox(x - stepsize * (gradr + manifold.jc(x, lam)), stepsize)
        x = manifold.a(x)
        s = x - x_p

        fval, gradf = obj_fun(x)
        grad_p = grad
        gradr = manifold.ja(x, gradf)

        stepsize_try = np.average(steps[np.maximum(0, jj - 10) :])
        stepsize_try = np.minimum(
            np.maximum(stepsize_try, 1e-5 / lip), 1e10 / lip
        )

        tol_aw = 1000 * manifold.feas_eval(x)
        lam = arrow_hurwicz_slpg(
            x, gradr, stepsize_try, prox, lam, manifold, tol=tol_aw
        )
        grad = gradr + manifold.jc(x, lam)
        y = grad - grad_p

        substationarity = norm(s / stepsize, "fro")
        feasibility = manifold.feas_eval(x)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        x = manifold.post_process(x)
        fval, gradf = obj_fun(x)
        feasibility = manifold.feas_eval(x)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return x, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def slpg_l21(
    obj_fun,
    manifold,
    xinit=None,
    maxit=100,
    gamma=0,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference SLPG for the l_{2,1} regularized objective."""

    def prox(x_input, eta):
        x_ref = np.sqrt(np.sum(x_input**2, axis=1, keepdims=True))
        x_ref_reduce = np.maximum(x_ref - gamma * eta, 0)
        return (x_ref_reduce / (x_ref + 1e-16)) * x_input

    def generate_lam(x_input):
        x_ref = 1 / (1e-14 + np.sqrt(np.sum(x_input**2, axis=1, keepdims=True)))
        return -x_input.T @ (x_ref * x_input)

    kkts, feas, fvals = [], [], []

    if xinit is None:
        xinit = manifold.init_point()

    x = xinit
    fval, gradf = obj_fun(x)
    gradr = manifold.ja(x, gradf)
    lip = norm(gradf, "fro") + norm(gradr, "fro")

    lam = gamma * generate_lam(x)
    grad = gradr + manifold.jc(x, lam)

    s = y = None
    for jj in range(maxit):
        if jj < 5:
            stepsize = 0.001 / lip
        else:
            stepsize = np.abs(np.sum(s * s) / np.sum(s * y))
            stepsize = np.min((stepsize, 1e5))

        x_p = x
        x = prox(x - stepsize * grad, stepsize)
        x = manifold.a(x)
        s = x - x_p

        fval, gradf = obj_fun(x)
        grad_p = grad
        gradr = manifold.ja(x, gradf)

        lam = gamma * generate_lam(x)
        grad = gradr + manifold.jc(x, lam)
        y = grad - grad_p

        substationarity = norm(s / stepsize, "fro")
        feasibility = manifold.feas_eval(x)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        x = manifold.post_process(x)
        fval, gradf = obj_fun(x)
        feasibility = manifold.feas_eval(x)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return x, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def pencf(
    xinit,
    obj_fun,
    manifold,
    beta=None,
    maxit=100,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference PenCF."""
    kkts, feas, fvals = [], [], []

    p = manifold._p
    x = xinit
    fval, gradf = obj_fun(x)

    if beta is None:
        beta = 0.1 * norm(gradf, "fro")

    gradr = manifold.ja(x, gradf) + beta * manifold.jc(x, manifold.c(x))
    lip = norm(gradf, "fro") + norm(gradr, "fro")

    s = y = None
    for jj in range(maxit):
        if jj < 3:
            stepsize = 0.01 / lip
        else:
            stepsize = np.abs(np.sum(s * y) / np.sum(y * y))
            stepsize = np.min((stepsize, 1e10))

        x_p = x
        x = x - stepsize * gradr

        xx = x.T @ x
        feas_tmp = manifold.feas_eval(x)
        if feas_tmp > 1e-1:
            if feas_tmp < 0.5:
                x = 1.5 * x - x @ (xx / 2)
            else:
                x = np.linalg.solve((xx + np.eye(p)) / 2, x.T).T

        if norm(x, "fro") > 1.001 * np.sqrt(p):
            x = x * (1.001 * np.sqrt(p) / norm(x, "fro"))

        s = x - x_p

        fval, gradf = obj_fun(x)
        gradr_p = gradr
        gradr = manifold.ja(x, gradf) + beta * manifold.jc(x, manifold.c(x))
        y = gradr - gradr_p

        substationarity = norm(gradr, "fro")
        feasibility = manifold.feas_eval(x)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        x = manifold.post_process(x)
        fval, gradf = obj_fun(x)
        gradr = manifold.ja(x, gradf)
        substationarity = norm(gradr, "fro")
        feasibility = manifold.feas_eval(x)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return x, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def prox_l1(x_input, eta, gamma=0):
    """Reference proximal operator of the l_1 norm."""
    return np.maximum(x_input - gamma * eta, 0) + np.minimum(
        x_input + gamma * eta, 0
    )


def prox_l21(x_input, eta, gamma=0):
    """Reference proximal operator of the l_{2,1} norm."""
    x_ref = np.sqrt(np.sum(x_input**2, axis=1, keepdims=True))
    x_ref_reduce = np.maximum(x_ref - gamma * eta, 0)
    return x_ref_reduce / (x_ref + 1e-14) * x_input
