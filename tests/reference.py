"""A NumPy transcription of the algorithm SMOPT ports to Fortran 77.

This module exists purely as a test oracle. It mirrors the original
implementation statement by statement so that the Fortran drivers can be
checked against it, and it is deliberately kept free of the validation,
packaging and reporting concerns that the shipped package handles.

The single intentional departure is :meth:`Stiefel.Init_point`, whose
``Xinit == None`` test raises on array input; it is spelled ``is None``
here so the reference can be driven from the tests.
"""

import numpy as np
from numpy.linalg import norm, svd


class Stiefel:
    """Reference geometry of the Stiefel manifold."""

    def __init__(self, n, p):
        self._n = n
        self._p = p
        self.dim = n * p

    def Phi(self, M):
        return (M + M.T) / 2

    def A(self, X):
        XX = X.T @ X
        feas_tmp = norm(XX - np.eye(self._p), "fro")
        if feas_tmp < 0.5:
            return 1.5 * X - X @ (XX / 2)
        return np.linalg.solve((XX + np.eye(self._p)) / 2, X.T).T

    def JA(self, X, G):
        return G - X @ self.Phi(X.T @ G)

    def JC(self, X, Lambda):
        return X @ self.Phi(Lambda)

    def JC_transpose(self, X, D):
        return self.Phi(X.T @ D)

    def C(self, X):
        return X.T @ X - np.eye(self._p)

    def Feas_eval(self, X):
        return norm(self.C(X), "fro")

    def Init_point(self, Xinit=None):
        if Xinit is None:
            Xinit = np.random.randn(self._n, self._p)
        if norm(Xinit.T @ Xinit - np.eye(self._p), "fro") > 1e-6:
            Xinit, _ = np.linalg.qr(Xinit)
        return Xinit

    def Post_process(self, X):
        UX, _, VX = svd(X, full_matrices=False)
        return UX @ VX


def SLPG_smooth(
    obj_fun,
    manifold,
    Xinit=None,
    maxit=100,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference SLPG for a smooth objective."""
    kkts, feas, fvals = [], [], []

    if Xinit is None:
        Xinit = manifold.Init_point()

    X = Xinit
    fval, gradf = obj_fun(X)
    gradr = manifold.JA(X, gradf)
    L = norm(gradf, "fro") + norm(gradr, "fro")

    S = Y = None
    for jj in range(maxit):
        if jj < 3:
            stepsize = 0.01 / L
        else:
            stepsize = np.abs(np.sum(S * Y) / np.sum(Y * Y))
            stepsize = np.min((stepsize, 1e10))

        X_p = X
        X = X - stepsize * gradr
        X = manifold.A(X)
        S = X - X_p

        fval, gradf = obj_fun(X)
        gradr_p = gradr
        gradr = manifold.JA(X, gradf)
        Y = gradr - gradr_p

        substationarity = norm(gradr, "fro")
        feasibility = manifold.Feas_eval(X)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        X = manifold.Post_process(X)
        fval, gradf = obj_fun(X)
        gradr = manifold.JA(X, gradf)
        substationarity = norm(gradr, "fro")
        feasibility = manifold.Feas_eval(X)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return X, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def Arrow_Hurwicz_SLPG(X, G, eta, prox, Lambda, manifold, tol=0):
    """Reference Arrow-Hurwicz multiplier update."""
    Lambda_temp = Lambda
    try_stepsize = eta
    Z_tmp = X - try_stepsize * G
    for _ in range(5):
        X_try = prox(
            Z_tmp - try_stepsize * manifold.JC(X, Lambda_temp), try_stepsize
        )
        D_X = 1 / try_stepsize * (X_try - X)
        Lambda_inc = manifold.JC_transpose(X, D_X)
        Lambda_temp = Lambda_temp + Lambda_inc
        if norm(Lambda_inc, "fro") < tol:
            break
    return Lambda_temp


def SLPG(
    obj_fun,
    manifold,
    Xinit=None,
    maxit=100,
    prox=lambda X, eta: X,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference SLPG for a proximable regularizer."""
    kkts, feas, fvals, steps = [], [], [], []

    if Xinit is None:
        Xinit = manifold.Init_point()

    p = manifold._p
    X = Xinit
    fval, gradf = obj_fun(X)
    gradr = manifold.JA(X, gradf)
    L = norm(gradf, "fro") + norm(gradr, "fro")

    Lambda_r = np.zeros([p, p])
    Lambda_r = Arrow_Hurwicz_SLPG(X, gradr, 0.01 / L, prox, Lambda_r, manifold)
    Grad = gradr + manifold.JC(X, Lambda_r)

    S = Y = None
    for jj in range(maxit):
        if jj < 5:
            stepsize = 0.01 / L
        else:
            stepsize = np.abs(np.sum(S * S) / np.sum(S * Y))
            stepsize = np.min((stepsize, 1e10))

        X_p = X
        steps.append(stepsize)

        X = prox(X - stepsize * (gradr + manifold.JC(X, Lambda_r)), stepsize)
        X = manifold.A(X)
        S = X - X_p

        fval, gradf = obj_fun(X)
        Grad_p = Grad
        gradr = manifold.JA(X, gradf)

        stepsize_try = np.average(steps[np.maximum(0, jj - 10) :])
        stepsize_try = np.minimum(np.maximum(stepsize_try, 1e-5 / L), 1e10 / L)

        tol_AW = 1000 * manifold.Feas_eval(X)
        Lambda_r = Arrow_Hurwicz_SLPG(
            X, gradr, stepsize_try, prox, Lambda_r, manifold, tol=tol_AW
        )
        Grad = gradr + manifold.JC(X, Lambda_r)
        Y = Grad - Grad_p

        substationarity = norm(S / stepsize, "fro")
        feasibility = manifold.Feas_eval(X)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        X = manifold.Post_process(X)
        fval, gradf = obj_fun(X)
        feasibility = manifold.Feas_eval(X)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return X, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def SLPG_l21(
    obj_fun,
    manifold,
    Xinit=None,
    maxit=100,
    gamma=0,
    gtol=1e-5,
    post_process=True,
    verbosity=0,
):
    """Reference SLPG for the l_{2,1} regularized objective."""

    def prox(X_input, eta):
        X_ref = np.sqrt(np.sum(X_input**2, axis=1, keepdims=True))
        X_ref_reduce = np.maximum(X_ref - gamma * eta, 0)
        return (X_ref_reduce / (X_ref + 1e-16)) * X_input

    def generate_Lambda_r(X_input):
        X_ref = 1 / (1e-14 + np.sqrt(np.sum(X_input**2, axis=1, keepdims=True)))
        return -X_input.T @ (X_ref * X_input)

    kkts, feas, fvals = [], [], []

    if Xinit is None:
        Xinit = manifold.Init_point()

    X = Xinit
    fval, gradf = obj_fun(X)
    gradr = manifold.JA(X, gradf)
    L = norm(gradf, "fro") + norm(gradr, "fro")

    Lambda_r = gamma * generate_Lambda_r(X)
    Grad = gradr + manifold.JC(X, Lambda_r)

    S = Y = None
    for jj in range(maxit):
        if jj < 5:
            stepsize = 0.001 / L
        else:
            stepsize = np.abs(np.sum(S * S) / np.sum(S * Y))
            stepsize = np.min((stepsize, 1e5))

        X_p = X
        X = prox(X - stepsize * Grad, stepsize)
        X = manifold.A(X)
        S = X - X_p

        fval, gradf = obj_fun(X)
        Grad_p = Grad
        gradr = manifold.JA(X, gradf)

        Lambda_r = gamma * generate_Lambda_r(X)
        Grad = gradr + manifold.JC(X, Lambda_r)
        Y = Grad - Grad_p

        substationarity = norm(S / stepsize, "fro")
        feasibility = manifold.Feas_eval(X)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        X = manifold.Post_process(X)
        fval, gradf = obj_fun(X)
        feasibility = manifold.Feas_eval(X)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return X, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def PenCF(
    Xinit,
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
    X = Xinit
    fval, gradf = obj_fun(X)

    if beta is None:
        beta = 0.1 * norm(gradf, "fro")

    gradr = manifold.JA(X, gradf) + beta * manifold.JC(X, manifold.C(X))
    L = norm(gradf, "fro") + norm(gradr, "fro")

    S = Y = None
    for jj in range(maxit):
        if jj < 3:
            stepsize = 0.01 / L
        else:
            stepsize = np.abs(np.sum(S * Y) / np.sum(Y * Y))
            stepsize = np.min((stepsize, 1e10))

        X_p = X
        X = X - stepsize * gradr

        XX = X.T @ X
        feas_tmp = manifold.Feas_eval(X)
        if feas_tmp > 1e-1:
            if feas_tmp < 0.5:
                X = 1.5 * X - X @ (XX / 2)
            else:
                X = np.linalg.solve((XX + np.eye(p)) / 2, X.T).T

        if norm(X, "fro") > 1.001 * np.sqrt(p):
            X = X * (1.001 * np.sqrt(p) / norm(X, "fro"))

        S = X - X_p

        fval, gradf = obj_fun(X)
        gradr_p = gradr
        gradr = manifold.JA(X, gradf) + beta * manifold.JC(X, manifold.C(X))
        Y = gradr - gradr_p

        substationarity = norm(gradr, "fro")
        feasibility = manifold.Feas_eval(X)

        kkts.append(substationarity)
        feas.append(feasibility)
        fvals.append(fval)

        if substationarity < gtol:
            break

    if post_process:
        X = manifold.Post_process(X)
        fval, gradf = obj_fun(X)
        gradr = manifold.JA(X, gradf)
        substationarity = norm(gradr, "fro")
        feasibility = manifold.Feas_eval(X)
        kkts[-1] = substationarity
        feas[-1] = feasibility
        fvals[-1] = fval

    return X, {
        "kkts": kkts,
        "fvals": fvals,
        "fea": feasibility,
        "kkt": substationarity,
        "fval": fval,
        "feas": feas,
    }


def prox_l1(X_input, eta, gamma=0):
    """Reference proximal operator of the l_1 norm."""
    return np.maximum(X_input - gamma * eta, 0) + np.minimum(
        X_input + gamma * eta, 0
    )


def prox_l21(X_input, eta, gamma=0):
    """Reference proximal operator of the l_{2,1} norm."""
    X_ref = np.sqrt(np.sum(X_input**2, axis=1, keepdims=True))
    X_ref_reduce = np.maximum(X_ref - gamma * eta, 0)
    return X_ref_reduce / (X_ref + 1e-14) * X_input
