"""Shared utilities for the ANPM (UTiEfkfNQ2) claim verifiers.

All metrics here are independent re-implementations (not re-exports of the
official code) so they serve as an *independent checker*.  They are numerically
cross-checked against the official ``anpm.metrics.sin_thetak`` in the test suite.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Make the vendored official package importable as `anpm`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPSTREAM_PKG = os.path.join(REPO_ROOT, "repro")
if UPSTREAM_PKG not in sys.path:
    sys.path.insert(0, UPSTREAM_PKG)

DATA_DIR = os.path.join(REPO_ROOT, "repro", "datasets")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, ".openresearch", "artifacts")


def sin_thetak_indep(U: np.ndarray, X: np.ndarray, k: int) -> float:
    """sin of the k-th principal angle between range(X) and range(U).

    X, U are (column-)orthonormal.  Uses the singular values of U^T X: the
    k-th principal angle satisfies cos(theta_k) = sigma_min(U_k^T X) when X
    spans (at least) a k-dim space, so sin(theta_k) = sqrt(1 - sigma_min^2).
    Equivalent to the smallest singular value of (I - U U^T) X projected.
    """
    p = min(X.shape[1], U.shape[1])
    s = np.linalg.svd(U.T @ X, compute_uv=False)
    # s sorted descending; cos(theta_k) (k-th/largest principal angle) = sigma_min.
    cos_k = s[-1]
    cos_k = float(min(1.0, max(0.0, cos_k)))
    return float(np.sqrt(max(0.0, 1.0 - cos_k * cos_k)))


def tan_thetak(Uk: np.ndarray, Uminus: np.ndarray, X: np.ndarray) -> float:
    """||H||_2 = tan(theta_k) where H = (U_{-k}^T X)(U_k^T X)^{-1}.

    This is the exact quantity the paper's proof bounds (Lemma C.4 onward).
    """
    UktX = Uk.T @ X
    s = np.linalg.svd(UktX, compute_uv=False)
    sigma_min = s[-1]
    if sigma_min < 1e-14:
        return float("inf")
    H = (Uminus.T @ X) @ np.linalg.inv(UktX)
    return float(np.linalg.norm(H, ord=2))


def anpm_manual(A: np.ndarray, beta: float, T: int, X0: np.ndarray,
                Xi: np.ndarray) -> np.ndarray:
    """Independent re-implementation of ANPM (Eq. 1) returning the X_t list.

    Uses the paper's sign-canonical QR (non-negative diagonal R, Section 1.3).
    Used as an independent checker alongside the official ``anpm.anpm``.
    """
    X_prev = X0.copy()
    Y = 0.5 * A @ X0 + Xi[0]
    X, R = np.linalg.qr(Y, mode="reduced")
    # fix QR signs so R has positive diagonal (matches a canonical QR)
    sg = np.sign(np.diag(R))
    sg[sg == 0] = 1.0
    X = X * sg
    R = R * sg[:, None]
    xs = [X0, X]
    for t in range(1, T):
        Y = A @ X - beta * X_prev @ np.linalg.inv(R) + Xi[t]
        X_new, R = np.linalg.qr(Y, mode="reduced")
        sg = np.sign(np.diag(R))
        sg[sg == 0] = 1.0
        X_new = X_new * sg
        R = R * sg[:, None]
        X, X_prev = X_new, X
        xs.append(X)
    return np.array(xs)


def cheb_p(t: int, x: float, beta: float) -> float:
    """Scaled Chebyshev-like polynomial p_t(x): p0=1, p1=x/2, p_{t+1}=x p_t - beta p_{t-1}."""
    if t == 0:
        return 1.0
    if t == 1:
        return x / 2.0
    a, b = 1.0, x / 2.0
    for _ in range(t - 1):
        a, b = b, x * b - beta * a
    return b


def cheb_q(t: int, x: float, beta: float) -> float:
    """q_t(x): q0=1, q1=x, q_{t+1}=x q_t - beta q_{t-1}."""
    if t == 0:
        return 1.0
    if t == 1:
        return x
    a, b = 1.0, x
    for _ in range(t - 1):
        a, b = b, x * b - beta * a
    return b


def rel_gap(lambda_k: float, beta: float) -> float:
    """Delta = (lambda_k - 2 sqrt(beta)) / lambda_k (paper's gap parameter)."""
    return (lambda_k - 2.0 * np.sqrt(beta)) / lambda_k


def anpm_first_hit(A, beta, X0, Uk, k, eps_target, Tcap, Xi=None,
                   noise_fn=None):
    """Run ANPM step-by-step, stopping the moment sin theta_k <= eps_target.

    Returns (T_hit, sin_theta_trajectory). T_hit = -1 if never reached within Tcap.
    Sign-canonical QR (paper Section 1.3).  ``noise_fn(t, X)`` optionally returns
    the t-th perturbation (for state-dependent noise); else ``Xi[t]`` is used.
    """
    X_prev = X0.copy()
    if noise_fn is not None:
        xi0 = noise_fn(0, X0)
    else:
        xi0 = Xi[0]
    Y = 0.5 * A @ X0 + xi0
    X, R = np.linalg.qr(Y, mode="reduced")
    sg = np.sign(np.diag(R)); sg[sg == 0] = 1.0
    X = X * sg; R = R * sg[:, None]
    errs = [sin_thetak_indep(Uk[:, :k] if Uk.shape[1] >= k else Uk, X[:, :k], k)]
    hit = 0 if errs[0] <= eps_target else -1
    for t in range(1, Tcap):
        xi = noise_fn(t, X) if noise_fn is not None else Xi[t]
        Y = A @ X - beta * X_prev @ np.linalg.inv(R) + xi
        X_new, R = np.linalg.qr(Y, mode="reduced")
        sg = np.sign(np.diag(R)); sg[sg == 0] = 1.0
        X_new = X_new * sg; R = R * sg[:, None]
        X, X_prev = X_new, X
        e = sin_thetak_indep(Uk[:, :k] if Uk.shape[1] >= k else Uk, X[:, :k], k)
        errs.append(e)
        if hit < 0 and e <= eps_target:
            hit = t
            break
    return hit, np.array(errs)
