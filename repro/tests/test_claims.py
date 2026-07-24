"""Cross-validation tests: independent re-implementations match the official code."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))       # repo root (for repro/)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from common import sin_thetak_indep, anpm_manual, rel_gap
from anpm.metrics import sin_thetak as sin_official
from anpm.anpm import anpm as anpm_official


def test_sin_thetak_matches_official():
    rng = np.random.default_rng(0)
    d, k = 40, 4
    U = np.linalg.qr(rng.standard_normal((d, d)))[0][:, :k]
    X = np.linalg.qr(rng.standard_normal((d, k)))[0]
    # introduce a known misalignment
    X = 0.9 * X + 0.1 * np.linalg.qr(rng.standard_normal((d, k)))[0]
    X, _ = np.linalg.qr(X)
    a = sin_thetak_indep(U, X, k)
    b = sin_official(U, X, k)
    assert abs(a - b) < 1e-6, f"indep {a} vs official {b}"


def test_anpm_manual_converges_to_same_subspace_as_official():
    """The sign-canonical QR (paper Sec 1.3) and numpy QR are both valid ANPM;
    both must converge to the same top-k subspace (up to the noise floor)."""
    rng = np.random.default_rng(1)
    d, k = 30, 3
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    lam = np.array([5.0, 5.0, 1.0, 0.9, 0.5] + [0.5] * (d - 5))
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    X0 = np.linalg.qr(rng.standard_normal((d, k)))[0]
    Xi = 1e-5 * rng.standard_normal((300, d, k))
    Xa = anpm_manual(A, 0.2, 300, X0, Xi)
    Xb = anpm_official(A, 0.2, 300, X0, Xi)
    U = Q[:, :k]
    sa = sin_thetak_indep(U, Xa[-1], k)
    sb = sin_thetak_indep(U, Xb[-1], k)
    assert sa < 1e-3 and sb < 1e-3, f"both must converge: manual={sa}, official={sb}"
    assert abs(sa - sb) < 5e-3, f"final subspaces differ: manual={sa}, official={sb}"


def test_rel_gap_and_convergence():
    # noiseless ANPM converges monotonically below initial error
    rng = np.random.default_rng(2)
    d, k = 30, 3
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    lam_kp1 = 0.9
    lam = np.array([5.0, 5.0, 1.0, lam_kp1, 0.5] + [0.5] * (d - 5))
    A = Q @ np.diag(lam) @ Q.T
    A = 0.5 * (A + A.T)
    X0 = np.linalg.qr(rng.standard_normal((d, k)))[0]
    beta = lam_kp1 ** 2 / 4
    assert rel_gap(1.0, beta) > 0
    Xi = np.zeros((100, d, k))
    Xl = anpm_manual(A, beta, 100, X0, Xi)
    U = Q[:, :k]
    errs = np.array([sin_thetak_indep(U, Xt, k) for Xt in Xl])
    assert errs[-1] < errs[0] * 0.01
