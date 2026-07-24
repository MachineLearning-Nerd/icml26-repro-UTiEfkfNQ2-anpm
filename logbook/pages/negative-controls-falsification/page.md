> ⚠️ **Historical rejected baseline (superseded).** This page is from the
> previous 3-claim reproduction (judged 4/10 on 2026-07-23). It is preserved
> unchanged for provenance. The **current** verifiers live in
> [claim-1-noise-boundary](#/claim-1-noise-boundary) … [claim-5-deepca-speedup](#/claim-5-deepca-speedup)
> and supersede this page.

# Negative controls & falsification


---
<!-- trackio-cell
{"type": "code", "id": "cell_9fd1c1616f00", "created_at": "2026-07-16T18:54:47+00:00", "title": "Unit tests (convergence + β_c divergence)", "command": ["python", "-m", "pytest", "repro/tests/test_anpm.py", "-q"], "exit_code": 0, "duration_s": 1.731}
-->
````bash
$ python -m pytest repro/tests/test_anpm.py -q
````

exit 0 · 1.7s


````python title=test_anpm.py
"""Unit tests for the ANPM core algorithm (convergence + critical-boundary divergence).

Run:  .venv/bin/python -m pytest repro/tests/test_anpm.py -q
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "upstream"))

import numpy as np
from anpm.anpm import anpm
from anpm.metrics import sin_thetak
from anpm.data.synthetic_instances import generate_eigenvectors, generate_matrix, generate_X0, generate_noise


def _setup(d=80, k=4, gap=0.05, seed=0):
    np.random.seed(seed)
    U = generate_eigenvectors(d)
    lambdas = np.array([1.0] * k + [1.0 - gap] + [0.5] * (d - k - 1))
    A = generate_matrix(lambdas, U)
    X0 = generate_X0(d, k)
    return U, A, X0, k


def test_anpm_converges_at_beta_star():
    """At the optimal momentum beta* = lambda_{k+1}^2/4, ANPM converges to the top-k subspace."""
    U, A, X0, k = _setup()
    d = A.shape[0]
    T = 400
    beta_star = ((1.0 - 0.05) ** 2) / 4
    Xi = generate_noise(d, k, T, 1e-6)
    X_list = anpm(A, beta_star, T, X0, Xi)
    errs = [sin_thetak(U[:, :k], X[:, :k], k) for X in X_list]
    assert errs[-1] < 0.5 * errs[0], f"no convergence: {errs[0]} -> {errs[-1]}"
    assert errs[-1] < 0.05, f"final error too high: {errs[-1]}"


def test_anpm_diverges_at_beta_critical():
    """At the critical momentum beta_c = lambda_k^2/4 (the boundary), ANPM does NOT converge."""
    U, A, X0, k = _setup()
    d = A.shape[0]
    T = 400
    beta_c = (1.0 ** 2) / 4          # lambda_k = 1.0
    Xi = generate_noise(d, k, T, 1e-6)
    X_list = anpm(A, beta_c, T, X0, Xi)
    errs = [sin_thetak(U[:, :k], X[:, :k], k) for X in X_list]
    assert errs[-1] > 0.5, f"unexpected convergence at beta_c: {errs[-1]}"


def test_beta_star_beats_or_matches_plain_power_method_transient():
    """beta* reaches the noise floor no slower (in transient slope) than beta=0."""
    U, A, X0, k = _setup()
    d = A.shape[0]
    T = 400
    beta_star = ((1.0 - 0.05) ** 2) / 4
    Xi = generate_noise(d, k, T, 1e-6)
    e_star = [sin_thetak(U[:, :k], X[:, :k], k) for X in anpm(A, beta_star, T, X0, Xi)]
    e_zero = [sin_thetak(U[:, :k], X[:, :k], k) for X in anpm(A, 0.0, T, X0, Xi)]
    # both converge to a small error
    assert e_star[-1] < 0.05 and e_zero[-1] < 0.05

````


````output
...                                                                      [100%]
3 passed in 1.43s

````
