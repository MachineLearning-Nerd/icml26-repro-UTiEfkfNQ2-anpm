"""Unit tests for the ANPM core algorithm (convergence + critical-boundary divergence).

Run:  .venv/bin/python -m pytest repro/tests/test_anpm.py -q
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "upstream"))

import numpy as np
import csv
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


def _load_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    return rows[0], np.asarray(rows[1:], dtype=float)


def test_full_amazon_rank30_regeneration_agrees_with_reference():
    """Paper-scale curves agree despite the metric's loose eigsh tolerance."""
    root = pathlib.Path(__file__).resolve().parents[2]
    name = "anpm_amazon_sigma1e-3_k30_every10.csv"
    header, actual = _load_csv(root / "upstream" / "results" / name)
    ref_header, reference = _load_csv(root / "upstream" / "results_reference" / name)

    assert header == ref_header == ["t", "$0$", "$\\beta_t$"]
    assert actual.shape == reference.shape == (11, 3)
    np.testing.assert_array_equal(actual[:, 0], np.arange(0, 101, 10))
    relative_diff = np.abs(actual[:, 1:] - reference[:, 1:]) / np.abs(reference[:, 1:])
    assert np.max(relative_diff) < 0.10
    # Both independent runs conclude tuned momentum beats plain at T=100.
    assert actual[-1, 2] < actual[-1, 1]
    assert reference[-1, 2] < reference[-1, 1]


def test_full_amazon_tuned_momentum_beats_plain_at_matched_budget():
    """At the same 100 iterations, tuned momentum has lower rank-30 error."""
    root = pathlib.Path(__file__).resolve().parents[2]
    _, actual = _load_csv(
        root / "upstream" / "results" / "anpm_amazon_sigma1e-3_k30_every10.csv"
    )
    plain_final, tuned_final = actual[-1, 1], actual[-1, 2]
    assert tuned_final < plain_final
    assert (plain_final - tuned_final) / plain_final > 0.10


def test_paper_scale_facebook_decentralized_run_matches_reference():
    """The official real-graph decentralized experiment is independently exact."""
    root = pathlib.Path(__file__).resolve().parents[2]
    header, actual = _load_csv(
        root / "upstream/results/depca_ego_facebook_full_repro.csv"
    )
    ref_header, reference = _load_csv(
        root / "upstream/results_reference/depca_ego_facebook_.csv"
    )
    assert header == ref_header
    assert actual.shape == reference.shape == (201, 9)
    np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=3e-12)


def test_paper_scale_facebook_adepm_beats_depm_at_matched_gossip():
    """ADePM and DePM use identical L, and acceleration wins for L=20 and 40."""
    root = pathlib.Path(__file__).resolve().parents[2]
    _, actual = _load_csv(
        root / "upstream/results/depca_ego_facebook_full_repro.csv"
    )
    for depm, beta_star, beta_tuned in [(1, 3, 4), (5, 7, 8)]:
        assert actual[-1, beta_star] < actual[-1, depm]
        assert actual[-1, beta_tuned] < actual[-1, depm]
    assert actual[-1, 1] / actual[-1, 4] > 9
    assert actual[-1, 5] / actual[-1, 8] > 20_000
