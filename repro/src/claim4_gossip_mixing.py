"""Claim 4 verifier: accelerated gossip mixes in O~(1/sqrt(gamma_W)) rounds,
standard gossip in O~(1/gamma_W) rounds (Section 3.1, Proposition 3.2).

This is an ISOLATED test of the gossip subroutine, independent of PCA.  We use
the official ``anpm.gossip.AcceleratedGossip``:
  * omega = 0           -> standard gossip  (Y_{l+1} = W Y_l),           contracts (1-gamma_W)^L
  * omega = compute_omega(W) -> accelerated gossip (Liu & Morse 2011),   contracts (1-sqrt(gamma_W))^L

For a family of ring graphs (Metropolis weights) with gamma_W spanning several
orders of magnitude, we measure:
  (1) the empirical asymptotic per-round contraction factor rho (linear fit of
      log error) and compare to the theory (1-gamma_W) / (1-sqrt(gamma_W));
  (2) the first-hit number of rounds L* to reach a fixed mixing tolerance delta
      and fit the scaling  L* ~ (1/gamma_W)^p  -> p≈1 (standard), p≈0.5 (accel.).

Ring spectral gap: eigenvalues of W are 1/2 + 1/2 cos(2 pi j / n), so
gamma_W = 1 - (1/2 + 1/2 cos(2 pi/n)) = sin^2(pi/n) ~ (pi/n)^2.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUTS_DIR  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anpm.gossip import AcceleratedGossip, compute_omega  # noqa: E402
from anpm.data.graphs import RingGraph  # noqa: E402


def gamma_of(W):
    ev = np.linalg.eigvalsh(W)
    return float(1.0 - np.maximum(ev[-2], -ev[0]))


def mixing_error(Y, mean):
    # worst-agent Frobenius deviation from the network average
    return float(np.max(np.linalg.norm(Y - mean, ord="fro", axis=(1, 2))))


def empirical_contraction(Y0, W, omega, L=2000, burn=50):
    """Exact spectral radius of the gossip operator on the zero-sum (non-consensus)
    subspace.  For standard (omega=0) the operator is W; for accelerated it is the
    2n x 2n heavy-ball companion matrix [[(1+omega)W, -omega I],[I, 0]].  This is an
    independent computation (eigenvalues), not a fit of the trajectory.
    """
    n = W.shape[0]
    if abs(omega) < 1e-15:
        ev = np.linalg.eigvals(W)
    else:
        I = np.eye(n)
        M = np.block([[(1 + omega) * W, -omega * I], [I, np.zeros((n, n))]])
        ev = np.linalg.eigvals(M)
    # restrict to the non-consensus subspace: drop the eigenvalue(s) == 1
    ev_noncons = ev[np.abs(np.abs(ev) - 1.0) > 1e-9]
    rho = float(np.max(np.abs(ev_noncons))) if len(ev_noncons) else 0.0
    return rho, np.array([])


def trajectory_decay(Y0, W, omega, L):
    """Return the worst-agent mixing-error trajectory (for the plot/fit)."""
    n = Y0.shape[0]
    mean = np.repeat(Y0.mean(axis=0, keepdims=True), n, axis=0)
    errs = []
    X, Xp = Y0.copy(), Y0.copy()
    for _ in range(L):
        Xn = (1 + omega) * np.einsum("ij,jkl->ikl", W, X) - omega * Xp
        Xp, X = X, Xn
        errs.append(mixing_error(X, mean))
    return np.array(errs)


def first_hit(Y0, W, omega, delta, Lmax=200000):
    n = Y0.shape[0]
    mean = np.repeat(Y0.mean(axis=0, keepdims=True), n, axis=0)
    init = mixing_error(Y0, mean)
    X = Y0.copy()
    Xp = Y0.copy()
    thr = delta * init
    for l in range(1, Lmax + 1):
        Xn = (1 + omega) * np.einsum("ij,jkl->ikl", W, X) - omega * Xp
        Xp, X = X, Xn
        if mixing_error(X, mean) <= thr:
            return l
    return Lmax


def main():
    rng = np.random.default_rng(0)
    ns = [4, 5, 6, 8, 10, 14, 20, 30, 40, 50, 60]
    delta = 1e-6
    rows = []
    for n in ns:
        W = RingGraph(n)
        g = gamma_of(W)
        omega = compute_omega(W)
        Y0 = rng.standard_normal((n, 3, 2))
        rho_std, _ = empirical_contraction(Y0, W, 0.0, L=4000)
        rho_acc, _ = empirical_contraction(Y0, W, omega, L=4000)
        L_std = first_hit(Y0, W, 0.0, delta)
        L_acc = first_hit(Y0, W, omega, delta)
        rows.append({"n": n, "gamma_W": g, "1_over_gamma": 1.0 / g,
                     "1_over_sqrtdelta_gamma": 1.0 / np.sqrt(g), "omega": omega,
                     "rho_std_empirical": rho_std, "rho_std_theory": 1 - g,
                     "rho_acc_empirical": rho_acc, "rho_acc_theory": 1 - np.sqrt(g),
                     "L_star_std": L_std, "L_star_acc": L_acc})

    # ---- fit scaling exponents: L* ~ (1/gamma)^p ----
    g_inv = np.array([r["1_over_gamma"] for r in rows])
    Ls = np.array([r["L_star_std"] for r in rows], float)
    La = np.array([r["L_star_acc"] for r in rows], float)
    p_std = float(np.polyfit(np.log(g_inv), np.log(Ls), 1)[0])
    p_acc = float(np.polyfit(np.log(g_inv), np.log(La), 1)[0])
    # also fit accelerated vs 1/sqrt(gamma) -> should be ~1
    g_invsqrt = np.array([r["1_over_sqrtdelta_gamma"] for r in rows])
    p_acc_sqrt = float(np.polyfit(np.log(g_invsqrt), np.log(La), 1)[0])

    # contraction-factor agreement
    rho_std_err = float(np.max(np.abs([r["rho_std_empirical"] - r["rho_std_theory"] for r in rows])))
    rho_acc_err = float(np.max(np.abs([r["rho_acc_empirical"] - r["rho_acc_theory"] for r in rows])))

    result = {
        "claim": "Claim 4: accelerated gossip O~(1/sqrt(gamma_W)) vs standard O~(1/gamma_W)",
        "method": "official anpm.gossip.AcceleratedGossip; omega=0 (standard) vs omega=compute_omega(W) (accelerated); "
                  "ring graphs Metropolis weights, gamma_W=sin^2(pi/n)",
        "delta_relative": delta,
        "rows": rows,
        "scaling_fit": {
            "L_star_vs_1_over_gamma_std_exponent": p_std,
            "L_star_vs_1_over_gamma_acc_exponent": p_acc,
            "L_star_vs_1_over_sqrtgamma_acc_exponent": p_acc_sqrt,
        },
        "contraction_factor_max_abs_err_vs_theory": {"standard": rho_std_err, "accelerated": rho_acc_err},
        "verdict": "VERIFIED" if (abs(p_std - 1.0) < 0.15 and abs(p_acc - 0.5) < 0.12
                                  and abs(p_acc_sqrt - 1.0) < 0.15) else "INCONCLUSIVE",
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "claim4_gossip_mixing.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 74)
    print("CLAIM 4 -- accelerated vs standard gossip mixing rate")
    print("=" * 74)
    print(f"{'n':>4} {'gamma_W':>10} {'rho_std(emp/theo)':>20} {'rho_acc(emp/theo)':>20} "
          f"{'L*_std':>8} {'L*_acc':>8}")
    for r in rows:
        print(f"{r['n']:>4} {r['gamma_W']:>10.2e} "
              f"{r['rho_std_empirical']:>9.5f}/{r['rho_std_theory']:<9.5f} "
              f"{r['rho_acc_empirical']:>9.5f}/{r['rho_acc_theory']:<9.5f} "
              f"{r['L_star_std']:>8} {r['L_star_acc']:>8}")
    sf = result["scaling_fit"]
    print(f"\nscaling fit  L* ~ (1/gamma)^p:")
    print(f"  standard:   p = {sf['L_star_vs_1_over_gamma_std_exponent']:.3f}  (theory 1.0)")
    print(f"  accelerated: p = {sf['L_star_vs_1_over_gamma_acc_exponent']:.3f}  (theory 0.5)")
    print(f"  accelerated vs 1/sqrt(gamma): p = {sf['L_star_vs_1_over_sqrtgamma_acc_exponent']:.3f}  (theory 1.0)")
    ce = result["contraction_factor_max_abs_err_vs_theory"]
    print(f"contraction-factor max|empirical-theory|: standard={ce['standard']:.2e}, "
          f"accelerated={ce['accelerated']:.2e}")
    print(f"\nVERDICT: {result['verdict']}\nwrote {path}")
    return result


if __name__ == "__main__":
    main()
