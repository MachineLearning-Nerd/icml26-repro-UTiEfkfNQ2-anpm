"""Claim 2 verifier: Theorems 2.4 & 2.5 (tightness of noise conditions 3 & 4).

Both theorems are *existential counterexample* statements (Appendix C.4): there
exist an initial point X_0 (cos theta_k(U_k, X_0) > 0) and perturbations Xi_t
that satisfy a *relaxed* version of the noise conditions (constant >> c=1/32 of
Theorem 2.2) yet for which ANPM NEVER reaches tan theta_k <= eps.

The proofs give *explicit* constructions, reconstructed exactly here:

  Shared setup:
      A = diag(lambda_k I_k, 2 sqrt(beta) I_{d-k}),  lambda_k > 2 sqrt(beta) > 0.
      X_0 cols 1..k-1 = v_1..v_{k-1}; col k = cos(theta0) v_k + sin(theta0) v_{k+1},
        theta0 = arctan(2 eps)  =>  tan theta_k(U_k, X_0) = 2 eps > eps.

  Theorem 2.4 (condition (3) tight):  Xi_t = 8 (lam_k - 2 sb) eps [0,..,v_{k+1}] (constant).
  Theorem 2.5 (condition (4) tight):  Xi_0  = -(1/2)(lam_k-2 sb) cos theta_k [0,..,v_k];
                                      Xi_t  = -    (lam_k-2 sb) cos theta_k [0,..,v_k], t>=1.

QR convention.  The paper (Section 1.3) defines QR with **non-negative diagonal**
R.  This sign-canonical QR is REQUIRED for the exact counterexample dynamics
(Theorem 2.5's perturbation uses cos theta_k, the principal-angle cosine which is
sign-invariant; `numpy.linalg.qr` has sign freedom that breaks the collapse).
We therefore use the paper-faithful sign-canonical QR as the primary runner and
report the official `anpm.anpm` (numpy QR) as a secondary cross-check.

For each theorem we verify: (A) relaxed noise condition holds (constant >> 1/32);
(B) the complementary condition holds; (C) cos theta_k(X_0) > 0; (D) ANPM keeps
tan theta_k > eps for the whole horizon (counterexample works); (E) the analytic
lower bound phi(t) from the proof is <= the observed h_t and itself > eps; and a
contrast showing the STRICT constant c=1/32 DOES converge.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (anpm_manual, rel_gap, sin_thetak_indep,  # noqa: E402
                    tan_thetak, OUTPUTS_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anpm.anpm import anpm as anpm_official  # noqa: E402


def build_setup(d, k, lam_k, beta, eps):
    sb = 2.0 * np.sqrt(beta)
    A = np.diag(np.concatenate([np.full(k, lam_k), np.full(d - k, sb)]))
    e = np.eye(d)
    Uk, Uminus = e[:, :k], e[:, k:d]
    theta0 = np.arctan(2.0 * eps)
    X0 = np.zeros((d, k))
    X0[:, :k - 1] = e[:, :k - 1]
    X0[:, k - 1] = np.cos(theta0) * e[:, k - 1] + np.sin(theta0) * e[:, k]
    return A, Uk, Uminus, X0, theta0


def phi_lower_bound(t, lam_k, beta, eps):
    lam_k_plus = (lam_k + np.sqrt(lam_k ** 2 - 4.0 * beta)) / 2.0
    gamma = np.sqrt(beta) / lam_k_plus
    return 2.0 * eps * (1.0 - t * (1.0 - gamma) * gamma ** t)


def run_one(name, d, k, lam_k, beta, eps, T, Xi):
    A, Uk, Uminus, X0, theta0 = build_setup(d, k, lam_k, beta, eps)
    X_faith = anpm_manual(A, beta, T, X0, Xi)          # paper-faithful QR (primary)
    X_off = anpm_official(A, beta, T, X0, Xi)          # official numpy QR (cross-check)
    h = np.array([tan_thetak(Uk, Uminus, Xt) for Xt in X_faith])
    h_off = np.array([tan_thetak(Uk, Uminus, Xt) for Xt in X_off])
    sin_h = np.array([sin_thetak_indep(Uk, Xt, k) for Xt in X_faith])
    cos0 = float(np.linalg.svd(Uk.T @ X0, compute_uv=False)[-1])

    gap = lam_k - 2.0 * np.sqrt(beta)
    c_strict = 1.0 / 32.0
    UbotXi = np.array([np.linalg.norm(Xi[t][k:, :], ord=2) for t in range(T)])
    UtopXi = np.array([np.linalg.norm(Xi[t][:k, :], ord=2) for t in range(T)])
    if name == "2.4":
        const = 8.0
        relaxed_ok = bool(UbotXi.max() <= const * gap * eps + 1e-12)
        comp_zero = bool(UtopXi.max() < 1e-12)
    else:
        const = 1.0
        cos_t = np.array([float(np.linalg.svd(Uk.T @ Xt, compute_uv=False)[-1])
                          for Xt in X_faith[:T]])
        relaxed_ok = bool(np.all(UtopXi <= const * gap * cos_t + 1e-12))
        comp_zero = bool(UbotXi.max() < 1e-12)

    phi = np.array([phi_lower_bound(t, lam_k, beta, eps) for t in range(len(h))])
    res = {
        "theorem": name,
        "params": {"d": d, "k": k, "lambda_k": lam_k, "beta": beta,
                   "two_sqrt_beta": 2 * np.sqrt(beta), "eps": eps,
                   "Delta": float(rel_gap(lam_k, beta))},
        "assumptions": {"cos_theta_k_X0": cos0, "cos_positive": bool(cos0 > 0),
                        "tan_theta_k_X0": float(h[0]), "equals_2eps": bool(abs(h[0] - 2 * eps) < 1e-9)},
        "noise": {"strict_c": c_strict, "relaxed_constant": const,
                  "ratio_relaxed_over_strict": const / c_strict,
                  "relaxed_condition_holds": relaxed_ok,
                  "complementary_condition_zero": comp_zero},
        "divergence": {"horizon": T, "min_tan_theta": float(np.min(h)),
                       "max_tan_theta": float(np.max(h)),
                       "min_sin_theta": float(np.min(sin_h)),
                       "never_reaches_eps": bool(np.all(h > eps - 1e-12)),
                       "official_qr_min_tan_theta": float(np.min(h_off)),
                       "official_qr_never_reaches_eps": bool(np.all(h_off > eps - 1e-12))},
        "analytic_bound_phi": {"phi_min": float(np.min(phi)),
                               "phi_above_eps": bool(np.min(phi) > eps),
                               "phi_le_observed_h": bool(np.all(phi <= h + 1e-9))},
    }
    return res, h, X_faith


def main():
    d, k, lam_k, beta, eps, T = 8, 2, 1.0, 0.16, 1e-2, 4000
    gap = lam_k - 2.0 * np.sqrt(beta)
    A, Uk, Uminus, X0, theta0 = build_setup(d, k, lam_k, beta, eps)
    cos0 = float(np.cos(theta0))

    # ---- Theorem 2.4 ----
    Xi24 = np.zeros((T, d, k)); Xi24[:, k, k - 1] = 8.0 * gap * eps      # constant, v_{k+1}, col k
    r24, h24, _ = run_one("2.4", d, k, lam_k, beta, eps, T, Xi24)

    # ---- Theorem 2.5 ----
    Xi25 = np.zeros((T, d, k))
    Xi25[0, k - 1, k - 1] = -0.5 * gap * cos0
    Xi25[1:, k - 1, k - 1] = -gap * cos0
    r25, h25, _ = run_one("2.5", d, k, lam_k, beta, eps, T, Xi25)

    # ---- contrast: STRICT constant c=1/32 along U_-k DOES converge ----
    c = 1.0 / 32.0
    Xi_strict = np.zeros((T, d, k)); Xi_strict[:, k, k - 1] = c * gap * eps
    Xs = anpm_manual(A, beta, T, X0, Xi_strict)
    h_s = np.array([tan_thetak(Uk, Uminus, Xt) for Xt in Xs])
    strict_converges = bool(h_s[-1] <= eps)

    verdict = "VERIFIED" if (r24["divergence"]["never_reaches_eps"]
                            and r25["divergence"]["never_reaches_eps"]
                            and strict_converges) else "FALSIFIED"
    result = {"claim": "Claim 2: Theorems 2.4 & 2.5 (noise conditions cannot be relaxed)",
              "thm_2_4": r24, "thm_2_5": r25,
              "strict_constant_converges": {"c": c, "final_tan_theta": float(h_s[-1]),
                                            "converges": strict_converges},
              "verdict": verdict}

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "claim2_counterexamples.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 74)
    print("CLAIM 2 -- Theorems 2.4 & 2.5 (noise conditions cannot be relaxed)")
    print("=" * 74)
    for r in (r24, r25):
        p, a, n, dv, lb = r["params"], r["assumptions"], r["noise"], r["divergence"], r["analytic_bound_phi"]
        print(f"\n[Theorem {r['theorem']}] {p}")
        print(f"  cos theta_k(X0)={a['cos_theta_k_X0']:.5f}>0:{a['cos_positive']}; "
              f"tan theta_k(X0)={a['tan_theta_k_X0']:.5f} (=2eps):{a['equals_2eps']}")
        print(f"  relaxed const={n['relaxed_constant']} (strict c=1/32={n['strict_c']:.5f}), "
              f"x{n['ratio_relaxed_over_strict']:.0f}; relaxed cond holds:{n['relaxed_condition_holds']}, "
              f"complementary=0:{n['complementary_condition_zero']}")
        print(f"  T={dv['horizon']}: min tan theta={dv['min_tan_theta']:.5f}, "
              f"max={dv['max_tan_theta']:.5f}; never<=eps={dv['never_reaches_eps']}")
        print(f"  official-numpy-QR: min tan theta={dv['official_qr_min_tan_theta']:.5e} "
              f"(QR sign freedom can differ; paper uses non-neg diagonal QR, Sec 1.3)")
        print(f"  analytic phi: min={lb['phi_min']:.5f}>eps:{lb['phi_above_eps']}, "
              f"phi<=h:{lb['phi_le_observed_h']}")
    sc = result["strict_constant_converges"]
    print(f"\n[Contrast] strict c=1/32: final tan theta={sc['final_tan_theta']:.3e}, "
          f"converges:{sc['converges']}")
    print(f"\nVERDICT: {verdict}\nwrote {path}")
    return result


if __name__ == "__main__":
    main()
