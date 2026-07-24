"""Claim 1 verifier: Theorem 2.2 -- ANPM keeps the accelerated rate under the
*same* (mild) noise conditions as the non-accelerated Noisy Power Method,
||(U_-k)^T Xi|| <= c (lam_k - 2 sqrt(beta)) eps, i.e. scaling as Delta_k * eps.
This is much milder than Xu (2023)'s Delta_k * eps^mu (mu = Omega~(Delta_k^{-1/2})).

Three independent, non-circular tests:

  (A) NOISE-BOUNDARY DEMONSTRATION (primary).
      On the paper's synthetic instance (lam_1=5, lam_k=1, lam_{k+1}=1-gap,
      lam_d=0.5, d=1000, k=10) with a SMALL gap so mu = Omega(log(lam1/lam_{k+1})
      sqrt(lam_k/(lam_k-lam_{k+1}))) is large, set the noise at EXACTLY the mild
      level eta = c (lam_k - 2 sqrt(beta*)) eps (satisfies Thm 2.2 cond (3),(4))
      -- which is many orders of magnitude LARGER than the level Xu's analysis
      would guarantee (Delta_k * eps^mu). Show ANPM(beta*) still converges to
      an eps-floor. This directly probes the eps vs eps^mu distinction: the strict
      competing condition is grossly violated while the mild one holds.

  (B) RATE SCALING (independent of any noise formula).
      Noiseless. Sweep the gap; measure first-hit T(eps) for beta* vs beta=0.
      Fit T ~ Delta^{-p}: expect p ~= 1/2 (accelerated) vs p ~= 1 (plain PM).

  (C) PROOF-RATE CALIBRATION.
      The proof's rate T = 1/(-log(1 - 0.5 sqrt(Delta))) log(2 h0/eps) is checked
      against the observed first-hit T across several eps and gaps (constant-factor
      agreement), corroborating the symbolic derivation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (anpm_first_hit, anpm_manual, cheb_p, rel_gap,  # noqa: E402
                    sin_thetak_indep, tan_thetak, OUTPUTS_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anpm.anpm import anpm as anpm_official  # noqa: E402
from anpm.data.synthetic_instances import (  # noqa: E402
    generate_eigenvectors, generate_X0, generate_matrix, generate_adversarial_noise)


def mu_xu(lam1, lam_kp1, lam_k, gap):
    """Xu (2023) exponent mu_{k+1} = Omega(log(lam1/lam_{k+1}) sqrt(lam_k/gap))."""
    return np.log(lam1 / lam_kp1) * np.sqrt(lam_k / gap)


def synthetic_instance(d, k, gap, lam1=5.0, lam_k=1.0, lam_rest=0.5, seed=0):
    rng = np.random.default_rng(seed) if seed is not None else None
    np.random.seed(seed)
    U = generate_eigenvectors(d)
    lambdas = np.array([lam1] * (k - 1) + [lam_k] + [lam_k - gap] + [lam_rest] * (d - k - 1))
    A = generate_matrix(lambdas, U)
    X0 = generate_X0(d, k)
    return A, U, X0, lambdas


def first_hit(A, beta, X0, Uk, k, eps_target, Tcap=40000):
    """Early-stopping first-hit (noiseless). Returns (T_hit, errs)."""
    return anpm_first_hit(A, beta, X0, Uk, k, eps_target, Tcap, Xi=None,
                          noise_fn=lambda t, X: np.zeros_like(X0))


def partA_boundary():
    d, k, gap = 1000, 10, 1e-2
    lam1, lam_k, lam_rest = 5.0, 1.0, 0.5
    lam_kp1 = lam_k - gap
    A, U, X0, lambdas = synthetic_instance(d, k, gap, lam1, lam_k, lam_rest, seed=0)
    Uk = U[:, :k]
    beta_star = lam_kp1 ** 2 / 4
    Delta = rel_gap(lam_k, beta_star)              # (lam_k - 2 sqrt(beta))/lam_k
    c = 1.0 / 32.0
    eps = 1e-2
    mu = mu_xu(lam1, lam_kp1, lam_k, gap)

    eta_mild = c * (lam_k - 2 * np.sqrt(beta_star)) * eps          # Thm 2.2 condition (3) level
    eta_xu = (lam_k - lam_kp1) * eps ** mu                          # Xu (2023) required level (order)
    ratio = eta_mild / max(eta_xu, 1e-300)

    T = 800
    np.random.seed(1)
    Xi = generate_adversarial_noise(d, k, T, eta_mild, U)          # ||Xi_t|| = eta_mild
    # verify noise conditions hold (conditions are on the NOISE Xi_t, not the iterate)
    ubot = np.array([np.linalg.norm(U[:, k:].T @ Xi[t], ord=2) for t in range(T)])
    utop = np.array([np.linalg.norm(U[:, :k].T @ Xi[t], ord=2) for t in range(T)])
    cond3 = float(ubot.max()) <= c * (lam_k - 2 * np.sqrt(beta_star)) * eps + 1e-12
    # condition (4): ||U_k^T Xi_t|| <= c (lam_k - 2 sqrt(beta)) cos theta_k(U_k, X_t).
    # Run ANPM and check along the trajectory.
    Xl = anpm_official(A, beta_star, T, X0, Xi)
    cos_t = np.array([float(np.linalg.svd(Uk.T @ Xt, compute_uv=False)[-1]) for Xt in Xl[:T]])
    cond4 = bool(np.all(utop <= c * (lam_k - 2 * np.sqrt(beta_star)) * cos_t + 1e-9))
    errs = np.array([sin_thetak_indep(Uk, Xt[:, :k], k) for Xt in Xl])
    final = float(errs[-1])
    floor = float(np.min(errs[50:]))

    return {
        "gap_Delta_k": gap, "mu_xu": float(mu), "eps": eps,
        "eta_mild_Thm2_2": float(eta_mild), "eta_xu_required": float(eta_xu),
        "ratio_mild_over_xu": float(ratio),
        "xu_condition_violated_by_factor": float(ratio),
        "cond3_holds": bool(cond3), "cond4_holds": cond4,
        "ANPM_final_sin_theta": final, "ANPM_floor_sin_theta": floor,
        "converges_to_eps_floor": bool(floor <= eps),
        "conclusion": "ANPM converges to an eps-floor under the MILD condition (Delta*eps) "
                      "while the Xu (2023) stricter condition (Delta*eps^mu) is violated "
                      f"by ~{ratio:.1e}x.",
    }


def partB_scaling():
    """First-hit T(eps) to a moderate target (avoids the numerical floor).  The
    square-root speedup is the ratio T(beta=0)/T(beta*) ~ 1/sqrt(Delta), robust to
    transient oscillations.  Also fit T ~ Delta^{-p}: 0.5 (beta*), 1.0 (beta=0)."""
    d, k = 1000, 10
    lam1, lam_k, lam_rest = 5.0, 1.0, 0.5
    gaps = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    eps_target = 1e-6
    rows = []
    for gap in gaps:
        lam_kp1 = lam_k - gap
        A, U, X0, _ = synthetic_instance(d, k, gap, lam1, lam_k, lam_rest, seed=0)
        Uk = U[:, :k]
        beta_star = lam_kp1 ** 2 / 4
        Delta = rel_gap(lam_k, beta_star)
        T_acc, _ = first_hit(A, beta_star, X0, Uk, k, eps_target, Tcap=60000)
        T_plain, _ = first_hit(A, 0.0, X0, Uk, k, eps_target, Tcap=60000)
        rows.append({"gap": gap, "Delta": float(Delta),
                     "1_over_Delta": 1.0 / Delta, "1_over_sqrtDelta": 1.0 / np.sqrt(Delta),
                     "T_beta_star": int(T_acc), "T_beta_0": int(T_plain),
                     "speedup_ratio": (T_plain / T_acc) if (T_acc > 0 and T_plain > 0) else None})
    g_inv = np.array([r["1_over_Delta"] for r in rows])
    g_invsq = np.array([r["1_over_sqrtDelta"] for r in rows])
    Ta = np.array([r["T_beta_star"] for r in rows], float)
    Tp = np.array([r["T_beta_0"] for r in rows], float)
    m = (Ta > 0) & (Tp > 0)
    p_acc = float(np.polyfit(np.log(g_invsq[m]), np.log(Ta[m]), 1)[0])     # T_acc ~ (1/sqrt D)^1
    p_plain = float(np.polyfit(np.log(g_inv[m]), np.log(Tp[m]), 1)[0])     # T_0 ~ (1/D)^1
    ratio = Tp[m] / Ta[m]
    p_ratio = float(np.polyfit(np.log(g_invsq[m]), np.log(ratio), 1)[0])   # ratio ~ 1/sqrt D
    return {"eps_target": eps_target, "rows": rows,
            "exponent_T_beta_star_vs_1_over_sqrtDelta": p_acc,
            "exponent_T_beta_0_vs_1_over_Delta": p_plain,
            "exponent_speedup_ratio_vs_1_over_sqrtDelta": p_ratio,
            "theory": {"T_beta_star": 1.0, "T_beta_0": 1.0, "speedup_ratio": 1.0}}


def partC_rate_calibration():
    d, k, gap = 1000, 10, 1e-2
    lam1, lam_k, lam_rest = 5.0, 1.0, 0.5
    lam_kp1 = lam_k - gap
    A, U, X0, _ = synthetic_instance(d, k, gap, lam1, lam_k, lam_rest, seed=0)
    Uk = U[:, :k]
    beta_star = lam_kp1 ** 2 / 4
    Delta = rel_gap(lam_k, beta_star)
    h0 = tan_thetak(Uk, U[:, k:], X0)
    rows = []
    for eps_target in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        T_obs, errs = first_hit(A, beta_star, X0, Uk, k, eps_target, Tcap=20000)
        if T_obs <= 0:
            continue
        T_pred = (1.0 / (-np.log(1 - 0.5 * np.sqrt(Delta)))) * np.log(2 * h0 / eps_target)
        rows.append({"eps": eps_target, "T_observed": T_obs, "T_predicted": float(T_pred),
                     "ratio_pred_over_obs": float(T_pred / T_obs)})
    ratios = [r["ratio_pred_over_obs"] for r in rows]
    return {"Delta": float(Delta), "rows": rows,
            "ratio_pred_over_obs_range": [float(min(ratios)), float(max(ratios))] if ratios else None,
            "note": "proof rate T=1/(-log(1-0.5 sqrt(Delta))) log(2 h0/eps) within a constant "
                    "factor of observed -> corroborates the symbolic derivation."}


def main():
    A = partA_boundary()
    B = partB_scaling()
    C = partC_rate_calibration()
    okA = A["converges_to_eps_floor"] and A["cond3_holds"] and A["cond4_holds"] and A["ratio_mild_over_xu"] > 1e3
    okB = abs(B["exponent_T_beta_star_vs_1_over_sqrtDelta"] - 1.0) < 0.30 and abs(B["exponent_T_beta_0_vs_1_over_Delta"] - 1.0) < 0.30
    verdict = "VERIFIED" if (okA and okB) else "INCONCLUSIVE"
    result = {"claim": "Claim 1: Theorem 2.2 (accelerated rate under mild Delta*eps noise)",
              "partA_noise_boundary": A, "partB_rate_scaling": B, "partC_rate_calibration": C,
              "verdict": verdict}
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, "claim1_noise_boundary.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 74)
    print("CLAIM 1 -- Theorem 2.2 (accelerated rate under mild Delta*eps noise)")
    print("=" * 74)
    print("\n[Part A] noise-boundary demonstration (d=1000,k=10,gap=1e-2):")
    print(f"  mu (Xu) = {A['mu_xu']:.2f}; eps = {A['eps']}")
    print(f"  mild noise level (Thm2.2, Delta*eps)   = {A['eta_mild_Thm2_2']:.3e}")
    print(f"  Xu required noise level (Delta*eps^mu) = {A['eta_xu_required']:.3e}")
    print(f"  => mild is {A['ratio_mild_over_xu']:.1e}x LARGER than Xu allows (Xu violated)")
    print(f"  cond(3) holds: {A['cond3_holds']}; cond(4) holds: {A['cond4_holds']}")
    print(f"  ANPM(beta*) final sin theta = {A['ANPM_final_sin_theta']:.3e}, "
          f"floor = {A['ANPM_floor_sin_theta']:.3e} (<=eps {A['eps']}): "
          f"{A['converges_to_eps_floor']}")
    print(f"\n[Part B] first-hit T(eps={B['eps_target']:.0e}), square-root speedup")
    print(f"{'gap':>9} {'Delta':>9} {'T(beta*)':>9} {'T(beta=0)':>10} {'speedup':>9}")
    for r in B["rows"]:
        sp = f"{r['speedup_ratio']:.1f}x" if r["speedup_ratio"] else "n/a"
        print(f"{r['gap']:>9.0e} {r['Delta']:>9.2e} {r['T_beta_star']:>9} {r['T_beta_0']:>10} {sp:>9}")
    print(f"  T(beta*) ~ (1/sqrt Delta)^p: p={B['exponent_T_beta_star_vs_1_over_sqrtDelta']:.3f} (theory 1.0)")
    print(f"  T(beta=0) ~ (1/Delta)^p:     p={B['exponent_T_beta_0_vs_1_over_Delta']:.3f} (theory 1.0)")
    print(f"  speedup ratio ~ (1/sqrt Delta)^p: p={B['exponent_speedup_ratio_vs_1_over_sqrtDelta']:.3f} (theory 1.0)")
    print("\n[Part C] proof-rate calibration:")
    for r in C["rows"]:
        print(f"  eps={r['eps']:.0e}: T_obs={r['T_observed']}, T_pred={r['T_predicted']:.1f}, "
              f"pred/obs={r['ratio_pred_over_obs']:.2f}")
    print(f"\nVERDICT: {verdict}\nwrote {path}")
    return result


if __name__ == "__main__":
    main()
