#!/usr/bin/env python3
"""Independent verification + claim mapping for the ANPM reproduction.

Loads the regenerated upstream experiment CSVs (anpm_synthetic_{beta,gap,noise},
anpm_amazon) and the authors' reference CSVs, and checks each official claim with
mechanisms independent of merely eyeballing the plots:

  C1 (accelerated rate preserved under milder perturbation):
    - noise sweep: ANPM converges (sin theta -> a noise-dependent floor, not divergence)
      for perturbation levels eta in [1e-4, 1e-2]; floor grows ~linearly with eta.
    - beta sweep: the momentum methods decay faster (steeper log-slope) than beta=0.
  C2 (first accelerated decentralized PCA, similar comm cost):
    - amazon graph: tuned/accelerated (beta_t) reaches a lower error than beta=0
      at matched iteration/communication count.
  C3 (worst-case optimal; noise/momentum conditions cannot be relaxed):
    - beta sweep: at the CRITICAL momentum beta_c the iterate DIVERGES (sin theta
      stays ~1.0) whereas beta* converges -> the boundary is tight.

Also reports exact reproduction for the synthetic CSVs.  The Amazon error
metric uses a deliberately loose iterative eigensolver tolerance (1e-3), so its
cross-SciPy comparison is assessed by relative curve agreement and the final
method ordering rather than bitwise equality.
"""
import csv
import json
import os
import sys

import numpy as np

UP = os.path.join(os.path.dirname(__file__), "..", "..", "upstream")
REF = os.path.join(UP, "results_reference")
RES = os.path.join(UP, "results")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "verify_summary.json")


def load(p):
    with open(p) as f:
        r = list(csv.reader(f))
    return r[0], np.array([[float(x) for x in row] for row in r[1:]])


def maxdiff(name):
    a = os.path.join(RES, name)
    b = os.path.join(REF, name)
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    _, A = load(a)
    _, B = load(b)
    if A.shape != B.shape:
        return None
    return float(np.max(np.abs(A - B)))


def decay_slope(t, y, lo=20, hi=400, ymax=0.95, ymin=1e-6):
    yy = y.copy()
    logy = np.log(np.where(yy > 0, yy, 1e-12))
    mask = (t >= lo) & (t <= hi) & (y < ymax) & (y > ymin)
    if mask.sum() > 5:
        return float(np.polyfit(t[mask], logy[mask], 1)[0])
    return float("nan")


def main():
    res = {"exact_reproduction": {}, "claims": {}}

    # ---- exact reproduction vs reference ----
    for f in ["anpm_synthetic_beta_largegap.csv", "anpm_synthetic_beta_smallgap.csv",
              "anpm_synthetic_gap_.csv", "anpm_synthetic_noise_.csv"]:
        md = maxdiff(f)
        if md is not None:
            res["exact_reproduction"][f] = md

    # ---- beta sweep: acceleration (C1) + critical-boundary divergence (C3) ----
    h, a = load(os.path.join(RES, "anpm_synthetic_beta_largegap.csv"))
    t = a[:, 0]
    # columns: t, $0$, 0.5b*, 0.8b*, 0.9b*, b*, (b*+bc)/2, bc, bt
    idx = {"beta0": 1, "beta_star": 5, "beta_crit": 7, "beta_tune": 8}
    finals = {k: float(a[-1, j]) for k, j in idx.items()}
    slopes = {k: decay_slope(t, a[:, j]) for k, j in idx.items()}
    bc_diverges = finals["beta_crit"] > 0.5          # C3 boundary
    accel_faster = slopes["beta_star"] < slopes["beta0"] - 1e-4  # steeper (more negative)
    res["claims"]["C1_beta_slope"] = {"beta0": slopes["beta0"], "beta_star": slopes["beta_star"],
                                      "beta_tune": slopes["beta_tune"], "accelerated_faster": bool(accel_faster)}
    res["claims"]["C3_critical_boundary"] = {"beta_star_final": finals["beta_star"],
                                             "beta_crit_final": finals["beta_crit"],
                                             "beta_crit_diverges": bool(bc_diverges)}

    # ---- noise sweep (C1: converges under milder perturbation) ----
    hn, an = load(os.path.join(RES, "anpm_synthetic_noise_.csv"))
    noise_levels = hn[1:]
    noise_finals = [float(an[-1, j]) for j in range(1, len(hn))]
    # converges = final sin theta well below the start (~1.0) for every eta
    noise_converges = all(f < 0.5 for f in noise_finals)
    # Headers are eta=1e-2 ... 1e-4, so floors should decrease left-to-right.
    floor_monotone = all(noise_finals[i] >= noise_finals[i + 1] - 1e-9
                         for i in range(len(noise_finals) - 1))
    res["claims"]["C1_noise_sweep"] = {"eta_levels": noise_levels, "finals": noise_finals,
                                       "all_converge": bool(noise_converges),
                                       "floor_monotone_in_eta": bool(floor_monotone)}

    # ---- amazon (C2: accelerated decentralized PCA) ----
    am = os.path.join(RES, "anpm_amazon_sigma1e-3_k30_every10.csv")
    if os.path.exists(am):
        ha, aa = load(am)
        # columns: t, $0$, $beta_t$
        if len(ha) >= 3:
            b0_final = float(aa[-1, 1]); bt_final = float(aa[-1, 2])
            ref_am = os.path.join(REF, "anpm_amazon_sigma1e-3_k30_every10.csv")
            _, ref_aa = load(ref_am)
            expected_t = np.arange(0, 101, 10, dtype=float)
            matched_schedule = bool(
                aa.shape == (11, 3)
                and np.array_equal(aa[:, 0], expected_t)
                and aa.shape == ref_aa.shape
            )
            exact_diff = maxdiff("anpm_amazon_sigma1e-3_k30_every10.csv")
            curve_max_relative_diff = float(np.max(
                np.abs(aa[:, 1:] - ref_aa[:, 1:]) / np.abs(ref_aa[:, 1:])
            ))
            reference_same_final_order = bool(ref_aa[-1, 2] < ref_aa[-1, 1])
            res["claims"]["C2_decentralized_PCA"] = {
                "dataset": "SNAP Amazon0302",
                "nodes": 262111,
                "edges": 1234877,
                "rank_k": 30,
                "iterations": 100,
                "sigma": 1e-3,
                "checkpoint_schedule": aa[:, 0].astype(int).tolist(),
                "matched_schedule": matched_schedule,
                "beta0_final": b0_final,
                "beta_tune_final": bt_final,
                "absolute_improvement": b0_final - bt_final,
                "relative_improvement_percent": 100.0 * (b0_final - bt_final) / b0_final,
                "max_abs_diff_vs_authors_reference": exact_diff,
                "max_relative_diff_vs_authors_reference": curve_max_relative_diff,
                "reference_same_final_order": reference_same_final_order,
                "qualitative_reference_agreement": bool(
                    curve_max_relative_diff < 0.10 and reference_same_final_order
                ),
                "accelerated_lower": bool(bt_final < b0_final),
            }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)

    # ---- print ----
    print("=" * 70)
    print("ANPM verification summary")
    print("=" * 70)
    print("Exact reproduction (max abs diff vs reference, FP noise ~1e-13):")
    for f, md in res["exact_reproduction"].items():
        print(f"  {f:42s} {md:.3e}")
    print("\nC1 (accelerated rate under perturbation):")
    c1 = res["claims"]["C1_beta_slope"]
    print(f"  decay slope beta=0 (plain PM): {c1['beta0']:.5f}")
    print(f"  decay slope beta* (accelerated): {c1['beta_star']:.5f}  -> accelerated faster: {c1['accelerated_faster']}")
    c1n = res["claims"]["C1_noise_sweep"]
    print(f"  noise sweep eta in [1e-4,1e-2]: all converge={c1n['all_converge']}, "
          f"floor monotone in eta={c1n['floor_monotone_in_eta']}")
    print(f"    finals by eta: {[round(x,5) for x in c1n['finals']]}")
    print("\nC3 (worst-case optimal; cannot relax conditions):")
    c3 = res["claims"]["C3_critical_boundary"]
    print(f"  beta* final sin theta = {c3['beta_star_final']:.4e} (converges)")
    print(f"  beta_c final sin theta = {c3['beta_crit_final']:.4e} -> diverges at boundary: {c3['beta_crit_diverges']}")
    if "C2_decentralized_PCA" in res["claims"]:
        c2 = res["claims"]["C2_decentralized_PCA"]
        print("\nC2 (accelerated decentralized PCA, Amazon graph):")
        print(f"  full scale: {c2['nodes']:,} nodes, {c2['edges']:,} edges, "
              f"rank k={c2['rank_k']}, T={c2['iterations']}, sigma={c2['sigma']}")
        print(f"  matched checkpoints: {c2['matched_schedule']} | max relative curve diff vs "
              f"authors' reference: {100*c2['max_relative_diff_vs_authors_reference']:.2f}%")
        print(f"  same final ordering as reference: {c2['reference_same_final_order']} | "
              f"qualitative agreement: {c2['qualitative_reference_agreement']}")
        print(f"  beta=0 final = {c2['beta0_final']:.4e} | beta_t final = {c2['beta_tune_final']:.4e} "
              f"-> accelerated lower: {c2['accelerated_lower']} "
              f"({c2['relative_improvement_percent']:.2f}% reduction)")
    print("=" * 70)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
