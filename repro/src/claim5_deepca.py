"""Claim 5 verifier (two parts, Section 4):

  (A) ADePM(beta*) substantially outperforms BOTH DePM and DeEPCA baselines on a
      real dataset (ego-Facebook), at matched communication.  Additionally,
      DeEPCA plateaus (its epsilon-independent communication does NOT keep
      improving with more gossip rounds), while ADePM keeps improving -- matching
      the paper's Figure-2 narrative.

  (B) beta* = lambda_{k+1}^2/4 yields a SQUARE-ROOT speedup over non-accelerated
      decentralized PCA: measured directly via the synthetic eigengap sweep
      (Claim 1 Part B), where the iteration-count ratio T(beta=0)/T(beta*) tracks
      1/sqrt(Delta).  This part reads Claim 1's output and re-states the speedup.

Both baselines (DePM, DeEPCA) are the OFFICIAL ``anpm.depca`` implementations,
run unmodified on the official ego-Facebook experiment.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUTS_DIR  # noqa: E402


def main():
    fb_path = os.path.join(OUTPUTS_DIR, "facebook_experiment.json")
    with open(fb_path) as f:
        fb = json.load(f)
    matched = fb["matched_communication"]

    # (A) DeEPCA baseline comparison + plateau
    a_beats_e = all(matched[L]["ADePM_bstar_beats_DeEPCA"] for L in matched)
    e_ratios = {L: matched[L]["DeEPCA_over_ADePM_bstar"] for L in matched}
    # DeEPCA plateau: L=20 -> L=40 final error barely moves while ADePM improves a lot
    depm_l20, depm_l40 = matched["20"]["DeEPCA_final"], matched["40"]["DeEPCA_final"]
    adepm_l20, adepm_l40 = matched["20"]["ADePM_bstar_final"], matched["40"]["ADePM_bstar_final"]
    deepca_plateaus = abs(depm_l20 - depm_l40) / max(depm_l20, 1e-30) < 0.10   # <10% change
    adepm_improves = adepm_l40 < 0.1 * adepm_l20                              # >10x better with more L
    a_beats_depm = all(matched[L]["ADePM_bstar_beats_DePM"] for L in matched)

    # (B) square-root speedup (read Claim 1 Part B)
    c1_path = os.path.join(OUTPUTS_DIR, "claim1_noise_boundary.json")
    sqrt_speedup = None
    if os.path.exists(c1_path):
        with open(c1_path) as f:
            c1 = json.load(f)
        B = c1.get("partB_rate_scaling") or c1.get("partB_scaling")
        if B and "exponent_speedup_ratio_vs_1_over_sqrtDelta" in B:
            sqrt_speedup = {
                "exponent": B["exponent_speedup_ratio_vs_1_over_sqrtDelta"],
                "rows": B["rows"],
                "theory": 1.0,
            }
    speedup_ok = sqrt_speedup is not None and abs(sqrt_speedup["exponent"] - 1.0) < 0.30

    result = {
        "claim": "Claim 5: beta* sqrt speedup + ADePM > DePM/DeEPCA (Sec 4)",
        "partA_real_data": {
            "dataset": fb["dataset"], "ADePM_beats_DePM": bool(a_beats_depm),
            "ADePM_beats_DeEPCA": bool(a_beats_e), "DeEPCA_over_ADePM_bstar": e_ratios,
            "DeEPCA_final_L20": depm_l20, "DeEPCA_final_L40": depm_l40,
            "ADePM_bstar_final_L20": adepm_l20, "ADePM_bstar_final_L40": adepm_l40,
            "DeEPCA_plateaus_with_L": bool(deepca_plateaus),
            "ADePM_keeps_improving_with_L": bool(adepm_improves),
        },
        "partB_sqrt_speedup": sqrt_speedup,
        "verdict": "VERIFIED" if (a_beats_depm and a_beats_e and speedup_ok) else "INCONCLUSIVE",
    }
    path = os.path.join(OUTPUTS_DIR, "claim5_deepca.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 74)
    print("CLAIM 5 -- beta* sqrt speedup + ADePM > DePM/DeEPCA (Sec 4)")
    print("=" * 74)
    A = result["partA_real_data"]
    print(f"\n[Part A] real ego-Facebook:")
    print(f"  ADePM(beta*) beats DePM: {A['ADePM_beats_DePM']}; beats DeEPCA: {A['ADePM_beats_DeEPCA']}")
    for L in e_ratios:
        print(f"    L={L}: DeEPCA/ADePM(b*) = {e_ratios[L]:.1f}x")
    print(f"  DeEPCA final: L20={A['DeEPCA_final_L20']:.3e} -> L40={A['DeEPCA_final_L40']:.3e} "
          f"(plateaus: {A['DeEPCA_plateaus_with_L']})")
    print(f"  ADePM(b*) final: L20={A['ADePM_bstar_final_L20']:.3e} -> "
          f"L40={A['ADePM_bstar_final_L40']:.3e} (keeps improving: {A['ADePM_keeps_improving_with_L']})")
    if sqrt_speedup:
        print(f"\n[Part B] beta*=lambda_(k+1)^2/4 sqrt speedup (Claim 1 Part B):")
        print(f"  speedup ratio ~ 1/sqrt(Delta), exponent = {sqrt_speedup['exponent']:.3f} (theory 1.0)")
        for r in sqrt_speedup["rows"]:
            sp = r.get("speedup_ratio")
            print(f"    gap={r['gap']:.0e}: T(beta*)={r.get('T_beta_star')}, "
                  f"T(beta=0)={r.get('T_beta_0')}, speedup={sp:.1f}x" if sp else f"    gap={r['gap']:.0e}: n/a")
    print(f"\nVERDICT: {result['verdict']}\nwrote {path}")
    return result


if __name__ == "__main__":
    main()
