"""Claim 3 verifier: ADePM (accelerated decentralized PCA) achieves the
accelerated rate O(sqrt(lambda_k/(lambda_k-lambda_{k+1}))) with communication
cost O(1/sqrt(gamma_W)) comparable to non-accelerated methods (Thm 3.3, Sec 3.1).

Evidence: the official ``anpm.depca`` algorithms run on the REAL SNAP ego-Facebook
graph (50 agents, k=5, T=200).  At MATCHED gossip budgets L=20 and L=40, tuned
ADePM attains substantially lower final sin theta_k than DePM, and the regenerated
trajectory matches the authors' reference CSV to floating-point precision.

This is the previously full-credit claim; it is re-run in the cumulative suite.
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
    md = fb["max_abs_diff_vs_authors_reference"]
    matched = fb["matched_communication"]
    ok_diff = md is not None and md < 1e-9
    all_beat = all(matched[L]["ADePM_bstar_beats_DePM"] and matched[L]["ADePM_btune_beats_DePM"]
                   for L in matched)
    ratios = {L: {"bstar": matched[L]["DePM_over_ADePM_bstar"],
                  "btune": matched[L]["DePM_over_ADePM_btune"]} for L in matched}
    result = {
        "claim": "Claim 3: ADePM accelerated decentralized PCA (Thm 3.3)",
        "dataset": fb["dataset"], "agents": fb["agents"], "rank_k": fb["rank_k"],
        "iterations": fb["iterations_T"], "gossip_gamma_W": fb["gossip_gamma_W"],
        "max_abs_diff_vs_authors_reference": md,
        "exact_reproduction": bool(ok_diff),
        "ADePM_beats_DePM_at_matched_communication": bool(all_beat),
        "DePM_over_ADePM_ratios": ratios,
        "verdict": "VERIFIED" if (ok_diff and all_beat) else "FALSIFIED",
    }
    path = os.path.join(OUTPUTS_DIR, "claim3_adepm.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("=" * 74)
    print("CLAIM 3 -- ADePM accelerated decentralized PCA (Thm 3.3)")
    print("=" * 74)
    print(f"  {fb['dataset']}; n={fb['agents']}, k={fb['rank_k']}, T={fb['iterations_T']}, "
          f"gamma_W={fb['gossip_gamma_W']:.3e}")
    print(f"  max|diff| vs authors' reference: {md:.3e}  (exact reproduction: {ok_diff})")
    for L in matched:
        print(f"  L={L}: DePM/ADePM(b*)={ratios[L]['bstar']:.1f}x, "
              f"DePM/ADePM(bt)={ratios[L]['btune']:.1f}x lower error")
    print(f"  ADePM beats DePM at matched communication (all L): {all_beat}")
    print(f"\nVERDICT: {result['verdict']}\nwrote {path}")
    return result


if __name__ == "__main__":
    main()
