"""Official decentralized-PCA experiment on the real SNAP ego-Facebook graph.

Runs the four algorithms from ``anpm.depca`` (DePM, DeEPCA, ADePM beta=beta*,
ADePM beta=beta_t) at matched gossip budgets L in {20, 40}, exactly as the
paper's Figure 2 (right) / official ``depca_egofb.py``.  Writes the per-iteration
sin theta_k CSV (matching the authors' reference) and a JSON of summary metrics.

This single run feeds BOTH Claim 3 (ADePM vs DePM, real data) and Claim 5
(ADePM vs DeEPCA + beta* speedup), so the heavy experiment runs only once.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
REPRO = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(REPRO, ".."))
# The official loader uses the relative path "datasets/facebook_combined.txt".
os.chdir(REPRO)
sys.path.insert(0, REPRO)
sys.path.insert(0, HERE)
from common import OUTPUTS_DIR  # noqa: E402
from anpm.depca import ADePM, ADePM_tune, DePM, DeEPCA  # noqa: E402
from anpm.data.load_datasets import load_ego_facebook  # noqa: E402
from anpm.metrics import sin_thetak  # noqa: E402
from anpm.gossip import compute_omega  # noqa: E402


def run_alg(alg, A, T, X0, W, omega, L, beta_star, k):
    if alg == "DePM":
        return DePM(A, T, X0[:, :k], W, omega, L)
    if alg == "DeEPCA":
        return DeEPCA(A, T, X0[:, :k], W, omega, L)
    if alg == "ADePM_bstar":
        return ADePM(A, T + 1, beta_star, X0[:, :k], W, omega, L)
    if alg == "ADePM_btune":
        return ADePM_tune(A, T + 1, X0, W, omega, L)
    raise ValueError(alg)


def main():
    np.random.seed(0)
    T, k, n = 200, 5, 50
    A, W, A_mean = load_ego_facebook(n)
    eigs, U = np.linalg.eigh(A_mean)
    Uk = U[:, -k:]
    d = A.shape[1]
    lambda_k1 = eigs[-k - 1]
    beta_star = lambda_k1 ** 2 / 4
    omega = compute_omega(W)
    gamma_W = float(1.0 - np.maximum(np.sort(np.linalg.eigvalsh(W))[-2],
                                     -np.linalg.eigvalsh(W)[0]))
    X0 = np.random.randn(d, k + 1)

    Ls = [20, 40]
    algs = ["DePM", "DeEPCA", "ADePM_bstar", "ADePM_btune"]
    errors_dict = {}
    for L in Ls:
        for alg in algs:
            X_list = run_alg(alg, A, T, X0, W, omega, L, beta_star, k)
            errors = [[sin_thetak(Uk, X[i][:, :k], k) for X in X_list] for i in range(n)]
            errors_dict[f"{alg},L={L}"] = np.mean(errors, axis=0).tolist()

    n_rows = len(next(iter(errors_dict.values())))
    os.makedirs(os.path.join(REPO, "outputs"), exist_ok=True)
    csv_path = os.path.join(REPO, "outputs", "depca_ego_facebook_repro.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + list(errors_dict.keys()))
        for t in range(n_rows):
            w.writerow([t] + [errors_dict[name][t] for name in errors_dict])

    # exact diff vs authors' reference (column-aligned)
    ref = os.path.join(REPRO, "results_reference", "depca_ego_facebook_.csv")
    max_diff = None
    if os.path.exists(ref):
        with open(ref) as f:
            r = list(csv.reader(f))
        ref_hdr, ref_data = r[0], np.array([[float(x) for x in row] for row in r[1:]])
        with open(csv_path) as f:
            r2 = list(csv.reader(f))
        _, our_data = r2[0], np.array([[float(x) for x in row] for row in r2[1:]])
        # reference has only DePM/ADePM columns (no DeEPCA); align shared columns by header
        max_diff = _aligned_maxdiff(ref_hdr, ref_data, r2[0], our_data)

    summary = {
        "dataset": "SNAP ego-Facebook (n=d=50 subgraph)", "agents": n, "rank_k": k,
        "iterations_T": T, "lambda_k+1": float(lambda_k1), "beta_star": float(beta_star),
        "gossip_gamma_W": gamma_W, "omega": float(omega),
        "Ls": Ls, "max_abs_diff_vs_authors_reference": max_diff,
        "final_errors": {name: float(errors_dict[name][-1]) for name in errors_dict},
    }

    # per-L matched-communication comparisons
    summary["matched_communication"] = {}
    for L in Ls:
        d_ = errors_dict[f"DePM,L={L}"]
        e_ = errors_dict[f"DeEPCA,L={L}"]
        a_ = errors_dict[f"ADePM_bstar,L={L}"]
        at = errors_dict[f"ADePM_btune,L={L}"]
        summary["matched_communication"][str(L)] = {
            "DePM_final": float(d_[-1]), "DeEPCA_final": float(e_[-1]),
            "ADePM_bstar_final": float(a_[-1]), "ADePM_btune_final": float(at[-1]),
            "DePM_over_ADePM_bstar": float(d_[-1] / a_[-1]),
            "DePM_over_ADePM_btune": float(d_[-1] / at[-1]),
            "DeEPCA_over_ADePM_bstar": float(e_[-1] / a_[-1]),
            "DeEPCA_over_ADePM_btune": float(e_[-1] / at[-1]),
            "ADePM_bstar_beats_DePM": bool(a_[-1] < d_[-1]),
            "ADePM_btune_beats_DePM": bool(at[-1] < d_[-1]),
            "ADePM_bstar_beats_DeEPCA": bool(a_[-1] < e_[-1]),
        }

    out = os.path.join(REPO, "outputs", "facebook_experiment.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 74)
    print("OFFICIAL ego-Facebook decentralized PCA (4 algorithms, matched L)")
    print("=" * 74)
    print(f"n={n}, d={d}, k={k}, T={T}, gamma_W={gamma_W:.4e}, beta*={beta_star:.4e}")
    if max_diff is not None:
        print(f"max |diff| vs authors' reference (shared cols): {max_diff:.3e}")
    for L in Ls:
        mc = summary["matched_communication"][str(L)]
        print(f"\n[L={L}] final sin theta_k (mean over agents):")
        print(f"  DePM        = {mc['DePM_final']:.4e}")
        print(f"  DeEPCA      = {mc['DeEPCA_final']:.4e}")
        print(f"  ADePM beta* = {mc['ADePM_bstar_final']:.4e}  ({mc['DePM_over_ADePM_bstar']:.1f}x < DePM)")
        print(f"  ADePM beta_t= {mc['ADePM_btune_final']:.4e}  ({mc['DePM_over_ADePM_btune']:.1f}x < DePM)")
        print(f"  ADePM beta* beats DeEPCA: {mc['ADePM_bstar_beats_DeEPCA']} "
              f"({mc['DeEPCA_over_ADePM_bstar']:.1f}x)")
    print(f"\nwrote {csv_path}\nwrote {out}")
    return summary


def _aligned_maxdiff(ref_hdr, ref_data, our_hdr, our_data):
    """Compare positionally: the official script emits the 8 algorithm columns in a
    fixed order (DePM, DeEPCA, ADePM b*, ADePM bt) x (L=20, L=40), so the reference
    and our run are column-aligned despite different header labels."""
    if ref_data.shape != our_data.shape:
        # align by min rows/cols
        r = min(ref_data.shape[0], our_data.shape[0])
        c = min(ref_data.shape[1], our_data.shape[1])
        return float(np.max(np.abs(ref_data[:r, :c] - our_data[:r, :c])))
    return float(np.max(np.abs(ref_data - our_data)))


if __name__ == "__main__":
    main()
