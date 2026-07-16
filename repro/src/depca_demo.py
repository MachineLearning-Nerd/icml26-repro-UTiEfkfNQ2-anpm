#!/usr/bin/env python3
"""Claim 2 demo: accelerated decentralized PCA (ADePM) vs plain (DePM) at MATCHED
communication cost, on a small synthetic problem (so it runs in seconds on CPU,
unlike the 262k-node Amazon graph whose eigsh baseline is compute-bound).

Both methods use the same L gossip rounds per iteration, so any convergence gap
is due to the acceleration (momentum), not extra communication -- exactly the
C2 claim: "first decentralized PCA with provably accelerated convergence and
similar communication costs to non-accelerated methods."
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "upstream"))

import numpy as np
import networkx as nx
from anpm.depca import ADePM, DePM
from anpm.gossip import compute_omega
from anpm.metrics import sin_thetak
from anpm.data.synthetic_instances import generate_eigenvectors, generate_matrix, generate_X0


def build_gossip(n=12, p=0.4, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1e9)))
        if nx.is_connected(G):
            break
    M = nx.to_numpy_array(G)
    D = np.diag(M.sum(1))
    Dinv = np.linalg.pinv(np.sqrt(D))
    W = 0.5 * (np.eye(n) + Dinv @ M @ Dinv)   # lazy symmetric gossip; top eigenvalue = 1
    return W


def main():
    seed = 0
    np.random.seed(seed)
    n, d, k = 12, 60, 4
    gap = 0.08
    T, L = 150, 6

    U = generate_eigenvectors(d)
    lambdas = np.array([1.0] * k + [1.0 - gap] + [0.5] * (d - k - 1))
    A_global = generate_matrix(lambdas, U)
    A = np.stack([A_global / n for _ in range(n)])   # homogeneous split; sum_i A_i = A_global
    X0 = generate_X0(d, k)

    W = build_gossip(n=n, seed=seed)
    omega = compute_omega(W)
    # ADePM momentum must be set from the LOCAL (per-node) spectrum. With the
    # homogeneous split A_i = A_global/n, lambda_{k+1}(A_i) = (1-gap)/n.
    beta_star = (((1.0 - gap) / n) ** 2) / 4

    Uk = U[:, :k]
    def err_path(X_list):
        # X_list: (T, n, d, k); average per-node sin theta to the global top-k
        return [float(np.mean([sin_thetak(Uk, X_list[t, i, :, :k], k)
                                for i in range(n)])) for t in range(len(X_list))]

    e_adepm = err_path(ADePM(A, T, beta_star, X0, W, omega, L))
    e_depm = err_path(DePM(A, T, X0, W, omega, L))

    res = dict(n=n, d=d, k=k, gap=gap, T=T, L=L, omega=float(omega), beta_star=float(beta_star),
               adepm_final=e_adepm[-1], depm_final=e_depm[-1],
               adepm_min=min(e_adepm), depm_min=min(e_depm),
               accelerated_lower=bool(e_adepm[-1] < e_depm[-1]))
    out = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "depca_demo.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    json.dump(res, open(out, "w"), indent=2)

    print("=" * 64)
    print("C2 demo: accelerated (ADePM) vs plain (DePM) decentralized PCA")
    print(f"  graph: {n} nodes, d={d}, k={k}, eigengap={gap}, {L} gossip rounds/iter (MATCHED comm)")
    print(f"  ADePM (accelerated) final avg sin theta = {e_adepm[-1]:.4e}")
    print(f"  DePM  (plain)       final avg sin theta = {e_depm[-1]:.4e}")
    print(f"  -> accelerated reaches lower error at same communication cost: "
          f"{res['accelerated_lower']}")
    print("=" * 64)
    print("wrote", out)


if __name__ == "__main__":
    main()
