"""Generate the figures for the ANPM reproduction report.

Reads the verifier output JSON/CSVs and writes PNGs into reports/anpm-repro/images/.
All figures are deterministic (seeded runs) and regenerate from the fixed command.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "outputs")
IMG = os.path.join(REPO, "reports", "anpm-repro", "images")
os.makedirs(IMG, exist_ok=True)


def load_csv(p):
    with open(p) as f:
        r = list(csv.reader(f))
    return r[0], np.array([[float(x) for x in row] for row in r[1:]])


def fig_claim1():
    c1 = json.load(open(os.path.join(OUT, "claim1_noise_boundary.json")))
    B = c1["partB_rate_scaling"]
    rows = B["rows"]
    gaps = np.array([r["gap"] for r in rows])
    Ta = np.array([r["T_beta_star"] for r in rows], float)
    Tp = np.array([r["T_beta_0"] for r in rows], float)
    Delta = np.array([(1 - (1 - g) ** 2 / 4 - 2 * np.sqrt((1 - g) ** 2 / 4) * 0) * 0 + (1 - (1 - g)) for g in gaps])
    # use stored Delta
    Delta = np.array([r["Delta"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: noise boundary bar
    A = c1["partA_noise_boundary"]
    ax1.bar(["Mild (Thm 2.2)\n$\Delta\\cdot\\varepsilon$",
             "Xu (2023) req.\n$\Delta\\cdot\\varepsilon^\\mu$"],
            [A["eta_mild_Thm2_2"], max(A["eta_xu_required"], 1e-300)],
            color=["#2ca02c", "#d62728"], log=True)
    ax1.set_title(f"Noise level allowed (gap=1e-2, $\\mu$={A['mu_xu']:.1f})\n"
                  f"ANPM converges to {A['ANPM_floor_sin_theta']:.1e} under the MILD condition")
    ax1.set_ylabel("noise spectral-norm level $\\eta$")
    ax1.annotate(f"{A['ratio_mild_over_xu']:.0e}x\nlarger", xy=(0.5, 0.6), xycoords="axes fraction",
                 ha="center", fontsize=11, fontweight="bold", color="#444")
    # right: speedup scaling
    ax2.loglog(1 / np.sqrt(Delta), Tp / Ta, "o-", color="#1f77b4", label="observed $T(\\beta{=}0)/T(\\beta^*)$")
    xx = 1 / np.sqrt(Delta)
    ax2.loglog(xx, xx * (Tp / Ta)[0] / xx[0], "k--", alpha=0.5, label="$\\propto 1/\\sqrt{\\Delta}$ (theory)")
    ax2.set_xlabel("$1/\\sqrt{\\Delta_k}$")
    ax2.set_ylabel("iteration speedup ratio")
    ax2.set_title(f"Square-root speedup of $\\beta^*=\\lambda_{{k+1}}^2/4$\n"
                  f"fit exponent {B['exponent_speedup_ratio_vs_1_over_sqrtDelta']:.2f} (theory 1.0)")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "claim1_noise_and_speedup.png"), dpi=130)
    plt.close(fig)
    print("wrote claim1_noise_and_speedup.png")


def fig_claim2():
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from common import anpm_manual, tan_thetak
    d, k, lam_k, beta, eps = 8, 2, 1.0, 0.16, 1e-2
    sb = 2 * np.sqrt(beta); gap = lam_k - sb
    e = np.eye(d); Uk, Um = e[:, :k], e[:, k:]
    th0 = np.arctan(2 * eps)
    X0 = np.zeros((d, k)); X0[:, :k - 1] = e[:, :k - 1]
    X0[:, k - 1] = np.cos(th0) * e[:, k - 1] + np.sin(th0) * e[:, k]
    cos0 = float(np.cos(th0))
    T = 60
    # Thm 2.4
    Xi4 = np.zeros((T, d, k)); Xi4[:, k, k - 1] = 8 * gap * eps
    h4 = [tan_thetak(Uk, Um, Xt) for Xt in anpm_manual(lam_k * np.eye(d) * 0 + np.diag([lam_k] * k + [sb] * (d - k)), beta, T, X0, Xi4)]
    # Thm 2.5
    Xi5 = np.zeros((T, d, k)); Xi5[0, k - 1, k - 1] = -0.5 * gap * cos0; Xi5[1:, k - 1, k - 1] = -gap * cos0
    A = np.diag([lam_k] * k + [sb] * (d - k))
    h5 = [tan_thetak(Uk, Um, Xt) for Xt in anpm_manual(A, beta, T, X0, Xi5)]
    # strict contrast
    Xis = np.zeros((T, d, k)); Xis[:, k, k - 1] = (1 / 32) * gap * eps
    hs = [tan_thetak(Uk, Um, Xt) for Xt in anpm_manual(A, beta, T, X0, Xis)]
    t = np.arange(len(h4))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.semilogy(t, h4, "-", color="#d62728", lw=2, label="Thm 2.4 counterex. (cond. 3 relaxed x256)")
    ax.semilogy(t, h5, "-", color="#9467bd", lw=2, label="Thm 2.5 counterex. (cond. 4 relaxed x32)")
    ax.semilogy(t, hs, "-", color="#2ca02c", lw=2, label="strict $c=1/32$ (converges)")
    ax.axhline(eps, color="k", ls="--", alpha=0.6, label=f"target $\\varepsilon$={eps}")
    ax.set_xlabel("iteration $t$"); ax.set_ylabel("$\\tan\\,\\theta_k(\\mathbf{U}_k,\\mathbf{X}_t)$")
    ax.set_title("Counterexamples: relaxed noise conditions break convergence\n"
                 "(the conditions cannot be relaxed without losing the guarantee)")
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "claim2_counterexamples.png"), dpi=130)
    plt.close(fig); print("wrote claim2_counterexamples.png")


def fig_claim4():
    c4 = json.load(open(os.path.join(OUT, "claim4_gossip_mixing.json")))
    rows = c4["rows"]
    ginv = np.array([r["1_over_gamma"] for r in rows])
    Ls = np.array([r["L_star_std"] for r in rows], float)
    La = np.array([r["L_star_acc"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.loglog(ginv, Ls, "o-", color="#d62728", lw=2, label=f"standard gossip ($\\omega$=0), slope {c4['scaling_fit']['L_star_vs_1_over_gamma_std_exponent']:.2f}")
    ax.loglog(ginv, La, "s-", color="#1f77b4", lw=2, label=f"accelerated gossip (Liu--Morse), slope {c4['scaling_fit']['L_star_vs_1_over_gamma_acc_exponent']:.2f}")
    xx = ginv
    ax.loglog(xx, xx * Ls[0] / xx[0], "k--", alpha=0.4, label="$\\propto 1/\\gamma_W$ (theory)")
    ax.loglog(xx, np.sqrt(xx) * La[0] / np.sqrt(xx[0]), "k:", alpha=0.4, label="$\\propto 1/\\sqrt{\\gamma_W}$ (theory)")
    ax.set_xlabel("$1/\\gamma_W$"); ax.set_ylabel("gossip rounds $L^*$ to mix to $10^{-6}$")
    ax.set_title("Isolated gossip mixing: $\\widetilde{O}(1/\\sqrt{\\gamma_W})$ vs $\\widetilde{O}(1/\\gamma_W)$")
    ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "claim4_gossip_mixing.png"), dpi=130)
    plt.close(fig); print("wrote claim4_gossip_mixing.png")


def fig_facebook():
    hdr, data = load_csv(os.path.join(OUT, "depca_ego_facebook_repro.csv"))
    t = data[:, 0]
    cols = {h: i for i, h in enumerate(hdr)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, L, color in [(axes[0], 20, None), (axes[1], 40, None)]:
        for name, c, ls in [("DePM", "DePM", "-"), ("DeEPCA", "DeEPCA", "--"),
                            ("ADePM $\\beta{=}\\beta^*$", "ADePM_bstar", "-"),
                            ("ADePM $\\beta{=}\\beta_t$", "ADePM_btune", ":")]:
            col = f"{c},L={L}"
            if col in cols:
                ax.semilogy(t, np.clip(data[:, cols[col]], 1e-12, 1), ls, lw=2, label=name)
        ax.set_title(f"ego-Facebook, matched $L$={L} gossip rounds/iter")
        ax.set_xlabel("outer iteration $t$"); ax.set_ylabel("$\\sin\\theta_k$ (mean over agents)")
        ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("ADePM vs DePM / DeEPCA on real SNAP ego-Facebook (50 agents, k=5)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "facebook_adepm_deepca.png"), dpi=130, bbox_inches="tight")
    plt.close(fig); print("wrote facebook_adepm_deepca.png")


if __name__ == "__main__":
    fig_claim1(); fig_claim2(); fig_claim4(); fig_facebook()
    print("all figures ->", IMG)
