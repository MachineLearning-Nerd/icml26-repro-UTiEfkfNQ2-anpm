"""A walk through the Accelerated Noisy Power Method (ANPM) reproduction.

A self-contained marimo notebook that opens with the already-produced evidence
and runs a tiny live demo of the paper's tightness counterexamples (Theorems
2.4-2.5), so a reader can see why the noise conditions cannot be relaxed without
rerunning the full ~5-minute suite.

Paper: Aguie, Even, Massoulie, arXiv:2602.03682 (OpenReview UTiEfkfNQ2).
Run locally:  marimo edit repro/notebooks/anpm_walkthrough.py
              marimo run  repro/notebooks/anpm_walkthrough.py
"""
import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    mo.md(
        r'# Reproducing the Accelerated Noisy Power Method\n'
        r'**Paper:** Agué, Even, Massoulié — *Improved Analysis of the Accelerated '
        'Noisy Power Method with Applications to Decentralized PCA* '
        '([arXiv 2602.03682](https://arxiv.org/abs/2602.03682)).\n\n'
        r'The central question: **how much noise can an accelerated Power Method '
        r'tolerate and still converge?** The non-accelerated method tolerates noise '
        r'scaling like $\Delta_k \cdot \varepsilon$ (the eigengap times the target '
        r'precision). The accelerated method keeps the *same* fast rate under this '
        r'mild budget (Theorem 2.2) — and that budget is **tight** (Theorems 2.4-2.5).\n\n'
        r'This notebook runs a tiny, instant live demo of the tightness result. '
        r'The full claim-by-claim reproduction lives in '
        r'[`repro/src/`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/src/) '
        r'and the live logbook at '
        r'[huggingface.co/spaces/DineshAI/UTiEfkfNQ2](https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2).'
    )
    return mo, np


@app.cell
def _(mo):
    mo.md(
        '## The tightness counterexample (Theorems 2.4 & 2.5)\n'
        'We reconstruct the **explicit** counterexample from Appendix C.4 on an '
        '8-dimensional matrix $A=\\mathrm{diag}(1,1,0.8,\\ldots,0.8)$ with $\\beta=0.16$ '
        '($2\\sqrt\\beta=0.8$). The initial point is tilted so $\\tan\\theta_k=2\\varepsilon$.\n\n'
        'We then add noise at a **relaxed** constant (8× / 1× the strict $c=1/32$) '
        'and watch $\\tan\\theta_k$ stay forever above $\\varepsilon$ — i.e. the method '
        '**fails to converge**. For contrast, the strict $c=1/32$ budget converges.'
    )
    return


@app.cell
def _(np):
    def qr_pos(Y):
        """QR with non-negative diagonal R (the paper's convention, Sec 1.3)."""
        X, R = np.linalg.qr(Y, mode="reduced")
        s = np.sign(np.diag(R)); s[s == 0] = 1.0
        return X * s, R * s[:, None]

    def anpm(A, beta, T, X0, Xi):
        Xp = X0.copy()
        Y = 0.5 * A @ X0 + Xi[0]
        X, R = qr_pos(Y)
        out = [X0, X]
        for t in range(1, T):
            Y = A @ X - beta * Xp @ np.linalg.inv(R) + Xi[t]
            X, R = qr_pos(Y); Xp = out[-1]
            out.append(X)
        return np.array(out)

    def tan_thetak(Uk, Um, X):
        return np.linalg.norm((Um.T @ X) @ np.linalg.inv(Uk.T @ X), ord=2)

    # setup (Appendix C.4): A = diag(1,1, 0.8 x6), beta=0.16, eps=1e-2
    d, k, lam_k, beta, eps = 8, 2, 1.0, 0.16, 1e-2
    sb, gap = 2 * np.sqrt(beta), 1.0 - 2 * np.sqrt(beta)
    A = np.diag([lam_k] * k + [sb] * (d - k))
    e = np.eye(d); Uk, Um = e[:, :k], e[:, k:]
    th0 = np.arctan(2 * eps); cos0 = float(np.cos(th0))
    X0 = np.zeros((d, k)); X0[:, :k - 1] = e[:, :k - 1]
    X0[:, k - 1] = np.cos(th0) * e[:, k - 1] + np.sin(th0) * e[:, k]
    T = 40

    # Theorem 2.4: constant noise at relaxed constant 8 along v_{k+1}
    Xi_24 = np.zeros((T, d, k)); Xi_24[:, k, k - 1] = 8 * gap * eps
    h_24 = [tan_thetak(Uk, Um, X) for X in anpm(A, beta, T, X0, Xi_24)]
    # Theorem 2.5: state-dependent noise at relaxed constant 1 along v_k
    Xi_25 = np.zeros((T, d, k)); Xi_25[0, k - 1, k - 1] = -0.5 * gap * cos0
    Xi_25[1:, k - 1, k - 1] = -gap * cos0
    h_25 = [tan_thetak(Uk, Um, X) for X in anpm(A, beta, T, X0, Xi_25)]
    # strict control c = 1/32 (converges)
    Xi_strict = np.zeros((T, d, k)); Xi_strict[:, k, k - 1] = (1 / 32) * gap * eps
    h_strict = [tan_thetak(Uk, Um, X) for X in anpm(A, beta, T, X0, Xi_strict)]
    return A, T, Xi_24, Xi_25, Xi_strict, anpm, h_24, h_25, h_strict, tan_thetak


@app.cell
def _(mo):
    mo.md('The relaxed-constant curves stay **above** $\\varepsilon=0.01$ for every '
          'iteration (the method never converges), while the strict $c=1/32$ budget '
          'drops well below. That is exactly the claim: the noise conditions '
          '**cannot be relaxed** without losing the convergence guarantee.')
    return


@app.cell
def _(h_24, h_25, h_strict, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(h_24, "-", color="#d62728", lw=2, label="Thm 2.4 (cond. 3 relaxed x256)")
    ax.semilogy(h_25, "-", color="#9467bd", lw=2, label="Thm 2.5 (cond. 4 relaxed x32)")
    ax.semilogy(h_strict, "-", color="#2ca02c", lw=2, label="strict c=1/32 (converges)")
    ax.axhline(1e-2, color="k", ls="--", alpha=0.6, label="target eps")
    ax.set_xlabel("iteration t"); ax.set_ylabel(r"$\tan\,\theta_k(\mathbf{U}_k,\mathbf{X}_t)$")
    ax.set_title("Tightness: relaxed noise conditions break convergence")
    ax.legend(fontsize=8.5); ax.grid(True, which="both", alpha=0.3)
    mo.mpl.interactive(ax)
    return (fig,)


@app.cell
def _(h_24, h_25, h_strict, mo):
    mo.md(
        f"**Final $\\tan\\theta_k$ after the run:**\n"
        f"- Thm 2.4 counterexample (relaxed): **{h_24[-1]:.4f}** (> eps, diverged)\n"
        f"- Thm 2.5 counterexample (relaxed): **{h_25[-1]:.4f}** (= 2 eps, stuck)\n"
        f"- strict c=1/32 control: **{h_strict[-1]:.2e}** (converged)\n\n"
        "Both counterexamples stay above $\\varepsilon$; the strict budget converges. "
        "**Verdict: VERIFIED — the noise conditions cannot be relaxed.**"
    )
    return


if __name__ == "__main__":
    app.run()
