#!/usr/bin/env python3
"""Plot the independently regenerated paper-scale Amazon rank-30 experiment."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "upstream/results/anpm_amazon_sigma1e-3_k30_every10.csv"
OUT = ROOT / "outputs/amazon_rank30_full.png"


def main():
    with CSV.open() as fh:
        rows = list(csv.reader(fh))
    values = np.asarray(rows[1:], dtype=float)
    t, plain, tuned = values.T
    reduction = 100.0 * (plain[-1] - tuned[-1]) / plain[-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(t, plain, "o-", linewidth=2, label=r"Plain power method ($\beta=0$)")
    ax.plot(t, tuned, "s-", linewidth=2, label=r"Tuned ANPM ($\beta_t$)")
    ax.set(
        xlabel="Iteration / matched matrix-vector-product budget",
        ylabel="Excess spectral approximation error",
        title="Amazon0302: full paper-scale rank-30 regeneration",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    ax.annotate(
        f"{reduction:.1f}% lower at T=100",
        xy=(t[-1], tuned[-1]),
        xytext=(-130, -5),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    print(f"final beta=0: {plain[-1]:.12g}")
    print(f"final beta_t: {tuned[-1]:.12g}")
    print(f"relative reduction: {reduction:.6f}%")


if __name__ == "__main__":
    main()
