#!/usr/bin/env python3
"""Build provenance and a plot for the paper-default decentralized PCA run."""

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "upstream/results/depca_ego_facebook_full_repro.csv"
REFERENCE = ROOT / "upstream/results_reference/depca_ego_facebook_.csv"
DATASET = ROOT / "upstream/datasets/facebook_combined.txt"
JSON_OUT = ROOT / "outputs/facebook_depca_full_provenance.json"
FIG_OUT = ROOT / "outputs/facebook_depca_full.png"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    with path.open() as fh:
        rows = list(csv.reader(fh))
    return rows[0], np.asarray(rows[1:], dtype=float)


def main():
    header, values = load(RESULT)
    ref_header, reference = load(REFERENCE)
    assert header == ref_header and values.shape == reference.shape == (201, 9)

    runs = {}
    for rounds, depm, beta_star, beta_tuned in [(20, 1, 3, 4), (40, 5, 7, 8)]:
        runs[str(rounds)] = {
            "gossip_rounds_per_iteration": rounds,
            "depm_final": float(values[-1, depm]),
            "adepm_beta_star_final": float(values[-1, beta_star]),
            "adepm_beta_tuned_final": float(values[-1, beta_tuned]),
            "depm_to_tuned_error_ratio": float(values[-1, depm] / values[-1, beta_tuned]),
        }

    commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True
    ).strip()
    record = {
        "experiment": "official paper-default decentralized PCA on ego-Facebook",
        "command": (
            "python -u anpm/experiments/depca_egofb.py --T 200 --k 5 "
            "--n 50 --exp_name full_repro"
        ),
        "official_script": "anpm/experiments/depca_egofb.py",
        "official_algorithms": "anpm/depca.py::{ADePM,DePM}",
        "official_code_commit": commit,
        "dataset": {
            "name": "SNAP ego-Facebook combined graph",
            "agents": 50,
            "local_matrix_dimension": 50,
            "source_file_sha256": sha256(DATASET),
        },
        "parameters": {"T": 200, "k": 5, "n": 50, "gossip_rounds": [20, 40]},
        "matched_communication": (
            "Within each L, ADePM and DePM call the identical AcceleratedGossip "
            "routine exactly L times per outer iteration."
        ),
        "result_shape": list(values.shape),
        "result_sha256": sha256(RESULT),
        "max_abs_diff_vs_authors_reference": float(np.max(np.abs(values - reference))),
        "runs": runs,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    JSON_OUT.write_text(json.dumps(record, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, (rounds, depm, beta_star, beta_tuned) in zip(
        axes, [(20, 1, 3, 4), (40, 5, 7, 8)]
    ):
        ax.semilogy(values[:, 0], values[:, depm], label="DePM", linewidth=2)
        ax.semilogy(values[:, 0], values[:, beta_star], label=r"ADePM $\beta^*$", linewidth=2)
        ax.semilogy(values[:, 0], values[:, beta_tuned], label=r"ADePM $\beta_t$", linewidth=2)
        ax.set(
            title=f"Matched communication: L={rounds}",
            xlabel="Outer iteration",
            ylabel="Mean subspace error",
        )
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Real ego-Facebook decentralized PCA: 50 agents, k=5")
    fig.savefig(FIG_OUT, dpi=180)
    print(json.dumps(record, indent=2))
    print(f"wrote {JSON_OUT}")
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
