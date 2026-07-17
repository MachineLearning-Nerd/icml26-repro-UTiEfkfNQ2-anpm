#!/usr/bin/env python3
"""Write machine-readable provenance for the full Amazon0302 regeneration."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "upstream/datasets/amazon0302.txt"
RESULT = ROOT / "upstream/results/anpm_amazon_sigma1e-3_k30_every10.csv"
OUT = ROOT / "outputs/amazon_rank30_provenance.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True
    ).strip()
    record = {
        "experiment": "Amazon0302 full paper-scale ANPM regeneration",
        "command": (
            "python -u anpm/experiments/anpm_amazon.py --T 100 --k 30 "
            "--sigma 1e-3 --exp_name sigma1e-3_k30_every10 --error_every 10"
        ),
        "dataset": {
            "name": "SNAP Amazon0302",
            "nodes": 262111,
            "edges": 1234877,
            "sha256": sha256(DATASET),
        },
        "parameters": {"T": 100, "k": 30, "sigma": 1e-3, "error_every": 10},
        "result_sha256": sha256(RESULT),
        "official_code_commit": upstream_commit,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
