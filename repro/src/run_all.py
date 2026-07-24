"""Master verifier for the ANPM (UTiEfkfNQ2) reproduction.

Runs every claim check in order, prints a consolidated summary, and exits
NONZERO if any claim verifier fails (so the run log is self-certifying).

Fixed run command (inherited by every experiment node):
    bash repro/run.sh
which bootstraps uv from the pinned uv.lock and runs this module.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from common import OUTPUTS_DIR  # noqa: E402


def env_info():
    import numpy, scipy  # noqa: E402
    sha = "unknown"
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass
    return {
        "python": sys.version.split()[0], "numpy": numpy.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "git_sha": sha, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def run(label, fn):
    t0 = time.time()
    print(f"\n{'#' * 78}\n# {label}\n{'#' * 78}", flush=True)
    res = fn()
    dt = time.time() - t0
    verdict = res.get("verdict", "?") if isinstance(res, dict) else "?"
    print(f"[{label}] verdict={verdict}  ({dt:.1f}s)", flush=True)
    return res, verdict, dt


def main():
    import facebook_experiment  # noqa: E402
    import claim1_noise_boundary  # noqa: E402
    import claim2_counterexamples  # noqa: E402
    import claim4_gossip_mixing  # noqa: E402
    import claim3_adepm  # noqa: E402
    import claim5_deepca  # noqa: E402

    info = env_info()
    print("=" * 78)
    print("ANPM (UTiEfkfNQ2) reproduction -- master verifier")
    print("=" * 78)
    print(json.dumps(info, indent=2))

    timings = {}
    # heavy shared experiment first (feeds claims 3 & 5)
    fb, _, dt = run("Facebook experiment (official, 4 algorithms)", facebook_experiment.main)
    timings["facebook"] = dt
    c1, v1, dt = run("Claim 1", claim1_noise_boundary.main); timings["claim1"] = dt
    c2, v2, dt = run("Claim 2", claim2_counterexamples.main); timings["claim2"] = dt
    c4, v4, dt = run("Claim 4", claim4_gossip_mixing.main); timings["claim4"] = dt
    c3, v3, dt = run("Claim 3", claim3_adepm.main); timings["claim3"] = dt
    c5, v5, dt = run("Claim 5", claim5_deepca.main); timings["claim5"] = dt

    summary = {
        "env": info, "timings_seconds": timings,
        "verdicts": {"claim1": v1, "claim2": v2, "claim3": v3, "claim4": v4, "claim5": v5},
    }
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "run_all_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    for c, v in summary["verdicts"].items():
        print(f"  {c:10s}: {v}   ({timings[c]:.1f}s)")
    total = sum(timings.values())
    print(f"  total wall time: {total:.1f}s")
    fails = [c for c, v in summary["verdicts"].items() if v not in ("VERIFIED", "FALSIFIED")]
    if fails:
        print(f"\nNONZERO EXIT -- claims not at terminal VERIFIED/FALSIFIED: {fails}")
        sys.exit(1)
    print("\nAll claims at a terminal verdict (VERIFIED/FALSIFIED).")


if __name__ == "__main__":
    main()
