# ANPM (UTiEfkfNQ2) — ICML 2026 reproduction

Reproduction of *Improved Analysis of the Accelerated Noisy Power Method with
Applications to Decentralized PCA* (Aguié, Even, Massoulié; arXiv
[2602.03682](https://arxiv.org/abs/2602.03682), OpenReview
[UTiEfkfNQ2](https://openreview.net/forum?id=UTiEfkfNQ2)). Official code
[`pierreaguie/ANPM@3623010`](https://github.com/pierreaguie/ANPM), vendored at
[`repro/anpm/`](repro/anpm/) and run unmodified. **CPU-only** (numpy/scipy).

## Result — all five claims VERIFIED

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/notebooks/anpm_walkthrough.py)

| Claim | Theorem | Paper claim | Observed | Assessment |
| --- | --- | --- | --- | --- |
| 1 | Thm 2.2 | accelerated rate under mild Δ·ε noise (vs Xu's Δ·ε^μ) | mild budget is 7.7×10²⁸× larger than Xu's, yet ANPM converges to 3.4e-7 | **VERIFIED** |
| 2 | Thm 2.4/2.5 | noise conditions cannot be relaxed | explicit counterexamples never reach ε; strict c=1/32 converges | **VERIFIED** |
| 3 | Thm 3.3 | ADePM accelerated decentralized PCA | ADePM 8.5×–1.2×10⁶× lower error than DePM (real ego-Facebook); exact-diff vs ref 6.8e-12 | **VERIFIED** |
| 4 | Prop 3.2 | gossip Õ(1/√γ_W) vs Õ(1/γ_W) | scaling exponents 0.52 vs 1.02; contraction matches theory to 1e-15 | **VERIFIED** |
| 5 | §4 | β\*=λ²_{k+1}/4 √-speedup + ADePM > DeEPCA | speedup exponent 1.00; ADePM beats DeEPCA 8.2×–1.2×10⁶×; DeEPCA plateaus | **VERIFIED** |

**Downscaling/substitutions:** all experiments are CPU-scale (d≤1000 synthetic;
real ego-Facebook n=50 subgraph). The paper's Fed-Heart-Disease experiment needs
the `flamby` data download and is omitted. Claims 1–2 are theorems; finite
experiments are scoped corroboration / constructive counterexample verification.

**Compute:** local CPU, 1 core, ~4.4 min total (`orx exp run --backend local`).
No GPU used.

Full report: [`reports/anpm-repro/report.md`](reports/anpm-repro/report.md).
Live logbook: [huggingface.co/spaces/DineshAI/UTiEfkfNQ2](https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2).
Tutorial notebook: [`repro/notebooks/anpm_walkthrough.py`](repro/notebooks/anpm_walkthrough.py) (`marimo edit` / `marimo run`).

## Experiment log

| Branch / experiment | Purpose | Exact run command | Outcome | Compute |
| --- | --- | --- | --- | --- |
| `master` @ `f0afde4` — [baseline `0b464e60`](.) | 5-claim verifier suite (pinned uv env + vendored official code) | `bash repro/run.sh` | 5/5 VERIFIED (run `6758a894`, 265.8s) | local CPU, 1 core |
| `master` (presentation surface) | README, report, notebook, HF logbook | _Not run as an experiment (publication surface)_ | — | — |

`repro/run.sh` bootstraps `uv`, syncs the pinned `uv.lock`, and runs
`repro/src/run_all.py`, which prints every claim's metrics to stdout and exits
nonzero unless all reach a terminal VERIFIED/FALSIFIED verdict.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv sync
uv run python repro/src/run_all.py     # = bash repro/run.sh without the uv bootstrap
uv run --with pytest python -m pytest repro/tests/ -q
```

Pinned environment (Python 3.12, numpy 2.5.1, scipy 1.18.0, networkx 3.6.1) in
[`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock).

## Logbook
https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2
