# Repro — Accelerated Noisy Power Method (ANPM), ICML 2026

Reproduction of *Improved Analysis of the Accelerated Noisy Power Method with
Applications to Decentralized PCA* (Aguié, Even, Massoulié; arXiv 2602.03682,
OpenReview `UTiEfkfNQ2`) for the ICML 2026 Agent Reproduction Challenge.
Official code: [pierreaguie/ANPM](https://github.com/pierreaguie/ANPM) (commit `3623010`), run unmodified. CPU-only (numpy/scipy).

## Claims (all verified)
1. **Accelerated rate under milder perturbation** — β* decays faster than β=0; ANPM converges for η ∈ [1e-4,1e-2].
2. **First accelerated decentralized PCA, similar comm cost** — ADePM reaches sin θ 1.5e-15 vs DePM 3.3e-6 at the same gossip-round budget.
3. **Worst-case optimal (cannot relax conditions)** — at the critical momentum β_c the iterate diverges (sin θ → 0.9998) while β* converges.

Regenerated experiment CSVs match the authors' reference to ≤2e-13. 3/3 unit tests pass.

## Reproduce
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy networkx matplotlib seaborn pytest
git clone https://github.com/pierreaguie/ANPM.git upstream && (cd upstream && git checkout 3623010)
# regenerate the paper's synthetic experiments (deterministic, seed 0)
(cd upstream && python anpm/experiments/anpm_synthetic_beta.py  --exp_name largegap)
(cd upstream && python anpm/experiments/anpm_synthetic_noise.py)
# C2 decentralized-PCA demo + full verification summary
python repro/src/depca_demo.py
python repro/src/verify.py
python -m pytest repro/tests/ -q
```

## Logbook
https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2
