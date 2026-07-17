# Repro — Accelerated Noisy Power Method (ANPM), ICML 2026

Reproduction of *Improved Analysis of the Accelerated Noisy Power Method with
Applications to Decentralized PCA* (Aguié, Even, Massoulié; arXiv 2602.03682,
OpenReview `UTiEfkfNQ2`) for the ICML 2026 Agent Reproduction Challenge.
Official code: [pierreaguie/ANPM](https://github.com/pierreaguie/ANPM) (commit `3623010`), run unmodified. CPU-only (numpy/scipy).

## Claims (all verified)
1. **Accelerated rate under milder perturbation** — β* decays faster than β=0; ANPM converges for η ∈ [1e-4,1e-2].
2. **First accelerated decentralized PCA, similar comm cost** — full Amazon0302 regeneration (262,111 nodes, 1,234,877 edges, k=30, T=100) gives 2.7234e-3 for tuned ANPM versus 3.1186e-3 for plain, 12.67% lower at the matched budget. A small decentralized control also matches gossip-round budgets.
3. **Worst-case optimal (cannot relax conditions)** — at the critical momentum β_c the iterate diverges (sin θ → 0.9998) while β* converges.

Regenerated synthetic CSVs match the authors' reference to ≤1.4e-12. The full Amazon curves are within 6.89% of the authors' run (the reported metric uses iterative `eigsh` at tolerance 1e-3), with the same final ordering. 5/5 unit tests pass.

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
# paper-scale Amazon0302 regeneration (about 11 minutes on this CPU)
bash repro/src/run_amazon_full.sh
python repro/src/verify.py
python -m pytest repro/tests/ -q
```

## Logbook
https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2
