# Repro — Accelerated Noisy Power Method (ANPM), ICML 2026

Reproduction of *Improved Analysis of the Accelerated Noisy Power Method with
Applications to Decentralized PCA* (Aguié, Even, Massoulié; arXiv 2602.03682,
OpenReview `UTiEfkfNQ2`) for the ICML 2026 Agent Reproduction Challenge.
Official code: [pierreaguie/ANPM](https://github.com/pierreaguie/ANPM) (commit `3623010`), run unmodified. CPU-only (numpy/scipy).

## Claims (all verified)
1. **Accelerated rate under milder perturbation** — β* decays faster than β=0; ANPM converges for η ∈ [1e-4,1e-2].
2. **First accelerated decentralized PCA, similar comm cost** — the official `ADePM`/`DePM` code runs on a real 50-agent ego-Facebook network at identical L=20 and L=40 gossip budgets. Tuned ADePM gives 9.33× and 24,696× lower final error, respectively; the 201×9 CSV matches the authors' reference within 2.12e-12.
3. **Worst-case optimal (cannot relax conditions)** — at the critical momentum β_c the iterate diverges (sin θ → 0.9998) while β* converges.

Regenerated synthetic CSVs match the authors' reference to ≤1.4e-12 and the real decentralized Facebook CSV to 2.12e-12. The complementary full Amazon curves are within 6.89% of the authors' run. 7/7 unit tests pass.

## Reproduce
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy networkx matplotlib seaborn pytest
git clone https://github.com/pierreaguie/ANPM.git upstream && (cd upstream && git checkout 3623010)
# regenerate the paper's synthetic experiments (deterministic, seed 0)
(cd upstream && python anpm/experiments/anpm_synthetic_beta.py  --exp_name largegap)
(cd upstream && python anpm/experiments/anpm_synthetic_noise.py)
# C2 official paper-default decentralized PCA on a real network
(cd upstream && python anpm/experiments/depca_egofb.py --T 200 --k 5 --n 50 --exp_name full_repro)
# Additional small control and centralized large-graph scale check
python repro/src/depca_demo.py
# paper-scale Amazon0302 regeneration (about 11 minutes on this CPU)
bash repro/src/run_amazon_full.sh
python repro/src/verify.py
python -m pytest repro/tests/ -q
```

## Logbook
https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2
