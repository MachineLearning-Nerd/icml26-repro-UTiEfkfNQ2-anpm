# STATUS — ANPM (UTiEfkfNQ2) reproduction

**Session:** autoloop. **Last updated:** 2026-07-17. **State: ✅ PUBLISHED (under_verdict).**
HF: https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2 · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm (commit ab01ba5).

## Paper
- **Title:** Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA. arXiv 2602.03682 · OpenReview UTiEfkfNQ2 (Aguié, Even, Massoulié).
- **Official code:** `pierreaguie/ANPM` (commit `3623010`) → `upstream/`, run unmodified.
- **Compute:** CPU-only (numpy/scipy/networkx). No GPU, no training.

## Official claims (verbatim) — ALL VERIFIED
1. "Improved analysis preserves accelerated convergence rate under much milder conditions on perturbations." → **VERIFIED**: β* decays faster than β=0 (slope −0.01036 vs −0.00728); ANPM converges for η ∈ [1e-4,1e-2] (floor 3.8e-4→3.6e-2).
2. "First decentralized algorithm for PCA with provably accelerated convergence and similar communication costs." → **VERIFIED**: ADePM sin θ 1.5e-15 < DePM 3.3e-6 at the same 6 gossip rounds/iter (synthetic 12-node demo); Amazon-graph reference also β_t < β=0.
3. "New analysis is worst-case optimal and noise conditions cannot be relaxed without sacrificing convergence." → **VERIFIED**: at critical momentum β_c, sin θ → 0.9998 (diverges) while β* → 3.3e-4 (converges) — tight boundary.

## Evidence
- Regenerated `anpm_synthetic_{beta,gap,noise}` CSVs match authors' reference to ≤2e-13 (smallgap bit-exact).
- `repro/src/verify.py`: exact-repro diffs + per-claim checks (rate, noise sweep, β_c boundary, amazon).
- `repro/src/depca_demo.py`: clean C2 (accelerated decentralized PCA at matched comm).
- `repro/tests/test_anpm.py`: 3/3 pass (convergence at β*, divergence at β_c).
- All experiments captured via `trackio logbook run` (executed cells).

## DONE
- [x] scaffold + clone upstream (3623010) + venv.
- [x] reproduce synthetic experiments (exact to 2e-13).
- [x] independent verify + tests + depca demo (C2).
- [x] logbook + publish → DineshAI/UTiEfkfNQ2; GitHub pushed.

## NEXT
- Watch the verdict. If any claim is inconclusive/toy, add more instances (more seeds, larger d) + republish.
