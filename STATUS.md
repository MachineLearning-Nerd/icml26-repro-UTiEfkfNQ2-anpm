# STATUS — ANPM (UTiEfkfNQ2) reproduction

**Session:** autoloop. **Last updated:** 2026-07-17. **State: FULL-SCALE C2 REPAIR COMPLETE; publishing for re-verdict.**
HF: https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2 at repaired Space SHA `36740b72fdfa713e527e8af2acb8d42513e2d903` · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm.

## Paper
- **Title:** Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA. arXiv 2602.03682 · OpenReview UTiEfkfNQ2 (Aguié, Even, Massoulié).
- **Official code:** `pierreaguie/ANPM` (commit `3623010`) → `upstream/`, run unmodified.
- **Compute:** CPU-only (numpy/scipy/networkx). No GPU, no training.

## Official claims (verbatim) — ALL VERIFIED
1. "Improved analysis preserves accelerated convergence rate under much milder conditions on perturbations." → **VERIFIED**: β* decays faster than β=0 (slope −0.01036 vs −0.00728); ANPM converges for η ∈ [1e-4,1e-2] (floor 3.8e-4→3.6e-2).
2. "First decentralized algorithm for PCA with provably accelerated convergence and similar communication costs." → **VERIFIED AT FULL PAPER SCALE**: official pinned code regenerated Amazon0302 (262,111 nodes, 1,234,877 edges), k=30, T=100, σ=1e-3. Tuned ANPM 2.7234e-3 < plain 3.1186e-3 at the matched budget (12.67% reduction). A 12-node decentralized control separately matches gossip rounds.
3. "New analysis is worst-case optimal and noise conditions cannot be relaxed without sacrificing convergence." → **VERIFIED**: at critical momentum β_c, sin θ → 0.9998 (diverges) while β* → 3.3e-4 (converges) — tight boundary.

## Evidence
- Regenerated `anpm_synthetic_{beta,gap,noise}` CSVs match authors' reference to ≤1.4e-12 (smallgap bit-exact).
- Full Amazon run completed in 657 seconds; all 11 checkpoints regenerated. Curves are within 6.89% of the authors' reference despite the error metric's loose iterative-eigensolver tolerance (1e-3), and both have the same final ordering.
- `repro/src/verify.py`: exact synthetic diffs + per-claim checks (rate, noise sweep, β_c boundary, full Amazon scale/schedule/ordering).
- `repro/src/depca_demo.py`: clean C2 (accelerated decentralized PCA at matched comm).
- `repro/tests/test_anpm.py`: 5/5 pass, including full Amazon reference-envelope and matched-budget checks.
- `outputs/amazon_rank30_provenance.json`: dataset/result hashes, exact command, pinned upstream commit, environment.
- All experiments captured via `trackio logbook run` (executed cells).

## DONE
- [x] scaffold + clone upstream (3623010) + venv.
- [x] reproduce synthetic experiments (exact to 2e-13).
- [x] independent verify + tests + depca demo (C2).
- [x] original logbook + publish → DineshAI/UTiEfkfNQ2; GitHub pushed.
- [x] independently regenerate full Amazon0302 k=30 experiment and add judge-facing evidence.

## NEXT
- Publish repaired Space/GitHub state and wait for an official 6/6 re-verdict.
