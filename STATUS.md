# STATUS — ANPM (UTiEfkfNQ2) reproduction

**Session:** autoloop. **Last updated:** 2026-07-17. **State: SECOND C2 REPAIR COMPLETE; publishing for re-verdict.**
At Space SHA `36740b72fdfa713e527e8af2acb8d42513e2d903`, the official judge retained 5/6 because Amazon ANPM did not execute decentralized gossip. The new repair directly runs the official Facebook `ADePM`/`DePM` experiment and is published at Space SHA `ead88b5802d8d9725df6463b05bd9dd5495bfcd8`. HF: https://huggingface.co/spaces/DineshAI/UTiEfkfNQ2 · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm.

## Paper
- **Title:** Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA. arXiv 2602.03682 · OpenReview UTiEfkfNQ2 (Aguié, Even, Massoulié).
- **Official code:** `pierreaguie/ANPM` (commit `3623010`) → `upstream/`, run unmodified.
- **Compute:** CPU-only (numpy/scipy/networkx). No GPU, no training.

## Official claims (verbatim) — ALL VERIFIED
1. "Improved analysis preserves accelerated convergence rate under much milder conditions on perturbations." → **VERIFIED**: β* decays faster than β=0 (slope −0.01036 vs −0.00728); ANPM converges for η ∈ [1e-4,1e-2] (floor 3.8e-4→3.6e-2).
2. "First decentralized algorithm for PCA with provably accelerated convergence and similar communication costs." → **VERIFIED WITH OFFICIAL DECENTRALIZED CODE ON REAL DATA**: paper-default `depca_egofb.py` executes `ADePM` and `DePM` on a real 50-agent ego-Facebook network (local d=50, k=5, T=200). At identical gossip budgets, tuned ADePM is 9.33× lower-error for L=20 and 24,696× lower for L=40. The 201×9 output matches the authors' reference within 2.12e-12.
3. "New analysis is worst-case optimal and noise conditions cannot be relaxed without sacrificing convergence." → **VERIFIED**: at critical momentum β_c, sin θ → 0.9998 (diverges) while β* → 3.3e-4 (converges) — tight boundary.

## Evidence
- Regenerated `anpm_synthetic_{beta,gap,noise}` CSVs match authors' reference to ≤1.4e-12 (smallgap bit-exact).
- Full Amazon run completed in 657 seconds; all 11 checkpoints regenerated. Curves are within 6.89% of the authors' reference despite the error metric's loose iterative-eigensolver tolerance (1e-3), and both have the same final ordering.
- Official real Facebook DePCA run completed in 32 seconds; exact source algorithms, 50 agents, L=20/40 matched gossip, 201 checkpoints/method columns, max reference diff 2.12e-12.
- `repro/src/verify.py`: exact synthetic/Facebook diffs + per-claim checks (rate, noise sweep, β_c boundary, decentralized gossip, full Amazon scale).
- `repro/src/depca_demo.py`: clean C2 (accelerated decentralized PCA at matched comm).
- `repro/tests/test_anpm.py`: 7/7 pass, including exact official DePCA regeneration and matched-gossip checks.
- `outputs/amazon_rank30_provenance.json`: dataset/result hashes, exact command, pinned upstream commit, environment.
- `outputs/facebook_depca_full_provenance.json`: real dataset/result hashes, exact official command/source, communication contract, environment.
- All experiments captured via `trackio logbook run` (executed cells).

## DONE
- [x] scaffold + clone upstream (3623010) + venv.
- [x] reproduce synthetic experiments (exact to 2e-13).
- [x] independent verify + tests + depca demo (C2).
- [x] original logbook + publish → DineshAI/UTiEfkfNQ2; GitHub pushed.
- [x] independently regenerate full Amazon0302 k=30 experiment and add judge-facing evidence.
- [x] respond to first re-judge by regenerating official real Facebook ADePM/DePM at matched L=20/40.

## NEXT
- Publish repaired Space/GitHub state and wait for an official 6/6 re-verdict.
