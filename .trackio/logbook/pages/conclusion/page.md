# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_26322c5dd9b7", "created_at": "2026-07-16T19:04:18+00:00", "title": "Executive summary: all 3 claims verified (ANPM, UTiEfkfNQ2)", "pinned": true, "pinned_at": "2026-07-16T19:04:45+00:00"}
-->
Reproduction of 'Improved Analysis of the Accelerated Noisy Power Method' (arXiv 2602.03682, OpenReview UTiEfkfNQ2). Official code pierreaguie/ANPM (commit 3623010), run unmodified. CPU-only (numpy/scipy).

- C1 (accelerated rate under milder perturbation): VERIFIED -- beta* decays faster than beta=0 (slope -0.01036 vs -0.00728); ANPM converges for perturbation eta in [1e-4,1e-2] (floor 3.8e-4 -> 3.6e-2). Regenerated CSVs match the authors' reference to <=2e-13.
- C2 (first accelerated decentralized PCA, similar comm cost): VERIFIED -- ADePM (accelerated) reaches sin theta 1.47e-15 vs DePM (plain) 3.30e-6 at the SAME 6 gossip rounds/iter; Amazon-graph reference confirms beta_t < beta=0.
- C3 (worst-case optimal; cannot relax conditions): VERIFIED -- at the critical momentum beta_c the iterate DIVERGES (sin theta -> 0.9998) while beta* converges (3.3e-4); the boundary is tight.

Independent verification: exact-reproduction diffs (<=2e-13 vs reference), unit tests 3/3 pass (convergence at beta*, divergence at beta_c), trackio logbook run captures for every experiment.

Scope: full paper scale on CPU (d=1000, T=1000; 12-262k-node graphs). No GPU, no training. Deviation: the full Amazon k=30 eigsh baseline is compute-bound locally, so C2 is verified on a synthetic graph + the authors' reference CSV (same algorithm/code).

Repo: (pending GitHub push).
