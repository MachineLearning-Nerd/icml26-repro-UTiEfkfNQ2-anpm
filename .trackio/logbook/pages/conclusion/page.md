# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_26322c5dd9b7", "created_at": "2026-07-16T19:04:18+00:00", "title": "Executive summary: all 3 claims verified (ANPM, UTiEfkfNQ2)", "pinned": true, "pinned_at": "2026-07-16T19:04:45+00:00"}
-->
Reproduction of 'Improved Analysis of the Accelerated Noisy Power Method' (arXiv 2602.03682, OpenReview UTiEfkfNQ2). Official code pierreaguie/ANPM (commit 3623010), run unmodified. CPU-only (numpy/scipy).

- C1 (accelerated rate under milder perturbation): VERIFIED -- beta* decays faster than beta=0 (slope -0.01036 vs -0.00728); ANPM converges for perturbation eta in [1e-4,1e-2] (floor 3.8e-4 -> 3.6e-2). Regenerated CSVs match the authors' reference to <=2e-13.
- C2 (first accelerated decentralized PCA, similar comm cost): VERIFIED AT FULL PAPER SCALE -- official code regenerated Amazon0302 (262,111 nodes, 1,234,877 edges), k=30, T=100, sigma=1e-3. Tuned ANPM reaches 2.7234e-3 versus 3.1186e-3 for beta=0 at the matched budget (12.67% lower). The small decentralized control also has identical gossip budgets.
- C3 (worst-case optimal; cannot relax conditions): VERIFIED -- at the critical momentum beta_c the iterate DIVERGES (sin theta -> 0.9998) while beta* converges (3.3e-4); the boundary is tight.

Independent verification: synthetic exact-reproduction diffs <=1.4e-12; full Amazon curves within 6.89% of the reference and same final ordering; unit tests 5/5 pass; Trackio captures the 657-second full-scale command, result artifact, verifier, hashes, and environment.

Scope: full paper scale on CPU (d=1000, T=1000; 12-262k-node graphs). No GPU, no training. The full Amazon k=30 experiment is independently regenerated, not inferred from the authors' bundled CSV.

Repo: (pending GitHub push).
