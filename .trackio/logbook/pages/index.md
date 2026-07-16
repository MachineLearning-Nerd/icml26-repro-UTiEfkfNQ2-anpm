# Repro - Accelerated Noisy Power Method (ANPM)

## Pages

| Page |
| --- |
| [Claim 1 — accelerated rate under perturbation](#/claim-1-accelerated-rate-under-perturbation) |
| [Methods & environment](#/methods-environment) |
| [Negative controls & falsification](#/negative-controls-falsification) |
| [Claim 2 — decentralized PCA](#/claim-2-decentralized-pca) |
| [Claim 3 — worst-case optimal boundary](#/claim-3-worst-case-optimal-boundary) |
| [Conclusion](#/conclusion) |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_36c54e4830b9", "created_at": "2026-07-16T18:58:51+00:00", "title": "ANPM (UTiEfkfNQ2) — ICML 2026 reproduction"}
-->
Reproduction of 'Improved Analysis of the Accelerated Noisy Power Method with Applications to Decentralized PCA' (Aguié, Even, Massoulié; arXiv 2602.03682, OpenReview UTiEfkfNQ2) for the ICML 2026 Agent Reproduction Challenge. Official code pierreaguie/ANPM (commit 3623010), run unmodified. CPU-only linear algebra (numpy/scipy).

Headlines: the regenerated experiment CSVs match the authors' reference to <=2e-13 (FP library-version noise; smallgap bit-exact). C1 accelerated rate VERIFIED (beta* decays faster than beta=0; ANPM converges for perturbation eta in [1e-4,1e-2]). C3 worst-case-optimal boundary VERIFIED (at the critical momentum beta_c the iterate DIVERGES, sin theta ~= 1.0, while beta* converges to 3e-4). C2 decentralized PCA executed on the Amazon graph. 3/3 unit tests pass.
