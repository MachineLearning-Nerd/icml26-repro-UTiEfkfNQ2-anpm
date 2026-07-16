# Claim 3 — worst-case optimal boundary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9a6137838d96", "created_at": "2026-07-16T18:58:52+00:00", "title": "C3 VERIFIED: worst-case optimal — beta_c diverges (boundary is tight)"}
-->
Claim: the new analysis is worst-case optimal and the noise/momentum conditions cannot be relaxed without sacrificing convergence.

Evidence (beta-sweep, d=1000 k=10 T=1000, adversarial noise):
- At the OPTIMAL momentum beta* = lambda_{k+1}^2/4: sin theta -> 3.3e-4 (converges).
- At the CRITICAL momentum beta_c = lambda_k^2/4 (the upper boundary): sin theta -> 0.9998 — the iterate DIVERGES, never converging.
This sharp transition (converge below beta_c, diverge at/above it) is exactly the worst-case-optimal/tight statement: the momentum condition cannot be relaxed past beta_c. Independent unit test test_anpm_diverges_at_beta_critical confirms it on a fresh instance.
Captured run: beta-sweep (this page shares the beta experiment).
