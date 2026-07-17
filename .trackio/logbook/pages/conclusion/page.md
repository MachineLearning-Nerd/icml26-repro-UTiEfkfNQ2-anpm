# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_26322c5dd9b7", "created_at": "2026-07-16T19:04:18+00:00", "title": "Executive summary: all 3 claims verified (ANPM, UTiEfkfNQ2)", "pinned": true, "pinned_at": "2026-07-16T19:04:45+00:00"}
-->
Reproduction of 'Improved Analysis of the Accelerated Noisy Power Method' (arXiv 2602.03682, OpenReview UTiEfkfNQ2). Official code pierreaguie/ANPM (commit 3623010), run unmodified. CPU-only (numpy/scipy).

- C1 (accelerated rate under milder perturbation): VERIFIED -- beta* decays faster than beta=0 (slope -0.01036 vs -0.00728); ANPM converges for perturbation eta in [1e-4,1e-2] (floor 3.8e-4 -> 3.6e-2). Regenerated CSVs match the authors' reference to <=2e-13.
- C2 (first accelerated decentralized PCA, similar comm cost): VERIFIED WITH THE OFFICIAL DECENTRALIZED ALGORITHMS -- `depca_egofb.py` runs `ADePM` and `DePM` from `depca.py` on a real ego-Facebook network with 50 agents, local d=50, k=5, T=200. Communication is identical within each comparison: both call the same accelerated-gossip routine L=20 or L=40 times per outer iteration. Tuned ADePM is 9.33x lower-error than DePM at L=20 and 24,696x lower at L=40. The regenerated 201x9 CSV matches the authors' reference within 2.12e-12.
- C3 (worst-case optimal; cannot relax conditions): VERIFIED -- at the critical momentum beta_c the iterate DIVERGES (sin theta -> 0.9998) while beta* converges (3.3e-4); the boundary is tight.

Independent verification: synthetic exact-reproduction diffs <=1.4e-12; official Facebook DePCA diff 2.12e-12; full Amazon curves within 6.89% of the reference; unit tests 7/7 pass. Trackio captures both full commands, result artifacts, verifier, hashes, plots, and environment.

Scope: full paper scale on CPU (d=1000, T=1000; real decentralized Facebook experiment and 262k-node Amazon graph). No GPU, no training. Claim 2's primary evidence directly executes ADePM/DePM with matched gossip; Amazon is complementary centralized-scale evidence.

Repo: (pending GitHub push).
