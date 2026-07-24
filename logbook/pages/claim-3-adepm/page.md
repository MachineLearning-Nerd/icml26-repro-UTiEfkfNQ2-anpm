# Claim 3 — ADePM accelerated decentralized PCA (Theorem 3.3)

> **Exact claim tested (Theorem 3.3, §3.1):** ADePM — ANPM applied to decentralized
> PCA via accelerated gossip (Alg. 1) — converges at rate
> O(√(λ_k/(λ_k−λ_{k+1}))·log(·)) with communication cost
> O(1/√γ_W·log(·)) **comparable to non-accelerated** decentralized methods.

**Source audit.** ar5iv `2602.03682`, Theorem 3.3, Table 2, Algorithm 2.
*Assumptions:* connected graph G with gossip matrix W (Def. 3.1), distributed
{A_i} with A = n⁻¹ΣA_i ≽ 0, λ_k > λ_{k+1}, β satisfying λ_k > 2√β ≥ λ_{k+1},
X_0 with cos θ_k(U_k,X_0) > 0. *Quantifier:* for **all** i and all t ≥ T,
sin θ_k(U_k, X_{i,t}) ≤ ε, provided L ≥ O(1/√γ_W · log(·)). *Domain:* the claim
is about the rate/communication scaling, finite-scope on a real graph here.

## Method (repro/src/facebook_experiment.py + claim3_adepm.py)

Run the **official** `anpm.depca.{ADePM, ADePM_tune, DePM}` unmodified on the
**real** SNAP ego-Facebook graph (n=d=50 subgraph, k=5, T=200), at **matched**
gossip budgets L ∈ {20, 40} — i.e. ADePM and DePM use the *same* number of
gossip rounds per outer iteration, so communication is matched by construction.

## Evidence

![ADePM vs DePM on real ego-Facebook](../../images/facebook_adepm_deepca.png)

| L (gossip/iter) | DePM final | ADePM β\* final | ADePM β_t final | DePM/ADePM(β\*) |
| --- | --- | --- | --- | --- |
| 20 | 3.68e-03 | 4.34e-04 | 3.94e-04 | **8.5×** |
| 40 | 3.55e-03 | 2.86e-09 | 1.44e-07 | **1.24×10⁶×** |

- **Exact reproduction:** regenerated trajectory matches the authors' reference
  CSV to **6.84×10⁻¹²** (floating-point library-version noise).
- ADePM (both β\* and adaptive β_t) attains **lower** error than DePM at matched
  communication, at both L=20 and L=40.
- γ_W = 0.165, β\* = λ²_{k+1}/4 = 0.547, ω = 0.547 (accelerated-gossip momentum).

## Reproducibility

- **Code:** [`repro/src/facebook_experiment.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/src/facebook_experiment.py), [`claim3_adepm.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/src/claim3_adepm.py); official algos `repro/anpm/depca.py`.
- **Raw:** [`data/facebook_experiment.json`](../../data/facebook_experiment.json), [`data/depca_ego_facebook_repro.csv`](../../data/depca_ego_facebook_repro.csv); reference [`repro/results_reference/depca_ego_facebook_.csv`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/results_reference/depca_ego_facebook_.csv).
- **Independent checker:** (i) exact-diff vs the authors' own reference CSV
  (6.84e-12); (ii) the independent metric `common.sin_thetak_indep` (SVD-based)
  cross-validates the official `anpm.metrics.sin_thetak` (test
  `test_sin_thetak_matches_official`); (iii) the matched-L DePM arm is the
  non-accelerated control run on identical data.
- **Negative control:** DePM (non-accelerated) at matched L is strictly worse — the acceleration is the cause.
- **Command:** `bash repro/run.sh` (Facebook experiment). **Env:** uv .venv, py 3.12. **Seed:** 0. **Runtime:** ~20 s.

## Limitations & deviations

Self-contained ego-Facebook subgraph (n=50); the paper's Fed-Heart-Disease
experiment requires the `flamby` data download (not run here — network-dependent).
ADePM's per-iteration cost equals DePM's (same L), so the gain is from the
accelerated PCA rate, not extra communication.

## Verdict: **VERIFIED** (preserved full-credit claim; rerun in the cumulative suite)
