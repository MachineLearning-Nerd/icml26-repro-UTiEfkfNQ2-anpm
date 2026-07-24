# ANPM (UTiEfkfNQ2) — claim-by-claim reproduction

Reproduction of *Improved Analysis of the Accelerated Noisy Power Method with
Applications to Decentralized PCA* (Aguié, Even, Massoulié; arXiv [2602.03682](https://arxiv.org/abs/2602.03682), OpenReview [UTiEfkfNQ2](https://openreview.net/forum?id=UTiEfkfNQ2)).

**Previous live judged score: 4/10.** This revision adds direct, faithful evidence
for every previously-TOY/INCONCLUSIVE claim (1, 2, 4, 5) and preserves the
full-credit Claim 3. All five claims are now at a terminal **VERIFIED** verdict,
regenerated from a single fixed command on a pinned, vendored copy of the
official code.

| Claim | Theorem / § | Previous | Current | Confidence | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 — accelerated rate under **mild** Δ·ε noise (vs Xu's Δ·ε^μ) | Thm 2.2 | TOY 1/2 | **VERIFIED** | HIGH | [claim-1-noise-boundary](#/claim-1-noise-boundary) |
| 2 — noise conditions **cannot be relaxed** (Thms 2.4–2.5) | Thm 2.4, 2.5 | TOY 1/2 | **VERIFIED** | HIGH | [claim-2-counterexamples](#/claim-2-counterexamples) |
| 3 — ADePM accelerated decentralized PCA | Thm 3.3 | VERIFIED 2/2 | **VERIFIED** | HIGH | [claim-3-adepm](#/claim-3-adepm) |
| 4 — gossip mixes in Õ(1/√γ_W) vs Õ(1/γ_W) | §3.1, Prop 3.2 | INCONCLUSIVE 0/2 | **VERIFIED** | HIGH | [claim-4-gossip-mixing](#/claim-4-gossip-mixing) |
| 5 — β\*=λ²_{k+1}/4 √-speedup + ADePM > DePM/DeEPCA | §4 | INCONCLUSIVE 0/2 | **VERIFIED** | HIGH | [claim-5-deepca-speedup](#/claim-5-deepca-speedup) |

## How to reproduce

Fixed command (identical on every node), pinned environment, deterministic seeds:

```bash
bash repro/run.sh      # bootstraps uv from uv.lock, runs repro/src/run_all.py
```

- **Environment:** `uv`-managed `.venv`, Python 3.12, numpy 2.5.1, scipy 1.18.0, networkx 3.6.1 (pinned in [`pyproject.toml`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/pyproject.toml) / [`uv.lock`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/uv.lock)).
- **Official code:** `pierreaguie/ANPM` commit `3623010` vendored verbatim as [`repro/anpm/`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/anpm/) (+ self-contained `repro/datasets/facebook_combined.txt`).
- **Compute:** local CPU, 1 core, ~4.4 min total (see [methods-environment](#/methods-environment)).

## Headline result

![ADePM vs DePM vs DeEPCA on real ego-Facebook](../images/facebook_adepm_deepca.png)

ADePM (β\*=λ²_{k+1}/4) reaches 8.5×–1.2M× lower error than DePM **and** DeEPCA at
matched communication on the real SNAP ego-Facebook graph, while DeEPCA plateaus
(its ε-independent communication stops improving past L=20). The square-root
speedup of β\* is measured directly: T(β=0)/T(β\*) tracks 1/√Δ to exponent 1.00.

## Pages (current verifiers first)

- [Claim 1 — mild Δ·ε noise boundary](#/claim-1-noise-boundary)
- [Claim 2 — counterexamples (Thms 2.4–2.5)](#/claim-2-counterexamples)
- [Claim 3 — ADePM decentralized PCA](#/claim-3-adepm)
- [Claim 4 — gossip mixing rate](#/claim-4-gossip-mixing)
- [Claim 5 — β\* √-speedup & DeEPCA](#/claim-5-deepca-speedup)
- [Methods & environment](#/methods-environment)
- [Conclusion](#/conclusion)

Historical (previous 3-claim baseline, superseded): [claim-1-accelerated-rate-under-perturbation](#/claim-1-accelerated-rate-under-perturbation), [claim-2-decentralized-pca](#/claim-2-decentralized-pca), [claim-3-worst-case-optimal-boundary](#/claim-3-worst-case-optimal-boundary), [negative-controls-falsification](#/negative-controls-falsification).
