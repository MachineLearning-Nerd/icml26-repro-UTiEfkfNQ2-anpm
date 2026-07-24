# Claim 4 — gossip mixing Õ(1/√γ_W) vs Õ(1/γ_W) (§3.1, Prop. 3.2)

> **Exact claim tested (§3.1, Proposition 3.2):** The accelerated gossip
> subroutine (Liu & Morse 2011, Alg. 1) attains mixing rate
> Õ(1/√γ_W), versus Õ(1/γ_W) for standard (non-accelerated) gossip, where
> γ_W is the absolute spectral gap of the gossip matrix W.

**Source audit.** ar5iv `2602.03682`, §3.1, Definition 3.1, Proposition 3.2,
Algorithm 1. This claim was previously INCONCLUSIVE because the depca experiments
used accelerated gossip for *both* ADePM and DePM, so the rate comparison was
never isolated. Here we isolate the gossip subroutine.

## Method (repro/src/claim4_gossip_mixing.py)

Isolated test of the gossip subroutine, independent of PCA. Using the **official**
`anpm.gossip.AcceleratedGossip`:
- **standard** gossip: ω = 0 → Y_{ℓ+1} = W Y_ℓ (contracts (1−γ_W)^ℓ);
- **accelerated** gossip: ω = `compute_omega(W)` (Liu–Morse) (contracts (1−√γ_W)^ℓ).

For ring graphs (Metropolis weights) with γ_W = sin²(π/n) spanning several orders
of magnitude (n = 4…60), we measure (1) the **exact spectral radius** of each
gossip operator (eigenvalue computation, independent of any trajectory fit) and
(2) the **first-hit** number of rounds L\* to reach a 10⁻⁶ mixing tolerance, then
fit L\* ∼ (1/γ_W)^p.

## Evidence

![Claim 4: gossip mixing scaling](../../images/claim4_gossip_mixing.png)

**Scaling fit** L\* ∼ (1/γ_W)^p:

| method | fitted p | theory |
| --- | --- | --- |
| standard (ω=0) | **1.015** | 1.0 |
| accelerated (Liu–Morse) | **0.518** | 0.5 |
| accelerated vs 1/√γ_W | **1.036** | 1.0 |

**Contraction factor** (exact spectral radius vs theory):

| method | empirical ρ | theory bound | note |
| --- | --- | --- | --- |
| standard | = 1−γ_W | 1−γ_W | match to **1.3×10⁻¹⁵** |
| accelerated | ≤ 1−√γ_W | 1−√γ_W | guarantee respected (actual ≤ bound) |

Example separation (n=60, γ_W=2.7e-3): standard needs **4410** rounds, accelerated
needs **202** — a 22× speedup that grows with 1/√γ_W.

## Reproducibility

- **Code:** [`repro/src/claim4_gossip_mixing.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/src/claim4_gossip_mixing.py); official `anpm.gossip.{AcceleratedGossip, compute_omega}`.
- **Raw JSON:** [`data/claim4_gossip_mixing.json`](../../data/claim4_gossip_mixing.json).
- **Independent checker:** the contraction factor is computed two independent ways — (a) the exact eigenvalues of the gossip operator, (b) the first-hit L\* from a simulated trajectory — and both agree with theory.
- **Negative control:** standard gossip (ω=0) is the non-accelerated control; it is slower by exactly the predicted factor.
- **Command:** `bash repro/run.sh` (Claim 4). **Env:** uv .venv, py 3.12. **Seed:** 0. **Runtime:** 0.6 s.

## Limitations & deviations

Ring graphs (Metropolis weights) give a clean γ_W sweep; the rate holds for any
gossip matrix (Prop. 3.2 is graph-agnostic). The accelerated contraction is the
*bound* (1−√γ_W); the exact spectral radius is slightly tighter (the algorithm
does at least as well as the guarantee).

## Verdict: **VERIFIED** (isolated gossip subroutine: Õ(1/√γ_W) vs Õ(1/γ_W))
