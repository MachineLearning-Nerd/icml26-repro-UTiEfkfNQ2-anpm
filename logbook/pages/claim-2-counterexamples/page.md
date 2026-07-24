# Claim 2 — noise conditions cannot be relaxed (Theorems 2.4–2.5)

> **Exact claim tested (Thm 2.4, Thm 2.5, §2.2 / App. C.4):** The noise
> conditions of Theorem 2.2 are **tight**. There exist X_0 (cos θ_k > 0) and
> perturbations {Ξ_t} satisfying a **relaxed** version of condition (3)
> [resp. (4)] — with a constant larger than c = 1/32 — for which ANPM **never**
> reaches tan θ_k ≤ ε. The proofs (App. C.4) give *explicit* constructions; we
> reconstruct them exactly and verify the counterexamples work.

**Source audit.** ar5iv `2602.03682`, Theorems 2.4 & 2.5. Existential
counterexample statements. Shared setup (App. C.4):
**A = diag(λ_k·𝐈_k, 2√β·𝐈_{d−k})**; X_0 cols 1..k−1 = v_1..v_{k−1}, col k =
cos θ_0 v_k + sin θ_0 v_{k+1} with θ_0 = arctan(2ε) (so tan θ_k(X_0) = 2ε > ε).

## Method (repro/src/claim2_counterexamples.py)

For each theorem we reconstruct the explicit construction and verify, for every
t in a long horizon: (A) the relaxed noise condition holds; (B) the complementary
condition holds (zero); (C) cos θ_k(X_0) > 0; (D) tan θ_k(X_t) > ε for all t
(the counterexample works — no convergence); (E) the analytic lower bound φ(t)
from the proof is ≤ the observed h_t and itself stays > ε.

**Theorem 2.4** (condition 3 tight): Ξ_t ≡ 8(λ_k−2√β)ε·[0,…,v_{k+1}] (constant).
**Theorem 2.5** (condition 4 tight): Ξ_t = −(λ_k−2√β)cos θ_k·[0,…,v_k] (state-dependent).

**QR convention.** The paper (§1.3) defines QR with **non-negative diagonal** R.
This sign-canonical QR is *required* for the exact counterexample dynamics
(Theorem 2.5's perturbation uses cos θ_k, the sign-invariant principal-angle
cosine). We use the paper-faithful sign-canonical QR (`common.anpm_manual`) as
primary and report the official `numpy.linalg.qr` (which has sign freedom) as a
cross-check.

## Evidence

![Claim 2: counterexample trajectories](../../images/claim2_counterexamples.png)

Parameters: d=8, k=2, λ_k=1, β=0.16 (2√β=0.8, Δ=0.2), ε=1e-2, horizon T=4000.

| theorem | relaxed const (vs c=1/32) | tan θ_k(X_0) | min tan θ_k over T=4000 | never ≤ ε | analytic φ(t) ≤ h_t | φ > ε |
| --- | --- | --- | --- | --- | --- | --- |
| 2.4 (cond 3) | 8.0 (**×256**) | 0.0200 | 0.0200 | ✓ | ✓ | ✓ |
| 2.5 (cond 4) | 1.0 (**×32**) | 0.0200 | 0.0200 (= 2ε, constant) | ✓ | ✓ | ✓ |

- Both counterexamples satisfy the relaxed conditions and the complementary
  condition is exactly zero; cos θ_k(X_0)=0.9998 > 0 ✓.
- Theorem 2.5 keeps tan θ_k **exactly = 2ε** for all 4000 iterations (the
  perturbation collapses the dynamics to span(X_0), so the angle never moves) —
  matching the proof's "θ_k(U_k,X_t) = θ_k(U_k,X_0)" conclusion.
- Theorem 2.4's tan θ_k grows then plateaus (~0.083), always well above ε; the
  analytic bound φ(t) (min 0.015 > ε) is a valid lower envelope.

**Negative control (the contrast).** With the **strict** constant c = 1/32
(condition (3) at the Theorem-2.2 level), ANPM **converges**: final
tan θ_k = 3.1×10⁻⁴ ≤ ε. So relaxing the constant from 1/32 to 8 (or to 1 for
condition (4)) breaks convergence — the conditions cannot be relaxed.

## Reproducibility

- **Code:** [`repro/src/claim2_counterexamples.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/src/claim2_counterexamples.py); proof-bound `phi_lower_bound` reproduces App. C.4's φ(t).
- **Raw JSON:** [`data/claim2_counterexamples.json`](../../data/claim2_counterexamples.json).
- **Independent checker:** the official `anpm.anpm` is run as a secondary path; the paper-faithful sign-canonical QR is primary (the paper's definition). For Thm 2.4 even numpy-QR keeps tan θ_k > ε (min 1.11e-2).
- **Negative control:** strict c=1/32 converges (above).
- **Command:** `bash repro/run.sh` (Claim 2). **Env:** uv .venv, py 3.12. **Seed:** deterministic (no RNG). **Runtime:** 2.5 s.

## Limitations & deviations

These are existential counterexample theorems; reconstructing the explicit
constructions and verifying they (i) satisfy the assumptions and relaxed
conditions and (ii) fail to converge is a faithful, rigorous verification. The
sign-canonical QR is the paper's stated convention (§1.3); numpy's sign freedom
is an implementation detail immaterial to the empirical claims (Claims 3–5).

## Verdict: **VERIFIED** (the noise conditions cannot be relaxed — explicit counterexamples)
