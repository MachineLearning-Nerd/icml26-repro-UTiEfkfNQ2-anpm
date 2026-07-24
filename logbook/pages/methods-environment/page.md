# Methods & environment

## Fixed reproduction command (identical on every node)

```bash
bash repro/run.sh
```

`repro/run.sh` bootstraps `uv`, syncs the locked environment from `uv.lock`, and
runs [`repro/src/run_all.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/src/run_all.py),
which executes all five claim verifiers and prints every metric to stdout. The run
exits **nonzero** if any claim is not at a terminal VERIFIED/FALSIFIED verdict.

## Pinned environment

| item | value |
| --- | --- |
| manager | `uv` (single repo-level `.venv`, no conda) |
| python | 3.12.11 |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| networkx | 3.6.1 |
| matplotlib | 3.x (figures only) |
| lockfile | [`uv.lock`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/uv.lock) |

## Official code (vendored, unmodified)

`pierreaguie/ANPM` commit **`3623010e6a3d35bced7fa45b89689753b23551df`**, vendored
as [`repro/anpm/`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/anpm/)
with the self-contained [`repro/datasets/facebook_combined.txt`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/datasets/facebook_combined.txt).
Authors' reference CSVs in [`repro/results_reference/`](https://github.com/MachineLearning-Nerd/icml26-repro-UTiEfkfNQ2-anpm/blob/master/repro/results_reference/).

## Compute

- **Backend:** local CPU (`orx exp run --backend local`), 1 core.
- **Wall time:** 265.8 s total (Claim 1: 220.9 s; Facebook: ~20 s; Claims 2/4: <3 s; Claims 3/5: <1 s).
- **GPU:** none (CPU-only linear algebra).

## Git SHA & seeds

- **Winning branch:** `master` @ `f0afde41b54a4f57ec594bad0ab498cf1b42deb8` (baseline experiment `orx/anpm-5-claim-baseline-uv-env-official-code`, run `6758a894`).
- **Seeds:** `np.random.seed(0)` for instances & the Facebook experiment; `seed=1` for Claim 1 Part A noise. Claim 2 is deterministic (no RNG).

## Evaluator-visible visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [claim-1](#/claim-1-noise-boundary) | ✓ | ✓ | ✓ json | official `anpm` + indep | β=0 slower | Thm 2.2 Eqs 3–4 | VERIFIED |
| 2 | [claim-2](#/claim-2-counterexamples) | ✓ | ✓ | ✓ json | official + sign-canonical QR | strict c=1/32 converges | Thm 2.4 & 2.5 | VERIFIED |
| 3 | [claim-3](#/claim-3-adepm) | ✓ | ✓ | ✓ csv+json | exact-diff vs ref | DePM worse | Thm 3.3 | VERIFIED |
| 4 | [claim-4](#/claim-4-gossip-mixing) | ✓ | ✓ | ✓ json | spectral radius + first-hit | standard slower | §3.1 Prop 3.2 | VERIFIED |
| 5 | [claim-5](#/claim-5-deepca-speedup) | ✓ | ✓ | ✓ json | DeEPCA official code | DePM/DeEPCA plateau | §4 | VERIFIED |
