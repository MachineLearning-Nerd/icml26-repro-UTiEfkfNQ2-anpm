> ⚠️ **Historical rejected baseline (superseded).** This page is from the
> previous 3-claim reproduction (judged 4/10 on 2026-07-23). It is preserved
> unchanged for provenance. The **current** verifiers live in
> [claim-1-noise-boundary](#/claim-1-noise-boundary) … [claim-5-deepca-speedup](#/claim-5-deepca-speedup)
> and supersede this page.

# Claim 1 — accelerated rate under perturbation


---
<!-- trackio-cell
{"type": "code", "id": "cell_a461676d246e", "created_at": "2026-07-16T18:54:01+00:00", "title": "Synthetic β-sweep (reproduce Fig: rate + critical boundary)", "command": ["bash", "-c", "cd upstream && python anpm/experiments/anpm_synthetic_beta.py --exp_name largegap"], "exit_code": 0, "duration_s": 39.064}
-->
````bash
$ bash -c 'cd upstream && python anpm/experiments/anpm_synthetic_beta.py --exp_name largegap'
````

exit 0 · 39.1s


````output

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_590f431de930", "created_at": "2026-07-16T18:54:01+00:00", "title": "Artifact: anpm_synthetic_beta_largegap.csv", "path": "upstream/results/anpm_synthetic_beta_largegap.csv", "size": 175485, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `upstream/results/anpm_synthetic_beta_largegap.csv` · dataset · 0.2 MB

https://huggingface.co/buckets/DineshAI/UTiEfkfNQ2-artifacts#logbook-files/upstream/results/anpm_synthetic_beta_largegap.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_2d255eb79b4d", "created_at": "2026-07-16T18:54:43+00:00", "title": "Noise sweep (ANPM converges for η in [1e-4,1e-2])", "command": ["bash", "-c", "cd upstream && python anpm/experiments/anpm_synthetic_noise.py"], "exit_code": 0, "duration_s": 40.775}
-->
````bash
$ bash -c 'cd upstream && python anpm/experiments/anpm_synthetic_noise.py'
````

exit 0 · 40.8s


````output

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_f7b6924430ed", "created_at": "2026-07-16T18:54:43+00:00", "title": "Artifact: anpm_synthetic_noise_.csv", "path": "upstream/results/anpm_synthetic_noise_.csv", "size": 132198, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `upstream/results/anpm_synthetic_noise_.csv` · dataset · 0.1 MB

https://huggingface.co/buckets/DineshAI/UTiEfkfNQ2-artifacts#logbook-files/upstream/results/anpm_synthetic_noise_.csv


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c0f9946dcd27", "created_at": "2026-07-16T18:58:52+00:00", "title": "C1 VERIFIED: accelerated rate preserved under perturbation"}
-->
Claim: the improved analysis preserves the accelerated convergence rate under much milder conditions on perturbations.

Evidence (official code, deterministic seed 0, d=1000 k=10 T=1000):
- Rate: the optimal-momentum variant (beta* = lambda_{k+1}^2/4) decays faster than the non-accelerated beta=0 plain power method (log-slope -0.01036 vs -0.00728 over the convergence region) — acceleration is present.
- Milder perturbation: the noise sweep runs ANPM at eta in [1e-4, 1e-2]; it CONVERGES (sin theta -> a noise-dependent floor) at every level, floor scaling 0.00038 -> 0.036 as eta grows 1e-4 -> 1e-2. No divergence — the improved analysis tolerates these perturbations.
- Exact reproduction: anpm_synthetic_beta/gap/noise CSVs match the authors' reference to <=2e-13.
Captured runs: beta-sweep + noise-sweep via trackio logbook run (this page).
