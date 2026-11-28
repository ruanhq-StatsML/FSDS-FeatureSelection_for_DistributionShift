# gen_fsds - generative conditional samplers for FSDS / CFPerm

Prototype + benchmark for replacing the marginal / residual permutations in the FSDS
covariate-shift and concept-drift arms with samplers of `X_j | X_-j` built on
**zuko** (normalizing flows) and **torchcfm** (conditional / rectified flow matching).

```
pip install -r requirements_gen.txt        # from python/
python -m gen_fsds.run_benchmark --suite covshift --out results_gen --reps 5
python -m gen_fsds.report results_gen      # -> results_gen/REPORT.md + figures
./run_gen_benchmark_all.sh                 # everything (about 2 h on 4 CPU cores, n = 5000 per batch)
```

## Modules

| file | content |
|---|---|
| `dgp.py` | SEM data-generating process with non-Gaussian, heteroscedastic, nonlinear mechanisms. Shifts are injected at the mechanism level (`mean`, `scale`, `shape`), so the exact set of features whose *full conditional* shifts is known (`S ∪ pa(S)`), as are descendants that only inherit a marginal shift. Concept drift (`mean` or `var` of `Y|X`) confined to `S`, confounded by an independent covariate shift. |
| `flows.py` | `NSF1D` (zuko NSF, exact log-density), `CFM1D` (torchcfm, sigma=0 = rectified flow, optional reflow), `AmortCFM` (one masked network for every `X_j | X_-j`), `JointOTCFM` (joint OT-CFM transport). |
| `samplers.py` | Common `fit(X)` / `sample(Xq, j, B, rng)` interface: `Marginal` (the repo's current `condperm`/`oob`), `LinGauss`, `RFResid` (PermuCATE-style), `QRF`, `NSF`, `CFM`, `CFM-reflow1`, `AmortCFM`. |
| `covshift.py` | MMD-LOCO (repo), DC-LOCO, `DC-CPI:<sampler>` (domain-classifier conditional permutation importance with t-test p-values), `CondKL:NSF`, `CondW1:CFM`, `JointOT:CFM`. |
| `conceptdrift.py` | `CD-LOCO`, `CD-CPI:<sampler>` (doubly-robust pseudo-outcome PO-risk, PermuCATE-style), `DistDrift:<sampler>` (distributional drift via a pseudo-outcome `log q1(y|x) - log q0(y|x)`). |
| `quality.py` | C2ST sampler-quality benchmark (paired-group CV). |
| `metrics.py` | AUC against exact truth, FPR / power at 0.05, BH FDP / power, paired one-sided t-test. |
| `run_benchmark.py`, `report.py` | Resumable, parallel driver and aggregation. |

## Statistical design

* Samplers are trained on the **pooled train split without the domain label**, so under
  `H0: T ⟂ X_j | X_-j` an exact sampler makes `(X_-j, X~_j)` exchangeable with `(X_-j, X_j)`;
  the CPI t-test is then valid. Test statistics are computed on a held-out split.
* Truth for covariate shift is `S ∪ pa(S)`: a shifted mechanism of `c` enters the full
  conditional of every parent of `c`. Descendants of `S` inherit a marginal shift only.
* Not implemented / known limitations: discrete or mixed-type features, `p` in the hundreds,
  real-data runs, finite-sample guarantee under sampler misspecification (the paper's GOF
  step is replaced here by the C2ST sampler benchmark).
