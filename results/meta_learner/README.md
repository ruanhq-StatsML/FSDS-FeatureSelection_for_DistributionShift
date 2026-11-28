# Attribution-guided meta-learner: technical roadmap and results

Status: toy-world evidence only. Everything below is simulated with known ground truth. Nothing here has been
run on real recommendation logs or on an LLM.

## 1. One idea, three scenarios

A selection rule that looks at one sample's loss confuses two things that need opposite treatment:

- drift: valuable, persistent, localised to a few coordinates, slices or contexts;
- noise: idiosyncratic, not persistent.

The meta-learner therefore runs the same five stages in every scenario:

| Stage | What it does | Recsys (E1) | Data selection (E2) | Generative / LM (E3) |
|---|---|---|---|---|
| 1. Gate | Is the new data surprising relative to a reference? | per-feature CUSUM on expected-permutation importance drop | slice-level importance z-score under a fresh-data reference | per-context G-test, slice vs fresh |
| 2. Localise | Which coordinates / contexts carry the conflict? | top-K candidates among the top-2S baseline features | features with z < -ZC | contexts with significant G-test |
| 3. Select / weight | What do we do with the affected data? | raise process noise Q_j on flagged features | reuse the slice on all other coordinates | drop flagged rows if localised, down-weight if diffuse |
| 4. Update | Apply the change | diag-EKF step (Sherman-Morrison, rank one) | logistic fit + slice-specific frozen offset on conflicting coordinates | count-based bigram update |
| 5. Validate | Does it help, and what did it cost? | excess KL + retention KL on the old concept | excess KL | KL to the true transition matrix |

Data-assimilation reading: the proximal fit is a 3D-Var/MAP step with the previous estimate as background. The
flags estimate the support of a sparse process-noise matrix Q (flow-dependent covariance inflation). Sample
weights play the role of observation-error inflation.

Catastrophic-forgetting reading: EWC stiffens parameters that mattered for the old task. This learner loosens the
parameters implicated in the shift. The two differ in whether the old task is still evaluated, which is why a
retention metric is reported.

## 2. Results

Scripts (each at most 100 lines and 120 characters per line):

- `python/meta_online_persample.py` -> `recsys_persample.txt` (24 seeds), `recsys_fixed_cfg.txt` (24 seeds)
- `python/meta_data_selection.py 30` -> `data_selection.txt` (30 seeds)
- `python/meta_generative_toy.py 100` -> `generative_toy.txt` (100 seeds)

Metric: exact expected excess KL (no label noise in the metric). Hyperparameters are tuned on separate seeds
(100-102, or 1000-1009 for E3). The standard errors are over evaluation seeds only and do not include tuning noise.

### E1: per-sample online recsys (p=1000, 6400 stream samples, sparse drift of K=4 coefficients)

`recsys_persample.txt` tunes each method separately for each scenario. `recsys_fixed_cfg.txt` uses one fixed
configuration everywhere, which is closer to deployment.

Fixed configuration (excess KL, lower is better; retention KL on the old concept in brackets):

| Scenario | static | ekf | samp (per-sample surprise weight) | attr (ours) |
|---|---|---|---|---|
| abrupt | 0.682 | 0.329 (0.446) | 0.321 (0.432) | **0.246** (0.605) |
| abrupt + 10% label flips | 0.590 | 0.331 (0.403) | 0.326 (0.383) | **0.276** (0.494) |
| gradual dense drift | 0.253 | **0.190** (0.195) | 0.199 | 0.192 (0.205) |
| null (no drift) | 0.228 | **0.182** (0.181) | 0.192 | 0.188 (0.186) |

Reading:

- Abrupt and noisy sparse drift: attribution gives a 17-25% reduction versus ekf. Per-sample surprise weighting
  (`samp`) is indistinguishable from ekf, which matches the pre-registered prediction.
- Gradual dense drift and null: no gain. Under a fixed configuration the null costs about 3% (0.188 versus 0.182).
- Retention cost is real: old-concept KL rises from 0.446 to 0.605 in the abrupt scenario. This is the
  plasticity/forgetting trade-off, not a free lunch.
- With per-scenario tuning, the null scenario selects a threshold that never fires (0 flags per check), so that
  null row is flattering. The fixed-configuration table is the honest one.
- Localisation is information-limited at p=1000. About 400-500 samples are needed to separate a flipped
  coefficient. Flag precision / recall are 0.36 / 0.70 (abrupt), 0.28 / 0.53 (noisy) and 0.08 / 0.15 (gradual),
  from the per-scenario-tuned run in `recsys_persample.txt`.
- A random-K control with the same flag count is worse than plain ekf (abrupt 0.345 versus 0.329, tuned run), so
  the gain comes from where the flags land and not from extra process noise.

### E2: data selection with slice attribution (logistic, p=500, S=20 signals, fresh n = 150 / 300 / 600)

Excess KL; `attr_partial` is ours; `oracle_F` knows the true conflicting features.

| Scenario | n_fresh | fresh_only | pool_all | loss_filter | slice_discard | attr_partial | oracle_F |
|---|---|---|---|---|---|---|---|
| 5 flipped, clean | 150 | 0.152 | 0.169 | 0.161 | 0.149 | **0.095** | 0.059 |
| 5 flipped, clean | 300 | 0.135 | 0.162 | 0.149 | 0.116 | **0.080** | 0.055 |
| 5 flipped, clean | 600 | 0.114 | 0.150 | 0.129 | 0.086 | **0.073** | 0.052 |
| 5 flipped + noisy slice | 150 | 0.152 | 0.138 | 0.118 | 0.149 | **0.102** | 0.084 |
| 5 flipped + noisy slice | 300 | 0.135 | 0.136 | 0.111 | 0.116 | **0.089** | 0.081 |
| 5 flipped + noisy slice | 600 | 0.115 | 0.129 | 0.097 | 0.087 | **0.083** | 0.075 |
| no flip + noisy slice | 150 | 0.149 | **0.065** | 0.078 | 0.141 | 0.077 | 0.065 |
| no flip + noisy slice | 300 | 0.135 | **0.063** | 0.070 | 0.121 | 0.068 | 0.063 |
| no flip + noisy slice | 600 | 0.115 | 0.061 | 0.062 | 0.084 | **0.056** | 0.061 |

Reading:

- When some coordinates conflict, `attr_partial` is best among non-oracle methods in every row. Versus the best
  baseline in the clean scenario the excess KL falls by 36% (n=150), 31% (n=300) and 15% (n=600).
- Pure sample-level `loss_filter` handles the noisy slice reasonably but cannot repair conflicting coordinates.
- With no true conflict, `attr_partial` loses to `pool_all` at n=150 and n=300 (0.077 versus 0.065, 0.068 versus
  0.063). The gate produces false conflicts when fresh data is scarce relative to p. At n=600 the ordering flips.
- `slice_discard` is weak because the reference model's sign errors flag even clean slices.
- The gap between `attr_partial` and `oracle` in noisy scenarios (0.102 versus 0.055) is mostly the missing
  noise-weighting, not the localisation (`oracle_F` closes only part of it).

### E3: bigram LM toy (V=40, 6 drifted contexts, noisy scraped pool, fresh pairs 500 / 2000 / 8000)

Stationary-weighted KL to the true transition matrix.

| Scenario | n_fresh | fresh_only | pool_all | token_filter | ctx_attr | oracle |
|---|---|---|---|---|---|---|
| 6 drifted contexts | 500 | 0.646 | 0.178 | 0.182 | **0.143** | 0.029 |
| 6 drifted contexts | 2000 | 0.280 | 0.171 | 0.178 | **0.062** | 0.026 |
| 6 drifted contexts | 8000 | 0.085 | 0.147 | 0.134 | **0.019** | 0.019 |
| no drift | 500 | 0.645 | **0.039** | 0.044 | 0.039 | 0.013 |
| no drift | 2000 | 0.279 | **0.038** | 0.047 | 0.038 | 0.013 |
| no drift | 8000 | 0.085 | 0.035 | 0.037 | **0.011** | 0.012 |

Reading:

- Context-level localisation is much better than token-level loss filtering. Dropping high-loss tokens biases the
  model against rare tokens, so `token_filter` is never better than `pool_all` at its best setting here.
- Under no drift, `ctx_attr` ties `pool_all` at n=500 and 2000 (it does not flag anything) and is better at n=8000.
- This is a bigram toy. It does not say anything about LLM fine-tuning dynamics.

## 3. Caveats

1. Toy worlds with known ground truth, chosen to have sparse and localised drift. Dense or gradual drift is a
   failure region (E1 gradual) and real logs may mostly look like that.
2. Gates are CUSUM / z-score thresholds. The online FDR procedures in OnlineRFPerm (SAFFRON / ADDIS / LOND) are
   not implemented, so there is no formal error control. A fixed false-positive rate per check is not guaranteed.
3. "PO-risk" is read as permutation-OOB risk because the OnlineRFPerm README does not define it.
4. The importance increment uses a second-order expected-permutation approximation and assumes standardised
   features. It is not valid as-is for trees, embeddings or sparse categorical features.
5. Small-sample gating (E2, n_fresh=150, p=500) produces false conflicts and can lose to doing nothing.
6. Standard errors ignore the variance from tuning; E1 tuning also selected hyperparameters per scenario in
   `recsys_persample.txt`.
7. Retention is worse under attribution-guided loosening. If the old task is still evaluated (continual learning
   rather than drift), the method needs a retention constraint.
8. Only the OnlineRFPerm logic was ported (reference model, per-batch loss, p-value gating, truncation idea). Its
   code was not reused and no claim is made about its correctness.

## 4. Technical roadmap

Phase 0 (done here): toy worlds with ground truth for the three scenarios; null and noisy controls; random-K
control; fixed-configuration check.

Phase 1: make the gate statistically honest.
- Replace thresholds with an online FDR procedure on slice/feature p-values, and add a power-versus-FDR curve.
- Calibrate the importance-drop statistic by permutation under the null on holdout data; sweep p/n to find the
  regime where gating beats `pool_all`.
- Pre-register: the attribution gain should vanish when drift is dense; report that as a pass, not a failure.

Phase 2: recommendation logs.
- Replay a public dataset with timestamped feedback (candidate: MovieLens, Criteo-style CTR logs) for a
  logistic / factorisation model; per-sample update with the diag-EKF step.
- Slices = item category, surface, user cohort. Metrics: online AUC, calibration, NDCG, plus retention on
  old-cohort holdout.
- Compare against sample-weighting (RHO-LOSS style), fixed forgetting windows, EWC, ADWIN retraining.

Phase 3: data selection / procurement.
- Slices become data sources. Output both the reuse decision and the per-source conflicting features, which
  feeds the data-valuation estimands (MTV, AV, RV) from the earlier toy world.
- Add winner's-curse correction when the source is chosen by its own attribution score.

Phase 4: generative models.
- Move from bigrams to a small transformer with per-module (or per-LoRA-subspace) attribution, with contexts
  replaced by domain/topic slices. Test whether localised slice-level drop/down-weight beats token-level
  selection on a held-out fresh domain, with a forgetting metric on the old domain.
- Only then consider an LLM fine-tuning pilot; that is a separate experiment with its own pre-registration.

Phase 5: write-up.
- One paper section on the gate-localise-select-update-validate pipeline, with the failure regions
  (dense drift, scarce fresh data, retention) stated up front.
