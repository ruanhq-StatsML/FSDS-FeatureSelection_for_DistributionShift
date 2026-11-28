# Attribution-guided online updating (CTR-style stream, sparse concept drift)

Script: `python/attr_guided_online.py [n_seeds] [batch_size]`. Raw output: `batch400.txt`, `batch1000.txt`, `batch2000.txt` (12 seeds each).

## Setup
- Logistic click model, p = 1000 features (factor structure, var 1), 20 signal coefficients of +-0.45, 60 rounds of M impressions, 8 warm-up rounds, prequential evaluation (predict on the new batch, then update).
- `abrupt`: every 12 rounds 4 of the 20 signal coefficients flip sign. `gradual`: 4 coefficients random-walk (sd 0.1 per round). `null`: no drift.
- Metrics: excess log loss over the Bayes (true-logit) loss, and AUC, averaged over rounds after warm-up and over seeds. Brackets in the raw files are s.e. across seeds.
- Updates: `uniform` = proximal step from the current model on the newest batch with one ridge anchor lambda for all features; `scratch` = sliding-window (4 batches) ridge refit from zero; `attr` = same proximal step, but the anchor is loose (lambda_lo) on the K features with the largest attribution scores and tight (lambda_hi) elsewhere; `random` = same with K random features (control); `oracle` = loose on the truly drifting features; `static` = no update.
- Attribution score of feature j: permutation risk increase of the current model on the new batch, compared with an exponentially weighted reference of the same quantity from earlier batches (z-score, reference frozen for features with z > 3). This is a loss-contrast permutation attribution under the fitted model, a simplified stand-in for PermuCATE (no random forest, no doubly robust pseudo-outcome).
- lambda, (lambda_lo, lambda_hi, K) tuned on separate seeds 100-103 (minimum excess log loss) for every scenario and batch size. `random` and `oracle` reuse the tuned (lambda_lo, lambda_hi) of `attr`.

## Results (excess log loss / AUC; lower / higher is better)

| scenario | M | static | scratch | uniform | random-K | attr-guided | oracle-guided |
|---|---|---|---|---|---|---|---|
| abrupt | 400 | 0.685 / 0.585 | 0.127 / 0.760 | 0.179 / 0.767 | 0.194 / 0.762 | 0.174 / 0.778 | 0.128 / 0.800 |
| abrupt | 1000 | 0.550 / 0.590 | 0.101 / 0.780 | 0.114 / 0.796 | 0.121 / 0.793 | 0.107 / 0.805 | 0.089 / 0.814 |
| abrupt | 2000 | 0.435 / 0.625 | 0.082 / 0.797 | 0.092 / 0.808 | 0.111 / 0.784 | 0.075 / 0.811 | 0.040 / 0.835 |
| gradual | 400 | 0.279 / 0.768 | 0.123 / 0.803 | 0.101 / 0.842 | 0.115 / 0.838 | 0.129 / 0.832 | 0.094 / 0.850 |
| gradual | 1000 | 0.131 / 0.804 | 0.076 / 0.819 | 0.056 / 0.841 | 0.072 / 0.838 | 0.078 / 0.836 | 0.063 / 0.843 |
| gradual | 2000 | 0.113 / 0.816 | 0.053 / 0.848 | 0.029 / 0.865 | 0.038 / 0.860 | 0.046 / 0.856 | 0.017 / 0.872 |
| null | 400 | 0.195 / 0.785 | 0.101 / 0.784 | 0.095 / 0.822 | 0.109 / 0.817 | 0.127 / 0.810 | 0.095 / 0.822 |
| null | 1000 | 0.074 / 0.827 | 0.069 / 0.813 | 0.037 / 0.844 | 0.068 / 0.832 | 0.078 / 0.828 | 0.061 / 0.835 |
| null | 2000 | 0.035 / 0.845 | 0.045 / 0.833 | 0.014 / 0.856 | 0.025 / 0.850 | 0.043 / 0.841 | 0.014 / 0.856 |

Top-K attribution quality in `abrupt` (rounds with at least one recently drifted feature): recall of the truly drifting features 0.50 / 0.42 / 0.31 and precision 0.25 / 0.21 / 0.15 at M = 400 / 1000 / 2000. In `gradual`: recall 0.06 / 0.29 / 0.27.

## Reading
- Abrupt sparse drift: attribution-guided updating beats uniform updating in AUC at all three batch sizes (+0.011, +0.009, +0.003) and beats the random-K control (+0.016, +0.012, +0.027). It also lowers log loss (0.174 vs 0.179, 0.107 vs 0.114, 0.075 vs 0.092). The gains are small and recover only part of the oracle's: about 33% / 50% / 11% of the AUC gap and 10% / 28% / 33% of the log-loss gap between uniform and oracle.
- Gradual drift and no drift: attribution-guided updating is worse than plain uniform updating (for example null at M = 400: 0.127 vs 0.095). Freeing K coordinates every round costs variance when nothing (or nothing detectable) drifted. Recall for slow drift is low (0.06 at M = 400), because the reference lags a slowly moving signal.
- A plain sliding-window ridge refit has the best log loss in abrupt drift at M = 400 (0.127 vs 0.174 for attribution-guided; the oracle ties it), at a lower AUC. At M = 2000 the guided update has the lower log loss (0.075 vs 0.082).
- The oracle gap is large (AUC 0.80 vs 0.77 at M = 400; log loss 0.040 vs 0.092 at M = 2000), so the potential of guidance is real; the binding constraint is attribution accuracy from one online batch. Recall did not improve with batch size here; I have not diagnosed why.

## Caveats
- Synthetic logistic DGP with factor structure; linear model, so the attribution is a permutation loss contrast and not the PermuCATE estimator used in the main benchmark.
- K (number of features freed) is a tuned budget, which assumes some prior knowledge of the drift sparsity.
- 12 evaluation seeds per cell; tuning seeds are separate from evaluation seeds. The design itself was iterated on while looking at small runs (update on newest batch only, top-K instead of a z threshold, reference decay 0.95 vs 0.8 chosen after seeing a 6-seed run), so mild forking-paths bias is possible. Earlier variants with a z-threshold gate (3-9 false flags per round) did worse than uniform updating and were dropped.
- `batch*.txt` were produced by the 113-line version of the script; the 100-line version in the repo was checked to give identical per-run numbers for five (scenario, method) pairs on seed 0.
