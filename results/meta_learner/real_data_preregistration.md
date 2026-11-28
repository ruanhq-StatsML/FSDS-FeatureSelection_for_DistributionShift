# Real-data pre-registration (written before any real-data run)

Data: SWE-bench Verified (500 instances, Hugging Face `princeton-nlp/SWE-bench_Verified`) and per-instance
outcomes of 135 leaderboard submissions (`SWE-bench/experiments`, `evaluation/verified/*/results/results.json`).
Instance label for a submission: instance id is in `resolved`.

## R1: online forecasting stream
Stream = (submission, instance) pairs, submissions in chronological order, instances in random order within a
submission. Features = instance-level only (repo one-hot, patch / test statistics, TF-IDF of the problem
statement). Prequential prediction of "resolved". Drift = model capability improving over time (resolve rate
0.04 to about 0.79), with repo-specific and difficulty-specific effects changing.

Predictions, based on the synthetic results (gradual and dense drift gave no gain):
1. All adaptive methods beat static on log loss by a large margin, mostly through the intercept.
2. `attr` is within +-3% of plain `ekf` on log loss and AUC; the difference is not significant over bootstrap
   resamples of submissions.
3. `attr` flags concentrate on repo indicators and size features; flags per check are non-zero in most checks.
4. Tuning is done only on submissions 20-59 of the stream and reporting on submissions 60 onwards.

## R2: leave-one-repo-out data selection
Target repo = held-out repo; its instances are split in half by instance. A small fresh sample of target pairs
(n = 150 / 600) comes from the first half. Pools are the other repos' pairs (one slice per repo). Evaluation on
the second half of target instances (instance-disjoint). Methods as in E2 (fresh_only, pool_all, loss_filter,
slice_discard, attr_partial).

Predictions:
1. `pool_all` beats `fresh_only` for small n.
2. `attr_partial` beats `pool_all` in a minority of repos (probably the ones with distinct difficulty
   structure, e.g. sympy / sphinx), and loses in others because of false conflicts at n=150. Average gain over
   repos is small (under 3%) and may be negative.

## R3: classical drift benchmarks (sanity check)
Electricity and Covertype streams (OpenML) with the same per-sample learners. p is 8 and 54, so there is little
sparse structure to localise. Prediction: no gain of `attr` over `ekf`.

## Reporting rules
Report all results, including losses. Negative results stay in the README.
