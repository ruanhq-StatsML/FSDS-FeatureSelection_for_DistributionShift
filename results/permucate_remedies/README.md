# PO-risk / PermuCATE: other outcome models vs RF and MLP

Setup: pure concept drift, |S| = 10 shifted features, `python/permucate_nuisance_remedies.py`, 5 replicates per cell,
numbers in `summary.md` (AUC, AP, nuisance R^2, paired differences, share of oracle excess AUC lost).
Variants: T-learner RF / MLP (the earlier baselines); R-learner with one pooled outcome model (`rl_lasso`, `rl_rf`,
`rl_mlp`); partial-linear stack (`pl_lasso_rf`: pooled lasso + arm RF on residuals); lasso-screened RF (`scr_rf`);
`oracle` = true mu0, mu1.

## Findings (honest reading)

* RF and MLP are statistically indistinguishable (paired AUC difference MLP-RF: -0.085 / -0.010 / 0.000 at
  p/n = 20 / 100 / 200, s.e. 0.02-0.06).
* No alternative beats RF consistently. At p/n = 100 the pooled R-learners help (`rl_lasso` +0.057, s.e. 0.017;
  `rl_rf` +0.047, s.e. 0.022; n = 5, 7 variants x 3 settings, not multiplicity-corrected). The sign flips at
  p/n = 20 and 200 (-0.04 to -0.06 at p/n = 200 for `pl_lasso_rf`). Treat the p/n = 100 gain as a hint, not a result.
* Nuisance R^2 stays 0.04-0.18 for every learnable model. The oracle is far ahead (AUC 0.79 vs <= 0.65 at p/n = 100);
  60-70% of the oracle's excess AUC is lost to nuisance estimation, and swapping the learner family barely moves that.
* The bottleneck is signal-to-noise of the outcome regression in p >> n, not the function class.

## What to do about it (only the first two are supported by measured evidence in this repo)

1. Reduce p before the PO stage (screening): see `results/screening_fsds/README.md`; that moves AUC from 0.58 to 0.75
   at p = 10000 in a different DGP.
2. Factor augmentation (`fa_*` variants, `results/permucate_mlp_fa`).
3. Untested: borrow strength from a larger old batch (an n_old = 2000 run was started and aborted unfinished; no claim
   is made), and a pooled R-learner when the stable part is dense.
