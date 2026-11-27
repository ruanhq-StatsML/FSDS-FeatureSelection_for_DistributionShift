"""How well can a well-specified sparse linear estimator learn mu0 = X beta at the cross-fitting sample size?

Fits LassoCV on the first n_arm rows of the old batch (n_arm = 125: one cross-fit fold of one arm; 250: a full arm of
the training half) and reports out-of-sample R^2 against the true mu0 on held-out rows of the same batch.
"""
import warnings

import numpy as np
from sklearn.linear_model import LassoCV

import benchmark_permucate_mlp as m

warnings.filterwarnings("ignore")
for r in [20, 100]:
    res = {125: [], 250: []}
    for rep in range(3):
        X1, Y1, _, _, _, truth = m.make_linear_complex_pair(500, r * 500, 7 + 100 * rep + r, s=10, b=100, shift_var=8)
        Xt = X1[250:]
        mu = truth["mu0"](Xt)
        for n_arm in res:
            fit = LassoCV(cv=5, n_jobs=4, max_iter=2000).fit(X1[:n_arm], Y1[:n_arm])
            res[n_arm].append(1 - np.mean((fit.predict(Xt) - mu) ** 2) / mu.var())
    print(f"p/n={r}: LassoCV R^2 vs true mu0 by n_arm: " + ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in res.items()))
