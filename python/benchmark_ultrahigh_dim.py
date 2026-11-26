"""
Ultra-high-dimensional (p / n >= 20) benchmark of feature-selection methods for distribution shift,
targeted at high-dimensional financial-factor / characteristic panels.

Setting
-------
n_exist = n_new = 500 (--n), p = (p/n) * n  ->  p/n = 20 (p = 10,000) and p/n = 100 (p = 50,000).
Outcome model everywhere an outcome / nuisance regression is needed: random forest.
Grid: np.linspace(0, 0.95, 20) over `rho`, the share of each characteristic's variance that is driven
by common latent factors (factor strength).

Messy DGP (`make_messy_pair`)
-----------------------------
* X: approximate factor model X = sqrt(rho) F L' + sqrt(1 - rho) E (K latent factors), then 30% of the
  columns get a random skewed / heavy-tailed / non-monotone marginal transform (exp, cubic, tanh, signed-sqrt).
* 10% of the features (`--frac_signal`) are *signal* features. The outcome mechanism is a random mix of
  linear / sin / tanh / ReLU / quadratic / signed-sqrt terms plus pairwise interactions among signal features,
  scaled to a total SNR of `--snr` (default 1, R^2 = 0.5), with heavy-tailed (t5) noise.
* The same 10% signal features are the *shifted* features (ground truth for selection). In the new batch
  they get (i) a nonlinear concept shift (sin + quadratic + interaction, total variance `--shift_ratio` x
  signal variance, spread over all shifted features), and (ii) for a random half of them, a covariate shift
  X_j += `--cov_gamma` * z_j^2 (same form as `synthetic_injection_nonlinear_interaction`).

Methods (compact set, `--methods` for more)
-------------------------------------------
rf_domain, permucate, loco_r_risk, sgshift_l1_spec, sgshift_l1_misspec, datafix, fslnet.
PermuCATE and LOCO-R-risk are computed for **all** p features:
* PermuCATE (n_perm = 25, RF outcome / propensity / tau models, 50/50 train-test split, DR pseudo-outcome
  risk). Exact shortcut: a feature that no tree of the tau forest splits on has importance exactly 0, and
  a permuted feature only changes the trees that split on it, so only those trees are re-evaluated.
  E[X_j | X_-j] is estimated by ridge on the `--n_neighbors` most correlated features (not a forest on
  the other p - 1 features).
* LOCO-R-risk: RF outcome model m(x) = E[Y|X] and RF propensity (2-fold cross-fitted once with all
  features), ridge R-learner final stage. The final stage is refit without feature j for every j through
  rank-one updates of the kernel (Gram) matrix; the RF nuisances are *not* refit per feature (that would
  be p x 4 forest fits).
* SGShift L1: same objective as SGShift.py (offset from a source RF, L1 penalty on delta, BIC over a
  lambda path) solved by coordinate descent instead of SCS.
* DataFix (DF-Locate) and FSL-Net are the upstream packages (AI-sandbox/DataFix, AI-sandbox/FSL-Net).
Not run: knockoffs (Gaussian knockoffs are infeasible at p >> n).

Usage
-----
python benchmark_ultrahigh_dim.py --p_over_n 20  --out ../results/ultrahigh_dim_p20n
python benchmark_ultrahigh_dim.py --p_over_n 100 --datafix_every 9 --out ../results/ultrahigh_dim_p100n
# pure concept drift (no covariate shift), quant alpha-decay formulation, 5 grid points
python benchmark_ultrahigh_dim.py --dgp quant_concept --p_over_n 20 --shift_ratio 2 --grid_index 0,5,10,15,19 \
    --out ../results/quant_concept_p20n
"""
import os
import sys
import copy
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import lasso_path
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")
N_JOBS = max(1, os.cpu_count() or 1)

# ----------------------------------------------------------------------------------------------
# Messy data generating process
# ----------------------------------------------------------------------------------------------


def _g(kind, z):
    if kind == 0:
        return z
    if kind == 1:
        return np.sin(1.5 * z)
    if kind == 2:
        return np.tanh(2.0 * z)
    if kind == 3:
        return np.maximum(z, 0.0) - 0.4
    if kind == 4:
        return z ** 2 - 1.0
    return np.sign(z) * np.sqrt(np.abs(z))


def _column_transforms(X, kinds):
    """Skewed / heavy-tailed / non-monotone marginals on the columns with kinds >= 0."""
    for k, fn in enumerate([lambda x: np.exp(0.6 * x), lambda x: x ** 3, lambda x: np.tanh(1.5 * x),
                            lambda x: np.sign(x) * np.sqrt(np.abs(x))]):
        idx = np.flatnonzero(kinds == k)
        if idx.size:
            X[:, idx] = fn(X[:, idx])
    return X


def make_messy_pair(n, p, rho, seed, frac_signal=0.1, snr=1.0, shift_ratio=1.0, cov_gamma=0.3,
                    n_factors=10, transform_frac=0.3):
    """Returns X1, Y1, X2, Y2, shifted (sorted indices of the shifted = signal features)."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((p, n_factors))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    kinds = np.where(rng.uniform(size=p) < transform_frac, rng.integers(0, 4, size=p), -1)

    def draw():
        F = rng.standard_normal((n, n_factors))
        X = np.sqrt(rho) * (F @ L.T) + np.sqrt(1.0 - rho) * rng.standard_normal((n, p))
        return _column_transforms(X, kinds)

    X1, X2 = draw(), draw()
    pooled = np.vstack([X1, X2])
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-8
    del pooled
    X1, X2 = (X1 - mu) / sd, (X2 - mu) / sd

    s = max(2, int(round(frac_signal * p)))
    S = np.sort(rng.choice(p, size=s, replace=False))
    kind = rng.integers(0, 6, size=s)
    amp = rng.uniform(0.5, 1.5, size=s) * rng.choice([-1.0, 1.0], size=s)
    perm = rng.permutation(S)
    pairs = perm[: 2 * (s // 2)].reshape(-1, 2)
    pair_amp = rng.uniform(0.3, 1.0, size=len(pairs)) * rng.choice([-1.0, 1.0], size=len(pairs))
    partner = rng.choice(S, size=s)
    c1, c2, c3 = rng.uniform(0.5, 1.0, s), rng.uniform(0.25, 0.5, s), rng.uniform(0.2, 0.35, s)
    cov_set = S[rng.uniform(size=s) < 0.5]

    def signal(X):
        f = np.zeros(len(X))
        for a, k, j in zip(amp, kind, S):
            f += a * _g(k, X[:, j])
        f += (pair_amp * X[:, pairs[:, 0]] * X[:, pairs[:, 1]]).sum(1)
        return f

    def concept_delta(X):
        d = np.zeros(len(X))
        for a, b, c, j, m in zip(c1, c2, c3, S, partner):
            z = X[:, j]
            d += a * np.sin(z) + b * (z ** 2 - 1.0) + c * z * X[:, m]
        return d

    f1 = signal(X1)
    scale = np.sqrt(snr / (f1.var() + 1e-12))
    X2[:, cov_set] += cov_gamma * X2[:, cov_set] ** 2
    d2 = concept_delta(X2)
    d_scale = np.sqrt(shift_ratio * snr / (d2.var() + 1e-12))
    noise = lambda m: rng.standard_t(5, size=m) / np.sqrt(5.0 / 3.0)  # noqa: E731
    Y1 = scale * f1 + noise(n)
    Y2 = scale * signal(X2) + d_scale * d2 + noise(n)
    return X1, Y1, X2, Y2, S


def make_quant_concept_pair(n, p, rho, seed, frac_signal=0.1, snr=1.0, shift_ratio=1.0, frac_drift=0.5,
                            n_factors=10, transform_frac=0.3, **_):
    """Pure concept drift ("alpha decay / regime change") on a factor-characteristic panel.

    X1 (estimation window) and X2 (new regime) are i.i.d. draws from the SAME characteristic distribution
    (approximate factor model + skewed marginals), so there is no covariate shift: P(X) is unchanged and
    only E[Y | X] moves. Y is the next-period excess return of a stock; the 10% alpha characteristics S drive it
    (random linear / saturating / convex / non-monotone terms + pairwise interactions, t5 noise, SNR = `snr`).
    A random `frac_drift` of the alphas D (the ground truth, 5% of p) change their return mechanism in the new
    regime, each drawing one of: alpha decay (loading x 0.2, crowded trade), sign flip (momentum crash /
    reversal), or shape change (e.g. linear -> convex / non-monotone payoff). The remaining alphas are
    stable. The size of the drift is calibrated so that Var(drift component of Y) = shift_ratio * snr.
    Returns X1, Y1, X2, Y2, D."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((p, n_factors))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    kinds = np.where(rng.uniform(size=p) < transform_frac, rng.integers(0, 4, size=p), -1)

    def draw():
        F = rng.standard_normal((n, n_factors))
        X = np.sqrt(rho) * (F @ L.T) + np.sqrt(1.0 - rho) * rng.standard_normal((n, p))
        return _column_transforms(X, kinds)

    X1, X2 = draw(), draw()
    pooled = np.vstack([X1, X2])
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-8
    del pooled
    X1, X2 = (X1 - mu) / sd, (X2 - mu) / sd

    s = max(2, int(round(frac_signal * p)))
    S = np.sort(rng.choice(p, size=s, replace=False))
    kind = rng.integers(0, 6, size=s)
    amp = rng.uniform(0.5, 1.5, size=s) * rng.choice([-1.0, 1.0], size=s)
    pos = rng.permutation(s)
    pairs_pos = pos[: 2 * (s // 2)].reshape(-1, 2)
    pair_amp = rng.uniform(0.3, 1.0, size=len(pairs_pos)) * rng.choice([-1.0, 1.0], size=len(pairs_pos))

    drift_pos = np.sort(rng.choice(s, size=max(1, int(round(frac_drift * s))), replace=False))
    dtype = rng.integers(0, 3, size=len(drift_pos))
    amp2, kind2, mult = amp.copy(), kind.copy(), np.ones(s)
    for pi, t in zip(drift_pos, dtype):
        if t == 0:
            amp2[pi], mult[pi] = 0.2 * amp[pi], 0.2
        elif t == 1:
            amp2[pi], mult[pi] = -amp[pi], -1.0
        else:
            kind2[pi] = rng.choice([k for k in range(6) if k != kind[pi]])
    m0, m1 = mult[pairs_pos[:, 0]], mult[pairs_pos[:, 1]]
    pair_mult = np.where(m0 != 1.0, m0, m1)

    def signal(X, a, k, pm):
        f = np.zeros(len(X))
        for ai, ki, j in zip(a, k, S):
            f += ai * _g(ki, X[:, j])
        f += (pair_amp * pm * X[:, S[pairs_pos[:, 0]]] * X[:, S[pairs_pos[:, 1]]]).sum(1)
        return f

    f1 = signal(X1, amp, kind, 1.0)
    scale = np.sqrt(snr / (f1.var() + 1e-12))
    f2_old = signal(X2, amp, kind, 1.0)
    f2_new = signal(X2, amp2, kind2, pair_mult)
    gap = f2_new - f2_old
    lam = np.sqrt(shift_ratio * snr / (scale ** 2 * gap.var() + 1e-12))
    noise = lambda m: rng.standard_t(5, size=m) / np.sqrt(5.0 / 3.0)  # noqa: E731
    Y1 = scale * f1 + noise(n)
    Y2 = scale * (f2_old + lam * gap) + noise(n)
    return X1, Y1, X2, Y2, S[drift_pos]


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------


def metrics_from_scores(scores, shifted, p):
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf, posinf=np.finfo(float).max)
    k = len(shifted)
    if k == 0:
        return {"k": 0}
    order = np.argsort(-scores, kind="stable")
    tp = len(set(order[:k].tolist()) & set(shifted.tolist()))
    precision = tp / k
    member = np.zeros(p, dtype=int)
    member[shifted] = 1
    finite = scores[np.isfinite(scores)]
    floor = (finite.min() - 1.0) if finite.size else 0.0
    s = np.where(np.isfinite(scores), scores, floor)
    return {
        "k": k,
        "TP": tp,
        "precision": precision,
        "recall": precision,
        "F1": precision,
        "FDR": 1.0 - precision,
        "lift": precision / (k / p),
        "recall@2k": len(set(order[: 2 * k].tolist()) & set(shifted.tolist())) / k,
        "AUC": float(roc_auc_score(member, s)),
        "AP": float(average_precision_score(member, s)),
    }


# ----------------------------------------------------------------------------------------------
# RF domain classifier family
# ----------------------------------------------------------------------------------------------


def rf_clf(seed, n_estimators=150, n_jobs=N_JOBS, n=1000):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=max(1, int(round(np.sqrt(n) / 2))),
        n_jobs=n_jobs, random_state=seed,
    )


def rf_reg(seed, n_estimators=100, n_jobs=N_JOBS):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=5, n_jobs=n_jobs, random_state=seed
    )


def rf_domain_importance(X, W, seed, n_estimators=300):
    return rf_clf(seed, n_estimators=n_estimators, n=len(X)).fit(X, W).feature_importances_


def rfdom_shadow(X, W, seed):
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    shadow = np.apply_along_axis(rng.permutation, 0, X)
    imp = rf_clf(seed, 300, n=len(X)).fit(np.hstack([X, shadow]), W).feature_importances_
    return imp[:p] - imp[p:].max()


def rfdom_pimp(X, W, seed, n_null=30):
    base = rf_clf(seed, 100, n=len(X)).fit(X, W).feature_importances_
    rng = np.random.default_rng(seed + 7)
    null = np.empty((n_null, X.shape[1]))
    for r in range(n_null):
        null[r] = rf_clf(seed + 11 + r, 100, n=len(X)).fit(X, rng.permutation(W)).feature_importances_
    return (base - null.mean(0)) / (null.std(0) + 1e-12)


def rfdom_boruta(X, W, seed, n_iter=15):
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    hits, cum = np.zeros(p), np.zeros(p)
    for it in range(n_iter):
        shadow = np.apply_along_axis(rng.permutation, 0, X)
        imp = rf_clf(seed + it, 100, n=len(X)).fit(np.hstack([X, shadow]), W).feature_importances_
        hits += imp[:p] > imp[p:].max()
        cum += imp[:p]
    return hits + cum / (cum.max() + 1e-12) * 1e-3


# ----------------------------------------------------------------------------------------------
# PermuCATE (all features, exact tree shortcut)
# ----------------------------------------------------------------------------------------------


def permucate_fit(X, Y, W, seed, n_estimators, clip_e=0.01):
    """50/50 stratified split; RF outcome models per arm, RF propensity, RF tau on the DR pseudo-outcome."""
    Xtr, Xte, Wtr, Wte, Ytr, Yte = train_test_split(X, W, Y, test_size=0.5, stratify=W, random_state=seed)
    mu0 = rf_reg(seed, n_estimators).fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
    mu1 = rf_reg(seed + 1, n_estimators).fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
    ps = rf_clf(seed + 2, n_estimators, n=len(Xtr)).fit(Xtr, Wtr)
    m0, m1 = mu0.predict(Xtr), mu1.predict(Xtr)
    pi = np.clip(ps.predict_proba(Xtr)[:, 1], clip_e, 1 - clip_e)
    pseudo = m1 - m0 + (Ytr - np.where(Wtr == 1, m1, m0)) * (Wtr - pi) / (pi * (1 - pi))
    tau = rf_reg(seed + 3, n_estimators).fit(Xtr, pseudo)
    return dict(Xtr=np.ascontiguousarray(Xtr, dtype=np.float32), Xte=np.ascontiguousarray(Xte, dtype=np.float32),
                Wte=Wte, Yte=Yte, mu0=mu0, mu1=mu1, ps=ps, tau=tau, clip_e=clip_e)


def top_neighbors(Xtr, targets, k, block=400):
    """For each target feature the k most correlated other features (train split)."""
    Z = ((Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-8)).astype(np.float32)
    n = Z.shape[0]
    out = np.empty((len(targets), k), dtype=np.int64)
    for s in range(0, len(targets), block):
        t = targets[s:s + block]
        C = np.abs(Z[:, t].T @ Z) / n
        C[np.arange(len(t)), t] = -1.0
        out[s:s + block] = np.argpartition(-C, k, axis=1)[:, :k]
    return out


def permucate_vimp(fit, seed, n_perm, n_neighbors):
    Xte, Xtr, Wte, Yte = fit["Xte"], fit["Xtr"], fit["Wte"], fit["Yte"]
    p = Xte.shape[1]
    clip_e = fit["clip_e"]
    mu0, mu1 = fit["mu0"].predict(Xte), fit["mu1"].predict(Xte)
    pi = np.clip(fit["ps"].predict_proba(Xte)[:, 1], clip_e, 1 - clip_e)
    pseudo = (Yte - np.where(Wte == 1, mu1, mu0)) * (Wte - pi) / (pi * (1 - pi)) + (mu1 - mu0)
    trees = [e.tree_ for e in fit["tau"].estimators_]
    T = len(trees)
    P = np.stack([t.predict(Xte)[:, 0] for t in trees])
    tot = P.sum(0)
    base = float(np.mean((pseudo - tot / T) ** 2))
    feat_trees = {}
    for t_idx, t in enumerate(trees):
        for j in np.unique(t.feature[t.feature >= 0]):
            feat_trees.setdefault(int(j), []).append(t_idx)
    used = np.array(sorted(feat_trees), dtype=np.int64)
    nbrs = top_neighbors(Xtr, used, n_neighbors)
    mu_tr = Xtr.mean(0)

    def worker(chunk):
        Xw = Xte.copy()
        res = []
        for ci in chunk:
            j = int(used[ci])
            nb = nbrs[ci]
            A = Xtr[:, nb] - mu_tr[nb]
            beta = np.linalg.solve(A.T @ A + np.eye(len(nb)), A.T @ (Xtr[:, j] - mu_tr[j]))
            nu = mu_tr[j] + (Xte[:, nb] - mu_tr[nb]) @ beta
            r = Xte[:, j] - nu
            rng = np.random.default_rng(seed * 104729 + j)
            ts = feat_trees[j]
            old = P[ts].sum(0)
            vals = np.empty(n_perm)
            for k in range(n_perm):
                Xw[:, j] = nu + rng.permutation(r)
                new = sum(trees[t].predict(Xw)[:, 0] for t in ts)
                vals[k] = np.mean((pseudo - (tot - old + new) / T) ** 2) - base
            Xw[:, j] = Xte[:, j]
            res.append(vals.mean())
        return res

    chunks = np.array_split(np.arange(len(used)), N_JOBS * 8)
    out = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(worker)(c) for c in chunks if len(c))
    v = np.zeros(p)
    v[used] = np.concatenate([np.asarray(o) for o in out])
    return v, len(used)


# ----------------------------------------------------------------------------------------------
# LOCO-R-risk (all features): RF nuisances once, ridge R-learner refit without feature j
# ----------------------------------------------------------------------------------------------


def loco_r_risk(X, Y, W, seed, n_estimators, alpha=1.0, n_folds=2, clip_e=0.01):
    n, p = X.shape
    X = np.asarray(X, dtype=np.float32)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(X))
    m_hat, e_hat = np.zeros(n), np.zeros(n)
    for k, (tr, te) in enumerate(folds):
        m_hat[te] = rf_reg(seed + k, n_estimators).fit(X[tr], Y[tr]).predict(X[te])
        e_hat[te] = rf_clf(seed + 5 + k, n_estimators, n=len(tr)).fit(X[tr], W[tr]).predict_proba(X[te])[:, 1]
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    Yt, Wt = Y - m_hat, W - e_hat

    prep = []
    for tr, te in folds:
        keep = np.abs(Wt[tr]) > 1e-3
        tr = tr[keep]
        w = Wt[tr] ** 2
        z = Yt[tr] / Wt[tr]
        sd = X[tr].std(0) + 1e-8
        mw = (w[:, None] * X[tr]).sum(0) / w.sum()
        Atr = ((X[tr] - mw) / sd).astype(np.float32)
        Ate = ((X[te] - mw) / sd).astype(np.float32)
        sw = np.sqrt(w)
        zbar = float((w * z).sum() / w.sum())
        A = sw[:, None].astype(np.float32) * Atr
        prep.append(dict(Atr=Atr, Ate=Ate, A=A, G=(A @ A.T).astype(np.float64),
                         C=(Ate @ Atr.T).astype(np.float64), sw=sw, y=sw * (z - zbar), zbar=zbar, te=te))

    def risk(j=None):
        total = 0.0
        for d in prep:
            G, C = d["G"], d["C"]
            if j is not None:
                a = d["A"][:, j].astype(np.float64)
                G = G - np.outer(a, a)
                C = C - np.outer(d["Ate"][:, j], d["Atr"][:, j])
            dual = cho_solve(cho_factor(G + alpha * np.eye(len(G))), d["y"])
            tau = C @ (d["sw"] * dual) + d["zbar"]
            te = d["te"]
            total += np.mean((Yt[te] - tau * Wt[te]) ** 2)
        return total / len(prep)

    base = risk()

    def chunk(js):
        return [risk(j) - base for j in js]

    with threadpool_limits(limits=1):
        out = Parallel(n_jobs=N_JOBS, prefer="threads")(
            delayed(chunk)(c) for c in np.array_split(np.arange(p), N_JOBS * 16))
    return np.concatenate([np.asarray(o) for o in out])


# ----------------------------------------------------------------------------------------------
# SGShift L1 (specified / misspecified), coordinate descent + BIC over a lambda path
# ----------------------------------------------------------------------------------------------


def _bic_path_select(Xd, r, lambdas):
    """Lasso path with sklearn alpha = lambda / 2 (mean-squared loss + lambda * |.|_1)."""
    _, coefs, _ = lasso_path(Xd, r, alphas=np.asarray(lambdas) / 2.0, max_iter=2000, tol=1e-5)
    n = len(r)
    best, best_bic = None, np.inf
    for a in range(coefs.shape[1]):
        c = coefs[:, a]
        k = int(np.sum(np.abs(c) > 1e-4))
        if k == 0:
            continue
        rss = float(np.sum((r - Xd @ c) ** 2))
        bic = n * np.log(rss / n + 1e-12) + k * np.log(n)
        if bic < best_bic:
            best, best_bic = a, bic
    if best is None:
        best = coefs.shape[1] - 1
    return coefs, best


def sgshift_l1(X1, Y1, X2, Y2, seed, lambdas=None):
    if lambdas is None:
        lambdas = np.sort(np.logspace(-2.5, 0.3, 8))[::-1]
    src = rf_reg(seed, 100).fit(X1, Y1)
    f1, f2 = src.predict(X1), src.predict(X2)
    p = X1.shape[1]
    coefs, best = _bic_path_select(X2, Y2 - f2, lambdas)
    spec = np.abs(coefs[:, best]) + 1e-6 * np.abs(coefs).max(axis=1)
    Xs = np.zeros((len(X1) + len(X2), 2 * p), dtype=np.float32)
    Xs[: len(X1), :p] = X1 / 0.9
    Xs[len(X1):, :p] = X2 / 0.9
    Xs[len(X1):, p:] = X2
    coefs, best = _bic_path_select(Xs, np.concatenate([Y1 - f1, Y2 - f2]), lambdas)
    mis = np.abs(coefs[p:, best]) + 1e-6 * np.abs(coefs[p:]).max(axis=1)
    return spec, mis


# ----------------------------------------------------------------------------------------------
# DataFix (DF-Locate) and FSL-Net
# ----------------------------------------------------------------------------------------------


def datafix_locate(X1, X2, seed):
    from datafix import DFLocate

    n = len(X1) + len(X2)
    loc = DFLocate(
        estimator=RandomForestClassifier(
            max_depth=5, n_estimators=150, min_samples_leaf=max(1, round(np.sqrt(n) // 2)),
            max_features="sqrt", random_state=seed,
        ),
        cv=5, test_size=None, random_state=seed, verbose=False, n_jobs=N_JOBS,
    )
    loc.shift_location(X1, X2)
    return np.asarray(loc.importances_, dtype=float).ravel(), np.asarray(loc.mask_, dtype=int).ravel()


def fslnet_locate(X1, X2, chunk=5000):
    """FSL-Net is a 1-D convolutional net along the feature axis, so it is applied in blocks of `chunk`
    consecutive features (RAM-bounded); a single p=50000 pass needs > 15 GB of RAM."""
    import torch
    from fslnet.fslnet import FSLNet

    model = FSLNet.from_pretrained(device="cpu")
    out = []
    with torch.no_grad():
        for s in range(0, X1.shape[1], chunk):
            prob, _ = model(torch.tensor(X1[:, s:s + chunk], dtype=torch.float32),
                            torch.tensor(X2[:, s:s + chunk], dtype=torch.float32))
            out.append(prob.detach().cpu().numpy().ravel())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------
# Optional extras (not in the default method set)
# ----------------------------------------------------------------------------------------------


def delta_shap(X1, Y1, X2, Y2, seed, n_estimators=100, test_size=0.25):
    import shap

    mean_shap = []
    for k, (X, Y) in enumerate([(X1, Y1), (X2, Y2)]):
        Xtr, Xte, Ytr, _ = train_test_split(X, Y, test_size=test_size, random_state=seed + 10 + 2 * k)
        rf = rf_reg(seed + k, n_estimators).fit(Xtr, Ytr)
        vals = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent").shap_values(
            Xte, check_additivity=False)
        mean_shap.append(np.mean(vals, axis=0))
    return np.abs(mean_shap[1] - mean_shap[0])


def causal_forest_vimp(X, Y, W, seed, n_estimators):
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LogisticRegression

    cf = CausalForestDML(
        model_y=rf_reg(seed, 100), model_t=LogisticRegression(max_iter=500, random_state=seed),
        n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=5, discrete_treatment=True,
        random_state=seed, n_jobs=N_JOBS,
    )
    cf.fit(Y, W, X=X)
    return np.asarray(cf.feature_importances_, dtype=float)


# ----------------------------------------------------------------------------------------------
# One replicate
# ----------------------------------------------------------------------------------------------

DEFAULT_METHODS = ["rf_domain", "permucate", "loco_r_risk", "sgshift_l1_spec", "sgshift_l1_misspec",
                   "datafix", "fslnet"]
ALL_METHODS = DEFAULT_METHODS + ["rfdom_shadowVIMP", "rfdom_vitaPIMP", "rfdom_boruta", "delta_shap", "grf"]


def run_one(X1, Y1, X2, Y2, shifted, args, seed):
    p = X1.shape[1]
    X1f, X2f = X1.astype(np.float32), X2.astype(np.float32)
    X = np.vstack([X1f, X2f])
    Y = np.concatenate([Y1, Y2])
    W = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))]).astype(int)
    methods = set(args.methods)
    scores, times, errors, native, extra = {}, {}, {}, {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # keep the sweep alive; record the failure
            errors[name] = repr(e)
            print(f"    [{name}] FAILED: {e!r}", flush=True)
            return None
        times[name] = time.perf_counter() - t0
        print(f"    {name}: {times[name]:.0f}s", flush=True)
        return out

    if "rf_domain" in methods:
        v = timed("rf_domain", lambda: rf_domain_importance(X, W, seed))
        if v is not None:
            scores["rf_domain"] = v
    if "permucate" in methods:
        def run_perm():
            fit = permucate_fit(X, Y, W, seed, args.n_estimators)
            return permucate_vimp(fit, seed + 5, args.n_perm, args.n_neighbors)
        out = timed("permucate", run_perm)
        if out is not None:
            scores["permucate"] = out[0]
            extra["permucate_n_used_features"] = out[1]
    if "loco_r_risk" in methods:
        v = timed("loco_r_risk", lambda: loco_r_risk(X, Y, W, seed + 6, args.n_estimators))
        if v is not None:
            scores["loco_r_risk"] = v
    if "sgshift_l1_spec" in methods or "sgshift_l1_misspec" in methods:
        out = timed("sgshift_l1", lambda: sgshift_l1(X1f, Y1, X2f, Y2, seed + 10))
        if out is not None:
            scores["sgshift_l1_spec"], scores["sgshift_l1_misspec"] = out
            times["sgshift_l1_spec"] = times["sgshift_l1_misspec"] = times.pop("sgshift_l1") / 2.0
    for name, fn in [
        ("rfdom_shadowVIMP", lambda: rfdom_shadow(X, W, seed + 1)),
        ("rfdom_vitaPIMP", lambda: rfdom_pimp(X, W, seed + 2, n_null=args.n_null)),
        ("rfdom_boruta", lambda: rfdom_boruta(X, W, seed + 3, n_iter=args.boruta_iter)),
        ("delta_shap", lambda: delta_shap(X1f, Y1, X2f, Y2, seed + 7)),
        ("grf", lambda: causal_forest_vimp(X, Y, W, seed + 9, args.n_estimators)),
    ]:
        if name in methods:
            v = timed(name, fn)
            if v is not None:
                scores[name] = v
    if "datafix" in methods:
        out = timed("datafix", lambda: datafix_locate(X1f, X2f, seed + 11))
        if out is not None:
            scores["datafix"] = out[0]
            native["datafix"] = np.flatnonzero(out[1] == 1)
    if "fslnet" in methods:
        v = timed("fslnet", lambda: fslnet_locate(X1f, X2f))
        if v is not None:
            scores["fslnet"] = v
            native["fslnet"] = np.flatnonzero(v > 0.5)

    metrics = {m: metrics_from_scores(s, shifted, p) for m, s in scores.items()}
    for m, sel in native.items():
        tp = len(set(sel.tolist()) & set(shifted.tolist()))
        metrics[m].update({"native_n_selected": len(sel), "native_TP": tp,
                           "native_precision": tp / len(sel) if len(sel) else 0.0,
                           "native_recall": tp / max(1, len(shifted))})
    return scores, metrics, times, errors, extra


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", choices=["messy", "quant_concept"], default="messy",
                    help="messy: concept + covariate shift on all signal features; "
                         "quant_concept: pure concept drift on half of the alpha characteristics")
    ap.add_argument("--n", type=int, default=500, help="n_exist = n_new")
    ap.add_argument("--p_over_n", type=float, default=20.0)
    ap.add_argument("--grid_len", type=int, default=20)
    ap.add_argument("--grid_max", type=float, default=0.95)
    ap.add_argument("--frac_signal", type=float, default=0.1)
    ap.add_argument("--snr", type=float, default=1.0)
    ap.add_argument("--shift_ratio", type=float, default=1.0)
    ap.add_argument("--cov_gamma", type=float, default=0.3)
    ap.add_argument("--n_factors", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=ALL_METHODS)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n_estimators", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=25)
    ap.add_argument("--n_neighbors", type=int, default=20)
    ap.add_argument("--n_null", type=int, default=30)
    ap.add_argument("--boruta_iter", type=int, default=15)
    ap.add_argument("--grid_index", type=str, default=None, help="comma-separated grid indices (debug / thinning)")
    ap.add_argument("--datafix_every", type=int, default=1,
                    help="run DataFix only on grid indices divisible by this (it dominates the runtime at p/n=100)")
    ap.add_argument("--out", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    p = int(round(args.p_over_n * args.n))
    grid = np.linspace(0, args.grid_max, args.grid_len)
    os.makedirs(os.path.join(args.out, "scores"), exist_ok=True)
    csv_path = os.path.join(args.out, "results_long.csv")
    rows, done = [], set()
    if os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["grid_idx"], r["rep"]) for r in rows}
    idxs = list(range(len(grid))) if args.grid_index is None else [int(i) for i in args.grid_index.split(",")]
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args) | {"p": p, "grid": grid.tolist()}, f, indent=2)
    print(f"n={args.n} per batch, p={p} (p/n={p / args.n:.0f}), methods={args.methods}", flush=True)
    for rep in range(args.reps):
        for gi in idxs:
            rho = float(grid[gi])
            if (gi, rep) in done:
                continue
            seed = args.seed + 1000 * rep + 10 * gi
            t0 = time.perf_counter()
            make_pair = make_quant_concept_pair if args.dgp == "quant_concept" else make_messy_pair
            X1, Y1, X2, Y2, shifted = make_pair(
                args.n, p, rho, seed, frac_signal=args.frac_signal, snr=args.snr, shift_ratio=args.shift_ratio,
                cov_gamma=args.cov_gamma, n_factors=args.n_factors)
            print(f"[rep {rep} grid {gi}/{len(grid) - 1}] rho={rho:.3f} #shifted={len(shifted)} "
                  f"(data {time.perf_counter() - t0:.0f}s)", flush=True)
            gargs = copy.copy(args)
            if gi % args.datafix_every != 0:
                gargs.methods = [m for m in args.methods if m != "datafix"]
            scores, metrics, times, errors, extra = run_one(X1, Y1, X2, Y2, shifted, gargs, seed)
            for m, mt in metrics.items():
                rows.append({"grid_idx": gi, "rho": rho, "rep": rep, "n": args.n, "p": p,
                             "n_shifted": len(shifted), "method": m, "time_sec": times.get(m, np.nan),
                             **extra, **mt})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            np.savez_compressed(os.path.join(args.out, "scores", f"rep{rep}_grid{gi:02d}.npz"),
                                shifted=shifted, **scores)
            with open(os.path.join(args.out, "errors.jsonl"), "a") as f:
                f.write(json.dumps({"rep": rep, "grid_idx": gi, "errors": errors}) + "\n")
            summ = ", ".join(f"{m}:{mt.get('AUC', float('nan')):.2f}" for m, mt in metrics.items())
            print(f"  done in {time.perf_counter() - t0:.0f}s | AUC {summ}", flush=True)


if __name__ == "__main__":
    main()
