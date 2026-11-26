"""
Ultra-high-dimensional (p / n >= 20) benchmark of feature-selection methods for distribution shift,
targeted at high-dimensional financial-factor / characteristic panels.

Setting
-------
n_exist = n_new = 500 (override with --n), p = ratio * n  (ratio = 20 -> p = 10,000).
Outcome model everywhere an outcome / nuisance regression is needed: random forest.

Grid
----
By default the grid is `np.linspace(0, 0.95, 20)` over `rho` (`--grid_param rho`); the shift is
sparse (`--n_shift` shifted features, default 20). Two designs for X (`--design`):
  factor (default): approximate factor model  X = sqrt(rho) * F @ L' + sqrt(1 - rho) * E  with K latent
                    factors, i.e. rho is the share of each characteristic's variance driven by common
                    factors (the usual "factor strength" of a factor-zoo panel). Heavy-tailed (t5) returns noise.
  ar1:              AR(1) correlation rho between neighbouring features, Gaussian noise.
`--grid_param prop` instead sweeps the *proportion* of shifted features over the same grid
(rho fixed by `--rho`); the 0.0 grid point is then a no-shift null.

Scalability adaptations (the reference implementations refit one model per feature, which is
O(p) forest fits and infeasible at p = 10,000)
---------------------------------------------------------------------------------------
* LOCO-MMD / LOCO-HSIC: exact, computed through an incremental squared-distance update.
* Per-feature-refit methods (PermuCATE, LOCO-R-risk, RF-domain LOCO / condperm):
  screen-then-refine. A pool of `--pool` candidate features is built from two cheap screens
  (RF domain-classifier importance and RF tau-model importance on the doubly-robust
  pseudo-outcome); features outside the pool get the lowest score. Pool recall is reported.
* Conditional permutation (PermuCATE, condperm): E[X_j | X_-j] is estimated by ridge on the
  `--n_neighbors` most correlated other features instead of a forest on all p-1 features.
* SGShift L1 (specified / misspecified): same objective as SGShift.py (offset from a source RF,
  L1 penalty on delta, BIC over a lambda path) solved with coordinate descent instead of SCS.
* DataFix (DF-Locate, RF estimator) and FSL-Net (pretrained) are the upstream packages
  (github.com/AI-sandbox/DataFix, github.com/AI-sandbox/FSL-Net).
Not run: the knockoff variant (Gaussian knockoffs are infeasible at p >> n). The default method set
(`--methods`) is the compact one: rf_domain, rfdom_condperm, permucate, loco_r_risk, sgshift_l1_*,
datafix, fslnet; the other implemented methods can be switched on through `--methods`.

Usage
-----
python benchmark_ultrahigh_dim.py --out ../results/ultrahigh_dim
"""
import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, lasso_path
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_DGP import synthetic_injection_nonlinear_interaction  # noqa: E402

warnings.filterwarnings("ignore")
N_JOBS = max(1, os.cpu_count() or 1)

# ----------------------------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------------------------


def ar1_design(n, p, rho, rng):
    Z = rng.standard_normal((n, p))
    if rho <= 0:
        return Z
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    c = np.sqrt(1.0 - rho ** 2)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + c * Z[:, j]
    return X


def factor_design(n, p, rho, loadings, rng):
    F = rng.standard_normal((n, loadings.shape[1]))
    E = rng.standard_normal((n, p))
    return np.sqrt(rho) * (F @ loadings.T) + np.sqrt(1.0 - rho) * E


def make_pair(n, p, rho, n_shift, seed, n_signal=10, noise=0.5, n_prop=None, design="factor", n_factors=10):
    """Existing / new batch with a shared sparse linear mechanism and a nonlinear shift injected on
    the new batch (`synthetic_injection_nonlinear_interaction`). Returns df1_X, df1_Y, df2_X, df2_Y, shifted."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    beta[rng.choice(p, size=n_signal, replace=False)] = rng.uniform(0.5, 1.5, size=n_signal)
    if design == "factor":
        L = rng.standard_normal((p, n_factors))
        L /= np.linalg.norm(L, axis=1, keepdims=True)
        X1 = factor_design(n, p, rho, L, rng)
        X2 = factor_design(n, p, rho, L, rng)
        draw = lambda m: noise * rng.standard_t(5, size=m) / np.sqrt(5.0 / 3.0)  # noqa: E731
    else:
        X1 = ar1_design(n, p, rho, rng)
        X2 = ar1_design(n, p, rho, rng)
        draw = lambda m: noise * rng.standard_normal(m)  # noqa: E731
    Y1 = X1 @ beta + draw(n)
    Y2 = X2 @ beta + draw(n)
    if n_shift <= 0:
        return X1, Y1, X2, Y2, np.array([], dtype=int)
    prop = (n_shift / p) if n_prop is None else n_prop
    df1, df2, shifted = synthetic_injection_nonlinear_interaction(X1, Y1, X2, Y2, n_prop=prop, seed=seed + 1)
    return df1[:, :-1], df1[:, -1], df2[:, :-1], df2[:, -1], np.sort(np.asarray(shifted, dtype=int))


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------


def metrics_from_scores(scores, shifted, p):
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf, posinf=np.finfo(float).max)
    k = len(shifted)
    if k == 0:
        return {"k": 0}
    order = np.argsort(-scores, kind="stable")
    sel = order[:k]
    tp = len(set(sel.tolist()) & set(shifted.tolist()))
    precision = tp / k
    member = np.zeros(p, dtype=int)
    member[shifted] = 1
    finite = scores[np.isfinite(scores)]
    floor = (finite.min() - 1.0) if finite.size else 0.0
    s = np.where(np.isfinite(scores), scores, floor)
    out = {
        "k": k,
        "TP": tp,
        "precision": precision,
        "recall": precision,
        "F1": precision,
        "FDR": 1.0 - precision,
        "recall@2k": len(set(order[: 2 * k].tolist()) & set(shifted.tolist())) / k,
        "AUC": float(roc_auc_score(member, s)),
        "AP": float(average_precision_score(member, s)),
    }
    return out


def pool_scores(scores_pool, pool, p):
    out = np.full(p, -np.inf)
    out[pool] = scores_pool
    return out


# ----------------------------------------------------------------------------------------------
# Domain-classifier (RF) family
# ----------------------------------------------------------------------------------------------


def rf_clf(seed, n_estimators=150, n_jobs=N_JOBS, n=1000, p=10000):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=max(1, int(round(np.sqrt(n) / 2))),
        n_jobs=n_jobs,
        random_state=seed,
    )


def rf_domain_importance(X, W, seed, n_estimators=300):
    clf = rf_clf(seed, n_estimators=n_estimators, n=len(X)).fit(X, W)
    return clf.feature_importances_


def rfdom_shadow(X, W, seed):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    shadow = np.apply_along_axis(rng.permutation, 0, X)
    imp = rf_clf(seed, 300, n=n).fit(np.hstack([X, shadow]), W).feature_importances_
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
    n, p = X.shape
    hits = np.zeros(p)
    cum = np.zeros(p)
    for it in range(n_iter):
        shadow = np.apply_along_axis(rng.permutation, 0, X)
        imp = rf_clf(seed + it, 100, n=n).fit(np.hstack([X, shadow]), W).feature_importances_
        hits += imp[:p] > imp[p:].max()
        cum += imp[:p]
    return hits + cum / (cum.max() + 1e-12) * 1e-3


def cond_resid_fn(Xtr, pool, n_neighbors):
    """Return f(j, Xte) -> (nu_hat, resid) with nu_hat from ridge on the most correlated features."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    Zp = (Xtr[:, pool] - mu[pool]) / sd[pool]
    Zall = (Xtr - mu) / sd
    corr = np.abs(Zp.T @ Zall) / len(Xtr)
    corr[np.arange(len(pool)), pool] = -1.0
    nbrs = np.argpartition(-corr, n_neighbors, axis=1)[:, :n_neighbors]

    def f(idx, Xte):
        j = pool[idx]
        nb = nbrs[idx]
        model = Ridge(alpha=1.0).fit(Xtr[:, nb], Xtr[:, j])
        nu = model.predict(Xte[:, nb])
        return nu, Xte[:, j] - nu

    return f


def rfdom_condperm_loco(X, W, pool, seed, n_perm, n_neighbors, n_estimators=100):
    Xtr, Xte, Wtr, Wte = train_test_split(X, W, test_size=0.5, stratify=W, random_state=seed)
    clf = rf_clf(seed, n_estimators, n_jobs=N_JOBS, n=len(Xtr)).fit(Xtr, Wtr)
    base_auc = roc_auc_score(Wte, clf.predict_proba(Xte)[:, 1])
    clf.n_jobs = 1
    resid = cond_resid_fn(Xtr, pool, n_neighbors)

    def cond_one(idx):
        j = pool[idx]
        nu, r = resid(idx, Xte)
        rng = np.random.default_rng(seed * 7919 + j)
        drops = []
        for _ in range(n_perm):
            Xp = Xte.copy()
            Xp[:, j] = nu + rng.permutation(r)
            drops.append(base_auc - roc_auc_score(Wte, clf.predict_proba(Xp)[:, 1]))
        return float(np.mean(drops))

    def loco_one(idx):
        j = pool[idx]
        keep = np.delete(np.arange(X.shape[1]), j)
        c = rf_clf(seed, n_estimators, n_jobs=1, n=len(Xtr)).fit(Xtr[:, keep], Wtr)
        return base_auc - roc_auc_score(Wte, c.predict_proba(Xte[:, keep])[:, 1])

    cond = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(cond_one)(i) for i in range(len(pool)))
    t0 = time.perf_counter()
    loco = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(loco_one)(i) for i in range(len(pool)))
    return np.array(cond), np.array(loco), time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------
# Delta-SHAP
# ----------------------------------------------------------------------------------------------


def delta_shap(X1, Y1, X2, Y2, seed, n_estimators=100, test_size=0.25):
    import shap

    p = X1.shape[1]
    mean_shap = []
    for k, (X, Y) in enumerate([(X1, Y1), (X2, Y2)]):
        Xtr, Xte, Ytr, _ = train_test_split(X, Y, test_size=test_size, random_state=seed + 10 + 2 * k)
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=5, n_jobs=N_JOBS, random_state=seed + k
        ).fit(Xtr, Ytr)
        expl = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
        vals = expl.shap_values(Xte, check_additivity=False)
        mean_shap.append(np.mean(vals, axis=0))
    return np.abs(mean_shap[1] - mean_shap[0])


# ----------------------------------------------------------------------------------------------
# LOCO-MMD and LOCO-HSIC (exact, incremental distances)
# ----------------------------------------------------------------------------------------------


def loco_mmd_hsic(X1, X2, seed, max_n=1000, sigma_w=1.0, block=64):
    rng = np.random.default_rng(seed)
    if len(X1) > max_n:
        X1 = X1[rng.choice(len(X1), max_n, replace=False)]
    if len(X2) > max_n:
        X2 = X2[rng.choice(len(X2), max_n, replace=False)]
    n1, n2 = len(X1), len(X2)
    Z = np.vstack([X1, X2]).astype(np.float32)
    n, p = Z.shape
    sq = (Z ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0).astype(np.float32)
    med = np.sqrt(np.median(D[D > 0]))
    gamma = np.float32(1.0 / (2.0 * med * med + 1e-12))
    W = np.concatenate([np.zeros(n1), np.ones(n2)])
    L = np.exp(-((W[:, None] - W[None, :]) ** 2) / (2.0 * sigma_w ** 2))
    H = np.eye(n) - np.ones((n, n)) / n
    Lc = (H @ L @ H).astype(np.float32)

    def stats(Dm):
        K = np.exp(-gamma * Dm)
        Kxx, Kyy, Kxy = K[:n1, :n1], K[n1:, n1:], K[:n1, n1:]
        txx = (Kxx.sum() - np.trace(Kxx)) / (n1 * (n1 - 1))
        tyy = (Kyy.sum() - np.trace(Kyy)) / (n2 * (n2 - 1))
        mmd = max(txx + tyy - 2.0 * Kxy.mean(), 0.0)
        hsic = float((K * Lc).sum() / (n - 1) ** 2)
        return mmd, hsic

    mmd0, hsic0 = stats(D)

    def chunk(js):
        out = []
        for j in js:
            col = Z[:, j]
            dj = (col[:, None] - col[None, :]) ** 2
            m, h = stats(np.maximum(D - dj, 0.0))
            out.append((mmd0 - m, hsic0 - h))
        return out

    chunks = np.array_split(np.arange(p), max(1, p // block))
    res = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(chunk)(c) for c in chunks)
    flat = np.array([v for r in res for v in r])
    return flat[:, 0], flat[:, 1]


# ----------------------------------------------------------------------------------------------
# Causal VIMPs: PermuCATE, LOCO-R-risk, causal forest
# ----------------------------------------------------------------------------------------------


def rf_reg(seed, n_estimators=100, n_jobs=N_JOBS):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=5, n_jobs=n_jobs, random_state=seed
    )


def po_risk(tau, Y, W, mu0, mu1, pi, clip=1e-3):
    pi = np.clip(pi, clip, 1 - clip)
    pseudo = (Y - np.where(W == 1, mu1, mu0)) * (W - pi) / (pi * (1 - pi)) + (mu1 - mu0)
    return float(np.mean((pseudo - tau) ** 2))


def permucate_fit(X, Y, W, seed, n_estimators, clip_e=0.01):
    """Train / test split (50/50, stratified by W); RF outcome models per arm, RF propensity,
    RF tau model on the doubly-robust pseudo-outcome."""
    Xtr, Xte, Wtr, Wte, Ytr, Yte = train_test_split(X, W, Y, test_size=0.5, stratify=W, random_state=seed)
    mu0 = rf_reg(seed, n_estimators).fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
    mu1 = rf_reg(seed + 1, n_estimators).fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
    ps = rf_clf(seed + 2, n_estimators, n=len(Xtr)).fit(Xtr, Wtr)
    m0, m1 = mu0.predict(Xtr), mu1.predict(Xtr)
    pi = np.clip(ps.predict_proba(Xtr)[:, 1], clip_e, 1 - clip_e)
    pseudo = m1 - m0 + (Ytr - np.where(Wtr == 1, m1, m0)) * (Wtr - pi) / (pi * (1 - pi))
    tau = rf_reg(seed + 3, n_estimators).fit(Xtr, pseudo)
    return dict(Xtr=Xtr, Xte=Xte, Wte=Wte, Yte=Yte, mu0=mu0, mu1=mu1, ps=ps, tau=tau, clip_e=clip_e)


def permucate_vimp(fit, pool, seed, n_perm, n_neighbors):
    Xte, Wte, Yte = fit["Xte"], fit["Wte"], fit["Yte"]
    mu0 = fit["mu0"].predict(Xte)
    mu1 = fit["mu1"].predict(Xte)
    pi = np.clip(fit["ps"].predict_proba(Xte)[:, 1], fit["clip_e"], 1 - fit["clip_e"])
    tau_model = fit["tau"]
    tau_model.n_jobs = 1
    base = po_risk(tau_model.predict(Xte), Yte, Wte, mu0, mu1, pi)
    resid = cond_resid_fn(fit["Xtr"], pool, n_neighbors)

    def one(idx):
        j = pool[idx]
        nu, r = resid(idx, Xte)
        rng = np.random.default_rng(seed * 104729 + j)
        vals = []
        for _ in range(n_perm):
            Xp = Xte.copy()
            Xp[:, j] = nu + rng.permutation(r)
            vals.append(po_risk(tau_model.predict(Xp), Yte, Wte, mu0, mu1, pi) - base)
        return float(np.mean(vals))

    out = Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(one)(i) for i in range(len(pool)))
    tau_model.n_jobs = N_JOBS
    return np.array(out)


def _rrisk(X, Y, W, seed, n_estimators, n_folds=2, clip_e=0.01):
    """Cross-fitted R-risk with an RF outcome model m(x)=E[Y|X], RF propensity, ridge R-learner tau."""
    n = len(X)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(X))
    m_hat, e_hat = np.zeros(n), np.zeros(n)
    for k, (tr, te) in enumerate(folds):
        m_hat[te] = rf_reg(seed + k, n_estimators, n_jobs=1).fit(X[tr], Y[tr]).predict(X[te])
        e_hat[te] = rf_clf(seed + 5 + k, n_estimators, n_jobs=1, n=len(tr)).fit(X[tr], W[tr]).predict_proba(X[te])[:, 1]
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    Yt, Wt = Y - m_hat, W - e_hat
    risks = []
    for tr, te in folds:
        mask = np.abs(Wt[tr]) > 1e-3
        sc = StandardScaler().fit(X[tr])
        rd = Ridge(alpha=1.0, random_state=seed).fit(
            sc.transform(X[tr][mask]), Yt[tr][mask] / Wt[tr][mask], sample_weight=Wt[tr][mask] ** 2
        )
        tau = rd.predict(sc.transform(X[te]))
        risks.append(np.mean((Yt[te] - tau * Wt[te]) ** 2))
    return float(np.mean(risks))


def loco_r_risk(X, Y, W, pool, seed, n_estimators):
    base = _rrisk(X, Y, W, seed, n_estimators)

    def one(j):
        keep = np.delete(np.arange(X.shape[1]), j)
        return _rrisk(X[:, keep], Y, W, seed, n_estimators) - base

    return np.array(Parallel(n_jobs=N_JOBS, prefer="threads")(delayed(one)(j) for j in pool))


def causal_forest_vimp(X, Y, W, seed, n_estimators):
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LogisticRegression

    cf = CausalForestDML(
        model_y=rf_reg(seed, 100),
        model_t=LogisticRegression(max_iter=500, random_state=seed),
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        discrete_treatment=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    cf.fit(Y, W, X=X)
    return np.asarray(cf.feature_importances_, dtype=float)


# ----------------------------------------------------------------------------------------------
# SGShift L1 (specified / misspecified), coordinate descent + BIC over a lambda path
# ----------------------------------------------------------------------------------------------


def _bic_path_select(Xd, r, lambdas, delta_slice, p_total):
    """Lasso path on (Xd, r) with sklearn's alpha = lambda / 2 (mean-squared loss + lambda * |.|_1)."""
    alphas, coefs, _ = lasso_path(Xd, r, alphas=np.asarray(lambdas) / 2.0, max_iter=2000, tol=1e-5)
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
    f2 = src.predict(X2)
    f1 = src.predict(X1)
    p = X1.shape[1]
    # specified: y_T - f(x_T) ~ X_T delta
    coefs, best = _bic_path_select(X2, Y2 - f2, lambdas, slice(0, p), p)
    spec = np.abs(coefs[:, best]) + 1e-6 * np.abs(coefs).max(axis=1)
    # misspecified: stacked [X_S 0; X_T X_T], penalty 0.9*lambda on beta, lambda on delta
    Xs = np.zeros((len(X1) + len(X2), 2 * p), dtype=np.float32)
    Xs[: len(X1), :p] = X1 / 0.9
    Xs[len(X1):, :p] = X2 / 0.9
    Xs[len(X1):, p:] = X2
    r = np.concatenate([Y1 - f1, Y2 - f2])
    coefs, best = _bic_path_select(Xs, r, lambdas, slice(p, 2 * p), 2 * p)
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


def fslnet_locate(X1, X2):
    import torch
    from fslnet.fslnet import FSLNet

    model = FSLNet.from_pretrained(device="cpu")
    with torch.no_grad():
        prob, _ = model(torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32))
    return prob.detach().cpu().numpy().ravel()


# ----------------------------------------------------------------------------------------------
# One replicate
# ----------------------------------------------------------------------------------------------


def build_pool(screens, size):
    """Round-robin union of the top features of each screen (higher score = more likely shifted)."""
    ranks = [np.argsort(-np.nan_to_num(np.asarray(sc, dtype=float), nan=-np.inf)) for sc in screens]
    pool, seen, i = [], set(), 0
    while len(pool) < size:
        for r in ranks:
            j = int(r[i])
            if j not in seen:
                seen.add(j)
                pool.append(j)
                if len(pool) >= size:
                    break
        i += 1
    return np.array(sorted(pool), dtype=int)


DEFAULT_METHODS = ["rf_domain", "rfdom_condperm", "permucate", "loco_r_risk",
                   "sgshift_l1_spec", "sgshift_l1_misspec", "datafix", "fslnet"]
ALL_METHODS = DEFAULT_METHODS + ["rfdom_shadowVIMP", "rfdom_vitaPIMP", "rfdom_boruta", "rfdom_loco",
                                 "delta_shap", "loco_mmd", "loco_hsic", "grf"]


def run_one(X1, Y1, X2, Y2, shifted, args, seed, pool_size):
    p = X1.shape[1]
    X = np.vstack([X1, X2]).astype(np.float32)
    Y = np.concatenate([Y1, Y2])
    W = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))]).astype(int)
    X1f, X2f = X1.astype(np.float32), X2.astype(np.float32)
    methods = set(args.methods)
    scores, times, errors, native = {}, {}, {}, {}

    def want(*names):
        return any(n in methods for n in names)

    def timed(name, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # keep the sweep alive; record the failure
            errors[name] = repr(e)
            print(f"    [{name}] FAILED: {e!r}", flush=True)
            return None
        times[name] = time.perf_counter() - t0
        return out

    # cheap screens (always computed: they define the candidate pool of the refit-based methods)
    screens = {}
    imp = timed("rf_domain", lambda: rf_domain_importance(X, W, seed))
    if imp is not None:
        screens["rf_domain"] = imp
    out = timed("sgshift_l1", lambda: sgshift_l1(X1f, Y1, X2f, Y2, seed + 10))
    if out is not None:
        screens["sgshift_l1_spec"], screens["sgshift_l1_misspec"] = out
        times["sgshift_l1_spec"] = times["sgshift_l1_misspec"] = times.pop("sgshift_l1") / 2.0
    if "loco_mmd" in methods or "loco_hsic" in methods:
        out = timed("loco_mmd_hsic", lambda: loco_mmd_hsic(X1, X2, seed + 8, max_n=args.max_mmd_n))
        if out is not None:
            screens["loco_mmd"], screens["loco_hsic"] = out
            times["loco_hsic"] = times["loco_mmd"] = times.pop("loco_mmd_hsic") / 2.0
    for m, sc in screens.items():
        if m in methods:
            scores[m] = sc

    fit = timed("nuisance", lambda: permucate_fit(X, Y, W, seed, args.n_estimators))
    pool = None
    if fit is not None:
        pool_screens = [fit["tau"].feature_importances_] + [screens[k] for k in
                                                           ("rf_domain", "sgshift_l1_spec") if k in screens]
        pool = build_pool(pool_screens, pool_size)
    pool_recall = float(np.mean(np.isin(shifted, pool))) if (pool is not None and len(shifted)) else np.nan

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

    if pool is not None:
        if want("rfdom_condperm", "rfdom_loco"):
            out = timed(
                "rfdom_condperm_loco",
                lambda: rfdom_condperm_loco(X, W, pool, seed + 4, args.n_perm, args.n_neighbors, args.n_estimators),
            )
            if out is not None:
                cond, loco, t_loco = out
                if "rfdom_condperm" in methods:
                    scores["rfdom_condperm"] = pool_scores(cond, pool, p)
                if "rfdom_loco" in methods:
                    scores["rfdom_loco"] = pool_scores(loco, pool, p)
                times["rfdom_loco"] = t_loco
                times["rfdom_condperm"] = times.pop("rfdom_condperm_loco") - t_loco
        if "permucate" in methods:
            v = timed("permucate", lambda: permucate_vimp(fit, pool, seed + 5, args.n_perm, args.n_neighbors))
            if v is not None:
                scores["permucate"] = pool_scores(v, pool, p)
        if "loco_r_risk" in methods:
            v = timed("loco_r_risk", lambda: loco_r_risk(X, Y, W, pool, seed + 6, args.loco_trees))
            if v is not None:
                scores["loco_r_risk"] = pool_scores(v, pool, p)

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

    times.pop("nuisance", None)
    metrics = {m: metrics_from_scores(s, shifted, p) for m, s in scores.items()}
    for m, sel in native.items():
        tp = len(set(sel.tolist()) & set(shifted.tolist()))
        metrics[m].update({"native_n_selected": len(sel), "native_TP": tp,
                           "native_precision": tp / len(sel) if len(sel) else 0.0,
                           "native_recall": tp / max(1, len(shifted))})
    return scores, metrics, times, errors, pool_recall, (pool if pool is not None else np.array([], dtype=int))


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="n_exist = n_new")
    ap.add_argument("--p_over_n", type=float, default=20.0, help="p / n (>= 20 for the ultra-high-dim regime)")
    ap.add_argument("--design", choices=["factor", "ar1"], default="factor")
    ap.add_argument("--n_factors", type=int, default=10)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=ALL_METHODS)
    ap.add_argument("--grid_param", choices=["rho", "prop"], default="rho")
    ap.add_argument("--grid_len", type=int, default=20)
    ap.add_argument("--grid_max", type=float, default=0.95)
    ap.add_argument("--rho", type=float, default=0.3, help="fixed rho when --grid_param prop")
    ap.add_argument("--n_shift", type=int, default=20, help="#shifted features when --grid_param rho")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pool", type=int, default=300, help="candidate pool for per-feature-refit methods")
    ap.add_argument("--n_estimators", type=int, default=100)
    ap.add_argument("--loco_trees", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=25)
    ap.add_argument("--n_neighbors", type=int, default=20)
    ap.add_argument("--n_null", type=int, default=30)
    ap.add_argument("--boruta_iter", type=int, default=15)
    ap.add_argument("--max_mmd_n", type=int, default=1000)
    ap.add_argument("--grid_index", type=str, default=None, help="comma-separated grid indices to run (debug)")
    ap.add_argument("--out", type=str, default="../results/ultrahigh_dim")
    return ap.parse_args()


def main():
    args = parse_args()
    p = int(round(args.p_over_n * args.n))
    grid = np.linspace(0, args.grid_max, args.grid_len)
    os.makedirs(os.path.join(args.out, "scores"), exist_ok=True)
    csv_path = os.path.join(args.out, "results_long.csv")
    done = set()
    rows = []
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        done = {(r["grid_idx"], r["rep"]) for r in rows}
    idxs = list(range(len(grid))) if args.grid_index is None else [int(i) for i in args.grid_index.split(",")]
    print(f"n={args.n} per batch, p={p} (p/n={p / args.n:.0f}), grid_param={args.grid_param}, "
          f"grid={np.round(grid, 3).tolist()}", flush=True)
    for rep in range(args.reps):
        for gi in idxs:
            g = float(grid[gi])
            if (gi, rep) in done:
                continue
            seed = args.seed + 1000 * rep + 10 * gi
            if args.grid_param == "rho":
                rho, n_shift, n_prop = g, args.n_shift, None
            else:
                rho, n_prop = args.rho, g
                n_shift = int(round(p * g))
            t0 = time.perf_counter()
            X1, Y1, X2, Y2, shifted = make_pair(args.n, p, rho, n_shift, seed, n_prop=n_prop,
                                              design=args.design, n_factors=args.n_factors)
            print(f"[rep {rep} grid {gi}/{len(grid) - 1}] {args.grid_param}={g:.4f} rho={rho:.3f} "
                  f"#shifted={len(shifted)}", flush=True)
            scores, metrics, times, errors, pool_recall, pool = run_one(X1, Y1, X2, Y2, shifted, args, seed, args.pool)
            for m, mt in metrics.items():
                rows.append({"grid_param": args.grid_param, "grid_idx": gi, "grid_value": g, "rho": rho, "rep": rep,
                             "n": args.n, "p": p, "n_shifted": len(shifted), "method": m,
                             "time_sec": times.get(m, np.nan), "pool_size": len(pool), "pool_recall": pool_recall,
                             **mt})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            np.savez_compressed(os.path.join(args.out, "scores", f"rep{rep}_grid{gi:02d}.npz"),
                                shifted=shifted, pool=pool, **scores)
            with open(os.path.join(args.out, "errors.jsonl"), "a") as f:
                f.write(json.dumps({"rep": rep, "grid_idx": gi, "errors": errors}) + "\n")
            summ = ", ".join(f"{m}:{mt.get('AUC', float('nan')):.2f}" for m, mt in metrics.items())
            print(f"  done in {time.perf_counter() - t0:.0f}s | pool_recall={pool_recall:.2f} | AUC {summ}", flush=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args) | {"p": p, "grid": grid.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
