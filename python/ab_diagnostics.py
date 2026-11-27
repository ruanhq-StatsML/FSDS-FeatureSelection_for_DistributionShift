"""Experiment-platform diagnostics built on the FSDS machinery.

The two FSDS arms map directly onto A/B testing questions once the batch
indicator is replaced by the variant indicator (or the gray-release stage):

- covariate arm (domain classifier + VIMP, LOCO-MMD): is the split balanced,
  and which features / modality blocks break the balance;
- concept arm (DR pseudo-outcome + PermuCATE-style VIMP): is there treatment
  effect heterogeneity, and which features drive it.

The DR pseudo-outcome also gives a cross-fitted AIPW estimator of the ATE,
which generalises CUPED to nonlinear, multi-covariate regression adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from VIMP_mmd_benchmark import MMD, _subsample_batch


def _rf_classifier(n, seed, ntree=200):
    return RandomForestClassifier(
        n_estimators=ntree,
        max_features="sqrt",
        min_samples_leaf=max(1, round(np.sqrt(n) / 2)),
        random_state=seed,
        n_jobs=-1,
    )


def _rf_regressor(n, p, seed, ntree=200):
    return RandomForestRegressor(
        n_estimators=ntree,
        max_features=max(1, round(p / 3)),
        min_samples_leaf=max(1, round(np.sqrt(n) / 4)),
        random_state=seed,
        n_jobs=-1,
    )


def _rf_tau(n, p, seed, ntree=300):
    # The DR pseudo-outcome is very noisy; large leaves keep tau_hat from
    # fitting that noise, otherwise the heterogeneity test loses all power.
    return RandomForestRegressor(
        n_estimators=ntree,
        max_features=max(1, round(p / 3)),
        min_samples_leaf=max(5, n // 40),
        random_state=seed,
        n_jobs=-1,
    )


def _as_blocks(p, blocks):
    if blocks is None:
        return {f"x{j}": [j] for j in range(p)}
    return {name: list(cols) for name, cols in blocks.items()}


@dataclass
class BalanceReport:
    auc: float
    p_value: float
    n_perm: int
    importance: Dict[str, float] = field(default_factory=dict)

    def top(self, k=5):
        return sorted(self.importance.items(), key=lambda kv: -kv[1])[:k]


def split_balance_c2st(
    X_control,
    X_treatment,
    blocks: Optional[Dict[str, Sequence[int]]] = None,
    n_perm=500,
    n_vimp_reps=10,
    test_size=0.5,
    seed=0,
):
    """Classifier two-sample test for the randomisation check (beyond SRM).

    SRM only checks counts; this tests whether P(X | control) = P(X | treatment).
    The classifier is fit on one half and scored on the held-out half, so
    permuting held-out labels against fixed scores gives a valid null for AUC.
    Block-level permutation importance (Brier-risk increase) localises the
    imbalance to features, or to modality blocks when ``blocks`` groups columns.
    """
    X = np.vstack([np.asarray(X_control, float), np.asarray(X_treatment, float)])
    W = np.r_[np.zeros(len(X_control), int), np.ones(len(X_treatment), int)]
    X_tr, X_te, W_tr, W_te = train_test_split(
        X, W, test_size=test_size, stratify=W, random_state=seed
    )
    clf = _rf_classifier(len(W_tr), seed).fit(X_tr, W_tr)
    score = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(W_te, score)

    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(rng.permutation(W_te), score) for _ in range(n_perm)])
    p_value = (1 + np.sum(null >= auc)) / (n_perm + 1)

    base = np.mean((score - W_te) ** 2)
    importance = {}
    for name, cols in _as_blocks(X.shape[1], blocks).items():
        incr = np.zeros(n_vimp_reps)
        for r in range(n_vimp_reps):
            X_perm = X_te.copy()
            X_perm[:, cols] = X_te[rng.permutation(len(X_te))][:, cols]
            incr[r] = np.mean((clf.predict_proba(X_perm)[:, 1] - W_te) ** 2) - base
        importance[name] = float(incr.mean())
    return BalanceReport(auc=float(auc), p_value=float(p_value), n_perm=n_perm, importance=importance)


def block_loco_mmd(X_ref, X_new, blocks: Dict[str, Sequence[int]], max_n=1000, seed=0):
    """LOCO-MMD at block level: MMD(all) - MMD(all without block).

    Comparing a block's LOCO score with the MMD computed on that block alone
    separates marginal drift inside a modality from drift that only appears
    jointly with the other modalities (cross-modal dependence shift).
    """
    rng = np.random.default_rng(seed)
    A = _subsample_batch(np.asarray(X_ref, float), max_n, rng)
    B = _subsample_batch(np.asarray(X_new, float), max_n, rng)
    mmd = MMD()
    full, _ = mmd(A, B)
    p = A.shape[1]
    out = {"__full__": {"loco": full, "marginal": full}}
    for name, cols in blocks.items():
        keep = [c for c in range(p) if c not in set(cols)]
        without, _ = mmd(A[:, keep], B[:, keep]) if keep else (0.0, None)
        alone, _ = mmd(A[:, cols], B[:, cols])
        out[name] = {"loco": float(full - without), "marginal": float(alone)}
    return out


def _block_grams(Z, blocks):
    grams = {}
    for name, cols in blocks.items():
        Zb = Z[:, cols]
        sq = np.sum(Zb * Zb, axis=1)
        D = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Zb @ Zb.T, 0.0)
        med = np.median(D[np.triu_indices_from(D, k=1)])
        grams[name] = np.exp(-D / (2.0 * med + 1e-12))
    return grams


def _mmd2_from_gram(K, A):
    """Unbiased MMD^2 for every label column of A (N x R, 1 = reference)."""
    B = 1.0 - A
    n, m = A.sum(axis=0), B.sum(axis=0)
    KA, KB = K @ A, K @ B
    diag = np.diag(K)[:, None]
    s_aa = np.sum(A * KA, axis=0) - np.sum(A * diag, axis=0)
    s_bb = np.sum(B * KB, axis=0) - np.sum(B * diag, axis=0)
    s_ab = np.sum(A * KB, axis=0)
    return s_aa / (n * (n - 1)) + s_bb / (m * (m - 1)) - 2.0 * s_ab / (n * m)


def _subsets(names):
    from itertools import combinations
    for r in range(1, len(names) + 1):
        for S in combinations(names, r):
            yield frozenset(S)


@dataclass
class ModalityDecomposition:
    value: Dict[frozenset, float]
    lomi: Dict[str, float]
    lomo: Dict[str, float]
    shapley: Dict[str, float]
    order_range: Dict[str, tuple]
    interaction: Dict[frozenset, float]
    p_joint: float
    p_lomi: Dict[str, float]


def modality_shift_decomposition(
    X_ref,
    X_new,
    blocks: Dict[str, Sequence[int]],
    max_n=800,
    n_perm=200,
    seed=0,
):
    """Set-function view of multimodal shift: v(S) = MMD^2 restricted to modalities S.

    Each modality gets its own median-heuristic RBF kernel and k_S is the
    product of the kernels in S, so v(S) are comparable across subsets
    (re-estimating one bandwidth on the concatenation changes the kernel with
    the subset and makes LOMO differences partly a bandwidth artefact).

    - LOMI_j = v({j})              marginal shift of modality j alone
    - LOMO_j = v(M) - v(M \\ j)    what j adds on top of all other modalities
    - Shapley_j                    average of v(S+j) - v(S) over all orders;
      order_range gives min / max over orders, i.e. the path non-uniqueness
    - interaction[{i, j}]          Moebius dividend v(ij) - v(i) - v(j):
      > 0 synergy (shift only visible jointly), < 0 redundancy (shared shift)

    MMD^2 is not monotone in S: multiplying in the kernel of an unshifted
    modality shrinks the statistic, so LOMO of an unshifted modality is
    negative and interactions mix redundancy with this dilution. Read signs
    of LOMO / interactions only for modalities whose LOMI is significant.

    Label permutation gives valid p-values for H0: P = Q (joint) and for
    H0: P_j = Q_j (LOMI). It does NOT give valid p-values for LOMO or the
    interactions, whose nulls are conditional; report those as effect sizes,
    and use dependence_shift_test for the cross-modal dependence question.
    """
    rng = np.random.default_rng(seed)
    A = _subsample_batch(np.asarray(X_ref, float), max_n, rng)
    B = _subsample_batch(np.asarray(X_new, float), max_n, rng)
    Z = np.vstack([A, B])
    labels = np.r_[np.ones(len(A)), np.zeros(len(B))]
    grams = _block_grams(Z, blocks)
    names = list(blocks)

    perms = np.column_stack([labels] + [rng.permutation(labels) for _ in range(n_perm)])
    value, null = {frozenset(): 0.0}, {}
    for S in _subsets(names):
        K = np.ones_like(grams[names[0]])
        for name in S:
            K = K * grams[name]
        stats_all = _mmd2_from_gram(K, perms)
        value[S] = float(stats_all[0])
        null[S] = stats_all[1:]

    full = frozenset(names)
    lomi = {j: value[frozenset([j])] for j in names}
    lomo = {j: value[full] - value[full - {j}] for j in names}

    from itertools import permutations
    contrib = {j: [] for j in names}
    for order in permutations(names):
        seen = frozenset()
        for j in order:
            contrib[j].append(value[seen | {j}] - value[seen])
            seen = seen | {j}
    shapley = {j: float(np.mean(c)) for j, c in contrib.items()}
    order_range = {j: (float(np.min(c)), float(np.max(c))) for j, c in contrib.items()}

    interaction = {}
    for S in _subsets(names):
        if len(S) == 2:
            i, j = sorted(S)
            interaction[S] = value[S] - value[frozenset([i])] - value[frozenset([j])]

    def _p(S):
        return float((1 + np.sum(null[S] >= value[S])) / (n_perm + 1))

    return ModalityDecomposition(
        value=value,
        lomi=lomi,
        lomo=lomo,
        shapley=shapley,
        order_range=order_range,
        interaction=interaction,
        p_joint=_p(full),
        p_lomi={j: _p(frozenset([j])) for j in names},
    )


def _rank_transform(X):
    ranks = np.argsort(np.argsort(X, axis=0), axis=0)
    return (ranks + 1.0) / (len(X) + 1.0)


def dependence_shift_test(X_ref, X_new, blocks, max_n=800, n_perm=200, seed=0):
    """Test for a change in cross-modal dependence that is robust to marginal shift.

    Each coordinate is rank-transformed within its own batch, which removes
    every per-coordinate marginal difference and leaves only the copula; a
    joint product-kernel MMD on the transformed data then tests
    H0: copula(P) = copula(Q). Within-modality dependence changes also
    register; compare with per-block results to attribute them.
    """
    rng = np.random.default_rng(seed)
    A = _rank_transform(_subsample_batch(np.asarray(X_ref, float), max_n, rng))
    B = _rank_transform(_subsample_batch(np.asarray(X_new, float), max_n, rng))
    res = modality_shift_decomposition(A, B, blocks, max_n=max_n, n_perm=n_perm, seed=seed)
    return {"statistic": res.value[frozenset(blocks)], "p_value": res.p_joint}


@dataclass
class AIPWResult:
    ate: float
    se: float
    ci: tuple
    p_value: float
    diff_in_means: float
    diff_in_means_se: float
    cuped: Optional[float]
    cuped_se: Optional[float]
    variance_ratio_vs_dim: float
    pseudo_outcome: np.ndarray = field(repr=False)


def _cuped(Y, W, x_pre):
    theta = np.cov(Y, x_pre)[0, 1] / np.var(x_pre, ddof=1)
    Y_adj = Y - theta * (x_pre - x_pre.mean())
    return _diff_in_means(Y_adj, W)


def _diff_in_means(Y, W):
    y1, y0 = Y[W == 1], Y[W == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(est), float(se)


def aipw_ate(
    X,
    Y,
    W,
    propensity: Optional[float] = None,
    pre_period_col: Optional[int] = None,
    n_folds=5,
    clip=0.01,
    seed=0,
    alpha=0.05,
):
    """Cross-fitted AIPW (doubly robust) ATE.

    In a randomised experiment pass the configured traffic share as
    ``propensity``; the estimator stays unbiased whatever the outcome models
    do, and the outcome models only buy variance reduction. With
    ``propensity=None`` the propensity is estimated (observational / gray
    release stage comparisons), which is where double robustness matters.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float).ravel()
    W = np.asarray(W, int).ravel()
    n, p = X.shape
    mu0, mu1, pi = np.zeros(n), np.zeros(n), np.zeros(n)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(folds.split(X, W)):
        t1, t0 = tr[W[tr] == 1], tr[W[tr] == 0]
        mu1[te] = _rf_regressor(len(t1), p, seed + k).fit(X[t1], Y[t1]).predict(X[te])
        mu0[te] = _rf_regressor(len(t0), p, seed + k).fit(X[t0], Y[t0]).predict(X[te])
        if propensity is None:
            pi[te] = _rf_classifier(len(tr), seed + k).fit(X[tr], W[tr]).predict_proba(X[te])[:, 1]
    if propensity is not None:
        pi[:] = propensity
    pi = np.clip(pi, clip, 1 - clip)

    psi = mu1 - mu0 + W * (Y - mu1) / pi - (1 - W) * (Y - mu0) / (1 - pi)
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha / 2)
    dim, dim_se = _diff_in_means(Y, W)
    cuped = cuped_se = None
    if pre_period_col is not None:
        cuped, cuped_se = _cuped(Y, W, X[:, pre_period_col])
    return AIPWResult(
        ate=float(ate),
        se=float(se),
        ci=(float(ate - z * se), float(ate + z * se)),
        p_value=float(2 * stats.norm.sf(abs(ate / se))),
        diff_in_means=dim,
        diff_in_means_se=dim_se,
        cuped=cuped,
        cuped_se=cuped_se,
        variance_ratio_vs_dim=float((se / dim_se) ** 2),
        pseudo_outcome=psi,
    )


@dataclass
class HTEReport:
    heterogeneity_gain: float
    heterogeneity_se: float
    heterogeneity_p_value: float
    importance: Dict[str, float] = field(default_factory=dict)

    def top(self, k=5):
        return sorted(self.importance.items(), key=lambda kv: -kv[1])[:k]


def hte_drivers(
    X,
    pseudo_outcome,
    blocks: Optional[Dict[str, Sequence[int]]] = None,
    n_perm=10,
    test_size=0.5,
    seed=0,
):
    """Which features drive treatment-effect heterogeneity (PermuCATE style).

    tau(x) is regressed on the DR pseudo-outcome on one half. On the other
    half the PO-risk E[(psi - tau_hat)^2] is compared with the constant-ATE
    risk: a positive, significant gain means real heterogeneity rather than
    noise. Feature importance is the PO-risk increase under conditional
    permutation (X_j = nu_j(X_-j) + permuted residual), which avoids crediting
    features that are merely correlated with the true drivers.
    """
    X = np.asarray(X, float)
    psi = np.asarray(pseudo_outcome, float).ravel()
    n, p = X.shape
    X_tr, X_te, psi_tr, psi_te = train_test_split(X, psi, test_size=test_size, random_state=seed)
    tau = _rf_tau(len(X_tr), p, seed).fit(X_tr, psi_tr)
    tau_te = tau.predict(X_te)

    loss_tau = (psi_te - tau_te) ** 2
    loss_const = (psi_te - psi_tr.mean()) ** 2
    diff = loss_const - loss_tau
    gain = diff.mean()
    gain_se = diff.std(ddof=1) / np.sqrt(len(diff))
    p_value = stats.norm.sf(gain / gain_se)

    rng = np.random.default_rng(seed)
    base = loss_tau.mean()
    importance = {}
    for name, cols in _as_blocks(p, blocks).items():
        others = [c for c in range(p) if c not in set(cols)]
        if others:
            target = X_tr[:, cols] if len(cols) > 1 else X_tr[:, cols[0]]
            nu = _rf_regressor(len(X_tr), len(others), seed + 1).fit(X_tr[:, others], target)
            nu_hat = nu.predict(X_te[:, others]).reshape(len(X_te), -1)
        else:
            nu_hat = np.tile(X_tr[:, cols].mean(axis=0), (len(X_te), 1))
        resid = X_te[:, cols] - nu_hat
        incr = np.zeros(n_perm)
        for r in range(n_perm):
            X_perm = X_te.copy()
            X_perm[:, cols] = nu_hat + resid[rng.permutation(len(X_te))]
            incr[r] = np.mean((psi_te - tau.predict(X_perm)) ** 2) - base
        importance[name] = float(incr.mean())
    return HTEReport(
        heterogeneity_gain=float(gain),
        heterogeneity_se=float(gain_se),
        heterogeneity_p_value=float(p_value),
        importance=importance,
    )


def stage_shift_report(
    X_ref,
    Y_ref,
    X_new,
    Y_new,
    blocks: Optional[Dict[str, Sequence[int]]] = None,
    seed=0,
):
    """Gray-release stage (or reference vs incoming batch) shift diagnosis.

    Covariate arm: did the population change between stages, and where.
    Concept arm: with the stage indicator as "treatment", did E[Y | X] change,
    and which features drive the change. A covariate-only alarm usually means
    a traffic-mix change; a concept alarm means the strategy or model behaves
    differently on the same kind of users.
    """
    X = np.vstack([np.asarray(X_ref, float), np.asarray(X_new, float)])
    Y = np.r_[np.asarray(Y_ref, float).ravel(), np.asarray(Y_new, float).ravel()]
    T = np.r_[np.zeros(len(X_ref), int), np.ones(len(X_new), int)]
    covariate = split_balance_c2st(X_ref, X_new, blocks=blocks, seed=seed)
    dr = aipw_ate(X, Y, T, propensity=None, seed=seed)
    concept = hte_drivers(X, dr.pseudo_outcome, blocks=blocks, seed=seed)
    return {"covariate": covariate, "conditional_shift": dr, "concept": concept}
