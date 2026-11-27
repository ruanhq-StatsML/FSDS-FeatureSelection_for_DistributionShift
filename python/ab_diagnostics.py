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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
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


def _domain_model(model, seed):
    if model == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, early_stopping=True, random_state=seed
        )
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=seed
        )
    raise ValueError(f"unknown model: {model}")


def _crossfit_logloss(X, W, folds, model, seed, clip):
    out = np.zeros(len(W))
    for k, (tr, te) in enumerate(folds):
        p = _domain_model(model, seed + k).fit(X[tr], W[tr]).predict_proba(X[te])[:, 1]
        p = np.clip(p, clip, 1 - clip)
        out[te] = -(W[te] * np.log(p) + (1 - W[te]) * np.log(1 - p))
    return out


@dataclass
class ModalityInfoDecomposition:
    """Each entry is (estimate, standard error), in nats of I(W; X_S)."""

    value: Dict[frozenset, tuple]
    lomi: Dict[str, tuple]
    lomo: Dict[str, tuple]
    shapley: Dict[str, tuple]
    order_range: Dict[str, tuple]
    interaction: Dict[frozenset, tuple]
    z_crit: float

    def _sig(self, est_se):
        est, se = est_se
        return est > self.z_crit * se

    def pattern(self, name):
        """unique / redundant / synergistic / none, from one-sided z-tests on (LOMI, LOMO)."""
        alone, drop = self._sig(self.lomi[name]), self._sig(self.lomo[name])
        if alone and drop:
            return "unique"
        if alone:
            return "redundant"
        if drop:
            return "synergistic"
        return "none"


def modality_info_decomposition(
    X_ref,
    X_new,
    blocks: Dict[str, Sequence[int]],
    model="hgb",
    n_folds=5,
    n_null=1,
    n_repeats=3,
    clip=1e-3,
    z_crit=2.0,
    seed=0,
):
    """Monotone modality decomposition: v(S) = I(W; X_S), W = batch label.

    v(S) is estimated as constant-predictor log-loss minus the cross-fitted
    log-loss of a domain classifier restricted to modalities S. In population
    the chain rule gives I(W; X_{S+j}) = I(W; X_S) + I(W; X_j | X_S) >= I(W; X_S),
    so LOMO_j = I(W; X_j | X_-j) >= 0, an unshifted modality contributes
    exactly zero, and interaction v(ij) - v(i) - v(j) = -co-information
    separates synergy (> 0) from redundancy (< 0) without the kernel dilution
    that affects modality_shift_decomposition.

    Finite-sample cross-fitting makes every v(S) slightly negative under the
    null, and that penalty is not additive across subsets (spurious
    interactions). It is estimated per subset by refitting on permuted labels
    ``n_null`` times and subtracted.

    Cross-fitting is repeated ``n_repeats`` times with fresh folds and null
    permutations; estimates are averaged and the standard error adds the
    between-repeat variance to the per-sample one, because fold and penalty
    noise are otherwise ignored and inflate false "unique" / "synergistic"
    labels. The one-sided z-tests are reliable away from zero and
    conservative-to-irregular at the null boundary.
    """
    X = np.vstack([np.asarray(X_ref, float), np.asarray(X_new, float)])
    W = np.r_[np.zeros(len(X_ref), int), np.ones(len(X_new), int)]
    pi = W.mean()
    l0 = -(W * np.log(pi) + (1 - W) * np.log(1 - pi))
    names = list(blocks)
    subsets = list(_subsets(names))
    full = frozenset(names)
    rng = np.random.default_rng(seed)

    gains = []
    for r in range(n_repeats):
        s = seed + 1000 * r
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s).split(X, W))
        W_null = [rng.permutation(W) for _ in range(n_null)]
        gain = {frozenset(): np.zeros(len(W))}
        for S in subsets:
            cols = [c for name in S for c in blocks[name]]
            g = l0 - _crossfit_logloss(X[:, cols], W, folds, model, s, clip)
            penalty = np.mean([
                np.mean(l0 - _crossfit_logloss(X[:, cols], Wn, folds, model, s, clip)) for Wn in W_null
            ]) if n_null > 0 else 0.0
            gain[S] = g - penalty
        gains.append(gain)

    def _est(fn):
        per = [fn(g) for g in gains]
        means = np.array([d.mean() for d in per])
        within = np.mean([d.var(ddof=1) / len(d) for d in per])
        between = means.var(ddof=1) if len(per) > 1 else 0.0
        return float(means.mean()), float(np.sqrt(within + between))

    value = {S: _est(lambda g, S=S: g[S]) for S in subsets}
    lomi = {j: _est(lambda g, j=j: g[frozenset([j])]) for j in names}
    lomo = {j: _est(lambda g, j=j: g[full] - g[full - {j}]) for j in names}

    from itertools import permutations
    orders = list(permutations(names))

    def _shapley_vec(g, j):
        acc = np.zeros(len(W))
        for order in orders:
            before = frozenset(order[: order.index(j)])
            acc += g[before | {j}] - g[before]
        return acc / len(orders)

    shapley = {j: _est(lambda g, j=j: _shapley_vec(g, j)) for j in names}
    order_range = {}
    for j in names:
        contrib = [
            float(np.mean([(g[frozenset(o[: o.index(j)]) | {j}] - g[frozenset(o[: o.index(j)])]).mean() for g in gains]))
            for o in orders
        ]
        order_range[j] = (min(contrib), max(contrib))

    interaction = {}
    for S in subsets:
        if len(S) == 2:
            i, j = sorted(S)
            interaction[S] = _est(lambda g, S=S, i=i, j=j: g[S] - g[frozenset([i])] - g[frozenset([j])])

    return ModalityInfoDecomposition(
        value=value,
        lomi=lomi,
        lomo=lomo,
        shapley=shapley,
        order_range=order_range,
        interaction=interaction,
        z_crit=z_crit,
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
    A = _subsample_batch(np.asarray(X_ref, float), max_n, rng)
    B = _subsample_batch(np.asarray(X_new, float), max_n, rng)
    Z = np.vstack([A, B])
    n_a = len(A)
    labels = np.r_[np.ones(n_a), np.zeros(len(B))][:, None]

    def _stat(rows):
        Za, Zb = Z[rows[:n_a]], Z[rows[n_a:]]
        U = np.vstack([_rank_transform(Za), _rank_transform(Zb)])
        K = np.ones((len(U), len(U)))
        for G in _block_grams(U, blocks).values():
            K = K * G
        return float(_mmd2_from_gram(K, labels)[0])

    # The rank transform sits inside the permutation loop: permuting after
    # it would compare batches whose marginals were already forced equal,
    # which makes the null too wide and the test needlessly conservative.
    observed = _stat(np.arange(len(Z)))
    null = np.array([_stat(rng.permutation(len(Z))) for _ in range(n_perm)])
    p_value = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {"statistic": observed, "p_value": float(p_value)}


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
    blp_slope: float
    blp_slope_se: float
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

    tau(x) is regressed on the DR pseudo-outcome on one half. The
    heterogeneity test is the best-linear-predictor (BLP) slope test of
    Chernozhukov et al. (generic ML inference on heterogeneous effects): on
    the held-out half, psi is regressed on 1 and tau_hat - mean(tau_hat), and
    H0: slope <= 0 is tested with a heteroskedasticity-robust SE. The
    PO-risk gain over the constant-ATE predictor is kept as an effect size
    only: as a test it pays for the variance of tau_hat and had about 10%
    power at a 0.6 effect gap with n = 3000 in calibration.

    Feature importance is the PO-risk increase under conditional permutation
    (X_j = nu_j(X_-j) + permuted residual), which avoids crediting features
    that are merely correlated with the true drivers.
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

    s = tau_te - tau_te.mean()
    if np.var(s) < 1e-12:
        slope, slope_se, p_value = 0.0, float("inf"), 1.0
    else:
        D = np.column_stack([np.ones_like(s), s])
        coef, *_ = np.linalg.lstsq(D, psi_te, rcond=None)
        resid = psi_te - D @ coef
        bread = np.linalg.inv(D.T @ D)
        meat = (D * resid[:, None] ** 2).T @ D
        cov = bread @ meat @ bread * len(s) / (len(s) - 2)
        slope, slope_se = float(coef[1]), float(np.sqrt(cov[1, 1]))
        p_value = float(stats.norm.sf(slope / slope_se))

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
        blp_slope=slope,
        blp_slope_se=slope_se,
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
