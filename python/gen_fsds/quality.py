"""Sampler-quality benchmark: how close is (X_-j, X~_j) to (X_-j, X_j)?

C2ST AUC of a gradient-boosting classifier separating real from resampled rows
(0.5 = indistinguishable), plus fit / sample wall time."""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .dgp import SEM
from .samplers import make_sampler


def c2st_auc(real: np.ndarray, fake: np.ndarray, seed: int = 0, folds: int = 3) -> float:
    """real[i] and fake[i] share the same conditioning row, so both go to the same fold."""
    Z = np.vstack([real, fake])
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    groups = np.r_[np.arange(len(real)), np.arange(len(fake))]
    aucs = []
    for tr, te in GroupKFold(folds, shuffle=True, random_state=seed).split(Z, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
        clf.fit(Z[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1]))
    return float(np.mean(aucs))


def run_quality(sampler_names, n: int, p: int, seed: int, n_feat: int = 8, sampler_kw=None,
                euler_steps=None, noise_scale: float = 1.0) -> list:
    """One replicate: fit each sampler on n pooled null rows, C2ST on n held-out rows."""
    sampler_kw = sampler_kw or {}
    sem = SEM(p=p, seed=seed, noise_scale=noise_scale)
    rng = np.random.default_rng(seed + 1000)
    Xtr, Xte = sem.sample(n, rng), sem.sample(n, rng)
    feats = np.random.default_rng(seed).choice(p, size=min(n_feat, p), replace=False)
    rows = []
    for name in sampler_names:
        s = make_sampler(name, p, seed=seed, **sampler_kw.get(name, {})).fit(Xtr)
        for K in (euler_steps or [None]):
            if K is not None and not hasattr(s, "m"):
                continue
            aucs, t0 = [], time.time()
            for j in feats:
                srng = np.random.default_rng(seed * 31 + int(j))
                kw = {} if K is None else {"n_steps": K}
                try:
                    xt = s.sample(Xte, int(j), 1, srng, **kw)[0]
                except TypeError:
                    continue
                real = np.column_stack([np.delete(Xte, j, 1), Xte[:, j]])
                fake = np.column_stack([np.delete(Xte, j, 1), xt])
                aucs.append(c2st_auc(real, fake, seed))
            if not aucs:
                continue
            rows.append(dict(sampler=name, n=n, p=p, seed=seed, noise_scale=noise_scale, steps=K if K else -1,
                             c2st=float(np.mean(aucs)), c2st_max=float(np.max(aucs)),
                             fit_time=s.fit_time, sample_time=(time.time() - t0) / len(aucs)))
    return rows
