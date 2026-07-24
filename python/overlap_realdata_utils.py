"""Shared overlap subsampling for real-data VIMP benchmarks."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

OVERLAP_MIXTURE_GRID = np.array(
    [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75]
)

N1 = 2000
N2 = 2000


def overlap_id(overlap: float) -> str:
    return f"overlap_{overlap:g}"


def overlap_batches(df1, df2, overlap_mixture, n1=N1, n2=N2, seed=0):
    """Subsample two domains with controlled distributional overlap."""
    rng = np.random.default_rng(seed)
    p = df1.shape[1] - 1
    X1 = np.asarray(df1[:, :p], dtype=float)
    X2 = np.asarray(df2[:, :p], dtype=float)
    Y1 = np.asarray(df1[:, p], dtype=float).ravel()
    Y2 = np.asarray(df2[:, p], dtype=float).ravel()

    X = np.vstack([X1, X2])
    W = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X, W)
    prob = clf.predict_proba(X)[:, 1]
    p1 = prob[: len(X1)]
    p2 = prob[len(X1) :]

    overlap = float(np.clip(overlap_mixture, 1e-3, 0.99))
    half_span = 0.5 * (1.0 - overlap)
    t0 = 0.5 - half_span
    t1 = 0.5 + half_span

    def _pick(X_pool, Y_pool, p_pool, target, k):
        k = min(k, len(X_pool))
        if k <= 0:
            return X_pool[:0], Y_pool[:0]
        order = np.argsort(np.abs(p_pool - target))
        idx = order[:k]
        if len(idx) < k:
            extra = rng.choice(len(X_pool), k - len(idx), replace=False)
            idx = np.concatenate([idx, extra])
        return X_pool[idx], Y_pool[idx]

    X1_s, Y1_s = _pick(X1, Y1, p1, t0, n1)
    X2_s, Y2_s = _pick(X2, Y2, p2, t1, n2)
    out1 = np.hstack([X1_s, Y1_s.reshape(-1, 1)])
    out2 = np.hstack([X2_s, Y2_s.reshape(-1, 1)])
    return out1, out2
