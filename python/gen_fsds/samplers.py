"""Conditional samplers for X_j | X_-j with one common interface.

    s = make_sampler("NSF", p=20, seed=0).fit(X_train)
    Xj_tilde = s.sample(X_query, j, B=5, rng)         # (B, n_query), original units

Under H0: T ⟂ X_j | X_-j an exact sampler makes (X_-j, X~_j) exchangeable with
(X_-j, X_j), which is what makes conditional permutation / CRT / CPI valid.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from .flows import AmortCFM, CFM1D, NSF1D


def _minus(X, j):
    return np.delete(X, j, axis=1)


class Sampler:
    name = "base"

    def __init__(self, p: int, seed: int = 0):
        self.p, self.seed = p, seed
        self.fit_time = 0.0

    def fit(self, X):
        t0 = time.time()
        self.X = np.asarray(X, dtype=float)
        self._fit(self.X)
        self.fit_time = time.time() - t0
        return self

    def _fit(self, X):
        pass

    def sample(self, Xq, j, B, rng):
        raise NotImplementedError


class MarginalPerm(Sampler):
    """Breaks the dependence on X_-j: what the existing 'condperm'/'oob' code does."""

    name = "Marginal"

    def sample(self, Xq, j, B, rng):
        return np.stack([rng.permutation(Xq[:, j]) for _ in range(B)])


class LinGauss(Sampler):
    name = "LinGauss"

    def _fit(self, X):
        self.m = []
        for j in range(self.p):
            r = Ridge(alpha=1.0).fit(_minus(X, j), X[:, j])
            self.m.append((r, np.std(X[:, j] - r.predict(_minus(X, j)))))

    def sample(self, Xq, j, B, rng):
        r, s = self.m[j]
        return r.predict(_minus(Xq, j))[None] + s * rng.standard_normal((B, len(Xq)))


class RFResid(Sampler):
    """E[X_j|X_-j] by RF + permuted OOB residuals (the PermuCATE-style conditional permutation)."""

    name = "RFResid"

    def __init__(self, p, seed=0, n_estimators=100, min_samples_leaf=5):
        super().__init__(p, seed)
        self.ne, self.msl = n_estimators, min_samples_leaf

    def _fit(self, X):
        self.m = []
        for j in range(self.p):
            rf = RandomForestRegressor(self.ne, min_samples_leaf=self.msl, max_features=0.5, oob_score=True,
                                       n_jobs=1, random_state=self.seed + j).fit(_minus(X, j), X[:, j])
            self.m.append((rf, X[:, j] - rf.oob_prediction_))

    def sample(self, Xq, j, B, rng):
        rf, res = self.m[j]
        mu = rf.predict(_minus(Xq, j))
        return mu[None] + res[rng.integers(0, len(res), (B, len(Xq)))]


class QRF(Sampler):
    """Quantile-regression-forest sampler: draw X_j from the training rows sharing a random leaf.

    Nonparametric, handles heteroscedastic / multimodal conditionals without deep learning."""

    name = "QRF"

    def __init__(self, p, seed=0, n_estimators=100, min_samples_leaf=10):
        super().__init__(p, seed)
        self.ne, self.msl = n_estimators, min_samples_leaf

    def _fit(self, X):
        self.m = []
        for j in range(self.p):
            rf = RandomForestRegressor(self.ne, min_samples_leaf=self.msl, max_features=0.5, n_jobs=1,
                                       random_state=self.seed + j).fit(_minus(X, j), X[:, j])
            leaves = rf.apply(_minus(X, j))
            order = [np.argsort(leaves[:, t], kind="stable") for t in range(leaves.shape[1])]
            sorted_leaves = [leaves[o, t] for t, o in enumerate(order)]
            self.m.append((rf, X[:, j].copy(), order, sorted_leaves))

    def sample(self, Xq, j, B, rng):
        rf, y, order, sleaves = self.m[j]
        lq = rf.apply(_minus(Xq, j))
        n, T = lq.shape
        out = np.empty((B, n))
        for b in range(B):
            tsel = rng.integers(0, T, n)
            for t in np.unique(tsel):
                rows = np.where(tsel == t)[0]
                lo = np.searchsorted(sleaves[t], lq[rows, t], "left")
                hi = np.searchsorted(sleaves[t], lq[rows, t], "right")
                pick = lo + np.floor(rng.random(len(rows)) * np.maximum(hi - lo, 1)).astype(int)
                out[b, rows] = y[order[t][np.minimum(pick, len(y) - 1)]]
        return out


class NSFSampler(Sampler):
    """zuko neural spline flow per feature."""

    name = "NSF"

    def __init__(self, p, seed=0, **kw):
        super().__init__(p, seed)
        self.kw = kw

    def _fit(self, X):
        self.m = [NSF1D(self.p - 1, seed=self.seed + j, **self.kw).fit(X[:, j], _minus(X, j))
                  for j in range(self.p)]

    def sample(self, Xq, j, B, rng):
        return self.m[j].sample(_minus(Xq, j), B, seed=int(rng.integers(1 << 31)))


class CFMSampler(Sampler):
    """torchcfm conditional (rectified) flow matching per feature; reflow>0 gives rectified-flow rounds."""

    def __init__(self, p, seed=0, reflow=0, n_euler=20, **kw):
        super().__init__(p, seed)
        self.reflow, self.n_euler, self.kw = reflow, n_euler, kw
        self.name = "CFM" if reflow == 0 else f"CFM-reflow{reflow}"

    def _fit(self, X):
        self.m = [CFM1D(self.p - 1, reflow=self.reflow, n_euler=self.n_euler, seed=self.seed + j, **self.kw)
                  .fit(X[:, j], _minus(X, j)) for j in range(self.p)]

    def sample(self, Xq, j, B, rng, n_steps=None):
        return self.m[j].sample(_minus(Xq, j), B, n_steps=n_steps, seed=int(rng.integers(1 << 31)))


class AmortCFMSampler(Sampler):
    """One masked flow-matching network for all conditionals (O(1) fits instead of O(p))."""

    name = "AmortCFM"

    def __init__(self, p, seed=0, **kw):
        super().__init__(p, seed)
        self.kw = kw

    def _fit(self, X):
        self.m = AmortCFM(self.p, seed=self.seed, **self.kw).fit(X)

    def sample(self, Xq, j, B, rng, n_steps=None):
        return self.m.sample(Xq, j, B, n_steps=n_steps, seed=int(rng.integers(1 << 31)))


SAMPLERS = {
    "Marginal": MarginalPerm,
    "LinGauss": LinGauss,
    "RFResid": RFResid,
    "QRF": QRF,
    "NSF": NSFSampler,
    "CFM": lambda p, seed=0, **kw: CFMSampler(p, seed, reflow=0, **kw),
    "CFM-reflow1": lambda p, seed=0, **kw: CFMSampler(p, seed, reflow=1, **kw),
    "AmortCFM": AmortCFMSampler,
}


def make_sampler(name: str, p: int, seed: int = 0, **kw) -> Sampler:
    return SAMPLERS[name](p, seed=seed, **kw)
