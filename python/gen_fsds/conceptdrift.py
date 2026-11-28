"""Concept-drift feature diagnosis (P(Y|X) shifts).

Mean-drift pipeline = a doubly-robust pseudo-outcome learner (the repo's PermuCATE/LOCO R-risk idea):
nuisances on split A, CATE tau(x) fitted on split B, importance evaluated on split C.

CD-CPI:<sampler>    PermuCATE-style: PO-risk increase when X_j is replaced by X~_j ~ P(X_j | X_-j)
CD-LOCO             refit tau without X_j on B, PO-risk difference on C
DistDrift:<sampler> distributional drift. Two conditional flows q(y|x,T) (zuko NSF). Pseudo-outcome
                    L = log q(y|x,1) - log q(y|x,0) on new-batch rows estimates D(x) = KL(q1||q0)(x);
                    CPI on a regression of D. Sees variance/shape drift that conditional-mean
                    methods cannot.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from .flows import NSF1D
from .metrics import evaluate, paired_one_sided_p
from .samplers import make_sampler


def _rf(seed, leaf=5, n=100):
    return RandomForestRegressor(n, min_samples_leaf=leaf, max_features=0.5, n_jobs=1, random_state=seed)


class CDCtx:
    def __init__(self, X0, Y0, X1, Y1, seed, clip=0.02):
        X = np.vstack([X0, X1])
        Y = np.r_[Y0, Y1]
        T = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        idx = np.arange(len(X))
        iA, rest = train_test_split(idx, train_size=0.3, stratify=T, random_state=seed)
        iB, iC = train_test_split(rest, train_size=0.3 / 0.7, stratify=T[rest], random_state=seed + 1)
        self.X, self.Y, self.T = X, Y, T
        self.iA, self.iB, self.iC = iA, iB, iC
        self.p, self.seed = X.shape[1], seed
        a = iA
        self.mu0 = _rf(seed).fit(X[a][T[a] == 0], Y[a][T[a] == 0])
        self.mu1 = _rf(seed + 1).fit(X[a][T[a] == 1], Y[a][T[a] == 1])
        self.pi = RandomForestClassifier(100, min_samples_leaf=5, max_features="sqrt", n_jobs=1,
                                         random_state=seed + 2).fit(X[a], T[a])
        self.clip = clip
        self.phiB, self.phiC = self.phi(iB), self.phi(iC)
        self.tau = _rf(seed + 3, leaf=20).fit(X[iB], self.phiB)
        self.XA_B = X[np.r_[iA, iB]]

    def phi(self, idx):
        X, Y, T = self.X[idx], self.Y[idx], self.T[idx]
        pi = np.clip(self.pi.predict_proba(X)[:, 1], self.clip, 1 - self.clip)
        m0, m1 = self.mu0.predict(X), self.mu1.predict(X)
        return (Y - np.where(T == 1, m1, m0)) * (T - pi) / (pi * (1 - pi)) + (m1 - m0)


def _pack(ds):
    st = [paired_one_sided_p(d) for d in ds]
    return dict(score=np.array([np.mean(d) for d in ds]), pval=np.array([s[1] for s in st]))


def m_cd_cpi(ctx: CDCtx, sampler="Marginal", B=5, **_):
    s = make_sampler(sampler, ctx.p, seed=ctx.seed).fit(ctx.XA_B)
    rng = np.random.default_rng(ctx.seed + 31)
    XC = ctx.X[ctx.iC]
    base = (ctx.phiC - ctx.tau.predict(XC)) ** 2
    ds = []
    for j in range(ctx.p):
        draws = s.sample(XC, j, B, rng)
        loss = np.zeros(len(XC))
        for b in range(B):
            Xt = XC.copy()
            Xt[:, j] = draws[b]
            loss += (ctx.phiC - ctx.tau.predict(Xt)) ** 2
        ds.append(loss / B - base)
    return _pack(ds)


def m_cd_loco(ctx: CDCtx, **_):
    XB, XC = ctx.X[ctx.iB], ctx.X[ctx.iC]
    base = (ctx.phiC - ctx.tau.predict(XC)) ** 2
    ds = []
    for j in range(ctx.p):
        keep = [c for c in range(ctx.p) if c != j]
        tau = _rf(ctx.seed + 50 + j, leaf=20).fit(XB[:, keep], ctx.phiB)
        ds.append((ctx.phiC - tau.predict(XC[:, keep])) ** 2 - base)
    return _pack(ds)


def m_dist_drift(ctx: CDCtx, sampler="NSF", B=5, **_):
    """Pseudo-outcome CPI on the distributional drift function D(x) = KL(q(.|x,T=1) || q(.|x,T=0)).

    Flows q(y|x,T) are fitted on split A. On new-batch rows L = log q1(y|x) - log q0(y|x) is an
    unbiased pseudo-outcome for D(x), a function of x only (so it is invariant to covariate shift).
    D is regressed on split B and the usual conditional-permutation risk increase is computed on C."""
    A = ctx.iA
    q = NSF1D(ctx.p + 1, seed=ctx.seed).fit(ctx.Y[A], np.column_stack([ctx.X[A], ctx.T[A]]))

    def pseudo(idx):
        idx = idx[ctx.T[idx] == 1]
        X, Y = ctx.X[idx], ctx.Y[idx]
        L = q.log_prob(Y, np.column_stack([X, np.ones(len(X))])) - q.log_prob(Y, np.column_stack([X, np.zeros(len(X))]))
        return X, L

    XB, LB = pseudo(ctx.iB)
    lo, hi = np.percentile(LB, [1, 99])
    LB = np.clip(LB, lo, hi)
    XC, LC = pseudo(ctx.iC)
    LC = np.clip(LC, lo, hi)
    D = _rf(ctx.seed + 5, leaf=20).fit(XB, LB)
    s = make_sampler(sampler, ctx.p, seed=ctx.seed).fit(ctx.XA_B)
    rng = np.random.default_rng(ctx.seed + 37)
    base = (LC - D.predict(XC)) ** 2
    ds = []
    for j in range(ctx.p):
        draws = s.sample(XC, j, B, rng)
        loss = np.zeros(len(XC))
        for b in range(B):
            Xt = XC.copy()
            Xt[:, j] = draws[b]
            loss += (LC - D.predict(Xt)) ** 2
        ds.append(loss / B - base)
    return _pack(ds)


def get_method(name):
    if name == "CD-LOCO":
        return m_cd_loco
    if name.startswith("CD-CPI:"):
        sp = name.split(":", 1)[1]
        return lambda ctx, **kw: m_cd_cpi(ctx, sampler=sp, **kw)
    if name.startswith("DistDrift:"):
        sp = name.split(":", 1)[1]
        return lambda ctx, **kw: m_dist_drift(ctx, sampler=sp, **kw)
    raise KeyError(name)


DEFAULT_METHODS = ["CD-LOCO", "CD-CPI:Marginal", "CD-CPI:RFResid", "CD-CPI:QRF", "CD-CPI:NSF",
                   "CD-CPI:CFM", "CD-CPI:AmortCFM", "DistDrift:Marginal", "DistDrift:QRF", "DistDrift:NSF"]


def run_concept(X0, Y0, X1, Y1, truth, methods, seed, meta=None) -> list:
    ctx = CDCtx(X0, Y0, X1, Y1, seed)
    rows = []
    for name in methods:
        t0 = time.time()
        r = get_method(name)(ctx)
        row = dict(meta or {}, method=name, seed=seed, time=time.time() - t0)
        row.update(evaluate(r["score"], r["pval"], truth, ctx.p))
        row["score"] = np.round(r["score"], 5).tolist()
        rows.append(row)
    return rows
