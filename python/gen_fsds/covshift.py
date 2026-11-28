"""Covariate-shift feature diagnosis: baselines vs generative conditional methods.

Every method returns a length-p ``score`` (larger = more shifted) and, where a valid
sampling/likelihood argument exists, one-sided p-values.

Methods
-------
MMD-LOCO            repo baseline (marginal, hard column deletion)
DC-LOCO             domain-classifier leave-one-covariate-out (refit)
DC-CPI:<sampler>    domain-classifier conditional permutation importance with a pluggable
                    X_j | X_-j sampler; Marginal == the repo's current `condperm` / `oob` code
CondKL:NSF          zuko flow p(x_j | x_-j, T): symmetrised conditional KL, the exact
                    "LOCO of KL" term  E_{x_-j}[KL(P1(x_j|x_-j) || P0(x_j|x_-j))]
CondW1:CFM          flow matching p(x_j | x_-j, T) with common noise: conditional W1 transport
JointOT:CFM         joint OT-CFM transport map exist -> new, per-feature displacement
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .flows import CFM1D, JointOTCFM, NSF1D
from .metrics import evaluate, paired_one_sided_p
from .samplers import make_sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _brier(clf, X, T):
    return (clf.predict_proba(X)[:, 1] - T) ** 2


class Ctx:
    """Shared split + domain classifier (fitted lazily, once per replicate)."""

    def __init__(self, X0, X1, seed, n_trees=100):
        self.X0, self.X1, self.seed, self.n_trees = X0, X1, seed, n_trees
        X = np.vstack([X0, X1])
        T = np.r_[np.zeros(len(X0)), np.ones(len(X1))]
        self.Xtr, self.Xte, self.Ttr, self.Tte = train_test_split(X, T, test_size=0.5, stratify=T,
                                                                  random_state=seed)
        self.p = X.shape[1]
        self._clf = None

    @property
    def clf(self):
        if self._clf is None:
            self._clf = RandomForestClassifier(self.n_trees, min_samples_leaf=5, max_features="sqrt",
                                               n_jobs=1, random_state=self.seed).fit(self.Xtr, self.Ttr)
        return self._clf

    def ctx_with_T(self, X, T):
        return np.column_stack([X, np.full(len(X), T) if np.isscalar(T) else T])


def _pack(per_feature_d):
    stats_ = [paired_one_sided_p(d) for d in per_feature_d]
    return dict(score=np.array([np.mean(d) for d in per_feature_d]), pval=np.array([s[1] for s in stats_]))


def m_mmd_loco(ctx: Ctx, **_):
    from VIMP_mmd_benchmark import compute_loco_mmd_batches
    vimp, _ = compute_loco_mmd_batches(ctx.X0, ctx.X1, max_n=1000, seed=ctx.seed)
    return dict(score=np.asarray(vimp), pval=None)


def m_dc_loco(ctx: Ctx, **_):
    base = _brier(ctx.clf, ctx.Xte, ctx.Tte)
    ds = []
    for j in range(ctx.p):
        keep = [c for c in range(ctx.p) if c != j]
        clf = RandomForestClassifier(ctx.n_trees, min_samples_leaf=5, max_features="sqrt", n_jobs=1,
                                     random_state=ctx.seed + 1 + j).fit(ctx.Xtr[:, keep], ctx.Ttr)
        ds.append((clf.predict_proba(ctx.Xte[:, keep])[:, 1] - ctx.Tte) ** 2 - base)
    return _pack(ds)


def m_dc_cpi(ctx: Ctx, sampler: str = "Marginal", B: int = 5, sampler_kw=None, **_):
    s = make_sampler(sampler, ctx.p, seed=ctx.seed, **(sampler_kw or {})).fit(ctx.Xtr)
    rng = np.random.default_rng(ctx.seed + 17)
    base = _brier(ctx.clf, ctx.Xte, ctx.Tte)
    ds = []
    for j in range(ctx.p):
        draws = s.sample(ctx.Xte, j, B, rng)
        loss = np.zeros(len(ctx.Xte))
        for b in range(B):
            Xt = ctx.Xte.copy()
            Xt[:, j] = draws[b]
            loss += _brier(ctx.clf, Xt, ctx.Tte)
        ds.append(loss / B - base)
    out = _pack(ds)
    out["fit_time_sampler"] = s.fit_time
    return out


def m_condkl_nsf(ctx: Ctx, **_):
    ds = []
    for j in range(ctx.p):
        c_tr = ctx.ctx_with_T(np.delete(ctx.Xtr, j, 1), ctx.Ttr)
        f = NSF1D(ctx.p, seed=ctx.seed + j).fit(ctx.Xtr[:, j], c_tr)
        Xm = np.delete(ctx.Xte, j, 1)
        lp_true = f.log_prob(ctx.Xte[:, j], ctx.ctx_with_T(Xm, ctx.Tte))
        lp_other = f.log_prob(ctx.Xte[:, j], ctx.ctx_with_T(Xm, 1.0 - ctx.Tte))
        ds.append(lp_true - lp_other)
    return _pack(ds)


def m_condw1_cfm(ctx: Ctx, B: int = 5, **_):
    rng = np.random.default_rng(ctx.seed + 23)
    sd = ctx.Xtr.std(0)
    score = np.zeros(ctx.p)
    for j in range(ctx.p):
        c_tr = ctx.ctx_with_T(np.delete(ctx.Xtr, j, 1), ctx.Ttr)
        f = CFM1D(ctx.p, seed=ctx.seed + j).fit(ctx.Xtr[:, j], c_tr)
        Xm = np.delete(ctx.Xte, j, 1)
        z = rng.standard_normal((B, len(Xm)))
        a = f.sample(ctx.ctx_with_T(Xm, 1.0), B, z=z)
        b = f.sample(ctx.ctx_with_T(Xm, 0.0), B, z=z)
        score[j] = np.mean(np.abs(a - b)) / sd[j]
    return dict(score=score, pval=None)


def m_joint_ot(ctx: Ctx, **_):
    f = JointOTCFM(ctx.p, seed=ctx.seed).fit(ctx.Xtr[ctx.Ttr == 0], ctx.Xtr[ctx.Ttr == 1])
    A = ctx.Xte[ctx.Tte == 0]
    disp = np.abs(f.transport(A) - f.standardise(A)).mean(0)
    return dict(score=disp, pval=None)


def get_method(name: str):
    if name == "MMD-LOCO":
        return m_mmd_loco
    if name == "DC-LOCO":
        return m_dc_loco
    if name.startswith("DC-CPI:"):
        sampler = name.split(":", 1)[1]
        return lambda ctx, **kw: m_dc_cpi(ctx, sampler=sampler, **kw)
    if name == "CondKL:NSF":
        return m_condkl_nsf
    if name == "CondW1:CFM":
        return m_condw1_cfm
    if name == "JointOT:CFM":
        return m_joint_ot
    raise KeyError(name)


DEFAULT_METHODS = ["MMD-LOCO", "DC-LOCO", "DC-CPI:Marginal", "DC-CPI:LinGauss", "DC-CPI:RFResid",
                   "DC-CPI:QRF", "DC-CPI:NSF", "DC-CPI:CFM", "DC-CPI:AmortCFM", "CondKL:NSF",
                   "CondW1:CFM", "JointOT:CFM"]


def run_covshift(X0, X1, truth, methods, seed, meta=None) -> list:
    ctx = Ctx(X0, X1, seed)
    rows = []
    for name in methods:
        t0 = time.time()
        r = get_method(name)(ctx)
        row = dict(meta or {}, method=name, seed=seed, time=time.time() - t0)
        row.update(evaluate(r["score"], r["pval"], truth, ctx.p))
        row["score"] = np.round(r["score"], 5).tolist()
        rows.append(row)
    return rows
