from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def paired_one_sided_p(d: np.ndarray) -> tuple:
    """One-sided paired t-test for E[d] > 0 (the CPI test of Watson & Wright)."""
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0 or n < 3:
        return 0.0, 0.5
    t = d.mean() / (sd / np.sqrt(n))
    return float(t), float(stats.t.sf(t, n - 1))


def bh_reject(pvals: np.ndarray, q: float = 0.1) -> np.ndarray:
    p = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, p + 1) / p)
    ok = np.where(pvals[order] <= thresh)[0]
    rej = np.zeros(p, dtype=bool)
    if len(ok):
        rej[order[: ok.max() + 1]] = True
    return rej


def evaluate(score: np.ndarray, pval: np.ndarray | None, truth: dict, p: int, alpha: float = 0.05,
             q: float = 0.1) -> dict:
    """Ranking and (when p-values exist) testing metrics against the exact conditional truth.

    cond   : S ∪ pa(S), the features whose full conditional truly shifts (null = complement)
    strict : S only
    desc   : descendants of S outside `cond` (inherit a marginal shift, conditional unchanged)
    """
    cond = np.zeros(p, bool)
    cond[truth["cond"]] = True
    strict = np.zeros(p, bool)
    strict[truth["strict"]] = True
    desc = np.zeros(p, bool)
    desc[truth["desc"]] = True
    k = int(cond.sum())
    top = np.argsort(-score)[:k]
    out = dict(
        auc_cond=float(roc_auc_score(cond, score)),
        auc_strict=float(roc_auc_score(strict, score)),
        prec_at_k=float(cond[top].mean()),
        desc_in_topk=float(desc[top].sum() / max(desc.sum(), 1)) if desc.any() else np.nan,
    )
    if pval is not None:
        rej = pval <= alpha
        out["fpr"] = float(rej[~cond].mean()) if (~cond).any() else np.nan
        out["fpr_desc"] = float(rej[desc].mean()) if desc.any() else np.nan
        out["power"] = float(rej[cond].mean())
        bh = bh_reject(pval, q)
        out["bh_n"] = int(bh.sum())
        out["bh_fdp"] = float((~cond[bh]).mean()) if bh.any() else 0.0
        out["bh_power"] = float(bh[cond].mean())
    return out
