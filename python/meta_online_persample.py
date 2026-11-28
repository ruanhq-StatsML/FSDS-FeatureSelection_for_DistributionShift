"""Per-sample attribution-guided online learner: logistic diag-EKF whose process noise Q_j is raised on the top-K
previously important features whose (expected-permutation) importance dropped persistently (per-feature CUSUM).
Compared with a per-sample conformal-surprise weighting rule, EWC-style stiffness, a uniform EKF and an oracle.
Usage: python3 meta_online_persample.py [n_seeds]"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

par = Parallel(n_jobs=4, backend="multiprocessing")
P, S, K, WARM, N, E, BUF, CHK, RS, KK, LAM = 1000, 20, 4, 3200, 9600, 1600, 400, 50, 1 / 2000, .12, 30.
sp, ar = lambda z: np.logaddexp(0, z), np.arange(P)

def make_stream(seed, mode):
    rng = np.random.default_rng(seed)
    A, sig = rng.normal(size=(P, 10)) * np.sqrt(.05), rng.choice(P, S, replace=False)
    b0 = rng.choice([-.45, .45], S)
    path, recent = np.tile(b0, (N, 1)), np.zeros((N, S), bool)
    if mode in ("abrupt", "noisy"):                                   # K signal coefficients flip sign every E samples
        for t in range(WARM, N, E):
            ch = rng.choice(S, K, replace=False)
            path[t:, ch] *= -1; recent[t:t + 800, ch] = True
    if mode == "gradual":                                             # K coefficients random-walk (sd .1 / 400 samples)
        ch = rng.choice(S, K, replace=False)
        path[WARM:, ch] += np.cumsum(rng.normal(0, .005, (N - WARM, K)), 0); recent[WARM:, ch] = True
    mk = lambda n: rng.normal(size=(n, 10)) @ A.T + np.sqrt(.5) * rng.normal(size=(n, P))
    X, X0 = mk(N), mk(1000)
    z = (X[:, sig] * path).sum(1)
    yc = rng.random(N) < expit(z)
    y = np.where(rng.random(N) < .1, ~yc, yc) if mode == "noisy" else yc  # noisy: 10% random label flips
    return X, y.astype(float), yc, z, recent, sig, X0, X0[:, sig] @ b0

def run(seed, mode, method, cfg):
    X, y, yc, z, recent, sig, X0, z0 = make_stream(seed, mode)
    nw, rng, (q_lo, q_hi, zc, kap, rr) = WARM - 800, np.random.default_rng(seed + 7), cfg
    def f(b):
        zz = X[:nw] @ b
        return np.sum(sp(zz) - y[:nw] * zz) + .5 * LAM * b @ b, X[:nw].T @ (expit(zz) - y[:nw]) + LAM * b
    b = minimize(f, np.zeros(P), jac=True, method="L-BFGS-B", options=dict(maxiter=100)).x
    ph = expit(X[:nw] @ b)
    Pv, prot = 1 / (LAM + (ph * (1 - ph)) @ X[:nw] ** 2), np.isin(ar, np.argsort(-abs(b))[:S])  # prot: EWC-style
    def incr(i, b):                                                   # E over permutations (2nd order): importance
        x, pp = X[i], expit(X[i] @ b)
        return (y[i] - pp) * x * b + .5 * pp * (1 - pp) * b ** 2 * (1 + x ** 2)
    D = np.array([incr(i, b) for i in range(nw, WARM)])
    slow, v, cs, flag = D.mean(0), D.var(0) + 1e-9, np.zeros(P), np.zeros(P, bool)
    hist = np.array([sp(X[i] @ b) - y[i] * (X[i] @ b) for i in range(nw, nw + BUF)])
    out = dict(kl=[], ret=[], zh=[], fl=[], prec=[], rcl=[])
    for i in range(WARM, N):
        x, zh = X[i], X[i] @ b; p = expit(zh); loss = sp(zh) - y[i] * zh
        out["kl"].append(sp(zh) - sp(z[i]) - expit(z[i]) * (zh - z[i])); out["zh"].append(zh)
        w = np.exp(kap * (1 - 2 * (1 + (hist >= loss).sum()) / (1 + BUF))) if method == "samp" else 1.
        hist[i % BUF] = loss                                          # w: conformal p-value vs rolling loss history
        if method in ("attr", "randK"):
            d = incr(i, b)
            cs = np.minimum(3 * zc, np.maximum(0, cs + (slow - d) / np.sqrt(v) - KK)); ok = cs < zc / 2
            slow += RS * (d - slow) * ok; v += RS * ((d - slow) ** 2 - v) * ok
            if i % CHK == 0:
                cand = np.argsort(-slow)[:2 * S]; top = cand[np.argsort(-cs[cand])[:int(kap)]]
                flag = np.zeros(P, bool); flag[top] = cs[top] > zc
                if method == "randK": flag = np.isin(ar, rng.choice(P, flag.sum(), replace=False))
        if method == "oracle": flag = np.isin(ar, sig[recent[i]])
        if i % CHK == 0 and method in ("attr", "oracle", "randK"): out["fl"].append(flag.sum())
        if i % CHK == 0 and method in ("attr", "oracle") and recent[i].any():
            tr = np.isin(ar, sig[recent[i]]); out["prec"].append((flag & tr).sum() / max(flag.sum(), 1))
            out["rcl"].append((flag & tr).sum() / tr.sum())
        if method == "static": continue
        Qv = np.where(flag, q_hi, q_lo) if method in ("attr", "oracle", "randK") else np.full(P, q_lo)
        if method == "ewc": Qv = np.where(prot, 0., q_lo)
        Pm = Pv + Qv; h = w * p * (1 - p) / rr; Px = Pm * x; s = 1 + h * (Px @ x)
        b = b + Px * (w * (y[i] - p) / rr) / s; Pv = Pm - h * Px ** 2 / s
        if i % 500 == 0: out["ret"].append(np.mean(sp(X0 @ b) - sp(z0) - expit(z0) * (X0 @ b - z0)))
    g = lambda k: np.mean(out[k]) if out[k] else np.nan
    auc = roc_auc_score(yc[WARM:], out["zh"])
    return dict(kl=g("kl"), ret=g("ret"), auc=auc, fl=g("fl"), prec=g("prec"), rcl=g("rcl"))

def tune(mode, method, grid):
    sc = [np.mean([r["kl"] for r in par(delayed(run)(s, mode, method, c) for s in range(100, 103))]) for c in grid]
    return grid[int(np.argmin(sc))]

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for mode in ("abrupt", "noisy", "gradual", "null"):
        ce = tune(mode, "ekf", [(q, q, 0, 0, r) for q in (1e-8, 1e-7, 1e-6) for r in (1, 3)]); q, r = ce[0], ce[4]
        cs = tune(mode, "samp", [(q, q, 0, k, r) for k in (-1.5, -.7, .7, 1.5)])
        ca = tune(mode, "attr", [(q, hi, zc, k, r) for hi in (1e-4, 1e-3) for zc in (10, 20, 40) for k in (4, 8)])
        cfg = {"ekf": ce, "samp": cs, "attr": ca, "oracle": ca, "randK": ca, "ewc": ce, "static": ce}
        print(f"\n== {mode} (p={P}, per-sample, {N - WARM} stream samples) cfg: {cfg}")
        print(f"{'method':7s} {'excess KL':>16s} {'AUC':>6s} {'retention KL':>13s}  flags/chk  prec/recall")
        for m in ("static", "ekf", "ewc", "samp", "randK", "attr", "oracle"):
            r = par(delayed(run)(s, mode, m, cfg[m]) for s in range(n_seeds)); g = lambda k: np.array([x[k] for x in r])
            fl = f"{np.nanmean(g('fl')):6.1f}" if m in ("attr", "randK", "oracle") else ""
            pr = f"{np.nanmean(g('prec')):.2f} / {np.nanmean(g('rcl')):.2f}" if m in ("attr", "oracle") else ""
            pr = pr if mode != "null" else ""
            print(f"{m:7s} {g('kl').mean():8.4f} ({g('kl').std() / n_seeds ** .5:.4f}) {g('auc').mean():6.3f} "
                  f"{g('ret').mean():10.4f}   {fl:>6s}    {pr}")
