"""Data selection with attribution-localised partial reuse (logistic, p=500). Scarce fresh data from Q plus pooled
slices: clean (from Q), stale (old concept: kc signal coefficients flipped), optional noisy (35% label flips).
A reference model fit on the fresh data attributes each slice's loss to features (expected-permutation importance);
features with significantly negative importance on a slice are 'conflicting'. A slice is reused on all other
coordinates, while the conflicting coordinates get a slice-specific frozen offset. Compared with sample-level
loss filtering, slice discard, no selection and oracles (oracle_F: true conflicts; oracle: also noise weights).
Usage: python3 meta_data_selection.py [n_seeds]"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit

par = Parallel(n_jobs=4, backend="multiprocessing")
P, S, NC, NS, NN, ETA, ZC = 500, 20, 1000, 4000, 4000, .35, 4.
sp = lambda z: np.logaddexp(0, z)

def world(seed, nf, kc, noisy):
    rng = np.random.default_rng(seed)
    A, sig = rng.normal(size=(P, 10)) * np.sqrt(.05), rng.choice(P, S, replace=False)
    bq = np.zeros(P); bq[sig] = rng.choice([-.45, .45], S)
    bo, dr = bq.copy(), rng.choice(sig, kc, replace=False)
    bo[dr] *= -1
    def draw(n, b, eta=0.):
        X = rng.normal(size=(n, 10)) @ A.T + np.sqrt(.5) * rng.normal(size=(n, P))
        y = rng.random(n) < expit(X @ b)
        return X, np.where(rng.random(n) < eta, ~y, y).astype(float)
    return draw(nf, bq), [draw(NC, bq), draw(NS, bo)] + ([draw(NN, bq, ETA)] if noisy else []), bq, dr, draw

def fit(X, y, lam, w=None, off=0.):
    w = np.ones(len(y)) if w is None else w
    def f(b):
        z = X @ b + off
        return np.sum(w * (sp(z) - y * z)) + .5 * lam * b @ b, X.T @ (w * (expit(z) - y)) + lam * b
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=150)).x

def conflicts(b, X, y):                                          # features with significantly negative importance
    p = expit(X @ b); a = (y - p)[:, None] * X * b + .5 * (p * (1 - p))[:, None] * b ** 2 * (1 + X ** 2)
    return a.mean(0) / (a.std(0) / np.sqrt(len(y)) + 1e-12) < -ZC

def run(seed, nf, kc, noisy, method, lam=30., keep=.5):
    (Xf, yf), slices, bq, dr, draw = world(seed, nf, kc, noisy)
    Xt = draw(4000, bq)[0]; zt = Xt @ bq
    b_ref = fit(Xf, yf, 30.); Xs, ys, ws, os_ = [Xf], [yf], [np.ones(nf)], [np.zeros(nf)]
    for k, (X, y) in enumerate(slices):
        w, off, Xm = np.ones(len(y)), np.zeros(len(y)), X
        if method == "pool_all": pass
        elif method == "fresh_only": continue
        elif method == "loss_filter":
            l = sp(X @ b_ref) - y * (X @ b_ref); w = (l <= np.quantile(l, keep)).astype(float)
        elif method == "slice_discard" and conflicts(b_ref, X, y).any(): continue
        elif method in ("attr_partial", "oracle_F", "oracle"):
            F = np.isin(np.arange(P), dr) if method != "attr_partial" else conflicts(b_ref, X, y)
            if method != "attr_partial" and k != 1: F[:] = False
            if method == "oracle" and k == 2: w = np.full(len(y), (1 - 2 * ETA) ** 2)
            if F.any():                                          # slice-specific fit on the localised coordinates
                bF = fit(X[:, F], y, 1., off=X[:, ~F] @ b_ref[~F]); off, Xm = X[:, F] @ bF, X * ~F
        Xs.append(Xm); ys.append(y); ws.append(w); os_.append(off)
    b = fit(np.vstack(Xs), np.concatenate(ys), lam, np.concatenate(ws), np.concatenate(os_))
    return float(np.mean(sp(Xt @ b) - sp(zt) - expit(zt) * (Xt @ b - zt)))

if __name__ == "__main__":
    ns, M = int(sys.argv[1]) if len(sys.argv) > 1 else 30, ("fresh_only", "pool_all", "loss_filter", "slice_discard",
                                                           "attr_partial", "oracle_F", "oracle")
    for kc, nz in ((5, False), (5, True), (0, True)):
        for nf in (150, 300, 600):
            ev = lambda m, a, ss: par(delayed(run)(s, nf, kc, nz, m, **a) for s in ss)
            tune = lambda m, g: min(g, key=lambda a: np.mean(ev(m, a, range(100, 103))))
            hp = {m: tune(m, [dict(lam=l) for l in (3., 30., 300.)]) for m in M if m != "loss_filter"}
            hp["loss_filter"] = tune("loss_filter", [dict(lam=l, keep=k) for l in (30., 300.) for k in (.3, .6, .9)])
            print(f"\n== stale slice: {kc} flipped signals of {S}; noisy slice: {nz}; fresh n={nf} (p={P}); hp={hp}")
            for m in M:
                r = np.array(ev(m, hp[m], range(ns)))
                print(f"{m:14s} excess KL {r.mean():.4f} ({r.std() / ns ** .5:.4f})")
