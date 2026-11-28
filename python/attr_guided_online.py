"""Attribution-guided online updating on a CTR-style stream with sparse concept drift.
Each round: predict (prequential), attribute the loss change to features, then update with a per-feature
proximal anchor that is loose on the flagged (drifting) features and tight elsewhere.
Usage: python3 attr_guided_online.py [n_seeds] [batch_size]"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

par = Parallel(n_jobs=4, backend="multiprocessing")
P, S, K, M, T, W, WARM = 1000, 20, 4, 400, 60, 4, 8      # features, signal, drifting, batch, rounds, window, warm-up
GRID_U, GRID_LO, GRID_HI, GRID_K = [(u, 0) for u in (.3, 3, 30, 300, 3000)], [3, 30], [300, 3000], [8, 24]

def make_stream(seed, mode):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(P, 10)) * np.sqrt(.05)                     # factor loadings, var(x_j) = 1
    sig, beta = rng.choice(P, S, replace=False), np.zeros(P)
    beta[sig] = rng.choice([-.45, .45], S)
    drift, last, out = rng.choice(sig, K, replace=False), np.full(P, -99), []
    for t in range(T):
        if mode == "abrupt" and t >= WARM and (t - WARM) % 12 == 0:  # K signal features flip sign
            ch = rng.choice(sig, K, replace=False)
            beta[ch], last[ch] = -beta[ch], t
        if mode == "gradual" and t >= WARM:                          # K features random-walk (alpha decay)
            beta[drift], last[drift] = beta[drift] + rng.normal(0, .1, K), t
        X = rng.normal(size=(M, 10)) @ A.T + np.sqrt(.5) * rng.normal(size=(M, P))
        z = X @ beta
        out.append((X, (rng.random(M) < expit(z)).astype(float), z, np.flatnonzero(t - last < W)))
    return out

nll = lambda z, y: np.logaddexp(0, z) - y * z

def fit(X, y, b0, lam):
    def f(b):
        z, d = X @ b, b - b0
        return np.sum(nll(z, y)) + .5 * lam @ d ** 2, X.T @ (expit(z) - y) + lam * d
    return minimize(f, b0, jac=True, method="L-BFGS-B", options=dict(maxiter=60)).x

def perm_importance(X, y, b, rng, R=5):  # permutation risk increase of each feature: (mean, s.e.)
    z, d = X @ b, 0.
    base = nll(z, y)
    for _ in range(R):
        zz = z[:, None] + (np.take_along_axis(X, rng.random(X.shape).argsort(0), 0) - X) * b
        d = d + nll(zz, y[:, None]) - base[:, None]
    return (d / R).mean(0), (d / R).std(0) / np.sqrt(M)

def run(seed, mode, method, cfg, zc=3.0):
    stream, rng = make_stream(seed, mode), np.random.default_rng(seed + 1)
    cat = lambda sl, i: np.concatenate([s[i] for s in stream[sl]])
    b, ref, rv, rec = np.zeros(P), None, 0., dict(exc=[], auc=[], prec=[], rec=[])
    for t, (X, y, z, orc) in enumerate(stream):
        if t >= WARM:
            zh = X @ b
            rec["exc"].append(np.mean(nll(zh, y)) - np.mean(nll(z, y)))
            rec["auc"].append(roc_auc_score(y, zh))
        if t == WARM - 1:                                            # common warm-up fit
            b = fit(cat(slice(0, WARM), 0), cat(slice(0, WARM), 1), b, np.full(P, 30.))
        if t < WARM or method == "static": continue
        flag = np.zeros(P, bool)
        if method in ("attr", "random"):
            pm, se = perm_importance(X, y, b, rng)
            zs = np.zeros(P) if ref is None else (ref - pm) / np.sqrt(se ** 2 + rv + 1e-12)
            if ref is not None:
                flag[np.argsort(-zs)[:cfg[2]]] = True                             # top-K attribution scores
            hit = flag[orc].sum()
            if method == "attr" and len(orc):
                rec["prec"].append(hit / cfg[2]); rec["rec"].append(hit / len(orc))
            strong = zs > zc                                                      # freeze reference on drifters
            ref, rv = (pm, se ** 2) if ref is None else (np.where(strong, ref, .8 * ref + .2 * pm),
                                                         np.where(strong, rv, .64 * rv + .04 * se ** 2))
            if method == "random":
                flag = np.isin(np.arange(P), rng.choice(P, cfg[2], replace=False))
        elif method == "oracle":
            flag[orc] = True
        lo = slice(max(0, t - W + 1), t + 1) if method == "scratch" else slice(t, t + 1)
        guided = method in ("attr", "random", "oracle")
        b = fit(cat(lo, 0), cat(lo, 1), np.zeros(P) if method == "scratch" else b,
                np.where(flag, cfg[0], cfg[1]) if guided else np.full(P, cfg[0]))
    return {k: np.mean(v) if v else np.nan for k, v in rec.items()}

def tune(mode, method, grid):
    sc = [np.mean([r["exc"] for r in par(delayed(run)(s, mode, method, c) for s in range(100, 104))]) for c in grid]
    return grid[int(np.argmin(sc))]

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    M = int(sys.argv[2]) if len(sys.argv) > 2 else M
    for mode in ("abrupt", "gradual", "null"):
        cfg = {"uniform": tune(mode, "uniform", GRID_U), "scratch": tune(mode, "scratch", GRID_U),
               "attr": tune(mode, "attr", [(lo, hi, k) for lo in GRID_LO for hi in GRID_HI for k in GRID_K])}
        print(f"\n== {mode}  (p={P}, window={W * M} obs, tuned cfg: {cfg})")
        print(f"{'method':9s} {'excess logloss':>16s} {'AUC':>6s}  top-K attribution: precision / recall")
        for m in ("static", "scratch", "uniform", "random", "attr", "oracle"):
            r = par(delayed(run)(s, mode, m, cfg.get(m, cfg["attr"])) for s in range(n_seeds))
            g = lambda k: np.array([x[k] for x in r])
            ex = f"{np.nanmean(g('prec')):.2f} / {np.nanmean(g('rec')):.2f}" if m == "attr" and mode != "null" else ""
            print(f"{m:9s} {g('exc').mean():8.3f} ({g('exc').std() / n_seeds ** .5:.3f}) {g('auc').mean():6.3f}  {ex}")
