"""R1: per-sample online forecasting of SWE-bench Verified outcomes (instance features -> 'resolved' by the next
leaderboard submission). Stream = (submission, instance) pairs in chronological submission order; prequential scoring.
Methods: static, ekf (uniform process noise), samp (conformal-surprise weight), randK, attr (attribution flags).
Tuned on submissions 20-59 only, reported on 60+. Needs real_data/swebench/verified.npz (swebench_prepare.py).
Usage: python3 meta_real_swebench.py [n_orders]"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

par = Parallel(n_jobs=4, backend="multiprocessing")
D_ = np.load(os.path.join(os.path.dirname(__file__), "..", "real_data", "swebench", "verified.npz"))
X0, R = np.hstack([D_["X"], np.ones((500, 1))]), D_["R"]               # last column = intercept
NS, P = R.shape[0], X0.shape[1]
W, TUNE_END, S, CHK, RS, KK, LAM, BUF, ar = 20, 60, 20, 50, 1 / 2000, .12, 30., 400, np.arange(P)
sp = lambda z: np.logaddexp(0, z)

def run(seed, method, cfg):
    q_lo, q_int, q_hi, zc, kap = cfg
    rng = np.random.default_rng(seed); order = np.stack([rng.permutation(500) for _ in range(NS)])
    idx = order.ravel(); X, y = X0[idx], R[np.arange(NS)[:, None], order].ravel().astype(float); nw = W * 500 - 1000
    f = lambda b: (np.sum(sp(X[:nw] @ b) - y[:nw] * (X[:nw] @ b)) + .5 * LAM * b @ b,
                   X[:nw].T @ (expit(X[:nw] @ b) - y[:nw]) + LAM * b)
    b = minimize(f, np.zeros(P), jac=True, method="L-BFGS-B", options=dict(maxiter=100)).x
    ph = expit(X[:nw] @ b); Pv = 1 / (LAM + (ph * (1 - ph)) @ X[:nw] ** 2)
    def incr(i, b):                                                    # expected-permutation importance increment
        pp = expit(X[i] @ b); return (y[i] - pp) * X[i] * b + .5 * pp * (1 - pp) * b ** 2 * (1 + X[i] ** 2)
    Dm = np.array([incr(i, b) for i in range(nw, W * 500)]); Dm[:, -1] = 0
    slow, v, cs, flag = Dm.mean(0), Dm.var(0) + 1e-9, np.zeros(P), np.zeros(P, bool)
    hist = np.array([sp(X[i] @ b) - y[i] * (X[i] @ b) for i in range(nw, nw + BUF)]); zs, fl, fc = [], [], np.zeros(P)
    Qb = np.full(P, q_lo); Qb[-1] = q_int
    for i in range(W * 500, NS * 500):
        x, zh = X[i], X[i] @ b; p = expit(zh); loss = sp(zh) - y[i] * zh; zs.append(zh)
        w = np.exp(kap * (1 - 2 * (1 + (hist >= loss).sum()) / (1 + BUF))) if method == "samp" else 1.
        hist[i % BUF] = loss
        if method in ("attr", "randK"):
            d = incr(i, b)
            cs = np.minimum(3 * zc, np.maximum(0, cs + (slow - d) / np.sqrt(v) - KK)); ok = cs < zc / 2
            slow += RS * (d - slow) * ok; v += RS * ((d - slow) ** 2 - v) * ok
            if i % CHK == 0:
                cand = np.argsort(-slow[:-1])[:2 * S]; top = cand[np.argsort(-cs[cand])[:int(kap)]]
                flag = np.zeros(P, bool); flag[top] = cs[top] > zc; fl.append(flag.sum()); fc += flag
                if method == "randK": flag[:] = False; flag[rng.choice(P - 1, fl[-1], replace=False)] = True
        if method == "static": continue
        Pm = Pv + np.where(flag, q_hi, Qb); h = w * p * (1 - p); Px = Pm * x; s = 1 + h * (Px @ x)
        b = b + Px * (w * (y[i] - p)) / s; Pv = Pm - h * Px ** 2 / s
    zs = np.array(zs).reshape(NS - W, 500); yy = y[W * 500:].reshape(NS - W, 500)
    return zs, yy, np.mean(fl) if fl else 0., fc

def score(zs, yy, a, b_):                                              # log loss and mean within-submission AUC
    z, y = zs[a - W:b_ - W], yy[a - W:b_ - W]
    auc = [roc_auc_score(u, v) for u, v in zip(y, z) if 0 < u.sum() < 500]
    return float(np.mean(sp(z) - y * z)), float(np.mean(auc))

def tr(m, c):
    return np.mean([score(*run(s, m, c)[:2], W, TUNE_END)[0] for s in range(100, 102)])

if __name__ == "__main__":
    no = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    def tune(m, grid):
        sc = par(delayed(tr)(m, c) for c in grid)
        print(m, "tuning logloss by config:", [(c, round(v, 4)) for c, v in zip(grid, sc)])
        return grid[int(np.argmin(sc))]
    ce = tune("ekf", [(q, qi, 0, 0, 0) for q in (1e-9, 1e-8, 1e-7, 1e-6) for qi in (1e-4, 1e-3, 1e-2)])
    cfg = {"static": ce, "ekf": ce, "samp": tune("samp", [(ce[0], ce[1], 0, 0, k) for k in (-1.5, -.7, .7, 1.5)])}
    ga = [(ce[0], ce[1], hi, zc, k) for hi in (1e-4, 1e-3) for zc in (10, 20, 40) for k in (4, 8)]
    cfg["attr"] = tune("attr", ga)
    cfg["randK"] = cfg["attr"]; print("cfg (q_lo,q_int,q_hi,zc,K/kappa):", cfg)
    out = {m: par(delayed(run)(s, m, cfg[m]) for s in range(no)) for m in cfg}
    print(f"{'method':8s} {'logloss':>9s} {'AUC':>7s}  (submissions {TUNE_END}-end, {no} instance orders)  flags/chk")
    pl = lambda m: np.mean([[score(zs, yy, a, a + 1)[0] for a in range(TUNE_END, NS)] for zs, yy, *_ in out[m]], 0)
    sub = {m: pl(m) for m in out}
    for m in out:
        sc = np.array([score(zs, yy, TUNE_END, NS) for zs, yy, *_ in out[m]]).mean(0)
        db = sub[m] - sub["ekf"]; bs = [np.random.default_rng(i).choice(db, len(db)).mean() for i in range(2000)]
        top = np.argsort(-sum(o[3] for o in out[m]))[:5] if m == "attr" else []
        print(f"{m:8s} {sc[0]:9.4f} {sc[1]:7.4f}   d_vs_ekf {db.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, "
              f"{np.percentile(bs, 97.5):+.4f}]  {np.mean([o[2] for o in out[m]]):6.1f}  "
              f"{[str(D_['names'][j]) if j < P - 1 else 'intercept' for j in top]}")
