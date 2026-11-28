"""R2: leave-one-repo-out data selection on SWE-bench Verified. Target repo = held-out repo; scarce fresh pairs
(instance, submission) from half of its instances; pools = other repos' pairs, one slice per repo. Evaluation on the
target's other half of instances (instance-disjoint, all submissions). Reuses fit/conflicts from meta_data_selection.
Features: repo indicators dropped; 'num' = 11 patch/test size features, 'all' = num + 600 TF-IDF terms; plus an
intercept. Usage: python3 meta_real_swebench_select.py [reps] [num|all]"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from meta_data_selection import fit, conflicts, sp

par = Parallel(n_jobs=4, backend="multiprocessing")
D_ = np.load(os.path.join(os.path.dirname(__file__), "..", "real_data", "swebench", "verified.npz"))
R, repo, NS = D_["R"], D_["repo"], D_["R"].shape[0]
FS = sys.argv[2] if len(sys.argv) > 2 else "num"
XF = D_["X"][:, len(set(repo)):]                                           # drop repo one-hots
X0 = np.hstack([XF[:, :11] if FS == "num" else XF, np.full((500, 1), 5.)])  # intercept column
TARGETS = ["django/django", "sympy/sympy", "sphinx-doc/sphinx", "matplotlib/matplotlib", "scikit-learn/scikit-learn",
           "astropy/astropy", "pydata/xarray", "pytest-dev/pytest"]

def pairs(rng, inst, n):
    i, s = rng.choice(inst, n), rng.integers(0, NS, n)
    return X0[i], R[s, i].astype(float)

def run(seed, tgt, nf, method, lam=30., keep=.6):
    rng = np.random.default_rng(seed); ti = rng.permutation(np.where(repo == tgt)[0]); h = len(ti) // 2
    Xf, yf = pairs(rng, ti[:h], nf); Xt = X0[ti[h:]].repeat(NS, 0); yt = R[:, ti[h:]].T.ravel().astype(float)
    b_ref = fit(Xf, yf, 30.); Xs, ys, ws, os_ = [Xf], [yf], [np.ones(nf)], [np.zeros(nf)]
    for r in sorted(set(repo) - {tgt}):
        if method == "fresh_only": break
        X, y = pairs(rng, np.where(repo == r)[0], 2000); w, off, Xm = np.ones(len(y)), np.zeros(len(y)), X
        if method == "loss_filter":
            l = sp(X @ b_ref) - y * (X @ b_ref); w = (l <= np.quantile(l, keep)).astype(float)
        elif method in ("slice_discard", "attr_partial"):
            F = conflicts(b_ref, X, y)
            if method == "slice_discard" and F.any(): continue
            if method == "attr_partial" and F.any():
                bF = fit(X[:, F], y, 1., off=X[:, ~F] @ b_ref[~F]); off, Xm = X[:, F] @ bF, X * ~F
        Xs.append(Xm); ys.append(y); ws.append(w); os_.append(off)
    b = fit(np.vstack(Xs), np.concatenate(ys), lam, np.concatenate(ws), np.concatenate(os_)); z = Xt @ b
    return float(np.mean(sp(z) - yt * z)), roc_auc_score(yt, z)

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    M = ("fresh_only", "pool_all", "loss_filter", "slice_discard", "attr_partial")
    for nf in (150, 600):
        ev = lambda m, tg, a, ss: par(delayed(run)(s, t, nf, m, **a) for t in tg for s in ss)
        tune = lambda m, g: min(g, key=lambda a: np.mean([r[0] for r in ev(m, TARGETS, a, range(100, 102))]))
        hp = {m: tune(m, [dict(lam=l) for l in (3., 30., 300., 3000.)]) for m in M if m != "loss_filter"}
        hp["loss_filter"] = tune("loss_filter", [dict(lam=l, keep=k) for l in (300., 3000.) for k in (.3, .6, .9)])
        print(f"\n== features={FS}, fresh pairs n={nf} per target repo; hp={hp}")
        print(f"log loss (AUC) on held-out target instances, {reps} reps")
        res = {m: np.array(ev(m, TARGETS, hp[m], range(reps))).reshape(len(TARGETS), reps, 2) for m in M}
        print(f"{'target':28s}" + "".join(f"{m:>17s}" for m in M))
        for k, t in enumerate(TARGETS):
            print(f"{t:28s}" + "".join(f"  {res[m][k, :, 0].mean():.3f} ({res[m][k, :, 1].mean():.3f})" for m in M))
        print(f"{'mean over repos':28s}" + "".join(f"  {res[m][..., 0].mean():.3f} ({res[m][..., 1].mean():.3f})"
                                              for m in M))
        d = res["attr_partial"][..., 0] - res["pool_all"][..., 0]
        print(f"attr_partial - pool_all log loss by repo: {np.round(d.mean(1), 3).tolist()}; repos where attr wins:"
              f" {(d.mean(1) < 0).sum()}/{len(TARGETS)}")
