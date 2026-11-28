"""Can the PO-risk (DR pseudo-outcome) stage of PermuCATE be helped by changing the outcome model, not just RF vs MLP?
Baselines (benchmark_permucate_mlp): rf, mlp (T-learner, learned pi), oracle (true mu, pi). New outcome models, all
2-fold cross-fitted, constant pi = mean(W) (the batch design; earlier ablation: pi barely matters):
  rl_*        R-learner: ONE pooled model m(x) = E[Y|x] on all rows (mu0 = mu1 = m, so the pseudo-outcome is
              (W-pi)(Y-m)/(pi(1-pi)))
  pl_lasso_rf partial-linear stack: beta shared by both arms (pooled arm-centred lasso) + arm-specific RF on residuals
  scr_rf      screen top-50 features by the pooled lasso, then arm-specific RF on them
Usage: python3 permucate_nuisance_remedies.py --p_over_n 20 100 200 --reps 4 --out ../results/permucate_remedies"""

import argparse, os, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, train_test_split
import benchmark_permucate_mlp as m
import benchmark_ultrahigh_dim as bu

warnings.filterwarnings("ignore")


class Fn:  # wraps a fitted predictor as a nuisance object
    def __init__(self, f):
        self.f = f

    def predict(self, X):
        return self.f(X)

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def lasso(X, y, fi=True):
    return LassoCV(cv=3, alphas=30, max_iter=500, tol=1e-3, fit_intercept=fi, n_jobs=4).fit(X, y)


def centred(X, Y, W):  # arm-centred pooled data: shared slope, free arm means
    mx, my = [X[W == w].mean(0) for w in (0, 1)], [Y[W == w].mean() for w in (0, 1)]
    return X - np.where(W[:, None] == 1, mx[1], mx[0]), Y - np.where(W == 1, my[1], my[0]), mx, my


def build(kind, X, Y, W, seed, ne):
    if kind.startswith("rl_"):
        f = {
            "rl_lasso": lambda: lasso(X, Y),
            "rl_rf": lambda: bu.rf_reg(seed, ne).fit(X, Y),
            "rl_mlp": lambda: m.TorchMLP("reg", seed).fit(X, Y),
        }[kind]()
        return Fn(f.predict), Fn(f.predict)
    Xc, Yc, mx, my = centred(X, Y, W)
    las = lasso(Xc, Yc, fi=False)
    pl = kind == "pl_lasso_rf"
    keep = np.arange(X.shape[1]) if pl else np.argsort(-abs(las.coef_))[:50]
    lin = (lambda Z, w: my[w] + (Z - mx[w]) @ las.coef_) if pl else (lambda Z, w: 0.0)
    rf = [bu.rf_reg(seed + w, ne).fit(X[W == w][:, keep], (Y - lin(X, w))[W == w]) for w in (0, 1)]
    return [Fn(lambda Z, w=w: lin(Z, w) + rf[w].predict(Z[:, keep])) for w in (0, 1)]


def fit_new(X, Y, W, seed, kind, ne, clip=0.01):
    Xtr, Xte, Wtr, Wte, Ytr, Yte = train_test_split(X, W, Y, test_size=0.5, stratify=W, random_state=seed)
    m0, m1, mods = np.zeros(len(Xtr)), np.zeros(len(Xtr)), []
    for k, (a, b) in enumerate(KFold(2, shuffle=True, random_state=seed).split(Xtr)):
        f0, f1 = build(kind, Xtr[a], Ytr[a], Wtr[a], seed + 10 * k, ne)
        m0[b], m1[b] = f0.predict(Xtr[b]), f1.predict(Xtr[b])
        mods.append((f0, f1))
    pi = np.clip(Wtr.mean(), clip, 1 - clip)
    pseudo = m1 - m0 + (Ytr - np.where(Wtr == 1, m1, m0)) * (Wtr - pi) / (pi * (1 - pi))
    tau = m._tau_forest(seed + 3, ne, 0.1).fit(Xtr, pseudo)
    avg = lambda i: m.Avg([mm[i] for mm in mods])
    return dict(
        Xtr=np.ascontiguousarray(Xtr, np.float32),
        Xte=np.ascontiguousarray(Xte, np.float32),
        Wte=Wte,
        Yte=Yte,
        mu0=avg(0),
        mu1=avg(1),
        ps=m.Oracle(pi=pi),
        tau=tau,
        clip_e=clip,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_over_n", type=float, nargs="+", default=[20, 100, 200])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument(
        "--variants", nargs="+", default=["rf", "mlp", "rl_lasso", "rl_rf", "rl_mlp", "pl_lasso_rf", "scr_rf", "oracle"]
    )
    ap.add_argument("--n_old", type=int, default=0, help="rows of the old batch (default: same as --n, the new batch)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, path = [], os.path.join(a.out, "results_long.csv")
    for r in a.p_over_n:
        for rep in range(a.reps):
            seed, p = 7 + 100 * rep + int(r), int(round(r * a.n))
            no = a.n_old or a.n
            X1, Y1, X2, Y2, S, truth = m.make_linear_complex_pair(no, p, seed, s=10, b=100, shift_var=8)
            X2, Y2 = X2[: a.n], Y2[: a.n]  # new batch keeps a.n rows, old has n_old
            X, Y, W = np.vstack([X1, X2]), np.concatenate([Y1, Y2]), np.r_[np.zeros(no), np.ones(a.n)].astype(int)
            for v in a.variants:
                t0 = time.perf_counter()
                base = v in ("rf", "mlp", "oracle")
                fit = (
                    m.permucate_fit_cf(X, Y, W, seed, v, 100, truth=truth, tau_mtry=0.1)
                    if base
                    else fit_new(X, Y, W, seed, v, 100)
                )
                mg = lambda Z: 0.5 * (truth["mu0"](Z) + truth["mu1"](Z))  # rl_*: compare m(x) with E[Y|x]
                tr = dict(truth, mu0=mg, mu1=mg) if v.startswith("rl_") else truth
                dg = m.nuisance_diagnostics(fit, tr)
                vimp, nu = bu.permucate_vimp(fit, seed + 1, 25, 20)
                mt = bu.metrics_from_scores(vimp, S, p)
                Z, t0v = fit["Xte"], tr["mu0"](fit["Xte"])  # R^2 on the old-arm regression only
                r0 = 1 - np.mean((fit["mu0"].predict(Z) - t0v) ** 2) / (np.var(t0v) + 1e-12)
                rows.append(
                    dict(
                        p_over_n=r,
                        n_old=no,
                        rep=rep,
                        variant=v,
                        sec=time.perf_counter() - t0,
                        mu_r2=dg["mu_r2"],
                        mu0_r2=r0,
                        **mt,
                    )
                )
                print(
                    f"p/n={r:g} rep {rep} {v:12s} AUC={mt['AUC']:.3f} AP={mt['AP']:.3f} mu_R2={dg['mu_r2']:.2f} "
                    f"({rows[-1]['sec']:.0f}s)",
                    flush=True,
                )
                pd.DataFrame(rows).to_csv(path, index=False)
