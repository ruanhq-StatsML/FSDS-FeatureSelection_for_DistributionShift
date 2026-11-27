"""
PermuCATE under pure concept drift at very high p/n: random-forest vs MLP nuisance models.

DGP (`make_linear_complex_pair`)
--------------------------------
X1 (existing) and X2 (new) are i.i.d. from the same approximate factor model, so P(X) does not change.
    Y_old = X beta + eps
    Y_new = X beta + f(X_S) + eps
beta is sparse and *stable* (b random features). The perturbation features S (|S| = s, disjoint from the
support of beta) are the ground truth for selection; f is a complex function of them: per-feature
sin / tanh / ReLU / quadratic / signed-sqrt terms, pairwise products, and tanh-of-group-sum interactions.
Var(X beta) = lin_var and Var(f(X_S)) = shift_var, eps ~ N(0, 1).

PermuCATE variants (W = batch indicator, 50/50 split, DR pseudo-outcome risk, RF tau model with the exact
tree shortcut of benchmark_ultrahigh_dim.permucate_vimp, n_perm = 25, all p features scored)
    rf_insample : the variant used in benchmark_ultrahigh_dim.py (RF nuisances, pseudo-outcome of the training
                  half built from in-sample nuisance predictions)
    rf          : RF outcome (per arm) + RF propensity, 2-fold cross-fitted pseudo-outcome
    mlp         : MLP outcome (per arm) + MLP propensity, 2-fold cross-fitted pseudo-outcome
    oracle      : true mu0 = X beta, mu1 = X beta + f, pi = 1/2 (diagnostic upper bound for the nuisance stage)

python benchmark_permucate_mlp.py --p_over_n 20 100 200 500 --reps 3 --out ../results/permucate_mlp
"""
import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_ultrahigh_dim as bu  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(bu.N_JOBS)


# ----------------------------------------------------------------------------------------------
# DGP
# ----------------------------------------------------------------------------------------------


def make_linear_complex_pair(n, p, seed, rho=0.3, n_factors=10, s=50, b=100, lin_var=2.0, shift_var=2.0):
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((p, n_factors))
    L /= np.linalg.norm(L, axis=1, keepdims=True)

    def draw():
        F = rng.standard_normal((n, n_factors))
        return (np.sqrt(rho) * (F @ L.T) + np.sqrt(1.0 - rho) * rng.standard_normal((n, p))).astype(np.float32)

    X1, X2 = draw(), draw()
    perm = rng.permutation(p)
    S = np.sort(perm[:s])
    B = np.sort(perm[s:s + b])
    beta = np.zeros(p)
    beta[B] = rng.choice([-1.0, 1.0], size=b) * rng.uniform(0.5, 1.5, size=b)
    Xref = np.vstack([X1, X2])
    lin_scale = np.sqrt(lin_var / ((Xref @ beta).var() + 1e-12))
    beta *= lin_scale
    beta32 = beta.astype(np.float32)

    kind = rng.integers(0, 5, size=s)
    amp = rng.uniform(0.5, 1.5, size=s) * rng.choice([-1.0, 1.0], size=s)
    sp = rng.permutation(s)
    pairs = sp[: 2 * (s // 2)].reshape(-1, 2)
    pair_amp = rng.uniform(0.3, 1.0, size=len(pairs)) * rng.choice([-1.0, 1.0], size=len(pairs))
    groups = rng.permutation(s)[: 5 * (s // 5)].reshape(-1, 5)
    gw = rng.standard_normal(groups.shape)
    gamp = rng.uniform(0.5, 1.0, size=len(groups)) * rng.choice([-1.0, 1.0], size=len(groups))

    def g(k, z):
        return [lambda x: np.sin(2.0 * x), lambda x: np.tanh(2.0 * x), lambda x: np.maximum(x, 0) - 0.4,
                lambda x: x ** 2 - 1.0, lambda x: np.sign(x) * np.sqrt(np.abs(x))][k](z)

    def shift_raw(X):
        Z = X[:, S].astype(np.float64)
        f = np.zeros(len(X))
        for j in range(s):
            f += amp[j] * g(kind[j], Z[:, j])
        f += (pair_amp * Z[:, pairs[:, 0]] * Z[:, pairs[:, 1]]).sum(1)
        for gi in range(len(groups)):
            f += gamp[gi] * np.tanh(Z[:, groups[gi]] @ gw[gi]) ** 2 * np.sin(Z[:, groups[gi, 0]])
        return f

    f_ref = shift_raw(X2)
    shift_mean, shift_sd = f_ref.mean(), f_ref.std() + 1e-12
    shift_scale = np.sqrt(shift_var) / shift_sd

    def shift(X):
        return shift_scale * (shift_raw(X) - shift_mean)

    def lin(X):
        return X @ beta32

    Y1 = lin(X1) + rng.standard_normal(n)
    Y2 = lin(X2) + shift(X2) + rng.standard_normal(n)
    truth = dict(mu0=lin, mu1=lambda X: lin(X) + shift(X))
    return X1, Y1, X2, Y2, S, truth


# ----------------------------------------------------------------------------------------------
# Nuisance models with an sklearn-like API
# ----------------------------------------------------------------------------------------------


class TorchMLP:
    """One-hidden-layer MLP on standardized inputs, AdamW with weight decay, early stopping on a 20% holdout."""

    def __init__(self, task, seed, hidden=64, wd=1e-2, lr=1e-3, max_epochs=150, patience=15):
        self.task, self.seed, self.hidden, self.wd, self.lr = task, seed, hidden, wd, lr
        self.max_epochs, self.patience = max_epochs, patience

    def _net(self, p):
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(torch.nn.Linear(p, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), (X.std(0) + 1e-6) * np.float32(np.sqrt(X.shape[1]))
        y = np.asarray(y, dtype=np.float32)
        if self.task == "reg":
            self.ym_, self.ys_ = float(y.mean()), float(y.std() + 1e-8)
            yt = (y - self.ym_) / self.ys_
        else:
            yt = y
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        nv = max(2, int(0.2 * len(X)))
        va, tr = idx[:nv], idx[nv:]
        Xt = torch.from_numpy((X - self.mu_) / self.sd_)
        yt = torch.from_numpy(yt)[:, None]
        self.net_ = self._net(X.shape[1])
        loss_fn = torch.nn.MSELoss() if self.task == "reg" else torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net_.train()
            opt.zero_grad()
            loss_fn(self.net_(Xt[tr]), yt[tr]).backward()
            opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net_(Xt[va]), yt[va]))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def _raw(self, X):
        Xt = torch.from_numpy((np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_)
        with torch.no_grad():
            return self.net_(Xt).numpy().ravel()

    def predict(self, X):
        return self._raw(X) * self.ys_ + self.ym_

    def predict_proba(self, X):
        q = 1.0 / (1.0 + np.exp(-self._raw(X)))
        return np.stack([1 - q, q], axis=1)


class Avg:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class Oracle:
    def __init__(self, fn=None, pi=None):
        self.fn, self.pi = fn, pi

    def predict(self, X):
        return self.fn(X)

    def predict_proba(self, X):
        return np.stack([np.full(len(X), 1 - self.pi), np.full(len(X), self.pi)], axis=1)


def _make_models(kind, seed, n_est, n):
    if kind == "mlp":
        return TorchMLP("reg", seed), TorchMLP("reg", seed + 1), TorchMLP("clf", seed + 2)
    return bu.rf_reg(seed, n_est), bu.rf_reg(seed + 1, n_est), bu.rf_clf(seed + 2, n_est, n=n)


def _tau_forest(seed, n_est, mtry):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_est, max_features=mtry, min_samples_leaf=5, n_jobs=bu.N_JOBS,
                                 random_state=seed)


def permucate_fit_cf(X, Y, W, seed, kind, n_est, truth=None, clip_e=0.01, n_folds=2, tau_mtry="sqrt"):
    Xtr, Xte, Wtr, Wte, Ytr, Yte = train_test_split(X, W, Y, test_size=0.5, stratify=W, random_state=seed)
    if kind == "rf_insample":
        mu0, mu1, ps = _make_models("rf", seed, n_est, len(Xtr))
        mu0.fit(Xtr[Wtr == 0], Ytr[Wtr == 0])
        mu1.fit(Xtr[Wtr == 1], Ytr[Wtr == 1])
        ps.fit(Xtr, Wtr)
        m0, m1, pi = mu0.predict(Xtr), mu1.predict(Xtr), ps.predict_proba(Xtr)[:, 1]
    elif kind == "oracle":
        mu0, mu1, ps = Oracle(truth["mu0"]), Oracle(truth["mu1"]), Oracle(pi=0.5)
        m0, m1, pi = mu0.predict(Xtr), mu1.predict(Xtr), np.full(len(Xtr), 0.5)
    else:
        m0, m1, pi = np.zeros(len(Xtr)), np.zeros(len(Xtr)), np.zeros(len(Xtr))
        folds_models = []
        for k, (a, b) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(Xtr)):
            f0, f1, fp = _make_models(kind, seed + 10 * k, n_est, len(a))
            a0, a1 = a[Wtr[a] == 0], a[Wtr[a] == 1]
            f0.fit(Xtr[a0], Ytr[a0])
            f1.fit(Xtr[a1], Ytr[a1])
            fp.fit(Xtr[a], Wtr[a])
            m0[b], m1[b], pi[b] = f0.predict(Xtr[b]), f1.predict(Xtr[b]), fp.predict_proba(Xtr[b])[:, 1]
            folds_models.append((f0, f1, fp))
        mu0, mu1, ps = (Avg([m[i] for m in folds_models]) for i in range(3))
    pi = np.clip(pi, clip_e, 1 - clip_e)
    pseudo = m1 - m0 + (Ytr - np.where(Wtr == 1, m1, m0)) * (Wtr - pi) / (pi * (1 - pi))
    tau = _tau_forest(seed + 3, n_est, tau_mtry).fit(Xtr, pseudo)
    return dict(Xtr=np.ascontiguousarray(Xtr, dtype=np.float32), Xte=np.ascontiguousarray(Xte, dtype=np.float32),
                Wte=Wte, Yte=Yte, mu0=mu0, mu1=mu1, ps=ps, tau=tau, clip_e=clip_e)


def nuisance_diagnostics(fit, truth):
    """Out-of-sample quality of the nuisance models on the test half (true mu available in simulation)."""
    Xte, Wte, Yte = fit["Xte"], fit["Wte"], fit["Yte"]
    mu_hat = np.where(Wte == 1, fit["mu1"].predict(Xte), fit["mu0"].predict(Xte))
    mu_true = np.where(Wte == 1, truth["mu1"](Xte), truth["mu0"](Xte))
    r2 = 1.0 - np.mean((mu_hat - mu_true) ** 2) / (np.var(mu_true) + 1e-12)
    pi = fit["ps"].predict_proba(Xte)[:, 1]
    return dict(mu_r2=float(r2), pi_mean=float(pi.mean()), pi_sd=float(pi.std()))


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_over_n", type=float, nargs="+", default=[20, 100, 200, 500])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--s", type=int, default=50, help="number of shifted (perturbation) features")
    ap.add_argument("--b", type=int, default=100, help="support size of the stable linear beta")
    ap.add_argument("--lin_var", type=float, default=2.0)
    ap.add_argument("--shift_var", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--variants", nargs="+", default=["rf_insample", "rf", "mlp", "oracle"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tau_mtry", default="sqrt", help="max_features of the tau forest: 'sqrt' or a fraction of p")
    ap.add_argument("--n_estimators", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=25)
    ap.add_argument("--n_neighbors", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "results_long.csv")
    rows = pd.read_csv(csv_path).to_dict("records") if os.path.exists(csv_path) else []
    done = {(r["p_over_n"], r["rep"]) for r in rows}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    for r in args.p_over_n:
        p = int(round(r * args.n))
        for rep in range(args.reps):
            if (float(r), rep) in done:
                continue
            seed = args.seed + 100 * rep + int(r)
            t0 = time.perf_counter()
            X1, Y1, X2, Y2, S, truth = make_linear_complex_pair(
                args.n, p, seed, rho=args.rho, s=args.s, b=args.b, lin_var=args.lin_var, shift_var=args.shift_var)
            X = np.vstack([X1, X2])
            Y = np.concatenate([Y1, Y2])
            W = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))]).astype(int)
            print(f"[p/n={r:g} p={p} rep {rep}] #shifted={len(S)} (data {time.perf_counter() - t0:.0f}s)", flush=True)
            for v in args.variants:
                t1 = time.perf_counter()
                try:
                    fit = permucate_fit_cf(X, Y, W, seed, v, args.n_estimators, truth=truth,
                                           tau_mtry=args.tau_mtry if args.tau_mtry == "sqrt" else float(args.tau_mtry))
                    diag = nuisance_diagnostics(fit, truth)
                    vimp, n_used = bu.permucate_vimp(fit, seed + 1, args.n_perm, args.n_neighbors)
                    m = bu.metrics_from_scores(vimp, S, p)
                except Exception as e:  # keep the sweep alive
                    print(f"    [{v}] FAILED: {e!r}", flush=True)
                    continue
                dt = time.perf_counter() - t1
                rows.append({"p_over_n": float(r), "p": p, "rep": rep, "variant": v, "time_sec": dt,
                             "n_used_features": n_used, **diag, **m})
                print(f"    {v}: AUC={m['AUC']:.3f} AP={m['AP']:.3f} prec@k={m['precision']:.3f} "
                      f"mu_R2={diag['mu_r2']:.2f} pi={diag['pi_mean']:.2f}±{diag['pi_sd']:.2f} ({dt:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
