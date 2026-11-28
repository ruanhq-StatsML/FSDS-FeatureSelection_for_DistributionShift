"""Compact ultra-high-dim drift localization: messy DGP (concept + covariate shift), RFDomain and PermuCATE on all
features (exact tree shortcut). Condensed from benchmark_ultrahigh_dim.py (LOCO-R-risk, SGShift, DataFix, FSL-Net,
quant DGP, rho grid; same DGP, draws not bit-identical).   python ultrahigh_dim_compact.py [p_over_n=20] [rho=0.5]"""
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

G = [lambda z: z, lambda z: np.sin(1.5 * z), lambda z: np.tanh(2 * z), lambda z: np.maximum(z, 0) - .4,
     lambda z: z ** 2 - 1, lambda z: np.sign(z) * np.sqrt(np.abs(z))]
MARG = [lambda x: np.exp(.6 * x), lambda x: x ** 3, lambda x: np.tanh(1.5 * x),
        lambda x: np.sign(x) * np.sqrt(np.abs(x))]


def make_data(n, p, rho, seed, snr=1., shift=1., gamma=.3, K=10):
    """Factor model + skewed marginals; the 10% signal features are the shifted ones (covariate shift on half)."""
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((p, K))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    kinds = np.where(rng.random(p) < .3, rng.integers(0, 4, p), -1)
    def draw():
        X = np.sqrt(rho) * rng.standard_normal((n, K)) @ L.T + np.sqrt(1 - rho) * rng.standard_normal((n, p))
        for k, f in enumerate(MARG):
            X[:, kinds == k] = f(X[:, kinds == k])
        return X
    X1, X2 = draw(), draw()
    P = np.vstack([X1, X2])
    X1, X2 = ((X - P.mean(0)) / (P.std(0) + 1e-8) for X in (X1, X2))
    s, sgn = p // 10, lambda m: rng.choice([-1, 1], m)
    S, kind, amp = np.sort(rng.choice(p, s, replace=False)), rng.integers(0, 6, s), rng.uniform(.5, 1.5, s)
    amp, pr = amp * sgn(s), rng.permutation(S)[:s // 2 * 2].reshape(-1, 2)
    pa, partner = rng.uniform(.3, 1, len(pr)) * sgn(len(pr)), rng.choice(S, s)
    c, cov = rng.uniform([[.5], [.25], [.2]], [[1], [.5], [.35]], (3, s)), S[rng.random(s) < .5]
    signal = lambda X: (sum(G[k](X[:, S[kind == k]]) @ amp[kind == k] for k in range(6))
                        + (pa * X[:, pr[:, 0]] * X[:, pr[:, 1]]).sum(1))
    delta = lambda X: np.sin(X[:, S]) @ c[0] + (X[:, S] ** 2 - 1) @ c[1] + (X[:, S] * X[:, partner]) @ c[2]
    X2[:, cov] += gamma * X2[:, cov] ** 2
    f1, d2 = signal(X1), delta(X2)
    sc, ds = np.sqrt(snr / f1.var()), np.sqrt(shift * snr / d2.var())
    eps = lambda: rng.standard_t(5, n) / np.sqrt(5 / 3)
    return X1, sc * f1 + eps(), X2, sc * signal(X2) + ds * d2 + eps(), S


def permucate(X, Y, W, seed, n_perm=25, T=100, nb=20, clip=.01):
    """Risk VIMP of a tau forest on the DR pseudo-outcome. Features no tree splits on get exactly 0 and a permuted
    feature only changes the trees that split on it; E[X_j | X_-j] is a ridge fit on the nb most correlated features."""
    Xtr, Xte, Wtr, Wte, Ytr, Yte = train_test_split(X, W, Y, test_size=.5, stratify=W, random_state=seed)
    Xtr, Xte = (np.ascontiguousarray(a, dtype=np.float32) for a in (Xtr, Xte))
    rfr = lambda s: RFR(T, max_features="sqrt", min_samples_leaf=5, n_jobs=-1, random_state=s)
    mu = [rfr(seed + a).fit(Xtr[Wtr == a], Ytr[Wtr == a]) for a in (0, 1)]
    ps = RFC(T, min_samples_leaf=16, n_jobs=-1, random_state=seed + 2).fit(Xtr, Wtr)
    def pseudo(Xa, Wa, Ya):
        m0, m1, pi = mu[0].predict(Xa), mu[1].predict(Xa), np.clip(ps.predict_proba(Xa)[:, 1], clip, 1 - clip)
        return m1 - m0 + (Ya - np.where(Wa == 1, m1, m0)) * (Wa - pi) / (pi * (1 - pi))
    trees = [e.tree_ for e in rfr(seed + 3).fit(Xtr, pseudo(Xtr, Wtr, Ytr)).estimators_]
    target, P = pseudo(Xte, Wte, Yte), np.stack([t.predict(Xte)[:, 0] for t in trees])
    tot, base, where = P.sum(0), np.mean((target - P.mean(0)) ** 2), {}
    for i, t in enumerate(trees):
        for j in np.unique(t.feature[t.feature >= 0]):
            where.setdefault(int(j), []).append(i)
    used, m = np.array(sorted(where)), Xtr.mean(0)
    Z = (Xtr - m) / (Xtr.std(0) + 1e-8)
    def worker(js):
        Xw, out, C = Xte.copy(), [], np.abs(Z[:, js].T @ Z)
        C[np.arange(len(js)), js] = -1
        for j, q in zip(js, np.argpartition(-C, nb, axis=1)[:, :nb]):
            A = Xtr[:, q] - m[q]
            nu = m[j] + (Xte[:, q] - m[q]) @ np.linalg.solve(A.T @ A + np.eye(nb), A.T @ (Xtr[:, j] - m[j]))
            r, rng, ts, v = Xte[:, j] - nu, np.random.default_rng(seed + j), where[j], []
            for _ in range(n_perm):
                Xw[:, j] = nu + rng.permutation(r)
                new = sum(trees[t].predict(Xw)[:, 0] for t in ts)
                v.append(np.mean((target - (tot - P[ts].sum(0) + new) / T) ** 2) - base)
            Xw[:, j] = Xte[:, j]
            out.append(np.mean(v))
        return out
    score = np.zeros(X.shape[1])
    chunks = np.array_split(used, 32)
    score[used] = np.concatenate(Parallel(n_jobs=-1, prefer="threads")(delayed(worker)(c) for c in chunks))
    return score


if __name__ == "__main__":
    n, seed, p_over_n, rho = 500, 0, *(float(v) for v in sys.argv[1:3]), *(20., .5)[len(sys.argv) - 1:]
    p = int(p_over_n * n)
    X1, Y1, X2, Y2, S = make_data(n, p, rho, seed)
    X, Y, y = np.vstack([X1, X2]), np.r_[Y1, Y2], np.isin(np.arange(p), S)
    W = np.r_[np.zeros(n), np.ones(n)].astype(int)
    scores = dict(rf_domain=RFC(300, min_samples_leaf=16, n_jobs=-1, random_state=seed).fit(X, W).feature_importances_,
                  permucate=permucate(X, Y, W, seed))
    print(f"n={n} p={p} rho={rho} #shifted={len(S)} (chance precision {len(S) / p:.2f})")
    for name, s in scores.items():
        top = y[np.argsort(-s, kind="stable")[:len(S)]].mean()
        print(f"{name:10s} AUC={roc_auc_score(y, s):.3f} AP={average_precision_score(y, s):.3f} precision@k={top:.3f}")
