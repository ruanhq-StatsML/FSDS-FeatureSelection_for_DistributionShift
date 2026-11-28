"""Selection vs aggregation for weak per-feature drift: group-pooled empirical-Bayes screening (no PCA / factor analysis on X)."""

import argparse
import numpy as np


def scan(X, y):
    Xc, yc = X - X.mean(0), y - y.mean()
    return (Xc.T @ yc) / (np.sqrt((Xc**2).sum(0) * (yc**2).sum()) + 1e-12) * np.sqrt(len(y))


def eb_scores(z, g, G, iters=60, m=20.0):
    """Two-group model z_j ~ (1-pi_g) N(0,1) + pi_g N(0,1+tau2) with group-specific pi_g; returns posterior P(drift)."""
    pi, tau2 = np.full(G, 0.01), 4.0
    cnt = np.bincount(g, minlength=G)
    for _ in range(iters):
        f0 = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        f1 = np.exp(-0.5 * z**2 / (1 + tau2)) / np.sqrt(2 * np.pi * (1 + tau2))
        r = pi[g] * f1 / (pi[g] * f1 + (1 - pi[g]) * f0)
        pbar = r.mean()
        pi = np.clip((np.bincount(g, r, G) + m * pbar) / (cnt + m), 1e-4, 0.5)
        tau2 = max((r * z**2).sum() / max(r.sum(), 1e-9) - 1, 0.25)
    return r, pi


def one(seed, p, n, na, K, G, hot, noise, delta, structured, factors):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), p // G)
    if structured:
        pool = np.where(np.isin(g, rng.choice(G, hot, replace=False)))[0]
    else:
        pool = np.arange(p)
    act = rng.choice(pool, na, replace=False)
    beta = np.zeros(p)
    beta[rng.choice(p, 50, replace=False)] = 0.1
    L = rng.normal(size=(p, factors)) * 0.6 if factors else None
    zs, zc = [], []
    drift = np.zeros(p)
    drift[act] = delta
    for b in range(2):
        X = rng.normal(size=(n, p))
        Fb = rng.normal(size=(n, factors)) if factors else None
        if factors:
            X += Fb @ L.T
        y = X @ (beta + b * drift) + noise * rng.normal(size=n)
        zs.append(scan(X, y))
        if factors:
            Q = np.linalg.qr(np.column_stack([np.ones(n), Fb]))[0]
            zc.append(scan(X - Q @ (Q.T @ X), y - Q @ (Q.T @ y)))
        else:
            zc.append(zs[-1])
    zm = (zs[1] - zs[0]) / np.sqrt(2)
    z = (zc[1] - zc[0]) / np.sqrt(2)
    truth = np.zeros(p, bool)
    truth[act] = True
    marg = np.argsort(-np.abs(zm))[:K]
    plain = np.argsort(-np.abs(z))[:K]
    post, pi = eb_scores(z, g, G)
    eb = np.argsort(-post)[:K]
    hotg = np.argsort(-np.bincount(g, z**2 - 1, G))[:hot]
    oracle = np.where(np.isin(g, hotg))[0]
    oracle = oracle[np.argsort(-np.abs(z[oracle]))[:K]]
    gtrue = np.zeros(G, bool)
    gtrue[np.unique(g[act])] = True
    gscore = np.bincount(g, z**2 - 1, G)
    from scipy.stats import rankdata

    ranks = rankdata(gscore)
    gauc = (ranks[gtrue].sum() - gtrue.sum() * (gtrue.sum() + 1) / 2) / (gtrue.sum() * (~gtrue).sum())
    return dict(
        nu=np.sqrt(n / 2) * delta / np.sqrt(0.5 + noise**2),
        marg=truth[marg].mean() * K / na,
        plain=truth[plain].mean() * K / na,
        eb=truth[eb].mean() * K / na,
        group_top=truth[oracle].mean() * K / na,
        group_auc=gauc,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--na", type=int, default=40)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--G", type=int, default=50)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    print(
        "recall@K of the %d drifting features (random baseline = %.3f); mean over %d reps" % (a.na, a.K / a.p, a.reps)
    )
    print("%-46s %5s %8s %8s %8s %9s %8s" % ("setting", "nu", "marginal", "plain|C", "EB|C", "grp-topK", "grp-AUC"))
    for structured in (1, 0):
        for delta in (0.15, 0.25, 0.40):
            for factors in (0, 5):
                rs = [
                    one(s, a.p, a.n, a.na, a.K, a.G, a.hot, a.noise, delta, structured, factors) for s in range(a.reps)
                ]
                m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
                name = "%s drift, delta=%.2f, %s" % (
                    "block-concentrated" if structured else "uniform",
                    delta,
                    "5 observed factors" if factors else "iid X",
                )
                print(
                    "%-46s %5.2f %8.3f %8.3f %8.3f %9.3f %8.3f"
                    % (name, m["nu"], m["marg"], m["plain"], m["eb"], m["group_top"], m["group_auc"]),
                    flush=True,
                )

    rng = np.random.default_rng(0)
    L = rng.normal(size=(a.p, 5)) * 0.6
    X = rng.normal(size=(a.n, a.p)) + rng.normal(size=(a.n, 5)) @ L.T
    drift = np.zeros(a.p)
    drift[rng.choice(a.p, a.na, replace=False)] = 1.0
    V = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
    for k in (5, 10, 50):
        print(
            "share of ||drift||^2 inside top-%d PCs of X: %.4f (uniform direction would give %.4f)"
            % (k, (V[:k] @ drift) @ (V[:k] @ drift) / (drift @ drift), k / a.p)
        )
