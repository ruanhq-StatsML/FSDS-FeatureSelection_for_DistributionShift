"""Generative-model toy (bigram LM): reuse of a stale synthetic pool and a noisy scraped pool next to scarce fresh data.
Context-level localisation (G-test per context) decides *where* a pool can be reused; compared with token-level
loss filtering (RHO-1 style), no filtering, fresh only and an oracle. Metric: stationary-weighted KL(T_Q || model).
Usage: python3 meta_generative_toy.py [n_seeds]"""
import sys
import numpy as np
from scipy.stats import chi2

V, KD, EPS, NS, NN, AL = 40, 6, .4, 60000, 20000, .1         # vocab, drifted ctx, noise mix, pools, smoothing

def world(seed, nf, kd):
    rng = np.random.default_rng(seed)
    TQ = rng.dirichlet(np.full(V, .3), V)
    TO, dr = TQ.copy(), rng.choice(V, kd, replace=False)
    TO[dr] = rng.dirichlet(np.full(V, .3), kd)                   # stale generator: kd contexts have changed
    pi = np.full(V, 1. / V)
    def draw(T, n, eps=0.):                                      # (context, next-token) pairs, uniform contexts
        c = rng.integers(0, V, n); u = rng.random(n)
        t = np.minimum((u[:, None] > np.cumsum(T[c], 1)).sum(1), V - 1)
        bad = rng.random(n) < eps
        return c, np.where(bad, rng.integers(0, V, n), t)
    cnt = lambda c, t: np.bincount(c * V + t, minlength=V * V).reshape(V, V).astype(float)
    return TQ, pi, dr, cnt(*draw(TQ, nf)), cnt(*draw(TO, NS)), cnt(*draw(TQ, NN, EPS)), draw(TO, NS), draw(TQ, NN, EPS)

def model(C):
    return (C + AL) / (C + AL).sum(1, keepdims=True)

def gtest(Cs, Cf):                                               # per-context G test: slice row vs fresh row
    ps, pf = model(Cs), model(Cf)
    ns, nf = Cs.sum(1, keepdims=True), Cf.sum(1, keepdims=True)
    pool = (ns * ps + nf * pf) / (ns + nf)
    G = 2 * ((Cs * np.log(ps / pool)).sum(1) + (Cf * np.log(pf / pool)).sum(1))
    return chi2.sf(G, V - 1)

def run(seed, nf, kd, method, keep=.5, om=.1, ctx_alpha=1e-3):
    TQ, pi, dr, Cf, Cs, Cn, ds, dn = world(seed, nf, kd)
    ref = model(Cf); C = Cf.copy()
    if method == "pool_all": C = Cf + Cs + Cn
    if method == "token_filter":
        for c, t in (ds, dn):
            loss = -np.log(ref[c, t]); k = loss <= np.quantile(loss, keep)
            C = C + np.bincount(c[k] * V + t[k], minlength=V * V).reshape(V, V)
    if method == "ctx_attr":
        for Cp in (Cs, Cn):
            flag = gtest(Cp, Cf) < ctx_alpha / V
            C = C + (Cp * (~flag)[:, None] if flag.mean() < .5 else om * Cp)  # local: drop; diffuse: downweight
    if method == "oracle":                                       # knows drifted contexts and the noise mix
        rows = np.ones(V, bool); rows[dr] = False
        Cd = np.clip(Cn - EPS * Cn.sum(1, keepdims=True) / V, 0, None) / (1 - EPS)
        C = Cf + Cs * rows[:, None] + (1 - EPS) ** 2 * Cd
    p = model(C)
    return float((pi * (TQ * np.log(TQ / p)).sum(1)).sum())

if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for kd in (KD, 0):
        for nf in (500, 2000, 8000):
            tune = lambda m, g: min(g, key=lambda a: np.mean([run(s, nf, kd, m, **a) for s in range(1000, 1010)]))
            hp = {"token_filter": tune("token_filter", [dict(keep=k) for k in (.8, .95, .99)]),
                  "ctx_attr": tune("ctx_attr", [dict(om=o) for o in (0., .1, .3)])}
            print(f"\n== drifted contexts={kd}/{V}, fresh pairs={nf}, hp={hp}")
            for m in ("fresh_only", "pool_all", "token_filter", "ctx_attr", "oracle"):
                r = np.array([run(s, nf, kd, m, **hp.get(m, {})) for s in range(ns)])
                print(f"{m:13s} KL {r.mean():.4f} ({r.std() / ns ** .5:.4f})")
