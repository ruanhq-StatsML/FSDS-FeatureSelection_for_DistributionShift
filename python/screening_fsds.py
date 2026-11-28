"""Screening-Enhanced FSDS (V1: conditional-mean drift). Stage 1: score every feature separately in the two labelled
batches (SIS, conditional SIS on the leading PCs, quantile-bin NIS), combine batches (union / pooled / one batch /
intersection / drift-aware U+|D|+C / one ISIS round / oracle / random) at a fixed budget K. Stage 2 (--fsds): PermuCATE
(FSDS) on the retained features. Metrics: drift-support recall (by type), retained size, seconds, FSDS AUC.
DGP: factor-correlated features, sg dense weak stable signals, 24 drift coordinates (decay, emerge, reverse, nonmono, inter).
Usage: python3 screening_fsds.py --p 10000 --K 100 500 --reps 20 [--fsds 5] --out ../results/screening_fsds"""

import argparse, os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import benchmark_permucate_mlp as m, benchmark_ultrahigh_dim as bu

TYPES = ["decay", "emerge", "reverse", "nonmono", "inter"]


def make(seed, n, p, rho, sg, snr, sd, nf=10, nl=False):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(p, nf))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    X = [
        (np.sqrt(rho) * rng.normal(size=(n, nf)) @ L.T + np.sqrt(1 - rho) * rng.normal(size=(n, p))).astype(np.float32)
        for _ in (0, 1)
    ]
    idx = rng.permutation(p)
    st, b, dr = idx[:sg], rng.normal(size=sg), idx[sg : sg + 24]
    g = dict(zip(TYPES, [dr[:5], dr[5:10], dr[10:15], dr[15:20], dr[20:24]]))
    f = {
        "decay": lambda Z, j, t: Z[:, j].sum(1) * (t == 0),
        "emerge": lambda Z, j, t: Z[:, j].sum(1) * (t == 1),
        "reverse": lambda Z, j, t: Z[:, j].sum(1) * (1 - 2 * t),
        "nonmono": lambda Z, j, t: (Z[:, j] ** 2 - 1).sum(1) * (t == 1),
        "inter": lambda Z, j, t: (Z[:, j[0]] * Z[:, j[1]] + Z[:, j[2]] * Z[:, j[3]]) * (t == 1),
    }
    if nl:  # highly nonlinear drift: |z| (decay), z^2 (emerge), sin(3z) (reverse), band 1{|z|<.5} (new), products
        f["decay"] = lambda Z, j, t: np.abs(Z[:, j]).sum(1) * (t == 0)
        f["emerge"] = lambda Z, j, t: (Z[:, j] ** 2).sum(1) * (t == 1)
        f["reverse"] = lambda Z, j, t: np.sin(3 * Z[:, j]).sum(1) * (1 - 2 * t)
        f["nonmono"] = lambda Z, j, t: (np.abs(Z[:, j]) < 0.5).sum(1) * (t == 1)
    Xp = np.vstack(X)
    b *= np.sqrt(snr / np.var(Xp[:, st] @ b))
    c = {
        k: np.sqrt(sd / 5 / np.var(f[k](Xp, g[k], 1) - f[k](Xp, g[k], 0))) for k in TYPES
    }  # equal drift energy per type
    y = [X[t][:, st] @ b + sum(c[k] * f[k](X[t], g[k], t) for k in TYPES) + rng.normal(size=n) for t in (0, 1)]
    return X[0], y[0], X[1], y[1], g


def resid(A, Q):
    return A - Q @ (Q.T @ A)


def core(X, nc=10):  # centred leading PCs (Gram trick): known core factors
    Xc = X - X.mean(0)
    return np.linalg.eigh((Xc @ Xc.T).astype(np.float64))[1][:, -nc:]


def scan(X, y, Q, kind, ch=8192):  # row 0: marginal gain G (or NIS), row 1: slope beta
    n, out, yr = len(y), np.zeros((2, X.shape[1])), resid(y - y.mean(), Q)
    for a in range(0, X.shape[1], ch):
        Z = X[:, a : a + ch].astype(np.float64)
        Z = resid(Z - Z.mean(0), Q)
        if kind == "sq":  # even-in-x signal: scan the centred square
            Z = Z**2 - (Z**2).mean(0)
        cv, vr = Z.T @ yr / n, (Z**2).mean(0) + 1e-12
        out[1, a : a + ch] = cv / vr
        if kind == "np":  # variance of E[y | quantile bin of x_j], 5 bins
            r = np.argsort(np.argsort(Z, 0), 0) * 5 // n
            out[0, a : a + ch] = sum(((r == k).T @ yr) ** 2 / np.maximum((r == k).sum(0), 1) for k in range(5)) / n
        else:
            out[0, a : a + ch] = cv**2 / vr
    return out


def top(s, k):
    return np.argpartition(-s, k)[:k]


def fill(S, s0, s1, K):  # fill budget by best cross-batch rank
    r = np.minimum(np.argsort(np.argsort(-s0)), np.argsort(np.argsort(-s1)))
    r[S] = 10**9
    return np.union1d(S, np.argsort(r)[: max(0, K - len(S))])


def union(s0, s1, K):
    return fill(np.union1d(top(s0, K // 2), top(s1, K // 2)), s0, s1, K)


def select(X0, y0, X1, y1, g, K, rng):
    t, p, allj = {}, X0.shape[1], np.concatenate(list(g.values()))
    Q = [core(X0), core(X1)]
    mk = lambda kind, q: [scan(X, y, qq, kind) for X, y, qq in ((X0, y0, q[0]), (X1, y1, q[1]))]
    t0 = time.perf_counter()
    z = [np.zeros((len(y0), 0)), np.zeros((len(y1), 0))]
    sis = mk("lin", z)
    t["sis"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cs = mk("lin", Q)
    t["csis"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    npx = mk("np", Q)
    t["np"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sq = mk("sq", Q)
    t["sq"] = time.perf_counter() - t0
    Xw, yw = np.vstack([X0 - X0.mean(0), X1 - X1.mean(0)]), np.r_[y0 - y0.mean(), y1 - y1.mean()]
    pool = scan(Xw, yw, core(Xw), "lin")[0]
    U, D, C = np.maximum(cs[0][0], cs[1][0]), np.abs(cs[0][0] - cs[1][0]), np.abs(cs[0][1] - cs[1][1])
    k3 = K // 3
    S1 = union(cs[0][0], cs[1][0], K // 2)
    res = []  # ISIS: ridge on S1, rescan residuals, add K/2 more
    for X, y, q in ((X0, y0, Q[0]), (X1, y1, Q[1])):
        Z = X[:, S1].astype(np.float64)
        Z -= Z.mean(0)
        b = np.linalg.solve(Z.T @ Z + len(y) / 10 * np.eye(len(S1)), Z.T @ (y - y.mean()))
        res.append(scan(X, y - Z @ b - y.mean(), q, "lin")[0])
    sel = {
        "sis_ref": top(sis[0][0], K),
        "sis_tgt": top(sis[1][0], K),
        "sis_union": union(sis[0][0], sis[1][0], K),
        "csis_ref": top(cs[0][0], K),
        "csis_tgt": top(cs[1][0], K),
        "csis_pooled": top(pool, K),
        "csis_union": union(cs[0][0], cs[1][0], K),
        "csis_inter": np.intersect1d(top(cs[0][0], K), top(cs[1][0], K)),
        "np_union": union(npx[0][0], npx[1][0], K),
        "sq_union": union(sq[0][0], sq[1][0], K),
        "nl_union": fill(
            np.union1d(
                np.union1d(union(cs[0][0], cs[1][0], K // 3), union(npx[0][0], npx[1][0], K // 3)),
                union(sq[0][0], sq[1][0], K // 3),
            ),
            npx[0][0],
            npx[1][0],
            K,
        ),
        "csis_np_union": np.union1d(union(cs[0][0], cs[1][0], K // 2), union(npx[0][0], npx[1][0], K // 2)),
        "drift_aware": fill(np.union1d(np.union1d(top(U, k3), top(D, k3)), top(C, k3)), U, U, K),
        "isis_union": np.union1d(S1, union(res[0], res[1], K // 2)),
        "oracle": np.concatenate([allj, rng.choice(np.setdiff1d(np.arange(p), allj), K - len(allj), replace=False)]),
        "random": rng.choice(p, K, replace=False),
    }
    return sel, t


def fsds(X0, y0, X1, y1, S, allj, seed, ne=100):  # PermuCATE importance on retained columns; others ranked last
    t0 = time.perf_counter()
    X, W, Y = np.vstack([X0[:, S], X1[:, S]]), np.r_[np.zeros(len(y0)), np.ones(len(y1))].astype(int), np.r_[y0, y1]
    fit = m.permucate_fit_cf(X, Y, W, seed, "rf", ne, tau_mtry=0.1)
    v = np.full(X0.shape[1], -np.inf)
    v[S] = bu.permucate_vimp(fit, seed + 1, 25, 20)[0]
    r = bu.metrics_from_scores(v, allj, X0.shape[1])
    inS, ind = np.isin(S, allj), np.isin(np.arange(X0.shape[1]), S)
    within = roc_auc_score(inS, v[S]) if 0 < inS.sum() < len(S) else np.nan  # AUC among retained columns only
    return dict(
        fsds_auc=r["AUC"],
        fsds_rec=r["recall"],
        fsds_sec=time.perf_counter() - t0,
        mech_auc=roc_auc_score(np.isin(np.arange(X0.shape[1]), allj), ind),  # retained-indicator alone, no FSDS ranking
        within_auc=within,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=10000)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--K", type=int, nargs="+", default=[100, 500])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--sg", type=int, default=200)
    ap.add_argument("--snr", type=float, default=0.15)
    ap.add_argument("--sd", type=float, default=0.2)
    ap.add_argument("--nl", action="store_true", help="highly nonlinear drift (see make)")
    ap.add_argument("--fsds", type=int, default=0, help="reps with the FSDS stage (plus no-screen if p <= 10000)")
    ap.add_argument("--fsds_K", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, frows = [], []
    for rep in range(a.reps):
        X0, y0, X1, y1, g = make(rep, a.n, a.p, a.rho, a.sg, a.snr, a.sd, nl=a.nl)
        allj = np.concatenate(list(g.values()))
        for K in a.K:
            sel, t = select(X0, y0, X1, y1, g, K, np.random.default_rng(rep))
            for mth, S in sel.items():
                rows.append(
                    dict(
                        rep=rep,
                        K=K,
                        method=mth,
                        size=len(S),
                        recall=np.isin(allj, S).mean(),
                        **{f"rec_{k}": np.isin(g[k], S).mean() for k in TYPES},
                        **{f"sec_{k}": v for k, v in t.items()},
                    )
                )
        if rep < a.fsds:
            sel, _ = select(X0, y0, X1, y1, g, a.fsds_K, np.random.default_rng(rep))
            sel = {
                k: sel[k]
                for k in (
                    "sis_union",
                    "csis_union",
                    "csis_pooled",
                    "drift_aware",
                    "csis_np_union",
                    "sq_union",
                    "nl_union",
                    "oracle",
                    "random",
                )
            }
            if a.p <= 10000:
                sel["no_screen"] = np.arange(a.p)
            for mth, S in sel.items():
                frows.append(dict(rep=rep, method=mth, size=len(S), **fsds(X0, y0, X1, y1, S, allj, rep)))
            pd.DataFrame(frows).to_csv(os.path.join(a.out, f"fsds_p{a.p}.csv"), index=False)
        print(f"rep {rep} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"screen_p{a.p}.csv"), index=False)
    d = pd.DataFrame(rows)
    for K in a.K:
        print(f"\n== p={a.p}, n={a.n}/batch, rho={a.rho}, K={K}: mean drift recall by type ({a.reps} reps)")
        print(
            d[d.K == K].groupby("method")[["size", "recall"] + [f"rec_{k}" for k in TYPES]].mean().round(2).to_string()
        )
    print(
        "\nscreening seconds per replicate (scan of both batches):",
        d[[c for c in d if c.startswith("sec_")]].mean().round(1).to_dict(),
    )
    if frows:
        print("\nFSDS stage (K=%d):" % a.fsds_K)
        print(
            pd.DataFrame(frows)
            .groupby("method")[["size", "fsds_auc", "fsds_rec", "fsds_sec", "mech_auc", "within_auc"]]
            .agg(["mean"])
            .round(3)
            .to_string()
        )
