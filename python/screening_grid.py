"""Screen-only grid for screening_fsds.py: per cell (p, n, rho, sd) x K, drift-support recall by method and scan seconds.
Usage: python3 screening_grid.py --reps 5 --out ../results/screening_fsds/grid [--cells small|large]"""

import argparse, itertools, os, time
import numpy as np, pandas as pd
import screening_fsds as sf

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--cells", default="small", help="small: p<=10000 full grid; large: p=50000 reduced grid")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.cells == "small":
    grid = itertools.product([2000, 10000], [500, 1000, 2000], [0.2, 0.5, 0.8], [0.2, 0.5, 1.0])
else:
    grid = itertools.product([50000], [1000, 2000], [0.5], [0.2, 0.5, 1.0])
path, rows = os.path.join(a.out, f"grid_{a.cells}.csv"), []
for p, n, rho, sd in grid:
    for rep in range(a.reps):
        X0, y0, X1, y1, g = sf.make(rep, n, p, rho, 200, 0.15, sd)
        allj = np.concatenate(list(g.values()))
        for K in ([100, 200] if p == 2000 else [100, 500] if p == 10000 else [200, 1000]):
            sel, t = sf.select(X0, y0, X1, y1, g, K, np.random.default_rng(rep))
            for mth, S in sel.items():
                rows.append(
                    dict(
                        p=p,
                        n=n,
                        rho=rho,
                        sd=sd,
                        K=K,
                        rep=rep,
                        method=mth,
                        size=len(S),
                        recall=np.isin(allj, S).mean(),
                        **{f"rec_{k}": np.isin(g[k], S).mean() for k in sf.TYPES},
                        **{f"sec_{k}": v for k, v in t.items()},
                    )
                )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"done p={p} n={n} rho={rho} sd={sd} ({time.strftime('%H:%M:%S')})", flush=True)
