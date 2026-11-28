"""Markdown summary of results/screening_fsds/grid/*.csv: recall by method x (sd, n, rho) and scan seconds.
Usage: python3 summarize_screening_grid.py ../results/screening_fsds/grid"""

import glob, os, sys
import pandas as pd

d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(sys.argv[1], "grid_*.csv")))])
M = ["random", "sis_union", "csis_pooled", "csis_union", "drift_aware", "csis_np_union", "isis_union"]
out = ["# Screening grid summary (mean recall of the 24 drift coordinates; 5 reps per cell)\n"]
for (p, K), x in d.groupby(["p", "K"]):
    x = x[x.method.isin(M)]
    out.append(f"\n## p={p}, K={K}\n")
    for by in ("sd", "n", "rho"):
        out.append(
            f"\nby {by} (averaged over the other factors)\n\n"
            + x.pivot_table(index="method", columns=by, values="recall").reindex(M).round(2).to_markdown()
            + "\n"
        )
s = d[d.method == "csis_union"].groupby(["p", "n", "K"])[["sec_sis", "sec_csis", "sec_np", "sec_sq"]].mean().round(2)
out.append("\n## Scan seconds per replicate (both batches), by p, n, K\n\n" + s.to_markdown() + "\n")
open(os.path.join(sys.argv[1], "summary.md"), "w").write("\n".join(out))
