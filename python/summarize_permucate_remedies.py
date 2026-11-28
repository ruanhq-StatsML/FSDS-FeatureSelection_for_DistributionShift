"""Summarise results/permucate_remedies/results_long.csv: AUC / AP / nuisance R^2 per outcome model, paired differences
to the RF and MLP T-learners (same data replicate), and the share of the oracle's excess AUC that is lost.
Usage: python3 summarize_permucate_remedies.py ../results/permucate_remedies"""

import sys
import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1] + "/results_long.csv")
order = ["rf", "mlp", "rl_lasso", "rl_rf", "rl_mlp", "pl_lasso_rf", "scr_rf", "oracle"]
ms = lambda x: f"{x.mean():.3f} ({x.std(ddof=1) / np.sqrt(len(x)):.3f})" if len(x) > 1 else f"{x.mean():.3f}"
out = [
    "# PermuCATE outcome-model variants (pure concept drift, |S| = 10, shift_var = 8)",
    "",
    "Mean (s.e.) over replicates. Chance AUC 0.5. `rl_*` = R-learner with one pooled outcome model;",
    "`mu_r2` of `rl_*` is measured against E[Y|x], the others against mu_w(x).",
    "",
]
for met in ("AUC", "AP", "mu_r2"):
    t = d.pivot_table(index="variant", columns="p_over_n", values=met, aggfunc=ms).reindex(order)
    out += (
        [
            f"## {met}",
            "",
            "| variant | " + " | ".join(f"p/n={c:g}" for c in t.columns) + " |",
            "|:--|" + "---:|" * len(t.columns),
        ]
        + [f"| {v} | " + " | ".join(t.loc[v]) + " |" for v in t.index]
        + [""]
    )
for base in ("rf", "mlp"):
    out += [
        f"## Paired AUC difference to `{base}` (variant minus {base}, same replicate)",
        "",
        "| variant | " + " | ".join(f"p/n={c:g}" for c in sorted(d.p_over_n.unique())) + " |",
        "|:--|" + "---:|" * d.p_over_n.nunique(),
    ]
    for v in order:
        if v == base:
            continue
        cells = []
        for r in sorted(d.p_over_n.unique()):
            a = d[(d.p_over_n == r) & (d.variant == v)].set_index("rep").AUC
            b = d[(d.p_over_n == r) & (d.variant == base)].set_index("rep").AUC
            x = (a - b).dropna()
            cells.append(f"{x.mean():+.3f} ({x.std(ddof=1) / np.sqrt(len(x)):.3f}), n={len(x)}" if len(x) > 1 else "-")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    out.append("")
rs = sorted(d.p_over_n.unique())
auc = lambda r, v: d[(d.p_over_n == r) & (d.variant == v)].AUC.mean() - 0.5
out += [
    "## Share of the oracle's excess AUC (over 0.5) lost to nuisance estimation",
    "",
    "| variant | " + " | ".join(f"p/n={c:g}" for c in rs) + " |",
    "|:--|" + "---:|" * len(rs),
]
for v in order[:-1]:
    out.append(f"| {v} | " + " | ".join(f"{100 * (1 - auc(r, v) / auc(r, 'oracle')):.0f}%" for r in rs) + " |")
open(sys.argv[1] + "/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
