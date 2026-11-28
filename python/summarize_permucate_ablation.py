"""Summarize the nuisance ablation of PermuCATE (results/permucate_mlp_ablation) next to the main RF/MLP/oracle run."""
import os
import sys

import numpy as np
import pandas as pd

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
main = pd.read_csv(os.path.join(root, "permucate_mlp", "results_long.csv"))
abl = pd.read_csv(os.path.join(root, "permucate_mlp_ablation", "results_long.csv"))
fa_path = os.path.join(root, "permucate_mlp_fa", "results_long.csv")
fa = pd.read_csv(fa_path) if os.path.exists(fa_path) else abl.iloc[:0]
d = pd.concat([main, abl, fa], ignore_index=True)
d = d[d.p_over_n.isin(sorted(set(abl.p_over_n) | set(fa.p_over_n)))]

order = [("oracle", "true mu, true pi"), ("oracle_rfpi", "true mu, RF pi"), ("oracle_r2_0.6", "true mu + noise, nominal R2 0.6"),
         ("oracle_r2_0.3", "true mu + noise, nominal R2 0.3"), ("oracle_r2_0.1", "true mu + noise, nominal R2 0.1"),
         ("rf_truepi", "RF mu, true pi"), ("rf", "RF mu, RF pi"), ("mlp", "MLP mu, MLP pi"),
         ("fa", "factor-augmented ridge/logistic (PCs, K by eigenvalue ratio)")]


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


lines = ["# PermuCATE nuisance ablation (pure concept drift, |S| = 10)", "",
         "AUC / AP mean (s.e.) over replicates; `mu R2` is the measured out-of-sample R^2 of the outcome model "
         "against the true regression. Chance: AUC 0.5, AP = 10/p.", ""]
for metric in ["AUC", "AP", "mu_r2"]:
    lines += [f"## {metric}", "", "| variant | " + " | ".join(f"p/n={int(r)}" for r in sorted(d.p_over_n.unique())) + " |",
              "|:--|" + "---:|" * d.p_over_n.nunique()]
    for key, label in order:
        sub = d[d.variant == key]
        if sub.empty:
            continue
        cells = []
        for r in sorted(d.p_over_n.unique()):
            x = sub[sub.p_over_n == r][metric]
            cells.append(f"{x.mean():.3f} ({se(x):.3f})" if len(x) > 1 else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

lines += ["## Share of the oracle's excess AUC (over 0.5) that is lost to nuisance estimation", "",
          "| p/n | oracle AUC-0.5 | RF mu+pi | RF mu, true pi | MLP |", "|---:|---:|---:|---:|---:|"]
for r in sorted(d.p_over_n.unique()):
    g = lambda v: d[(d.variant == v) & (d.p_over_n == r)].AUC.mean() - 0.5
    o = g("oracle")
    lines.append(f"| {int(r)} | {o:.3f} | {1 - g('rf') / o:.0%} | {1 - g('rf_truepi') / o:.0%} | {1 - g('mlp') / o:.0%} |")
open(os.path.join(root, "permucate_mlp_ablation", "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
