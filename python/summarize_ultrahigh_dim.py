"""Aggregate results of benchmark_ultrahigh_dim.py into figures and a markdown table.

python summarize_ultrahigh_dim.py --out ../results/ultrahigh_dim
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_ORDER = [
    "rf_domain", "rfdom_shadowVIMP", "rfdom_vitaPIMP", "rfdom_boruta", "rfdom_condperm", "rfdom_loco",
    "delta_shap", "loco_mmd", "loco_hsic", "permucate", "loco_r_risk", "grf",
    "sgshift_l1_spec", "sgshift_l1_misspec",
]
POOLED = {"rfdom_condperm", "rfdom_loco", "permucate", "loco_r_risk"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/ultrahigh_dim")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.out, "results_long.csv"))
    gp = df["grid_param"].iloc[0]
    n, p = int(df["n"].iloc[0]), int(df["p"].iloc[0])
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    agg = df.groupby(["method", "grid_value"])[["AUC", "AP", "precision", "recall@2k", "time_sec"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    cmap = plt.get_cmap("tab20")
    for ax, metric, title in zip(axes, ["AUC", "AP", "precision"],
                                 ["Selection AUC", "Average precision", "Precision@k (= recall@k = F1@k)"]):
        for i, m in enumerate(methods):
            d = agg[agg.method == m].sort_values("grid_value")
            ax.plot(d.grid_value, d[metric], marker="o", ms=3.5, lw=1.5, color=cmap(i),
                    ls="--" if m in POOLED else "-", label=m)
        ax.set_xlabel(f"{gp} grid value")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, ncol=2)
    k = int(df["n_shifted"].max())
    fig.suptitle(f"Ultra-high-dim feature selection for distribution shift: n_exist = n_new = {n}, p = {p} "
                 f"(p/n = {p // n}), {k} shifted features, RF outcome model; dashed = screen-then-refine methods")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ultrahigh_dim_benchmark.png"), dpi=150)

    def tab(metric):
        t = df.pivot_table(index="method", columns="grid_value", values=metric, aggfunc="mean").reindex(methods)
        t["mean"] = t.mean(axis=1)
        return t

    ov = df.groupby("method").agg(AUC=("AUC", "mean"), AP=("AP", "mean"), precision=("precision", "mean"),
                                  recall_2k=("recall@2k", "mean"), time_sec=("time_sec", "mean")).reindex(methods)
    ov = ov.sort_values("AUC", ascending=False)
    lines = [f"# Ultra-high-dim benchmark (n={n} per batch, p={p}, grid over {gp})", "",
             f"Grid points completed: {df['grid_value'].nunique()}, replicates per point: {df['rep'].nunique()}", "",
             "## Overall (mean over the grid)", "", ov.round(3).to_markdown(), "",
             f"Mean candidate-pool recall (ceiling for screen-then-refine methods): "
             f"{df.groupby(['grid_idx', 'rep'])['pool_recall'].first().mean():.3f}", ""]
    for metric in ["AUC", "AP", "precision"]:
        lines += [f"## {metric} by {gp}", "", tab(metric).round(2).to_markdown(), ""]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
