"""Aggregate results of benchmark_ultrahigh_dim.py (one or more output dirs) into figures and a markdown summary.

python summarize_ultrahigh_dim.py --dirs ../results/ultrahigh_dim_p20n ../results/ultrahigh_dim_p100n --out ../results
"""
import os
import argparse
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_ORDER = [
    "rf_domain", "permucate", "loco_r_risk", "sgshift_l1_spec", "sgshift_l1_misspec", "datafix", "fslnet",
    "rfdom_shadowVIMP", "rfdom_vitaPIMP", "rfdom_boruta", "delta_shap", "grf",
]
COLORS = {m: c for m, c in zip(METHOD_ORDER, plt.get_cmap("tab10").colors + plt.get_cmap("Set2").colors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["../results/ultrahigh_dim_p20n", "../results/ultrahigh_dim_p100n"])
    ap.add_argument("--out", default="../results")
    ap.add_argument("--name", default="ultrahigh_dim", help="prefix of the output png / md")
    ap.add_argument("--title", default="messy nonlinear DGP, concept + covariate shift, 10% signal")
    args = ap.parse_args()
    dfs = [pd.read_csv(os.path.join(d, "results_long.csv")) for d in args.dirs if
           os.path.exists(os.path.join(d, "results_long.csv"))]

    metrics = ["AUC", "AP", "precision", "lift"]
    titles = {"AUC": "Selection AUC", "AP": "Average precision", "precision": "Precision@k (= recall@k = F1@k)",
              "lift": "Lift of precision@k over chance (0.1)"}
    fig, axes = plt.subplots(len(dfs), len(metrics), figsize=(5.2 * len(metrics), 4.3 * len(dfs)), squeeze=False)
    lines = [f"# Ultra-high-dim benchmark ({args.title}, RF outcome model)", ""]
    for r, df in enumerate(dfs):
        n, p = int(df["n"].iloc[0]), int(df["p"].iloc[0])
        methods = [m for m in METHOD_ORDER if m in set(df["method"])]
        agg = df.groupby(["method", "rho"])[metrics + ["time_sec"]].mean().reset_index()
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            for m in methods:
                d = agg[agg.method == m].sort_values("rho")
                ax.plot(d.rho, d[metric], marker="o", ms=3.5, lw=1.4, color=COLORS[m], label=m)
            if metric == "AUC":
                ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.set_xlabel("factor strength rho (grid = np.linspace(0, 0.95, 20))")
            ax.set_title(f"p/n={p // n} (p={p}): {titles[metric]}", fontsize=10)
            ax.grid(alpha=0.3)
        axes[r, 0].legend(fontsize=7.5)

        ov = df.groupby("method").agg(
            AUC=("AUC", "mean"), AP=("AP", "mean"), precision=("precision", "mean"), lift=("lift", "mean"),
            FDR=("FDR", "mean"), recall_2k=("recall@2k", "mean"), time_sec=("time_sec", "mean"),
            n_grid=("rho", "nunique")).reindex(methods).sort_values("AUC", ascending=False)
        lines += [f"## p/n = {p // n} (n_exist = n_new = {n}, p = {p}, {int(df['n_shifted'].max())} shifted features, chance precision = {df['n_shifted'].max() / p:.2f})", "",
                  f"Grid points completed: {df['rho'].nunique()}, replicates per point: {df['rep'].nunique()}", "",
                  "### Overall (mean over completed grid points)", "", ov.round(3).to_markdown(), ""]
        if "native_n_selected" in df:
            nat = df[df["method"].isin(["datafix", "fslnet"])].groupby("method")[
                ["native_n_selected", "native_precision", "native_recall"]].mean()
            lines += ["### Native selection rule (DataFix mask / FSL-Net prob > 0.5)", "", nat.round(3).to_markdown(), ""]
        if "permucate_n_used_features" in df:
            u = df.groupby("rho")["permucate_n_used_features"].first()
            lines += [f"PermuCATE: tau-forest splits on {u.mean():.0f} of {p} features on average "
                      f"({u.mean() / p:.1%}); the rest get importance exactly 0.", ""]
        for metric in ["AUC", "precision"]:
            t = df.pivot_table(index="method", columns="rho", values=metric, aggfunc="mean").reindex(methods)
            t.columns = [f"{c:.2f}" for c in t.columns]
            lines += [f"### {metric} by rho", "", t.round(2).to_markdown(), ""]
    fig.suptitle(f"Ultra-high-dim feature selection for distribution shift: {args.title}, RF outcome model")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.name}_benchmark.png"), dpi=140)
    with open(os.path.join(args.out, f"{args.name}_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
