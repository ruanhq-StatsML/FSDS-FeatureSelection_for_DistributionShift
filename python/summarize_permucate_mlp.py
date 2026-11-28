"""Summarize benchmark_permucate_mlp.py: AUC / AP of PermuCATE with RF vs MLP nuisance models vs p/n.

python summarize_permucate_mlp.py --dir ../results/permucate_mlp
"""
import os
import argparse
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["rf_insample", "rf", "mlp", "oracle"]
LABEL = {"rf_insample": "RF nuisances (in-sample pseudo-outcome)", "rf": "RF nuisances (cross-fitted)",
         "mlp": "MLP nuisances (cross-fitted)", "oracle": "oracle nuisances (diagnostic)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/permucate_mlp")
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(args.dir, "results_long.csv"))
    variants = [v for v in ORDER if v in set(df["variant"])]
    cols = ["AUC", "AP", "precision", "mu_r2", "pi_sd", "time_sec"]
    g = df.groupby(["p_over_n", "variant"])[cols]
    mean, sd = g.mean(), g.std()
    nrep = df.groupby("p_over_n")["rep"].nunique()
    s = int(df["k"].iloc[0])
    p_of = df.groupby("p_over_n")["p"].first()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric in zip(axes, ["AUC", "AP"]):
        for v in variants:
            m = mean.xs(v, level="variant")[metric]
            e = sd.xs(v, level="variant")[metric] / (nrep.reindex(m.index) ** 0.5)
            ax.errorbar(m.index, m.values, yerr=e.values, marker="o", capsize=2, label=LABEL[v],
                        ls="--" if v == "oracle" else "-")
        ax.set_xscale("log")
        ax.set_xlabel("p / n")
        ax.set_ylabel("selection " + metric)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[1].axhline(s / p_of.iloc[0], color="k", ls=":", lw=0.8)
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"PermuCATE under concept drift Y_new = X beta + f(X_S), |S| = {s}: RF vs MLP nuisances "
                 f"(mean +- s.e. over {int(nrep.min())} reps)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "permucate_mlp.png"), dpi=140)

    def fmt(metric):
        t = pd.DataFrame({v: [f"{mean.loc[(r, v), metric]:.3f} ± {sd.loc[(r, v), metric] / nrep[r] ** 0.5:.3f}"
                              for r in mean.index.get_level_values(0).unique()] for v in variants},
                         index=[f"p/n={r:g} (p={p_of[r]})" for r in mean.index.get_level_values(0).unique()])
        return t.rename(columns=LABEL)

    lines = [f"# PermuCATE: RF vs MLP nuisance models, pure concept drift (|S| = {s} shifted features)", "",
             f"Replicates per p/n: {int(nrep.min())}-{int(nrep.max())}. Mean ± s.e.", ""]
    for metric in ["AUC", "AP", "precision"]:
        lines += [f"## {metric}", "", fmt(metric).to_markdown(), ""]
    lines += ["## Nuisance quality on the test half (R^2 of the outcome model against the true mu; spread of the propensity)",
              "", mean[["mu_r2", "pi_sd"]].round(3).unstack("variant").to_markdown(), "",
              "## Wall-clock seconds per replicate", "", mean["time_sec"].unstack("variant").round(0).to_markdown(), ""]
    with open(os.path.join(args.dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
