"""Aggregate the JSON results into REPORT.md + figures.   python -m gen_fsds.report results_gen"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(root: str, suite: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(root, suite, "*.json"))):
        rows.extend(json.load(open(f)))
    return pd.DataFrame(rows)


def ms(x: pd.Series, d: int = 2) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "-"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"{x.mean():.{d}f}±{se:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def agg(df: pd.DataFrame, group, metrics: dict) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(group, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group, key))
        row["reps"] = g["seed"].nunique()
        for col, (label, d) in metrics.items():
            if col in g:
                row[label] = ms(g[col], d)
        out.append(row)
    return pd.DataFrame(out)


COV_METRICS = {
    "auc_cond": ("AUC(cond)", 2), "auc_strict": ("AUC(S)", 2), "prec_at_k": ("P@k", 2),
    "fpr": ("FPR@.05", 2), "fpr_desc": ("FPR desc", 2), "power": ("power@.05", 2),
    "bh_fdp": ("BH FDP", 2), "bh_power": ("BH power", 2), "time": ("sec", 0),
}
CD_METRICS = {
    "auc_strict": ("AUC(S)", 2), "prec_at_k": ("P@k", 2), "fpr": ("FPR@.05", 2),
    "power": ("power@.05", 2), "bh_fdp": ("BH FDP", 2), "bh_power": ("BH power", 2), "time": ("sec", 0),
}


def fig_quality(q: pd.DataFrame, root: str):
    q = q[q.steps == -1]
    if q.empty:
        return
    q = q.assign(cfg=q.n.astype(str) + np.where(q.get("noise_scale", 1.0) < 1, " strongdep", ""))
    cfgs = list(dict.fromkeys(q.cfg))
    samplers = list(dict.fromkeys(q.sampler))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        m = q[q.cfg == c].groupby("sampler").c2st.mean().reindex(samplers)
        e = q[q.cfg == c].groupby("sampler").c2st.sem().reindex(samplers)
        ax.bar(np.arange(len(samplers)) + i * w, m, w, yerr=e, label=f"n={c}")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(len(samplers)) + 0.4 - w / 2)
    ax.set_xticklabels(samplers, rotation=30)
    ax.set_ylabel("C2ST AUC (0.5 = perfect)")
    ax.set_ylim(0.45, 0.9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "fig_sampler_quality.png"), dpi=140)
    plt.close(fig)


def fig_steps(s: pd.DataFrame, root: str):
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for name, g in s.groupby("sampler"):
        m = g.groupby("steps").c2st.mean()
        e = g.groupby("steps").c2st.sem()
        ax.errorbar(m.index, m.values, yerr=e.values, marker="o", label=name)
    ax.set_xscale("log")
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xlabel("Euler steps (NFE)")
    ax.set_ylabel("C2ST AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "fig_sampler_steps.png"), dpi=140)
    plt.close(fig)


def fig_detection(df: pd.DataFrame, root: str, name: str, key_metric: str, scenarios=None):
    if df.empty:
        return
    scen = scenarios or list(dict.fromkeys(df.scenario))
    methods = list(dict.fromkeys(df.method))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 1.1 * len(scen) * 2), 6.5), sharex=True)
    w = 0.85 / len(methods)
    for ax, metric, title in zip(axes, [key_metric, "fpr"], [f"{key_metric} (higher better)",
                                                              "FPR@0.05 on true nulls (target 0.05)"]):
        for i, m in enumerate(methods):
            vals = [df[(df.scenario == s) & (df.method == m)][metric].mean() for s in scen]
            errs = [df[(df.scenario == s) & (df.method == m)][metric].sem() for s in scen]
            ax.bar(np.arange(len(scen)) + i * w, vals, w, yerr=errs, label=m)
        ax.set_title(title, fontsize=10)
    axes[1].axhline(0.05, color="k", ls="--", lw=1)
    axes[1].set_xticks(np.arange(len(scen)) + 0.4)
    axes[1].set_xticklabels(scen, rotation=20, fontsize=8)
    axes[0].legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(root, f"fig_{name}.png"), dpi=140)
    plt.close(fig)


def main(root: str):
    out = ["# Generative conditional samplers for FSDS - benchmark report\n"]
    q = load(root, "quality")
    if not q.empty:
        out.append("## 1. Sampler quality: C2ST AUC of (X_-j, X~_j) vs (X_-j, X_j) (0.5 = indistinguishable)\n")
        q["cfg"] = q.n.astype(str) + np.where(q.get("noise_scale", 1.0) < 1, " strongdep", "")
        t = q[q.steps == -1].groupby(["sampler", "cfg"], sort=False).apply(
            lambda g: pd.Series(dict(c2st=ms(g.c2st, 3), fit_s=ms(g.fit_time, 0), reps=g.seed.nunique())),
            include_groups=False).reset_index()
        out.append(md_table(t.pivot(index="sampler", columns="cfg", values="c2st").reset_index()
                            .reindex(columns=["sampler"] + list(dict.fromkeys(q.cfg)))))
        out.append("\nFit time for all p=20 conditionals (seconds, n=5000):\n")
        tt = t[t.cfg == "5000"][["sampler", "fit_s", "reps"]]
        out.append(md_table(tt))
        fig_quality(q, root)
    s = load(root, "steps")
    if not s.empty:
        out.append("\n## 2. Rectified flow: C2ST vs number of Euler steps (n=5000)\n")
        t = s.groupby(["sampler", "steps"], sort=False).c2st.apply(lambda x: ms(x, 3)).reset_index()
        out.append(md_table(t.pivot(index="sampler", columns="steps", values="c2st").reset_index()))
        fig_steps(s, root)
    cv = load(root, "covshift")
    if not cv.empty:
        out.append("\n## 3. Covariate-shift diagnosis (exact conditional truth)\n")
        out.append("cond = S ∪ pa(S) = features whose full conditional shifts; FPR/FDP are measured against "
                   "its complement. 'desc' = descendants of S that inherit a *marginal* shift but whose "
                   "conditional is unchanged.\n")
        for scen, g in cv.groupby("scenario", sort=False):
            out.append(f"\n### {scen}\n")
            out.append(md_table(agg(g, ["method"], COV_METRICS)))
        fig_detection(cv, root, "covshift", "auc_cond")
    cd = load(root, "concept")
    if not cd.empty:
        out.append("\n## 4. Concept-drift diagnosis (truth = S)\n")
        for scen, g in cd.groupby("scenario", sort=False):
            out.append(f"\n### {scen}\n")
            out.append(md_table(agg(g, ["method"], CD_METRICS)))
        fig_detection(cd, root, "concept", "auc_strict")
    with open(os.path.join(root, "REPORT.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", os.path.join(root, "REPORT.md"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_gen")
